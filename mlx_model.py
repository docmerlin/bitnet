"""Experimental MLX port of the dense BitNet PaTH-FoX model path."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint as activation_checkpoint

from mlx_path_kernel import path_triangular_solve, reference_triangular_solve
from mlx_rfmoe_kernel import compacted_grouped_linear, masked_grouped_linear
from mlx_ternary_kernel import (
    activation_levels as _activation_level_pair,
    pack_ternary_weight,
    ste_activation_quantize,
    ternary_fused_ffn_m1,
    ternary_fused_linear_m1,
    ternary_quantized_linear,
)


_effective_weight_cache: ContextVar[dict[tuple[object, str], object] | None] = ContextVar(
    "effective_weight_cache",
    default=None,
)
_recurrent_quantized_matmul: ContextVar[bool] = ContextVar("recurrent_quantized_matmul", default=False)
# "last" = last-query PaTH decode (default); "recompute" = full open-chunk path_chunk baseline.
_path_decode_mode: ContextVar[str] = ContextVar("path_decode_mode", default="last")


_MEMORY_EPS = 1e-6


def _safe_norm(x: mx.array, axis: int, keepdims: bool = False) -> mx.array:
    """L2 norm whose gradient is finite at zero.

    ``mx.linalg.norm`` differentiates to ``x / ||x||``, which is 0/0 at the
    origin. Guarding the *result* -- ``mx.maximum(norm(x), eps)`` -- fixes the
    forward and does nothing for the backward, because ``maximum`` then
    multiplies the already-NaN cotangent by zero and NaN survives it. Putting
    eps inside the square root instead keeps the whole thing smooth.
    """
    return mx.sqrt(mx.sum(x * x, axis=axis, keepdims=keepdims) + 1e-12)


def _safe_normalize(x: mx.array, axis: int = -1, floor: float = 1e-6) -> mx.array:
    """``x`` scaled to unit norm, finite gradient at zero. See :func:`_safe_norm`."""
    return x / mx.maximum(_safe_norm(x, axis=axis, keepdims=True), floor)


def _infini_sigma(x: mx.array) -> mx.array:
    """ELU+1 feature map (Linear Transformer / Infini-attention paper)."""
    return mx.where(x > 0, x + 1.0, mx.exp(x))


@dataclass(frozen=True)
class MLXBitNetConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    num_attention_heads: int = 16
    intermediate_size: int = 1024
    num_prelude_layers: int = 2
    num_recurrent_layers: int = 4
    num_coda_layers: int = 2
    num_loops: int = 4
    block_size: int = 8
    path_window_size: int = 1024
    # Legacy unused slot count (paper Infini uses head_dim×head_dim associative M).
    infini_memory_dim: int = 64
    infini_memory_expand: int = 0  # Infini key-feature width; 0 = head_dim. See config.py.
    infini_feature_map: str = "elu"  # "elu" (paper) or "favor" (exp kernel). See config.py.
    infini_delta_rule: bool = True
    # Optional block-granular top-k retrieval branch (training only). See config.py.
    use_topk_blocks: bool = False
    topk_blocks: int = 4
    topk_block_size: int = 64
    activation_bits: int = 4
    use_4bit_activations: bool = True
    use_hadamard: bool = True
    use_path_kernel: bool = True
    rms_norm_eps: float = 1e-5
    use_engram: bool = True
    engram_layer_ids: tuple[int, ...] = (1, 15)
    engram_vocab_size: int | None = None  # None → ~engram_param_fraction of body
    engram_param_fraction: float = 0.05
    engram_max_ngram_size: int = 3
    engram_num_heads: int = 4
    engram_head_dim: int = 16
    engram_kernel_size: int = 4
    engram_pad_id: int = 257
    engram_seed: int = 0
    use_rfmoe: bool = False
    rfmoe_num_experts: int = 8
    rfmoe_expert_dim: int | None = None
    rfmoe_rank: int | None = None
    rfmoe_theta: float = 0.01
    rfmoe_backend: str = "auto"
    mtp_depth: int = 0
    # Residual path: "kimi" = Block AttnRes (arXiv:2603.15031); "sandwich" = legacy scale residual.
    attn_res_mode: str = "kimi"
    # Transformer layers per AttnRes depth-block (None → max(1, unique_layers // 8)).
    attn_res_group_size: int | None = None
    # Share the output projection with the input embedding. False (untied) is the
    # modded-nanogpt configuration: an untied head plus a much higher embedding
    # learning rate was their largest win after Muon. Costs vocab*hidden extra
    # parameters (16.7M at vocab 32768 / hidden 512).
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.path_window_size < 1:
            raise ValueError("path_window_size must be positive")
        if min(self.num_prelude_layers, self.num_recurrent_layers, self.num_coda_layers) < 0:
            raise ValueError("layer counts must be non-negative")
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if self.engram_max_ngram_size < 2:
            raise ValueError("invalid Engram table or N-gram size")
        if self.rfmoe_num_experts < 1:
            raise ValueError("rfmoe_num_experts must be positive")
        if self.rfmoe_backend not in {"auto", "metal", "hybrid", "host"}:
            raise ValueError("rfmoe_backend must be auto, metal, hybrid, or host")
        if not 0.0 <= float(self.engram_param_fraction) <= 1.0:
            raise ValueError("engram_param_fraction must be in [0, 1]")
        mode = str(self.attn_res_mode).lower()
        if mode not in {"kimi", "sandwich"}:
            raise ValueError("attn_res_mode must be 'kimi' or 'sandwich'")
        object.__setattr__(self, "attn_res_mode", mode)
        if self.attn_res_group_size is None:
            object.__setattr__(
                self, "attn_res_group_size", max(1, self.num_hidden_layers // 8)
            )
        elif int(self.attn_res_group_size) < 1:
            raise ValueError("attn_res_group_size must be >= 1")
        else:
            object.__setattr__(self, "attn_res_group_size", int(self.attn_res_group_size))

        # Filter Engram injects to unique stack; auto table size ~fraction of body.
        L = self.num_hidden_layers
        ids = tuple(int(i) for i in self.engram_layer_ids if 0 <= int(i) < L)
        if self.use_engram and not ids:
            ids = (0,) if L == 1 else tuple(dict.fromkeys((min(1, L - 1), L // 2)))
        object.__setattr__(self, "engram_layer_ids", ids)
        if self.engram_vocab_size is not None:
            if int(self.engram_vocab_size) < 1:
                raise ValueError("engram_vocab_size must be positive")
            object.__setattr__(self, "engram_vocab_size", int(self.engram_vocab_size))
        elif self.use_engram and ids:
            from config import _nearest_odd_table_size

            H = self.hidden_size
            I = self.intermediate_size
            emb = self.vocab_size * H
            ffn = H * (2 * I) + I * I + I * H
            per_layer = 6 * H * H + ffn + 10 * H + (4 * H if mode == "kimi" else 2 * H)
            body = emb + L * per_layer + 4 * 4 * H
            target = float(self.engram_param_fraction) * max(1, body)
            num_tables = max(1, (self.engram_max_ngram_size - 1) * self.engram_num_heads)
            hd = self.engram_head_dim
            fixed = 2 * num_tables * hd * H + H * self.engram_kernel_size + 3 * H
            budget = max(0.0, target / len(ids) - fixed)
            raw_v = int(budget / (num_tables * hd)) if num_tables * hd else 17
            object.__setattr__(self, "engram_vocab_size", _nearest_odd_table_size(raw_v))
        else:
            object.__setattr__(self, "engram_vocab_size", 4093)

    @property
    def num_hidden_layers(self) -> int:
        return self.num_prelude_layers + self.num_recurrent_layers + self.num_coda_layers

    @property
    def effective_depth(self) -> int:
        return self.num_prelude_layers + self.num_recurrent_layers * self.num_loops + self.num_coda_layers


class MLXDepthAttnMix(nn.Module):
    """Kimi AttnRes depth mix: h = softmax(wᵀ RMSNorm(V)) · V over depth axis."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False)
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, completed: list[mx.array], partial: mx.array) -> mx.array:
        if not completed:
            return partial
        v = mx.stack([*completed, partial], axis=0)  # [N+1, B, T, D]
        k = self.norm(v)
        w = mx.reshape(self.proj.weight, (-1,))  # [D]
        logits = mx.sum(k * w, axis=-1)  # [N+1, B, T]
        weights = mx.softmax(logits, axis=0)
        return mx.sum(weights[..., None] * v, axis=0)


@dataclass
class MLXAttnResStream:
    """Mutable Block AttnRes state for one depth segment (train or decode pass)."""

    completed: list[mx.array]
    partial: mx.array | None
    layers_in_block: int
    group_size: int
    last_hidden: mx.array
    attn_mix: MLXDepthAttnMix
    mlp_mix: MLXDepthAttnMix

    @classmethod
    def start(
        cls,
        seed: mx.array,
        *,
        group_size: int,
        attn_mix: MLXDepthAttnMix,
        mlp_mix: MLXDepthAttnMix,
    ) -> "MLXAttnResStream":
        if group_size < 1:
            raise ValueError("attn_res_group_size must be >= 1")
        return cls(
            completed=[seed],
            partial=seed,
            layers_in_block=0,
            group_size=int(group_size),
            last_hidden=seed,
            attn_mix=attn_mix,
            mlp_mix=mlp_mix,
        )

    def mix_attn(self) -> mx.array:
        partial = self.partial if self.partial is not None else self.completed[-1]
        h = self.attn_mix(self.completed, partial)
        self.last_hidden = h
        return h

    def mix_mlp(self) -> mx.array:
        partial = self.partial if self.partial is not None else self.completed[-1]
        h = self.mlp_mix(self.completed, partial)
        self.last_hidden = h
        return h

    def add_sublayer(self, delta: mx.array) -> None:
        self.partial = delta if self.partial is None else self.partial + delta

    def close_layer(self) -> None:
        self.layers_in_block += 1
        if self.layers_in_block < self.group_size:
            return
        if self.partial is None:
            raise RuntimeError("MLXAttnResStream.close_layer with empty partial")
        self.completed.append(self.partial)
        self.partial = None
        self.layers_in_block = 0

    def hidden(self) -> mx.array:
        if self.partial is not None:
            return self.mlp_mix(self.completed, self.partial)
        return self.completed[-1]


class MLXHBitLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, config: MLXBitNetConfig):
        super().__init__()
        self.input_dims = input_dims
        self.config = config
        self.weight = mx.random.uniform(
            low=-input_dims**-0.5,
            high=input_dims**-0.5,
            shape=(output_dims, input_dims),
        )
        self.weight_mix = mx.array(1.0)
        self.activation_mix = mx.array(1.0)
        # When set, set_quantization_state leaves weight_mix alone. Used by the
        # identity-initialised FFN mid; see MLXHybridBlock.
        self.pinned_weight_mix: float | None = None
        self._pinned_full_weight_quant = False
        # [positive, negative] for the fused quantiser. Module state, not a
        # locally built constant, so mx.compile treats it as an input and a
        # mid-run bit-width change flows through instead of being baked in.
        self.activation_level_pair = _activation_level_pair(config.activation_bits)
        self.freeze(
            keys=["weight_mix", "activation_mix", "activation_level_pair"],
            recurse=False,
        )
        # Pinned inference weights: set by pin_inference_weights(); avoid per-token rematerialization.
        self._pinned_dense: mx.array | None = None
        self._pinned_packed: tuple | None = None
        self._full_activation_quant = False
        self._act_levels_f = float((2 ** (config.activation_bits - 1)) - 1)

    def prepare_input(self, x: mx.array) -> mx.array:
        if self.config.use_hadamard and self.input_dims & (self.input_dims - 1) == 0:
            x = mx.hadamard_transform(x)
        if self.config.use_4bit_activations:
            # One fused kernel rather than the nine elementwise passes the
            # expression form costs -- absmax, divide, round, clip, rescale, and
            # the straight-through add. Measured 2-7x on the quantiser alone,
            # and it is 41-75% of a whole HBitLinear. The levels stay a runtime
            # array so ramping the bit width does not recompile the graph.
            #
            # float32 on purpose. The expression this replaced read the levels
            # from a float32 array, so bfloat16 activations were promoted and the
            # whole quantisation -- and the matmul consuming it -- ran at float32.
            # The kernel works in the input dtype, so passing x through unchanged
            # would silently drop this stack to bfloat16 quantisation: measured
            # 1180 of 1200 elements differing on a [4, 300] bf16 input. bfloat16
            # carries 8 mantissa bits and an 8-bit grid needs 255 levels, so that
            # is not a safe change to make by accident. See todo.md -- the
            # implied float32 matmul may itself be worth revisiting deliberately.
            quantized_x = ste_activation_quantize(
                x.astype(mx.float32), self.activation_level_pair
            ).astype(mx.float32)
            if self._full_activation_quant:
                return quantized_x
            x = x + self.activation_mix * mx.stop_gradient(quantized_x - x)
        return x

    def effective_weight(
        self,
        dtype,
        weight: mx.array | None = None,
        cache_key: object | None = None,
    ) -> mx.array:
        if self._pinned_dense is not None and weight is None:
            return self._pinned_dense
        cache = _effective_weight_cache.get()
        key = None
        if cache is not None:
            key = (id(self) if weight is None else cache_key, str(dtype))
            if key[0] is not None and key in cache:
                return cache[key]
        weight = self.weight if weight is None else weight
        weight_scale = mx.maximum(mx.mean(mx.abs(weight), axis=-1, keepdims=True), 1e-5)
        normalized = weight / weight_scale
        ternary = mx.where(normalized > 0.5, 1.0, mx.where(normalized < -0.5, -1.0, 0.0))
        quantized_weight = ternary * weight_scale
        # stop_gradient(q) + mix*(w - stop_gradient(w)) at mix=1, rather than
        # w + mix*stop_gradient(q - w): same value and gradient, but the latter
        # computes q - w, and the identity-initialised FFN mid has w ~ I against
        # q ~ 1, so that subtraction loses most of its significant digits.
        if self._pinned_full_weight_quant:
            effective = (mx.stop_gradient(quantized_weight) + (weight - mx.stop_gradient(weight))).astype(dtype)
        else:
            effective = (weight + self.weight_mix * mx.stop_gradient(quantized_weight - weight)).astype(dtype)
        if cache is not None and key is not None and key[0] is not None:
            cache[key] = effective
        return effective

    def pin_inference_weight(self, dtype, *, prefer_packed: bool = True) -> None:
        """Materialize one effective/packed weight for the whole generation lifetime."""
        self._pinned_dense = None
        self._pinned_packed = None
        can_pack = (
            prefer_packed
            and _recurrent_quantized_matmul.get()
            and self.weight.shape[-1] % 32 == 0
            # Prefer packed ternary GEMV whenever packing is valid (decode M=1 path uses fused kernel).
            and min(self.weight.shape) >= 32
        )
        if can_pack:
            try:
                self._pinned_packed = pack_ternary_weight(self.weight)
                return
            except ValueError:
                self._pinned_packed = None
        self._pinned_dense = self.effective_weight(dtype)

    def clear_pinned_inference_weight(self) -> None:
        self._pinned_dense = None
        self._pinned_packed = None

    def _packed_weight(self, x: mx.array):
        """Return (packed, scales, group_size) from pin or generation weight cache."""
        if self._pinned_packed is not None:
            return self._pinned_packed
        if not (_recurrent_quantized_matmul.get() and self.weight.shape[-1] % 32 == 0):
            return None
        cache = _effective_weight_cache.get()
        key = (id(self), f"packed-{x.dtype}")
        packed_weight = cache.get(key) if cache is not None else None
        if packed_weight is None:
            try:
                packed_weight = pack_ternary_weight(self.weight)
            except ValueError:
                return None
            if cache is not None:
                cache[key] = packed_weight
        return packed_weight

    def __call__(self, x: mx.array) -> mx.array:
        # Decode fast path: M=1 fused act-quant + ternary add/sub GEMV (one Metal dispatch).
        in_dim = int(self.weight.shape[1])
        out_dim = int(self.weight.shape[0])
        tokens = int(x.size // max(in_dim, 1))
        # Custom ternary M=1 kernel helps small/medium shapes (launch-bound + fused act quant).
        # At 1B widths (K/N ~1024+) mx.quantized_matmul is faster — skip the custom GEMV.
        use_fused_m1 = (
            tokens == 1
            and in_dim <= 512
            and out_dim <= 1024
            and in_dim % 32 == 0
        )
        if use_fused_m1:
            packed_weight = self._packed_weight(x)
            if packed_weight is not None:
                packed, scales, group_size = packed_weight
                if self.config.use_hadamard and in_dim & (in_dim - 1) == 0:
                    x = mx.hadamard_transform(x)
                if self.config.use_4bit_activations and not self._full_activation_quant:
                    # Partial quant mix (training): keep STE prepare + generic quantized matmul.
                    x = self.prepare_input(x)
                    return ternary_quantized_linear(x, self.weight, packed, scales)
                quantize_acts = bool(self.config.use_4bit_activations and self._full_activation_quant)
                return ternary_fused_linear_m1(
                    x,
                    packed,
                    scales,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    group_size=int(group_size),
                    quantize_acts=quantize_acts,
                    act_levels=self._act_levels_f,
                    dtype=x.dtype,
                )

        x = self.prepare_input(x)
        if self._pinned_packed is not None:
            packed, scales, _ = self._pinned_packed
            return ternary_quantized_linear(x, self.weight, packed, scales)
        if self._pinned_dense is not None:
            return x @ self._pinned_dense.T
        packed_weight = self._packed_weight(x)
        if packed_weight is not None:
            packed, scales, _ = packed_weight
            return ternary_quantized_linear(x, self.weight, packed, scales)
        return x @ self.effective_weight(x.dtype).T

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        if self.pinned_weight_mix is not None:
            weight_mix = self.pinned_weight_mix
        self._pinned_full_weight_quant = self.pinned_weight_mix is not None and weight_mix >= 1.0
        self.weight_mix = mx.array(weight_mix)
        self.activation_mix = mx.array(activation_mix)
        levels = float((2 ** (max(bits, 2) - 1)) - 1)
        self.activation_level_pair = _activation_level_pair(bits)
        self._act_levels_f = levels
        self._full_activation_quant = activation_mix >= 1.0
        self.clear_pinned_inference_weight()


class MLXEngram(nn.Module):
    def __init__(self, config: MLXBitNetConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.num_tables = (config.engram_max_ngram_size - 1) * config.engram_num_heads
        self.embedding = nn.Embedding(
            self.num_tables * config.engram_vocab_size,
            config.engram_head_dim,
        )
        self.offsets = mx.arange(self.num_tables, dtype=mx.int64) * config.engram_vocab_size
        rng = np.random.default_rng(config.engram_seed + 10_007 * layer_id)
        multipliers = rng.integers(
            1,
            2**31,
            size=(config.engram_max_ngram_size, config.engram_num_heads),
            dtype=np.int64,
        )
        self.multipliers = mx.array(multipliers | 1)
        memory_size = self.num_tables * config.engram_head_dim
        self.key_proj = nn.Linear(memory_size, config.hidden_size, bias=False)
        self.value_proj = nn.Linear(memory_size, config.hidden_size, bias=False)
        self.key_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.query_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.short_conv_weight = mx.random.uniform(
            low=-config.engram_kernel_size**-0.5,
            high=config.engram_kernel_size**-0.5,
            shape=(config.hidden_size, config.engram_kernel_size),
        )
        self.freeze(keys=["offsets", "multipliers"], recurse=False)

    def hash_ids(self, input_ids: mx.array, segment_ids: mx.array | None = None) -> mx.array:
        hashes = []
        for ngram_size in range(2, self.config.engram_max_ngram_size + 1):
            mixed = input_ids[..., None] * self.multipliers[0]
            for lag in range(1, ngram_size):
                if lag >= input_ids.shape[1]:
                    shifted = mx.full(input_ids.shape, self.config.engram_pad_id, dtype=input_ids.dtype)
                    keep = mx.zeros(input_ids.shape, dtype=mx.bool_)
                else:
                    shifted = mx.concatenate(
                        (
                            mx.full((input_ids.shape[0], lag), self.config.engram_pad_id, dtype=input_ids.dtype),
                            input_ids[:, :-lag],
                        ),
                        axis=1,
                    )
                    keep = mx.concatenate(
                        (
                            mx.zeros((input_ids.shape[0], lag), dtype=mx.bool_),
                            mx.ones((input_ids.shape[0], input_ids.shape[1] - lag), dtype=mx.bool_),
                        ),
                        axis=1,
                    )
                if segment_ids is not None:
                    if lag >= segment_ids.shape[1]:
                        same = mx.zeros(segment_ids.shape, dtype=mx.bool_)
                    else:
                        same = segment_ids[:, lag:] == segment_ids[:, :-lag]
                        same = mx.concatenate((mx.zeros((input_ids.shape[0], lag), dtype=mx.bool_), same), axis=1)
                    keep = keep & same
                shifted = mx.where(keep, shifted, self.config.engram_pad_id)
                mixed = mx.bitwise_xor(mixed, shifted[..., None] * self.multipliers[lag])
            hashes.append(mx.remainder(mixed, self.config.engram_vocab_size))
        return mx.concatenate(hashes, axis=-1)

    def _values(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        segment_ids: mx.array | None,
        tail_length: int | None = None,
    ):
        hashes = self.hash_ids(input_ids, segment_ids)
        hashes = hashes[:, -tail_length:] if tail_length is not None else hashes
        memory = self.embedding(hashes + self.offsets)
        memory = memory.reshape(*memory.shape[:-2], -1)
        key = self.key_norm(self.key_proj(memory))
        query = self.query_norm(hidden_states)
        score = mx.sum(key * query, axis=-1) / math.sqrt(hidden_states.shape[-1])
        score = mx.sign(score) * mx.sqrt(mx.maximum(mx.abs(score), 1e-6))
        value = mx.sigmoid(score)[..., None] * self.value_proj(memory)
        normalized = self.conv_norm(value)
        return value, normalized

    def _convolve(self, normalized: mx.array, segment_ids: mx.array | None = None) -> mx.array:
        convolved = mx.zeros_like(normalized)
        for lag in range(self.config.engram_kernel_size):
            if lag == 0:
                shifted = normalized
            elif lag >= normalized.shape[1]:
                shifted = mx.zeros_like(normalized)
            else:
                shifted = mx.concatenate((mx.zeros_like(normalized[:, :lag]), normalized[:, :-lag]), axis=1)
            if segment_ids is not None and 0 < lag < segment_ids.shape[1]:
                same = segment_ids[:, lag:] == segment_ids[:, :-lag]
                same = mx.concatenate((mx.zeros((segment_ids.shape[0], lag), dtype=mx.bool_), same), axis=1)
                shifted = shifted * same[..., None]
            weight = self.short_conv_weight[:, self.config.engram_kernel_size - 1 - lag]
            convolved = convolved + shifted * weight
        return convolved

    def __call__(self, hidden_states: mx.array, input_ids: mx.array, segment_ids: mx.array | None) -> mx.array:
        value, normalized = self._values(hidden_states, input_ids, segment_ids)
        return value + nn.silu(self._convolve(normalized, segment_ids))

    def prefill(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: "MLXEngramInferenceCache",
    ) -> mx.array:
        value, normalized = self._values(hidden_states, input_ids, None)
        cache.normalized = normalized[:, -(self.config.engram_kernel_size - 1) :]
        return value + nn.silu(self._convolve(normalized))

    def extend(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        token_history: mx.array,
        cache: "MLXEngramInferenceCache",
    ) -> mx.array:
        ids = mx.concatenate((token_history, input_ids), axis=1)
        value, normalized = self._values(hidden_states, ids, None, input_ids.shape[1])
        combined = normalized if cache.normalized is None else mx.concatenate((cache.normalized, normalized), axis=1)
        convolved = self._convolve(combined)[:, -input_ids.shape[1] :]
        cache.normalized = combined[:, -(self.config.engram_kernel_size - 1) :]
        return value + nn.silu(convolved)

    def incremental(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        token_history: mx.array,
        cache: "MLXEngramInferenceCache",
    ) -> mx.array:
        ids = mx.concatenate((token_history, input_ids), axis=1)
        if ids.shape[1] < self.config.engram_max_ngram_size:
            padding = mx.full(
                (ids.shape[0], self.config.engram_max_ngram_size - ids.shape[1]),
                self.config.engram_pad_id,
                dtype=ids.dtype,
            )
            ids = mx.concatenate((padding, ids), axis=1)
        value, normalized = self._values(hidden_states, ids, None, 1)
        history = cache.normalized
        kernel = self.config.engram_kernel_size
        convolved = normalized * self.short_conv_weight[:, kernel - 1]
        for lag in range(1, kernel):
            if history is not None and history.shape[1] >= lag:
                index = history.shape[1] - lag
                convolved = convolved + history[:, index : index + 1] * self.short_conv_weight[:, kernel - 1 - lag]
        if history is None:
            cache.normalized = normalized
        else:
            cache.normalized = mx.concatenate((history, normalized), axis=1)
        keep = self.config.engram_kernel_size - 1
        if cache.normalized.shape[1] > keep:
            start = cache.normalized.shape[1] - keep
            cache.normalized = cache.normalized[:, start : start + keep]
        return value + nn.silu(convolved)


@dataclass
class MLXEngramInferenceCache:
    normalized: mx.array | None = None

    def clone(self) -> "MLXEngramInferenceCache":
        return MLXEngramInferenceCache(self.normalized)


class MLXRFMoEExpert(nn.Module):
    def __init__(self, config: MLXBitNetConfig, expert_dim: int, rank: int):
        super().__init__()
        hidden = config.hidden_size
        self.a_gate = MLXHBitLinear(hidden, rank, config)
        self.b_gate = MLXHBitLinear(rank, expert_dim, config)
        self.w_up = MLXHBitLinear(hidden, expert_dim, config)
        self.w_mid = MLXHBitLinear(expert_dim, expert_dim, config)
        # Cold start: identity mid ~ classic 2-mat expert body.
        # eye(D)*D, not eye(D): the per-output-channel scale is mean(|row|), so a
        # plain identity row (one 1, D-1 zeros) has scale 1/D and quantises to
        # eye(D)/D -- an attenuator, not a pass-through. See TernaryMLP.mid_proj.
        self.w_mid.weight = mx.eye(expert_dim, dtype=self.w_mid.weight.dtype) * expert_dim
        self.w_mid.pinned_weight_mix = 1.0
        self.w_mid.weight_mix = mx.array(1.0)
        self.w_mid._pinned_full_weight_quant = True
        self.w_down = MLXHBitLinear(expert_dim, hidden, config)
        self.bias = mx.array([1e-6])

    def __call__(self, x: mx.array):
        z = self.a_gate(x)
        gate = mx.maximum(_safe_norm(z, axis=-1) - self.bias, 0.0)
        hidden = mx.sigmoid(self.b_gate(z)) * self.w_up(x)
        contribution = self.w_down(nn.silu(self.w_mid(hidden)))
        return gate, contribution


class MLXRFMoE(nn.Module):
    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        expert_dim = config.rfmoe_expert_dim or config.intermediate_size // 4
        rank = config.rfmoe_rank or max(1, config.hidden_size // 16)
        self.theta = config.rfmoe_theta
        self.backend = (
            "hybrid"
            if config.rfmoe_backend == "auto" and mx.metal.is_available()
            else "host"
            if config.rfmoe_backend == "auto"
            else config.rfmoe_backend
        )
        self.experts = [MLXRFMoEExpert(config, expert_dim, rank) for _ in range(config.rfmoe_num_experts)]
        self.usage_ema = mx.full((config.rfmoe_num_experts,), 1.0 / config.rfmoe_num_experts)
        self.last_usage = mx.zeros((config.rfmoe_num_experts,))
        self.last_gate = mx.zeros((config.rfmoe_num_experts, 1))
        self.last_density = mx.array(0.0)
        self.freeze(keys=["usage_ema", "last_usage", "last_gate", "last_density"], recurse=False)

    @staticmethod
    def _grouped_linear(x: mx.array, layers: list[MLXHBitLinear], expert_indices: mx.array) -> mx.array:
        prepared = layers[0].prepare_input(x)
        weights = mx.stack([layer.weight for layer in layers])
        weights = layers[0].effective_weight(
            prepared.dtype,
            weights,
            tuple(id(layer) for layer in layers),
        ).swapaxes(-1, -2)
        return mx.squeeze(
            mx.gather_mm(
                prepared[:, None, :],
                weights,
                rhs_indices=expert_indices,
            ),
            axis=1,
        )

    @staticmethod
    def _metal_grouped_linear(x: mx.array, layers: list[MLXHBitLinear], active: mx.array) -> mx.array:
        shape = x.shape
        prepared = layers[0].prepare_input(x.reshape(-1, shape[-1])).reshape(shape)
        weights = layers[0].effective_weight(
            prepared.dtype,
            mx.stack([layer.weight for layer in layers]),
            tuple(id(layer) for layer in layers),
        )
        return masked_grouped_linear(prepared, weights, active)

    @staticmethod
    def _hybrid_grouped_linear(
        x: mx.array,
        layers: list[MLXHBitLinear],
        active: mx.array,
        expert_indices: mx.array,
        token_indices: mx.array,
        expert_offsets: mx.array,
    ) -> mx.array:
        shape = x.shape
        prepared = layers[0].prepare_input(x.reshape(-1, shape[-1])).reshape(shape)
        weights = layers[0].effective_weight(
            prepared.dtype,
            mx.stack([layer.weight for layer in layers]),
            tuple(id(layer) for layer in layers),
        )
        return compacted_grouped_linear(
            prepared,
            weights,
            active,
            expert_indices,
            token_indices,
            expert_offsets,
        )

    def _scores(self, flat: mx.array):
        a_gates = [expert.a_gate for expert in self.experts]
        score_input = a_gates[0].prepare_input(flat)
        a_gate = a_gates[0].effective_weight(
            score_input.dtype,
            mx.stack([layer.weight for layer in a_gates]),
            tuple(id(layer) for layer in a_gates),
        )
        z = mx.einsum("td,erd->etr", score_input, a_gate)
        biases = mx.stack([expert.bias for expert in self.experts])
        gate_stack = mx.maximum(_safe_norm(z, axis=-1) - biases, 0.0)
        return z, gate_stack

    def _metal_forward_arrays(self, x: mx.array):
        flat = x.reshape(-1, x.shape[-1])
        z, gate_stack = self._scores(flat)
        active = gate_stack >= self.theta
        b_gates = [expert.b_gate for expert in self.experts]
        w_ups = [expert.w_up for expert in self.experts]
        w_mids = [expert.w_mid for expert in self.experts]
        w_downs = [expert.w_down for expert in self.experts]
        gate = mx.sigmoid(self._metal_grouped_linear(z, b_gates, active))
        hidden = gate * self._metal_grouped_linear(flat[None, :, :], w_ups, active)
        hidden = nn.silu(self._metal_grouped_linear(hidden, w_mids, active))
        contribution = self._metal_grouped_linear(hidden, w_downs, active)
        output = mx.sum(gate_stack[..., None] * contribution, axis=0)
        return output.reshape(x.shape), gate_stack, mx.mean(active)

    def _host_forward_arrays(self, x: mx.array):
        flat = x.reshape(-1, x.shape[-1])
        z, gate_stack = self._scores(flat)

        active = np.argwhere(np.array(gate_stack >= self.theta))
        output = mx.zeros_like(flat)
        if active.size:
            expert_indices = mx.array(active[:, 0], dtype=mx.uint32)
            token_indices = mx.array(active[:, 1], dtype=mx.uint32)
            selected_z = z[expert_indices, token_indices]
            selected_gates = gate_stack[expert_indices, token_indices]
            b_gates = [expert.b_gate for expert in self.experts]
            w_ups = [expert.w_up for expert in self.experts]
            w_mids = [expert.w_mid for expert in self.experts]
            w_downs = [expert.w_down for expert in self.experts]
            gate = mx.sigmoid(self._grouped_linear(selected_z, b_gates, expert_indices))
            hidden = gate * self._grouped_linear(flat[token_indices], w_ups, expert_indices)
            hidden = nn.silu(self._grouped_linear(hidden, w_mids, expert_indices))
            contribution = self._grouped_linear(hidden, w_downs, expert_indices)
            output = output.at[token_indices].add(selected_gates[:, None] * contribution)
        hard_density = mx.array(active.shape[0] / max(gate_stack.size, 1))
        return output.reshape(x.shape), gate_stack, hard_density

    def _hybrid_forward_arrays(self, x: mx.array):
        flat = x.reshape(-1, x.shape[-1])
        z, gate_stack = self._scores(flat)
        active = gate_stack >= self.theta
        active_pairs = np.argwhere(np.array(active))
        expert_indices = mx.array(active_pairs[:, 0], dtype=mx.uint32)
        token_indices = mx.array(active_pairs[:, 1], dtype=mx.uint32)
        expert_offsets = mx.array(
            np.searchsorted(active_pairs[:, 0], np.arange(len(self.experts) + 1)),
            dtype=mx.uint32,
        )
        b_gates = [expert.b_gate for expert in self.experts]
        w_ups = [expert.w_up for expert in self.experts]
        w_mids = [expert.w_mid for expert in self.experts]
        w_downs = [expert.w_down for expert in self.experts]
        gate = mx.sigmoid(
            self._hybrid_grouped_linear(
                z,
                b_gates,
                active,
                expert_indices,
                token_indices,
                expert_offsets,
            )
        )
        hidden = gate * self._hybrid_grouped_linear(
            flat[None, :, :],
            w_ups,
            active,
            expert_indices,
            token_indices,
            expert_offsets,
        )
        hidden = nn.silu(self._hybrid_grouped_linear(
            hidden,
            w_mids,
            active,
            expert_indices,
            token_indices,
            expert_offsets,
        ))
        contribution = self._hybrid_grouped_linear(
            hidden,
            w_downs,
            active,
            expert_indices,
            token_indices,
            expert_offsets,
        )
        output = mx.sum(gate_stack[..., None] * contribution, axis=0)
        return output.reshape(x.shape), gate_stack, mx.mean(active)

    def forward_arrays(self, x: mx.array):
        if self.backend == "metal":
            return self._metal_forward_arrays(x)
        if self.backend == "hybrid":
            return self._hybrid_forward_arrays(x)
        return self._host_forward_arrays(x)

    def update_stats(self, gate_stack: mx.array, hard_density: mx.array) -> None:
        if self.training:
            self.last_gate = gate_stack
            self.last_usage = mx.mean(gate_stack, axis=1)
            self.last_density = hard_density
            self.usage_ema = 0.99 * self.usage_ema + 0.01 * mx.stop_gradient(self.last_usage)

    def __call__(self, x: mx.array) -> mx.array:
        output, gate_stack, hard_density = self.forward_arrays(x)
        self.update_stats(gate_stack, hard_density)
        return output

    def aux_losses(self, s: float, alpha: float):
        density = mx.mean(self.last_usage)
        usage = self.last_usage / mx.maximum(mx.sum(self.last_usage), 1e-8)
        order = mx.argsort(self.usage_ema)[::-1]
        ranked = usage[order]
        ranks = mx.arange(1, len(self.experts) + 1, dtype=mx.float32)
        target = ranks ** (-s)
        target = (1.0 - alpha) * target / mx.sum(target) + alpha / len(self.experts)
        locality = mx.sum(target * (mx.log(mx.maximum(target, 1e-8)) - mx.log(mx.maximum(ranked, 1e-8))))
        centered = self.last_gate - mx.mean(self.last_gate, axis=1, keepdims=True)
        normalized = _safe_normalize(centered, axis=1, floor=1e-8)
        correlation = normalized @ normalized.T
        off_diagonal = correlation - mx.diag(mx.diag(correlation))
        diversity = mx.sum(mx.square(mx.maximum(off_diagonal, 0.0))) / max(
            len(self.experts) * (len(self.experts) - 1), 1
        )
        return density, locality, diversity


class MLXLoopHyperConnection(nn.Module):
    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        flat = 4 * config.hidden_size
        self.norm = nn.RMSNorm(flat, eps=config.rms_norm_eps)
        self.w_pre = nn.Linear(flat, 4)
        self.w_post = nn.Linear(flat, 4)
        self.w_res = nn.Linear(flat, 4)
        self.alpha_pre = mx.array([1.0])
        self.alpha_post = mx.array([1.0])
        self.alpha_res = mx.array([1.0])
        self.loop_embed = nn.Embedding(64, config.hidden_size)
        self.w_pre.weight = mx.zeros_like(self.w_pre.weight)
        self.w_pre.bias = mx.full(self.w_pre.bias.shape, -math.log(3.0))
        self.w_post.weight = mx.zeros_like(self.w_post.weight)
        self.w_post.bias = mx.full(self.w_post.bias.shape, -3.0)
        self.w_res.weight = mx.zeros_like(self.w_res.weight)
        self.w_res.bias = mx.full(self.w_res.bias.shape, 3.0)
        self.loop_embed.weight = mx.zeros_like(self.loop_embed.weight)

    @staticmethod
    def expand(x: mx.array) -> mx.array:
        return mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], 4, x.shape[-1]))

    @staticmethod
    def fold(y: mx.array) -> mx.array:
        return mx.mean(y, axis=2)

    def project_in(self, y: mx.array):
        z = self.norm(y.reshape(*y.shape[:2], -1))
        pre = mx.sigmoid(self.alpha_pre * self.w_pre(z))
        post = 2.0 * mx.sigmoid(self.alpha_post * self.w_post(z))
        residual = mx.sigmoid(self.alpha_res * self.w_res(z))
        return mx.sum(pre[..., None] * y, axis=2), post, residual

    @staticmethod
    def write_back(y: mx.array, output: mx.array, post: mx.array, residual: mx.array) -> mx.array:
        return residual[..., None] * y + post[..., None] * output[:, :, None, :]


class MLXPaTHAttention(nn.Module):
    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        self.config = config
        self.num_blocks = config.block_size
        self.fixed_block_width = None
        hidden = config.hidden_size
        self.head_dim = hidden // config.num_attention_heads
        self.qkv = MLXHBitLinear(hidden, hidden * 3, config)
        self.out = MLXHBitLinear(hidden, hidden, config)
        path_rank = min(32, hidden)
        self.path_down = MLXHBitLinear(hidden, path_rank, config)
        self.path_up = MLXHBitLinear(path_rank, hidden, config)
        self.path_conv_weight = mx.random.normal((hidden, 3)) * 0.02
        self.path_beta = nn.Linear(hidden, config.num_attention_heads)
        self.path_forget = nn.Linear(hidden, config.num_attention_heads)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.memory_gate = mx.zeros((config.num_attention_heads,))
        if config.use_topk_blocks:
            # -2.0 (sigmoid ~0.12) so retrieval starts as a nudge the model can grow
            # into. The memory gate's 0.0 init puts half the output into an untrained
            # branch, which measurably costs perplexity before it learns its way out.
            # Only registered when enabled, so existing checkpoints still resume.
            self.topk_gate = mx.full((config.num_attention_heads,), -2.0)
        self.out.weight = self.out.weight * 0.01
        # Paper Infini: M (B,H,Dm,Dv), z (B,H,Dm). Dm is the key-feature dim and is
        # what bounds capacity — M holds Dm*Dv numbers per head, so it stores the
        # window losslessly once Dm >= tokens-per-reset. Dv stays head_dim.
        self.memory_dim = int(config.infini_memory_expand) or self.head_dim
        if self.memory_dim != self.head_dim:
            # Shared across heads on purpose: capacity belongs in the per-head M,
            # not in the projection, so widening Dm costs ~no extra parameters.
            self.memory_proj = nn.Linear(self.head_dim, self.memory_dim, bias=False)
        d = self.head_dim
        h = config.num_attention_heads
        self.memory_m = mx.zeros((0, h, self.memory_dim, d), dtype=mx.float32)
        self.memory_z = mx.zeros((0, h, self.memory_dim), dtype=mx.float32)
        self.memory_initialized = mx.zeros((0,), dtype=mx.bool_)
        self.freeze(keys=["memory_m", "memory_z", "memory_initialized"], recurse=False)

    def reset_memory(self, batch_size: int | None = None) -> None:
        batch_size = self.memory_m.shape[0] if batch_size is None else batch_size
        d = self.head_dim
        h = self.config.num_attention_heads
        self.memory_m = mx.zeros((batch_size, h, self.memory_dim, d), dtype=mx.float32)
        self.memory_z = mx.zeros((batch_size, h, self.memory_dim), dtype=mx.float32)
        self.memory_initialized = mx.zeros((batch_size,), dtype=mx.bool_)

    def new_inference_cache(self, batch_size: int) -> "MLXPaTHInferenceCache":
        if self.config.use_topk_blocks:
            raise NotImplementedError(
                "use_topk_blocks is a training-only A/B branch: the decode cache keeps "
                "only the open chunk and the Infini banks, so past blocks are not "
                "retrievable. Train with it to compare perplexity, then disable it to "
                "generate."
            )
        d = self.head_dim
        h = self.config.num_attention_heads
        return MLXPaTHInferenceCache(
            memory_m=mx.zeros((batch_size, h, self.memory_dim, d), dtype=mx.float32),
            memory_z=mx.zeros((batch_size, h, self.memory_dim), dtype=mx.float32),
            memory_initialized=mx.zeros((batch_size,), dtype=mx.bool_),
        )

    def _memory_features(self, x: mx.array) -> mx.array:
        """phi() of the key/query features, widened to ``memory_dim`` when asked."""
        projected = self.memory_proj(x) if self.memory_dim != self.head_dim else x
        if self.config.infini_feature_map == "favor":
            # exp kernel: phi(q).phi(k) approximates exp(q.k), the kernel softmax
            # attention uses. ELU+1's constant baseline makes the weights nearly
            # uniform, so every query reads back the same vector regardless of how
            # big M is. Row max is subtracted for overflow safety.
            norm = mx.sum(x * x, axis=-1, keepdims=True) / 2
            shift = mx.max(projected, axis=-1, keepdims=True)
            return mx.exp(projected - norm - shift)
        return _infini_sigma(projected)

    def _retrieve_memory(
        self,
        q: mx.array,
        memory_m: mx.array,
        memory_z: mx.array,
    ) -> mx.array:
        """A_mem = σ(Q) M / (σ(Q) z).

        M and z stay in the graph so gradient reaches whatever wrote them. Detaching
        here leaves only σ(Q) trainable, which turns the memory into a fixed
        content-free blur worth ~0 perplexity. ``_next_memory`` bounds how far back
        that gradient runs.
        """
        sq = self._memory_features(q.astype(mx.float32))
        numerator = sq @ memory_m
        denominator = mx.sum(sq * memory_z[:, :, None, :], axis=-1, keepdims=True)
        denominator = mx.maximum(denominator, _MEMORY_EPS)
        return numerator / denominator

    def _next_memory(
        self,
        keys: mx.array,
        values: mx.array,
        rows: mx.array,
        memory_m: mx.array,
        memory_z: mx.array,
        memory_initialized: mx.array,
    ):
        """Paper Linear or Linear+Delta update; rows masks batch items that may write.

        Truncated BPTT, window 1: prior history is detached but this chunk's K/V stay
        live, so the next chunk's read still teaches the model *what* to store. Full
        BPTT across every chunk would be the textbook version, but the chain grows
        with block_size and mx.compile dies on it ("unordered_map::at: key not found")
        at block_size >= 7 under kimi AttnRes. Depth here is constant in block_size.
        """
        memory_m = mx.stop_gradient(memory_m)
        memory_z = mx.stop_gradient(memory_z)
        sk = self._memory_features(keys.astype(mx.float32))
        vf = values.astype(mx.float32)
        if self.config.infini_delta_rule:
            den = mx.sum(sk * memory_z[:, :, None, :], axis=-1, keepdims=True)
            den = mx.maximum(den, _MEMORY_EPS)
            retrieved = (sk @ memory_m) / den
            delta_v = vf - retrieved
            m_update = sk.swapaxes(-1, -2) @ delta_v
        else:
            m_update = sk.swapaxes(-1, -2) @ vf
        z_update = mx.sum(sk, axis=2)
        rows_m = rows[:, None, None, None]
        rows_z = rows[:, None, None]
        next_m = mx.where(rows_m, memory_m + m_update, memory_m)
        next_z = mx.where(rows_z, memory_z + z_update, memory_z)
        return next_m, next_z, memory_initialized | rows

    def _topk_context(
        self, q_chunk: mx.array, keys: mx.array, values: mx.array, start: int
    ) -> mx.array | None:
        """Softmax attention over the top-k most similar whole blocks before ``start``.

        Past tokens are split into ``topk_block_size`` blocks scored by mean-key
        against the chunk's mean query; the best ``topk_blocks`` are attended in
        full. Every selected token precedes the chunk, so causality needs no mask.
        Returns None when no whole block has accumulated yet.
        """
        block = self.config.topk_block_size
        num_blocks = start // block
        if num_blocks == 0:
            return None
        take = min(self.config.topk_blocks, num_blocks)
        batch, heads, _, dim = keys.shape
        shape = (batch, heads, num_blocks, block, dim)
        key_blocks = keys[:, :, : num_blocks * block].reshape(shape)
        value_blocks = values[:, :, : num_blocks * block].reshape(shape)

        summary = mx.mean(key_blocks.astype(mx.float32), axis=3)
        probe = mx.mean(q_chunk.astype(mx.float32), axis=2, keepdims=True)
        scores = mx.sum(summary * probe, axis=-1)
        chosen = mx.argpartition(-scores, kth=take - 1, axis=-1)[..., :take]
        chosen = chosen[:, :, :, None, None]
        selected_k = mx.take_along_axis(key_blocks, chosen, axis=2).reshape(
            batch, heads, take * block, dim
        )
        selected_v = mx.take_along_axis(value_blocks, chosen, axis=2).reshape(
            batch, heads, take * block, dim
        )
        logits = (q_chunk @ selected_k.swapaxes(-1, -2)) * self.head_dim**-0.5
        weights = mx.softmax(logits.astype(mx.float32), axis=-1).astype(values.dtype)
        return weights @ selected_v

    @staticmethod
    def _shift(projected: mx.array, segment_ids: mx.array | None, offset: int) -> mx.array:
        padding = mx.zeros_like(projected[:, :offset])
        shifted = mx.concatenate((padding, projected[:, :-offset]), axis=1)
        if segment_ids is not None:
            same = segment_ids[:, offset:] == segment_ids[:, :-offset]
            leading = mx.zeros(segment_ids[:, :offset].shape, dtype=mx.bool_)
            same = mx.concatenate((leading, same), axis=1)
            shifted = shifted * same[..., None]
        return shifted

    def path_vectors(self, x: mx.array, segment_ids: mx.array | None) -> mx.array:
        vectors, _ = self._path_vectors(x, segment_ids)
        return vectors

    def _path_vectors(self, x: mx.array, segment_ids: mx.array | None):
        projected = self.path_up(self.path_down(x))
        convolved = (
            projected * self.path_conv_weight[:, 2]
            + self._shift(projected, segment_ids, 1) * self.path_conv_weight[:, 1]
            + self._shift(projected, segment_ids, 2) * self.path_conv_weight[:, 0]
        )
        convolved = nn.silu(convolved.astype(mx.float32)).reshape(
            x.shape[0], x.shape[1], self.config.num_attention_heads, self.head_dim
        )
        vectors = _safe_normalize(convolved)
        return vectors, projected

    def _project(self, x: mx.array, segment_ids: mx.array | None):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            length,
            3,
            self.config.num_attention_heads,
            self.head_dim,
        )
        q = self.q_norm(qkv[:, :, 0].transpose(0, 2, 1, 3))
        k = self.k_norm(qkv[:, :, 1].transpose(0, 2, 1, 3))
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        w, projected = self._path_vectors(x, segment_ids)
        beta = 2.0 * mx.sigmoid(self.path_beta(x).astype(mx.float32))
        forget_logits = self.path_forget(x).astype(mx.float32)
        log_forget = -mx.logaddexp(mx.zeros_like(forget_logits), -forget_logits)
        return q, k, v, w, beta, log_forget, projected

    def _project_extension(self, x: mx.array, cache: "MLXPaTHInferenceCache"):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            length,
            3,
            self.config.num_attention_heads,
            self.head_dim,
        )
        q = self.q_norm(qkv[:, :, 0].transpose(0, 2, 1, 3))
        k = self.k_norm(qkv[:, :, 1].transpose(0, 2, 1, 3))
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        projected = self.path_up(self.path_down(x))
        combined = projected if cache.path_projected is None else mx.concatenate((cache.path_projected, projected), axis=1)
        convolved = combined * self.path_conv_weight[:, 2]
        for lag, weight_index in ((1, 1), (2, 0)):
            if lag < combined.shape[1]:
                shifted = mx.concatenate((mx.zeros_like(combined[:, :lag]), combined[:, :-lag]), axis=1)
                convolved = convolved + shifted * self.path_conv_weight[:, weight_index]
        convolved = nn.silu(convolved[:, -length:].astype(mx.float32)).reshape(
            batch, length, self.config.num_attention_heads, self.head_dim
        )
        w = _safe_normalize(convolved)
        beta = 2.0 * mx.sigmoid(self.path_beta(x).astype(mx.float32))
        forget_logits = self.path_forget(x).astype(mx.float32)
        log_forget = -mx.logaddexp(mx.zeros_like(forget_logits), -forget_logits)
        cache.path_projected = combined[:, -2:]
        return q, k, v, w, beta, log_forget

    def incremental(
        self,
        x: mx.array,
        cache: "MLXPaTHInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        batch, _, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, 1, 3, self.config.num_attention_heads, self.head_dim)
        q = self.q_norm(qkv[:, :, 0].transpose(0, 2, 1, 3))
        k = self.k_norm(qkv[:, :, 1].transpose(0, 2, 1, 3))
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        projected = self.path_up(self.path_down(x))
        convolved = projected * self.path_conv_weight[:, 2]
        if cache.path_projected is not None and cache.path_projected.shape[1] > 0:
            history = cache.path_projected
            hist_len = history.shape[1]
            if hist_len >= 1:
                convolved = convolved + history[:, hist_len - 1 : hist_len] * self.path_conv_weight[:, 1]
            if hist_len >= 2:
                convolved = convolved + history[:, hist_len - 2 : hist_len - 1] * self.path_conv_weight[:, 0]
            combined = mx.concatenate((history, projected), axis=1)
            clen = combined.shape[1]
            cache.path_projected = combined if clen <= 2 else combined[:, clen - 2 : clen]
        else:
            cache.path_projected = projected
        w = nn.silu(convolved.astype(mx.float32)).reshape(
            batch, 1, self.config.num_attention_heads, self.head_dim
        )
        w = _safe_normalize(w)
        beta = 2.0 * mx.sigmoid(self.path_beta(x).astype(mx.float32))
        forget_logits = self.path_forget(x).astype(mx.float32)
        log_forget = -mx.logaddexp(mx.zeros_like(forget_logits), -forget_logits)

        if cache.q is None or cache.open_len == 0:
            cache.q, cache.k, cache.v = q, k, v
            cache.w, cache.beta, cache.log_forget = w, beta, log_forget
            cache.open_len = 1
            cache.t_inverse = None
        else:
            cache.q = mx.concatenate((cache.q, q), axis=2)
            cache.k = mx.concatenate((cache.k, k), axis=2)
            cache.v = mx.concatenate((cache.v, v), axis=2)
            cache.w = mx.concatenate((cache.w, w), axis=1)
            cache.beta = mx.concatenate((cache.beta, beta), axis=1)
            cache.log_forget = mx.concatenate((cache.log_forget, log_forget), axis=1)
            cache.open_len = cache.open_len + 1

        if _path_decode_mode.get() == "recompute":
            full = self.path_chunk(cache.q, cache.k, cache.v, cache.w, cache.beta, cache.log_forget, None)
            last = cache.open_len - 1
            local = full[:, :, last : last + 1]
            # Keep running T in sync for mixed-mode / branch clones.
            cache.t_inverse = self.path_system_t_inverse(cache.w, cache.beta)
        else:
            cache.t_inverse = self.path_border_update_t(cache.t_inverse, cache.w, cache.beta)
            local = self.path_chunk_last_with_t(
                cache.q, cache.k, cache.v, cache.w, cache.beta, cache.log_forget, cache.t_inverse, None
            )
        memory_context = self._retrieve_memory(q, cache.memory_m, cache.memory_z).astype(v.dtype)
        gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
        mixed = (1.0 - gate) * local + gate * memory_context
        local = mx.where(cache.memory_initialized[:, None, None, None], mixed, local)
        context = local.transpose(0, 2, 1, 3).reshape(batch, 1, hidden)

        chunk_width = min(self.fixed_block_width or self.config.path_window_size, self.config.path_window_size)
        if cache.open_len == chunk_width:
            if update_memory:
                cache.memory_m, cache.memory_z, cache.memory_initialized = self._next_memory(
                    cache.k,
                    cache.v,
                    mx.ones((batch,), dtype=mx.bool_),
                    cache.memory_m,
                    cache.memory_z,
                    cache.memory_initialized,
                )
            cache.clear_open_chunk()
        return self.out(context)

    def extend(
        self,
        x: mx.array,
        cache: "MLXPaTHInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        batch, length, hidden = x.shape
        q, k, v, w, beta, log_forget = self._project_extension(x, cache)
        chunk_width = min(self.fixed_block_width or self.config.path_window_size, self.config.path_window_size)
        outputs = []
        offset = 0
        while offset < length:
            open_length = 0 if cache.q is None else cache.q.shape[2]
            take = min(chunk_width - open_length, length - offset)
            q_new = q[:, :, offset : offset + take]
            k_new = k[:, :, offset : offset + take]
            v_new = v[:, :, offset : offset + take]
            w_new = w[:, offset : offset + take]
            beta_new = beta[:, offset : offset + take]
            forget_new = log_forget[:, offset : offset + take]
            if open_length:
                chunk_q = mx.concatenate((cache.q, q_new), axis=2)
                chunk_k = mx.concatenate((cache.k, k_new), axis=2)
                chunk_v = mx.concatenate((cache.v, v_new), axis=2)
                chunk_w = mx.concatenate((cache.w, w_new), axis=1)
                chunk_beta = mx.concatenate((cache.beta, beta_new), axis=1)
                chunk_forget = mx.concatenate((cache.log_forget, forget_new), axis=1)
            else:
                chunk_q, chunk_k, chunk_v = q_new, k_new, v_new
                chunk_w, chunk_beta, chunk_forget = w_new, beta_new, forget_new
            local = self.path_chunk(
                chunk_q,
                chunk_k,
                chunk_v,
                chunk_w,
                chunk_beta,
                chunk_forget,
                None,
            )[:, :, open_length:]
            memory_context = self._retrieve_memory(q_new, cache.memory_m, cache.memory_z).astype(v.dtype)
            gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
            mixed = (1.0 - gate) * local + gate * memory_context
            outputs.append(mx.where(cache.memory_initialized[:, None, None, None], mixed, local))
            if chunk_q.shape[2] == chunk_width:
                if update_memory:
                    cache.memory_m, cache.memory_z, cache.memory_initialized = self._next_memory(
                        chunk_k,
                        chunk_v,
                        mx.ones((batch,), dtype=mx.bool_),
                        cache.memory_m,
                        cache.memory_z,
                        cache.memory_initialized,
                    )
                cache.clear_open_chunk()
            else:
                cache.q, cache.k, cache.v = chunk_q, chunk_k, chunk_v
                cache.w, cache.beta, cache.log_forget = chunk_w, chunk_beta, chunk_forget
                cache.open_len = chunk_q.shape[2]
                cache.t_inverse = self.path_system_t_inverse(cache.w, cache.beta)
            offset += take
        context = mx.concatenate(outputs, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.out(context)

    def prefill(
        self,
        x: mx.array,
        cache: "MLXPaTHInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        batch, length, hidden = x.shape
        q, k, v, w, beta, log_forget, projected = self._project(x, None)
        chunks = []
        chunk_width = min(self.fixed_block_width or self.config.path_window_size, self.config.path_window_size)
        for start in range(0, length, chunk_width):
            end = min(start + chunk_width, length)
            local = self.path_chunk(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                w[:, start:end],
                beta[:, start:end],
                log_forget[:, start:end],
                None,
            )
            memory_context = self._retrieve_memory(
                q[:, :, start:end], cache.memory_m, cache.memory_z
            ).astype(v.dtype)
            gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
            mixed = (1.0 - gate) * local + gate * memory_context
            local = mx.where(cache.memory_initialized[:, None, None, None], mixed, local)
            if end - start == chunk_width:
                if update_memory:
                    cache.memory_m, cache.memory_z, cache.memory_initialized = self._next_memory(
                        k[:, :, start:end],
                        v[:, :, start:end],
                        mx.ones((batch,), dtype=mx.bool_),
                        cache.memory_m,
                        cache.memory_z,
                        cache.memory_initialized,
                    )
            else:
                cache.q = q[:, :, start:end]
                cache.k = k[:, :, start:end]
                cache.v = v[:, :, start:end]
                cache.w = w[:, start:end]
                cache.beta = beta[:, start:end]
                cache.log_forget = log_forget[:, start:end]
                cache.open_len = end - start
                cache.t_inverse = self.path_system_t_inverse(cache.w, cache.beta)
            chunks.append(local)
        plen = projected.shape[1]
        cache.path_projected = projected if plen <= 2 else projected[:, plen - 2 : plen]
        context = mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.out(context)

    def _path_solve(self, system: mx.array, diagonal: mx.array, *, compile_friendly: bool = False) -> mx.array:
        """Solve ``S T = diag(beta)`` by forward substitution.

        Do not "optimize" this into a Neumann series (T = prod_i (I + (-L)^(2^i))).
        It is algebraically exact for nilpotent L and benchmarks ~2.7x faster, but
        it is not backward stable: once the learned path vectors align, entries of
        L approach beta ~ 2 and the intermediate powers M^16 / M^32 overflow into
        the cancellation that produces the small true inverse. Training NaNs on
        step 1 (grad norm 3456 vs 70). Random-unit-vector tests do not catch it.
        """
        if self.config.use_path_kernel:
            if compile_friendly:
                from mlx_path_kernel import _run_kernel, _LOWER_SOLVE
                return _run_kernel(_LOWER_SOLVE, system, diagonal)
            return path_triangular_solve(system, diagonal)
        return reference_triangular_solve(system, diagonal)

    def path_chunk(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        w: mx.array,
        beta: mx.array,
        log_forget: mx.array,
        segment_ids: mx.array | None,
    ) -> mx.array:
        length = q.shape[2]
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        wf = w.transpose(0, 2, 1, 3)
        beta = beta.transpose(0, 2, 1)
        gram = wf @ wf.transpose(0, 1, 3, 2)
        eye = mx.eye(length, dtype=mx.float32)
        system = eye + mx.tril(beta[..., None] * gram, k=-1)
        t_inverse = self._path_solve(system, eye * beta[..., :, None], compile_friendly=False)
        qk = qf @ kf.transpose(0, 1, 3, 2)
        qw = mx.tril(qf @ wf.transpose(0, 1, 3, 2))
        wk = mx.tril(wf @ kf.transpose(0, 1, 3, 2), k=-1)
        logits = (qk - (qw @ t_inverse) @ wk) * self.head_dim**-0.5
        prefix = mx.cumsum(log_forget.transpose(0, 2, 1), axis=-1)
        logits = logits + prefix[..., :, None] - prefix[..., None, :]
        keep = mx.tril(mx.ones((length, length), dtype=mx.bool_))
        if segment_ids is not None:
            keep = keep & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
        logits = mx.where(keep, logits, -1e9)
        return mx.softmax(logits, axis=-1).astype(v.dtype) @ v

    def path_system_t_inverse(self, w: mx.array, beta: mx.array) -> mx.array:
        """Full open-chunk T = S^{-1} D (used to seed running state after prefill/extend)."""
        length = w.shape[1]
        wf = w.transpose(0, 2, 1, 3).astype(mx.float32)
        beta_h = beta.transpose(0, 2, 1).astype(mx.float32)
        gram = wf @ wf.transpose(0, 1, 3, 2)
        eye = mx.eye(length, dtype=mx.float32)
        system = eye + mx.tril(beta_h[..., None] * gram, k=-1)
        return self._path_solve(system, eye * beta_h[..., :, None], compile_friendly=True)

    def path_border_update_t(
        self,
        t_prev: mx.array | None,
        w: mx.array,
        beta: mx.array,
    ) -> mx.array:
        """Grow T by one token via lower-triangular border update (O(L^2), not O(L^3)).

        For S = [[S_prev, 0], [s^T, 1]] and D = diag(beta):
        T = [[T_prev, 0], [-s^T T_prev, beta_new]] with s_j = beta_new (w_new · w_j).
        """
        length = w.shape[1]
        batch = w.shape[0]
        heads = self.config.num_attention_heads
        beta_h = beta.transpose(0, 2, 1).astype(mx.float32)  # B,H,L
        beta_new = beta_h[:, :, length - 1]  # B,H
        if length == 1 or t_prev is None:
            return beta_new[:, :, None, None]
        wf = w.transpose(0, 2, 1, 3).astype(mx.float32)  # B,H,L,D
        w_new = wf[:, :, length - 1 : length, :]  # B,H,1,D
        w_prev = wf[:, :, : length - 1, :]  # B,H,L-1,D
        # s_j = beta_new * (w_new · w_j)
        dots = mx.sum(w_new * w_prev, axis=-1)  # B,H,L-1
        s = beta_new[:, :, None] * dots  # B,H,L-1
        # t_row = -s @ T_prev  -> (B,H,L-1)
        t_row = -mx.sum(s[:, :, :, None] * t_prev, axis=2)
        zeros_col = mx.zeros((batch, heads, length - 1, 1), dtype=mx.float32)
        top = mx.concatenate((t_prev, zeros_col), axis=-1)
        bottom = mx.concatenate((t_row[:, :, None, :], beta_new[:, :, None, None]), axis=-1)
        return mx.concatenate((top, bottom), axis=2)

    def path_chunk_last_with_t(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        w: mx.array,
        beta: mx.array,
        log_forget: mx.array,
        t_inverse: mx.array,
        segment_ids: mx.array | None = None,
    ) -> mx.array:
        """Last-query PaTH using a provided running T (no system rebuild or re-solve)."""
        length = q.shape[2]
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        wf = w.transpose(0, 2, 1, 3).astype(mx.float32)
        q_last = qf[:, :, length - 1 : length]
        qk = q_last @ kf.transpose(0, 1, 3, 2)
        qw_last = q_last @ wf.transpose(0, 1, 3, 2)
        wk = mx.tril(wf @ kf.transpose(0, 1, 3, 2), k=-1)
        corrected = (qw_last @ t_inverse) @ wk
        logits = (qk - corrected) * self.head_dim**-0.5
        prefix = mx.cumsum(log_forget.transpose(0, 2, 1).astype(mx.float32), axis=-1)
        logits = logits + prefix[:, :, length - 1 : length, None] - prefix[:, :, None, :]
        if segment_ids is not None:
            keep = segment_ids[:, None, length - 1 : length, None] == segment_ids[:, None, None, :]
            logits = mx.where(keep, logits, -1e9)
        return mx.softmax(logits, axis=-1).astype(v.dtype) @ v

    def path_chunk_last(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        w: mx.array,
        beta: mx.array,
        log_forget: mx.array,
        segment_ids: mx.array | None,
        t_inverse: mx.array | None = None,
    ) -> mx.array:
        """Last-query PaTH; uses running T when provided, else seeds T then attends."""
        if t_inverse is None:
            t_inverse = self.path_system_t_inverse(w, beta)
        return self.path_chunk_last_with_t(q, k, v, w, beta, log_forget, t_inverse, segment_ids)

    def _chunk_bounds(self, length: int) -> list[tuple[int, int]]:
        """Half-open [start, end) spans of each local-attention chunk."""
        block_width = self.fixed_block_width or (length + self.num_blocks - 1) // self.num_blocks
        bounds = []
        for block_start in range(0, length, block_width):
            block_end = min(block_start + block_width, length)
            for start in range(block_start, block_end, self.config.path_window_size):
                bounds.append((start, min(start + self.config.path_window_size, block_end)))
        return bounds

    def _batched_path_chunks(self, q, k, v, w, beta, log_forget, segment_ids, count):
        """Run all equal-width chunks through ``path_chunk`` as one batched call.

        A single 64x64 chunk nowhere near saturates the GPU — the triangular solve
        gets one thread per column — so the loop was launch-bound rather than
        FLOP-bound. Stacking chunks onto the batch axis is the same arithmetic with
        ``count`` times the parallelism. Returns (batch, count, heads, width, dim).
        """
        batch, heads, length, dim = q.shape
        width = length // count
        merged = batch * count

        def fold(t):  # (B, H, count*width, D) -> (B*count, H, width, D)
            return (
                t.reshape(batch, heads, count, width, dim)
                .transpose(0, 2, 1, 3, 4)
                .reshape(merged, heads, width, dim)
            )

        stacked = self.path_chunk(
            fold(q),
            fold(k),
            fold(v),
            w.reshape(merged, width, heads, dim),
            beta.reshape(merged, width, heads),
            log_forget.reshape(merged, width, heads),
            None if segment_ids is None else segment_ids.reshape(merged, width),
        )
        return stacked.reshape(batch, count, heads, width, dim)

    def forward_arrays(
        self,
        x: mx.array,
        segment_ids: mx.array | None,
        memory_m: mx.array,
        memory_z: mx.array,
        memory_initialized: mx.array,
        update_memory: bool,
    ):
        batch, length, hidden = x.shape
        q, k, v, w, beta, log_forget, _ = self._project(x, segment_ids)
        chunks = []
        bounds = self._chunk_bounds(length)
        memory_safe = (
            mx.ones((batch,), dtype=mx.bool_)
            if segment_ids is None
            else mx.all(segment_ids == segment_ids[:, :1], axis=1)
        )
        # Local attention is chunk-independent, so run every chunk in one batched
        # call; only the Infini carry below has to stay sequential.
        widths = {end - start for start, end in bounds}
        stacked = (
            self._batched_path_chunks(q, k, v, w, beta, log_forget, segment_ids, len(bounds))
            if len(widths) == 1 and len(bounds) > 1
            else None
        )
        for index, (start, end) in enumerate(bounds):
            if stacked is None:
                chunk_segments = segment_ids[:, start:end] if segment_ids is not None else None
                local = self.path_chunk(
                    q[:, :, start:end],
                    k[:, :, start:end],
                    v[:, :, start:end],
                    w[:, start:end],
                    beta[:, start:end],
                    log_forget[:, start:end],
                    chunk_segments,
                )
            else:
                local = stacked[:, index]
            memory_context = self._retrieve_memory(
                q[:, :, start:end], memory_m, memory_z
            ).astype(v.dtype)
            gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
            mixed = (1.0 - gate) * local + gate * memory_context
            use_memory = memory_initialized & memory_safe
            local = mx.where(use_memory[:, None, None, None], mixed, local)
            if self.config.use_topk_blocks:
                retrieved = self._topk_context(q[:, :, start:end], k, v, start)
                if retrieved is not None:
                    topk_gate = mx.sigmoid(self.topk_gate)[None, :, None, None]
                    blended = (1.0 - topk_gate) * local + topk_gate * retrieved
                    # Cross-block retrieval would leak across packed documents.
                    local = mx.where(memory_safe[:, None, None, None], blended, local)
            if update_memory:
                memory_m, memory_z, memory_initialized = self._next_memory(
                    k[:, :, start:end],
                    v[:, :, start:end],
                    memory_safe,
                    memory_m,
                    memory_z,
                    memory_initialized,
                )
            chunks.append(local)
        context = mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.out(context), memory_m, memory_z, memory_initialized

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
        checkpoint_activations: bool = False,
        *,
        persist_memory: bool = False,
    ) -> mx.array:
        """PaTH+Infini forward.

        Within-sequence memory carries through local tensors in ``forward_arrays``.
        Persisting updated banks onto ``self.memory_*`` is opt-in
        (``persist_memory=True``) for eager inspection; default is off because
        writing multi-block carries into ``model.state`` under ``mx.compile``
        fails for ``block_size >= 7``. Training resets memory each forward;
        decode uses ``MLXPaTHInferenceCache``.
        """
        if self.memory_m.shape[0] != x.shape[0]:
            self.reset_memory(x.shape[0])
        # Detach only where memory enters from module state. These banks are frozen
        # buffers inside mx.compile's `inputs=model.state`; leaving them in the graph
        # makes autodiff look for a gradient slot they do not have ("key not found").
        # The chunk-to-chunk carry built in forward_arrays stays differentiable, which
        # is what lets gradient reach the K/V that wrote the memory.
        memory = (
            mx.stop_gradient(self.memory_m),
            mx.stop_gradient(self.memory_z),
            self.memory_initialized,
        )
        if checkpoint_activations:
            run = activation_checkpoint(self, self.forward_arrays)
        else:
            run = self.forward_arrays
        output, memory_m, memory_z, memory_initialized = run(
            x,
            segment_ids,
            *memory,
            update_memory,
        )
        # Default: do not write new memory tensors into Module state. Callers that
        # need inspectable buffers (unit tests, eager single-layer use) pass
        # persist_memory=True. Inference decode uses MLXPaTHInferenceCache.
        if update_memory and persist_memory:
            self.memory_m = mx.stop_gradient(memory_m)
            self.memory_z = mx.stop_gradient(memory_z)
            self.memory_initialized = memory_initialized
        return output


@dataclass
class MLXPaTHInferenceCache:
    memory_m: mx.array
    memory_z: mx.array
    memory_initialized: mx.array
    path_projected: mx.array | None = None
    q: mx.array | None = None
    k: mx.array | None = None
    v: mx.array | None = None
    w: mx.array | None = None
    beta: mx.array | None = None
    log_forget: mx.array | None = None
    # Running S^{-1} D for the open chunk; border-updated O(L^2) per decode token.
    t_inverse: mx.array | None = None
    open_len: int = 0

    def clear_open_chunk(self) -> None:
        self.q = self.k = self.v = self.w = self.beta = self.log_forget = None
        self.t_inverse = None
        self.open_len = 0

    def clone(self) -> "MLXPaTHInferenceCache":
        return MLXPaTHInferenceCache(
            self.memory_m,
            self.memory_z,
            self.memory_initialized,
            self.path_projected,
            self.q,
            self.k,
            self.v,
            self.w,
            self.beta,
            self.log_forget,
            self.t_inverse,
            self.open_len,
        )

    def arrays(self) -> list[mx.array]:
        return [
            value
            for value in (
                self.memory_m,
                self.memory_z,
                self.memory_initialized,
                self.path_projected,
                self.q,
                self.k,
                self.v,
                self.w,
                self.beta,
                self.log_forget,
                self.t_inverse,
            )
            if value is not None
        ]


class MLXHybridBlock(nn.Module):
    def __init__(self, config: MLXBitNetConfig, layer_id: int):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        self.config = config
        self.layer_id = layer_id
        self.attn_res_mode = config.attn_res_mode
        self.engram = MLXEngram(config, layer_id) if config.use_engram and layer_id in config.engram_layer_ids else None
        self.attn_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.attn = MLXPaTHAttention(config)
        self.attn_gate = mx.array([0.0])
        self.mlp_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.moe = MLXRFMoE(config) if config.use_rfmoe else None
        if self.moe is None:
            self.up = MLXHBitLinear(hidden, intermediate * 2, config)
            self.mid = MLXHBitLinear(intermediate, intermediate, config)
            # Cold start: identity mid ~ classic 2-mat path (silu pass-through on
            # expand), scaled so ternarisation lands on a *true* identity. The
            # per-output-channel scale is mean(|row|); a plain eye(I) row is one 1
            # and I-1 zeros, so the scale is 1/I and the quantised weight comes out
            # as eye(I)/I -- a 1/512 attenuator, not a pass-through. eye(I)*I gives
            # mean(|row|) = 1, so the quantised weight is exactly eye(I). The
            # weight mix is pinned with it: the straight-through blend only means
            # anything when raw and quantised share a scale, and here they differ
            # by I.
            self.mid.weight = mx.eye(intermediate, dtype=self.mid.weight.dtype) * intermediate
            self.mid.pinned_weight_mix = 1.0
            self.mid.weight_mix = mx.array(1.0)
            self.mid._pinned_full_weight_quant = True
            self.down = MLXHBitLinear(intermediate, hidden, config)
            self.down.weight = self.down.weight * 0.01
        if self.attn_res_mode == "kimi":
            self.attn_res_mix = MLXDepthAttnMix(hidden, eps=config.rms_norm_eps)
            self.mlp_res_mix = MLXDepthAttnMix(hidden, eps=config.rms_norm_eps)
            # Keep sandwich attrs for convert soft-load / sandwich mode compatibility.
            self.attn_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
            self.attn_scale = mx.array([0.1])
            self.mlp_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
            self.mlp_scale = mx.array([0.1])
        else:
            self.attn_res_mix = None
            self.mlp_res_mix = None
            self.attn_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
            self.attn_scale = mx.array([0.1])
            self.mlp_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
            self.mlp_scale = mx.array([0.1])

    def _dense_mlp(self, x: mx.array) -> mx.array:
        # Decode M=1: one Metal dispatch for up + mid + down ternary FFN.
        hidden = int(x.shape[-1])
        tokens = int(x.size // max(hidden, 1))
        inter = int(self.mid.weight.shape[0])
        # Fused FFN shared-memory kernel is for small/medium widths; 1B (h=1024, I=2048) uses qmm.
        if (
            tokens == 1
            and hidden <= 512
            and inter <= 1024
            and self.up._full_activation_quant
            and self.mid._full_activation_quant
            and self.down._full_activation_quant
        ):
            up_p = self.up._packed_weight(x)
            mid_p = self.mid._packed_weight(x)
            down_p = self.down._packed_weight(x)
            if up_p is not None and mid_p is not None and down_p is not None:
                try:
                    return ternary_fused_ffn_m1(
                        x,
                        up_p[0],
                        up_p[1],
                        mid_p[0],
                        mid_p[1],
                        down_p[0],
                        down_p[1],
                        hidden=hidden,
                        intermediate=inter,
                        quantize_acts=bool(self.up.config.use_4bit_activations),
                        act_levels=self.up._act_levels_f,
                        dtype=x.dtype,
                    )
                except ValueError:
                    pass
        gate, value = mx.split(self.up(x), 2, axis=-1)
        hidden_act = nn.silu(self.mid(nn.silu(gate) * value))
        return self.down(hidden_act)

    def _mlp(self, x: mx.array, checkpoint_activations: bool = False) -> mx.array:
        if self.moe is not None:
            if checkpoint_activations and self.moe.backend != "hybrid":
                output, gate_stack, hard_density = activation_checkpoint(
                    self.moe,
                    self.moe.forward_arrays,
                )(x)
                self.moe.update_stats(gate_stack, hard_density)
                return output
            return self.moe(x)
        run_mlp = activation_checkpoint(self, self._dense_mlp) if checkpoint_activations else self._dense_mlp
        return run_mlp(x)

    def new_inference_cache(self, batch_size: int) -> "MLXBlockInferenceCache":
        return MLXBlockInferenceCache(
            attention=self.attn.new_inference_cache(batch_size),
            engram=MLXEngramInferenceCache() if self.engram is not None else None,
        )

    def _mixer(
        self,
        x_norm: mx.array,
        segment_ids: mx.array | None,
        update_memory: bool,
        checkpoint_activations: bool,
    ) -> mx.array:
        return self.attn(
            x_norm,
            segment_ids,
            update_memory,
            checkpoint_activations,
        )

    def forward_sandwich(
        self,
        x: mx.array,
        input_ids: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
        checkpoint_activations: bool = False,
    ) -> mx.array:
        if self.engram is not None:
            run_engram = activation_checkpoint(self.engram) if checkpoint_activations else self.engram
            x = x + run_engram(x, input_ids, segment_ids)
        attention = self._mixer(
            self.attn_norm(x),
            segment_ids,
            update_memory,
            checkpoint_activations,
        )
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        output = self._mlp(self.mlp_norm(x), checkpoint_activations)
        return self.mlp_post(x + self.mlp_scale * output)

    def forward_kimi(
        self,
        stream: MLXAttnResStream,
        input_ids: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
        checkpoint_activations: bool = False,
        *,
        attn_runner=None,
        engram_runner=None,
    ) -> MLXAttnResStream:
        """Kimi Block AttnRes step. ``attn_runner`` overrides train attn for decode paths."""
        stream.attn_mix = self.attn_res_mix
        stream.mlp_mix = self.mlp_res_mix

        h = stream.mix_attn()
        if self.engram is not None:
            if engram_runner is not None:
                h = h + engram_runner(h)
            else:
                run_engram = activation_checkpoint(self.engram) if checkpoint_activations else self.engram
                h = h + run_engram(h, input_ids, segment_ids)

        if attn_runner is not None:
            attention = attn_runner(self.attn_norm(h))
        else:
            attention = self._mixer(
                self.attn_norm(h),
                segment_ids,
                update_memory,
                checkpoint_activations,
            )
        stream.add_sublayer(mx.sigmoid(self.attn_gate) * attention)

        h = stream.mix_mlp()
        stream.add_sublayer(self._mlp(self.mlp_norm(h), checkpoint_activations))
        stream.close_layer()
        return stream

    def step_kimi_stream(
        self,
        stream: MLXAttnResStream,
        input_ids: mx.array,
        token_history: mx.array | None,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
        mode: str,
    ) -> MLXAttnResStream:
        """Apply one block through AttnRes stream for prefill/extend/incremental."""

        def engram_runner(h: mx.array) -> mx.array:
            if self.engram is None:
                return mx.zeros_like(h)
            if mode == "incremental":
                return self.engram.incremental(h, input_ids, token_history, cache.engram)
            if mode == "extend":
                return self.engram.extend(h, input_ids, token_history, cache.engram)
            return self.engram.prefill(h, input_ids, cache.engram)

        def attn_runner(h_norm: mx.array) -> mx.array:
            if mode == "incremental":
                return self.attn.incremental(h_norm, cache.attention, update_memory)
            if mode == "extend":
                return self.attn.extend(h_norm, cache.attention, update_memory)
            return self.attn.prefill(h_norm, cache.attention, update_memory)

        return self.forward_kimi(
            stream,
            input_ids,
            None,
            update_memory,
            False,
            attn_runner=attn_runner,
            engram_runner=engram_runner if self.engram is not None else None,
        )

    def incremental(
        self,
        x: mx.array,
        input_ids: mx.array,
        token_history: mx.array,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        if self.attn_res_mode == "kimi":
            raise RuntimeError("use MLXBitNet stream decode for kimi AttnRes")
        if self.engram is not None:
            x = x + self.engram.incremental(x, input_ids, token_history, cache.engram)
        attention = self.attn.incremental(self.attn_norm(x), cache.attention, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        return self.mlp_post(x + self.mlp_scale * self._mlp(self.mlp_norm(x)))

    def extend(
        self,
        x: mx.array,
        input_ids: mx.array,
        token_history: mx.array,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        if self.attn_res_mode == "kimi":
            raise RuntimeError("use MLXBitNet stream decode for kimi AttnRes")
        if self.engram is not None:
            x = x + self.engram.extend(x, input_ids, token_history, cache.engram)
        attention = self.attn.extend(self.attn_norm(x), cache.attention, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        return self.mlp_post(x + self.mlp_scale * self._mlp(self.mlp_norm(x)))

    def prefill(
        self,
        x: mx.array,
        input_ids: mx.array,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        if self.attn_res_mode == "kimi":
            raise RuntimeError("use MLXBitNet stream decode for kimi AttnRes")
        if self.engram is not None:
            x = x + self.engram.prefill(x, input_ids, cache.engram)
        attention = self.attn.prefill(self.attn_norm(x), cache.attention, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        return self.mlp_post(x + self.mlp_scale * self._mlp(self.mlp_norm(x)))

    def __call__(
        self,
        x: mx.array | MLXAttnResStream,
        input_ids: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
        checkpoint_activations: bool = False,
    ) -> mx.array | MLXAttnResStream:
        if self.attn_res_mode == "kimi":
            if not isinstance(x, MLXAttnResStream):
                raise TypeError("kimi AttnRes mode requires MLXAttnResStream input")
            return self.forward_kimi(
                x,
                input_ids,
                segment_ids,
                update_memory,
                checkpoint_activations,
            )
        if isinstance(x, MLXAttnResStream):
            raise TypeError("sandwich mode expects a hidden array, not MLXAttnResStream")
        return self.forward_sandwich(
            x,
            input_ids,
            segment_ids,
            update_memory,
            checkpoint_activations,
        )


@dataclass
class MLXBlockInferenceCache:
    attention: MLXPaTHInferenceCache
    engram: MLXEngramInferenceCache | None

    def clone(self) -> "MLXBlockInferenceCache":
        return MLXBlockInferenceCache(
            self.attention.clone(),
            self.engram.clone() if self.engram is not None else None,
        )

    def arrays(self) -> list[mx.array]:
        values = self.attention.arrays()
        if self.engram is not None and self.engram.normalized is not None:
            values.append(self.engram.normalized)
        return values


@dataclass
class MLXInferenceCache:
    layers: list[MLXBlockInferenceCache]
    token_history: mx.array
    num_loops: int
    weight_cache: dict[tuple[object, str], object]
    position: int = 0

    def clone(self) -> "MLXInferenceCache":
        return MLXInferenceCache(
            layers=[layer.clone() for layer in self.layers],
            token_history=self.token_history,
            num_loops=self.num_loops,
            weight_cache=self.weight_cache,
            position=self.position,
        )

    def arrays(self) -> list[mx.array]:
        return [self.token_history, *(value for layer in self.layers for value in layer.arrays())]


class MLXBitNet(nn.Module):
    """MLX BitNet with PaTH/Infini, Engram, RFMoE, Hyperloop, and MTP."""

    def __init__(
        self,
        config: MLXBitNetConfig,
        *,
        reuse_recurrent_weights: bool = True,
        recurrent_quantized_matmul: bool = False,
    ):
        super().__init__()
        self.config = config
        self.reuse_recurrent_weights = reuse_recurrent_weights
        self.recurrent_quantized_matmul = recurrent_quantized_matmul
        self.path_decode_mode = "last"
        self._compiled_inference_step = None
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Untied: a separate output projection, initialised small so the head
        # starts near-uniform instead of inheriting the embedding's scale.
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head.weight = self.lm_head.weight * 0.01
        self.subln = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.blocks = [MLXHybridBlock(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        self.loop_hc = MLXLoopHyperConnection(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_transforms = [
            nn.Sequential(
                nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            )
            for _ in range(config.mtp_depth)
        ]

    def reset_memory(self, batch_size: int) -> None:
        for block in self.blocks:
            if block.attn is not None:
                block.attn.reset_memory(batch_size)

    @property
    def uses_engram(self) -> bool:
        """Whether any block hashes token n-grams, the one thing needing ids."""
        return any(block.engram is not None for block in self.blocks)

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        def update(_, module):
            if isinstance(module, MLXHBitLinear):
                module.set_quantization_state(weight_mix, activation_mix, bits)

        self.apply_to_modules(update)

    def set_active_blocks(self, blocks: int) -> None:
        for block in self.blocks:
            if block.attn is not None:
                block.attn.num_blocks = max(1, blocks)

    def set_inference_block_width(self, width: int | None) -> None:
        for block in self.blocks:
            if block.attn is not None:
                block.attn.fixed_block_width = width

    def pin_inference_weights(self, dtype=None, *, prefer_packed: bool = True) -> None:
        """Pin one effective/packed weight per HBitLinear for the generation lifetime."""
        if dtype is None:
            sample = self.embedding.weight
            dtype = sample.dtype
        token = _recurrent_quantized_matmul.set(self.recurrent_quantized_matmul)
        try:
            def pin(_, module):
                if isinstance(module, MLXHBitLinear):
                    module.pin_inference_weight(dtype, prefer_packed=prefer_packed)

            self.apply_to_modules(pin)
            pinned = [
                module._pinned_dense
                for _, module in self.named_modules()
                if isinstance(module, MLXHBitLinear) and module._pinned_dense is not None
            ]
            packed = [
                module._pinned_packed[0]
                for _, module in self.named_modules()
                if isinstance(module, MLXHBitLinear) and module._pinned_packed is not None
            ]
            if pinned or packed:
                mx.eval(*(pinned + packed))
        finally:
            _recurrent_quantized_matmul.reset(token)

    def set_path_decode_mode(self, mode: str) -> None:
        if mode not in {"last", "recompute"}:
            raise ValueError("path decode mode must be 'last' or 'recompute'")
        self.path_decode_mode = mode

    def new_inference_cache(self, batch_size: int = 1, num_loops: int | None = None) -> MLXInferenceCache:
        loops = self.config.num_loops if num_loops is None else num_loops
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        execution = [
            *self.blocks[:prelude_end],
            *(block for _ in range(loops) for block in self.blocks[prelude_end:recurrent_end]),
            *self.blocks[recurrent_end:],
        ]
        return MLXInferenceCache(
            layers=[block.new_inference_cache(batch_size) for block in execution],
            token_history=mx.zeros((batch_size, 0), dtype=mx.int32),
            num_loops=loops,
            weight_cache={},
        )

    @contextmanager
    def _inference_weight_context(self, cache: MLXInferenceCache):
        weight_token = _effective_weight_cache.set(cache.weight_cache if self.reuse_recurrent_weights else None)
        quantized_token = _recurrent_quantized_matmul.set(self.recurrent_quantized_matmul)
        mode = getattr(self, "path_decode_mode", "last")
        path_token = _path_decode_mode.set(mode)
        try:
            yield
        finally:
            _effective_weight_cache.reset(weight_token)
            _recurrent_quantized_matmul.reset(quantized_token)
            _path_decode_mode.reset(path_token)

    def inference_step(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        with self._inference_weight_context(cache):
            step = getattr(self, "_compiled_inference_step", None)
            if step is not None:
                return step(tokens, cache)
            return self._inference_step(tokens, cache)

    def _cache_layout(self, cache: MLXInferenceCache) -> list[bool]:
        """Whether each expanded layer has an Engram cache."""
        return [layer.engram is not None for layer in cache.layers]

    def _flatten_inference_cache(self, cache: MLXInferenceCache) -> list[mx.array]:
        batch = cache.token_history.shape[0]
        hidden = self.config.hidden_size
        heads = self.config.num_attention_heads
        head_dim = hidden // heads
        dtype = self.embedding.weight.dtype
        arrays: list[mx.array] = [cache.token_history]
        for layer in cache.layers:
            att = layer.attention
            arrays.extend([att.memory_m, att.memory_z, att.memory_initialized])
            arrays.append(
                att.path_projected
                if att.path_projected is not None
                else mx.zeros((batch, 0, hidden), dtype=dtype)
            )
            arrays.append(att.q if att.q is not None else mx.zeros((batch, heads, 0, head_dim), dtype=dtype))
            arrays.append(att.k if att.k is not None else mx.zeros((batch, heads, 0, head_dim), dtype=dtype))
            arrays.append(att.v if att.v is not None else mx.zeros((batch, heads, 0, head_dim), dtype=dtype))
            arrays.append(att.w if att.w is not None else mx.zeros((batch, 0, heads, head_dim), dtype=dtype))
            arrays.append(att.beta if att.beta is not None else mx.zeros((batch, 0, heads), dtype=mx.float32))
            arrays.append(
                att.log_forget if att.log_forget is not None else mx.zeros((batch, 0, heads), dtype=mx.float32)
            )
            open_len = 0 if att.q is None else att.q.shape[2]
            if att.t_inverse is not None:
                arrays.append(att.t_inverse)
            else:
                arrays.append(mx.zeros((batch, heads, open_len, open_len), dtype=mx.float32))
            if layer.engram is not None:
                eng = layer.engram.normalized
                if eng is None:
                    eng_dim = self.config.engram_num_heads * self.config.engram_head_dim
                    eng = mx.zeros((batch, 0, eng_dim), dtype=dtype)
                arrays.append(eng)
        return arrays

    def _unflatten_inference_cache(
        self,
        arrays: list[mx.array],
        layout: list[bool],
        num_loops: int,
        weight_cache: dict,
        position: int,
    ) -> MLXInferenceCache:
        idx = 0
        token_history = arrays[idx]
        idx += 1
        layers: list[MLXBlockInferenceCache] = []

        def nonempty(array: mx.array, axis: int):
            return None if array.shape[axis] == 0 else array

        for has_engram in layout:
            memory_m, memory_z, memory_initialized = arrays[idx], arrays[idx + 1], arrays[idx + 2]
            idx += 3
            path_projected = arrays[idx]
            idx += 1
            q, k, v, w, beta, log_forget = arrays[idx : idx + 6]
            idx += 6
            q_arr = nonempty(q, 2)
            t_inv = arrays[idx]
            idx += 1
            open_len = 0 if q_arr is None else int(q_arr.shape[2])
            attention = MLXPaTHInferenceCache(
                memory_m,
                memory_z,
                memory_initialized,
                nonempty(path_projected, 1),
                q_arr,
                nonempty(k, 2),
                nonempty(v, 2),
                nonempty(w, 1),
                nonempty(beta, 1),
                nonempty(log_forget, 1),
                t_inverse=None if open_len == 0 else t_inv,
                open_len=open_len,
            )
            engram = None
            if has_engram:
                eng = arrays[idx]
                idx += 1
                engram = MLXEngramInferenceCache(nonempty(eng, 1))
            layers.append(MLXBlockInferenceCache(attention, engram))
        return MLXInferenceCache(layers, token_history, num_loops, weight_cache, position)

    def _apply_flat_to_inference_cache(self, cache: MLXInferenceCache, arrays: list[mx.array]) -> None:
        """Write functionalized step outputs back into an existing cache (no realloc of layer list)."""
        idx = 0
        cache.token_history = arrays[idx]
        idx += 1
        for layer in cache.layers:
            att = layer.attention
            att.memory_m = arrays[idx]
            att.memory_z = arrays[idx + 1]
            att.memory_initialized = arrays[idx + 2]
            idx += 3
            pp = arrays[idx]
            idx += 1
            q, k, v, w, beta, log_forget = arrays[idx : idx + 6]
            idx += 6
            t_inv = arrays[idx]
            idx += 1
            att.path_projected = None if pp.shape[1] == 0 else pp
            att.q = None if q.shape[2] == 0 else q
            att.k = None if k.shape[2] == 0 else k
            att.v = None if v.shape[2] == 0 else v
            att.w = None if w.shape[1] == 0 else w
            att.beta = None if beta.shape[1] == 0 else beta
            att.log_forget = None if log_forget.shape[1] == 0 else log_forget
            att.open_len = 0 if att.q is None else att.q.shape[2]
            att.t_inverse = None if att.open_len == 0 else t_inv
            if layer.engram is not None:
                eng = arrays[idx]
                idx += 1
                layer.engram.normalized = None if eng.shape[1] == 0 else eng

    def enable_compiled_inference(self) -> bool:

        """Enable functionalized compiled steps specialized lazily by open-chunk length."""
        self._compiled_inference_step = None
        self._compiled_by_open_len = None
        try:
            # Dense pins avoid custom quantized transforms that break mx.compile.
            for _, module in self.named_modules():
                if isinstance(module, MLXHBitLinear):
                    module.pin_inference_weight(self.embedding.weight.dtype, prefer_packed=False)
            loops = getattr(self, "inference_num_loops", self.config.num_loops)
            path_attn = next((b.attn for b in self.blocks), None)
            default_w = self.config.path_window_size
            width = max(
                1,
                int(
                    (path_attn.fixed_block_width if path_attn is not None else None) or default_w
                ),
            )
            width = min(width, self.config.path_window_size)
            probe = self.new_inference_cache(num_loops=loops)
            layout = self._cache_layout(probe)
            weight_cache = probe.weight_cache
            compiled: dict[int, object] = {}

            def compile_open_before(open_before: int):
                if open_before in compiled:
                    return compiled[open_before]

                def pure_step(step_tokens: mx.array, *flat: mx.array):
                    cache = self._unflatten_inference_cache(
                        list(flat), layout, loops, weight_cache, position=1
                    )
                    for layer in cache.layers:
                        layer.attention.open_len = (
                            0 if layer.attention.q is None else layer.attention.q.shape[2]
                        )
                    with self._inference_weight_context(cache):
                        hidden = self._inference_step(step_tokens, cache)
                    return (hidden, *self._flatten_inference_cache(cache))

                warm = self.new_inference_cache(num_loops=loops)
                if open_before:
                    pref = mx.array([list(range(1, open_before + 1))], dtype=mx.int32)
                    with self._inference_weight_context(warm):
                        states = self._prefill(pref, warm)
                        mx.eval(states, *warm.arrays())
                for layer in warm.layers:
                    layer.attention.open_len = (
                        0 if layer.attention.q is None else layer.attention.q.shape[2]
                    )
                compiled_fn = mx.compile(pure_step)
                flat = self._flatten_inference_cache(warm)
                out = compiled_fn(mx.array([[10_000 + open_before]], dtype=mx.int32), *flat)
                mx.eval(out[0])
                compiled[open_before] = compiled_fn
                return compiled_fn

            # Eagerly specialize every open length so decode does not pay first-use compile cost.
            for open_before in range(width):
                compile_open_before(open_before)

            def _first_path_open_len(cache: MLXInferenceCache) -> int:
                for layer in cache.layers:
                    return 0 if layer.attention.q is None else int(layer.attention.q.shape[2])
                return 0

            def step(tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
                open_before = _first_path_open_len(cache)
                open_before = int(open_before)
                if open_before < 0 or open_before >= width:
                    return self._inference_step(tokens, cache)
                fn = compile_open_before(open_before)
                flat_in = self._flatten_inference_cache(cache)
                result = fn(tokens, *flat_in)
                mx.eval(result)
                hidden = result[0]
                self._apply_flat_to_inference_cache(cache, list(result[1:]))
                cache.position += 1
                return hidden

            self._compiled_inference_step = step
            self._compiled_by_open_len = compiled
            return True
        except Exception:
            self._compiled_inference_step = None
            self._compiled_by_open_len = None
            return False


    def _inference_step(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("inference_step requires one token per batch row")
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        cache_index = 0
        n_pre = prelude_end
        x = self._run_decode_stack(
            self.blocks[:prelude_end],
            x,
            tokens,
            cache.token_history,
            cache.layers[cache_index : cache_index + n_pre],
            [True] * n_pre,
            "incremental",
        )
        cache_index += n_pre
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            n_rec = recurrent_end - prelude_end
            for loop_index in range(cache.num_loops):
                x, post, residual = self.loop_hc.project_in(streams)
                # Fresh AttnRes history each loop (matches train).
                x = self._run_decode_stack(
                    self.blocks[prelude_end:recurrent_end],
                    x,
                    tokens,
                    cache.token_history,
                    cache.layers[cache_index : cache_index + n_rec],
                    [loop_index == cache.num_loops - 1] * n_rec,
                    "incremental",
                )
                cache_index += n_rec
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        n_coda = len(self.blocks) - recurrent_end
        x = self._run_decode_stack(
            self.blocks[recurrent_end:],
            x,
            tokens,
            cache.token_history,
            cache.layers[cache_index : cache_index + n_coda],
            [True] * n_coda,
            "incremental",
        )
        history_keep = self.config.engram_max_ngram_size - 1
        combined_history = mx.concatenate((cache.token_history, tokens), axis=1)
        if combined_history.shape[1] > history_keep:
            start = combined_history.shape[1] - history_keep
            cache.token_history = combined_history[:, start : start + history_keep]
        else:
            cache.token_history = combined_history
        cache.position += 1
        return self.norm(x)

    def inference_extend(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        with self._inference_weight_context(cache):
            return self._inference_extend(tokens, cache)

    def _inference_extend(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("inference_extend requires a non-empty rank-2 token array")
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        cache_index = 0
        n_pre = prelude_end
        x = self._run_decode_stack(
            self.blocks[:prelude_end],
            x,
            tokens,
            cache.token_history,
            cache.layers[cache_index : cache_index + n_pre],
            [True] * n_pre,
            "extend",
        )
        cache_index += n_pre
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            n_rec = recurrent_end - prelude_end
            for loop_index in range(cache.num_loops):
                x, post, residual = self.loop_hc.project_in(streams)
                x = self._run_decode_stack(
                    self.blocks[prelude_end:recurrent_end],
                    x,
                    tokens,
                    cache.token_history,
                    cache.layers[cache_index : cache_index + n_rec],
                    [loop_index == cache.num_loops - 1] * n_rec,
                    "extend",
                )
                cache_index += n_rec
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        n_coda = len(self.blocks) - recurrent_end
        x = self._run_decode_stack(
            self.blocks[recurrent_end:],
            x,
            tokens,
            cache.token_history,
            cache.layers[cache_index : cache_index + n_coda],
            [True] * n_coda,
            "extend",
        )
        cache.token_history = mx.concatenate((cache.token_history, tokens), axis=1)[
            :, -(self.config.engram_max_ngram_size - 1) :
        ]
        cache.position += tokens.shape[1]
        return self.norm(x)

    def prefill(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        with self._inference_weight_context(cache):
            return self._prefill(tokens, cache)

    def _prefill(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("prefill requires a non-empty rank-2 token array")
        if cache.position:
            raise ValueError("prefill requires an empty inference cache")
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        cache_index = 0
        n_pre = prelude_end
        empty_hist = mx.zeros((tokens.shape[0], 0), dtype=mx.int32)
        x = self._run_decode_stack(
            self.blocks[:prelude_end],
            x,
            tokens,
            empty_hist,
            cache.layers[cache_index : cache_index + n_pre],
            [True] * n_pre,
            "prefill",
        )
        cache_index += n_pre
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            n_rec = recurrent_end - prelude_end
            for loop_index in range(cache.num_loops):
                x, post, residual = self.loop_hc.project_in(streams)
                x = self._run_decode_stack(
                    self.blocks[prelude_end:recurrent_end],
                    x,
                    tokens,
                    empty_hist,
                    cache.layers[cache_index : cache_index + n_rec],
                    [loop_index == cache.num_loops - 1] * n_rec,
                    "prefill",
                )
                cache_index += n_rec
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        n_coda = len(self.blocks) - recurrent_end
        x = self._run_decode_stack(
            self.blocks[recurrent_end:],
            x,
            tokens,
            empty_hist,
            cache.layers[cache_index : cache_index + n_coda],
            [True] * n_coda,
            "prefill",
        )
        cache.token_history = tokens[:, -(self.config.engram_max_ngram_size - 1) :]
        cache.position = tokens.shape[1]
        return self.norm(x)

    def rfmoe_aux_losses(self, s: float, alpha: float):
        density = mx.array(0.0)
        locality = mx.array(0.0)
        diversity = mx.array(0.0)
        hard_densities = []
        for block in self.blocks:
            if block.moe is not None:
                layer_density, layer_locality, layer_diversity = block.moe.aux_losses(s, alpha)
                density = density + layer_density
                locality = locality + layer_locality
                diversity = diversity + layer_diversity
                hard_densities.append(block.moe.last_density)
        hard = mx.mean(mx.stack(hard_densities)) if hard_densities else mx.array(0.0)
        return density, locality, diversity, hard

    def logits_from(self, hidden: mx.array) -> mx.array:
        """Unembed. The one place that knows whether the head is tied."""
        if self.lm_head is None:
            return hidden @ self.embedding.weight.T
        return self.lm_head(hidden)

    def mtp_logits(self, hidden: mx.array) -> list[mx.array]:
        return [self.logits_from(transform(hidden)) for transform in self.mtp_transforms]

    def selected_mtp_logits(self, hidden: mx.array, selector: mx.array) -> mx.array:
        transformed = mx.stack([transform(hidden) for transform in self.mtp_transforms], axis=2)
        selected = mx.sum(transformed * selector[None, None, :, None], axis=2)
        return self.logits_from(selected)

    def draft_logits(self, hidden: mx.array) -> mx.array:
        last = hidden[:, -1:]
        transformed = mx.concatenate([transform(last) for transform in self.mtp_transforms], axis=1)
        return self.logits_from(transformed)

    @property
    def _kimi_mode(self) -> bool:
        return self.config.attn_res_mode == "kimi"

    def _new_attn_stream(self, seed: mx.array) -> MLXAttnResStream:
        for block in self.blocks:
            if block.attn_res_mix is not None and block.mlp_res_mix is not None:
                return MLXAttnResStream.start(
                    seed,
                    group_size=int(self.config.attn_res_group_size),
                    attn_mix=block.attn_res_mix,
                    mlp_mix=block.mlp_res_mix,
                )
        raise RuntimeError("kimi AttnRes requires hybrid blocks with depth mixes")

    def _run_block_stack(
        self,
        blocks: list[MLXHybridBlock],
        seed: mx.array,
        tokens: mx.array,
        segment_ids: mx.array | None,
        update_memory: bool,
        checkpoint: bool,
    ) -> mx.array:
        """Run a stack of blocks; Kimi mode uses a fresh AttnRes stream from seed."""
        if not blocks:
            return seed
        if not self._kimi_mode:
            x = seed
            for block in blocks:
                x = block(x, tokens, segment_ids, update_memory, checkpoint)
            return x
        stream = self._new_attn_stream(seed)
        for block in blocks:
            stream = block(stream, tokens, segment_ids, update_memory, checkpoint)
        return stream.hidden()

    def _run_decode_stack(
        self,
        blocks: list[MLXHybridBlock],
        seed: mx.array,
        tokens: mx.array,
        token_history: mx.array,
        caches: list,
        update_flags: list[bool],
        mode: str,
    ) -> mx.array:
        if not blocks:
            return seed
        if not self._kimi_mode:
            x = seed
            for block, cache, upd in zip(blocks, caches, update_flags):
                if mode == "incremental":
                    x = block.incremental(x, tokens, token_history, cache, upd)
                elif mode == "extend":
                    x = block.extend(x, tokens, token_history, cache, upd)
                else:
                    x = block.prefill(x, tokens, cache, upd)
            return x
        stream = self._new_attn_stream(seed)
        for block, cache, upd in zip(blocks, caches, update_flags):
            stream = block.step_kimi_stream(stream, tokens, token_history, cache, upd, mode)
        return stream.hidden()

    def hidden_states(
        self,
        tokens: mx.array | None = None,
        segment_ids: mx.array | None = None,
        num_loops: int | None = None,
        reset_memory: bool = True,
        checkpoint_activations: bool | str = False,
        *,
        inputs_embeds: mx.array | None = None,
    ) -> mx.array:
        """Run the block stack, from token ids or from embeddings already computed.

        ``inputs_embeds`` lets this stack serve as BLT's global transformer,
        which operates on pooled patch latents rather than a token sequence.
        Nothing in the stack needs discrete ids except Engram, whose n-gram
        hashing is defined over a vocabulary -- everything else (PaTH, Infini,
        RFMoE) reads hidden states only. So ``tokens`` stays required
        exactly when Engram is active and is otherwise optional.
        """
        if (tokens is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of tokens or inputs_embeds")
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        checkpoint_scope = "all" if checkpoint_activations is True else checkpoint_activations or "none"
        if checkpoint_scope not in ("none", "recurrent", "all"):
            raise ValueError("checkpoint_activations must be none, recurrent, or all")

        if inputs_embeds is not None and self.uses_engram:
            raise ValueError(
                "Engram hashes token n-grams and cannot run on embeddings alone; "
                "disable it (use_engram=False) to drive this stack from inputs_embeds"
            )
        batch_size = tokens.shape[0] if tokens is not None else inputs_embeds.shape[0]
        if reset_memory:
            self.reset_memory(batch_size)
        x = self.subln(self.embedding(tokens) if inputs_embeds is None else inputs_embeds)
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        # Prelude: one AttnRes segment (or sandwich stack).
        x = self._run_block_stack(
            self.blocks[:prelude_end],
            x,
            tokens,
            segment_ids,
            True,
            checkpoint_scope == "all",
        )
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            cache_token = _effective_weight_cache.set({}) if self.reuse_recurrent_weights and loops > 1 else None
            quantized_token = _recurrent_quantized_matmul.set(self.recurrent_quantized_matmul)
            try:
                for loop_index in range(loops):
                    # Reset AttnRes history each Hyperloop iteration (HC owns cross-loop mix).
                    x, post, residual = self.loop_hc.project_in(streams)
                    x = self._run_block_stack(
                        self.blocks[prelude_end:recurrent_end],
                        x,
                        tokens,
                        segment_ids,
                        loop_index == loops - 1,
                        checkpoint_scope in ("recurrent", "all"),
                    )
                    embedding_index = min(loop_index, 63)
                    output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                    streams = self.loop_hc.write_back(streams, output, post, residual)
            finally:
                if cache_token is not None:
                    _effective_weight_cache.reset(cache_token)
                _recurrent_quantized_matmul.reset(quantized_token)
            x = self.loop_hc.fold(streams)
        # Coda: fresh AttnRes segment after HC fold.
        x = self._run_block_stack(
            self.blocks[recurrent_end:],
            x,
            tokens,
            segment_ids,
            True,
            checkpoint_scope == "all",
        )
        return self.norm(x)

    def __call__(
        self,
        tokens: mx.array,
        segment_ids: mx.array | None = None,
        num_loops: int | None = None,
        return_mtp: bool = False,
        reset_memory: bool = True,
        checkpoint_activations: bool | str = False,
    ):
        hidden = self.hidden_states(
            tokens,
            segment_ids,
            num_loops,
            reset_memory,
            checkpoint_activations,
        )
        logits = self.logits_from(hidden)
        if return_mtp:
            return logits, self.mtp_logits(hidden)
        return logits
