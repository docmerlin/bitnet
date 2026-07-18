"""Experimental MLX port of the dense BitNet PaTH-FoX model path."""

from __future__ import annotations

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
from mlx_ternary_kernel import pack_ternary_weight, ternary_quantized_linear


_effective_weight_cache: ContextVar[dict[tuple[object, str], object] | None] = ContextVar(
    "effective_weight_cache",
    default=None,
)
_recurrent_quantized_matmul: ContextVar[bool] = ContextVar("recurrent_quantized_matmul", default=False)


@lru_cache(maxsize=32)
def _memory_pool_weights(length: int, memory_dim: int) -> mx.array:
    weights = np.zeros((memory_dim, length), dtype=np.float32)
    for index in range(memory_dim):
        start = index * length // memory_dim
        end = max(start + 1, ((index + 1) * length + memory_dim - 1) // memory_dim)
        weights[index, start:end] = 1.0 / (end - start)
    return mx.array(weights)


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
    path_window_size: int = 64
    infini_memory_dim: int = 64
    activation_bits: int = 4
    use_4bit_activations: bool = True
    use_hadamard: bool = True
    use_path_kernel: bool = True
    rms_norm_eps: float = 1e-5
    use_engram: bool = True
    engram_layer_ids: tuple[int, ...] = (1, 15)
    engram_vocab_size: int = 4093
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

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.path_window_size < 1:
            raise ValueError("path_window_size must be positive")
        if min(self.num_prelude_layers, self.num_recurrent_layers, self.num_coda_layers) < 0:
            raise ValueError("layer counts must be non-negative")
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        if self.block_size < 1 or self.infini_memory_dim < 1:
            raise ValueError("block_size and infini_memory_dim must be positive")
        if self.activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if self.engram_vocab_size < 1 or self.engram_max_ngram_size < 2:
            raise ValueError("invalid Engram table or N-gram size")
        if self.rfmoe_num_experts < 1:
            raise ValueError("rfmoe_num_experts must be positive")
        if self.rfmoe_backend not in {"auto", "metal", "hybrid", "host"}:
            raise ValueError("rfmoe_backend must be auto, metal, hybrid, or host")

    @property
    def num_hidden_layers(self) -> int:
        return self.num_prelude_layers + self.num_recurrent_layers + self.num_coda_layers

    @property
    def effective_depth(self) -> int:
        return self.num_prelude_layers + self.num_recurrent_layers * self.num_loops + self.num_coda_layers


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
        self.activation_levels = mx.array(float((2 ** (config.activation_bits - 1)) - 1))
        self.freeze(keys=["weight_mix", "activation_mix", "activation_levels"], recurse=False)

    def prepare_input(self, x: mx.array) -> mx.array:
        if self.config.use_hadamard and self.input_dims & (self.input_dims - 1) == 0:
            x = mx.hadamard_transform(x)
        if self.config.use_4bit_activations:
            levels = self.activation_levels
            negative_levels = levels + 1
            activation_scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5) / levels
            quantized_x = mx.clip(mx.round(x / activation_scale), -negative_levels, levels) * activation_scale
            x = x + self.activation_mix * mx.stop_gradient(quantized_x - x)
        return x

    def effective_weight(
        self,
        dtype,
        weight: mx.array | None = None,
        cache_key: object | None = None,
    ) -> mx.array:
        cache = _effective_weight_cache.get()
        if cache is not None:
            key = (id(self) if weight is None else cache_key, str(dtype))
            if key[0] is not None and key in cache:
                return cache[key]
        weight = self.weight if weight is None else weight
        weight_scale = mx.maximum(mx.mean(mx.abs(weight), axis=-1, keepdims=True), 1e-5)
        normalized = weight / weight_scale
        ternary = mx.where(normalized > 0.5, 1.0, mx.where(normalized < -0.5, -1.0, 0.0))
        quantized_weight = ternary * weight_scale
        effective = (weight + self.weight_mix * mx.stop_gradient(quantized_weight - weight)).astype(dtype)
        if cache is not None and key[0] is not None:
            cache[key] = effective
        return effective

    def __call__(self, x: mx.array) -> mx.array:
        x = self.prepare_input(x)
        if _recurrent_quantized_matmul.get() and min(self.weight.shape) >= 512:
            cache = _effective_weight_cache.get()
            key = (id(self), f"packed-{x.dtype}")
            packed_weight = cache.get(key) if cache is not None else None
            if packed_weight is None:
                packed_weight = pack_ternary_weight(self.weight)
                if cache is not None:
                    cache[key] = packed_weight
            packed, scales, _ = packed_weight
            return ternary_quantized_linear(x, self.weight, packed, scales)
        return x @ self.effective_weight(x.dtype).T

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        self.weight_mix = mx.array(weight_mix)
        self.activation_mix = mx.array(activation_mix)
        self.activation_levels = mx.array(float((2 ** (max(bits, 2) - 1)) - 1))


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
        convolved = normalized * self.short_conv_weight[:, -1]
        for lag in range(1, self.config.engram_kernel_size):
            if history is not None and history.shape[1] >= lag:
                index = history.shape[1] - lag
                convolved = convolved + history[:, index : index + 1] * self.short_conv_weight[:, -1 - lag]
        cache.normalized = normalized if history is None else mx.concatenate((history, normalized), axis=1)
        cache.normalized = cache.normalized[:, -(self.config.engram_kernel_size - 1) :]
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
        self.w_down = MLXHBitLinear(expert_dim, hidden, config)
        self.bias = mx.array([1e-6])

    def __call__(self, x: mx.array):
        z = self.a_gate(x)
        gate = mx.maximum(mx.linalg.norm(z, axis=-1) - self.bias, 0.0)
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
        gate_stack = mx.maximum(mx.linalg.norm(z, axis=-1) - biases, 0.0)
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
        normalized = centered / mx.maximum(mx.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
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
        self.out.weight = self.out.weight * 0.01
        self.memory_k = mx.zeros((0, config.num_attention_heads, config.infini_memory_dim, self.head_dim))
        self.memory_v = mx.zeros_like(self.memory_k)
        self.memory_initialized = mx.zeros((0,), dtype=mx.bool_)
        self.freeze(keys=["memory_k", "memory_v", "memory_initialized"], recurse=False)

    def reset_memory(self, batch_size: int | None = None) -> None:
        batch_size = self.memory_k.shape[0] if batch_size is None else batch_size
        shape = (
            batch_size,
            self.config.num_attention_heads,
            self.config.infini_memory_dim,
            self.head_dim,
        )
        self.memory_k = mx.zeros(shape, dtype=mx.float32)
        self.memory_v = mx.zeros(shape, dtype=mx.float32)
        self.memory_initialized = mx.zeros((batch_size,), dtype=mx.bool_)

    def new_inference_cache(self, batch_size: int) -> "MLXPaTHInferenceCache":
        shape = (
            batch_size,
            self.config.num_attention_heads,
            self.config.infini_memory_dim,
            self.head_dim,
        )
        return MLXPaTHInferenceCache(
            memory_k=mx.zeros(shape, dtype=mx.float32),
            memory_v=mx.zeros(shape, dtype=mx.float32),
            memory_initialized=mx.zeros((batch_size,), dtype=mx.bool_),
        )

    def _next_memory(
        self,
        keys: mx.array,
        values: mx.array,
        rows: mx.array,
        memory_k: mx.array,
        memory_v: mx.array,
        memory_initialized: mx.array,
    ):
        length = keys.shape[2]
        memory_dim = self.config.infini_memory_dim
        if length == memory_dim:
            key_update, value_update = keys, values
        elif length % memory_dim == 0:
            pooled_shape = (*keys.shape[:2], memory_dim, length // memory_dim, keys.shape[-1])
            key_update = mx.mean(keys.reshape(pooled_shape), axis=3)
            value_update = mx.mean(values.reshape(pooled_shape), axis=3)
        elif memory_dim % length == 0:
            repeats = memory_dim // length
            key_update = mx.repeat(keys, repeats, axis=2)
            value_update = mx.repeat(values, repeats, axis=2)
        else:
            weights = _memory_pool_weights(length, memory_dim)
            key_update = mx.einsum("ml,bhld->bhmd", weights, keys.astype(mx.float32))
            value_update = mx.einsum("ml,bhld->bhmd", weights, values.astype(mx.float32))
        key_update = key_update.astype(mx.float32)
        value_update = value_update.astype(mx.float32)
        rows = rows[:, None, None, None]
        next_k = mx.where(rows, 0.99 * memory_k + 0.01 * mx.stop_gradient(key_update), memory_k)
        next_v = mx.where(rows, 0.99 * memory_v + 0.01 * mx.stop_gradient(value_update), memory_v)
        return next_k, next_v, memory_initialized | rows[:, 0, 0, 0]

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
        vectors = convolved / mx.maximum(mx.linalg.norm(convolved, axis=-1, keepdims=True), 1e-6)
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
        w = convolved / mx.maximum(mx.linalg.norm(convolved, axis=-1, keepdims=True), 1e-6)
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
        if cache.path_projected is not None:
            history = cache.path_projected
            if history.shape[1] >= 1:
                convolved = convolved + history[:, -1:] * self.path_conv_weight[:, 1]
            if history.shape[1] >= 2:
                convolved = convolved + history[:, -2:-1] * self.path_conv_weight[:, 0]
            cache.path_projected = mx.concatenate((history, projected), axis=1)[:, -2:]
        else:
            cache.path_projected = projected
        w = nn.silu(convolved.astype(mx.float32)).reshape(
            batch, 1, self.config.num_attention_heads, self.head_dim
        )
        w = w / mx.maximum(mx.linalg.norm(w, axis=-1, keepdims=True), 1e-6)
        beta = 2.0 * mx.sigmoid(self.path_beta(x).astype(mx.float32))
        forget_logits = self.path_forget(x).astype(mx.float32)
        log_forget = -mx.logaddexp(mx.zeros_like(forget_logits), -forget_logits)

        cache.q = q if cache.q is None else mx.concatenate((cache.q, q), axis=2)
        cache.k = k if cache.k is None else mx.concatenate((cache.k, k), axis=2)
        cache.v = v if cache.v is None else mx.concatenate((cache.v, v), axis=2)
        cache.w = w if cache.w is None else mx.concatenate((cache.w, w), axis=1)
        cache.beta = beta if cache.beta is None else mx.concatenate((cache.beta, beta), axis=1)
        cache.log_forget = (
            log_forget if cache.log_forget is None else mx.concatenate((cache.log_forget, log_forget), axis=1)
        )

        local = self.path_chunk(cache.q, cache.k, cache.v, cache.w, cache.beta, cache.log_forget, None)
        local = local[:, :, -1:]
        scores = q.astype(mx.float32) @ cache.memory_k.swapaxes(-1, -2)
        scores = scores * self.head_dim**-0.5
        memory_context = mx.softmax(scores, axis=-1).astype(v.dtype) @ cache.memory_v.astype(v.dtype)
        gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
        mixed = (1.0 - gate) * local + gate * memory_context
        local = mx.where(cache.memory_initialized[:, None, None, None], mixed, local)
        context = local.transpose(0, 2, 1, 3).reshape(batch, 1, hidden)

        chunk_width = min(self.fixed_block_width or self.config.path_window_size, self.config.path_window_size)
        if cache.q.shape[2] == chunk_width:
            if update_memory:
                cache.memory_k, cache.memory_v, cache.memory_initialized = self._next_memory(
                    cache.k,
                    cache.v,
                    mx.ones((batch,), dtype=mx.bool_),
                    cache.memory_k,
                    cache.memory_v,
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
            scores = q_new.astype(mx.float32) @ cache.memory_k.swapaxes(-1, -2)
            scores = scores * self.head_dim**-0.5
            memory_context = mx.softmax(scores, axis=-1).astype(v.dtype) @ cache.memory_v.astype(v.dtype)
            gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
            mixed = (1.0 - gate) * local + gate * memory_context
            outputs.append(mx.where(cache.memory_initialized[:, None, None, None], mixed, local))
            if chunk_q.shape[2] == chunk_width:
                if update_memory:
                    cache.memory_k, cache.memory_v, cache.memory_initialized = self._next_memory(
                        chunk_k,
                        chunk_v,
                        mx.ones((batch,), dtype=mx.bool_),
                        cache.memory_k,
                        cache.memory_v,
                        cache.memory_initialized,
                    )
                cache.clear_open_chunk()
            else:
                cache.q, cache.k, cache.v = chunk_q, chunk_k, chunk_v
                cache.w, cache.beta, cache.log_forget = chunk_w, chunk_beta, chunk_forget
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
            scores = q[:, :, start:end].astype(mx.float32) @ cache.memory_k.swapaxes(-1, -2)
            scores = scores * self.head_dim**-0.5
            memory_context = mx.softmax(scores, axis=-1).astype(v.dtype) @ cache.memory_v.astype(v.dtype)
            gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
            mixed = (1.0 - gate) * local + gate * memory_context
            local = mx.where(cache.memory_initialized[:, None, None, None], mixed, local)
            if end - start == chunk_width:
                if update_memory:
                    cache.memory_k, cache.memory_v, cache.memory_initialized = self._next_memory(
                        k[:, :, start:end],
                        v[:, :, start:end],
                        mx.ones((batch,), dtype=mx.bool_),
                        cache.memory_k,
                        cache.memory_v,
                        cache.memory_initialized,
                    )
            else:
                cache.q = q[:, :, start:end]
                cache.k = k[:, :, start:end]
                cache.v = v[:, :, start:end]
                cache.w = w[:, start:end]
                cache.beta = beta[:, start:end]
                cache.log_forget = log_forget[:, start:end]
            chunks.append(local)
        cache.path_projected = projected[:, -2:]
        context = mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.out(context)

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
        diagonal = eye * beta[..., :, None]
        solve = path_triangular_solve if self.config.use_path_kernel else reference_triangular_solve
        t_inverse = solve(system, diagonal)
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

    def forward_arrays(
        self,
        x: mx.array,
        segment_ids: mx.array | None,
        memory_k: mx.array,
        memory_v: mx.array,
        memory_initialized: mx.array,
        update_memory: bool,
    ):
        batch, length, hidden = x.shape
        q, k, v, w, beta, log_forget, _ = self._project(x, segment_ids)
        chunks = []
        block_width = self.fixed_block_width or (length + self.num_blocks - 1) // self.num_blocks
        memory_safe = (
            mx.ones((batch,), dtype=mx.bool_)
            if segment_ids is None
            else mx.all(segment_ids == segment_ids[:, :1], axis=1)
        )
        for block_start in range(0, length, block_width):
            block_end = min(block_start + block_width, length)
            for start in range(block_start, block_end, self.config.path_window_size):
                end = min(start + self.config.path_window_size, block_end)
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
                scores = q[:, :, start:end].astype(mx.float32) @ memory_k.swapaxes(-1, -2)
                scores = scores * self.head_dim**-0.5
                memory_context = mx.softmax(scores, axis=-1).astype(v.dtype) @ memory_v.astype(v.dtype)
                gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
                mixed = (1.0 - gate) * local + gate * memory_context
                use_memory = memory_initialized & memory_safe
                local = mx.where(use_memory[:, None, None, None], mixed, local)
                if update_memory:
                    memory_k, memory_v, memory_initialized = self._next_memory(
                        k[:, :, start:end],
                        v[:, :, start:end],
                        memory_safe,
                        memory_k,
                        memory_v,
                        memory_initialized,
                    )
                chunks.append(local)
        context = mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.out(context), memory_k, memory_v, memory_initialized

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
        checkpoint_activations: bool = False,
    ) -> mx.array:
        memory = (self.memory_k, self.memory_v, self.memory_initialized)
        if checkpoint_activations:
            run = activation_checkpoint(self, self.forward_arrays)
        else:
            run = self.forward_arrays
        output, memory_k, memory_v, memory_initialized = run(
            x,
            segment_ids,
            *memory,
            update_memory,
        )
        if update_memory:
            self.memory_k = memory_k
            self.memory_v = memory_v
            self.memory_initialized = memory_initialized
        return output


@dataclass
class MLXPaTHInferenceCache:
    memory_k: mx.array
    memory_v: mx.array
    memory_initialized: mx.array
    path_projected: mx.array | None = None
    q: mx.array | None = None
    k: mx.array | None = None
    v: mx.array | None = None
    w: mx.array | None = None
    beta: mx.array | None = None
    log_forget: mx.array | None = None

    def clear_open_chunk(self) -> None:
        self.q = self.k = self.v = self.w = self.beta = self.log_forget = None

    def clone(self) -> "MLXPaTHInferenceCache":
        return MLXPaTHInferenceCache(
            self.memory_k,
            self.memory_v,
            self.memory_initialized,
            self.path_projected,
            self.q,
            self.k,
            self.v,
            self.w,
            self.beta,
            self.log_forget,
        )

    def arrays(self) -> list[mx.array]:
        return [
            value
            for value in (
                self.memory_k,
                self.memory_v,
                self.memory_initialized,
                self.path_projected,
                self.q,
                self.k,
                self.v,
                self.w,
                self.beta,
                self.log_forget,
            )
            if value is not None
        ]


class MLXHybridBlock(nn.Module):
    def __init__(self, config: MLXBitNetConfig, layer_id: int):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        self.engram = MLXEngram(config, layer_id) if config.use_engram and layer_id in config.engram_layer_ids else None
        self.attn_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.attn = MLXPaTHAttention(config)
        self.attn_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.attn_scale = mx.array([0.1])
        self.attn_gate = mx.array([0.0])
        self.mlp_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.moe = MLXRFMoE(config) if config.use_rfmoe else None
        if self.moe is None:
            self.up = MLXHBitLinear(hidden, intermediate * 2, config)
            self.mid = MLXHBitLinear(intermediate, intermediate, config)
            self.down = MLXHBitLinear(intermediate, hidden, config)
            self.down.weight = self.down.weight * 0.01
        self.mlp_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.mlp_scale = mx.array([0.1])

    def _dense_mlp(self, x: mx.array) -> mx.array:
        gate, value = mx.split(self.up(x), 2, axis=-1)
        return self.down(nn.silu(self.mid(nn.silu(gate) * value)))

    def new_inference_cache(self, batch_size: int) -> "MLXBlockInferenceCache":
        return MLXBlockInferenceCache(
            attention=self.attn.new_inference_cache(batch_size),
            engram=MLXEngramInferenceCache() if self.engram is not None else None,
        )

    def incremental(
        self,
        x: mx.array,
        input_ids: mx.array,
        token_history: mx.array,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        if self.engram is not None:
            x = x + self.engram.incremental(x, input_ids, token_history, cache.engram)
        attention = self.attn.incremental(self.attn_norm(x), cache.attention, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        normalized = self.mlp_norm(x)
        output = self.moe(normalized) if self.moe is not None else self._dense_mlp(normalized)
        return self.mlp_post(x + self.mlp_scale * output)

    def extend(
        self,
        x: mx.array,
        input_ids: mx.array,
        token_history: mx.array,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        if self.engram is not None:
            x = x + self.engram.extend(x, input_ids, token_history, cache.engram)
        attention = self.attn.extend(self.attn_norm(x), cache.attention, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        normalized = self.mlp_norm(x)
        output = self.moe(normalized) if self.moe is not None else self._dense_mlp(normalized)
        return self.mlp_post(x + self.mlp_scale * output)

    def prefill(
        self,
        x: mx.array,
        input_ids: mx.array,
        cache: "MLXBlockInferenceCache",
        update_memory: bool,
    ) -> mx.array:
        if self.engram is not None:
            x = x + self.engram.prefill(x, input_ids, cache.engram)
        attention = self.attn.prefill(self.attn_norm(x), cache.attention, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        normalized = self.mlp_norm(x)
        output = self.moe(normalized) if self.moe is not None else self._dense_mlp(normalized)
        return self.mlp_post(x + self.mlp_scale * output)

    def __call__(
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
        attention = self.attn(
            self.attn_norm(x),
            segment_ids,
            update_memory,
            checkpoint_activations,
        )
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        normalized = self.mlp_norm(x)
        if self.moe is not None:
            if checkpoint_activations and self.moe.backend != "hybrid":
                output, gate_stack, hard_density = activation_checkpoint(
                    self.moe,
                    self.moe.forward_arrays,
                )(normalized)
                self.moe.update_stats(gate_stack, hard_density)
            else:
                output = self.moe(normalized)
        else:
            run_mlp = activation_checkpoint(self, self._dense_mlp) if checkpoint_activations else self._dense_mlp
            output = run_mlp(normalized)
        return self.mlp_post(x + self.mlp_scale * output)


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
    position: int = 0

    def clone(self) -> "MLXInferenceCache":
        return MLXInferenceCache(
            [layer.clone() for layer in self.layers],
            self.token_history,
            self.num_loops,
            self.position,
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
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
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
            block.attn.reset_memory(batch_size)

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        def update(_, module):
            if isinstance(module, MLXHBitLinear):
                module.set_quantization_state(weight_mix, activation_mix, bits)

        self.apply_to_modules(update)

    def set_active_blocks(self, blocks: int) -> None:
        for block in self.blocks:
            block.attn.num_blocks = max(1, blocks)

    def set_inference_block_width(self, width: int | None) -> None:
        for block in self.blocks:
            block.attn.fixed_block_width = width

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
            [block.new_inference_cache(batch_size) for block in execution],
            mx.zeros((batch_size, 0), dtype=mx.int32),
            loops,
        )

    def inference_step(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("inference_step requires one token per batch row")
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        cache_index = 0
        for block in self.blocks[:prelude_end]:
            x = block.incremental(x, tokens, cache.token_history, cache.layers[cache_index], True)
            cache_index += 1
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            for loop_index in range(cache.num_loops):
                x, post, residual = self.loop_hc.project_in(streams)
                for block in self.blocks[prelude_end:recurrent_end]:
                    x = block.incremental(
                        x,
                        tokens,
                        cache.token_history,
                        cache.layers[cache_index],
                        loop_index == cache.num_loops - 1,
                    )
                    cache_index += 1
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        for block in self.blocks[recurrent_end:]:
            x = block.incremental(x, tokens, cache.token_history, cache.layers[cache_index], True)
            cache_index += 1
        cache.token_history = mx.concatenate((cache.token_history, tokens), axis=1)[
            :, -(self.config.engram_max_ngram_size - 1) :
        ]
        cache.position += 1
        return self.norm(x)

    def inference_extend(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("inference_extend requires a non-empty rank-2 token array")
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        cache_index = 0
        for block in self.blocks[:prelude_end]:
            x = block.extend(x, tokens, cache.token_history, cache.layers[cache_index], True)
            cache_index += 1
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            for loop_index in range(cache.num_loops):
                x, post, residual = self.loop_hc.project_in(streams)
                for block in self.blocks[prelude_end:recurrent_end]:
                    x = block.extend(
                        x,
                        tokens,
                        cache.token_history,
                        cache.layers[cache_index],
                        loop_index == cache.num_loops - 1,
                    )
                    cache_index += 1
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        for block in self.blocks[recurrent_end:]:
            x = block.extend(x, tokens, cache.token_history, cache.layers[cache_index], True)
            cache_index += 1
        cache.token_history = mx.concatenate((cache.token_history, tokens), axis=1)[
            :, -(self.config.engram_max_ngram_size - 1) :
        ]
        cache.position += tokens.shape[1]
        return self.norm(x)

    def prefill(self, tokens: mx.array, cache: MLXInferenceCache) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("prefill requires a non-empty rank-2 token array")
        if cache.position:
            raise ValueError("prefill requires an empty inference cache")
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        cache_index = 0
        for block in self.blocks[:prelude_end]:
            x = block.prefill(x, tokens, cache.layers[cache_index], True)
            cache_index += 1
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            for loop_index in range(cache.num_loops):
                x, post, residual = self.loop_hc.project_in(streams)
                for block in self.blocks[prelude_end:recurrent_end]:
                    x = block.prefill(
                        x,
                        tokens,
                        cache.layers[cache_index],
                        loop_index == cache.num_loops - 1,
                    )
                    cache_index += 1
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        for block in self.blocks[recurrent_end:]:
            x = block.prefill(x, tokens, cache.layers[cache_index], True)
            cache_index += 1
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

    def mtp_logits(self, hidden: mx.array) -> list[mx.array]:
        return [transform(hidden) @ self.embedding.weight.T for transform in self.mtp_transforms]

    def selected_mtp_logits(self, hidden: mx.array, selector: mx.array) -> mx.array:
        transformed = mx.stack([transform(hidden) for transform in self.mtp_transforms], axis=2)
        selected = mx.sum(transformed * selector[None, None, :, None], axis=2)
        return selected @ self.embedding.weight.T

    def draft_logits(self, hidden: mx.array) -> mx.array:
        last = hidden[:, -1:]
        transformed = mx.concatenate([transform(last) for transform in self.mtp_transforms], axis=1)
        return transformed @ self.embedding.weight.T

    def hidden_states(
        self,
        tokens: mx.array,
        segment_ids: mx.array | None = None,
        num_loops: int | None = None,
        reset_memory: bool = True,
        checkpoint_activations: bool | str = False,
    ) -> mx.array:
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        checkpoint_scope = "all" if checkpoint_activations is True else checkpoint_activations or "none"
        if checkpoint_scope not in ("none", "recurrent", "all"):
            raise ValueError("checkpoint_activations must be none, recurrent, or all")
        if reset_memory:
            self.reset_memory(tokens.shape[0])
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        for block in self.blocks[:prelude_end]:
            x = block(x, tokens, segment_ids, True, checkpoint_scope == "all")
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            cache_token = _effective_weight_cache.set({}) if self.reuse_recurrent_weights and loops > 1 else None
            quantized_token = _recurrent_quantized_matmul.set(self.recurrent_quantized_matmul)
            try:
                for loop_index in range(loops):
                    x, post, residual = self.loop_hc.project_in(streams)
                    for block in self.blocks[prelude_end:recurrent_end]:
                        x = block(
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
        for block in self.blocks[recurrent_end:]:
            x = block(x, tokens, segment_ids, True, checkpoint_scope == "all")
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
        logits = hidden @ self.embedding.weight.T
        if return_mtp:
            return logits, self.mtp_logits(hidden)
        return logits
