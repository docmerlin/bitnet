"""MLX transformer primitives for the ternary BLT student.

A numerical mirror of :mod:`blt.layers.transformer_block` and
:mod:`blt.layers.cross_attention`, not of the BitNet stack's ``MLXHBitLinear``.
The two stacks agree on ternary weight quantization and on the activation
quantization scheme (width from ``config.activation_bits``), and ``mx.hadamard_transform`` matches the torch dense Hadamard
matmul to float32 precision, but BLT's blocks differ in shape (SwiGLU with an
identity-initialised ``mid_proj``, cross-attention with a projected residual), so
these are written against the BLT torch modules and parity-tested against them.

Conventions worth stating because they are easy to get silently wrong:

- RoPE rotates *interleaved pairs* (``x[0::2]``/``x[1::2]``), the GPT-NeoX form,
  not the half-split Llama form. ``cos``/``sin`` are repeat-interleaved to match.
- ``RMSNorm`` uses the float32 machine epsilon, because ``torch.nn.RMSNorm``
  defaults ``eps=None`` and falls back to ``finfo(dtype).eps``. MLX's own default
  is 1e-5, which would quietly diverge.
- Fully masked query rows are zeroed after attention rather than left to softmax
  into a uniform distribution over dropped keys.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from blt.config import TernaryBLTConfig
from blt.ngram_hash import HASH_MODULUS, HASH_PAD, hash_bases
from mlx_ternary_kernel import activation_levels, ste_activation_quantize

# torch.nn.RMSNorm(dim) leaves eps=None and uses finfo(x.dtype).eps at runtime.
_TORCH_FLOAT32_EPS = 1.1920928955078125e-07

_MASK_FLOOR = -3.4028234663852886e38


def rotate_half(x: mx.array) -> mx.array:
    """Rotate the last dimension by pairs, matching ``utils.rotate_half``."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return mx.stack([-x_odd, x_even], axis=-1).reshape(x.shape)


def build_rope_cache(seq_len: int, dim: int, *, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    positions = mx.arange(seq_len, dtype=mx.float32)
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    freqs = positions[:, None] * inv_freq[None, :]
    return mx.repeat(mx.cos(freqs), 2, axis=-1), mx.repeat(mx.sin(freqs), 2, axis=-1)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """``x`` is [batch, heads, seq, head_dim]."""
    cos = cos[None, None].astype(x.dtype)
    sin = sin[None, None].astype(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def causal_window_attention_bias(seq_len: int, window: int | None, *, dtype=mx.float32) -> mx.array:
    """Additive bias for causal attention, optionally limited to a sliding window."""
    q_pos = mx.arange(seq_len).reshape(seq_len, 1)
    k_pos = mx.arange(seq_len).reshape(1, seq_len)
    invalid = k_pos > q_pos
    if window is not None and window > 0:
        invalid = invalid | (q_pos - k_pos >= window)
    return mx.where(invalid, mx.array(_MASK_FLOOR, dtype=dtype), mx.array(0.0, dtype=dtype))


def windowed_block_attention_bias(window: int, *, dtype=mx.float32) -> mx.array:
    """Mask for one query block against ``[previous block, current block]``.

    Query at offset ``p`` in a block covering absolute positions ``[jw, jw+w)``
    attends to absolute ``[jw+p-w+1, jw+p]``, which in the concatenated pair is
    exactly ``[p+1, p+w]`` -- a fixed band, the same for every block. Shape
    ``[w, 2w]``.
    """
    p = mx.arange(window).reshape(window, 1)
    c = mx.arange(2 * window).reshape(1, 2 * window)
    keep = (c >= p + 1) & (c <= p + window)
    return mx.where(keep, mx.array(0.0, dtype=dtype), mx.array(_MASK_FLOOR, dtype=dtype))


def combine_attention_bias(
    attention_mask: mx.array | None,
    *,
    base_bias: mx.array | None,
    batch_size: int,
    q_len: int,
    k_len: int,
    dtype=mx.float32,
) -> tuple[mx.array | None, mx.array | None]:
    """Fold a caller mask into ``base_bias``; mirrors ``utils.combine_attention_bias``.

    Returns ``(attn_bias, query_valid)``. ``query_valid`` marks rows the caller
    must zero afterwards: a row with every key dropped still softmaxes to a
    uniform distribution over nothing, which is not zero.
    """
    if attention_mask is None:
        return base_bias, None

    if attention_mask.ndim == 2:
        keep = attention_mask[:, None, None, :k_len].astype(mx.bool_)
        query_valid = attention_mask[:, None, :q_len, None].astype(mx.bool_)
    else:
        keep = attention_mask[:, None, :q_len, :k_len].astype(mx.bool_)
        query_valid = mx.any(keep, axis=-1, keepdims=True)

    if base_bias is None:
        attn_bias = mx.zeros((batch_size, 1, q_len, k_len), dtype=dtype)
    else:
        attn_bias = mx.broadcast_to(base_bias, (batch_size, 1, q_len, k_len))
    attn_bias = mx.where(keep, attn_bias, mx.array(_MASK_FLOOR, dtype=dtype))
    return attn_bias, query_valid


class MLXHBitLinear(nn.Module):
    """Ternary linear with Hadamard-preconditioned, fake-quantized activations.

    Matches ``layers.h_bitlinear.HBitLinear``: per-output-channel abs-mean weight
    scale, symmetric per-token activation scale, straight-through estimators on
    both. Weight is stored ``[out, in]`` so checkpoints transfer directly.
    """

    def __init__(self, in_features: int, out_features: int, *, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_hadamard = bool(config.use_hadamard) and in_features & (in_features - 1) == 0
        self.quantize_activations = bool(config.use_4bit_activations)
        self.activation_bits = int(config.activation_bits)
        # Quantisation strength, ramped during training rather than fixed.
        # Fully quantised activations from step 0 diverge: at activation_mix=1.0
        # the model collapses to uniform output after one update and NaNs on the
        # next, while activation_mix=0.0 trains normally at any weight_mix.
        # Ternary weights are not the problem. These mirror
        # layers.h_bitlinear.HBitLinear, which has had them all along.
        self.weight_mix = 1.0
        self.activation_mix = 1.0
        # When set, set_quantization_state leaves weight_mix alone. Used by the
        # identity-initialised FFN mid; see MLXTernaryMLP.
        self.pinned_weight_mix: float | None = None
        # The ramped values also live as module state, because the trainer wraps
        # the step in mx.compile and a Python float read inside a traced function
        # is baked in as a graph constant. Reassigning the float then changes
        # nothing: measured, a 16-bit and a 4-bit forward returned identical
        # output under compile, i.e. the whole quantisation ramp was inert.
        # Arrays in module state are graph *inputs*, so they track. The plain
        # floats above stay for the branch decisions, which genuinely do need a
        # retrace when they flip.
        self.weight_mix_value = mx.array(1.0)
        self.activation_mix_value = mx.array(1.0)
        self.activation_level_pair = activation_levels(self.activation_bits)
        self.freeze(
            keys=["weight_mix_value", "activation_mix_value", "activation_level_pair"],
            recurse=False,
        )

        # kaiming_uniform_(a=sqrt(5)) reduces to U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        bound = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(low=-bound, high=bound, shape=(out_features, in_features))
        # Generation-only: materialize ternarized weight once per generate() call
        # instead of redoing abs-mean scale + threshold every matmul. None while training.
        self._pinned_weight: mx.array | None = None

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        """Ramp quantisation strength. 0.0 is full precision, 1.0 fully quantised."""
        if self.pinned_weight_mix is None:
            self.weight_mix = float(min(max(weight_mix, 0.0), 1.0))
        else:
            self.weight_mix = self.pinned_weight_mix
        self.activation_mix = float(min(max(activation_mix, 0.0), 1.0))
        self.activation_bits = max(int(bits), 2)
        self.weight_mix_value = mx.array(self.weight_mix)
        self.activation_mix_value = mx.array(self.activation_mix)
        self.activation_level_pair = activation_levels(self.activation_bits)

    def prepare_input(self, x: mx.array) -> mx.array:
        if self.use_hadamard:
            x = mx.hadamard_transform(x)
        if not self.quantize_activations:
            return x
        # One fused kernel rather than the nine elementwise passes the
        # expression form costs -- absmax, divide, round, clip, rescale, and the
        # straight-through add. Measured 2-7x on the quantiser and 41-75% of a
        # whole HBitLinear. Bit-identical output; the STE is the kernel's vjp.
        quantized = ste_activation_quantize(x, self.activation_level_pair)
        # One branch-free expression rather than early-returning on the mix. A
        # Python `if` on a ramped value is evaluated once, at trace time, and
        # baked into the compiled graph: the trainer starts the ramp at
        # activation_mix=0.0, so the compiled step took the "return x" branch and
        # never quantised activations again for the whole run. Measured -- a
        # compiled forward at mix=0 and at mix=1 returned identical output.
        # At mix=1 this is x + (q - x) = q with an identity gradient, and at
        # mix=0 it is x, so nothing is lost but the early return.
        return x + self.activation_mix_value * mx.stop_gradient(quantized - x)

    def effective_weight(self) -> mx.array:
        if self._pinned_weight is not None:
            return self._pinned_weight
        weight = self.weight
        if self.weight_mix <= 0.0:
            return weight
        scale = mx.maximum(mx.mean(mx.abs(mx.stop_gradient(weight)), axis=-1, keepdims=True), 1e-5)
        normalized = weight / scale
        ternary = mx.where(normalized > 0.5, 1.0, mx.where(normalized < -0.5, -1.0, 0.0))
        if self.weight_mix >= 1.0:
            # stop_gradient(q) + (w - stop_gradient(w)), not w + stop_gradient(q - w).
            # Same value and same identity gradient, but the second form computes
            # q - w, and the identity-initialised FFN mid has w ~ N against q ~ 1,
            # so that subtraction loses most of its significant digits.
            return mx.stop_gradient(ternary * scale) + (weight - mx.stop_gradient(weight))
        return weight + self.weight_mix_value * mx.stop_gradient(ternary * scale - weight)

    def pin_inference_weight(self) -> None:
        """Materialize one effective weight for the generation lifetime."""
        self._pinned_weight = None  # force recompute from current master
        self._pinned_weight = self.effective_weight()

    def clear_pinned_inference_weight(self) -> None:
        self._pinned_weight = None

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward_prepared(self.prepare_input(x))

    def forward_prepared(self, x: mx.array) -> mx.array:
        """Project input prepared by a layer with matching activation settings."""
        return x @ self.effective_weight().T


def _rms_norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=_TORCH_FLOAT32_EPS)


def _attend(q: mx.array, k: mx.array, v: mx.array, bias: mx.array | None, valid: mx.array | None) -> mx.array:
    scale = 1.0 / math.sqrt(q.shape[-1])
    context = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
    if valid is not None:
        context = mx.where(valid, context, 0.0)
    return context


class MLXTernarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        config: TernaryBLTConfig,
        local_window: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_window = local_window
        self.causal = causal
        self.rope_theta = config.rope_theta

        self.q_proj = MLXHBitLinear(dim, dim, config=config)
        self.k_proj = MLXHBitLinear(dim, dim, config=config)
        self.v_proj = MLXHBitLinear(dim, dim, config=config)
        self.o_proj = MLXHBitLinear(dim, dim, config=config)

    def _chunkable(self, seq_len: int, attention_mask: mx.array | None) -> bool:
        """Whether the windowed path can run block-by-block instead of dense.

        Needs a causal window that tiles the sequence and leaves at least two
        blocks -- below that the dense mask is already the smaller matrix. A
        caller mask is folded per key, which the shared band mask cannot express,
        so those fall back.
        """
        window = self.local_window
        return (
            self.causal
            and attention_mask is None
            and window is not None
            and 0 < window < seq_len
            and seq_len % window == 0
        )

    def _windowed_attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """Causal sliding-window attention over query blocks of ``local_window``.

        The dense path scores every query against every key and then throws away
        everything outside the band -- at sequence 1024 and window 256 that is
        75% of the work. Here each query block sees only the previous and current
        key blocks, folded onto the batch axis. Identical output to the dense mask.

        Block 0 runs separately rather than sharing the fold. Its "previous"
        block is zero padding that has to be masked away entirely, and
        broadcasting a per-block-instance mask to express that costs
        ``[batch*blocks, 1, window, 2*window]`` -- 33 MB at batch 16, sequence
        1024, window 256, against 4 MB for the dense mask this is replacing.
        Two calls with two shared masks allocate neither.
        """
        window = self.local_window
        batch, heads, seq_len, head_dim = q.shape
        blocks = seq_len // window
        scale = 1.0 / math.sqrt(head_dim)

        def blocked(t):
            return t.reshape(batch, heads, blocks, window, head_dim)

        def fold(t, count):
            return t.transpose(0, 2, 1, 3, 4).reshape(batch * count, heads, -1, head_dim)

        queries, keys, values = blocked(q), blocked(k), blocked(v)

        # Block 0: plain causal attention over its own keys.
        head_context = mx.fast.scaled_dot_product_attention(
            queries[:, :, 0], keys[:, :, 0], values[:, :, 0], scale=scale, mask="causal"
        )

        if blocks == 1:
            return head_context

        # Blocks 1..n-1: each against [previous, current], one shared band mask.
        def pair(t):
            return fold(mx.concatenate([t[:, :, :-1], t[:, :, 1:]], axis=3), blocks - 1)

        tail_context = mx.fast.scaled_dot_product_attention(
            fold(queries[:, :, 1:], blocks - 1),
            pair(keys),
            pair(values),
            scale=scale,
            mask=windowed_block_attention_bias(window, dtype=q.dtype),
        )
        tail_context = tail_context.reshape(batch, blocks - 1, heads, window, head_dim)
        context = mx.concatenate([head_context[:, :, None], tail_context.transpose(0, 2, 1, 3, 4)], axis=2)
        return context.reshape(batch, heads, seq_len, head_dim)

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

        def heads(t):
            return t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        same_preparation = all(
            layer.activation_mix == self.q_proj.activation_mix
            and layer.activation_bits == self.q_proj.activation_bits
            for layer in (self.k_proj, self.v_proj)
        )
        prepared = self.q_proj.prepare_input(x) if same_preparation else None
        q, k, v = (
            heads(layer(x) if prepared is None else layer.forward_prepared(prepared))
            for layer in (self.q_proj, self.k_proj, self.v_proj)
        )
        cos, sin = build_rope_cache(seq_len, self.head_dim, theta=self.rope_theta)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self._chunkable(seq_len, attention_mask):
            context = self._windowed_attend(q, k, v)
        else:
            base_bias = (
                causal_window_attention_bias(seq_len, self.local_window, dtype=q.dtype)
                if self.causal
                else None
            )
            bias, valid = combine_attention_bias(
                attention_mask,
                base_bias=base_bias,
                batch_size=batch_size,
                q_len=seq_len,
                k_len=seq_len,
                dtype=q.dtype,
            )
            context = _attend(q, k, v, bias, valid)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        return self.o_proj(context)


class MLXTernaryMLP(nn.Module):
    """SwiGLU expand -> mid (silu) -> down, with ``mid_proj`` seeded to identity."""

    def __init__(self, dim: int, multiplier: float, *, config: TernaryBLTConfig) -> None:
        super().__init__()
        hidden_dim = max(int(dim * multiplier), dim)
        self.hidden_dim = hidden_dim
        self.gate_proj = MLXHBitLinear(dim, hidden_dim, config=config)
        self.up_proj = MLXHBitLinear(dim, hidden_dim, config=config)
        self.mid_proj = MLXHBitLinear(hidden_dim, hidden_dim, config=config)
        # Scaled so ternarisation lands on a *true* identity. The per-output-channel
        # scale is mean(|row|); a plain eye(N) row is one 1 and N-1 zeros, so the
        # scale is 1/N and the quantised weight comes out as eye(N)/N -- a 1/1024
        # attenuator, not a pass-through. eye(N)*N gives mean(|row|) = 1, so the
        # quantised weight is exactly eye(N).
        #
        # The weight mix is pinned with it: the straight-through blend
        # (1-mix)*raw + mix*quantised only means anything when raw and quantised
        # share a scale, and here they differ by N. Ramping this particular matrix
        # would put it at 768x identity a quarter of the way through the ramp.
        self.mid_proj.weight = mx.eye(hidden_dim) * hidden_dim
        self.mid_proj.pinned_weight_mix = 1.0
        self.mid_proj.weight_mix = 1.0
        self.down_proj = MLXHBitLinear(hidden_dim, dim, config=config)

    def __call__(self, x: mx.array) -> mx.array:
        same_preparation = (
            self.gate_proj.activation_mix == self.up_proj.activation_mix
            and self.gate_proj.activation_bits == self.up_proj.activation_bits
        )
        prepared = self.gate_proj.prepare_input(x) if same_preparation else None
        gate = self.gate_proj(x) if prepared is None else self.gate_proj.forward_prepared(prepared)
        up = self.up_proj(x) if prepared is None else self.up_proj.forward_prepared(prepared)
        hidden = nn.silu(gate) * up
        return self.down_proj(nn.silu(self.mid_proj(hidden)))


class MLXTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        config: TernaryBLTConfig,
        ffn_multiplier: float,
        local_window: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = _rms_norm(dim)
        self.mlp_norm = _rms_norm(dim)
        self.attn = MLXTernarySelfAttention(
            dim, num_heads, config=config, local_window=local_window, causal=causal
        )
        self.mlp = MLXTernaryMLP(dim, ffn_multiplier, config=config)

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        return x + self.mlp(self.mlp_norm(x))


class MLXTernaryCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        *,
        hidden_dim: int,
        num_heads: int,
        config: TernaryBLTConfig,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (hidden_dim // num_heads) % 2 != 0:
            raise ValueError("cross-attention head_dim must be even")

        self.output_dim = output_dim or query_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_norm = _rms_norm(query_dim)
        self.kv_norm = _rms_norm(kv_dim)
        self.q_proj = MLXHBitLinear(query_dim, hidden_dim, config=config)
        self.k_proj = MLXHBitLinear(kv_dim, hidden_dim, config=config)
        self.v_proj = MLXHBitLinear(kv_dim, hidden_dim, config=config)
        self.out_proj = MLXHBitLinear(hidden_dim, self.output_dim, config=config)
        self.residual_proj = (
            MLXHBitLinear(query_dim, self.output_dim, config=config)
            if query_dim != self.output_dim
            else None
        )

    def __call__(self, query: mx.array, key_value: mx.array, *, mask: mx.array | None = None) -> mx.array:
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]
        normed_kv = self.kv_norm(key_value)

        def heads(t, length):
            return t.reshape(batch_size, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = heads(self.q_proj(self.query_norm(query)), query_len)
        same_preparation = (
            self.k_proj.activation_mix == self.v_proj.activation_mix
            and self.k_proj.activation_bits == self.v_proj.activation_bits
        )
        prepared = self.k_proj.prepare_input(normed_kv) if same_preparation else None
        k = heads(
            self.k_proj(normed_kv) if prepared is None else self.k_proj.forward_prepared(prepared),
            kv_len,
        )
        v = heads(
            self.v_proj(normed_kv) if prepared is None else self.v_proj.forward_prepared(prepared),
            kv_len,
        )

        bias, valid = combine_attention_bias(
            mask, base_bias=None, batch_size=batch_size, q_len=query_len, k_len=kv_len, dtype=q.dtype
        )
        context = _attend(q, k, v, bias, valid)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, query_len, self.hidden_dim)
        residual = self.residual_proj(query) if self.residual_proj is not None else query
        return residual + self.out_proj(context)


class MLXTernaryPatchGather(nn.Module):
    """Byte reads the value of the one patch it belongs to.

    Mirrors :class:`blt.layers.cross_attention.TernaryPatchGather`, which
    replaced a cross-attention whose mask was one-hot by construction. With a
    single permitted key the attention weights are a constant 1, so the query and
    key projections could not affect the output and received exactly zero
    gradient. Only ``kv_norm``, ``v_proj``, ``out_proj`` and the residual
    projection survive.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        *,
        hidden_dim: int,
        num_heads: int,
        config: TernaryBLTConfig,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.output_dim = output_dim or query_dim
        self.hidden_dim = hidden_dim
        self.kv_norm = _rms_norm(kv_dim)
        self.v_proj = MLXHBitLinear(kv_dim, hidden_dim, config=config)
        self.out_proj = MLXHBitLinear(hidden_dim, self.output_dim, config=config)
        self.residual_proj = (
            MLXHBitLinear(query_dim, self.output_dim, config=config)
            if query_dim != self.output_dim
            else None
        )

    def __call__(
        self,
        query: mx.array,
        key_value: mx.array,
        patch_ids: mx.array,
        *,
        valid: mx.array | None = None,
    ) -> mx.array:
        values = self.v_proj(self.kv_norm(key_value))
        # patch_ids is -1 on padded bytes; clamp to keep the gather in range and
        # zero those rows, matching what a fully masked attention row produced.
        safe = mx.maximum(patch_ids, 0)[..., None]
        gathered = mx.take_along_axis(
            values, mx.broadcast_to(safe, (*patch_ids.shape, values.shape[-1])), axis=1
        )
        if valid is not None:
            gathered = mx.where(valid[..., None], gathered, 0.0)
        residual = self.residual_proj(query) if self.residual_proj is not None else query
        return residual + self.out_proj(gathered)


class MLXHashNgramEmbedding(nn.Module):
    """Hashed byte n-gram embeddings summed into the byte embedding.

    Meta's BLT, eq. 3: ``e_i = (x_i + sum_n E_n[Hash(g_{i,n})]) / (|sizes| + 1)``.
    See :mod:`blt.ngram_hash` for why this exists and what the hash guarantees.

    One table holds every size, offset by slot, which keeps the whole thing to a
    single gather. The tables are ``ngram_dim`` wide rather than ``local_dim``
    and a shared projection lifts the sum back up -- at this scale Meta's
    full-width tables would outweigh the model.
    """

    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.sizes = tuple(config.ngram_sizes)
        self.vocab_size = int(config.ngram_vocab_size)
        self.dim = int(config.ngram_dim)
        self.embedding = nn.Embedding(len(self.sizes) * self.vocab_size, self.dim)
        self.proj = (
            MLXHBitLinear(self.dim, config.local_dim, config=config)
            if self.dim != config.local_dim
            else None
        )
        # Derived constants, deliberately not module state: storing them would
        # put them in parameters() on this side and not in the torch state dict,
        # breaking the name-for-name parity both stacks are built on.
        self.bases = hash_bases(max(self.sizes))

    def hashes(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """``(indices, valid)`` per n-gram size, both ``[len(sizes), B, L]``."""
        batch, length = input_ids.shape
        ids = input_ids.astype(mx.int64)
        running = mx.zeros((batch, length), dtype=mx.int64)
        wanted = {size: slot for slot, size in enumerate(self.sizes)}
        indices: list[mx.array] = [None] * len(self.sizes)
        valid: list[mx.array] = [None] * len(self.sizes)
        positions = mx.arange(length)
        for lag in range(max(self.sizes)):
            if lag == 0:
                shifted = ids
            elif lag < length:
                pad = mx.full((batch, lag), HASH_PAD, dtype=mx.int64)
                shifted = mx.concatenate((pad, ids[:, : length - lag]), axis=1)
            else:
                shifted = mx.full((batch, length), HASH_PAD, dtype=mx.int64)
            running = mx.remainder(running + shifted * self.bases[lag], HASH_MODULUS)
            size = lag + 1
            if size in wanted:
                slot = wanted[size]
                indices[slot] = mx.remainder(running, self.vocab_size)
                valid[slot] = mx.broadcast_to(positions[None, :] >= size - 1, (batch, length))
        return mx.stack(indices), mx.stack(valid)

    def __call__(self, byte_embeddings: mx.array, input_ids: mx.array, attention_mask: mx.array | None = None):
        indices, valid = self.hashes(input_ids)
        if attention_mask is not None:
            # Padding is a suffix, so a position being real implies its whole
            # backward window is real; masking the position is enough.
            valid = valid & attention_mask.astype(mx.bool_)[None]
        offsets = (mx.arange(len(self.sizes), dtype=indices.dtype) * self.vocab_size)[:, None, None]
        gathered = self.embedding(indices + offsets)
        gathered = mx.where(valid[..., None], gathered, 0.0)
        total = mx.sum(gathered, axis=0).astype(byte_embeddings.dtype)
        if self.proj is not None:
            total = self.proj(total)
        # Eq. 3 is a plain sum. An earlier draft averaged over |sizes|+1, which
        # measured as dividing the byte embedding by 7 -- std 0.994 -> 0.204 --
        # because the projected n-gram term is much smaller than the byte term,
        # so the mean mostly just shrinks the latter.
        return byte_embeddings + total
