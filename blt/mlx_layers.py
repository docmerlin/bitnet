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

        # kaiming_uniform_(a=sqrt(5)) reduces to U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        bound = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(low=-bound, high=bound, shape=(out_features, in_features))

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        """Ramp quantisation strength. 0.0 is full precision, 1.0 fully quantised."""
        self.weight_mix = float(min(max(weight_mix, 0.0), 1.0))
        self.activation_mix = float(min(max(activation_mix, 0.0), 1.0))
        self.activation_bits = max(int(bits), 2)

    def _prepare_input(self, x: mx.array) -> mx.array:
        if self.use_hadamard:
            x = mx.hadamard_transform(x)
        if not self.quantize_activations or self.activation_bits < 2 or self.activation_mix <= 0.0:
            return x
        positive_levels = (2 ** (self.activation_bits - 1)) - 1
        negative_levels = 2 ** (self.activation_bits - 1)
        scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5) / max(positive_levels, 1)
        quantized = mx.clip(mx.round(x / scale), -negative_levels, positive_levels) * scale
        if self.activation_mix >= 1.0:
            return x + mx.stop_gradient(quantized - x)
        return x + self.activation_mix * mx.stop_gradient(quantized - x)

    def effective_weight(self) -> mx.array:
        weight = self.weight
        if self.weight_mix <= 0.0:
            return weight
        scale = mx.maximum(mx.mean(mx.abs(mx.stop_gradient(weight)), axis=-1, keepdims=True), 1e-5)
        normalized = weight / scale
        ternary = mx.where(normalized > 0.5, 1.0, mx.where(normalized < -0.5, -1.0, 0.0))
        if self.weight_mix >= 1.0:
            return weight + mx.stop_gradient(ternary * scale - weight)
        return weight + self.weight_mix * mx.stop_gradient(ternary * scale - weight)

    def __call__(self, x: mx.array) -> mx.array:
        return self._prepare_input(x) @ self.effective_weight().T


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

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

        def heads(t):
            return t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        cos, sin = build_rope_cache(seq_len, self.head_dim, theta=self.rope_theta)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        base_bias = (
            causal_window_attention_bias(seq_len, self.local_window, dtype=q.dtype) if self.causal else None
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
        self.mid_proj.weight = mx.eye(hidden_dim)
        self.down_proj = MLXHBitLinear(hidden_dim, dim, config=config)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = nn.silu(self.gate_proj(x)) * self.up_proj(x)
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
        k = heads(self.k_proj(normed_kv), kv_len)
        v = heads(self.v_proj(normed_kv), kv_len)

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
