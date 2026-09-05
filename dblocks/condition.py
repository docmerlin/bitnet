"""Fourier σ embedding and AdaRMS (DiT-style conditioning on RMSNorm).

Affine is zero-initialized so a freshly built block is an identity residual
norm: ``x * (1 + 0) + 0``. Full-precision, never HBitLinear.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

# Transformer / DiT geometric grid. Raw 2**k hits 2^31 at default fourier_dim=64
# and float32 sin/cos stop tracking log σ (~[-6, 4]) past ~2^15.
_FOURIER_MAX_PERIOD = 10_000.0


def fourier_frequencies(half: int, max_period: float = _FOURIER_MAX_PERIOD) -> mx.array:
    """Geometric frequencies in ``[1, max_period]``. ``half`` is fourier_dim/2."""
    if half < 1:
        raise ValueError("half must be positive")
    if half == 1:
        return mx.array([1.0], dtype=mx.float32)
    scale = mx.arange(half).astype(mx.float32) / float(half - 1)
    return mx.exp(math.log(max_period) * scale)


class MLXSigmaEmbed(nn.Module):
    """``log σ`` Fourier features → MLP → cond vector."""

    def __init__(self, fourier_dim: int, cond_dim: int):
        super().__init__()
        if fourier_dim < 2 or fourier_dim % 2:
            raise ValueError("fourier_dim must be a positive even integer")
        if cond_dim < 1:
            raise ValueError("cond_dim must be positive")
        self.fourier_dim = int(fourier_dim)
        self.cond_dim = int(cond_dim)
        self.in_proj = nn.Linear(self.fourier_dim, self.cond_dim, bias=True)
        self.out_proj = nn.Linear(self.cond_dim, self.cond_dim, bias=True)

    def __call__(self, sigma: mx.array) -> mx.array:
        log_sigma = mx.log(mx.maximum(sigma.astype(mx.float32), 1e-8))
        flat = mx.reshape(log_sigma, (-1, 1))
        freqs = fourier_frequencies(self.fourier_dim // 2)
        angle = flat * freqs.reshape((1, -1))
        features = mx.concatenate([mx.sin(angle), mx.cos(angle)], axis=-1)
        hidden = nn.silu(self.in_proj(features))
        cond = self.out_proj(hidden)
        if sigma.ndim == 0:
            return cond[0]
        return cond


class MLXAdaRMS(nn.Module):
    """``x ← x * (1 + scale(cond)) + shift(cond)``. Zero-init → identity."""

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        if min(hidden_size, cond_dim) < 1:
            raise ValueError("hidden_size and cond_dim must be positive")
        self.proj = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.proj.bias = mx.zeros_like(self.proj.bias)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        style = self.proj(cond.astype(x.dtype))
        scale, shift = mx.split(style, 2, axis=-1)
        # Insert length-1 axes in front of H so (H,) → (1, 1, H) and (B, H) → (B, 1, H).
        while scale.ndim < x.ndim:
            scale = mx.expand_dims(scale, axis=-2)
            shift = mx.expand_dims(shift, axis=-2)
        return x * (1.0 + scale) + shift


def apply_ada(norm_out: mx.array, ada: MLXAdaRMS | None, cond: mx.array | None) -> mx.array:
    """Identity when either half of the conditioner is missing."""
    if ada is None or cond is None:
        return norm_out
    return ada(norm_out, cond)
