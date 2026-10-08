"""Fourier σ embedding and AdaRMS (DiT-style conditioning on RMSNorm).

Affine is zero-initialized so a freshly built block is an identity residual
norm: ``x * (1 + 0) + 0``. Full-precision, never HBitLinear.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

# DiT timestep grid, as the official DiffusionBlocks code: frequencies fall from 1
# to 1/max_period. Applied to EDM's c_noise = ln(σ)/4 (|c_noise| <= 1.6 on
# [0.002, 80]) no feature wraps, so σ beyond a block's trained range (the sampler
# starts at σ_max, which training almost never draws) still reads as "very noisy".
# The old grid ran 1..max_period on ln σ: even its slowest feature wrapped every 2π,
# so σ=80 encoded like σ≈0.15 and block 0 trusted pure noise.
_FOURIER_MAX_PERIOD = 10_000.0


def fourier_frequencies(half: int, max_period: float = _FOURIER_MAX_PERIOD) -> mx.array:
    """DiT frequencies ``exp(-ln(max_period) · i / half)``, in ``(1/max_period, 1]``."""
    if half < 1:
        raise ValueError("half must be positive")
    scale = mx.arange(half).astype(mx.float32) / float(half)
    return mx.exp(-math.log(max_period) * scale)


def sigma_noise_input(sigma: mx.array) -> mx.array:
    """EDM ``c_noise = ln(σ) / 4``."""
    return 0.25 * mx.log(mx.maximum(sigma.astype(mx.float32), 1e-8))


class MLXSigmaEmbed(nn.Module):
    """``c_noise = ln(σ)/4`` Fourier features (DiT grid) → MLP → cond vector."""

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
        flat = mx.reshape(sigma_noise_input(sigma), (-1, 1))
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
