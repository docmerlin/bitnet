"""Utility helpers for the deep ternary BitNet model."""

from __future__ import annotations

from typing import Tuple

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by pairs for RoPE."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def build_rope_cache(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine caches for rotary embeddings.

    ``scaling_factor`` provides a simple NTK/YaRN-style position stretch.
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.float32) / scaling_factor
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to ``x``.

    ``x`` is expected to be shaped ``[batch, heads, seq, head_dim]``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def ternary_quantize(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dense weights to ternary values with abs-mean scaling."""
    scale = weights.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    ternary = torch.where(
        weights > 0.5 * scale,
        torch.ones_like(weights),
        torch.where(weights < -0.5 * scale, -torch.ones_like(weights), torch.zeros_like(weights)),
    )
    return ternary.to(torch.int8), scale.squeeze(-1)
