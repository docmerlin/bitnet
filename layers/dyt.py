"""Dynamic Tanh (DyT) — drop-in replacement for RMSNorm / LayerNorm.

Paper: Zhu et al., "Transformers without Normalization" (CVPR 2025),
arXiv:2503.10622.

    DyT(x) = γ ⊙ tanh(α x) + β

``α`` is a learnable scalar (init 0.5); ``γ`` and ``β`` are per-channel affine
parameters (ones / zeros), matching the usual RMSNorm affine interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """Elementwise DyT over the last dimension."""

    def __init__(self, normalized_shape: int, alpha_init: float = 0.5):
        super().__init__()
        if normalized_shape < 1:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = int(normalized_shape)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x) * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, alpha_init={float(self.alpha.data):.4f}"


def make_norm(
    dim: int,
    *,
    norm_type: str = "rms",
    eps: float = 1e-5,
    alpha_init: float = 0.5,
) -> nn.Module:
    """Build RMSNorm or DyT from a config-style string."""
    kind = str(norm_type).lower()
    if kind in {"rms", "rmsnorm", "rms_norm"}:
        return nn.RMSNorm(dim, eps=eps)
    if kind in {"dyt", "dynamic_tanh", "dynamictanh"}:
        return DynamicTanh(dim, alpha_init=alpha_init)
    raise ValueError(f"unknown norm_type {norm_type!r}; expected 'rms' or 'dyt'")
