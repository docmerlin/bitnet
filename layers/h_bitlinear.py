"""H-BitLinear for ternary BitNet-style projections.

This layer keeps a floating-point master weight for training stability and
quantizes it to ternary values ``{-1, 0, 1}`` with a straight-through estimator
during the forward pass. Optional Hadamard preprocessing and 4-bit activation
quantization reduce compute while preserving a simple, fully-PyTorch
implementation that still runs on CPU-class hardware.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DefaultBitLinearConfig:
    use_hadamard = True
    use_4bit_activations = True


def hadamard_matrix(size: int) -> torch.Tensor:
    """Return a normalized Hadamard matrix for power-of-two ``size``."""
    if size < 1 or size & (size - 1) != 0:
        raise ValueError("Hadamard size must be a positive power of two")
    if size == 1:
        return torch.ones(1, 1)

    half = hadamard_matrix(size // 2)
    top = torch.cat((half, half), dim=1)
    bottom = torch.cat((half, -half), dim=1)
    return torch.cat((top, bottom), dim=0) / math.sqrt(2.0)


def ternary_quantize_ste(weight: torch.Tensor) -> torch.Tensor:
    """Quantize a floating-point weight tensor to ternary values with STE.

    A per-output-channel abs-mean scale preserves some dynamic range while the
    straight-through estimator lets gradients flow to the master weights.
    """
    scale = weight.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    normalized = weight / scale
    ternary = torch.where(
        normalized > 0.5,
        torch.ones_like(normalized),
        torch.where(normalized < -0.5, -torch.ones_like(normalized), torch.zeros_like(normalized)),
    )
    quantized = ternary * scale
    return weight + (quantized - weight).detach()


def quantize_activations_4bit(x: torch.Tensor) -> torch.Tensor:
    """Apply symmetric 4-bit activation quantization with STE.

    Scaling is computed per token vector over the last dimension, which is more
    stable than a single scale for the entire batch.
    """
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 7.0
    quantized = (x / scale).round().clamp(-8, 7) * scale
    return x + (quantized - x).detach()


class HBitLinear(nn.Module):
    """Hadamard-preconditioned ternary linear projection.

    The master weight stays in floating point so the module can be trained with
    standard optimizers. During the forward pass it is projected to ternary
    values using ``ternary_quantize_ste``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Any | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or _DefaultBitLinearConfig()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        use_hadamard = bool(getattr(self.config, "use_hadamard", True))
        if use_hadamard and in_features & (in_features - 1) == 0:
            self.register_buffer("hadamard", hadamard_matrix(in_features), persistent=False)
        else:
            self.register_buffer("hadamard", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the floating-point master weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optional Hadamard, 4-bit activations, and ternary matmul."""
        if self.hadamard is not None:
            x = x @ self.hadamard.to(dtype=x.dtype)

        if bool(getattr(self.config, "use_4bit_activations", True)):
            x = quantize_activations_4bit(x)

        quantized_weight = ternary_quantize_ste(self.weight).to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, quantized_weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"ternary=True, hadamard={self.hadamard is not None}"
        )


if __name__ == "__main__":
    layer = HBitLinear(1024, 1024)
    x = torch.randn(2, 128, 1024)
    y = layer(x)
    print(f"HBitLinear test passed. Output shape: {y.shape}")
