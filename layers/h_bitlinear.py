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


_HADAMARD_BASE_CACHE: dict[int, torch.Tensor] = {}
_HADAMARD_DEVICE_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


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


def get_hadamard_tensor(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a shared Hadamard tensor for the requested size/device/dtype."""
    device = torch.device(device)
    cache_key = (size, device.type, device.index, dtype)
    cached = _HADAMARD_DEVICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base = _HADAMARD_BASE_CACHE.get(size)
    if base is None:
        base = hadamard_matrix(size)
        _HADAMARD_BASE_CACHE[size] = base

    cached = base.to(device=device, dtype=dtype)
    _HADAMARD_DEVICE_CACHE[cache_key] = cached
    return cached


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


def quantize_activations(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Apply symmetric activation quantization with STE.

    Scaling is computed per token vector over the last dimension. ``bits`` can
    be increased during warmup to keep training slightly softer before the model
    transitions into full 4-bit activation quantization.
    """
    if bits < 2:
        return x

    positive_levels = (2 ** (bits - 1)) - 1
    negative_levels = 2 ** (bits - 1)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / max(positive_levels, 1)
    quantized = (x / scale).round().clamp(-negative_levels, positive_levels) * scale
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
        self.weight_quantization_mix = 1.0
        self.activation_quantization_mix = 1.0
        self.activation_bits = 4
        self.enable_weight_quantization = True
        self.enable_activation_quantization = True

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        use_hadamard = bool(getattr(self.config, "use_hadamard", True))
        if use_hadamard and in_features & (in_features - 1) == 0:
            self.hadamard_size = in_features
        else:
            self.hadamard_size = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the floating-point master weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard (on input), activation quantization, and ternary weight matmul.

        Following BitNet b1.58 best practices, Hadamard is applied to the input
        before the ternary weight multiplication for improved quantization stability.
        """
        if self.hadamard_size is not None:
            x = x @ get_hadamard_tensor(self.hadamard_size, x.device, x.dtype)

        if bool(getattr(self.config, "use_4bit_activations", True)) and self.enable_activation_quantization:
            quantized_x = quantize_activations(x, bits=self.activation_bits)
            if self.activation_quantization_mix >= 1.0:
                x = quantized_x
            elif self.activation_quantization_mix > 0.0:
                x = torch.lerp(x, quantized_x, self.activation_quantization_mix)

        if self.enable_weight_quantization:
            quantized_weight = ternary_quantize_ste(self.weight).to(dtype=x.dtype)
            if self.weight_quantization_mix >= 1.0:
                weight = quantized_weight
            elif self.weight_quantization_mix > 0.0:
                weight = torch.lerp(self.weight.to(dtype=x.dtype), quantized_weight, self.weight_quantization_mix)
            else:
                weight = self.weight.to(dtype=x.dtype)
        else:
            weight = self.weight.to(dtype=x.dtype)

        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def set_quantization_state(
        self,
        *,
        weight_mix: float | None = None,
        activation_mix: float | None = None,
        activation_bits: int | None = None,
        enable_weight_quantization: bool | None = None,
        enable_activation_quantization: bool | None = None,
    ) -> None:
        """Update runtime quantization settings for staged training."""
        if weight_mix is not None:
            self.weight_quantization_mix = float(min(max(weight_mix, 0.0), 1.0))
        if activation_mix is not None:
            self.activation_quantization_mix = float(min(max(activation_mix, 0.0), 1.0))
        if activation_bits is not None:
            self.activation_bits = max(int(activation_bits), 2)
        if enable_weight_quantization is not None:
            self.enable_weight_quantization = bool(enable_weight_quantization)
        if enable_activation_quantization is not None:
            self.enable_activation_quantization = bool(enable_activation_quantization)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"ternary=True, hadamard={self.hadamard_size is not None}"
        )


if __name__ == "__main__":
    layer = HBitLinear(1024, 1024)
    x = torch.randn(2, 128, 1024)
    y = layer(x)
    print(f"HBitLinear test passed. Output shape: {y.shape}")
