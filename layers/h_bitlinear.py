"""H-BitLinear for ternary BitNet-style projections.

This layer keeps a floating-point master weight for training stability and
quantizes it to ternary values ``{-1, 0, 1}`` with a straight-through estimator
during the forward pass. Optional Hadamard preprocessing is applied to inputs.
"""

from __future__ import annotations

import contextlib
import math
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_HADAMARD_BASE_CACHE: dict[int, torch.Tensor] = {}
_HADAMARD_DEVICE_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}
_EFFECTIVE_WEIGHT_CACHE: ContextVar[dict[tuple[int, torch.dtype], torch.Tensor] | None] = ContextVar(
    "effective_weight_cache", default=None
)
_GROUPED_WEIGHT_CACHE: ContextVar[dict[tuple[int, int, int, torch.dtype], torch.Tensor] | None] = ContextVar(
    "grouped_weight_cache", default=None
)


@contextlib.contextmanager
def reuse_effective_weights():
    """Reuse quantized weights within one recurrent forward scope."""
    token = _EFFECTIVE_WEIGHT_CACHE.set({})
    grouped = _GROUPED_WEIGHT_CACHE.set({})
    try:
        yield
    finally:
        _EFFECTIVE_WEIGHT_CACHE.reset(token)
        _GROUPED_WEIGHT_CACHE.reset(grouped)


def grouped_weight_cache() -> dict[tuple[int, int, int, torch.dtype], torch.Tensor] | None:
    """Forward-scoped grouped ternary stacks, or None outside a reuse scope."""
    return _GROUPED_WEIGHT_CACHE.get()


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
    scale = weight.detach().abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    normalized = weight / scale
    ternary = torch.where(
        normalized > 0.5,
        torch.ones_like(normalized),
        torch.where(normalized < -0.5, -torch.ones_like(normalized), torch.zeros_like(normalized)),
    )
    quantized = ternary * scale
    # quantized.detach() + (weight - weight.detach()), not weight + (quantized -
    # weight).detach(). Same value and same identity gradient, but the second
    # form computes quantized - weight, and the identity-initialised FFN mid has
    # weight ~ N against quantized ~ 1, so that subtraction loses most of its
    # significant digits.
    return quantized.detach() + (weight - weight.detach())


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
        *,
        config: Any,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.enable_weight_quantization = True

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        use_hadamard = bool(config.use_hadamard)
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

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this layer's Hadamard, if enabled."""
        if self.hadamard_size is not None:
            # Dense cached matmul, not a fast Walsh-Hadamard transform. At the
            # sizes this model uses (n = in_features = 1024, and 2048 for
            # ffn_down) a single GEMM against a cached n x n matrix beats the
            # O(n log n) FWHT by ~2.4-4.5x, because the FWHT's log2(n) sequential
            # butterfly stages are memory-bound and pay per-stage kernel-launch
            # overhead that a fused GEMM avoids. The GEMM also accumulates in
            # fp32 on tensor cores, and the cached matrix is pre-normalized to
            # 1/sqrt(n), so there is no fp16 overflow risk.
            #
            # Switch to a FWHT once n grows past ~4096: there the O(n^2) matmul
            # flops and the n x n matrix's memory (4096^2 fp32 = 64MB/dtype)
            # dominate and O(n log n) wins. Crossover is device-specific
            # (measured on MPS) -- re-benchmark on the training GPU before moving
            # the threshold. A FWHT implementation was removed from this module;
            # recover it from git history if the size regime changes.
            x = x @ get_hadamard_tensor(self.hadamard_size, x.device, x.dtype)
        return x

    def effective_weight(self, dtype: torch.dtype, weight: torch.Tensor | None = None) -> torch.Tensor:
        """Return ternary STE weight, including grouped leading dimensions."""
        cache = _EFFECTIVE_WEIGHT_CACHE.get()
        key = (id(self), dtype)
        if weight is None and cache is not None and key in cache:
            return cache[key]

        weight = self.weight if weight is None else weight
        if self.enable_weight_quantization:
            result = ternary_quantize_ste(weight).to(dtype=dtype)
        else:
            result = weight.to(dtype=dtype)
        if weight is self.weight and cache is not None:
            cache[key] = result
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard (on input) and ternary weight matmul."""
        return self.forward_prepared(self.prepare_input(x))

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this projection to input already prepared by an equivalent layer."""
        weight = self.effective_weight(x.dtype)

        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def set_quantization_state(self, *, enable_weight_quantization: bool | None = None) -> None:
        """Update runtime weight-quantization settings."""
        if enable_weight_quantization is not None:
            self.enable_weight_quantization = bool(enable_weight_quantization)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"ternary=True, hadamard={self.hadamard_size is not None}"
        )
