"""Helpers for FFN square-mid identity init (fresh start + soft resume)."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


FFN_MID_KEY_TOKENS = ("ffn_mid", "w_mid", "mid_proj")


def is_ffn_mid_key(key: str) -> bool:
    """True for the 3-stage FFN mid mats (dense / RFMoE / BLT)."""
    return any(token in key for token in FFN_MID_KEY_TOKENS)


def filter_ffn_mid_keys(keys: Iterable[str]) -> List[str]:
    return [k for k in keys if is_ffn_mid_key(k)]


@torch.no_grad()
def copy_square_identity_(weight: torch.Tensor) -> bool:
    """In-place set a square 2D master weight to ``I``. Returns whether applied."""
    if weight.ndim != 2 or weight.size(0) != weight.size(1):
        return False
    weight.copy_(torch.eye(weight.size(0), device=weight.device, dtype=weight.dtype))
    return True


@torch.no_grad()
def init_all_ffn_mid_identity(model: nn.Module) -> List[str]:
    """Set every square FFN mid master weight on ``model`` to identity.

    Used for cold starts so 3-mat FFN begins near the classic 2-mat path:
    ``silu(I @ h)`` is a mild pointwise nonlinearity on the expanded features
    (before ternary/Hadamard). Non-square mid tensors are left unchanged.
    """
    upgraded: List[str] = []
    for name, param in model.named_parameters():
        if not is_ffn_mid_key(name):
            continue
        if copy_square_identity_(param):
            upgraded.append(name)
    return upgraded


@torch.no_grad()
def init_missing_ffn_mid_identity(
    model: nn.Module,
    missing_keys: Sequence[str],
) -> List[str]:
    """Initialize missing square mid weights to identity (continuity-friendly).

    Random Kaiming mid + ternary quantization scrambles a warm-started FFN body.
    Identity master weights make mid ≈ pass-through before training adapts them:
    ``silu(I @ h)`` is a mild pointwise nonlinearity on the expanded features.

    Returns the parameter names that were upgraded (subset of ``missing_keys``).
    Non-square mid tensors (unexpected) are left unchanged.
    """
    mid_missing = set(filter_ffn_mid_keys(missing_keys))
    if not mid_missing:
        return []

    upgraded: List[str] = []
    # state_dict keys use the module path; parameters share those names.
    for name, param in model.named_parameters():
        if name not in mid_missing:
            continue
        if copy_square_identity_(param):
            upgraded.append(name)
    return upgraded
