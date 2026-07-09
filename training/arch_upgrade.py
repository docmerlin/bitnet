"""Helpers for soft architecture upgrades at checkpoint resume."""

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
        if param.ndim != 2 or param.size(0) != param.size(1):
            continue
        param.copy_(torch.eye(param.size(0), device=param.device, dtype=param.dtype))
        upgraded.append(name)
    return upgraded
