"""Helpers for FFN square-mid identity init (fresh start + soft resume)."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


FFN_MID_KEY_TOKENS = ("ffn_mid", "w_mid", "mid_proj")

# The BLT local decoder's cross-attention had a one-hot mask by construction --
# every byte reads exactly the patch it belongs to -- so softmax over its single
# permitted key was a constant 1 and these three could not affect the output.
# They received exactly zero gradient. TernaryPatchGather replaced the module
# with the gather it always was; checkpoints written before that still carry
# these tensors and must be allowed to load past them.
RETIRED_DECODER_CROSS_ATTN_TOKENS = ("query_norm", "q_proj", "k_proj")


def is_ffn_mid_key(key: str) -> bool:
    """True for the 3-stage FFN mid mats (dense / RFMoE / BLT)."""
    return any(token in key for token in FFN_MID_KEY_TOKENS)


def filter_ffn_mid_keys(keys: Iterable[str]) -> List[str]:
    return [k for k in keys if is_ffn_mid_key(k)]


def is_retired_decoder_cross_attn_key(key: str) -> bool:
    """True for a query/key tensor of the retired BLT decoder cross-attention."""
    if "local_decoder.cross_attn_layers" not in key:
        return False
    return any(f".{token}." in key for token in RETIRED_DECODER_CROSS_ATTN_TOKENS)


def filter_retired_decoder_cross_attn_keys(keys: Iterable[str]) -> List[str]:
    return [k for k in keys if is_retired_decoder_cross_attn_key(k)]


@torch.no_grad()
def copy_square_identity_(weight: torch.Tensor) -> bool:
    """In-place set a square 2D master weight so it *quantises* to ``I``.

    ``eye(N) * N``, not ``eye(N)``. These are ternary weights: the per-output-
    channel scale is ``mean(|row|)``, which for a plain identity row (one 1 and
    N-1 zeros) is ``1/N``, so the quantised weight would be ``eye(N)/N`` -- a
    1/N attenuator rather than the pass-through this helper exists to create.
    Scaling by N makes ``mean(|row|) = 1``. Callers that own the module should
    also pin its weight mix; see ``layers.hybrid_block``.

    Returns whether applied.
    """
    if weight.ndim != 2 or weight.size(0) != weight.size(1):
        return False
    size = weight.size(0)
    weight.copy_(torch.eye(size, device=weight.device, dtype=weight.dtype) * size)
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
