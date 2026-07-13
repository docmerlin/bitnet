"""Checkpoint save/load for BitNet training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from config import TernaryConfig
from model import BitNetDeep
from training.arch_upgrade import (
    filter_ffn_mid_keys,
    init_missing_ffn_mid_identity,
    is_ffn_mid_key,
)
from training.memory import reset_infini_memory
from utils import atomic_torch_save, load_checkpoint_payload

# Re-export for callers / BLT that shared the predicate.
__all__ = [
    "TrainerState",
    "filter_ffn_mid_keys",
    "is_ffn_mid_key",
    "load_checkpoint",
    "save_checkpoint",
]


@dataclass
class TrainerState:
    step: int = 0
    tokens_processed: int = 0
    samples_processed: int = 0
    best_val_loss: float = float("inf")


def save_checkpoint(
    output_dir: Path,
    model: BitNetDeep,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[Any],
    state: TrainerState,
    model_config: TernaryConfig,
    args: Any,
    checkpoint_name: str,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "trainer_state": asdict(state),
        "model_config": asdict(model_config),
        "args": vars(args),
    }
    atomic_torch_save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: BitNetDeep,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[Any],
) -> TrainerState:
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    # Tolerate architecture evolution (e.g. qk-norm or FFN mid layers added after a
    # checkpoint was written): load non-strict and surface any mismatch.
    incompatible = model.load_state_dict(payload["model"], strict=False)
    mid_missing = filter_ffn_mid_keys(incompatible.missing_keys)
    unsupported_missing = [key for key in incompatible.missing_keys if key not in mid_missing]
    if unsupported_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint model does not match current architecture: "
            f"missing keys: {unsupported_missing}; unexpected keys: {incompatible.unexpected_keys}"
        )
    if incompatible.missing_keys:
        print(
            f"Warning: resumed checkpoint did not match the model exactly. "
            f"Missing keys: {incompatible.missing_keys}; "
            f"unexpected keys: {incompatible.unexpected_keys}",
            flush=True,
        )

    architecture_bump = bool(mid_missing)
    if architecture_bump:
        upgraded = init_missing_ffn_mid_identity(model, incompatible.missing_keys)
        print(
            "Architecture bump: checkpoint predates 3-stage FFN mid layers. "
            f"Applied identity init to {len(upgraded)} mid weight(s) "
            f"({len(mid_missing)} mid keys were missing). "
            "Optimizer, LR scheduler, and grad scaler will NOT be restored "
            "(param groups / step state would be inconsistent). "
            "Trainer step/tokens ARE restored so quant and block-growth schedules "
            "continue from prior progress — LR re-warms from a fresh LambdaLR. "
            "Prefer a same-architecture checkpoint for a true warm resume.",
            flush=True,
        )

    reset_infini_memory(model)

    if architecture_bump:
        # Fresh optimizer/scheduler for new mid params; keep TrainerState for token schedules.
        return TrainerState(**payload["trainer_state"])

    try:
        optimizer.load_state_dict(payload["optimizer"])
    except (ValueError, RuntimeError) as exc:
        print(
            f"Warning: optimizer state not restored ({exc}). "
            "Continuing with a freshly initialized optimizer. "
            "LR scheduler will also be left at its post-construction state.",
            flush=True,
        )
        return TrainerState(**payload["trainer_state"])

    try:
        scheduler.load_state_dict(payload["scheduler"])
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"Warning: scheduler state not restored ({exc}).", flush=True)
    if scaler is not None and payload.get("scaler") is not None:
        try:
            scaler.load_state_dict(payload["scaler"])
        except (ValueError, RuntimeError, KeyError) as exc:
            print(f"Warning: grad scaler state not restored ({exc}).", flush=True)
    return TrainerState(**payload["trainer_state"])
