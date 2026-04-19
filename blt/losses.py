"""Distillation losses for the ternary BLT student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from blt.model import TernaryBLTOutput
from blt.patching.teacher_patcher import patch_presence_mask


@dataclass(slots=True)
class DistillationLossWeights:
    hard_ce: float = 1.0
    logits_kl: float = 1.0
    encoder_patch_mse: float = 1.0
    global_patch_mse: float = 1.0
    decoder_hidden_mse: float = 0.5


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    total = (values * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return total / denom


def _masked_state_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((student - teacher).pow(2).mean(dim=-1), mask)


def compute_blt_distillation_loss(
    student: TernaryBLTOutput,
    teacher: TernaryBLTOutput | None,
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: DistillationLossWeights,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = student.logits.new_tensor(0.0)
    metrics: Dict[str, float] = {}

    if weights.hard_ce > 0.0:
        token_ce = F.cross_entropy(
            student.logits.reshape(-1, student.logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view_as(attention_mask)
        hard_ce = _masked_mean(token_ce, attention_mask)
        loss = loss + weights.hard_ce * hard_ce
        metrics["hard_ce"] = float(hard_ce.detach().item())

    if teacher is None:
        metrics["loss"] = float(loss.detach().item())
        return loss, metrics

    if weights.logits_kl > 0.0:
        student_log_probs = F.log_softmax(student.logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher.logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        logits_kl = _masked_mean(kl, attention_mask) * (temperature**2)
        loss = loss + weights.logits_kl * logits_kl
        metrics["logits_kl"] = float(logits_kl.detach().item())

    patch_mask = patch_presence_mask(teacher.patch_lengths)
    if weights.encoder_patch_mse > 0.0 and student.encoder_patches.shape == teacher.encoder_patches.shape:
        encoder_patch_mse = _masked_state_mse(student.encoder_patches, teacher.encoder_patches, patch_mask)
        loss = loss + weights.encoder_patch_mse * encoder_patch_mse
        metrics["encoder_patch_mse"] = float(encoder_patch_mse.detach().item())
    elif weights.encoder_patch_mse > 0.0:
        metrics["encoder_patch_mse_skipped"] = 1.0

    if weights.global_patch_mse > 0.0 and student.global_hidden.shape == teacher.global_hidden.shape:
        global_patch_mse = _masked_state_mse(student.global_hidden, teacher.global_hidden, patch_mask)
        loss = loss + weights.global_patch_mse * global_patch_mse
        metrics["global_patch_mse"] = float(global_patch_mse.detach().item())
    elif weights.global_patch_mse > 0.0:
        metrics["global_patch_mse_skipped"] = 1.0

    if weights.decoder_hidden_mse > 0.0 and student.decoder_hidden.shape == teacher.decoder_hidden.shape:
        decoder_hidden_mse = _masked_state_mse(student.decoder_hidden, teacher.decoder_hidden, attention_mask)
        loss = loss + weights.decoder_hidden_mse * decoder_hidden_mse
        metrics["decoder_hidden_mse"] = float(decoder_hidden_mse.detach().item())
    elif weights.decoder_hidden_mse > 0.0:
        metrics["decoder_hidden_mse_skipped"] = 1.0

    metrics["loss"] = float(loss.detach().item())
    return loss, metrics
