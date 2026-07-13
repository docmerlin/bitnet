"""Training-time loss terms for BitNet."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.rfmoe import (
    iter_rfmoe,
    rfmoe_aux_activity,
    rfmoe_diversity_loss,
    rfmoe_locality_loss,
)


def language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    z_loss_coef: float = 0.0,
) -> torch.Tensor:
    """Token cross-entropy plus an optional z-loss regularizer.

    The z-loss penalizes the squared log-partition ``logsumexp(logits)`` so the
    softmax normalizer stays near one. It keeps logits from drifting in
    low-precision ternary training and lets the optimizer run a higher learning
    rate without diverging. It is a training-time term only; evaluation reports
    plain cross-entropy so perplexity stays comparable.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if not torch.any(valid):
        return flat_logits.sum() * 0.0
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    if z_loss_coef > 0.0:
        log_z = torch.logsumexp(flat_logits.float(), dim=-1)
        loss = loss + z_loss_coef * log_z[valid].pow(2).mean()
    return loss


def multi_token_loss(
    mtp_logits: List[torch.Tensor],
    labels: torch.Tensor,
    segment_ids: Optional[torch.Tensor] = None,
    label_segment_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean cross-entropy over the extra multi-token-prediction heads.

    ``mtp_logits[i]`` predicts the token at offset ``i+2`` from each position.
    Since ``labels[t]`` is the token at ``t+1``, depth ``d = i+2`` targets
    ``labels[:, d-1:]`` aligned with the first ``S-(d-1)`` predictions; the tail
    positions have no target and are dropped (no ignore_index needed). Averaged
    across depths so the coefficient's scale is independent of ``mtp_depth``.

    Packed targets from another document are ignored.
    """
    total = None
    counted = 0
    for i, depth_logits in enumerate(mtp_logits):
        shift = i + 1                              # depth d=i+2 predicts labels[t + (i+1)]
        if shift >= depth_logits.size(1):          # no positions have a target this deep
            continue                               # else empty slice -> cross_entropy NaN
        pred = depth_logits[:, : depth_logits.size(1) - shift]   # (B, S-shift, V)
        target = labels[:, shift:]                               # (B, S-shift)
        if segment_ids is not None and label_segment_ids is not None:
            target = target.masked_fill(
                segment_ids[:, : target.size(1)].ne(label_segment_ids[:, shift:]),
                -100,
            )
        if not torch.any(target.ne(-100)):
            continue
        depth_loss = language_modeling_loss(pred, target)
        total = depth_loss if total is None else total + depth_loss
        counted += 1
    if counted == 0:
        if mtp_logits:
            return sum((head.sum() for head in mtp_logits), mtp_logits[0].new_zeros(())) * 0.0
        return labels.new_zeros((), dtype=torch.float32)
    return total / counted


def compute_train_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    mtp_logits: Optional[Sequence[torch.Tensor]] = None,
    segment_ids: Optional[torch.Tensor] = None,
    label_segment_ids: Optional[torch.Tensor] = None,
    z_loss_coef: float = 0.0,
    mtp_loss_coef: float = 0.0,
    density_lam: float = 0.0,
    locality_coef: float = 0.0,
    diversity_coef: float = 0.0,
    rfmoe_s: float = 1.0,
    rfmoe_alpha: float = 0.1,
) -> torch.Tensor:
    """Compose CE (+ z-loss), optional MTP, and independent RFMoE aux terms.

    RFMoE density, locality, and diversity are separate knobs: enabling locality
    or diversity no longer depends on the density controller object existing.
    When the model has no RFMoE layers the aux terms are skipped entirely.
    """
    if segment_ids is not None and label_segment_ids is not None:
        labels = labels.masked_fill(segment_ids.ne(label_segment_ids), -100)
    loss = language_modeling_loss(logits, labels, z_loss_coef=z_loss_coef)
    if mtp_logits:
        loss = loss + mtp_loss_coef * multi_token_loss(
            list(mtp_logits), labels, segment_ids, label_segment_ids
        )

    if not any(iter_rfmoe(model)):
        return loss

    # Density is always applied when RFMoE is present (controller owns ``lam``).
    loss = loss + density_lam * rfmoe_aux_activity(model)
    if locality_coef > 0.0:
        loss = loss + locality_coef * rfmoe_locality_loss(
            model, s=rfmoe_s, alpha=rfmoe_alpha
        )
    if diversity_coef > 0.0:
        loss = loss + diversity_coef * rfmoe_diversity_loss(model)
    return loss
