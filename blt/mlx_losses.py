"""Distillation loss for the MLX BLT student, against a cached teacher.

Mirrors :func:`blt.losses.compute_blt_distillation_loss` for the two terms a
cache can support -- hard cross-entropy and logit KL -- and deliberately does not
pretend to support the rest. The torch loss also has encoder/global/decoder
hidden-state MSE terms, but those are gated on the student and teacher tensors
having *identical shapes*, which at any realistic student width they do not:
against ``facebook/blt-1b`` all three self-skip. Storing 1B-wide hidden states to
disk to feed terms that then skip themselves is not a trade worth making, so the
cache holds logits only and :func:`blt_distillation_loss` refuses weights it
cannot honour rather than silently dropping them.

The KL is truncated: teacher probabilities come from the retained top-k,
renormalised. The student side stays a full-vocabulary log-softmax, so the
student's normaliser -- and therefore its gradient -- is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MLXDistillationLossWeights:
    hard_ce: float = 1.0
    logits_kl: float = 1.0


def masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    mask = mask.astype(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    return mx.sum(values * mask) / mx.maximum(mx.sum(mask), 1.0)


def truncated_kl(
    student_logits: mx.array,
    teacher_topk_indices: mx.array,
    teacher_topk_logits: mx.array,
    *,
    temperature: float,
) -> mx.array:
    """KL(teacher || student) over the teacher's retained top-k, per position.

    ``teacher_topk_logits`` arrive already shifted by their row maximum, which
    softmax is invariant to. The student's log-probabilities are computed over
    the whole vocabulary and only then gathered, so truncation never touches the
    student's normalising constant.
    """
    student_log_probs = nn.log_softmax(student_logits.astype(mx.float32) / temperature, axis=-1)
    gathered = mx.take_along_axis(student_log_probs, teacher_topk_indices, axis=-1)

    scaled = teacher_topk_logits.astype(mx.float32) / temperature
    teacher_log_probs = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)
    return mx.sum(teacher_probs * (teacher_log_probs - gathered), axis=-1)


def blt_distillation_loss(
    student_logits: mx.array,
    *,
    labels: mx.array,
    attention_mask: mx.array,
    weights: MLXDistillationLossWeights,
    temperature: float = 1.0,
    teacher_topk_indices: mx.array | None = None,
    teacher_topk_logits: mx.array | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Total loss plus its parts. Metrics stay as arrays so callers batch the sync."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    has_teacher = teacher_topk_indices is not None and teacher_topk_logits is not None
    if weights.logits_kl > 0.0 and not has_teacher:
        raise ValueError("logits_kl requires teacher_topk_indices and teacher_topk_logits")

    loss = mx.array(0.0)
    metrics: dict[str, mx.array] = {}
    # Projection precision must not reduce log-softmax or loss accumulation precision.
    student_logits = student_logits.astype(mx.float32)

    if weights.hard_ce > 0.0:
        token_ce = nn.losses.cross_entropy(student_logits, labels, reduction="none")
        hard_ce = masked_mean(token_ce, attention_mask)
        loss = loss + weights.hard_ce * hard_ce
        metrics["hard_ce"] = hard_ce

    if weights.logits_kl > 0.0:
        divergence = truncated_kl(
            student_logits, teacher_topk_indices, teacher_topk_logits, temperature=temperature
        )
        # The T^2 factor keeps soft-target gradients on the same scale as the
        # hard-CE ones as temperature moves; without it the KL term's influence
        # shrinks as 1/T^2 and the weight silently stops meaning what it says.
        logits_kl = masked_mean(divergence, attention_mask) * (temperature**2)
        loss = loss + weights.logits_kl * logits_kl
        metrics["logits_kl"] = logits_kl

    metrics["loss"] = loss
    return loss, metrics
