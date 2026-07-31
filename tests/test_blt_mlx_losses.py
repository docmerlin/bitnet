"""The MLX distillation loss must agree with the torch one where they overlap.

The torch loss takes full teacher logits; this one takes a cached top-k. With
``top_k == vocab_size`` the truncation vanishes and the two must agree to
floating-point noise -- that is the test that pins the temperature handling, the
T^2 factor and the masking. With a real top-k the KL is an approximation, so the
remaining tests assert the properties the approximation must still have.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch

from blt.losses import DistillationLossWeights, compute_blt_distillation_loss
from blt.mlx_losses import MLXDistillationLossWeights, blt_distillation_loss, truncated_kl
from blt.model import TernaryBLTOutput
from blt.teacher_cache import TeacherCache, TeacherCacheWriter

VOCAB, BATCH, SEQ = 64, 2, 12


def _fixtures(seed=0):
    rng = np.random.default_rng(seed)
    student = rng.standard_normal((BATCH, SEQ, VOCAB)).astype(np.float32) * 2
    teacher = rng.standard_normal((BATCH, SEQ, VOCAB)).astype(np.float32) * 2
    labels = rng.integers(0, VOCAB, size=(BATCH, SEQ)).astype(np.int64)
    mask = np.ones((BATCH, SEQ), dtype=bool)
    mask[1, 9:] = False
    return student, teacher, labels, mask


def _cache(tmp_path, teacher, top_k):
    with TeacherCacheWriter(
        tmp_path,
        num_sequences=BATCH,
        seq_len=SEQ,
        max_patches=3,
        top_k=top_k,
        vocab_size=VOCAB,
    ) as writer:
        writer.add(
            tokens=np.zeros((BATCH, SEQ), np.int32),
            mask=np.ones((BATCH, SEQ), bool),
            patch_lengths=np.full((BATCH, 3), 4, np.int32),
            logits=teacher,
        )
    return TeacherCache(tmp_path).batch(np.arange(BATCH))


def _torch_loss(student, teacher, labels, mask, temperature):
    def output(logits):
        zeros = torch.zeros(BATCH, 3)
        return TernaryBLTOutput(
            logits=torch.from_numpy(logits),
            patch_lengths=torch.full((BATCH, 3), 4),
            patch_ids=torch.zeros(BATCH, SEQ, dtype=torch.long),
            encoder_hidden=zeros,
            encoder_patches=zeros,
            global_hidden=zeros,
            decoder_hidden=zeros,
        )

    weights = DistillationLossWeights(
        hard_ce=1.0,
        logits_kl=1.0,
        encoder_patch_mse=0.0,
        global_patch_mse=0.0,
        decoder_hidden_mse=0.0,
    )
    loss, metrics = compute_blt_distillation_loss(
        output(student),
        output(teacher),
        labels=torch.from_numpy(labels),
        attention_mask=torch.from_numpy(mask),
        weights=weights,
        temperature=temperature,
    )
    return float(loss), metrics


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_matches_torch_when_nothing_is_truncated(tmp_path, temperature):
    student, teacher, labels, mask = _fixtures()
    batch = _cache(tmp_path, teacher, top_k=VOCAB)

    expected, expected_metrics = _torch_loss(student, teacher, labels, mask, temperature)
    actual, metrics = blt_distillation_loss(
        mx.array(student),
        labels=mx.array(labels.astype(np.int32)),
        attention_mask=mx.array(mask),
        weights=MLXDistillationLossWeights(),
        temperature=temperature,
        teacher_topk_indices=mx.array(batch["topk_indices"]),
        teacher_topk_logits=mx.array(batch["topk_logits"]),
    )
    assert float(actual) == pytest.approx(expected, rel=2e-3)
    assert float(metrics["hard_ce"]) == pytest.approx(expected_metrics["hard_ce"], rel=1e-4)
    assert float(metrics["logits_kl"]) == pytest.approx(expected_metrics["logits_kl"], rel=2e-3)


def _mean_kl(student, batch):
    return float(
        mx.mean(
            truncated_kl(
                mx.array(student),
                mx.array(batch["topk_indices"]),
                mx.array(batch["topk_logits"]),
                temperature=1.0,
            )
        )
    )


def _peaked_teacher(seed=0):
    """A byte LM's next-byte distribution is sharp, not uniform-ish."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((BATCH, SEQ, VOCAB)).astype(np.float32) * 1.5
    peak = rng.integers(0, VOCAB, size=(BATCH, SEQ))
    np.put_along_axis(logits, peak[..., None], 12.0, axis=-1)
    return logits


def test_truncation_stays_close_to_the_full_kl_on_a_peaked_teacher(tmp_path):
    student, _, _, _ = _fixtures()
    teacher = _peaked_teacher()
    full = _cache(tmp_path / "full", teacher, top_k=VOCAB)
    cut = _cache(tmp_path / "cut", teacher, top_k=16)

    assert TeacherCache(tmp_path / "cut").coverage(np.arange(BATCH)) > 0.99
    assert _mean_kl(student, cut) == pytest.approx(_mean_kl(student, full), rel=0.02)


def test_truncation_error_tracks_coverage(tmp_path):
    # The approximation is only as good as the mass retained. On a flat teacher,
    # top-16 of 64 keeps well under 99% and the KL is visibly wrong -- which is
    # exactly why the cache stores the full-row log-sum-exp and exposes
    # `coverage`. Watch that number, not the value of k.
    student, flat_teacher, _, _ = _fixtures()
    full = _cache(tmp_path / "full", flat_teacher, top_k=VOCAB)
    cut = _cache(tmp_path / "cut", flat_teacher, top_k=16)

    coverage = TeacherCache(tmp_path / "cut").coverage(np.arange(BATCH))
    assert coverage < 0.99
    error = abs(_mean_kl(student, cut) - _mean_kl(student, full)) / _mean_kl(student, full)
    assert error > 0.05


def test_kl_is_non_negative_and_zero_on_a_perfect_match(tmp_path):
    _, teacher, labels, mask = _fixtures()
    batch = _cache(tmp_path, teacher, top_k=VOCAB)
    divergence = truncated_kl(
        mx.array(teacher),  # student == teacher
        mx.array(batch["topk_indices"]),
        mx.array(batch["topk_logits"]),
        temperature=1.0,
    )
    assert float(mx.max(mx.abs(divergence))) < 2e-3
    other = truncated_kl(
        mx.array(_fixtures(seed=3)[0]),
        mx.array(batch["topk_indices"]),
        mx.array(batch["topk_logits"]),
        temperature=1.0,
    )
    assert float(mx.min(other)) > -1e-4


def test_mask_excludes_padded_positions(tmp_path):
    student, teacher, labels, mask = _fixtures()
    batch = _cache(tmp_path, teacher, top_k=VOCAB)
    kwargs = dict(
        labels=mx.array(labels.astype(np.int32)),
        weights=MLXDistillationLossWeights(),
        teacher_topk_indices=mx.array(batch["topk_indices"]),
        teacher_topk_logits=mx.array(batch["topk_logits"]),
    )
    masked, _ = blt_distillation_loss(mx.array(student), attention_mask=mx.array(mask), **kwargs)

    # Corrupting only masked positions must not move the loss.
    poisoned = student.copy()
    poisoned[1, 9:] = 50.0
    again, _ = blt_distillation_loss(mx.array(poisoned), attention_mask=mx.array(mask), **kwargs)
    assert float(masked) == pytest.approx(float(again), rel=1e-6)


def test_kl_without_a_teacher_is_refused():
    student, _, labels, mask = _fixtures()
    with pytest.raises(ValueError, match="logits_kl requires"):
        blt_distillation_loss(
            mx.array(student),
            labels=mx.array(labels.astype(np.int32)),
            attention_mask=mx.array(mask),
            weights=MLXDistillationLossWeights(logits_kl=1.0),
        )


def test_hard_ce_only_needs_no_teacher():
    student, _, labels, mask = _fixtures()
    loss, metrics = blt_distillation_loss(
        mx.array(student),
        labels=mx.array(labels.astype(np.int32)),
        attention_mask=mx.array(mask),
        weights=MLXDistillationLossWeights(logits_kl=0.0),
    )
    assert "logits_kl" not in metrics
    assert float(loss) == pytest.approx(float(metrics["hard_ce"]))


def test_non_positive_temperature_is_refused():
    student, _, labels, mask = _fixtures()
    with pytest.raises(ValueError, match="temperature must be positive"):
        blt_distillation_loss(
            mx.array(student),
            labels=mx.array(labels.astype(np.int32)),
            attention_mask=mx.array(mask),
            weights=MLXDistillationLossWeights(logits_kl=0.0),
            temperature=0.0,
        )
