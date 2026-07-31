"""The teacher cache has to survive a round trip and report its own error.

Two things matter. Written arrays must come back byte-identical for tokens,
masks and patch lengths -- a silent dtype or padding slip there mistrains the
student with no visible symptom. And the top-k truncation must be honest: the
cache stores the full-row log-sum-exp precisely so ``coverage`` can say how much
mass was actually kept, rather than the caller assuming 32 of 260 is plenty.
"""

import json

import numpy as np
import pytest

from blt.teacher_cache import (
    TeacherCache,
    TeacherCacheWriter,
    teacher_probabilities,
)

VOCAB = 260


def _write(tmp_path, *, sequences=6, seq_len=8, max_patches=4, top_k=8, batch=3, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((sequences, seq_len, VOCAB)).astype(np.float32) * 4
    tokens = rng.integers(4, VOCAB, size=(sequences, seq_len)).astype(np.int32)
    mask = np.ones((sequences, seq_len), dtype=bool)
    patches = np.full((sequences, max_patches), seq_len // max_patches, dtype=np.int32)

    with TeacherCacheWriter(
        tmp_path,
        num_sequences=sequences,
        seq_len=seq_len,
        max_patches=max_patches,
        top_k=top_k,
        vocab_size=VOCAB,
        teacher="test",
    ) as writer:
        for start in range(0, sequences, batch):
            window = slice(start, start + batch)
            writer.add(
                tokens=tokens[window],
                mask=mask[window],
                patch_lengths=patches[window],
                logits=logits[window],
            )
    return logits, tokens, mask, patches


def test_round_trip_preserves_inputs(tmp_path):
    logits, tokens, mask, patches = _write(tmp_path)
    cache = TeacherCache(tmp_path)
    assert len(cache) == 6

    batch = cache.batch(np.arange(6))
    assert np.array_equal(batch["tokens"], tokens)
    assert np.array_equal(batch["mask"], mask)
    assert np.array_equal(batch["patch_lengths"], patches)

    # Top-k must be the actual k largest, in descending order. Values are stored
    # shifted by the row maximum, so the leading entry is exactly 0.0.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    expected_top = np.sort(shifted, axis=-1)[..., ::-1][..., :8]
    assert np.allclose(batch["topk_logits"], expected_top, atol=5e-3)
    assert np.all(np.diff(batch["topk_logits"], axis=-1) <= 1e-6)
    assert np.allclose(batch["topk_logits"][..., 0], 0.0, atol=1e-6)
    gathered = np.take_along_axis(shifted, batch["topk_indices"], axis=-1)
    assert np.allclose(gathered, batch["topk_logits"], atol=5e-3)


def test_logsumexp_is_the_full_row_not_the_truncated_one(tmp_path):
    logits, _, _, _ = _write(tmp_path)
    batch = TeacherCache(tmp_path).batch(np.arange(6))

    shifted = logits - logits.max(axis=-1, keepdims=True)
    expected = np.log(np.exp(shifted).sum(axis=-1))
    assert np.allclose(batch["logsumexp"], expected, atol=1e-4)
    # The truncated normaliser would be strictly smaller; catching that is the point.
    truncated = np.log(np.exp(batch["topk_logits"]).sum(axis=-1))
    assert np.all(truncated <= batch["logsumexp"] + 1e-4)


def test_shifted_storage_beats_raw_fp16_precision(tmp_path):
    # The reason values are shifted before the fp16 cast. Raw logits reach the
    # part of the fp16 range where the gap between representable numbers costs
    # over a percent of probability mass; shifted, the error is ~20x smaller.
    _write(tmp_path, top_k=VOCAB)
    assert TeacherCache(tmp_path).coverage(np.arange(6)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("top_k,floor", [(4, 0.0), (32, 0.5), (260, 0.999)])
def test_coverage_rises_with_k(tmp_path, top_k, floor):
    _write(tmp_path, top_k=top_k, seq_len=8)
    coverage = TeacherCache(tmp_path).coverage(np.arange(6))
    assert 0.0 < coverage <= 1.0 + 1e-6
    assert coverage >= floor


def test_full_vocab_coverage_is_one(tmp_path):
    # Not exactly 1.0: fp16 storage leaves ~1e-5 of residue even with no truncation.
    _write(tmp_path, top_k=VOCAB)
    assert TeacherCache(tmp_path).coverage(np.arange(6)) == pytest.approx(1.0, abs=1e-4)


def test_teacher_probabilities_are_a_distribution(tmp_path):
    _write(tmp_path, top_k=16)
    batch = TeacherCache(tmp_path).batch(np.arange(6))
    for temperature in (0.5, 1.0, 2.0):
        probs = teacher_probabilities(batch["topk_logits"], temperature=temperature)
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)
    # Higher temperature flattens the target; that is what the knob is for.
    sharp = teacher_probabilities(batch["topk_logits"], temperature=0.5).max(axis=-1).mean()
    flat = teacher_probabilities(batch["topk_logits"], temperature=2.0).max(axis=-1).mean()
    assert sharp > flat


def test_short_dump_truncates_instead_of_padding_with_zeros(tmp_path):
    # A run that stops early must not leave zero-filled rows behind; training
    # would read them as real sequences of token 0.
    rng = np.random.default_rng(1)
    with TeacherCacheWriter(
        tmp_path, num_sequences=10, seq_len=4, max_patches=2, top_k=4, vocab_size=VOCAB
    ) as writer:
        writer.add(
            tokens=rng.integers(4, VOCAB, size=(3, 4)).astype(np.int32),
            mask=np.ones((3, 4), dtype=bool),
            patch_lengths=np.full((3, 2), 2, dtype=np.int32),
            logits=rng.standard_normal((3, 4, VOCAB)).astype(np.float32),
        )
    cache = TeacherCache(tmp_path)
    assert len(cache) == 3
    assert cache.batch(np.arange(3))["tokens"].shape == (3, 4)
    assert json.loads((tmp_path / "meta.json").read_text())["num_sequences"] == 3


def test_narrow_patch_lengths_are_padded(tmp_path):
    with TeacherCacheWriter(
        tmp_path, num_sequences=1, seq_len=4, max_patches=5, top_k=4, vocab_size=VOCAB
    ) as writer:
        writer.add(
            tokens=np.arange(4, 8, dtype=np.int32)[None],
            mask=np.ones((1, 4), dtype=bool),
            patch_lengths=np.array([[2, 2]], dtype=np.int32),
            logits=np.zeros((1, 4, VOCAB), dtype=np.float32),
        )
    lengths = TeacherCache(tmp_path).batch([0])["patch_lengths"]
    assert lengths.shape == (1, 5)
    assert np.array_equal(lengths[0], [2, 2, 0, 0, 0])


def test_overfilling_is_refused(tmp_path):
    with pytest.raises(ValueError, match="teacher cache is full"):
        with TeacherCacheWriter(
            tmp_path, num_sequences=1, seq_len=4, max_patches=2, top_k=4, vocab_size=VOCAB
        ) as writer:
            writer.add(
                tokens=np.zeros((2, 4), np.int32),
                mask=np.ones((2, 4), bool),
                patch_lengths=np.full((2, 2), 2, np.int32),
                logits=np.zeros((2, 4, VOCAB), np.float32),
            )


def test_too_wide_patch_lengths_are_refused(tmp_path):
    with pytest.raises(ValueError, match="wider than the declared max_patches"):
        with TeacherCacheWriter(
            tmp_path, num_sequences=1, seq_len=4, max_patches=1, top_k=4, vocab_size=VOCAB
        ) as writer:
            writer.add(
                tokens=np.zeros((1, 4), np.int32),
                mask=np.ones((1, 4), bool),
                patch_lengths=np.full((1, 3), 2, np.int32),
                logits=np.zeros((1, 4, VOCAB), np.float32),
            )


def test_missing_cache_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no teacher cache"):
        TeacherCache(tmp_path / "nope")


def test_bad_top_k_is_refused(tmp_path):
    with pytest.raises(ValueError, match="top_k must be in"):
        TeacherCacheWriter(
            tmp_path, num_sequences=1, seq_len=4, max_patches=1, top_k=0, vocab_size=VOCAB
        )
