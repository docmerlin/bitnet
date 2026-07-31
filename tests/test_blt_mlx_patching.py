"""The MLX patch utilities must agree with the torch originals exactly.

They are a rewrite, not a transcription -- the per-row Python loops became cumsum
arithmetic and a membership matmul -- so equality against
:mod:`blt.patching.teacher_patcher` is the only thing keeping the rewrite honest.
Randomised patch lists cover the ragged cases (empty rows, trailing zero-length
patches, over- and under-long totals) that hand-written examples miss.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch

from blt import mlx_patching as mlx_p
from blt.patching import teacher_patcher as torch_p


def _pair(lengths):
    """Same patch-length table as a torch tensor and an MLX array."""
    array = np.asarray(lengths, dtype=np.int64)
    return torch.from_numpy(array), mx.array(array.astype(np.int32))


CASES = [
    [[4, 4, 4]],
    [[6, 1]],
    [[4, 4, 0]],  # trailing empty patch
    [[0, 0, 0]],  # wholly empty row
    [[3, 5, 2], [10, 0, 0], [1, 1, 1]],  # ragged batch
    [[7]],
]


@pytest.mark.parametrize("lengths", CASES)
@pytest.mark.parametrize("target", [0, 1, 5, 12, 20])
def test_normalize_patch_lengths_matches_torch(lengths, target):
    torch_lengths, mlx_lengths = _pair(lengths)
    expected = torch_p.normalize_patch_lengths(torch_lengths, target)
    actual = mlx_p.normalize_patch_lengths(mlx_lengths, target)
    assert np.array_equal(np.asarray(actual), expected.numpy().astype(np.int32))
    # The contract the callers rely on: rows sum to the target.
    assert np.all(np.asarray(mx.sum(actual, axis=-1)) == target)


@pytest.mark.parametrize("lengths", CASES)
def test_normalize_to_targets_matches_torch(lengths):
    torch_lengths, mlx_lengths = _pair(lengths)
    rows = torch_lengths.shape[0]
    for targets in ([0] * rows, [5] * rows, list(range(rows))):
        expected = torch_p.normalize_patch_lengths_to_targets(torch_lengths, torch.tensor(targets))
        actual = mlx_p.normalize_patch_lengths_to_targets(mlx_lengths, mx.array(np.asarray(targets, np.int32)))
        assert np.array_equal(np.asarray(actual), expected.numpy().astype(np.int32))


@pytest.mark.parametrize("lengths", CASES)
def test_patch_ids_and_masks_match_torch(lengths):
    torch_lengths, mlx_lengths = _pair(lengths)
    seq_len = int(torch_lengths.sum(dim=-1).max().item()) or 1

    torch_ids = torch_p.patch_ids_from_lengths(torch_lengths, seq_len)
    mlx_ids = mlx_p.patch_ids_from_lengths(mlx_lengths, seq_len)
    assert np.array_equal(np.asarray(mlx_ids), torch_ids.numpy().astype(np.int32))

    num_patches = torch_lengths.shape[1]
    for as_queries in (True, False):
        expected = torch_p.patch_membership_mask(torch_ids, num_patches, patches_as_queries=as_queries)
        actual = mlx_p.patch_membership_mask(mlx_ids, num_patches, patches_as_queries=as_queries)
        assert np.array_equal(np.asarray(actual), expected.numpy())

    assert np.array_equal(
        np.asarray(mlx_p.patch_presence_mask(mlx_lengths)),
        torch_p.patch_presence_mask(torch_lengths).numpy(),
    )


def test_uniform_patch_lengths_matches_torch():
    for seq_len in (1, 7, 8, 13):
        for patch_size in (1, 4, 6):
            expected = torch_p.build_uniform_patch_lengths(3, seq_len, patch_size, device=torch.device("cpu"))
            actual = mlx_p.build_uniform_patch_lengths(3, seq_len, patch_size)
            assert np.array_equal(np.asarray(actual), expected.numpy().astype(np.int32))


@pytest.mark.parametrize("pooling", ["mean", "sum"])
@pytest.mark.parametrize("lengths", CASES)
def test_pooling_matches_torch(pooling, lengths):
    # Pooling assumes normalised lengths -- the torch original indexes straight
    # into the patch axis and raises when a row sums to less than seq_len, which
    # is why every caller normalises first. Test the domain the callers use.
    torch_lengths, mlx_lengths = _pair(lengths)
    batch, num_patches = torch_lengths.shape
    seq_len = int(torch_lengths.sum(dim=-1).max().item()) or 1
    torch_lengths = torch_p.normalize_patch_lengths(torch_lengths, seq_len)
    mlx_lengths = mlx_p.normalize_patch_lengths(mlx_lengths, seq_len)

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((batch, seq_len, 8)).astype(np.float32)

    expected = torch_p.pool_patch_representations(
        torch.from_numpy(hidden), torch_lengths, pooling=pooling
    )
    actual = mlx_p.pool_patch_representations(mx.array(hidden), mlx_lengths, pooling=pooling)
    assert np.allclose(np.asarray(actual), expected.numpy(), atol=1e-5)
    assert actual.shape == (batch, num_patches, 8)


def test_pooling_honours_the_token_mask():
    torch_lengths, mlx_lengths = _pair([[4, 4]])
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((1, 8, 8)).astype(np.float32)
    mask = np.array([[True] * 6 + [False] * 2])

    expected = torch_p.pool_patch_representations(
        torch.from_numpy(hidden), torch_lengths, token_mask=torch.from_numpy(mask)
    )
    actual = mlx_p.pool_patch_representations(
        mx.array(hidden), mlx_lengths, token_mask=mx.array(mask)
    )
    assert np.allclose(np.asarray(actual), expected.numpy(), atol=1e-5)


def test_patch_start_mask_matches_torch():
    for lengths in ([[4, 4, 4]], [[6, 1]], [[3, 5, 2], [10, 0, 0]]):
        torch_lengths, mlx_lengths = _pair(lengths)
        seq_len = int(torch_lengths.sum(dim=-1).max().item())
        expected = torch_p.patch_start_mask_from_lengths(torch_lengths, seq_len)
        actual = mlx_p.patch_start_mask_from_lengths(mlx_lengths, seq_len)
        assert np.array_equal(np.asarray(actual), expected.numpy())


def test_normalize_rejects_a_negative_target():
    with pytest.raises(ValueError, match="target_length must be non-negative"):
        mlx_p.normalize_patch_lengths(mx.array([[4, 4]], dtype=mx.int32), -1)
