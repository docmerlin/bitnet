"""Regression tests for BLT patch utilities."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt.patching.teacher_patcher import (
    build_uniform_patch_lengths,
    normalize_patch_lengths,
    patch_ids_from_lengths,
    patch_membership_mask,
    patch_start_mask_from_lengths,
)
from blt.patching.student_entropy import StudentEntropyModel
from blt.config import TernaryBLTConfig


def test_patch_length_normalization_and_ids() -> bool:
    patch_lengths = torch.tensor([[3, 3, 2], [4, 4, 0]], dtype=torch.long)
    normalized = normalize_patch_lengths(patch_lengths, target_length=7)
    assert normalized.tolist() == [[3, 3, 1], [4, 3, 0]], normalized.tolist()

    trailing_trim = normalize_patch_lengths(torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long), target_length=4)
    assert trailing_trim.tolist() == [[1, 1, 1, 1, 0]], trailing_trim.tolist()

    patch_ids = patch_ids_from_lengths(normalized, seq_len=7)
    assert patch_ids.tolist() == [[0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 1, 1]], patch_ids.tolist()

    query_mask = patch_membership_mask(patch_ids, num_patches=3, patches_as_queries=True)
    key_mask = patch_membership_mask(patch_ids, num_patches=3, patches_as_queries=False)
    assert query_mask.shape == (2, 3, 7)
    assert key_mask.shape == (2, 7, 3)
    assert query_mask[0, 2].sum().item() == 1
    assert key_mask[1, :, 2].sum().item() == 0

    uniform = build_uniform_patch_lengths(2, 10, 4, device=torch.device("cpu"))
    assert uniform.tolist() == [[4, 4, 2], [4, 4, 2]], uniform.tolist()

    start_mask = patch_start_mask_from_lengths(torch.tensor([[6, 1, 0]], dtype=torch.long), seq_len=7)
    assert start_mask.tolist() == [[True, False, False, False, False, False, True]], start_mask.tolist()

    patcher = StudentEntropyModel(
        TernaryBLTConfig(max_patch_length=3),
        dim=32,
        num_layers=1,
        num_heads=4,
    )
    boundary_logits = torch.full((1, 10), -1.0)
    boundary_logits[:, 0] = 1.0
    capped_patch_lengths = patcher.predict_patch_lengths_from_logits(boundary_logits)
    assert capped_patch_lengths.tolist() == [[3, 3, 3, 1]], capped_patch_lengths.tolist()

    print("BLT patching utility tests passed")
    return True


if __name__ == "__main__":
    test_patch_length_normalization_and_ids()
