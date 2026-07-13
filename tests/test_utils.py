"""Regression tests for shared utility helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from utils import (
    atomic_torch_save,
    build_rope_cache,
    clear_rope_cache,
    load_checkpoint_payload,
    seed_everything,
    validate_suffix_padded_mask,
)


def test_rope_cache_reuses_tensors() -> bool:
    clear_rope_cache()
    first_cos, first_sin = build_rope_cache(seq_len=16, dim=8, device=torch.device("cpu"))
    second_cos, second_sin = build_rope_cache(seq_len=16, dim=8, device=torch.device("cpu"))
    assert first_cos.data_ptr() == second_cos.data_ptr()
    assert first_sin.data_ptr() == second_sin.data_ptr()
    clear_rope_cache()
    print("RoPE cache reuse tests passed")
    return True


def test_seed_everything_is_idempotent_for_torch() -> bool:
    seed_everything(123)
    first = torch.rand(4)
    seed_everything(123)
    second = torch.rand(4)
    assert torch.allclose(first, second)
    print("seed_everything tests passed")
    return True


def test_load_checkpoint_payload_roundtrip() -> bool:
    with TemporaryDirectory() as tempdir:
        checkpoint_path = Path(tempdir) / "payload.pt"
        payload = {"step": 3, "metrics": {"loss": 1.25}, "tensor": torch.ones(2, 2)}
        torch.save(payload, checkpoint_path)
        loaded = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        assert loaded["step"] == 3
        assert loaded["metrics"]["loss"] == 1.25
        assert torch.allclose(loaded["tensor"], torch.ones(2, 2))
    print("load_checkpoint_payload tests passed")
    return True


def test_atomic_torch_save_preserves_destination_on_failure(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"valid")

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    try:
        atomic_torch_save({"step": 1}, checkpoint_path)
    except OSError:
        pass
    else:
        raise AssertionError("failed checkpoint save should propagate")
    assert checkpoint_path.read_bytes() == b"valid"


def test_validate_suffix_padded_mask_rejects_interior_padding() -> bool:
    valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    validate_suffix_padded_mask(valid)
    invalid = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool)
    try:
        validate_suffix_padded_mask(invalid)
    except ValueError as exc:
        assert "suffix-padded" in str(exc)
    else:
        raise AssertionError("Expected suffix-padding validation to fail")
    print("suffix-padding validation tests passed")
    return True


if __name__ == "__main__":
    test_rope_cache_reuses_tensors()
    test_seed_everything_is_idempotent_for_torch()
    test_load_checkpoint_payload_roundtrip()
    test_validate_suffix_padded_mask_rejects_interior_padding()
