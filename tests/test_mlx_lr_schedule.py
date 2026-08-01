"""Shape of the LR multiplier for both schedules."""

from __future__ import annotations

import pytest

from mlx_train import lr_multiplier


TOTAL, WARMUP, COOLDOWN = 1000, 100, 400


def _wsd(step: int) -> float:
    return lr_multiplier(step, TOTAL, WARMUP, COOLDOWN, 0.1, "wsd")


def _cosine(step: int) -> float:
    return lr_multiplier(step, TOTAL, WARMUP, COOLDOWN, 0.1, "cosine")


def test_wsd_warms_up_holds_then_decays_to_the_floor() -> None:
    assert _wsd(0) == pytest.approx(0.01)
    assert _wsd(WARMUP - 1) == pytest.approx(1.0)
    # Plateau: every step between warmup and the decay sits at the peak rate.
    assert all(_wsd(step) == pytest.approx(1.0) for step in range(WARMUP, TOTAL - COOLDOWN))
    # Halfway through the decay is halfway between the peak and the 0.1 floor.
    assert _wsd(TOTAL - COOLDOWN // 2) == pytest.approx(0.55)
    # The tail lands on the floor, not on zero -- the last steps still learn.
    assert _wsd(TOTAL) == pytest.approx(0.1)


def test_cosine_still_decays_from_the_first_post_warmup_step() -> None:
    # The distinguishing property: cosine starts falling immediately, WSD does not.
    assert _cosine(WARMUP + 1) < 1.0
    assert _wsd(WARMUP + 1) == pytest.approx(1.0)


def test_wsd_holds_the_floor_past_the_end() -> None:
    assert _wsd(TOTAL * 2) == pytest.approx(0.1)


def test_the_two_implementations_agree() -> None:
    # mlx_convert reconstructs a PyTorch run's LR from training.schedules and
    # refuses the checkpoint on a mismatch, so a schedule that exists on only one
    # side is a conversion failure waiting to happen. WSD was added to the MLX
    # copy first and this is what keeps the pair honest.
    from training.schedules import lr_schedule_multiplier

    for schedule in ("cosine", "wsd"):
        for step in range(0, TOTAL + 1, 7):
            mine = lr_multiplier(step, TOTAL, WARMUP, COOLDOWN, 0.1, schedule)
            shared = lr_schedule_multiplier(step, TOTAL, WARMUP, COOLDOWN, 0.1, schedule)
            assert abs(mine - shared) < 1e-9, (schedule, step, mine, shared)
