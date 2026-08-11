"""Pure wall-clock / momentum schedule helpers (no MLX required)."""

from __future__ import annotations

from training.token_progress import (
    estimate_total_steps,
    momentum_warmup,
    resolve_ramp_bounds,
    scheduled_int,
    scheduled_value,
    snap_sequence_length,
    wall_clock_shapes,
)


def test_scheduled_value_endpoints_and_mid() -> None:
    assert scheduled_value(0.0, 10.0, 0.0, 1.0) == 0.0
    assert scheduled_value(0.0, 10.0, 1.0, 1.0) == 10.0
    assert scheduled_value(0.0, 10.0, 0.5, 1.0) == 5.0
    assert scheduled_value(0.0, 10.0, 0.0, 0.0) == 10.0  # ratio 0 → jump to end


def test_scheduled_int_never_below_one() -> None:
    assert scheduled_int(1, 4, 0.0, 1.0) == 1
    assert scheduled_int(1, 4, 1.0, 1.0) == 4
    assert scheduled_int(0, 0, 0.5, 1.0) == 1


def test_snap_sequence_nearest_not_floor() -> None:
    # Floor-only would stick at 64 for most of 64→128; nearest crosses mid-way.
    assert snap_sequence_length(64, 64) == 64
    assert snap_sequence_length(96, 64) == 128
    assert snap_sequence_length(95, 64) == 64
    assert snap_sequence_length(97, 64) == 128
    assert snap_sequence_length(128, 64) == 128
    assert snap_sequence_length(1, 64) == 64  # floor at window


def test_resolve_ramp_bounds_none_means_fixed() -> None:
    assert resolve_ramp_bounds(None, 4) == (4, 4)
    assert resolve_ramp_bounds(1, 4) == (1, 4)


def test_wall_clock_shapes_batch_ramp() -> None:
    early = wall_clock_shapes(
        0.0,
        initial_batch=1,
        final_batch=4,
        batch_growth_ratio=1.0,
        initial_seq=128,
        final_seq=128,
        seq_growth_ratio=1.0,
        path_window=64,
    )
    late = wall_clock_shapes(
        1.0,
        initial_batch=1,
        final_batch=4,
        batch_growth_ratio=1.0,
        initial_seq=128,
        final_seq=128,
        seq_growth_ratio=1.0,
        path_window=64,
    )
    assert early.batch == 1 and early.seq == 128
    assert late.batch == 4 and late.seq == 128
    assert early.tokens_per_step(2) == 256
    assert late.tokens_per_step(2) == 1024


def test_estimate_total_steps_matches_live_progress_definition() -> None:
    """Replay estimate loop: progress = tokens_before / total (same as mlx_train)."""
    total = 10_000
    initial_batch, final_batch = 1, 4
    initial_seq, final_seq = 128, 128
    path_window = 64
    accum = 2
    steps = estimate_total_steps(
        total,
        initial_batch=initial_batch,
        final_batch=final_batch,
        batch_growth_ratio=1.0,
        initial_seq=initial_seq,
        final_seq=final_seq,
        seq_growth_ratio=1.0,
        path_window=path_window,
        grad_accumulation_steps=accum,
    )
    tokens = 0
    for _ in range(steps):
        progress = tokens / total
        shapes = wall_clock_shapes(
            progress,
            initial_batch=initial_batch,
            final_batch=final_batch,
            batch_growth_ratio=1.0,
            initial_seq=initial_seq,
            final_seq=final_seq,
            seq_growth_ratio=1.0,
            path_window=path_window,
        )
        tokens += shapes.tokens_per_step(accum)
    assert tokens >= total
    # One fewer step would undershoot.
    tokens2 = 0
    for _ in range(steps - 1):
        progress = tokens2 / total
        shapes = wall_clock_shapes(
            progress,
            initial_batch=initial_batch,
            final_batch=final_batch,
            batch_growth_ratio=1.0,
            initial_seq=initial_seq,
            final_seq=final_seq,
            seq_growth_ratio=1.0,
            path_window=path_window,
        )
        tokens2 += shapes.tokens_per_step(accum)
    assert tokens2 < total


def test_estimate_fixed_batch_matches_ceil() -> None:
    import math

    total = 100_000
    batch, seq, accum = 4, 128, 1
    per_step = batch * seq * accum
    expected = math.ceil(total / per_step)
    got = estimate_total_steps(
        total,
        initial_batch=batch,
        final_batch=batch,
        batch_growth_ratio=1.0,
        initial_seq=seq,
        final_seq=seq,
        seq_growth_ratio=1.0,
        path_window=64,
        grad_accumulation_steps=accum,
    )
    assert got == expected


def test_momentum_warmup_r9() -> None:
    assert momentum_warmup(0, 10, start=0.85, peak=0.95) == 0.85 + 0.1 * 0.1
    assert abs(momentum_warmup(9, 10, start=0.85, peak=0.95) - 0.95) < 1e-12
    assert momentum_warmup(10, 10, start=0.85, peak=0.95) == 0.95
    assert momentum_warmup(0, 0, start=0.85, peak=0.95) == 0.95  # no warmup
    assert momentum_warmup(3, 10, start=0.95, peak=0.95) == 0.95  # disabled
