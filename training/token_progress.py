"""Torch-free token-progress ramps (shared by MLX and PyTorch trainers)."""

from __future__ import annotations

from dataclasses import dataclass


def scheduled_value(start: float, end: float, progress: float, ratio: float) -> float:
    """Linear ramp from ``start``→``end`` over the first ``ratio`` of token progress.

    ``ratio <= 0`` jumps to ``end`` immediately. ``progress`` is tokens_done / budget
    in ``[0, 1+]`` (values above 1 clamp to the end of the ramp).
    """
    if ratio <= 0:
        return end
    fraction = min(max(progress / ratio, 0.0), 1.0)
    return start + fraction * (end - start)


def scheduled_int(start: int, end: int, progress: float, ratio: float) -> int:
    """Integer ramp; never below 1 (batch / positive counts)."""
    return max(1, int(round(scheduled_value(float(start), float(end), progress, ratio))))


def snap_sequence_length(target: int, path_window: int) -> int:
    """Snap to a positive multiple of ``path_window`` (packing / PaTH constraint).

    Uses nearest multiple so a 64→128 ramp with window 64 actually spends time
    at both lengths (floor-only would stay at 64 until the final step).
    """
    window = max(int(path_window), 1)
    length = max(int(target), window)
    return max(1, int(round(length / window))) * window


def resolve_ramp_bounds(initial: int | None, final: int) -> tuple[int, int]:
    """``initial is None`` → fixed at ``final`` (no ramp)."""
    end = int(final)
    start = end if initial is None else int(initial)
    return start, end


@dataclass(frozen=True)
class WallClockShapes:
    """Micro-batch and sequence length at a given token progress."""

    batch: int
    seq: int

    def tokens_per_step(self, grad_accumulation_steps: int) -> int:
        return self.batch * self.seq * max(int(grad_accumulation_steps), 1)


def wall_clock_shapes(
    progress: float,
    *,
    initial_batch: int,
    final_batch: int,
    batch_growth_ratio: float,
    initial_seq: int,
    final_seq: int,
    seq_growth_ratio: float,
    path_window: int,
) -> WallClockShapes:
    """Batch/seq at ``progress`` (R46 / R72 wall-clock curricula)."""
    batch = scheduled_int(initial_batch, final_batch, progress, batch_growth_ratio)
    seq = snap_sequence_length(
        scheduled_int(initial_seq, final_seq, progress, seq_growth_ratio),
        path_window,
    )
    return WallClockShapes(batch=batch, seq=seq)


def estimate_total_steps(
    total_tokens: int,
    *,
    initial_batch: int,
    final_batch: int,
    batch_growth_ratio: float,
    initial_seq: int,
    final_seq: int,
    seq_growth_ratio: float,
    path_window: int,
    grad_accumulation_steps: int,
) -> int:
    """Simulate the batch/seq curriculum to size the LR schedule for a token budget.

    Progress for step *k* is tokens accumulated *before* that step — same as the
    live MLX train loop — so LR total_steps matches actual token delivery.
    """
    total = max(int(total_tokens), 1)
    tokens = 0
    steps = 0
    # Step cap: worst case 1 token/step would need ``total`` steps; allow headroom.
    limit = total + 10
    while tokens < total and steps < limit:
        progress = tokens / total
        shapes = wall_clock_shapes(
            progress,
            initial_batch=initial_batch,
            final_batch=final_batch,
            batch_growth_ratio=batch_growth_ratio,
            initial_seq=initial_seq,
            final_seq=final_seq,
            seq_growth_ratio=seq_growth_ratio,
            path_window=path_window,
        )
        tokens += shapes.tokens_per_step(grad_accumulation_steps)
        steps += 1
    return max(steps, 1)


def momentum_warmup(
    step: int,
    warmup_steps: int,
    *,
    start: float,
    peak: float,
) -> float:
    """Linear momentum ramp start→peak over LR warmup (NanoGPT-speedrun R9).

    ``step`` is 0-based optimizer step index. Holds ``peak`` after warmup, or
    when warmup is disabled / start equals peak.
    """
    if warmup_steps <= 0 or abs(start - peak) < 1e-12:
        return peak
    if step < warmup_steps:
        return start + (peak - start) * float(step + 1) / float(warmup_steps)
    return peak
