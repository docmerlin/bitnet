"""Equi-probability schedule and equal unique-layer partitions."""

from __future__ import annotations

import math
import random

import pytest

from dblocks.schedule import NoiseSchedule, blocks_for_layer_width, layer_ranges


def test_layer_ranges_huginn_is_the_whole_stack() -> None:
    assert layer_ranges(48, 1) == [(0, 48)]


def test_layer_ranges_equal_width_remainder_on_last() -> None:
    assert layer_ranges(48, 4) == [(0, 12), (12, 24), (24, 36), (36, 48)]
    assert layer_ranges(50, 4) == [(0, 12), (12, 24), (24, 36), (36, 50)]
    assert layer_ranges(8, 3) == [(0, 2), (2, 4), (4, 8)]


def test_layer_ranges_rejects_too_many_blocks() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        layer_ranges(4, 5)


def test_blocks_for_layer_width_ceils() -> None:
    assert blocks_for_layer_width(48, 4) == 12
    assert blocks_for_layer_width(48, 12) == 4
    assert blocks_for_layer_width(10, 4) == 3


def test_probability_mass_is_equal() -> None:
    schedule = NoiseSchedule(num_blocks=4, overlap=0.0)
    masses = [schedule.probability_mass(b) for b in range(4)]
    assert pytest.approx(sum(masses), rel=1e-6) == 1.0
    for mass in masses:
        assert mass == pytest.approx(0.25, rel=1e-5)


def test_edges_run_from_sigma_max_to_sigma_min() -> None:
    schedule = NoiseSchedule(num_blocks=3, overlap=0.0)
    edges = schedule.sigma_edges
    assert edges[0] == pytest.approx(schedule.sigma_max, rel=1e-5)
    assert edges[-1] == pytest.approx(schedule.sigma_min, rel=1e-5)
    assert all(earlier > later for earlier, later in zip(edges, edges[1:]))


def test_overlap_expands_log_sigma_interval() -> None:
    tight = NoiseSchedule(num_blocks=3, overlap=0.0)
    wide = NoiseSchedule(num_blocks=3, overlap=0.1)
    lo_t, hi_t = tight.train_interval(1)
    lo_w, hi_w = wide.train_interval(1)
    assert lo_w < lo_t
    assert hi_w > hi_t
    # Appendix C: α = (σ_hi / σ_lo)^γ
    lo, hi = tight.interval(1)
    alpha = (hi / lo) ** 0.1
    assert lo_w == pytest.approx(max(lo / alpha, wide.sigma_min), rel=1e-6)
    assert hi_w == pytest.approx(min(hi * alpha, wide.sigma_max), rel=1e-6)


def test_sample_stays_in_train_interval() -> None:
    schedule = NoiseSchedule(num_blocks=4, overlap=0.1)
    rng = random.Random(0)
    for block_id in range(4):
        lo, hi = schedule.train_interval(block_id)
        for _ in range(32):
            sigma = schedule.sample(block_id, rng)
            assert lo <= sigma <= hi


def test_edm_weight_and_c_in() -> None:
    schedule = NoiseSchedule(sigma_data=0.5)
    sigma = 2.0
    assert schedule.weight(sigma) == pytest.approx(
        (sigma**2 + 0.25) / ((sigma * 0.5) ** 2)
    )
    assert schedule.c_in(sigma) == pytest.approx(1.0 / math.sqrt(sigma**2 + 0.25))


def test_euler_sigmas_are_decreasing_and_bracketed() -> None:
    schedule = NoiseSchedule(num_blocks=1)
    sigmas = schedule.euler_sigmas(4)
    assert len(sigmas) == 5
    assert sigmas[0] == pytest.approx(schedule.sigma_max, rel=1e-5)
    assert sigmas[-1] == pytest.approx(schedule.sigma_min, rel=1e-5)
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
