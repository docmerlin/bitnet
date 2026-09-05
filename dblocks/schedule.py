"""Equi-probability VE noise schedule for DiffusionBlocks.

Noise levels follow EDM (Karras et al. 2022): log σ ~ N(P_mean, P_std²),
truncated to [σ_min, σ_max]. Block edges split that truncated mass into B
equal pieces. Overlap expands each interval in log-σ space (paper γ).

Unique-layer partitions are **equal width**, not prelude|recurrent|coda.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


def _phi(z: float) -> float:
    return _NORMAL.cdf(z)


def _ndtri(q: float) -> float:
    q = min(max(float(q), 1e-7), 1.0 - 1e-7)
    return _NORMAL.inv_cdf(q)


def layer_ranges(unique_layers: int, num_blocks: int) -> list[tuple[int, int]]:
    """Inclusive-exclusive unique-layer slices, equal width.

    Remainder layers go on the last block. ``num_blocks == 1`` is the whole
    unique stack (Huginn / Stage 1).
    """
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    unique = int(unique_layers)
    if unique < 1:
        raise ValueError("need at least one unique layer")
    if num_blocks > unique:
        raise ValueError("num_blocks cannot exceed unique layer count")
    width = unique // num_blocks
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(num_blocks):
        end = unique if index == num_blocks - 1 else start + width
        ranges.append((start, end))
        start = end
    return ranges


def blocks_for_layer_width(unique_layers: int, layers_per_block: int) -> int:
    """``B = ceil(L / N)`` for the optional thin-block ablation."""
    if layers_per_block < 1:
        raise ValueError("layers_per_block must be positive")
    unique = int(unique_layers)
    if unique < 1:
        raise ValueError("need at least one unique layer")
    return (unique + layers_per_block - 1) // layers_per_block


@dataclass(frozen=True)
class NoiseSchedule:
    """VE / EDM schedule plus equi-probability block edges.

    ``sigma_edges[0] == sigma_max`` (noisiest) down to
    ``sigma_edges[B] == sigma_min``. Block ``b`` (0-indexed) owns
    ``[sigma_edges[b+1], sigma_edges[b]]``.
    """

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2
    overlap: float = 0.1
    num_blocks: int = 1

    def __post_init__(self) -> None:
        if self.sigma_min <= 0 or self.sigma_max <= self.sigma_min:
            raise ValueError("need 0 < sigma_min < sigma_max")
        if self.sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        if self.p_std <= 0:
            raise ValueError("p_std must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be positive")

    def _q_of_log_sigma(self, log_sigma: float) -> float:
        return _phi((log_sigma - self.p_mean) / self.p_std)

    def _sigma_of_q(self, q: float) -> float:
        return math.exp(self.p_mean + self.p_std * _ndtri(q))

    @property
    def q_min(self) -> float:
        return self._q_of_log_sigma(math.log(self.sigma_min))

    @property
    def q_max(self) -> float:
        return self._q_of_log_sigma(math.log(self.sigma_max))

    @property
    def sigma_edges(self) -> tuple[float, ...]:
        # b=0 → q_max → σ_max; b=B → q_min → σ_min.
        span = self.q_max - self.q_min
        edges = []
        for b in range(self.num_blocks + 1):
            q = self.q_max - (b / self.num_blocks) * span
            edges.append(self._sigma_of_q(q))
        return tuple(edges)

    def interval(self, block_id: int) -> tuple[float, float]:
        """Non-overlapped [σ_lo, σ_hi] for block ``block_id`` (σ_lo < σ_hi)."""
        if not 0 <= block_id < self.num_blocks:
            raise ValueError("block_id out of range")
        edges = self.sigma_edges
        hi = edges[block_id]
        lo = edges[block_id + 1]
        return float(lo), float(hi)

    def train_interval(self, block_id: int) -> tuple[float, float]:
        """Overlapped training interval. γ=0 recovers ``interval``."""
        lo, hi = self.interval(block_id)
        if self.overlap == 0.0 or lo <= 0.0:
            return lo, hi
        alpha = (hi / lo) ** self.overlap
        return max(lo / alpha, self.sigma_min), min(hi * alpha, self.sigma_max)

    def sample(self, block_id: int = 0, rng: random.Random | None = None) -> float:
        """Draw σ from the truncated log-normal on block ``block_id``'s train interval."""
        rng = rng or random
        lo, hi = self.train_interval(block_id)
        q_lo = self._q_of_log_sigma(math.log(lo))
        q_hi = self._q_of_log_sigma(math.log(hi))
        q = rng.uniform(min(q_lo, q_hi), max(q_lo, q_hi))
        return float(min(max(self._sigma_of_q(q), lo), hi))

    def weight(self, sigma: float) -> float:
        """EDM loss weight w(σ) = (σ² + σ_data²) / (σ · σ_data)²."""
        sigma = float(sigma)
        data = float(self.sigma_data)
        return (sigma * sigma + data * data) / ((sigma * data) ** 2)

    def c_in(self, sigma: float) -> float:
        """EDM input scale 1 / sqrt(σ² + σ_data²)."""
        sigma = float(sigma)
        data = float(self.sigma_data)
        return 1.0 / math.sqrt(sigma * sigma + data * data)

    def probability_mass(self, block_id: int) -> float:
        """Non-overlapped mass as a fraction of the truncated [σ_min, σ_max] mass."""
        lo, hi = self.interval(block_id)
        mass = abs(self._q_of_log_sigma(math.log(hi)) - self._q_of_log_sigma(math.log(lo)))
        total = abs(self.q_max - self.q_min)
        return mass / total if total else 0.0

    def euler_sigmas(self, steps: int) -> tuple[float, ...]:
        """Decreasing σ sequence of length ``steps + 1`` from σ_max to σ_min.

        Stage 1 uses ``steps = K`` Euler evaluations of the shared stack.
        Stage 2 uses ``steps = B`` (one block per step).
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        span = self.q_max - self.q_min
        return tuple(
            self._sigma_of_q(self.q_max - (i / steps) * span) for i in range(steps + 1)
        )
