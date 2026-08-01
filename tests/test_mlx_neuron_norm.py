"""MUD's neuron-axis imbalance on tall matrices, and the opt-in fix."""

from __future__ import annotations

import mlx.core as mx

from mlx_optim import CMUD, mud_decorrelate


def _row_cv(matrix: mx.array) -> float:
    norms = mx.linalg.norm(matrix, axis=1)
    return float(mx.std(norms) / (mx.mean(norms) + 1e-12))


def _gradient(rows: int, columns: int) -> mx.array:
    mx.random.seed(0)
    rank = min(rows, columns)
    spectrum = mx.array([1.0 / (1 + i) ** 0.7 for i in range(rank)])
    return (mx.random.normal((rows, rank)) * spectrum) @ mx.random.normal((rank, columns))


def test_tall_matrices_come_out_neuron_imbalanced_without_the_fix() -> None:
    # Whitening normalises rows of the orientation it worked in, and tall
    # matrices are transposed first -- so the neuron axis is the uneven one.
    tall = mud_decorrelate(_gradient(2048, 512), block_size=64)
    mx.eval(tall)
    assert _row_cv(tall) > 0.05


def test_wide_and_square_matrices_are_already_neuron_uniform() -> None:
    for rows, columns in ((512, 512), (512, 2048)):
        result = mud_decorrelate(_gradient(rows, columns), block_size=64)
        mx.eval(result)
        assert _row_cv(result) < 1e-5, (rows, columns)


def test_neuron_norm_flattens_the_tall_case_without_reversing_the_step() -> None:
    gradient = _gradient(2048, 512)
    plain = mud_decorrelate(gradient, block_size=64)
    normed = mud_decorrelate(gradient, block_size=64, neuron_norm=True)
    mx.eval(plain, normed)
    assert _row_cv(normed) < 1e-5
    # A rescale per neuron, not a different direction.
    cosine = float(mx.sum(plain * normed) / (mx.linalg.norm(plain) * mx.linalg.norm(normed)))
    assert cosine > 0.99


def test_flag_round_trips_and_defaults_off() -> None:
    assert CMUD(
        mud_learning_rate=1e-3, fallback_learning_rate=3e-4, weight_decay=0.0
    ).optimizers[0].neuron_norm is False
    config = CMUD(
        mud_learning_rate=1e-3, fallback_learning_rate=3e-4, weight_decay=0.0, neuron_norm=True
    ).checkpoint_config()
    assert config["neuron_norm"] is True
    assert CMUD(**config).optimizers[0].neuron_norm is True
