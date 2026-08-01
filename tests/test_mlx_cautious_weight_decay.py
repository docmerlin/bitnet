"""Cautious Weight Decay in MUD (modded-nanogpt records 43 and 50)."""

from __future__ import annotations

import mlx.core as mx

from mlx_optim import MUD, cautious_decay


def test_mask_allows_decay_only_where_the_step_already_shrinks_the_coordinate() -> None:
    # The parameter moves by -lr*update, so update and parameter sharing a sign
    # means the step is already pulling that coordinate toward zero.
    update = mx.array([[1.0, -1.0, 1.0, -1.0]])
    parameter = mx.array([[2.0, 2.0, -2.0, -2.0]])
    assert (cautious_decay(update, parameter) == mx.array([[1.0, 0.0, 0.0, 1.0]])).all().item()


def _decayed_magnitude(*, cautious: bool) -> float:
    optimizer = MUD(learning_rate=0.1, weight_decay=0.5, cautious_weight_decay=cautious)
    parameter = mx.ones((64, 64)) * 3.0
    state: dict = {}
    optimizer.init_single(parameter, state)
    updated = optimizer.apply_single(mx.zeros_like(parameter), parameter, state)
    mx.eval(updated)
    return float(mx.max(mx.abs(updated)))


def test_cautious_decay_leaves_a_parameter_with_no_gradient_alone() -> None:
    # Plain decay shrinks every coordinate every step regardless of the gradient,
    # which is the behaviour CWD exists to stop.
    assert _decayed_magnitude(cautious=True) == 3.0
    assert _decayed_magnitude(cautious=False) < 3.0


def test_flag_round_trips_through_the_checkpoint_config() -> None:
    from mlx_optim import CMUD

    optimizer = CMUD(
        mud_learning_rate=1e-3,
        fallback_learning_rate=3e-4,
        weight_decay=0.01,
        cautious_weight_decay=False,
    )
    assert optimizer.checkpoint_config()["cautious_weight_decay"] is False
    assert CMUD(**optimizer.checkpoint_config()).optimizers[0].cautious_weight_decay is False
