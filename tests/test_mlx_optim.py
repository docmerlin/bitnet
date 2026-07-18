"""Parity checks for MLX C-MUD/C-Lion."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_optim import CMUD, CLion, MUD, cautious_mask, dequantize_blockwise, mud_decorrelate, quantize_blockwise
from optim import cautious_mask as torch_cautious_mask
from optim import mud_decorrelate as torch_mud_decorrelate


def test_mlx_mud_and_cautious_mask_match_pytorch() -> None:
    values = np.random.default_rng(4).normal(size=(12, 8)).astype(np.float32)
    mlx_update = mud_decorrelate(mx.array(values), passes=2)
    torch_update = torch_mud_decorrelate(torch.from_numpy(values), passes=2)
    mx.eval(mlx_update)
    assert np.allclose(np.array(mlx_update), torch_update.numpy(), rtol=2e-4, atol=2e-4)

    gradient = np.random.default_rng(5).normal(size=(12, 8)).astype(np.float32)
    mlx_masked = cautious_mask(mx.array(values), mx.array(gradient))
    torch_masked = torch_cautious_mask(torch.from_numpy(values), torch.from_numpy(gradient))
    mx.eval(mlx_masked)
    assert np.allclose(np.array(mlx_masked), torch_masked.numpy(), rtol=1e-6, atol=1e-6)


def test_mlx_mud_blocking_matches_independent_blocks() -> None:
    update = mx.random.normal((8, 12))
    expected = mx.concatenate(
        [mud_decorrelate(update[start : start + 4], passes=2) for start in range(0, 8, 4)]
    )
    actual = mud_decorrelate(update, passes=2, block_size=4)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def test_mlx_clion_block_quantization_and_routing() -> None:
    values = mx.random.normal((5000,))
    quantized, scale = quantize_blockwise(values)
    restored = dequantize_blockwise(quantized, scale, values.shape)
    mx.eval(restored)
    relative_error = mx.mean(mx.abs(restored - values)) / mx.mean(mx.abs(values))
    assert relative_error.item() < 0.02

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(512, 8)
            self.projection = nn.Linear(8, 8)

        def __call__(self, tokens):
            return self.projection(self.embedding(tokens))

    model = Model()
    optimizer = CMUD(
        mud_learning_rate=0.05,
        fallback_learning_rate=0.02,
        weight_decay=0.0,
        block_size=4,
    )
    optimizer.init(model.trainable_parameters())
    mud_keys = {key for key, _ in tree_flatten(optimizer.state["states"][0])}
    lion_state = dict(tree_flatten(optimizer.state["states"][1]))
    assert "projection.weight.momentum_buffer" in mud_keys
    assert "embedding.weight.exp_avg_q" in lion_state
    assert lion_state["embedding.weight.exp_avg_q"].dtype == mx.int8
    assert optimizer.checkpoint_config()["block_size"] == 4
    assert not optimizer.checkpoint_config()["mud_eight_bit"]


def test_mlx_mud_block_quantization() -> None:
    parameter = mx.ones((64, 64), dtype=mx.bfloat16)
    gradient = mx.random.normal(parameter.shape).astype(mx.bfloat16)
    mud = MUD(learning_rate=0.1, weight_decay=0.0, eight_bit=True)
    state = {}
    mud.init_single(parameter, state)

    updated = mud.apply_single(gradient, parameter, state)
    restored = dequantize_blockwise(
        state["momentum_buffer_q"],
        state["momentum_buffer_scale"],
        parameter.shape,
    )
    mx.eval(updated, restored)

    assert state["momentum_buffer_q"].dtype == mx.int8
    assert "momentum_buffer" not in state
    relative_error = mx.mean(mx.abs(restored - gradient.astype(mx.float32))) / mx.mean(mx.abs(gradient))
    assert relative_error.item() < 0.02


def test_mlx_cmud_reduces_loss() -> None:
    model = nn.Linear(8, 8)
    optimizer = CMUD(
        mud_learning_rate=0.05,
        fallback_learning_rate=0.02,
        weight_decay=0.0,
        eight_bit=False,
    )
    inputs = mx.random.normal((8, 8))
    targets = mx.random.normal((8, 8))

    def loss_fn():
        return mx.mean(mx.square(model(inputs) - targets))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    first = None
    for _ in range(30):
        loss, gradients = loss_and_grad()
        optimizer.update(model, gradients)
        mx.eval(loss, model.state, optimizer.state)
        first = loss.item() if first is None else first
    assert loss.item() < first


def test_mlx_optimizer_state_stays_float32_for_bfloat16_parameters() -> None:
    parameter = mx.ones((64, 64), dtype=mx.bfloat16)
    gradient = mx.ones_like(parameter)

    mud = MUD(learning_rate=0.1, weight_decay=0.1)
    mud_state = {}
    mud.init_single(parameter, mud_state)
    updated = mud.apply_single(gradient, parameter, mud_state)

    lion = CLion(learning_rate=0.1, eight_bit=True)
    lion_state = {}
    lion.init_single(parameter, lion_state)
    lion.apply_single(gradient, parameter, lion_state)
    mx.eval(updated, mud_state, lion_state)

    assert mud_state["momentum_buffer"].dtype == mx.float32
    assert mud_state["master_parameter"].dtype == mx.float32
    assert lion_state["exp_avg_scale"].dtype == mx.float32
    assert lion_state["master_parameter"].dtype == mx.float32
    assert mx.all(updated < parameter).item()

    small_step_lion = CLion(learning_rate=1e-3, eight_bit=False)
    small_state = {}
    small_step_lion.init_single(parameter, small_state)
    small_update = parameter
    for _ in range(8):
        small_update = small_step_lion.apply_single(gradient, small_update, small_state)
    mx.eval(small_update)
    assert mx.all(small_update < parameter).item()
