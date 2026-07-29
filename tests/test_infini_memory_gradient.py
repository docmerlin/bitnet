"""Infini memory must be trainable, and the PaTH window knob must not lie."""

import mlx.core as mx
import pytest
import torch

from config import TernaryConfig, effective_path_window
from layers.infini_attention import InfiniAttention
from mlx_model import MLXBitNetConfig, MLXPaTHAttention


BATCH, HEADS, CHUNK, HEAD_DIM = 2, 4, 32, 32


def test_effective_path_window_is_capped_by_block_size():
    # The regression that made the path-window ablation measure nothing: at seq
    # 1024 with 16 blocks, windows of 64 and 1024 produce the same 64-wide chunk.
    narrow, wide = (
        effective_path_window(path_window_size=size, block_size=16, sequence_length=1024)
        for size in (64, 1024)
    )
    assert narrow == wide == 64
    # It does bind once the block count leaves room for it.
    assert effective_path_window(path_window_size=128, block_size=4, sequence_length=1024) == 128


@pytest.mark.parametrize("delta_rule", [True, False])
def test_mlx_memory_write_path_is_differentiable(delta_rule):
    config = MLXBitNetConfig(
        hidden_size=128,
        num_attention_heads=HEADS,
        vocab_size=256,
        block_size=4,
        path_window_size=1024,
        infini_delta_rule=delta_rule,
    )
    attention = MLXPaTHAttention(config)
    keys = mx.random.normal((BATCH, HEADS, CHUNK, HEAD_DIM))
    values = mx.random.normal((BATCH, HEADS, CHUNK, HEAD_DIM))
    queries = mx.random.normal((BATCH, HEADS, CHUNK, HEAD_DIM))
    rows = mx.ones((BATCH,), dtype=mx.bool_)

    def loss(keys, values):
        memory_m, memory_z, _ = attention._next_memory(
            keys,
            values,
            rows,
            mx.zeros((BATCH, HEADS, HEAD_DIM, HEAD_DIM)),
            mx.zeros((BATCH, HEADS, HEAD_DIM)),
            mx.zeros((BATCH,), dtype=mx.bool_),
        )
        return attention._retrieve_memory(queries, memory_m, memory_z).square().mean()

    key_grad, value_grad = mx.grad(loss, argnums=(0, 1))(keys, values)
    mx.eval(key_grad, value_grad)
    # Detaching the carry here leaves only sigma(Q) trainable, which turns the
    # memory into a fixed content-free blur.
    assert float(mx.abs(key_grad).mean()) > 0.0
    assert float(mx.abs(value_grad).mean()) > 0.0


def test_torch_memory_write_path_matches_mlx():
    config = TernaryConfig(
        hidden_size=128,
        num_attention_heads=HEADS,
        vocab_size=256,
        block_size=4,
        path_window_size=1024,
    )
    attention = InfiniAttention(config)
    keys = torch.randn(BATCH, HEADS, CHUNK, HEAD_DIM, requires_grad=True)
    values = torch.randn(BATCH, HEADS, CHUNK, HEAD_DIM, requires_grad=True)
    queries = torch.randn(BATCH, HEADS, CHUNK, HEAD_DIM)

    memory_m, memory_z = attention._update_memory_state(
        keys,
        values,
        torch.zeros(BATCH, HEADS, HEAD_DIM, HEAD_DIM),
        torch.zeros(BATCH, HEADS, HEAD_DIM),
    )
    attention._retrieve_memory(queries, memory_m, memory_z).pow(2).mean().backward()

    assert keys.grad.abs().mean() > 0.0
    assert values.grad.abs().mean() > 0.0
