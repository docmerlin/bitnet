"""Widening the Infini memory's key-feature dim is what changes its capacity.

M holds ``memory_dim * head_dim`` numbers per head, so ``infini_memory_expand`` is
the only knob that changes how much the memory can store. Default 0 keeps the
paper's head_dim so existing checkpoints and configs are untouched.
"""

import mlx.core as mx
import pytest
import torch

from config import TernaryConfig
from layers.infini_attention import InfiniAttention
from mlx_model import MLXBitNetConfig, MLXPaTHAttention


HIDDEN, HEADS, SEQ = 256, 8, 256
HEAD_DIM = HIDDEN // HEADS


def _mlx(expand):
    return MLXBitNetConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        vocab_size=256,
        block_size=4,
        path_window_size=1024,
        infini_memory_expand=expand,
    )


def test_default_is_unchanged_head_dim_memory():
    attention = MLXPaTHAttention(_mlx(0))
    assert attention.memory_dim == HEAD_DIM
    assert not hasattr(attention, "memory_proj")
    attention.reset_memory(2)
    assert attention.memory_m.shape == (2, HEADS, HEAD_DIM, HEAD_DIM)


@pytest.mark.parametrize("expand", [128, 512])
def test_expanded_memory_scales_the_bank(expand):
    attention = MLXPaTHAttention(_mlx(expand))
    attention.reset_memory(2)
    assert attention.memory_dim == expand
    assert attention.memory_m.shape == (2, HEADS, expand, HEAD_DIM)
    assert attention.memory_z.shape == (2, HEADS, expand)
    # Capacity is the point: M must hold expand/head_dim times as many numbers.
    assert attention.memory_m.size == 2 * HEADS * expand * HEAD_DIM


def test_expanded_memory_round_trips_and_stays_trainable():
    attention = MLXPaTHAttention(_mlx(256))
    mx.eval(attention.parameters())
    x = mx.random.normal((1, SEQ, HIDDEN))
    segments = mx.zeros((1, SEQ), dtype=mx.int32)
    out = attention(x, segments)
    mx.eval(out)
    assert out.shape == (1, SEQ, HIDDEN)

    def loss(params):
        attention.update(params)
        return attention(x, segments).square().mean()

    grads = mx.grad(loss)(attention.trainable_parameters())
    mx.eval(grads)
    assert float(mx.abs(grads["memory_proj"]["weight"]).mean()) > 0.0


def test_torch_memory_dim_matches_mlx():
    for expand, want in ((0, HEAD_DIM), (128, 128)):
        config = TernaryConfig(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            vocab_size=256,
            block_size=4,
            path_window_size=1024,
            infini_memory_expand=expand,
        )
        torch_attention = InfiniAttention(config)
        mlx_attention = MLXPaTHAttention(_mlx(expand))
        assert torch_attention.memory_dim == mlx_attention.memory_dim == want
        torch_attention._ensure_memory_batch(2)
        mlx_attention.reset_memory(2)
        assert tuple(torch_attention.memory_m.shape) == tuple(mlx_attention.memory_m.shape)
        assert tuple(torch_attention.memory_z.shape) == tuple(mlx_attention.memory_z.shape)
        assert tuple(torch_attention.memory_m.shape) == (2, HEADS, want, HEAD_DIM)


def test_negative_expand_is_rejected():
    with pytest.raises(ValueError, match="infini_memory_expand"):
        TernaryConfig(hidden_size=HIDDEN, num_attention_heads=HEADS, infini_memory_expand=-1)


def test_torch_expanded_memory_is_trainable():
    config = TernaryConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        vocab_size=256,
        block_size=4,
        path_window_size=1024,
        infini_memory_expand=128,
    )
    attention = InfiniAttention(config)
    keys = torch.randn(1, HEADS, 32, HEAD_DIM, requires_grad=True)
    values = torch.randn(1, HEADS, 32, HEAD_DIM)
    queries = torch.randn(1, HEADS, 32, HEAD_DIM)
    memory_m, memory_z = attention._update_memory_state(
        keys,
        values,
        torch.zeros(1, HEADS, 128, HEAD_DIM),
        torch.zeros(1, HEADS, 128),
    )
    assert memory_m.shape == (1, HEADS, 128, HEAD_DIM)
    attention._retrieve_memory(queries, memory_m, memory_z).pow(2).mean().backward()
    assert keys.grad.abs().mean() > 0.0
