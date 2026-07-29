"""Optional top-k block retrieval branch (A/B against local PaTH + Infini memory).

Reimplementation, not a restoration: the original top-k/branch_gates variant was
never committed. Only its hyperparameters survive, in runs/path-win-1024's
checkpoint (branch_gates was (num_heads, 3), topk_blocks=4, topk_block_size=64).
"""

import mlx.core as mx
import pytest
import torch

from config import TernaryConfig
from layers.infini_attention import InfiniAttention
from mlx_model import MLXBitNetConfig, MLXPaTHAttention


HIDDEN, HEADS, SEQ = 128, 4, 256


def _mlx_config(**over):
    base = dict(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        vocab_size=256,
        block_size=4,
        path_window_size=1024,
        use_topk_blocks=True,
        topk_blocks=2,
        topk_block_size=32,
    )
    base.update(over)
    return MLXBitNetConfig(**base)


def test_topk_selects_whole_past_blocks_only():
    attention = MLXPaTHAttention(_mlx_config())
    keys = mx.random.normal((1, HEADS, SEQ, attention.head_dim))
    values = mx.random.normal((1, HEADS, SEQ, attention.head_dim))
    queries = mx.random.normal((1, HEADS, 32, attention.head_dim))

    # Nothing to retrieve until a whole block has accumulated.
    assert attention._topk_context(queries, keys, values, 0) is None
    assert attention._topk_context(queries, keys, values, 31) is None

    retrieved = attention._topk_context(queries, keys, values, 64)
    mx.eval(retrieved)
    assert retrieved.shape == (1, HEADS, 32, attention.head_dim)


def test_topk_retrieval_ignores_the_future():
    # Rewriting tokens at/after `start` must not change what the branch returns:
    # only strictly-past whole blocks are eligible.
    attention = MLXPaTHAttention(_mlx_config())
    dim = attention.head_dim
    keys = mx.random.normal((1, HEADS, SEQ, dim))
    values = mx.random.normal((1, HEADS, SEQ, dim))
    queries = mx.random.normal((1, HEADS, 32, dim))
    start = 128

    before = attention._topk_context(queries, keys, values, start)
    tainted_k = mx.concatenate([keys[:, :, :start], mx.random.normal((1, HEADS, SEQ - start, dim))], axis=2)
    tainted_v = mx.concatenate([values[:, :, :start], mx.random.normal((1, HEADS, SEQ - start, dim))], axis=2)
    after = attention._topk_context(queries, tainted_k, tainted_v, start)
    mx.eval(before, after)
    assert mx.allclose(before, after).item()


def test_topk_branch_changes_the_output_and_is_off_by_default():
    x = mx.random.normal((1, SEQ, HIDDEN))
    segments = mx.zeros((1, SEQ), dtype=mx.int32)

    off = MLXPaTHAttention(_mlx_config(use_topk_blocks=False))
    on = MLXPaTHAttention(_mlx_config())
    on.update(off.parameters())  # identical weights; only the branch differs
    mx.eval(off.parameters(), on.parameters())

    baseline, retrieved = off(x, segments), on(x, segments)
    mx.eval(baseline, retrieved)
    assert not mx.allclose(baseline, retrieved, atol=1e-5).item()
    assert MLXBitNetConfig(hidden_size=HIDDEN, num_attention_heads=HEADS).use_topk_blocks is False


def test_topk_branch_is_trainable():
    attention = MLXPaTHAttention(_mlx_config())
    mx.eval(attention.parameters())
    x = mx.random.normal((1, SEQ, HIDDEN))

    def loss(params):
        attention.update(params)
        return attention(x, mx.zeros((1, SEQ), dtype=mx.int32)).square().mean()

    grads = mx.grad(loss)(attention.trainable_parameters())
    mx.eval(grads)
    assert float(mx.abs(grads["topk_gate"]).mean()) > 0.0


def test_decode_refuses_topk_rather_than_silently_dropping_it():
    attention = MLXPaTHAttention(_mlx_config())
    with pytest.raises(NotImplementedError, match="training-only"):
        attention.new_inference_cache(1)


def test_torch_topk_matches_mlx_selection():
    config = TernaryConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        vocab_size=256,
        block_size=4,
        path_window_size=1024,
        use_topk_blocks=True,
        topk_blocks=2,
        topk_block_size=32,
    )
    torch_attention = InfiniAttention(config)
    mlx_attention = MLXPaTHAttention(_mlx_config())
    dim = torch_attention.head_dim
    start = 128

    keys = torch.randn(1, HEADS, SEQ, dim)
    values = torch.randn(1, HEADS, SEQ, dim)
    queries = torch.randn(1, HEADS, 32, dim)
    expected = torch_attention._topk_context(queries, keys, values, start)

    to_mlx = lambda t: mx.array(t.detach().numpy())
    actual = mlx_attention._topk_context(
        to_mlx(queries), to_mlx(keys), to_mlx(values), start
    )
    mx.eval(actual)
    assert torch.allclose(
        expected, torch.tensor(actual.tolist()), atol=1e-4
    ), "backends disagree on retrieved context"
