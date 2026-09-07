"""Chunked sliding-window attention must match the dense window mask exactly."""

from __future__ import annotations

import mlx.core as mx
import pytest

from blt.config import TernaryBLTConfig
from blt.mlx_layers import MLXTernarySelfAttention, windowed_block_attention_bias


def _attention(dim, heads, window):
    config = TernaryBLTConfig(
        local_dim=dim, global_dim=512, decoder_dim=dim, patch_size=4,
        n_heads_local_encoder=heads, n_heads_global=8,
        n_heads_local_decoder=heads, n_heads_cross=4,
    )
    mx.random.seed(0)
    layer = MLXTernarySelfAttention(dim, heads, config=config, local_window=window, causal=True)
    mx.eval(layer.parameters())
    return layer


def _dense(layer, x):
    chunkable = layer._chunkable
    layer._chunkable = lambda *args, **kwargs: False
    try:
        return layer(x)
    finally:
        layer._chunkable = chunkable


@pytest.mark.parametrize("seq,window,heads,dim", [(1024, 256, 4, 256), (512, 128, 4, 256), (256, 64, 8, 64)])
def test_chunked_matches_dense(seq, window, heads, dim) -> None:
    layer = _attention(dim, heads, window)
    x = mx.random.normal((3, seq, dim))
    assert layer._chunkable(seq, None)
    chunked, dense = layer(x), _dense(layer, x)
    mx.eval(chunked, dense)
    assert float(mx.max(mx.abs(chunked - dense))) < 2e-5


def test_first_block_cannot_see_the_zero_padded_previous_block() -> None:
    # Block 0 has no predecessor; if the shared band mask leaked, its early
    # queries would attend to zeros and the outputs would differ from dense.
    layer = _attention(64, 4, 32)
    x = mx.random.normal((2, 128, 64))
    chunked, dense = layer(x), _dense(layer, x)
    mx.eval(chunked, dense)
    assert float(mx.max(mx.abs(chunked[:, :32] - dense[:, :32]))) < 2e-5


def test_falls_back_when_the_window_does_not_tile_or_a_mask_is_given() -> None:
    layer = _attention(64, 4, 48)
    assert not layer._chunkable(128, None)          # 128 % 48 != 0
    full = _attention(64, 4, 128)
    assert not full._chunkable(128, None)           # window >= seq, dense is smaller
    windowed = _attention(64, 4, 32)
    assert not windowed._chunkable(128, mx.ones((2, 128), dtype=mx.bool_))


def test_band_mask_admits_exactly_window_keys_per_query() -> None:
    bias = windowed_block_attention_bias(8)
    assert bias.shape == (8, 16)
    assert (mx.sum((bias == 0.0).astype(mx.int32), axis=1) == 8).all().item()


def test_whole_model_unpadded_dispatch_and_gradients(monkeypatch):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from blt.mlx_model import MLXTernaryBLTModel

    config = TernaryBLTConfig(
        local_dim=32, global_dim=32, decoder_dim=32, local_window=8,
        n_layers_local_encoder=1, n_layers_global=1, n_layers_local_decoder=1,
        n_heads_local_encoder=4, n_heads_global=4, n_heads_local_decoder=4,
        n_heads_cross=4, patch_size=4,
    )
    model = MLXTernaryBLTModel(config)
    model.validate_inputs = False
    tokens = mx.random.randint(config.offset, config.offset + 256, (2, 32))
    mask = mx.ones(tokens.shape, dtype=mx.bool_)
    calls = []
    original = MLXTernarySelfAttention._windowed_attend

    def tracked(self, q, k, v):
        calls.append(q.shape[2])
        return original(self, q, k, v)

    monkeypatch.setattr(MLXTernarySelfAttention, "_windowed_attend", tracked)
    expected = model(tokens, attention_mask=mask).logits
    actual = model(tokens, attention_mask=mask, unpadded=True).logits
    mx.eval(expected, actual)
    assert calls == [32, 32]  # Both local stacks, through the real model entrypoint.
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
    def loss(m, trusted):
        return mx.mean(m(tokens, attention_mask=mask, unpadded=trusted).logits ** 2)
    _, dense_grads = nn.value_and_grad(model, lambda m: loss(m, False))(model)
    _, chunk_grads = nn.value_and_grad(model, lambda m: loss(m, True))(model)
    mx.eval(dense_grads, chunk_grads)
    for (name, dense), (_, chunk) in zip(tree_flatten(dense_grads), tree_flatten(chunk_grads)):
        assert mx.allclose(dense, chunk, atol=2e-5, rtol=2e-4).item(), name
    calls.clear()
    # Explicit suffix padding must stay dense.
    padded = mx.arange(32)[None, :] < mx.array([[20], [32]])
    out = model(tokens, attention_mask=padded)
    mx.eval(out.logits)
    assert not calls
    assert mx.all(out.decoder_hidden[0, 20:] == 0).item()


@pytest.mark.parametrize("window", [None, 4])
@pytest.mark.parametrize("new_len", [1, 3, 6])
def test_cached_multirow_extend_matches_dense(window, new_len):
    layer = _attention(32, 4, window)
    x = mx.random.normal((2, 12 + new_len, 32))
    _, cache = layer.prefill(x[:, :8])
    _, cache = layer.extend(x[:, 8:12], cache, offset=8)
    actual, _ = layer.extend(x[:, 12:], cache, offset=12)
    expected = _dense(layer, x)[:, 12:]
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
