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
