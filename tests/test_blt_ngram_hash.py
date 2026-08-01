"""Hashed byte n-gram embeddings: hash agreement, causality, wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from blt.config import TernaryBLTConfig
from blt.layers.ngram import HashNgramEmbedding
from blt.mlx_layers import MLXHashNgramEmbedding
from blt.mlx_model import MLXTernaryBLTModel
from blt.model import TernaryBLTModel
from blt.ngram_hash import reference_ngram_hashes


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(
        local_dim=32, global_dim=32, decoder_dim=32,
        n_layers_local_encoder=1, n_layers_global=1, n_layers_local_decoder=1,
        n_heads_local_encoder=4, n_heads_global=4, n_heads_local_decoder=4,
        n_heads_cross=4, local_window=None, patch_size=4,
        ngram_vocab_size=97, ngram_dim=8,
    )
    base.update(overrides)
    return TernaryBLTConfig(**base)


def _numpy(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _tokens(batch=3, length=24, seed=0):
    return np.random.default_rng(seed).integers(4, 260, size=(batch, length), dtype=np.int64)


def test_both_stacks_match_the_reference_hash() -> None:
    # The two stacks index the same tables, so a divergence here would be a
    # silent correctness bug rather than a crash.
    config = _config()
    ids = _tokens()
    expected_idx, expected_valid = reference_ngram_hashes(
        ids, config.ngram_sizes, config.ngram_vocab_size
    )

    torch_idx, torch_valid = HashNgramEmbedding(config).hashes(torch.from_numpy(ids))
    assert np.array_equal(torch_idx.numpy(), expected_idx)
    assert np.array_equal(torch_valid.numpy(), expected_valid)

    mlx_idx, mlx_valid = MLXHashNgramEmbedding(config).hashes(mx.array(ids))
    mx.eval(mlx_idx, mlx_valid)
    assert np.array_equal(np.asarray(mlx_idx), expected_idx)
    assert np.array_equal(np.asarray(mlx_valid), expected_valid)


def test_hash_is_causal() -> None:
    # g_{i,n} ends at i, so changing a later byte must not move an earlier index.
    config = _config()
    ids = _tokens()
    layer = MLXHashNgramEmbedding(config)
    before, _ = layer.hashes(mx.array(ids))
    bumped = ids.copy()
    bumped[:, 17:] = (bumped[:, 17:] + 7) % 260
    after, _ = layer.hashes(mx.array(bumped))
    mx.eval(before, after)
    assert np.array_equal(np.asarray(before)[:, :, :17], np.asarray(after)[:, :, :17])


def test_short_prefixes_are_dropped_not_hashed() -> None:
    # An n-gram reaching before the sequence start contributes nothing.
    config = _config()
    _, valid = reference_ngram_hashes(_tokens(), config.ngram_sizes, config.ngram_vocab_size)
    for slot, size in enumerate(config.ngram_sizes):
        assert not valid[slot, :, : size - 1].any()
        assert valid[slot, :, size - 1 :].all()


def test_embedding_changes_the_byte_embedding_and_respects_the_mask() -> None:
    config = _config()
    ids = mx.array(_tokens(batch=2, length=16))
    layer = MLXHashNgramEmbedding(config)
    mx.eval(layer.parameters())
    base = mx.random.normal((2, 16, config.local_dim))
    mask = mx.concatenate(
        [mx.ones((2, 10), dtype=mx.bool_), mx.zeros((2, 6), dtype=mx.bool_)], axis=1
    )
    out = layer(base, ids, mask)
    mx.eval(out)
    assert out.shape == base.shape
    # Masked positions get no n-gram contribution, so they are unchanged.
    assert float(mx.max(mx.abs(out[:, 10:] - base[:, 10:]))) < 1e-5


def test_models_route_generation_and_training_through_embed_bytes() -> None:
    # A path that called byte_embeddings directly would silently skip n-grams.
    for config, model_cls, wrap in (
        (_config(), TernaryBLTModel, lambda a: torch.from_numpy(a)),
        (_config(), MLXTernaryBLTModel, lambda a: mx.array(a)),
    ):
        model = model_cls(config)
        assert model.ngram_embeddings is not None
        ids = wrap(_tokens(batch=1, length=16))
        plain = _numpy(model.byte_embeddings(ids))
        embedded = _numpy(model.embed_bytes(ids))
        assert float(np.abs(plain - embedded).max()) > 1e-6


def test_disabling_the_flag_removes_the_module_and_the_normaliser() -> None:
    config = _config(use_ngram_embeddings=False)
    model = MLXTernaryBLTModel(config)
    assert model.ngram_embeddings is None
    ids = mx.array(_tokens(batch=1, length=16))
    assert float(mx.max(mx.abs(model.embed_bytes(ids) - model.byte_embeddings(ids)))) == 0.0
