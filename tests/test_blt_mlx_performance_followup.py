"""Numerical and cache-lifetime contracts for the second performance pass."""

import mlx.core as mx
import numpy as np
import pytest

from blt.mlx_entropy_model import MLXByteEntropyModel, EntropyGenerationCache
from blt.mlx_generate import GenerationStats, _DraftCache, _Patching, _run_global, _ensure_draft_prefill, generate
from blt.mlx_layers import MLXHBitLinear
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_patching import build_uniform_patch_lengths, pool_patch_representations
from blt.patching.teacher_patcher import UniformPatcher
from test_blt_mlx_generate import _config


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_ternary_precision_and_ste(dtype):
    layer = MLXHBitLinear(32, 32, config=_config(use_hadamard=False))
    layer.set_dtype(dtype)
    x = mx.ones((1, 32), dtype=dtype)
    assert layer.effective_weight().dtype == dtype
    assert layer(x).dtype == dtype
    weight = layer.weight

    def loss(w):
        layer.weight = w
        return layer.effective_weight().sum()

    gradient = mx.grad(loss)(weight)
    layer.weight = weight
    assert bool(mx.all(gradient == 1))
    expected = layer(x)
    layer.pin_inference_weight()
    assert layer._pinned_weight.dtype == dtype
    assert bool(mx.all(layer(x) == expected))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [16, 19])
@pytest.mark.parametrize("pooling", ["sum", "mean"])
def test_uniform_pool_matches_ragged_reference_and_gradient(dtype, length, pooling):
    mx.random.seed(10)
    hidden = mx.random.normal((2, length, 32)).astype(dtype)
    lengths = build_uniform_patch_lengths(2, length, 4)
    mask = mx.arange(length)[None] < mx.array([[length], [length - 6]])

    def pool(x, hint):
        return pool_patch_representations(x, lengths, token_mask=mask, pooling=pooling, uniform_patch_size=hint)

    assert bool(mx.allclose(pool(hidden, 4), pool(hidden, None), atol=0.02, rtol=0.01))
    reference = mx.grad(lambda x: mx.square(pool(x, None)).sum())(hidden)
    actual = mx.grad(lambda x: mx.square(pool(x, 4)).sum())(hidden)
    assert bool(mx.allclose(actual, reference, atol=0.05, rtol=0.02))


@pytest.mark.parametrize("threshold", [0.0, 5.4, 100.0])
def test_entropy_cache_extension_replacement_and_truncation(threshold, monkeypatch):
    mx.random.seed(8)
    model = MLXByteEntropyModel(_config(max_patch_length=3), dim=32, num_layers=2, num_heads=4)
    cache = EntropyGenerationCache(model, threshold)
    lengths_scored = []
    original = model.forward_cached

    def score(tokens, *args, **kwargs):
        lengths_scored.append(tokens.shape[1])
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_cached", score)
    rows = [[4, 8, 9, 12], [4, 8, 9, 12, 10], [4, 8, 9, 12, 10],
            [4, 8, 7, 12, 10], [4, 8], [4, 8, 20, 30, 40, 50, 60]]
    for row in rows:
        tokens = mx.array([row])
        assert bool(mx.array_equal(cache.patch_lengths(tokens), model.predict_patch_lengths(tokens, threshold=threshold)))
        assert cache.opens_new_patch(tokens) == bool(model.opens_new_patch(tokens, threshold=threshold)[0])
    assert lengths_scored == [4, 1, 3, 5]


@pytest.mark.parametrize("cross_attn_k", [1, 2])
def test_global_prefill_and_decoder_projections_are_reused(cross_attn_k, monkeypatch):
    model = MLXTernaryBLTModel(_config(cross_attn_k=cross_attn_k, local_window=4))
    tokens = mx.array([[4, 5, 6, 7, 8, 9, 10, 11]])
    stats, cache = GenerationStats(), _DraftCache()
    calls = []
    original = model.local_encoder.encode_bytes_prefill

    def prefill(x):
        calls.append(x.shape[1])
        return original(x)

    monkeypatch.setattr(model.local_encoder, "encode_bytes_prefill", prefill)
    projections = []
    layer = model.local_decoder.cross_attn_layers[0]
    name = "project_kv" if cross_attn_k > 1 else "project_values"
    project = getattr(layer, name)

    def projected(x):
        projections.append(x.shape[1])
        return project(x)

    monkeypatch.setattr(layer, name, projected)
    latents, ids = _run_global(model, tokens, _Patching(UniformPatcher(4)), stats, cache)
    _ensure_draft_prefill(model, tokens, latents, ids, stats, cache)
    _ensure_draft_prefill(model, tokens, latents, ids, stats, cache)
    mx.eval(cache.last_hidden)
    assert calls == [8]
    assert len(projections) == 1
    assert stats.draft_encoder == 0


def test_bfloat16_model_and_windowed_speculation():
    mx.random.seed(15)
    model = MLXTernaryBLTModel(_config(local_window=4, cross_attn_k=2))
    model.set_dtype(mx.bfloat16)
    tokens = mx.array([[4, 5, 6, 7, 8, 9, 10, 11]])
    assert model(tokens, unpadded=True).logits.dtype == mx.bfloat16
    baseline, _ = generate(model, tokens, max_new_bytes=12, speculation_window=0, eos_id=-1)
    speculative, _ = generate(model, tokens, max_new_bytes=12, speculation_window=4, eos_id=-1)
    default, _ = generate(model, tokens, max_new_bytes=12, eos_id=-1)
    assert np.array_equal(np.asarray(baseline), np.asarray(speculative))
    assert np.array_equal(np.asarray(default), np.asarray(speculative))


def test_low_precision_loss_uses_float32_reductions():
    from blt.mlx_losses import MLXDistillationLossWeights, blt_distillation_loss
    logits = mx.random.normal((2, 16, 260)).astype(mx.bfloat16)
    arguments = dict(labels=mx.ones((2, 16), dtype=mx.int32), attention_mask=mx.ones((2, 16)),
                     weights=MLXDistillationLossWeights(logits_kl=0))
    actual, _ = blt_distillation_loss(logits, **arguments)
    expected, _ = blt_distillation_loss(logits.astype(mx.float32), **arguments)
    assert actual.dtype == mx.float32
    assert bool(mx.array_equal(actual, expected))
