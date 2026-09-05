"""Zero-init AdaRMS is an identity; Fourier embed is finite."""

from __future__ import annotations

import mlx.core as mx
import pytest

from dblocks.condition import MLXAdaRMS, MLXSigmaEmbed, apply_ada, fourier_frequencies


def test_adarms_is_identity_at_construction() -> None:
    ada = MLXAdaRMS(hidden_size=8, cond_dim=16)
    x = mx.random.normal((2, 4, 8))
    cond = mx.random.normal((16,))
    mx.eval(ada.parameters(), x, cond)
    out = ada(x, cond)
    mx.eval(out)
    assert mx.allclose(out, x, rtol=0, atol=0).item()


def test_apply_ada_skips_when_missing() -> None:
    x = mx.ones((1, 2, 4))
    ada = MLXAdaRMS(4, 8)
    mx.eval(ada.parameters(), x)
    assert mx.array_equal(apply_ada(x, None, mx.ones((8,))), x).item()
    assert mx.array_equal(apply_ada(x, ada, None), x).item()


def test_sigma_embed_is_finite_for_scalar_and_batch() -> None:
    embed = MLXSigmaEmbed(fourier_dim=16, cond_dim=8)
    mx.eval(embed.parameters())
    scalar = embed(mx.array(1.5))
    batched = embed(mx.array([0.1, 1.0, 10.0]))
    mx.eval(scalar, batched)
    assert scalar.shape == (8,)
    assert batched.shape == (3, 8)
    assert bool(mx.all(mx.isfinite(scalar)).item())
    assert bool(mx.all(mx.isfinite(batched)).item())


def test_fourier_frequencies_stay_in_float32_range() -> None:
    freqs = fourier_frequencies(32)
    mx.eval(freqs)
    values = [float(v) for v in freqs]
    assert values[0] == pytest.approx(1.0)
    assert max(values) == pytest.approx(10_000.0, rel=1e-5)
    assert min(values) >= 1.0 - 1e-5
    # Default half=32 would have been 2**31 with the old 2**k grid.
    assert max(values) < 2**15


def test_adarms_broadcasts_batched_cond_on_the_feature_axis() -> None:
    ada = MLXAdaRMS(hidden_size=4, cond_dim=4)
    ada.proj.weight = mx.ones_like(ada.proj.weight) * 0.05
    ada.proj.bias = mx.zeros_like(ada.proj.bias)
    x = mx.ones((3, 5, 4))
    cond = mx.arange(12).astype(mx.float32).reshape((3, 4))
    mx.eval(ada.parameters(), x, cond)
    out = ada(x, cond)
    mx.eval(out)
    assert out.shape == x.shape
    # Different batch rows must differ; tokens in a row share the same affine.
    assert not mx.allclose(out[0], out[1], rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(out[0, 0], out[0, 1], rtol=1e-5, atol=1e-5).item()


def test_adarms_can_leave_identity_after_a_nonzero_update() -> None:
    ada = MLXAdaRMS(hidden_size=4, cond_dim=4)
    ada.proj.weight = ada.proj.weight + 0.1
    x = mx.ones((1, 1, 4))
    cond = mx.ones((4,))
    mx.eval(ada.parameters(), x, cond)
    out = ada(x, cond)
    mx.eval(out)
    assert not mx.allclose(out, x, rtol=0, atol=0).item()
