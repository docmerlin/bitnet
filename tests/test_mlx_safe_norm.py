"""L2 norms in the model must differentiate at zero.

``mx.linalg.norm`` has vjp ``x / ||x||``, which is 0/0 at the origin. Guarding
the result with ``mx.maximum(norm(x), eps)`` fixes the forward and does nothing
for the backward: ``maximum`` multiplies the already-NaN cotangent by zero, and
NaN survives that. This cost the repo a training divergence -- two or more BLT
local encoder layers died on the first update, because a collapsed activation
made a PaTH Householder vector exactly zero and the NaN propagated back through
every parameter feeding it.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_model import _safe_norm, _safe_normalize


def _grad_finite(fn, value):
    grad = mx.grad(lambda a: mx.sum(fn(a) ** 2))(value)
    mx.eval(grad)
    return bool(mx.all(mx.isfinite(grad)))


def test_the_unguarded_form_is_the_trap_this_replaces() -> None:
    # Pinning the failure mode so the fix is not silently reverted.
    def guarded(v):
        return v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-6)

    assert _grad_finite(guarded, mx.array([[1.0, 2.0, 3.0]]))
    assert not _grad_finite(guarded, mx.zeros((1, 3)))


@pytest.mark.parametrize("value", [mx.zeros((2, 4)), mx.array([[0.0, 0.0], [3.0, 4.0]])])
def test_safe_normalize_differentiates_at_zero(value) -> None:
    assert _grad_finite(_safe_normalize, value)
    out = _safe_normalize(value)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))


def test_safe_norm_differentiates_at_zero() -> None:
    assert _grad_finite(lambda v: _safe_norm(v, axis=-1), mx.zeros((2, 4)))


def test_safe_forms_agree_with_the_plain_ones_away_from_zero() -> None:
    value = mx.random.normal((4, 8)) * 3.0
    mx.eval(value)
    assert float(mx.max(mx.abs(_safe_norm(value, axis=-1) - mx.linalg.norm(value, axis=-1)))) < 1e-5
    plain = value / mx.linalg.norm(value, axis=-1, keepdims=True)
    assert float(mx.max(mx.abs(_safe_normalize(value) - plain))) < 1e-5


def test_a_zero_path_vector_no_longer_poisons_the_gradient() -> None:
    # The concrete shape that failed: one head's Householder vector at exactly
    # zero while the rest are normal.
    vectors = mx.concatenate([mx.zeros((1, 1, 1, 16)), mx.random.normal((1, 1, 3, 16))], axis=2)
    mx.eval(vectors)
    assert _grad_finite(_safe_normalize, vectors)
