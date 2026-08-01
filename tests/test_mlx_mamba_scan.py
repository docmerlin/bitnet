"""Metal selective-scan parity smoke (forward + backward)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mamba_scan_kernel import (
    _scan_vjp_mlx,
    selective_scan,
    selective_scan_bwd_metal,
    selective_scan_fwd_metal,
    selective_scan_reference,
)


def _rand_scan_inputs(b=1, l=48, h=8, p=16, n=32, seed=0):
    mx.random.seed(seed)
    x = mx.random.normal((b, l, h, p)).astype(mx.float32)
    decay = (mx.sigmoid(mx.random.normal((b, l, h))) * 0.9 + 0.05).astype(mx.float32)
    dt = (mx.logaddexp(mx.random.normal((b, l, h)), 0.0) * 0.05).astype(mx.float32)
    bmat = (mx.random.normal((b, l, h, n)) * 0.1).astype(mx.float32)
    cmat = (mx.random.normal((b, l, h, n)) * 0.1).astype(mx.float32)
    trap = mx.sigmoid(mx.random.normal((b, l, h))).astype(mx.float32)
    return x, decay, dt, bmat, cmat, trap


def test_metal_scan_matches_reference() -> None:
    x, decay, dt, bmat, cmat, trap = _rand_scan_inputs()
    y_ref = selective_scan_reference(x, decay, dt, bmat, cmat, trap)
    y_met = selective_scan_fwd_metal(x, decay, dt, bmat, cmat, trap)
    mx.eval(y_ref, y_met)
    err = float(mx.max(mx.abs(y_ref - y_met)))
    assert err < 1e-5, err


def test_metal_bwd_matches_mlx_reference() -> None:
    x, decay, dt, bmat, cmat, trap = _rand_scan_inputs(l=32, h=4, p=16, n=32)
    dy = mx.random.normal(x.shape).astype(mx.float32)
    g_met = selective_scan_bwd_metal(x, decay, dt, bmat, cmat, trap, dy)
    g_mlx = _scan_vjp_mlx((x, decay, dt, bmat, cmat, trap), dy)
    mx.eval(*g_met, *g_mlx)
    for a, b in zip(g_met, g_mlx):
        err = float(mx.max(mx.abs(a - b)))
        assert err < 1e-4, err


def test_selective_scan_custom_function_grad() -> None:
    x, decay, dt, bmat, cmat, trap = _rand_scan_inputs(l=24, h=4, p=8, n=16)

    def loss(x_):
        return mx.mean(selective_scan(x_, decay, dt, bmat, cmat, trap) ** 2)

    val, grad = mx.value_and_grad(loss)(x)
    mx.eval(val, grad)
    assert float(val) == float(val)
    assert float(mx.mean(mx.abs(grad))) >= 0.0


@pytest.mark.parametrize("headdim", [1, 2, 3])
def test_backward_is_correct_at_small_head_widths(headdim) -> None:
    # The scalar reductions are striped across the threadgroup rather than
    # assigned to threads 0/1/2: the threadgroup is headdim threads wide, so a
    # fixed assignment left ddt and dtrap unwritten below 3 and returned
    # uninitialised device memory. config.py permits mamba_headdim >= 1.
    inputs = _rand_scan_inputs(b=1, l=8, h=2, p=headdim, n=16)
    dy = mx.random.normal(inputs[0].shape).astype(mx.float32)
    metal = selective_scan_bwd_metal(*inputs, dy)
    reference = _scan_vjp_mlx(inputs, dy)
    mx.eval(*metal, *reference)
    for got, want in zip(metal, reference):
        assert float(mx.max(mx.abs(got - want))) < 1e-4
