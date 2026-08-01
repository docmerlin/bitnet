"""Fused straight-through activation quantiser vs the expression it replaces."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_ternary_kernel import activation_levels, ste_activation_quantize


def _reference(x: mx.array, bits: int) -> mx.array:
    positive = (2 ** (bits - 1)) - 1
    negative = 2 ** (bits - 1)
    scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5) / positive
    return mx.clip(mx.round(x / scale), -negative, positive) * scale


SHAPES = [(4, 1024, 256), (2, 7, 1024), (3, 33), (16, 512, 1024), (5, 1, 17), (1, 1, 2048)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bits", [16, 8, 4, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_bit_identical_to_the_expression_form(shape, bits, dtype) -> None:
    # Bit-identical, not merely close: this replaces the expression in every
    # HBitLinear, so any drift would silently change training everywhere.
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(dtype)
    fused = ste_activation_quantize(x, activation_levels(bits))
    reference = _reference(x, bits)
    mx.eval(fused, reference)
    assert float(mx.max(mx.abs(fused.astype(mx.float32) - reference.astype(mx.float32)))) == 0.0


def test_gradient_is_straight_through() -> None:
    # The quantiser is a step function; its true derivative is zero almost
    # everywhere. The STE passes the cotangent through unchanged.
    x = mx.random.normal((4, 128))
    upstream = mx.random.normal((4, 128))
    levels = activation_levels(8)

    def loss(a):
        return mx.sum(ste_activation_quantize(a, levels) * upstream)

    grad = mx.grad(loss)(x)
    mx.eval(grad)
    assert float(mx.max(mx.abs(grad - upstream))) == 0.0


def test_rows_are_scaled_independently() -> None:
    # Per-row absmax: a large row must not shrink the grid of a small one.
    x = mx.concatenate([mx.ones((1, 64)) * 100.0, mx.ones((1, 64)) * 0.01], axis=0)
    out = ste_activation_quantize(x, activation_levels(8))
    mx.eval(out)
    assert float(mx.max(mx.abs(out[0] - 100.0))) < 1.0
    assert float(mx.max(mx.abs(out[1] - 0.01))) < 1e-4


def test_a_zero_row_does_not_divide_by_zero() -> None:
    out = ste_activation_quantize(mx.zeros((2, 32)), activation_levels(8))
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))
    assert float(mx.max(mx.abs(out))) == 0.0


def test_levels_carry_no_gradient() -> None:
    levels = activation_levels(4)
    grads = mx.grad(lambda l: mx.sum(ste_activation_quantize(mx.random.normal((2, 16)), l)))(levels)
    mx.eval(grads)
    assert float(mx.max(mx.abs(grads))) == 0.0


def test_bitnet_prepare_input_matches_the_expression_it_replaced() -> None:
    """The test the first pass was missing.

    The parametrised cases above check the kernel against a reference written
    alongside it, which is not the same as checking it against the code it
    replaced. That expression read its level count from a float32 array, so a
    bfloat16 activation was promoted and the whole quantisation ran at float32;
    the kernel works in the input dtype. Comparing kernel-to-reference could not
    see the difference, and the stack silently dropped to bfloat16 quantisation.
    """
    from mlx_model import MLXBitNetConfig, MLXHBitLinear

    config = MLXBitNetConfig(
        hidden_size=64, num_attention_heads=4, intermediate_size=128, use_engram=False
    )
    layer = MLXHBitLinear(300, 64, config)
    layer.set_quantization_state(1.0, 1.0, 8)
    mx.eval(layer.parameters())

    x = mx.random.normal((4, 300)).astype(mx.bfloat16)
    levels = mx.array(127.0)  # float32, exactly as set_quantization_state stored it
    scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5) / levels
    expected = mx.clip(mx.round(x / scale), -(levels + 1), levels) * scale
    actual = layer.prepare_input(x)
    mx.eval(expected, actual)

    assert actual.dtype == mx.float32, "bfloat16 here would coarsen an 8-bit grid"
    assert float(mx.max(mx.abs(expected - actual))) == 0.0
