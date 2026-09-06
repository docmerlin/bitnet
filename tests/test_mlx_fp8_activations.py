"""Native fp8 e4m3 activations (mlx.to_fp8), not absmax fake-quant."""

from __future__ import annotations

import mlx.core as mx

from mlx_model import MLXHBitLinear, MLXBitNetConfig


def _config(**overrides) -> MLXBitNetConfig:
    base = dict(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        num_loops=1,
        use_engram=False,
        use_hadamard=False,
        mtp_depth=0,
    )
    base.update(overrides)
    return MLXBitNetConfig(**base)


def test_fp8_roundtrip_is_not_identity() -> None:
    x = mx.random.normal((4, 64)).astype(mx.bfloat16)
    q = mx.from_fp8(mx.to_fp8(x), mx.bfloat16)
    mx.eval(q)
    assert float(mx.max(mx.abs(x.astype(mx.float32) - q.astype(mx.float32)))) > 1e-3


def test_fp8_ste_has_identity_gradient() -> None:
    x = mx.random.normal((2, 32)).astype(mx.float32)

    def loss(a):
        quantized = mx.from_fp8(mx.to_fp8(a), a.dtype)
        y = a + mx.stop_gradient(quantized - a)
        return mx.sum(y)

    grad = mx.grad(loss)(x)
    mx.eval(grad)
    assert bool(mx.allclose(grad, mx.ones_like(x), rtol=1e-5, atol=1e-5))


def test_linear_prepare_input_encodes_fp8() -> None:
    layer = MLXHBitLinear(64, 64, _config())
    x = mx.random.normal((2, 8, 64)).astype(mx.bfloat16)
    encoded = layer.prepare_input(x)
    mx.eval(encoded)
    assert float(mx.max(mx.abs(x.astype(mx.float32) - encoded.astype(mx.float32)))) > 1e-3
