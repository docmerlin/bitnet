"""MLX Mamba-3 hybrid schedule smoke (requires mlx)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_model import MLXBitNet, MLXBitNetConfig


def test_mlx_mamba_every_third_layer_and_train_step() -> None:
    cfg = MLXBitNetConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=2,
        num_coda_layers=1,
        num_loops=1,
        block_size=2,
        path_window_size=8,
        use_engram=False,
        use_mamba3_layers=True,
        mamba_layer_period=3,
        mamba_d_state=16,
        mamba_headdim=16,
        mamba_expand=2,
        use_path_kernel=False,
        attn_res_mode="kimi",
    )
    model = MLXBitNet(cfg)
    flags = [block.use_mamba3 for block in model.blocks]
    assert flags == [True, False, False, True]

    tokens = mx.random.randint(0, cfg.vocab_size, (2, 12))

    def loss_fn(model, toks):
        return mx.mean(model(toks) ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens)
    mx.eval(loss, grads)
    assert float(loss) == float(loss)  # finite


def test_mlx_mamba_can_disable() -> None:
    cfg = MLXBitNetConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=1,
        use_engram=False,
        use_mamba3_layers=False,
        use_path_kernel=False,
    )
    model = MLXBitNet(cfg)
    assert all(not b.use_mamba3 for b in model.blocks)
    assert all(b.attn is not None for b in model.blocks)
