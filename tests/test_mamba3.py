"""Mamba-3 mixer schedule and forward smoke tests."""

import torch

from config import TernaryConfig
from layers.hybrid_block import HybridTransformerBlock
from layers.mamba3 import Mamba3Mixer, heavy_tail_activation, is_mamba3_layer
from model import BitNetDeep


def test_is_mamba3_layer_every_third_from_zero() -> None:
    assert is_mamba3_layer(0, 3)
    assert not is_mamba3_layer(1, 3)
    assert not is_mamba3_layer(2, 3)
    assert is_mamba3_layer(3, 3)
    assert is_mamba3_layer(6, 3)
    assert not is_mamba3_layer(None, 3)


def test_heavy_tail_positive_and_smooth_at_zero() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = heavy_tail_activation(x)
    assert torch.all(y > 0)
    assert torch.isclose(y[2], torch.tensor(1.0))


def test_mamba3_mixer_shapes_and_grads() -> None:
    torch.manual_seed(0)
    cfg = TernaryConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=False,
        use_mamba3_layers=True,
        mamba_d_state=16,
        mamba_headdim=16,
        mamba_expand=2,
    )
    mixer = Mamba3Mixer(cfg)
    x = torch.randn(2, 12, 64, requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
    assert mixer.out_proj.weight.grad is not None


def test_hybrid_schedule_one_third_mamba() -> None:
    cfg = TernaryConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=False,
        use_mamba3_layers=True,
        mamba_layer_period=3,
        mamba_d_state=16,
        mamba_headdim=16,
        attn_res_mode="sandwich",
    )
    blocks = [HybridTransformerBlock(cfg, layer_id=i) for i in range(6)]
    flags = [b.use_mamba3 for b in blocks]
    assert flags == [True, False, False, True, False, False]
    assert blocks[0].mamba is not None and blocks[0].infini_attn is None
    assert blocks[1].mamba is None and blocks[1].infini_attn is not None


def test_model_forward_with_mamba_layers() -> None:
    torch.manual_seed(1)
    cfg = TernaryConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=False,
        use_mamba3_layers=True,
        mamba_layer_period=3,
        mamba_d_state=16,
        mamba_headdim=16,
        attn_res_mode="sandwich",
        block_size=4,
        path_window_size=8,
        infini_memory_dim=8,
    )
    model = BitNetDeep(cfg)
    n_mamba = sum(1 for layer in model.layers if layer.use_mamba3)
    assert n_mamba == 2  # layers 0, 3 of 6
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
