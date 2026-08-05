"""Dynamic Tanh (DyT) residual-stream norm."""

from __future__ import annotations

import torch

from config import TernaryConfig
from layers.dyt import DynamicTanh, make_norm
from model import BitNetDeep


def test_dyt_matches_closed_form() -> None:
    layer = DynamicTanh(8, alpha_init=0.5)
    x = torch.randn(2, 4, 8)
    expected = torch.tanh(0.5 * x)  # weight=1, bias=0 at init
    assert torch.allclose(layer(x), expected)


def test_make_norm_switches() -> None:
    assert type(make_norm(16, norm_type="rms")).__name__ == "RMSNorm"
    assert isinstance(make_norm(16, norm_type="dyt"), DynamicTanh)


def test_bitnet_builds_with_dyt() -> None:
    config = TernaryConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        use_engram=False,
        mtp_depth=0,
        norm_type="dyt",
        num_loops=1,
    )
    model = BitNetDeep(config)
    assert isinstance(model.norm, DynamicTanh)
    assert isinstance(model.layers[0].attn_norm, DynamicTanh)
    ids = torch.randint(0, 512, (1, 32))
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[-1] == 512
    assert torch.isfinite(logits).all()
