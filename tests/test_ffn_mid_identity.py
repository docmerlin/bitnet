"""The FFN mid must be an identity in the forward pass, not just in storage.

The raw weight being eye(N) proves nothing: these are ternary layers, and the
per-output-channel scale is mean(|row|). A plain eye(N) row is one 1 and N-1
zeros, so the scale is 1/N and the *quantised* weight -- the one the matmul
actually uses -- is eye(N)/N. That is a 1/1024 attenuator, not a pass-through.
"""

from __future__ import annotations

import pytest
import torch

mx = pytest.importorskip("mlx.core")

from blt.config import TernaryBLTConfig
from blt.layers.transformer_block import TernaryMLP
from blt.mlx_layers import MLXTernaryMLP
from mlx_model import MLXBitNetConfig, MLXHybridBlock


def _blt_config() -> TernaryBLTConfig:
    return TernaryBLTConfig(
        local_dim=64, global_dim=64, decoder_dim=64, patch_size=4,
        n_heads_local_encoder=4, n_heads_global=4, n_heads_local_decoder=4, n_heads_cross=4,
    )


def test_mlx_blt_mid_quantises_to_the_identity() -> None:
    mlp = MLXTernaryMLP(64, 4.0, config=_blt_config())
    mx.eval(mlp.parameters())
    size = mlp.mid_proj.weight.shape[0]
    assert bool(mx.allclose(mlp.mid_proj.effective_weight(), mx.eye(size), atol=1e-6))


def test_torch_blt_mid_quantises_to_the_identity() -> None:
    mlp = TernaryMLP(64, 4.0, config=_blt_config())
    size = mlp.mid_proj.weight.shape[0]
    with torch.no_grad():
        effective = mlp.mid_proj.effective_weight(torch.float32)
    assert torch.allclose(effective, torch.eye(size), atol=1e-6)


def test_bitnet_mid_quantises_to_the_identity() -> None:
    config = MLXBitNetConfig(
        vocab_size=256, hidden_size=64, num_attention_heads=4, intermediate_size=128,
        num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1, num_loops=1,
        use_engram=False, mtp_depth=0,
    )
    block = MLXHybridBlock(config, 1)
    mx.eval(block.parameters())
    size = block.mid.weight.shape[0]
    assert bool(mx.allclose(block.mid.effective_weight(mx.float32), mx.eye(size), atol=1e-6))


def test_a_signal_passes_through_the_mid_unattenuated() -> None:
    # The property the identity init exists to provide: a cold-start mid behaves
    # like the classic two-matrix SwiGLU path.
    mlp = MLXTernaryMLP(64, 4.0, config=_blt_config())
    mx.eval(mlp.parameters())
    x = mx.random.normal((2, 32, mlp.hidden_dim))
    out = mlp.mid_proj(x)
    mx.eval(x, out)
    assert float(mx.std(out)) / float(mx.std(x)) > 0.5


def test_the_ramp_does_not_move_the_mid() -> None:
    # (1-mix)*raw + mix*quantised is only meaningful when raw and quantised share
    # a scale; for the mid they differ by N, so mixing would put it at 768x
    # identity a quarter of the way through the ramp.
    mlp = MLXTernaryMLP(64, 4.0, config=_blt_config())
    mx.eval(mlp.parameters())
    size = mlp.mid_proj.weight.shape[0]
    for weight_mix in (0.0, 0.25, 0.5, 1.0):
        mlp.mid_proj.set_quantization_state(weight_mix, 0.0, 8)
        assert bool(mx.allclose(mlp.mid_proj.effective_weight(), mx.eye(size), atol=1e-6)), weight_mix
    # Sibling projections still ramp normally.
    mlp.up_proj.set_quantization_state(0.25, 0.0, 8)
    assert float(mlp.up_proj.weight_mix) == 0.25


def test_rfmoe_expert_mids_quantise_to_the_identity() -> None:
    # Same square mid, same trap: the experts had the identical bug.
    import torch as _torch
    from config import TernaryConfig
    from layers.rfmoe import RFMoEExpert
    from mlx_model import MLXRFMoEExpert

    torch_config = TernaryConfig(
        hidden_size=32, num_attention_heads=4, head_dim=8,
        use_rfmoe=True, rfmoe_expert_dim=16, rfmoe_rank=4,
    )
    expert = RFMoEExpert(32, expert_dim=16, rank=4, config=torch_config)
    with _torch.no_grad():
        effective = expert.w_mid.effective_weight(_torch.float32)
    assert _torch.allclose(effective, _torch.eye(16), atol=1e-6)

    mlx_config = MLXBitNetConfig(
        vocab_size=256, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1, num_loops=1,
        use_engram=False, mtp_depth=0, use_rfmoe=True, rfmoe_expert_dim=16, rfmoe_rank=4,
    )
    mlx_expert = MLXRFMoEExpert(mlx_config, expert_dim=16, rank=4)
    mx.eval(mlx_expert.parameters())
    assert bool(mx.allclose(mlx_expert.w_mid.effective_weight(mx.float32), mx.eye(16), atol=1e-6))


def test_torch_dense_bitnet_ffn_mid_quantises_to_the_identity() -> None:
    # The site the first pass missed: it is named ffn_mid, not mid or mid_proj.
    from config import TernaryConfig
    from layers.hybrid_block import HybridTransformerBlock

    config = TernaryConfig(
        hidden_size=64, num_attention_heads=4, head_dim=16, intermediate_size=128,
    )
    block = HybridTransformerBlock(config, 0)
    with torch.no_grad():
        effective = block.ffn_mid.effective_weight(torch.float32)
    assert torch.allclose(effective, torch.eye(128), atol=1e-6)
