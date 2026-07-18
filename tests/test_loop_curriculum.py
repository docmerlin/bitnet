"""Tests for recurrent loop depth curriculum."""

from __future__ import annotations

from training.schedules import loop_count_for_progress


def test_loop_curriculum_ramp() -> bool:
    assert loop_count_for_progress(0.0, min_loops=1, max_loops=4, curriculum_ratio=0.2) == 1
    assert loop_count_for_progress(0.1, min_loops=1, max_loops=4, curriculum_ratio=0.2) == 2
    assert loop_count_for_progress(0.2, min_loops=1, max_loops=4, curriculum_ratio=0.2) == 4
    assert loop_count_for_progress(0.5, min_loops=1, max_loops=4, curriculum_ratio=0.2) == 4
    assert loop_count_for_progress(1.0, min_loops=1, max_loops=4, curriculum_ratio=0.2) == 4
    print("Loop curriculum ramp tests passed")
    return True


def test_loop_curriculum_disabled() -> bool:
    assert loop_count_for_progress(0.0, min_loops=1, max_loops=4, curriculum_ratio=0.0) == 4
    assert loop_count_for_progress(0.0, min_loops=2, max_loops=2, curriculum_ratio=0.5) == 2
    print("Loop curriculum disabled tests passed")
    return True


def test_loop_curriculum_delayed() -> bool:
    kwargs = {
        "min_loops": 1,
        "max_loops": 4,
        "curriculum_start_ratio": 0.7,
        "curriculum_ratio": 0.9,
    }
    assert loop_count_for_progress(0.0, **kwargs) == 1
    assert loop_count_for_progress(0.7, **kwargs) == 1
    assert loop_count_for_progress(0.8, **kwargs) == 3
    assert loop_count_for_progress(0.9, **kwargs) == 4
    assert loop_count_for_progress(1.0, **kwargs) == 4
    return True


def test_small_sublayer_output_init() -> bool:
    import torch

    from config import TernaryConfig
    from layers.hybrid_block import HybridTransformerBlock
    from layers.h_bitlinear import HBitLinear

    cfg = TernaryConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        use_hadamard=False,
    )
    # Reference kaiming magnitude vs post-scale.
    ref = HBitLinear(64, 64, bias=False, config=cfg)
    block = HybridTransformerBlock(cfg)
    ref_rms = ref.weight.float().pow(2).mean().sqrt().item()
    from layers.hybrid_block import SUBLAYER_OUT_INIT_SCALE

    out_rms = block.infini_attn.o_proj.weight.float().pow(2).mean().sqrt().item()
    assert out_rms < 0.2 * ref_rms, f"o_proj should be downscaled, {out_rms} vs {ref_rms}"
    assert 0.0 < SUBLAYER_OUT_INIT_SCALE < 1.0
    x = torch.randn(1, 4, 64)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print("Small sublayer output init tests passed")
    return True


if __name__ == "__main__":
    test_loop_curriculum_ramp()
    test_loop_curriculum_disabled()
    test_loop_curriculum_delayed()
    test_small_sublayer_output_init()
