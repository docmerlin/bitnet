"""Unit tests for Kimi Block AttnRes depth mix."""

from __future__ import annotations

import torch

from config import TernaryConfig
from layers.attn_res import AttnResStream, DepthAttnMix
from layers.hybrid_block import HybridTransformerBlock
from model import BitNetDeep


def test_depth_attn_mix_is_convex_combination() -> bool:
    torch.manual_seed(0)
    mix = DepthAttnMix(16)
    b, t, d = 2, 3, 16
    completed = [torch.randn(b, t, d), torch.randn(b, t, d)]
    partial = torch.randn(b, t, d)
    out = mix(completed, partial)
    assert out.shape == (b, t, d)

    # With zero proj, logits equal → uniform softmax over N+1 states.
    with torch.no_grad():
        mix.proj.weight.zero_()
    out0 = mix(completed, partial)
    stacked = torch.stack([*completed, partial], dim=0)
    expected = stacked.mean(dim=0)
    assert torch.allclose(out0, expected, atol=1e-5), "zero query → uniform mix"

    # Gradients reach the pseudo-query.
    loss = mix(completed, partial).sum()
    loss.backward()
    assert mix.proj.weight.grad is not None
    assert float(mix.proj.weight.grad.abs().sum()) > 0.0
    print("DepthAttnMix convex / grad tests passed")
    return True


def test_stream_closes_depth_blocks() -> bool:
    torch.manual_seed(0)
    d = 8
    mix_a, mix_m = DepthAttnMix(d), DepthAttnMix(d)
    seed = torch.randn(1, 2, d)
    stream = AttnResStream.start(seed, group_size=2, attn_mix=mix_a, mlp_mix=mix_m)
    assert len(stream.completed) == 1
    stream.add_sublayer(torch.ones_like(seed))
    stream.close_layer()
    assert stream.layers_in_block == 1
    assert stream.partial is not None
    stream.add_sublayer(torch.ones_like(seed))
    stream.close_layer()
    assert stream.layers_in_block == 0
    assert stream.partial is None
    assert len(stream.completed) == 2
    print("AttnResStream block boundary tests passed")
    return True


def test_bitnet_kimi_forward_and_loop_reset() -> bool:
    torch.manual_seed(0)
    cfg = TernaryConfig(
        vocab_size=64,
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=2,
        num_coda_layers=1,
        num_loops=2,
        block_size=4,
        path_window_size=8,
        infini_memory_dim=8,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=False,
        attn_res_mode="kimi",
        attn_res_group_size=1,
    )
    model = BitNetDeep(cfg)
    ids = torch.randint(0, 64, (2, 8))
    logits = model(ids)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()

    # Gradients through depth mixes.
    loss = logits.float().sum()
    loss.backward()
    got = False
    for layer in model.layers:
        if layer.attn_res_mix is not None and layer.attn_res_mix.proj.weight.grad is not None:
            if float(layer.attn_res_mix.proj.weight.grad.abs().sum()) > 0:
                got = True
                break
    assert got, "expected nonzero grad on AttnRes pseudo-query"
    print("BitNetDeep kimi AttnRes forward/grad tests passed")
    return True


def test_sandwich_mode_still_works() -> bool:
    torch.manual_seed(0)
    cfg = TernaryConfig(
        vocab_size=32,
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=2,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=False,
        attn_res_mode="sandwich",
        attn_res_init_scale=1.0,
    )
    block = HybridTransformerBlock(cfg, layer_id=0)
    x = torch.randn(1, 4, 32)
    y = block(x)
    assert y.shape == x.shape
    print("Sandwich residual mode still works")
    return True


if __name__ == "__main__":
    test_depth_attn_mix_is_convex_combination()
    test_stream_closes_depth_blocks()
    test_bitnet_kimi_forward_and_loop_reset()
    test_sandwich_mode_still_works()
    print("All AttnRes tests passed")
