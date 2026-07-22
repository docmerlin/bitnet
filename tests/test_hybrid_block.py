"""Simple test script for the HybridTransformerBlock.

Tests Infini-Attention + residual path (Kimi Block AttnRes default; sandwich legacy).
Run with: python3 tests/test_hybrid_block.py
"""

import torch

from config import TernaryConfig
from layers.attn_res import AttnResStream, SandwichResidual
from layers.hybrid_block import HybridTransformerBlock


def reset_memory(block: HybridTransformerBlock) -> None:
    block.infini_attn.memory_k.zero_()
    block.infini_attn.memory_v.zero_()


def test_hybrid_block():
    torch.manual_seed(0)
    config = TernaryConfig(
        vocab_size=1024,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        head_dim=32,
        intermediate_size=512,
        block_size=4,
        infini_memory_dim=32,
        attn_res_mode="kimi",
        attn_res_group_size=2,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=False,
    )

    block = HybridTransformerBlock(config, layer_id=0)
    with torch.no_grad():
        block.infini_attn.o_proj.weight.mul_(100.0)
        block.ffn_down.weight.mul_(100.0)

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    stream = AttnResStream.start(
        x,
        group_size=2,
        attn_mix=block.attn_res_mix,
        mlp_mix=block.mlp_res_mix,
    )
    stream = block(stream, attention_mask)
    output = stream.hidden()
    assert output.shape == (batch_size, seq_len, config.hidden_size), f"Wrong output shape: {output.shape}"

    masked_attention = attention_mask.clone()
    masked_attention[:, -1] = 0
    additive_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=x.dtype)
    additive_mask[:, :, :, -1] = torch.finfo(x.dtype).min

    probe_pos = min(3, seq_len - 1)
    baseline = x.detach().clone()
    perturbed = baseline.clone()
    perturbed[:, 0, :] += 1.0

    block.num_blocks = seq_len  # block size 1: each token only sees itself locally
    reset_memory(block)
    s0 = AttnResStream.start(
        baseline, group_size=2, attn_mix=block.attn_res_mix, mlp_mix=block.mlp_res_mix
    )
    isolated_output = block(s0, update_memory=False).hidden()
    reset_memory(block)
    s1 = AttnResStream.start(
        perturbed, group_size=2, attn_mix=block.attn_res_mix, mlp_mix=block.mlp_res_mix
    )
    isolated_perturbed_output = block(s1, update_memory=False).hidden()
    assert torch.allclose(
        isolated_output[:, probe_pos, :],
        isolated_perturbed_output[:, probe_pos, :],
        atol=1e-4,
    ), "Local-only attention should not mix positions"

    assert hasattr(block, "infini_attn"), "Missing InfiniAttention"
    assert block.attn_res_mix is not None and block.mlp_res_mix is not None
    assert hasattr(block, "gate"), "Missing learned gate"
    assert hasattr(block, "ffn_up") and hasattr(block, "ffn_mid") and hasattr(block, "ffn_down")
    d, inter = config.hidden_size, config.intermediate_size
    assert block.ffn_up.weight.shape == (inter * 2, d)
    assert block.ffn_mid.weight.shape == (inter, inter)
    eye = torch.eye(inter, dtype=block.ffn_mid.weight.dtype)
    assert torch.allclose(block.ffn_mid.weight.detach().cpu(), eye, atol=1e-5), (
        "ffn_mid must cold-start as identity"
    )
    assert block.ffn_down.weight.shape == (d, inter)

    gate_value = torch.sigmoid(block.gate)
    assert 0.0 <= gate_value.item() <= 1.0

    # Gradient through mid + AttnRes query
    stream = AttnResStream.start(
        x.detach().requires_grad_(True),
        group_size=2,
        attn_mix=block.attn_res_mix,
        mlp_mix=block.mlp_res_mix,
    )
    out = block(stream, attention_mask).hidden()
    out.sum().backward()
    assert block.ffn_mid.weight.grad is not None
    assert block.attn_res_mix.proj.weight.grad is not None

    print("Hybrid block Kimi AttnRes tests passed")
    print(f"   - Learned gate value: {gate_value.item():.4f}")
    return True


def test_sandwich_residual_renormalizes_stream() -> bool:
    """post_norm(x + scale * y) bounds residual growth vs branch-only post-norm."""
    torch.manual_seed(0)
    hidden = 32
    res = SandwichResidual(hidden, init_scale=1.0, eps=1e-5)
    with torch.no_grad():
        res.norm.weight.fill_(1.0)

    x = torch.randn(2, 4, hidden)
    delta = torch.randn(2, 4, hidden) * 3.0
    y = x
    for _ in range(16):
        y = res(y, delta)
    rms = y.pow(2).mean(dim=-1).sqrt().mean().item()
    assert 0.5 < rms < 2.0, f"sandwich residual stream RMS out of band: {rms}"

    x0 = torch.randn(1, 2, hidden)
    s0 = torch.randn(1, 2, hidden)
    out = res(x0, s0)
    expected = res.norm(x0 + res.scale * s0)
    assert torch.allclose(out, expected, atol=1e-6), "SandwichResidual must be sandwich post form"
    print("Sandwich residual renormalization tests passed")
    return True


if __name__ == "__main__":
    test_hybrid_block()
    test_sandwich_residual_renormalizes_stream()
    print("\nHybrid block residual paths working correctly.")
