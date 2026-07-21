"""Simple test script for the HybridTransformerBlock.

This tests that every layer contains both Infini-Attention and Attention Residuals (AttnRes).
Run with: python3 tests/test_hybrid_block.py
"""

import torch

from config import TernaryConfig
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
        attn_res_init_scale=0.1,
    )

    block = HybridTransformerBlock(config)
    # Near-identity residual init is for training depth; unscale for path tests.
    with torch.no_grad():
        block.infini_attn.o_proj.weight.mul_(100.0)
        block.ffn_down.weight.mul_(100.0)
        block.attn_res.scale.fill_(1.0)
        block.mlp_res.scale.fill_(1.0)

    # Test 1: Forward pass shape
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    output = block(x, attention_mask)
    assert output.shape == (batch_size, seq_len, config.hidden_size), f"Wrong output shape: {output.shape}"

    masked_attention = attention_mask.clone()
    masked_attention[:, -1] = 0
    additive_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=x.dtype)
    additive_mask[:, :, :, -1] = torch.finfo(x.dtype).min
    block.eval()
    reset_memory(block)
    key_mask_output = block(x, masked_attention)
    reset_memory(block)
    additive_mask_output = block(x, additive_mask)
    assert torch.allclose(
        key_mask_output[:, :-1, :],
        additive_mask_output[:, :-1, :],
        atol=1e-5,
        rtol=1e-4,
    ), "4D additive masks should match equivalent key padding masks on valid query rows"

    padded_x = torch.randn(1, 4, config.hidden_size)
    padded_x[:, 2:, :] = 0.0
    padded_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    block.num_blocks = 1
    reset_memory(block)
    padded_output = block(padded_x, padded_mask)
    assert torch.allclose(
        padded_output[:, 2:, :],
        torch.zeros_like(padded_output[:, 2:, :]),
        atol=1e-6,
        rtol=1e-5,
    ), "2D padding masks should suppress padded query rows"

    additive_query_mask = torch.zeros(1, 4, 4, dtype=padded_x.dtype)
    additive_query_mask[:, 2:, :] = -1e4
    reset_memory(block)
    additive_query_output = block(padded_x, additive_query_mask)
    assert torch.allclose(
        padded_output,
        additive_query_output,
        atol=1e-6,
        rtol=1e-5,
    ), "3D additive masks should suppress fully masked query rows even with finite negative mask values"

    block.train()
    block.num_blocks = config.block_size

    # Test 2: Contains both mechanisms + sandwich norms + 3-stage dense FFN layout
    assert hasattr(block, 'infini_attn'), "Missing InfiniAttention"
    assert hasattr(block, 'attn_res'), "Missing AttentionResidual for attention"
    assert hasattr(block, 'mlp_res'), "Missing AttentionResidual for MLP"
    assert hasattr(block, 'attn_norm') and hasattr(block, 'mlp_norm'), "Missing pre-norms"
    assert hasattr(block.attn_res, 'norm') and hasattr(block.mlp_res, 'norm'), (
        "Missing sandwich post-norms on residual branches"
    )
    assert hasattr(block, 'gate'), "Missing learned gate"
    assert hasattr(block, 'ffn_up') and hasattr(block, 'ffn_mid') and hasattr(block, 'ffn_down'), (
        "Dense FFN should be three HBitLinear layers (up/mid/down)"
    )
    d, inter = config.hidden_size, config.intermediate_size
    assert block.ffn_up.weight.shape == (inter * 2, d), block.ffn_up.weight.shape
    assert block.ffn_mid.weight.shape == (inter, inter), block.ffn_mid.weight.shape
    assert block.ffn_down.weight.shape == (d, inter), block.ffn_down.weight.shape
    eye = torch.eye(inter, dtype=block.ffn_mid.weight.dtype)
    assert torch.allclose(block.ffn_mid.weight.detach().cpu(), eye, atol=1e-5), (
        "ffn_mid must cold-start as identity"
    )

    # Test 3: Learned gate is between 0 and 1
    gate_value = torch.sigmoid(block.gate)
    assert 0.0 <= gate_value.item() <= 1.0, f"Gate value out of range: {gate_value.item()}"

    # Test 4: Progressive block growth
    original_blocks = block.num_blocks
    block.num_blocks = original_blocks * 2
    assert block.num_blocks == original_blocks * 2, "Block growth not working"

    probe_pos = min(3, seq_len - 1)
    baseline = x.detach().clone()
    perturbed = baseline.clone()
    perturbed[:, 0, :] += 1.0

    block.num_blocks = seq_len  # block size 1: each token only sees itself locally
    reset_memory(block)
    isolated_output = block(baseline, update_memory=False)
    reset_memory(block)
    isolated_perturbed_output = block(perturbed, update_memory=False)
    assert torch.allclose(
        isolated_output[:, probe_pos, :],
        isolated_perturbed_output[:, probe_pos, :],
        atol=1e-5,
        rtol=1e-4,
    ), "num_blocks should limit local attention when blocks grow"

    block.num_blocks = 1  # full causal attention across the sequence
    reset_memory(block)
    global_output = block(baseline)
    reset_memory(block)
    global_perturbed_output = block(perturbed)
    assert not torch.allclose(
        global_output[:, probe_pos, :],
        global_perturbed_output[:, probe_pos, :],
        atol=1e-5,
        rtol=1e-4,
    ), "num_blocks=1 should restore full causal attention"

    # Test 5: Gradient flows through input and through ffn_mid (mid is not dead weight)
    x.requires_grad_(True)
    reset_memory(block)
    output = block(x, attention_mask)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None, "Gradients not flowing"
    assert block.ffn_mid.weight.grad is not None, "ffn_mid must participate in backward"
    assert float(block.ffn_mid.weight.grad.abs().sum()) > 0.0, "ffn_mid grad should be nonzero"

    print("✅ All HybridTransformerBlock tests passed!")
    print(f"   - Every layer contains both Infini-Attention and AttnRes")
    print(f"   - Learned gate value: {gate_value.item():.4f}")
    print(f"   - Progressive block growth supported (num_blocks = {block.num_blocks})")
    return True


def test_sandwich_residual_renormalizes_stream() -> bool:
    """post_norm(x + scale * y) bounds residual growth vs branch-only post-norm."""
    from layers.hybrid_block import AttentionResidual

    torch.manual_seed(0)
    hidden = 32
    res = AttentionResidual(hidden, init_scale=1.0, eps=1e-5)
    # Force identity-ish post-norm scale for a clean magnitude check after many adds.
    with torch.no_grad():
        res.norm.weight.fill_(1.0)

    x = torch.randn(2, 4, hidden)
    delta = torch.randn(2, 4, hidden) * 3.0
    # Repeated sandwich residuals should not explode RMS the way unnormalized add does.
    y = x
    for _ in range(16):
        y = res(y, delta)
    rms = y.pow(2).mean(dim=-1).sqrt().mean().item()
    # RMSNorm keeps last-dim RMS near 1 (weight=1); allow small slack.
    assert 0.5 < rms < 2.0, f"sandwich residual stream RMS out of band: {rms}"

    # Contract: output is post-norm of (x + scale * sublayer)
    x0 = torch.randn(1, 2, hidden)
    s0 = torch.randn(1, 2, hidden)
    out = res(x0, s0)
    expected = res.norm(x0 + res.scale * s0)
    assert torch.allclose(out, expected, atol=1e-6), "AttentionResidual must be sandwich post form"
    print("Sandwich residual renormalization tests passed")
    return True


if __name__ == "__main__":
    test_hybrid_block()
    test_sandwich_residual_renormalizes_stream()
    print("\nHybrid block with both mechanisms in every layer is working correctly.")
