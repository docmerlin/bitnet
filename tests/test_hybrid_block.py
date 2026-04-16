"""Simple test script for the HybridTransformerBlock.

This tests that every layer contains both Infini-Attention and Attention Residuals (AttnRes).
Run with: python3 tests/test_hybrid_block.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    block = HybridTransformerBlock(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        memory_dim=config.infini_memory_dim,
        init_scale=config.attn_res_init_scale,
        config=config,
    )

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

    # Test 2: Contains both mechanisms
    assert hasattr(block, 'infini_attn'), "Missing InfiniAttention"
    assert hasattr(block, 'attn_res'), "Missing AttentionResidual for attention"
    assert hasattr(block, 'mlp_res'), "Missing AttentionResidual for MLP"
    assert hasattr(block, 'gate'), "Missing learned gate"

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
    isolated_output = block(baseline)
    reset_memory(block)
    isolated_perturbed_output = block(perturbed)
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

    # Test 5: Basic gradient flow
    x.requires_grad_(True)
    reset_memory(block)
    output = block(x, attention_mask)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None, "Gradients not flowing"

    print("✅ All HybridTransformerBlock tests passed!")
    print(f"   - Every layer contains both Infini-Attention and AttnRes")
    print(f"   - Learned gate value: {gate_value.item():.4f}")
    print(f"   - Progressive block growth supported (num_blocks = {block.num_blocks})")
    return True


if __name__ == "__main__":
    test_hybrid_block()
    print("\nHybrid block with both mechanisms in every layer is working correctly.")
