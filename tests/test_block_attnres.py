"""Basic coverage for ``BlockAttentionResidual``."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TernaryConfig
from layers.block_attnres import BlockAttentionResidual


def test_block_attention_residual() -> bool:
    torch.manual_seed(0)
    config = TernaryConfig(
        vocab_size=1024,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        head_dim=32,
        intermediate_size=512,
        block_size=4,
    )

    block = BlockAttentionResidual(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        block_size=config.block_size,
        config=config,
    )

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    output = block(x, attention_mask)
    assert output.shape == (batch_size, seq_len, config.hidden_size), f"Wrong output shape: {output.shape}"

    masked_attention = attention_mask.clone()
    masked_attention[:, -1] = 0
    additive_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=x.dtype)
    additive_mask[:, :, :, -1] = torch.finfo(x.dtype).min
    key_mask_output = block(x, masked_attention)
    additive_mask_output = block(x, additive_mask)
    assert torch.allclose(
        key_mask_output[:, :-1, :],
        additive_mask_output[:, :-1, :],
        atol=1e-5,
        rtol=1e-4,
    ), "4D additive masks should match equivalent key padding masks on valid query rows"

    block.num_blocks = 1
    padded_x = torch.randn(1, 4, config.hidden_size)
    padded_x[:, 2:, :] = 0.0
    padded_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    padded_output = block(padded_x, padded_mask)
    assert torch.allclose(
        padded_output[:, 2:, :],
        torch.zeros_like(padded_output[:, 2:, :]),
        atol=1e-6,
        rtol=1e-5,
    ), "2D padding masks should suppress padded query rows"

    additive_query_mask = torch.zeros(1, 4, 4, dtype=padded_x.dtype)
    additive_query_mask[:, 2:, :] = -1e4
    additive_query_output = block(padded_x, additive_query_mask)
    assert torch.allclose(
        padded_output,
        additive_query_output,
        atol=1e-6,
        rtol=1e-5,
    ), "3D additive masks should suppress fully masked query rows even with finite negative mask values"

    probe_pos = min(3, seq_len - 1)
    baseline = x.detach().clone()
    perturbed = baseline.clone()
    perturbed[:, 0, :] += 1.0

    block.num_blocks = seq_len  # block size 1: each token only sees itself locally
    isolated_output = block(baseline)
    isolated_perturbed_output = block(perturbed)
    assert torch.allclose(
        isolated_output[:, probe_pos, :],
        isolated_perturbed_output[:, probe_pos, :],
        atol=1e-5,
        rtol=1e-4,
    ), "num_blocks should isolate later tokens when blocks are size 1"

    block.num_blocks = 1  # full causal attention across the sequence
    global_output = block(baseline)
    global_perturbed_output = block(perturbed)
    assert not torch.allclose(
        global_output[:, probe_pos, :],
        global_perturbed_output[:, probe_pos, :],
        atol=1e-5,
        rtol=1e-4,
    ), "num_blocks=1 should restore full causal attention"

    x.requires_grad_(True)
    output = block(x, attention_mask)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None, "Gradients not flowing"

    print("BlockAttentionResidual tests passed")
    return True


if __name__ == "__main__":
    test_block_attention_residual()
