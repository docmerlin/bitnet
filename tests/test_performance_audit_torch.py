"""Correctness tripwires for the September performance audit."""

import copy

import pytest
import torch

from config import TernaryConfig
from model import BitNetDeep
from layers.h_bitlinear import reuse_effective_weights
from layers.rfmoe import RFMoE, rfmoe_padding_waste


@pytest.mark.parametrize("granularity", ["layer", "loop"])
@pytest.mark.parametrize("group_size", [1, 2])
@pytest.mark.parametrize("moe", [False, True])
def test_kimi_checkpoint_output_gradient_and_memory_parity(granularity, group_size, moe):
    torch.manual_seed(5)
    config = TernaryConfig(
        vocab_size=32, hidden_size=16, num_attention_heads=2, head_dim=8,
        intermediate_size=32, num_prelude_layers=2, num_recurrent_layers=3,
        num_coda_layers=2, num_loops=2, path_window_size=4, infini_memory_dim=4,
        block_size=2, use_engram=False, use_hadamard=False,
        attn_res_mode="kimi", attn_res_group_size=group_size,
        use_rfmoe=moe, rfmoe_num_experts=2, rfmoe_rank=2, rfmoe_expert_dim=8,
    )
    baseline = BitNetDeep(config).train()
    checkpointed = copy.deepcopy(baseline)
    checkpointed.gradient_checkpointing = True
    checkpointed.checkpoint_granularity = granularity
    # Two calls exercise an already initialized memory bank as well as reset.
    for reset in (True, False):
        ids = torch.randint(0, 32, (1, 8))
        baseline.zero_grad(set_to_none=True)
        checkpointed.zero_grad(set_to_none=True)
        expected = baseline(ids, reset_memory=reset)
        actual = checkpointed(ids, reset_memory=reset)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, a), (_, b) in zip(baseline.named_parameters(), checkpointed.named_parameters()):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)
        for a, b in zip(baseline.layers, checkpointed.layers):
            for key, value in a.infini_attn.get_memory_state().items():
                torch.testing.assert_close(value, b.infini_attn.get_memory_state()[key])


def test_grouped_reuse_gradients_growth_and_padding():
    torch.manual_seed(7)
    baseline = RFMoE(16, expert_dim=8, num_experts=3, rank=2, theta=0.0)
    cached = copy.deepcopy(baseline)
    x = torch.randn(2, 4, 16)
    expected = baseline(x) + baseline(x * 0.7)
    with reuse_effective_weights():
        actual = cached(x) + cached(x * 0.7)
        layers = [expert.w_up for expert in cached.experts]
        assert cached._grouped_weight(layers, x.dtype) is cached._grouped_weight(layers, x.dtype)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    for a, b in zip(baseline.parameters(), cached.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert rfmoe_padding_waste(cached) == 1.0
    with reuse_effective_weights():
        before = cached._grouped_weight(layers, x.dtype)
        cached.add_expert()
        after = cached._grouped_weight([expert.w_up for expert in cached.experts], x.dtype)
        assert after.shape[0] == before.shape[0] + 1
    cached.theta = float("inf")
    cached(x)
    assert rfmoe_padding_waste(cached) == 0.0


@pytest.mark.parametrize("window", [None, 4])
@pytest.mark.parametrize("new_len", [1, 3, 6])
def test_torch_attention_extend_matches_full_prefix(window, new_len):
    from blt.config import TernaryBLTConfig
    from blt.layers.transformer_block import TernarySelfAttention

    config = TernaryBLTConfig(local_dim=32, global_dim=32, decoder_dim=32)
    layer = TernarySelfAttention(32, 4, config=config, local_window=window, causal=True).eval()
    x = torch.randn(2, 12 + new_len, 32)
    with torch.no_grad():
        _, cache = layer.prefill(x[:, :8])
        _, cache = layer.extend(x[:, 8:12], cache, offset=8)
        actual, _ = layer.extend(x[:, 12:], cache, offset=12)
        torch.testing.assert_close(actual, layer(x)[:, 12:], atol=2e-6, rtol=2e-5)


def test_torch_checkpoint_alias_is_immutable(tmp_path):
    from training.checkpoint import alias_checkpoint
    from utils import atomic_torch_save

    directory = tmp_path / "checkpoints"
    directory.mkdir()
    source = directory / "step_0000001.pt"
    atomic_torch_save({"step": 1}, source)
    alias = alias_checkpoint(tmp_path, source.name, "last.pt")
    assert source.stat().st_ino == alias.stat().st_ino
    atomic_torch_save({"step": 2}, alias)
    assert torch.load(source, weights_only=True)["step"] == 1
    assert torch.load(alias, weights_only=True)["step"] == 2
