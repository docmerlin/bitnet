"""TernaryPatchGather must reproduce the cross-attention it replaced.

The decoder's cross-attention mask was one-hot by construction: every byte reads
exactly the patch it belongs to. Softmax over a single permitted key is 1
whatever the query and key projections produce, so ``query_norm``, ``q_proj`` and
``k_proj`` could not influence the output and received exactly zero gradient --
0.5M parameters at production width that never learned anything.

These tests are the evidence for that claim, and the guard on it: if the decoder
ever attends across more than one patch, the equivalence breaks here first.
"""

import numpy as np
import pytest
import torch

from blt.config import TernaryBLTConfig
from blt.layers.cross_attention import TernaryCrossAttention, TernaryPatchGather
from blt.model import TernaryBLTModel
from blt.patching.teacher_patcher import patch_membership_mask
from training.arch_upgrade import (
    filter_retired_decoder_cross_attn_keys,
    is_retired_decoder_cross_attn_key,
)


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(
        local_dim=32,
        global_dim=32,
        decoder_dim=32,
        n_layers_local_encoder=1,
        n_layers_global=1,
        n_layers_local_decoder=2,
        n_heads_local_encoder=4,
        n_heads_global=4,
        n_heads_local_decoder=4,
        n_heads_cross=4,
        local_window=None,
        patch_size=4,
    )
    base.update(overrides)
    return TernaryBLTConfig(**base)


@pytest.mark.parametrize("with_padding", [False, True])
def test_gather_matches_one_hot_cross_attention(with_padding):
    torch.manual_seed(0)
    config = _config()
    kwargs = dict(hidden_dim=32, num_heads=4, config=config)
    attention = TernaryCrossAttention(32, 32, **kwargs).eval()
    gather = TernaryPatchGather(32, 32, **kwargs).eval()

    # Copy across only what the gather kept; the rest exists in `attention` alone.
    shared = {
        name: tensor
        for name, tensor in attention.state_dict().items()
        if name.split(".")[0] in ("kv_norm", "v_proj", "out_proj")
    }
    missing_and_unexpected = gather.load_state_dict(shared, strict=False)
    assert not missing_and_unexpected.missing_keys
    assert not missing_and_unexpected.unexpected_keys

    query = torch.randn(2, 12, 32)
    latents = torch.randn(2, 4, 32)
    patch_ids = torch.arange(4).repeat_interleave(3).unsqueeze(0).repeat(2, 1)
    if with_padding:
        patch_ids[1, 9:] = -1
    valid = patch_ids >= 0

    mask = patch_membership_mask(patch_ids.clamp_min(0), 4, patches_as_queries=False) & valid.unsqueeze(-1)
    with torch.no_grad():
        expected = attention(query, latents, mask=mask)
        actual = gather(query, latents, patch_ids, valid=valid)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_gather_keeps_only_the_projections_that_mattered():
    gather = TernaryPatchGather(32, 32, hidden_dim=32, num_heads=4, config=_config())
    roots = {name.split(".")[0] for name, _ in gather.named_parameters()}
    assert roots == {"kv_norm", "v_proj", "out_proj"}


def test_k1_decoder_carries_no_dead_parameters():
    model = TernaryBLTModel(_config(cross_attn_k=1))
    dead = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("local_decoder.cross_attn_layers")
        and name.rsplit(".", 2)[1] in ("query_norm", "q_proj", "k_proj")
    ]
    assert dead == []


@pytest.mark.parametrize("cross_attn_k", [1, 2, 4])
def test_every_decoder_parameter_receives_gradient(cross_attn_k):
    # At k=1 that holds because the inert projections are gone; at k>1 because
    # the byte now chooses among its patch's slots, so query and key matter.
    # torch floors masked logits at finfo.min rather than -inf, so a genuinely
    # dead parameter reads ~1e-9 here rather than exactly 0 -- hence a threshold.
    torch.manual_seed(0)
    model = TernaryBLTModel(_config(cross_attn_k=cross_attn_k))
    model(torch.randint(4, 36, (2, 16))).logits.square().mean().backward()

    inert = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("local_decoder")
        and (parameter.grad is None or parameter.grad.abs().sum().item() < 1e-7)
    ]
    assert inert == []


@pytest.mark.parametrize("cross_attn_k", [2, 4])
def test_higher_k_actually_changes_the_output(cross_attn_k):
    # Guard against a wiring slip where the extra slots are allocated but the
    # mask still admits only one of them -- the model would train fine and the
    # slots would be silently unused.
    torch.manual_seed(0)
    model = TernaryBLTModel(_config(cross_attn_k=cross_attn_k)).eval()
    tokens = torch.randint(4, 36, (2, 16))
    with torch.no_grad():
        before = model(tokens).logits.clone()
        # Perturb only the slots beyond the first for every patch.
        weight = model.local_decoder.patch_state_proj.weight
        slot_width = weight.size(0) // cross_attn_k
        weight[slot_width:] += 5.0
        after = model(tokens).logits
    assert not torch.allclose(before, after, atol=1e-4)


def test_k1_sheds_the_inert_projections():
    config = TernaryBLTConfig(cross_attn_k=1)
    # 2 square [decoder_dim, decoder_dim] projections plus one norm per layer.
    shed = config.n_layers_local_decoder * (2 * config.decoder_dim**2 + config.decoder_dim)
    assert shed == config.n_layers_local_decoder * 131_328
    # Named parameters rather than a total: an absolute count breaks whenever
    # anything else in the model changes size, and k also rescales
    # patch_state_proj, which is present either way and is not what is shed.
    gathered = dict(TernaryBLTModel(config).named_parameters())
    attended = dict(TernaryBLTModel(TernaryBLTConfig(cross_attn_k=2)).named_parameters())
    shed_names = set(attended) - set(gathered)
    assert shed_names and all("cross_attn_layers" in name for name in shed_names)
    assert sum(attended[name].numel() for name in shed_names) == shed


def test_default_k_restores_real_cross_attention():
    config = TernaryBLTConfig()
    assert config.cross_attn_k == 2
    model = TernaryBLTModel(config)
    roots = {
        name.split(".")[3]
        for name, _ in model.named_parameters()
        if name.startswith("local_decoder.cross_attn_layers")
    }
    assert {"q_proj", "k_proj", "query_norm"} <= roots


def test_non_positive_k_is_refused():
    with pytest.raises(ValueError, match="cross_attn_k must be positive"):
        TernaryBLTConfig(cross_attn_k=0)


@pytest.mark.parametrize(
    "key,retired",
    [
        ("local_decoder.cross_attn_layers.0.q_proj.weight", True),
        ("local_decoder.cross_attn_layers.3.query_norm.weight", True),
        ("local_decoder.cross_attn_layers.1.k_proj.weight", True),
        # Kept tensors of the same module must not be swept up.
        ("local_decoder.cross_attn_layers.0.v_proj.weight", False),
        ("local_decoder.cross_attn_layers.0.out_proj.weight", False),
        # The encoder's cross-attention is real attention and keeps its q/k.
        ("local_encoder.patch_cross_attn.q_proj.weight", False),
        # Self-attention everywhere is untouched.
        ("local_decoder.blocks.0.attn.q_proj.weight", False),
    ],
)
def test_retired_key_filter_is_precise(key, retired):
    assert is_retired_decoder_cross_attn_key(key) is retired
    assert bool(filter_retired_decoder_cross_attn_keys([key])) is retired


def test_old_checkpoint_state_dict_loads_past_retired_keys():
    # Resume has to survive the architecture bump: the old tensors are reported
    # as unexpected and dropped, and nothing legitimate goes missing.
    config = _config()
    torch.manual_seed(0)
    model = TernaryBLTModel(config)
    old_state = dict(model.state_dict())
    for layer in range(config.n_layers_local_decoder):
        prefix = f"local_decoder.cross_attn_layers.{layer}"
        old_state[f"{prefix}.query_norm.weight"] = torch.ones(config.decoder_dim)
        old_state[f"{prefix}.q_proj.weight"] = torch.zeros(config.decoder_dim, config.decoder_dim)
        old_state[f"{prefix}.k_proj.weight"] = torch.zeros(config.decoder_dim, config.decoder_dim)

    incompatible = TernaryBLTModel(config).load_state_dict(old_state, strict=False)
    assert not incompatible.missing_keys
    assert set(incompatible.unexpected_keys) == set(
        filter_retired_decoder_cross_attn_keys(incompatible.unexpected_keys)
    )
