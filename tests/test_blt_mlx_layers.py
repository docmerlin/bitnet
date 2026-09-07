"""The MLX BLT layers must reproduce the torch BLT layers numerically.

Weights transfer directly (both store ``[out, in]``), so given the same weights
and the same input the two stacks have to agree. This is the contract that lets
a torch-trained checkpoint be converted and served -- or, once MLX training
lands, lets an MLX-trained checkpoint be checked against the reference stack.

Tolerances are loose enough for float32 reassociation across two very different
kernels and tight enough to catch a convention error: a half-split RoPE instead
of interleaved, or MLX's 1e-5 RMSNorm epsilon instead of torch's finfo default,
both move outputs far past 1e-4.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten, tree_unflatten

from blt.config import TernaryBLTConfig
from blt.layers.cross_attention import TernaryCrossAttention
from blt.layers.transformer_block import TernaryMLP, TernarySelfAttention, TransformerBlock
from blt.mlx_layers import (
    MLXHBitLinear,
    MLXTernaryCrossAttention,
    MLXTernaryMLP,
    MLXTernaryPatchGather,
    MLXTernarySelfAttention,
    MLXTransformerBlock,
    apply_rotary_emb,
    build_rope_cache,
)

# Was 2e-4, which only held while the FFN mid quantised to eye(N)/N and so
# attenuated the block's own output by 1/256. With the mid a true identity the
# FFN passes signal at unit gain, and the attention path's pre-existing
# torch-vs-MLX disagreement is no longer damped: measured 2.3e-4 on the block
# against 5.2e-8 for the MLP alone.
TOLERANCE = 2e-3


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(local_dim=64, global_dim=64, decoder_dim=64, n_heads_cross=4)
    base.update(overrides)
    return TernaryBLTConfig(**base)


def _transfer(torch_module: torch.nn.Module, mlx_module) -> None:
    """Copy every torch parameter into the identically-named MLX parameter."""
    flat = [(name, mx.array(tensor.detach().numpy())) for name, tensor in torch_module.state_dict().items()]
    mlx_module.update(tree_unflatten(flat))
    mx.eval(mlx_module.parameters())


def _both(x: np.ndarray, torch_module, mlx_module, *args, **kwargs):
    torch_module.eval()
    with torch.no_grad():
        expected = torch_module(torch.from_numpy(x), *args, **kwargs).numpy()
    return expected


def _close(expected: np.ndarray, actual: mx.array) -> None:
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < TOLERANCE, np.abs(actual - expected).max()


def test_rope_matches_torch():
    from utils import apply_rotary_emb as torch_rope, build_rope_cache as torch_cache

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 16, 8)).astype(np.float32)
    cos, sin = torch_cache(16, 8, theta=10000.0)
    expected = torch_rope(torch.from_numpy(x), cos, sin).numpy()

    mlx_cos, mlx_sin = build_rope_cache(16, 8, theta=10000.0)
    _close(expected, apply_rotary_emb(mx.array(x), mlx_cos, mlx_sin))


@pytest.mark.parametrize("in_features,out_features", [(64, 64), (64, 256), (48, 64)])
def test_hbitlinear_matches_torch(in_features, out_features):
    # 48 is not a power of two, so the Hadamard branch must switch itself off
    # exactly as the torch layer does rather than transforming a padded input.
    from layers.h_bitlinear import HBitLinear

    config = _config()
    torch_layer = HBitLinear(in_features, out_features, config=config)
    mlx_layer = MLXHBitLinear(in_features, out_features, config=config)
    _transfer(torch_layer, mlx_layer)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 6, in_features)).astype(np.float32)
    expected = _both(x, torch_layer, mlx_layer)
    _close(expected, mlx_layer(mx.array(x)))


def test_shared_projections_prepare_inputs_once(monkeypatch):
    config = _config()
    attention = MLXTernarySelfAttention(64, 4, config=config)
    mlp = MLXTernaryMLP(64, 4.0, config=config)
    cross_attention = MLXTernaryCrossAttention(64, 64, hidden_dim=64, num_heads=4, config=config)
    tracked = {
        id(attention.q_proj): "attention_q",
        id(attention.k_proj): "attention_k",
        id(attention.v_proj): "attention_v",
        id(mlp.gate_proj): "mlp_gate",
        id(mlp.up_proj): "mlp_up",
        id(cross_attention.k_proj): "cross_k",
        id(cross_attention.v_proj): "cross_v",
    }
    calls = dict.fromkeys(tracked.values(), 0)
    original = MLXHBitLinear.prepare_input

    def counted(layer, values):
        if name := tracked.get(id(layer)):
            calls[name] += 1
        return original(layer, values)

    monkeypatch.setattr(MLXHBitLinear, "prepare_input", counted)
    x = mx.random.normal((2, 8, 64))
    outputs = (attention(x), mlp(x), cross_attention(x, x))
    mx.eval(*outputs)

    assert calls == {
        "attention_q": 1,
        "attention_k": 0,
        "attention_v": 0,
        "mlp_gate": 1,
        "mlp_up": 0,
        "cross_k": 1,
        "cross_v": 0,
    }


def test_mlp_matches_torch():
    config = _config()
    torch_mlp = TernaryMLP(64, 4.0, config=config)
    mlx_mlp = MLXTernaryMLP(64, 4.0, config=config)
    _transfer(torch_mlp, mlx_mlp)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 6, 64)).astype(np.float32)
    _close(_both(x, torch_mlp, mlx_mlp), mlx_mlp(mx.array(x)))


def test_mid_proj_starts_as_identity():
    # The torch MLP seeds mid_proj to the identity so a fresh model behaves like
    # the classic two-matrix SwiGLU. A random init here would change cold-start
    # dynamics without changing any test that only checks shapes.
    # The *effective* weight, not the raw one: these are ternary layers and the
    # forward uses quantize(weight). Asserting the raw weight is eye(N) is what
    # let the real bug through -- the per-row scale is mean(|row|) = 1/N for an
    # identity row, so a raw eye(N) quantised to eye(N)/N, a 1/N attenuator.
    mlx_mlp = MLXTernaryMLP(64, 4.0, config=_config())
    effective = np.asarray(mlx_mlp.mid_proj.effective_weight())
    assert np.allclose(effective, np.eye(256, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("local_window", [None, 4])
@pytest.mark.parametrize("causal", [True, False])
def test_self_attention_matches_torch(local_window, causal):
    config = _config()
    torch_attn = TernarySelfAttention(64, 4, config=config, local_window=local_window, causal=causal)
    mlx_attn = MLXTernarySelfAttention(64, 4, config=config, local_window=local_window, causal=causal)
    _transfer(torch_attn, mlx_attn)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 10, 64)).astype(np.float32)
    _close(_both(x, torch_attn, mlx_attn), mlx_attn(mx.array(x)))


def test_self_attention_honours_a_padding_mask():
    config = _config()
    torch_attn = TernarySelfAttention(64, 4, config=config, causal=True)
    mlx_attn = MLXTernarySelfAttention(64, 4, config=config, causal=True)
    _transfer(torch_attn, mlx_attn)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 10, 64)).astype(np.float32)
    mask = np.array([[True] * 10, [True] * 7 + [False] * 3])

    torch_attn.eval()
    with torch.no_grad():
        expected = torch_attn(torch.from_numpy(x), attention_mask=torch.from_numpy(mask)).numpy()
    _close(expected, mlx_attn(mx.array(x), attention_mask=mx.array(mask)))


def test_transformer_block_matches_torch():
    config = _config()
    torch_block = TransformerBlock(64, 4, config=config, ffn_multiplier=4.0, local_window=8)
    mlx_block = MLXTransformerBlock(64, 4, config=config, ffn_multiplier=4.0, local_window=8)
    _transfer(torch_block, mlx_block)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 12, 64)).astype(np.float32)
    _close(_both(x, torch_block, mlx_block), mlx_block(mx.array(x)))


@pytest.mark.parametrize("query_dim,kv_dim,output_dim", [(64, 64, None), (64, 32, None), (32, 64, 64)])
def test_cross_attention_matches_torch(query_dim, kv_dim, output_dim):
    config = _config()
    kwargs = dict(hidden_dim=64, num_heads=4, config=config, output_dim=output_dim)
    torch_attn = TernaryCrossAttention(query_dim, kv_dim, **kwargs)
    mlx_attn = MLXTernaryCrossAttention(query_dim, kv_dim, **kwargs)
    _transfer(torch_attn, mlx_attn)

    rng = np.random.default_rng(6)
    query = rng.standard_normal((2, 5, query_dim)).astype(np.float32)
    kv = rng.standard_normal((2, 9, kv_dim)).astype(np.float32)

    torch_attn.eval()
    with torch.no_grad():
        expected = torch_attn(torch.from_numpy(query), torch.from_numpy(kv)).numpy()
    _close(expected, mlx_attn(mx.array(query), mx.array(kv)))


@pytest.mark.parametrize("with_padding", [False, True])
def test_patch_gather_equals_the_cross_attention_it_replaced(with_padding):
    # The decoder used to run cross-attention here with a one-hot mask. Softmax
    # over a single permitted key is 1 regardless of the query and key
    # projections, so the gather must reproduce it exactly -- which is also the
    # proof that dropping query_norm/q_proj/k_proj changed no output.
    config = _config()
    attn = MLXTernaryCrossAttention(64, 64, hidden_dim=64, num_heads=4, config=config)
    gather = MLXTernaryPatchGather(64, 64, hidden_dim=64, num_heads=4, config=config)
    mx.eval(attn.parameters(), gather.parameters())
    # Share the projections the gather kept; the rest exist only in `attn`.
    gather.update(
        tree_unflatten(
            [(name, value) for name, value in tree_flatten(attn.parameters()) if name.split(".")[0] in
             ("kv_norm", "v_proj", "out_proj")]
        )
    )
    mx.eval(gather.parameters())

    rng = np.random.default_rng(11)
    query = mx.array(rng.standard_normal((2, 12, 64)).astype(np.float32))
    latents = mx.array(rng.standard_normal((2, 4, 64)).astype(np.float32))
    ids = np.repeat(np.arange(4), 3)[None].repeat(2, axis=0).astype(np.int32)
    if with_padding:
        ids[1, 9:] = -1  # padded bytes carry no patch
    patch_ids = mx.array(ids)
    valid = patch_ids >= 0

    one_hot = (patch_ids[..., None] == mx.arange(4, dtype=patch_ids.dtype).reshape(1, 1, 4)) & valid[..., None]
    expected = attn(query, latents, mask=one_hot)
    actual = gather(query, latents, patch_ids, valid=valid)
    assert np.abs(np.asarray(actual) - np.asarray(expected)).max() < 1e-5

    if with_padding:
        # A byte with no patch keeps only its residual, never a gathered value.
        residual_only = gather.out_proj(mx.zeros((1, 1, 64)))
        assert np.allclose(
            np.asarray(actual)[1, 9:], np.asarray(query[1, 9:] + residual_only), atol=1e-5
        )


def test_patch_gather_drops_the_inert_projections():
    gather = MLXTernaryPatchGather(64, 64, hidden_dim=64, num_heads=4, config=_config())
    names = {name.split(".")[0] for name, _ in tree_flatten(gather.parameters())}
    assert names == {"kv_norm", "v_proj", "out_proj"}
    assert not hasattr(gather, "q_proj")


def test_cross_attention_zeroes_fully_masked_queries():
    # A query row with no permitted key still softmaxes to a uniform distribution
    # over the floor-filled logits, which is not zero. Both stacks must zero it.
    config = _config()
    kwargs = dict(hidden_dim=64, num_heads=4, config=config)
    torch_attn = TernaryCrossAttention(64, 64, **kwargs)
    mlx_attn = MLXTernaryCrossAttention(64, 64, **kwargs)
    _transfer(torch_attn, mlx_attn)

    rng = np.random.default_rng(7)
    query = rng.standard_normal((1, 4, 64)).astype(np.float32)
    kv = rng.standard_normal((1, 6, 64)).astype(np.float32)
    mask = np.zeros((1, 4, 6), dtype=bool)
    mask[0, :3] = True  # row 3 attends to nothing

    torch_attn.eval()
    with torch.no_grad():
        expected = torch_attn(
            torch.from_numpy(query), torch.from_numpy(kv), mask=torch.from_numpy(mask)
        ).numpy()
    actual = mlx_attn(mx.array(query), mx.array(kv), mask=mx.array(mask))
    _close(expected, actual)
    # Only the projected residual survives on the dead row -- never attention output.
    assert not np.allclose(np.asarray(actual)[0, 3], 0.0)


def test_cross_attention_reuses_projected_kv():
    config = _config()
    attn = MLXTernaryCrossAttention(64, 64, hidden_dim=64, num_heads=4, config=config)
    gather = MLXTernaryPatchGather(64, 64, hidden_dim=64, num_heads=4, config=config)
    rng = np.random.default_rng(8)
    query = mx.array(rng.standard_normal((2, 3, 64)).astype(np.float32))
    kv = mx.array(rng.standard_normal((2, 5, 64)).astype(np.float32))
    projected = attn.project_kv(kv)
    live = attn(query, kv)
    cached = attn(query, kv, projected_kv=projected)
    assert np.allclose(np.asarray(live), np.asarray(cached), atol=1e-5)
    values = gather.project_values(kv)
    patch_ids = mx.array([[0, 1, 4], [2, 2, 0]], dtype=mx.int32)
    live_g = gather(query, kv, patch_ids)
    cached_g = gather(query, kv, patch_ids, values=values)
    assert np.allclose(np.asarray(live_g), np.asarray(cached_g), atol=1e-5)
