"""Whole-model parity between the MLX and torch BLT students.

The layer tests pin each primitive; this pins the assembly -- patch pooling, the
one-patch latent shift into the decoder, and the padding contract. Names match
across the two stacks, so a torch ``state_dict`` loads into the MLX model with no
renaming, and that is exactly how the checkpoint conversion will work.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_unflatten

from blt.config import TernaryBLTConfig
from blt.mlx_model import MLXTernaryBLTModel
from blt.model import TernaryBLTModel

# Loose because activation fake-quantisation makes this a comparison of two
# rounding implementations, not two matmuls. At activation_bits=8 the scale is
# 127x smaller, so x/scale is 127x larger and float differences between torch
# and MLX cross an integer boundary far more often; a flip costs one full
# quantisation step. The drift is concentrated in a handful of positions (the
# rest sit at ~1e-8), which is the signature of boundary flips rather than a
# systematic divergence.
TOLERANCE = 2e-2


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(
        local_dim=64,
        global_dim=64,
        decoder_dim=64,
        n_layers_local_encoder=2,
        n_layers_global=2,
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


def _pair(config):
    torch.manual_seed(0)
    torch_model = TernaryBLTModel(config)
    torch_model.eval()
    mlx_model = MLXTernaryBLTModel(config)
    flat = [(name, mx.array(t.detach().numpy())) for name, t in torch_model.state_dict().items()]
    mlx_model.update(tree_unflatten(flat))
    mx.eval(mlx_model.parameters())
    return torch_model, mlx_model


def _tokens(config, batch=2, seq=16):
    rng = np.random.default_rng(1)
    return rng.integers(config.offset, config.offset + 256, size=(batch, seq)).astype(np.int64)


def _close(expected, actual, tolerance=TOLERANCE):
    actual = np.asarray(actual)
    assert actual.shape == tuple(expected.shape)
    assert np.abs(actual - np.asarray(expected)).max() < tolerance, np.abs(actual - np.asarray(expected)).max()


@pytest.mark.parametrize("cross_attn_k", [1, 2, 4])
@pytest.mark.parametrize(
    "dims",
    [
        dict(),  # uniform widths, no projections
        dict(local_dim=32, global_dim=64, decoder_dim=32),  # every projection live
        dict(local_dim=64, global_dim=32, decoder_dim=64),
    ],
)
def test_full_forward_matches_torch(dims, cross_attn_k):
    # k=1 takes the gather path in both stacks, k>1 real cross-attention over
    # the patch's slots. Both have to agree with torch.
    config = _config(cross_attn_k=cross_attn_k, **dims)
    torch_model, mlx_model = _pair(config)
    tokens = _tokens(config)

    with torch.no_grad():
        expected = torch_model(torch.from_numpy(tokens))
    actual = mlx_model(mx.array(tokens))

    _close(expected.logits, actual.logits)
    _close(expected.encoder_hidden, actual.encoder_hidden)
    _close(expected.encoder_patches, actual.encoder_patches)
    _close(expected.global_hidden, actual.global_hidden)
    _close(expected.decoder_hidden, actual.decoder_hidden)
    assert np.array_equal(np.asarray(actual.patch_ids), expected.patch_ids.numpy().astype(np.int32))


def test_explicit_patch_lengths_match_torch():
    config = _config()
    torch_model, mlx_model = _pair(config)
    tokens = _tokens(config, seq=12)
    lengths = np.array([[5, 3, 4], [2, 6, 4]], dtype=np.int64)

    with torch.no_grad():
        expected = torch_model(torch.from_numpy(tokens), patch_lengths=torch.from_numpy(lengths))
    actual = mlx_model(mx.array(tokens), patch_lengths=mx.array(lengths.astype(np.int32)))
    _close(expected.logits, actual.logits)
    assert np.array_equal(
        np.asarray(actual.patch_lengths), expected.patch_lengths.numpy().astype(np.int32)
    )


def test_padded_batch_matches_torch():
    config = _config(pad_id=300)
    torch_model, mlx_model = _pair(config)
    tokens = _tokens(config, seq=16)
    tokens[1, 11:] = 300  # suffix padding on the second row only

    with torch.no_grad():
        expected = torch_model(torch.from_numpy(tokens))
    actual = mlx_model(mx.array(tokens))
    _close(expected.logits, actual.logits)
    # Padded positions carry no decoder state in either stack.
    assert np.abs(np.asarray(actual.decoder_hidden)[1, 11:]).max() == 0.0


def test_interior_padding_is_refused():
    config = _config(pad_id=300)
    _, mlx_model = _pair(config)
    tokens = _tokens(config, seq=16)
    mask = np.ones((2, 16), dtype=bool)
    mask[0, 5] = False  # hole, not a suffix
    with pytest.raises(ValueError, match="suffix-padded"):
        mlx_model(mx.array(tokens), attention_mask=mx.array(mask))


def test_mismatched_mask_shape_is_refused():
    config = _config()
    _, mlx_model = _pair(config)
    with pytest.raises(ValueError, match="same shape as input_ids"):
        mlx_model(mx.array(_tokens(config)), attention_mask=mx.ones((2, 4), dtype=mx.bool_))


def test_torch_state_dict_loads_without_renaming():
    # The whole conversion story rests on this: identical parameter names and
    # identical [out, in] weight layout in both stacks.
    config = _config()
    torch_model, mlx_model = _pair(config)
    torch_names = set(torch_model.state_dict())
    from mlx.utils import tree_flatten

    # Runtime quantisation state is excluded: weight_mix_value,
    # activation_mix_value and activation_level_pair exist as MLX module state
    # only so mx.compile treats them as graph inputs rather than baking them in
    # as constants. The torch side holds the same values as plain Python
    # attributes. They are scaffolding, not weights, and never transfer.
    runtime_state = ("weight_mix_value", "activation_mix_value", "activation_level_pair")
    mlx_names = {
        name
        for name, _ in tree_flatten(mlx_model.parameters())
        if not name.endswith(runtime_state)
    }
    assert torch_names == mlx_names
