"""The BitNet stack wired in as BLT's global transformer.

BLT's global model is a transformer over patch latents, which is what MLXBitNet
is once it stops being fed token ids. These pin the two things the wiring has to
get right: that the stack really can run on embeddings, and that padded patches
are refused rather than silently perturbing the real ones.
"""

import mlx.core as mx
import numpy as np
import pytest

from blt.config import TernaryBLTConfig
from blt.mlx_global import MLXBitNetGlobalTransformer, global_config_for
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_patching import build_uniform_patch_lengths, pad_patch_lengths_to_bucket
from mlx_model import MLXBitNet, MLXBitNetConfig


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(
        local_dim=64,
        global_dim=64,
        decoder_dim=64,
        n_layers_local_encoder=1,
        n_layers_global=4,
        n_layers_local_decoder=1,
        n_heads_local_encoder=4,
        n_heads_global=4,
        n_heads_local_decoder=4,
        n_heads_cross=4,
        local_window=None,
        patch_size=4,
    )
    base.update(overrides)
    return TernaryBLTConfig(**base)


def _model(config=None):
    config = config or _config()
    mx.random.seed(0)
    backbone = MLXBitNetGlobalTransformer(
        config, global_config_for(config, block_size=2, path_window_size=32)
    )
    model = MLXTernaryBLTModel(config, global_transformer=backbone)
    mx.eval(model.parameters())
    return model, backbone


def _tokens(batch=2, seq=64, seed=0):
    return mx.array(np.random.default_rng(seed).integers(4, 260, size=(batch, seq)).astype(np.int32))


def test_embeddings_path_matches_the_token_path():
    # The change that makes the wiring possible: nothing in the stack needs
    # discrete ids except Engram.
    config = MLXBitNetConfig(
        hidden_size=64,
        num_attention_heads=4,
        vocab_size=260,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=1,
        use_engram=False,
        block_size=2,
        path_window_size=32,
    )
    mx.random.seed(0)
    model = MLXBitNet(config)
    mx.eval(model.parameters())
    ids = _tokens(2, 16)
    from_ids = model.hidden_states(ids)
    from_embeds = model.hidden_states(inputs_embeds=model.embedding(ids))
    assert float(mx.max(mx.abs(from_ids - from_embeds))) == 0.0


def test_exactly_one_input_is_required():
    config = MLXBitNetConfig(
        hidden_size=64, num_attention_heads=4, vocab_size=260, use_engram=False, block_size=2
    )
    mx.random.seed(0)
    model = MLXBitNet(config)
    mx.eval(model.parameters())
    with pytest.raises(ValueError, match="exactly one of tokens or inputs_embeds"):
        model.hidden_states()
    with pytest.raises(ValueError, match="exactly one of tokens or inputs_embeds"):
        model.hidden_states(_tokens(1, 8), inputs_embeds=mx.zeros((1, 8, 64)))


def test_engram_cannot_run_on_embeddings():
    # It hashes token n-grams; patches have no ids. Raise instead of hashing
    # whatever integers happen to be lying around.
    config = MLXBitNetConfig(
        hidden_size=64,
        num_attention_heads=4,
        vocab_size=260,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=1,
        use_engram=True,
        engram_layer_ids=(0,),
        block_size=2,
    )
    mx.random.seed(0)
    model = MLXBitNet(config)
    mx.eval(model.parameters())
    assert model.uses_engram
    with pytest.raises(ValueError, match="Engram hashes token n-grams"):
        model.hidden_states(inputs_embeds=mx.zeros((1, 8, 64)))


def test_blt_runs_end_to_end_on_the_bitnet_backbone():
    model, backbone = _model()
    output = model(_tokens())
    mx.eval(output.logits)
    assert output.logits.shape == (2, 64, model.config.vocab_size)
    assert not backbone.global_config.use_engram


def test_global_config_refuses_a_width_mismatch():
    config = _config()
    with pytest.raises(ValueError, match="!= BLT global_dim"):
        MLXBitNetGlobalTransformer(config, global_config_for(config, hidden_size=128))


def test_global_config_refuses_engram():
    config = _config()
    with pytest.raises(ValueError, match="Engram needs token ids"):
        MLXBitNetGlobalTransformer(
            config, global_config_for(config, use_engram=True, engram_layer_ids=(0,))
        )


def test_global_config_refuses_patch_level_mtp():
    config = _config(mtp_depth=2)
    with pytest.raises(ValueError, match="patch latents have no discrete future-token targets"):
        MLXBitNetGlobalTransformer(config, global_config_for(config, mtp_depth=2))


def test_padded_patches_are_refused_not_silently_absorbed():
    # PaTH chunks by count, so its boundaries move with sequence length and
    # padding perturbs the real patches by ~1e-4 -- which the decoder's 4-bit
    # activation quantisation amplifies into ~2e-1 on the logits.
    model, _ = _model()
    padded = pad_patch_lengths_to_bucket(build_uniform_patch_lengths(2, 64, 4), 32)
    with pytest.raises(ValueError, match="cannot take padded patches"):
        model(_tokens(), patch_lengths=padded)


def test_unpadded_patches_are_accepted():
    model, _ = _model()
    lengths = build_uniform_patch_lengths(2, 64, 4)
    output = model(_tokens(), patch_lengths=lengths)
    mx.eval(output.logits)
    assert output.global_hidden.shape[1] == lengths.shape[1]


def test_fixed_patch_count_needs_no_padding():
    # The option that lets this backbone run compiled: a fixed count gives one
    # shape per batch without the zero-padding it cannot tolerate.
    from blt.mlx_entropy_model import MLXByteEntropyModel

    config = _config()
    mx.random.seed(0)
    patcher = MLXByteEntropyModel(config, dim=64, num_layers=1, num_heads=4, max_seq_len=128)
    mx.eval(patcher.parameters())

    shapes, totals = set(), set()
    for seed in range(6):
        tokens = _tokens(2, 64, seed=seed)
        lengths = patcher.predict_patch_lengths(tokens, num_patches=16)
        shapes.add(lengths.shape)
        totals.add(tuple(np.asarray(mx.sum(lengths, axis=1)).tolist()))
        assert bool(mx.all(lengths > 0))  # no padding, so the backbone accepts it
    assert shapes == {(2, 16)}
    assert totals == {(64, 64)}


def test_fixed_patch_count_takes_precedence_over_length_cap(monkeypatch):
    from blt.mlx_entropy_model import MLXByteEntropyModel

    config = _config(max_patch_length=2)
    patcher = MLXByteEntropyModel(config, dim=64, num_layers=1, num_heads=4, max_seq_len=64)
    entropy = mx.array([[10.0, 9.0, 8.0, 7.0] + [0.0] * 12])
    monkeypatch.setattr(patcher, "entropy", lambda _: entropy)

    lengths = patcher.predict_patch_lengths(mx.zeros((1, 16), dtype=mx.int32), num_patches=4)
    mx.eval(lengths)

    assert lengths.shape == (1, 4)
    assert bool(mx.all(lengths > 0))
    assert int(mx.sum(lengths)) == 16


def test_boundaries_by_count_is_uniform_and_ignores_future_entropy():
    from blt.mlx_entropy_model import boundaries_by_count

    entropy = mx.array([[0.1, 9.0, 0.2, 8.0, 0.3, 0.4]])
    # Same shape/count fixes positions; suffix entropy cannot displace a start.
    assert np.asarray(boundaries_by_count(entropy, 3))[0].tolist() == [
        True, False, True, False, True, False
    ]
    changed = mx.array([[0.1, 9.0, 0.2, 80.0, 90.0, 100.0]])
    assert np.array_equal(
        np.asarray(boundaries_by_count(entropy, 3)),
        np.asarray(boundaries_by_count(changed, 3)),
    )


@pytest.mark.parametrize("count", [1, 3, 7])
def test_fixed_count_is_uniform_without_running_entropy(monkeypatch, count):
    from blt.mlx_entropy_model import MLXByteEntropyModel

    patcher = MLXByteEntropyModel(_config(), dim=32, num_layers=1, num_heads=4)

    def fail(_):
        raise AssertionError("Fixed positional patching must not run the entropy model")

    monkeypatch.setattr(patcher, "entropy", fail)
    lengths = np.asarray(patcher.predict_patch_lengths(_tokens(2, 7), num_patches=count))
    assert lengths.shape == (2, count)
    assert np.all(lengths.sum(axis=1) == 7)
    assert lengths.max() - lengths.min() <= 1


def test_boundaries_by_count_rejects_an_impossible_count():
    from blt.mlx_entropy_model import boundaries_by_count

    with pytest.raises(ValueError, match="num_patches must be in"):
        boundaries_by_count(mx.zeros((1, 8)), 9)
