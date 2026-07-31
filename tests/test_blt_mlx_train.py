"""The MLX distillation loop has to actually learn, and route parameters right.

A smoke test that only checks "loss is finite" passes just as happily when the
optimizer is updating nothing. These assert that loss falls on a memorisable
batch, that every parameter receives gradient, and that the embedding is kept
away from MUD -- CMUD's router keys on ``embedding.weight`` and BLT's is
``byte_embeddings.weight``, so the exclusion needs its own test.
"""

import json

import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten

from blt.config import TernaryBLTConfig
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_train import (
    BLTCMUD,
    MLXBLTTrainer,
    TrainingConfig,
    clip_gradients,
    learning_rate_at,
    shifted_labels,
)
from blt.teacher_cache import TeacherCache, TeacherCacheWriter

SEQ, PATCHES, TOP_K = 16, 4, 8


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(
        local_dim=32,
        global_dim=32,
        decoder_dim=32,
        n_layers_local_encoder=1,
        n_layers_global=1,
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


def _cache(tmp_path, config, sequences=4, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(config.offset, config.offset + 32, size=(sequences, SEQ)).astype(np.int32)
    # A teacher that agrees with the data: peak the logit on the next byte.
    logits = rng.standard_normal((sequences, SEQ, config.vocab_size)).astype(np.float32)
    nxt = np.concatenate([tokens[:, 1:], tokens[:, :1]], axis=1)
    np.put_along_axis(logits, nxt[..., None], 10.0, axis=-1)

    with TeacherCacheWriter(
        tmp_path,
        num_sequences=sequences,
        seq_len=SEQ,
        max_patches=PATCHES,
        top_k=TOP_K,
        vocab_size=config.vocab_size,
    ) as writer:
        writer.add(
            tokens=tokens,
            mask=np.ones((sequences, SEQ), bool),
            patch_lengths=np.full((sequences, PATCHES), SEQ // PATCHES, np.int32),
            logits=logits,
        )
    return TeacherCache(tmp_path)


def _trainer(tmp_path, *, cross_attn_k=2, **training_overrides):
    config = _config(cross_attn_k=cross_attn_k)
    mx.random.seed(0)
    model = MLXTernaryBLTModel(config)
    mx.eval(model.parameters())
    settings = dict(steps=30, batch_size=4, learning_rate=2e-3, warmup_steps=2, log_every=0)
    settings.update(training_overrides)
    return MLXBLTTrainer(model, _cache(tmp_path, config), TrainingConfig(**settings))


def test_training_reduces_loss(tmp_path):
    trainer = _trainer(tmp_path)
    history = trainer.train(log=lambda *_: None)
    first = np.mean([h["loss"] for h in history[:5]])
    last = np.mean([h["loss"] for h in history[-5:]])
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


@pytest.mark.parametrize("cross_attn_k", [1, 2])
def test_no_parameter_lacks_gradient(tmp_path, cross_attn_k):
    # A parameter with no gradient is a wiring bug and it survives a
    # loss-goes-down test unnoticed. Nothing should be inert at either k: at k=1
    # the decoder takes the gather path and the query/key projections do not
    # exist; at k>1 the byte chooses among its patch's slots, so they matter.
    trainer = _trainer(tmp_path, cross_attn_k=cross_attn_k)
    _, gradients = trainer._loss_and_grad(trainer.sample_batch())
    mx.eval(gradients)

    dead = sorted(name for name, g in tree_flatten(gradients) if float(mx.sum(mx.abs(g))) == 0.0)
    assert dead == []


def test_encoder_cross_attention_is_not_dead(tmp_path):
    # Contrast with the decoder: patches attend to every byte they contain, so
    # this one is real attention and must learn.
    trainer = _trainer(tmp_path)
    _, gradients = trainer._loss_and_grad(trainer.sample_batch())
    mx.eval(gradients)
    grads = dict(tree_flatten(gradients))
    assert float(mx.sum(mx.abs(grads["local_encoder.patch_cross_attn.q_proj.weight"]))) > 0.0


def test_embedding_is_not_routed_to_mud():
    embedding = mx.zeros((260, 32))
    assert not BLTCMUD._is_mud_parameter("byte_embeddings.weight", embedding)
    # A genuine 2D projection still goes to MUD.
    assert BLTCMUD._is_mud_parameter("local_encoder.blocks.0.attn.q_proj.weight", mx.zeros((32, 32)))
    # RMSNorm scales are 1D and belong to the fallback either way.
    assert not BLTCMUD._is_mud_parameter("local_encoder.output_norm.weight", mx.zeros((32,)))


def test_shifted_labels_drop_the_final_position():
    tokens = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    mask = mx.array([[True, True, True, True]])
    labels, loss_mask = shifted_labels(tokens, mask, pad_id=-1)
    assert np.array_equal(np.asarray(labels)[0, :3], [6, 7, 8])
    # The last position predicts nothing, so it must not contribute.
    assert not bool(loss_mask[0, 3])
    assert np.asarray(loss_mask).sum() == 3


def test_shifted_labels_respect_existing_padding():
    tokens = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    mask = mx.array([[True, True, False, False]])
    _, loss_mask = shifted_labels(tokens, mask, pad_id=-1)
    assert np.array_equal(np.asarray(loss_mask)[0], [True, False, False, False])


def test_gradient_clipping_caps_the_norm():
    gradients = {"a": mx.array([3.0, 4.0]), "b": mx.array([12.0])}
    clipped, norm = clip_gradients(gradients, 1.0)
    assert float(norm) == pytest.approx(13.0)
    total = mx.sqrt(sum(mx.sum(g**2) for _, g in tree_flatten(clipped)))
    assert float(total) == pytest.approx(1.0, abs=1e-5)


def test_gradient_clipping_leaves_small_gradients_alone():
    gradients = {"a": mx.array([0.3, 0.4])}
    clipped, norm = clip_gradients(gradients, 1.0)
    assert float(norm) == pytest.approx(0.5)
    assert np.allclose(np.asarray(clipped["a"]), [0.3, 0.4])


def test_learning_rate_warms_up_then_decays():
    config = TrainingConfig(steps=100, warmup_steps=10, learning_rate=1.0)
    assert learning_rate_at(0, config) == pytest.approx(0.1)
    assert learning_rate_at(9, config) == pytest.approx(1.0)
    assert learning_rate_at(10, config) == pytest.approx(1.0)
    assert learning_rate_at(99, config) < 0.01
    # Monotone decay after warmup.
    after = [learning_rate_at(s, config) for s in range(10, 100)]
    assert all(b <= a + 1e-9 for a, b in zip(after, after[1:]))


def test_checkpoint_round_trips(tmp_path):
    trainer = _trainer(tmp_path / "cache", steps=2)
    trainer.train(log=lambda *_: None)
    trainer.save(tmp_path / "out")

    saved = mx.load(str(tmp_path / "out" / "model.safetensors"))
    live = dict(tree_flatten(trainer.model.parameters()))
    assert set(saved) == set(live)
    for name, value in live.items():
        assert np.allclose(np.asarray(saved[name]), np.asarray(value))


def test_vocab_mismatch_is_refused(tmp_path):
    # A cache dumped from a teacher with a different vocabulary would train the
    # student against top-k indices that mean different bytes.
    from blt.mlx_train import main

    config = _config()
    cache_dir = tmp_path / "cache"
    _cache(cache_dir, config)
    meta = json.loads((cache_dir / "meta.json").read_text())
    meta["vocab_size"] = config.vocab_size + 8
    (cache_dir / "meta.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="student vocab"):
        main(["--teacher-cache", str(cache_dir), "--steps", "1", "--output", str(tmp_path / "out")])
