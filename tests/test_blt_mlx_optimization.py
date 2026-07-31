"""Guards on the two changes that made MLX BLT training viable.

Both are the kind of regression that costs an order of magnitude while every
other test stays green, because neither changes any output:

- MUD's whitening block. Left unset it takes each parameter as a single block
  and does a [rows, rows] triangular solve, measured at 8.4s per step against a
  195ms forward+backward -- 96% of training inside the optimizer.
- Patch-count bucketing. ``mx.compile`` caches on shape, and entropy patching
  produces a different patch count nearly every batch, so without rounding the
  width up the graph is rebuilt continuously.
"""

import mlx.core as mx
import numpy as np
import pytest
import pathlib

from blt.config import TernaryBLTConfig
from blt.mlx_data import ByteCorpus, write_byte_corpus
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_patching import build_uniform_patch_lengths, pad_patch_lengths_to_bucket
from blt.mlx_train import MUD_BLOCK_SIZE, BLTCMUD, MLXBLTTrainer, TrainingConfig

SEQ = 64


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


def _corpus(tmp_path, size=8192, seed=0):
    payload = np.random.default_rng(seed).integers(0, 256, size=size, dtype=np.uint8).tobytes()
    return ByteCorpus(write_byte_corpus(tmp_path / "c.bin", payload), seq_len=SEQ, offset=4)


def test_mud_block_size_defaults_to_a_real_block():
    # None means "the whole matrix", which is the 26x-slower path.
    optimizer = BLTCMUD(mud_learning_rate=1e-3, fallback_learning_rate=1e-4, weight_decay=0.0)
    assert MUD_BLOCK_SIZE == 64
    assert optimizer.optimizers[0].block_size == MUD_BLOCK_SIZE
    assert optimizer.optimizers[0].block_size is not None


def test_mud_block_size_is_still_overridable():
    optimizer = BLTCMUD(
        mud_learning_rate=1e-3, fallback_learning_rate=1e-4, weight_decay=0.0, block_size=32
    )
    assert optimizer.optimizers[0].block_size == 32


def test_trainer_passes_the_configured_block_size(tmp_path):
    mx.random.seed(0)
    model = MLXTernaryBLTModel(_config())
    mx.eval(model.parameters())
    trainer = MLXBLTTrainer(
        model,
        _corpus(tmp_path),
        TrainingConfig(steps=1, batch_size=2, logits_kl=0.0, log_every=0, mud_block_size=16),
    )
    assert trainer.optimizer.optimizers[0].block_size == 16


@pytest.mark.parametrize("width,bucket,expected", [(130, 32, 160), (128, 32, 128), (1, 16, 16), (65, 64, 128)])
def test_bucket_rounds_the_patch_axis_up(width, bucket, expected):
    lengths = mx.ones((2, width), dtype=mx.int32)
    padded = pad_patch_lengths_to_bucket(lengths, bucket)
    assert padded.shape == (2, expected)
    # Padding is zero-length patches, so the byte count is untouched.
    assert np.array_equal(np.asarray(mx.sum(padded, axis=1)), np.asarray(mx.sum(lengths, axis=1)))


def test_bucket_is_a_no_op_at_an_exact_multiple():
    lengths = mx.ones((1, 64), dtype=mx.int32)
    assert pad_patch_lengths_to_bucket(lengths, 32).shape == (1, 64)


def test_bad_bucket_is_refused():
    with pytest.raises(ValueError, match="bucket must be positive"):
        pad_patch_lengths_to_bucket(mx.ones((1, 4), dtype=mx.int32), 0)


def test_zero_length_patches_do_not_change_the_model_output():
    # The whole basis for bucketing: padding the patch axis is inert. If this
    # ever stops holding, compiled runs are silently computing something else.
    config = _config()
    mx.random.seed(0)
    model = MLXTernaryBLTModel(config)
    mx.eval(model.parameters())

    tokens = mx.array(np.random.default_rng(1).integers(4, 260, size=(2, SEQ)).astype(np.int32))
    lengths = build_uniform_patch_lengths(2, SEQ, config.patch_size)
    padded = pad_patch_lengths_to_bucket(lengths, 32)
    assert padded.shape[1] > lengths.shape[1]

    plain = model(tokens, patch_lengths=lengths).logits
    with_padding = model(tokens, patch_lengths=padded).logits
    assert np.abs(np.asarray(plain) - np.asarray(with_padding)).max() < 1e-5


def test_compiled_and_uncompiled_training_agree(tmp_path):
    # Compilation must not change the arithmetic. Same seed, same data order,
    # same optimizer -- the loss trajectories should track to float32 noise.
    def run(compile_step):
        mx.random.seed(0)
        model = MLXTernaryBLTModel(_config())
        mx.eval(model.parameters())
        trainer = MLXBLTTrainer(
            model,
            _corpus(tmp_path),
            TrainingConfig(
                steps=8,
                batch_size=2,
                learning_rate=1e-3,
                warmup_steps=1,
                logits_kl=0.0,
                log_every=0,
                compile_step=compile_step,
            ),
        )
        return [h["loss"] for h in trainer.train(log=lambda *_: None)]

    plain, compiled = run(False), run(True)
    assert np.allclose(plain, compiled, rtol=2e-3), f"{plain}\n{compiled}"


def test_compiling_turns_off_the_syncing_validation(tmp_path):
    # The forward validates its mask by reading it, which forces a GPU sync and
    # blocks compilation outright.
    mx.random.seed(0)
    model = MLXTernaryBLTModel(_config())
    mx.eval(model.parameters())
    assert model.validate_inputs is True
    MLXBLTTrainer(
        model,
        _corpus(tmp_path),
        TrainingConfig(steps=1, batch_size=2, logits_kl=0.0, log_every=0, compile_step=True),
    )
    assert model.validate_inputs is False


def test_validation_stays_on_without_compilation(tmp_path):
    mx.random.seed(0)
    model = MLXTernaryBLTModel(_config())
    mx.eval(model.parameters())
    MLXBLTTrainer(
        model,
        _corpus(tmp_path),
        TrainingConfig(steps=1, batch_size=2, logits_kl=0.0, log_every=0, compile_step=False),
    )
    assert model.validate_inputs is True


def test_loss_breakdown_is_available_on_demand(tmp_path):
    # _loss has to stay pure for mx.compile, so the per-term metrics come from a
    # separate call rather than a side effect.
    mx.random.seed(0)
    model = MLXTernaryBLTModel(_config())
    mx.eval(model.parameters())
    trainer = MLXBLTTrainer(
        model, _corpus(tmp_path), TrainingConfig(steps=1, batch_size=2, logits_kl=0.0, log_every=0)
    )
    batch = trainer.sample_batch()
    metrics = trainer.step(batch, 0, breakdown=True)
    assert "hard_ce" in metrics and "loss" in metrics
    assert trainer.step(batch, 1, breakdown=False).keys() == {"loss", "grad_norm", "learning_rate"}
