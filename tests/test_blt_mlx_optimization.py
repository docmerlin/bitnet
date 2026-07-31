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


@pytest.mark.parametrize("global_backbone", [False, True])
def test_training_survives_the_quantisation_ramp(tmp_path, global_backbone):
    # 4-bit activations from a cold start collapse the model to uniform output
    # after one update and NaN on the next. The ramp is what keeps it finite;
    # without it this diverges at every width and learning rate tried.
    from blt.mlx_global import MLXBitNetGlobalTransformer, global_config_for

    config = _config(local_dim=64, global_dim=128, decoder_dim=64, n_layers_global=4)
    corpus = _corpus(tmp_path, size=1 << 15)
    # 8 patches over SEQ bytes is the ratio that diverged at 4-bit activations.
    mx.random.seed(0)
    backbone = (
        MLXBitNetGlobalTransformer(config, global_config_for(config, block_size=2, path_window_size=32))
        if global_backbone
        else None
    )
    model = MLXTernaryBLTModel(config, global_transformer=backbone)
    mx.eval(model.parameters())
    trainer = MLXBLTTrainer(
        model,
        corpus,
        TrainingConfig(
            steps=20, batch_size=2, learning_rate=1e-3, warmup_steps=3,
            logits_kl=0.0, log_every=0, patches_per_sequence=8,
        ),
    )
    losses = [h["loss"] for h in trainer.train(log=lambda *_: None)]
    assert np.all(np.isfinite(losses)), losses


def test_quantisation_ramps_from_soft_to_full():
    config = TrainingConfig(steps=100, quant_ramp_ratio=0.25)
    mx.random.seed(0)
    model = MLXTernaryBLTModel(_config())
    mx.eval(model.parameters())
    trainer = MLXBLTTrainer.__new__(MLXBLTTrainer)
    trainer.config = config

    start_w, start_a, start_bits = trainer.quantization_at(0)
    end_w, end_a, end_bits = trainer.quantization_at(99)
    # Activations must start unquantised -- that is the whole point.
    assert start_a == 0.0
    assert start_w == pytest.approx(0.25)
    assert start_bits == 16
    assert (end_w, end_a, end_bits) == (1.0, 1.0, 8)


def test_activation_mix_zero_skips_quantisation():
    from blt.mlx_layers import MLXHBitLinear

    layer = MLXHBitLinear(64, 64, config=_config())
    mx.eval(layer.parameters())
    x = mx.array(np.random.default_rng(0).standard_normal((1, 4, 64)).astype(np.float32))
    layer.set_quantization_state(1.0, 0.0, 8)
    unquantised = layer._prepare_input(x)
    layer.set_quantization_state(1.0, 1.0, 4)
    quantised = layer._prepare_input(x)
    assert float(mx.max(mx.abs(unquantised - quantised))) > 1e-4


def test_eight_bit_activations_are_the_default():
    # 4-bit buys no speed -- quantisation here is fake, so the matmul is float x
    # ternary either way -- and diverges where 8-bit does not. Measured on the
    # BitNet backbone at 8 patches over 64 bytes: 4-bit reaches NaN, 8-bit
    # trains. Set 4 only to match a deployment that truly runs 4-bit kernels.
    from blt.config import TernaryBLTConfig
    from blt.mlx_layers import MLXHBitLinear

    config = TernaryBLTConfig()
    assert config.activation_bits == 8
    assert MLXHBitLinear(64, 64, config=config).activation_bits == 8


def test_activation_bits_must_be_representable():
    from blt.config import TernaryBLTConfig

    with pytest.raises(ValueError, match="activation_bits must be at least 2"):
        TernaryBLTConfig(activation_bits=1)
