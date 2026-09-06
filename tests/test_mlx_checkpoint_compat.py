"""Checkpoint save/load: what is stored, and what old files still load."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_optim import CMUD
from mlx_train import (
    _RUNTIME_QUANT_NAMES,
    config_from_saved,
    load_checkpoint,
    migrate_two_group_optimizer_state,
    save_checkpoint,
)


def _config(**overrides) -> MLXBitNetConfig:
    base = dict(
        vocab_size=256, hidden_size=64, num_attention_heads=4, intermediate_size=128,
        num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1, num_loops=1,
        use_engram=False, mtp_depth=0,
    )
    base.update(overrides)
    return MLXBitNetConfig(**base)


def _optimizer() -> CMUD:
    return CMUD(mud_learning_rate=1e-3, fallback_learning_rate=3e-4, weight_decay=0.0)


def test_runtime_quantisation_state_is_not_written_to_checkpoints(tmp_path) -> None:
    # Old builds stored mix/level arrays as module state. New models do not, and
    # the save path still skips those names so leftover keys are not rewritten.
    config = _config()
    model = MLXBitNet(config)
    optimizer = _optimizer()
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters())

    path = save_checkpoint(tmp_path, model, optimizer, config, {"step": 1}, "ckpt")
    stored = mx.load(str(path))
    assert not [key for key in stored if key.endswith(_RUNTIME_QUANT_NAMES)]


def test_a_checkpoint_round_trips(tmp_path) -> None:
    config = _config()
    model = MLXBitNet(config)
    optimizer = _optimizer()
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters())
    path = save_checkpoint(tmp_path, model, optimizer, config, {"step": 3}, "ckpt")

    restored = MLXBitNet(config)
    restored_optimizer = _optimizer()
    restored_optimizer.init(restored.trainable_parameters())
    mx.eval(restored.parameters())
    load_checkpoint(path, restored, restored_optimizer)

    before = dict(tree_flatten(model.parameters()))
    after = dict(tree_flatten(restored.parameters()))
    for key, value in before.items():
        # Infini memory buffers start empty and are excluded from the file too.
        if key.endswith(_RUNTIME_QUANT_NAMES) or value.size == 0:
            continue
        assert float(mx.max(mx.abs(value - after[key]))) == 0.0, key


def test_cli_resume_keeps_legacy_vocab_dimensions(monkeypatch, tmp_path):
    import mlx_train

    checkpoint = tmp_path / "legacy.safetensors"
    checkpoint.with_suffix(".json").write_text(json.dumps({
        "model_config": asdict(_config(vocab_size=32768)),
        "training_args": {"tokenizer_max_patch_size": 8},
    }))

    class Tokenizer:
        def __init__(self, **kwargs):
            assert kwargs == {"max_patch_size": 8, "vocab_size_target": 32768}

        def __len__(self):
            return 1067

    class ConfigVerified(Exception):
        pass

    def build_model(config):
        assert config.vocab_size == 32768
        raise ConfigVerified

    monkeypatch.setattr(mlx_train, "HierarchicalTokenizer", Tokenizer)
    monkeypatch.setattr(mlx_train, "MLXBitNet", build_model)
    monkeypatch.setattr("sys.argv", ["mlx_train", "--resume-from", str(checkpoint),
                                    "--output-dir", str(tmp_path)])
    with pytest.raises(ConfigVerified):
        mlx_train.main()


def test_config_from_saved_drops_retired_fields() -> None:
    saved = {
        "vocab_size": 256,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "use_ffn_mid": True,
        "use_mamba3_layers": False,
        "mamba_layer_period": 3,
        "mamba_d_state": 64,
        "mamba_expand": 2,
        "mamba_headdim": 32,
        "mamba_d_conv": 4,
        "mamba_dt_min": 0.001,
        "mamba_dt_max": 0.1,
        "mamba_a_floor": 1e-4,
        "use_mamba_scan_kernel": True,
    }
    config = config_from_saved(saved)
    assert config.hidden_size == 64
    assert not hasattr(config, "use_ffn_mid")
    assert not hasattr(config, "use_mamba3_layers")


def test_config_from_saved_drops_retired_activation_fields() -> None:
    config = config_from_saved(
        {
            "vocab_size": 256,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_4bit_activations": False,
            "quantize_activations": True,
            "activation_bits": 4,
            "activation_dtype": "compute",
        }
    )
    assert not hasattr(config, "use_4bit_activations")
    assert not hasattr(config, "quantize_activations")
    assert not hasattr(config, "activation_bits")
    assert not hasattr(config, "activation_dtype")


def test_config_from_saved_still_rejects_an_unknown_field() -> None:
    # Only *known* retired names are dropped; a typo or real drift must fail.
    with pytest.raises(TypeError):
        config_from_saved({"vocab_size": 256, "hidden_size": 64, "nonsense_field": 1})


def test_two_group_optimizer_state_migrates_to_three() -> None:
    # CMUD was [MUD, C-Lion]; it is now [MUD, embeddings, everything else].
    config = _config()
    model = MLXBitNet(config)
    optimizer = _optimizer()
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters())
    expected = dict(tree_flatten(optimizer.state))

    # Fold the two C-Lion groups back into one, as the old build wrote them.
    old_style = {}
    for key, value in expected.items():
        if key.startswith("states.2."):
            name = key[len("states.2."):]
            # Per-group scalars collapse rather than collide.
            old_style["states.1." + name] = value
        else:
            old_style[key] = value
    assert {k.split(".")[1] for k in old_style if k.startswith("states.")} == {"0", "1"}

    migrated = migrate_two_group_optimizer_state(old_style, expected, optimizer)
    assert migrated.keys() == expected.keys()


def test_migration_leaves_a_current_checkpoint_alone() -> None:
    config = _config()
    model = MLXBitNet(config)
    optimizer = _optimizer()
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters())
    expected = dict(tree_flatten(optimizer.state))
    assert migrate_two_group_optimizer_state(dict(expected), expected, optimizer) == expected
