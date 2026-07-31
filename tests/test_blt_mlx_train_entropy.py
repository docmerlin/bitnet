"""Training and round-tripping the entropy patcher.

The saved artefact has to be self-describing. A threshold that does not travel
with the weights eventually gets paired with the wrong ones, and a patcher
running at the wrong cutoff produces plausible-looking patches of the wrong
width -- which degrades the student without ever raising an error.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from blt.config import TernaryBLTConfig
from blt.mlx_data import ByteCorpus, write_byte_corpus
from blt.mlx_entropy_model import MLXByteEntropyModel, load_entropy_model
from blt.mlx_train_entropy import build_parser, main, save_entropy_model, train_entropy_model


def _corpus(tmp_path, size=4096, seq_len=32, seed=0):
    payload = np.random.default_rng(seed).integers(0, 256, size=size, dtype=np.uint8).tobytes()
    path = write_byte_corpus(tmp_path / "corpus.bin", payload)
    return ByteCorpus(path, seq_len=seq_len, offset=4)


def _model(config=None, **kwargs):
    settings = dict(dim=32, num_layers=1, num_heads=4, max_seq_len=64)
    settings.update(kwargs)
    mx.random.seed(0)
    model = MLXByteEntropyModel(config or TernaryBLTConfig(), **settings)
    mx.eval(model.parameters())
    return model


def test_training_reduces_loss(tmp_path):
    # A repeating corpus is memorisable, so loss must fall well inside the budget.
    path = write_byte_corpus(tmp_path / "c.bin", bytes(range(32)) * 128)
    corpus = ByteCorpus(path, seq_len=32, offset=4)
    model = _model()
    history = train_entropy_model(
        model, corpus, steps=60, batch_size=4, learning_rate=3e-3, warmup_steps=5, log_every=0
    )
    first = np.mean([h["loss"] for h in history[:5]])
    last = np.mean([h["loss"] for h in history[-5:]])
    assert np.isfinite(last)
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


def test_bits_per_byte_tracks_loss(tmp_path):
    # The readable form of the objective: 8.0 means nothing learnt about a byte.
    model = _model()
    history = train_entropy_model(
        _model(), _corpus(tmp_path), steps=3, batch_size=2, warmup_steps=1, log_every=0
    )
    for entry in history:
        assert entry["bits_per_byte"] == pytest.approx(entry["loss"] / np.log(2))


def test_save_and_load_round_trip(tmp_path):
    model = _model()
    model.set_threshold(3.75)
    path = save_entropy_model(
        model, tmp_path / "e.safetensors", meta={"dim": 32, "layers": 1, "heads": 4, "max_seq_len": 64}
    )

    restored = load_entropy_model(path, TernaryBLTConfig())
    assert restored.default_threshold == pytest.approx(3.75)
    tokens = mx.array(np.random.default_rng(0).integers(4, 260, size=(2, 24)).astype(np.int32))
    assert np.allclose(np.asarray(restored(tokens)), np.asarray(model(tokens)), atol=1e-5)


def test_loading_without_the_sidecar_is_refused(tmp_path):
    model = _model()
    path = tmp_path / "bare.safetensors"
    mx.save_safetensors(str(path), dict(__import__("mlx.utils", fromlist=["x"]).tree_flatten(model.parameters())))
    with pytest.raises(FileNotFoundError, match="has no bare.json"):
        load_entropy_model(path, TernaryBLTConfig())


def test_sidecar_records_the_calibration(tmp_path):
    model = _model()
    model.set_threshold(2.5)
    save_entropy_model(
        model,
        tmp_path / "e.safetensors",
        meta={"dim": 32, "layers": 1, "heads": 4, "target_patch_size": 4.0},
    )
    meta = json.loads((tmp_path / "e.json").read_text())
    assert meta["threshold"] == pytest.approx(2.5)
    assert meta["target_patch_size"] == 4.0


def test_cli_trains_and_calibrates(tmp_path, capsys):
    payload = (bytes(range(64)) * 200)
    write_byte_corpus(tmp_path / "c.bin", payload)
    output = tmp_path / "entropy.safetensors"
    main(
        [
            "--corpus", str(tmp_path / "c.bin"),
            "--output", str(output),
            "--seq-len", "64",
            "--steps", "40",
            "--batch-size", "4",
            "--warmup-steps", "5",
            "--log-every", "0",
            "--dim", "32",
            "--layers", "1",
            "--heads", "4",
            "--target-patch-size", "4",
        ]
    )
    assert output.is_file()
    meta = json.loads(output.with_suffix(".json").read_text())
    assert meta["dim"] == 32 and meta["layers"] == 1
    assert "threshold" in meta and "mean_patch_width" in meta

    # The saved model must load and patch without any flags being restated.
    restored = load_entropy_model(output, TernaryBLTConfig())
    lengths = restored.predict_patch_lengths(
        mx.array(np.random.default_rng(0).integers(4, 260, size=(2, 64)).astype(np.int32))
    )
    assert np.all(np.asarray(mx.sum(lengths, axis=1)) == 64)


def test_cli_requires_a_corpus():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "1"])
