"""Training the BLT student from raw bytes, with no teacher anywhere.

The distillation path is now one of two sources. These cover the other: a
memory-mapped byte corpus, entropy patching from a trained byte LM, and the loop
that ties them together. The guards matter as much as the happy path -- asking
for a KL term against a corpus that has no teacher should fail loudly rather
than quietly train on cross-entropy alone.
"""

import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten

from blt.config import TernaryBLTConfig
from blt.mlx_data import ByteCorpus, write_byte_corpus
from blt.mlx_entropy_model import MLXByteEntropyModel, calibrate_threshold
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_train import MLXBLTTrainer, TrainingConfig

SEQ = 32


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


def _corpus_file(tmp_path, size=4096, seed=0, name="corpus.bin"):
    payload = np.random.default_rng(seed).integers(0, 256, size=size, dtype=np.uint8).tobytes()
    return write_byte_corpus(tmp_path / name, payload)


def test_corpus_maps_bytes_into_the_vocabulary(tmp_path):
    path = write_byte_corpus(tmp_path / "abc.bin", bytes([0, 1, 255, 7, 9, 3, 4, 5]))
    corpus = ByteCorpus(path, seq_len=4, offset=4)
    batch = corpus.batch([0])
    # Byte value + offset, and every position is real.
    assert batch["tokens"][0].tolist() == [4, 5, 259, 11]
    assert batch["mask"].all()
    assert batch["tokens"].dtype == np.int32


def test_corpus_chunks_without_overlap_by_default(tmp_path):
    path = write_byte_corpus(tmp_path / "c.bin", bytes(range(20)))
    corpus = ByteCorpus(path, seq_len=5)
    assert len(corpus) == 4
    tokens = corpus.batch(range(4))["tokens"]
    assert tokens[0].tolist() == [4, 5, 6, 7, 8]
    assert tokens[1].tolist() == [9, 10, 11, 12, 13]


def test_stride_is_opt_in(tmp_path):
    path = write_byte_corpus(tmp_path / "c.bin", bytes(range(20)))
    assert len(ByteCorpus(path, seq_len=5, stride=1)) == 16


def test_sequences_never_straddle_two_files(tmp_path):
    # A sequence spanning two unrelated documents teaches a transition that does
    # not exist, so the unfillable tail of each file is dropped instead.
    first = write_byte_corpus(tmp_path / "a.bin", bytes([1] * 7))
    second = write_byte_corpus(tmp_path / "b.bin", bytes([2] * 7))
    corpus = ByteCorpus([first, second], seq_len=5)
    assert len(corpus) == 2
    tokens = corpus.batch([0, 1])["tokens"]
    assert set(tokens[0].tolist()) == {5}
    assert set(tokens[1].tolist()) == {6}


def test_short_files_are_skipped(tmp_path):
    tiny = write_byte_corpus(tmp_path / "tiny.bin", b"ab")
    usable = write_byte_corpus(tmp_path / "ok.bin", bytes(range(20)))
    assert len(ByteCorpus([tiny, usable], seq_len=10)) == 2


def test_corpus_with_nothing_usable_is_refused(tmp_path):
    tiny = write_byte_corpus(tmp_path / "tiny.bin", b"ab")
    with pytest.raises(ValueError, match="holds 10 bytes"):
        ByteCorpus(tiny, seq_len=10)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus file not found"):
        ByteCorpus(tmp_path / "nope.bin", seq_len=4)


def test_degenerate_seq_len_is_refused(tmp_path):
    path = _corpus_file(tmp_path)
    with pytest.raises(ValueError, match="seq_len must exceed 1"):
        ByteCorpus(path, seq_len=1)


def test_corpus_index_is_per_file_not_per_sequence(tmp_path):
    paths = [write_byte_corpus(tmp_path / f"{i}.bin", bytes(range(40))) for i in range(3)]
    corpus = ByteCorpus(paths, seq_len=8, stride=2)
    assert not hasattr(corpus, "_all_starts")
    assert not hasattr(corpus, "_all_files")
    assert corpus._counts.shape == (3,)
    assert corpus._cumulative.shape == (3,)
    # 40-byte file, seq 8, stride 2 -> (40-8)//2+1 = 17 sequences each.
    assert corpus._counts.tolist() == [17, 17, 17]
    assert len(corpus) == 51


def test_corpus_searchsorted_matches_explicit_file_offsets(tmp_path):
    first = write_byte_corpus(tmp_path / "a.bin", bytes(range(20)))
    second = write_byte_corpus(tmp_path / "b.bin", bytes(range(50, 80)))
    seq_len, stride, offset = 5, 3, 4
    corpus = ByteCorpus([first, second], seq_len=seq_len, stride=stride, offset=offset)
    files = [np.frombuffer(first.read_bytes(), dtype=np.uint8), np.frombuffer(second.read_bytes(), dtype=np.uint8)]
    expected_rows = []
    for data in files:
        for start in range(0, data.size - seq_len + 1, stride):
            expected_rows.append((data[start : start + seq_len].astype(np.int32) + offset).tolist())
    assert len(corpus) == len(expected_rows)
    order = [0, len(corpus) - 1, 1, len(corpus) // 2]
    tokens = corpus.batch(order)["tokens"]
    for row, index in enumerate(order):
        assert tokens[row].tolist() == expected_rows[index]


def _trainer(tmp_path, *, patcher=None, **overrides):
    config = _config()
    mx.random.seed(0)
    model = MLXTernaryBLTModel(config)
    mx.eval(model.parameters())
    corpus = ByteCorpus(_corpus_file(tmp_path), seq_len=SEQ, offset=config.offset)
    settings = dict(
        steps=30, batch_size=4, learning_rate=2e-3, warmup_steps=2, log_every=0, logits_kl=0.0
    )
    settings.update(overrides)
    return MLXBLTTrainer(model, corpus, TrainingConfig(**settings), patcher=patcher)


def test_from_scratch_training_reduces_loss(tmp_path):
    trainer = _trainer(tmp_path)
    history = trainer.train(log=lambda *_: None)
    first = np.mean([h["loss"] for h in history[:5]])
    last = np.mean([h["loss"] for h in history[-5:]])
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


def test_from_scratch_batches_carry_no_teacher_fields(tmp_path):
    trainer = _trainer(tmp_path)
    batch = trainer.sample_batch()
    assert not trainer.has_teacher
    assert "topk_indices" not in batch
    assert "logits_kl" not in trainer._metrics if hasattr(trainer, "_metrics") else True
    # Patch lengths still have to be supplied, and still sum to the sequence.
    assert np.all(np.asarray(mx.sum(batch["patch_lengths"], axis=1)) == SEQ)


def test_kl_against_a_corpus_is_refused(tmp_path):
    with pytest.raises(ValueError, match="logits_kl > 0 needs a TeacherCache"):
        _trainer(tmp_path, logits_kl=1.0)


def test_corpus_training_needs_hard_ce(tmp_path):
    with pytest.raises(ValueError, match="needs hard_ce > 0"):
        _trainer(tmp_path, hard_ce=0.0, logits_kl=0.0)


def test_entropy_patcher_drives_the_patch_lengths(tmp_path):
    config = _config()
    mx.random.seed(0)
    patcher = MLXByteEntropyModel(config, dim=32, num_layers=1, num_heads=4, max_seq_len=128)
    mx.eval(patcher.parameters())
    corpus_tokens = mx.array(
        np.random.default_rng(0).integers(4, 260, size=(4, SEQ)).astype(np.int32)
    )
    calibrate_threshold(patcher, corpus_tokens, target_patch_size=4.0)

    trainer = _trainer(tmp_path, patcher=patcher)
    lengths = trainer.sample_batch()["patch_lengths"]
    assert np.all(np.asarray(mx.sum(lengths, axis=1)) == SEQ)
    widths = np.asarray(lengths)
    # Entropy patching is content-dependent, so widths must not all be identical
    # the way the uniform fallback's are.
    assert len(set(widths[widths > 0].tolist())) > 1


def test_uniform_fallback_is_used_without_a_patcher(tmp_path):
    trainer = _trainer(tmp_path)
    lengths = np.asarray(trainer.sample_batch()["patch_lengths"])
    assert set(lengths[lengths > 0].tolist()) == {4}


def test_patching_does_not_flow_gradient(tmp_path):
    # The patcher decides how the input is segmented; the student's loss must not
    # be able to move it. A gradient here would let the model widen patches to
    # make its own job easier.
    config = _config()
    mx.random.seed(0)
    patcher = MLXByteEntropyModel(config, dim=32, num_layers=1, num_heads=4, max_seq_len=128)
    mx.eval(patcher.parameters())
    trainer = _trainer(tmp_path, patcher=patcher)

    before = {name: np.array(value) for name, value in tree_flatten(patcher.parameters())}
    trainer.train(log=lambda *_: None)
    after = dict(tree_flatten(patcher.parameters()))
    for name, value in before.items():
        assert np.array_equal(value, np.asarray(after[name])), f"{name} moved"


def test_cli_trains_from_a_corpus(tmp_path):
    from blt.mlx_train import main

    corpus = _corpus_file(tmp_path, size=8192)
    main(
        [
            "--corpus", str(corpus),
            "--seq-len", "32",
            "--steps", "2",
            "--batch-size", "2",
            "--logits-kl", "0",
            "--log-every", "0",
            "--local-dim", "32",
            "--global-dim", "32",
            "--decoder-dim", "32",
            "--output", str(tmp_path / "out"),
        ]
    )
    assert (tmp_path / "out" / "model.safetensors").is_file()
    assert (tmp_path / "out" / "config.json").is_file()


def test_cli_refuses_both_sources(tmp_path, capsys):
    from blt.mlx_train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--corpus", "a.bin", "--teacher-cache", "b"])


def test_cli_requires_a_source():
    from blt.mlx_train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "1"])
