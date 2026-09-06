"""The holdout is independent of shuffle, source aliases, and stream restarts."""

import hashlib
import json
from itertools import islice

import pytest
from datasets import IterableDataset

import data.streams as streams
from data.presets import DatasetSource


def test_text_partitions_are_disjoint_before_shuffle_and_after_resume(monkeypatch):
    documents = [{"text": f"document-{index}"} for index in range(2000)]
    documents += [{"text": "  document-7  "}, {"text": "document-100"}]
    monkeypatch.setattr(
        streams, "load_dataset",
        lambda *_args, **_kwargs: IterableDataset.from_generator(lambda: iter(documents)),
    )

    def make(partition, seed=17, restart=False, skip=0):
        return streams.TextDatasetStream(
            DatasetSource(str(seed), "test", None, "train", "text"),
            seed=seed, shuffle=partition == "train", shuffle_buffer_size=31,
            skip_examples=skip, restart_on_eof=restart, partition=partition,
        )

    validation = set(make("validation"))
    training = set(make("train"))
    assert validation and training
    assert not training & validation
    assert training | validation == {row["text"].strip() for row in documents}
    assert set(make("train", seed=999)) == training
    assert set(make("validation", skip=100)) <= validation
    restarted = make("train", restart=True)
    assert set(islice(restarted, 5000)) == training
    state = json.loads(json.dumps(restarted.state_dict()))
    expected = list(islice(restarted, 100))
    restored = make("train", restart=True)
    restored.load_state_dict(state)
    assert list(islice(restored, 100)) == expected
    assert not set(expected) & validation
    state.pop("partition")
    with pytest.raises(ValueError, match="partition changed"):
        restored.load_state_dict(state)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_trainers_partition_both_curriculum_stages(monkeypatch, tmp_path, backend):
    if backend == "mlx":
        pytest.importorskip("mlx.core")
        import mlx_train as trainer
        flags = ["--num-prelude-layers", "0", "--num-recurrent-layers", "1",
                 "--num-coda-layers", "0", "--num-loops", "1", "--mtp-depth", "0"]
    else:
        import train as trainer
        flags = ["--num-layers", "1", "--disable-hadamard"]

    class Tokenizer:
        def __init__(self, **kwargs):
            pass

        def __len__(self):
            return 260

    class StreamsBuilt(Exception):
        pass

    calls = []

    def build_stream(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise StreamsBuilt
        return iter(())

    monkeypatch.setattr(trainer, "HierarchicalTokenizer", Tokenizer)
    model_name = "MLXBitNet" if backend == "mlx" else "BitNetDeep"
    model_class = getattr(trainer, model_name)

    def build_model(config):
        assert config.vocab_size == 260
        return model_class(config)

    monkeypatch.setattr(trainer, model_name, build_model)
    monkeypatch.setattr(trainer, "build_batch_stream", build_stream)
    monkeypatch.setattr("sys.argv", [
        "train", "--output-dir", str(tmp_path), "--hidden-size", "16",
        "--num-heads", "2", "--intermediate-size", "32", "--sequence-length", "8",
        "--path-window-size", "4", "--no-engram", "--no-gradient-checkpointing",
        "--late-train-mixture", "fineweb_edu=1", *flags,
    ])
    with pytest.raises(StreamsBuilt):
        trainer.main()
    assert [call["partition"] for call in calls] == ["train", "train"]


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_trainers_use_validation_partition(monkeypatch, backend):
    class StreamBuilt(ValueError):
        pass

    def build_stream(*args, **kwargs):
        assert kwargs["partition"] == "validation"
        raise StreamBuilt

    if backend == "mlx":
        pytest.importorskip("mlx.core")
        import mlx_train as trainer
        monkeypatch.setattr(trainer, "build_batch_stream", build_stream)
        with pytest.raises(StreamBuilt):
            trainer.build_validation_batches(None, trainer.build_parser().parse_args([]))
    else:
        import torch
        from train import build_arg_parser
        import training.runtime as runtime
        monkeypatch.setattr(runtime, "build_batch_stream", build_stream)
        runner = torch.nn.Linear(1, 1)
        with pytest.raises(StreamBuilt):
            runtime.evaluate(runner, None, [], build_arg_parser().parse_args([]),
                             torch.device("cpu"), False, None)
        assert runner.training


def _partition_text(held_out):
    for index in range(10000):
        text = f"tiny-{index}"
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        if (int.from_bytes(digest, "big") % 100 == 0) == held_out:
            return text
    raise AssertionError("test corpus has no matching partition")


@pytest.mark.parametrize("partition", ["train", "validation"])
@pytest.mark.parametrize("skip", [0, 1])
def test_empty_partition_fails_without_restarting(monkeypatch, partition, skip):
    loads = []
    documents = [{"text": _partition_text(partition != "validation")}]

    def load_dataset(*args, **kwargs):
        loads.append(args)
        return IterableDataset.from_generator(lambda: iter(documents))

    monkeypatch.setattr(streams, "load_dataset", load_dataset)
    stream = streams.TextDatasetStream(
        DatasetSource("tiny", "tiny", None, "train", "text"),
        seed=17, shuffle=partition == "train", shuffle_buffer_size=5,
        skip_examples=skip, restart_on_eof=True, partition=partition,
    )
    for _ in range(2):
        with pytest.raises(streams.EmptyPartitionError) as exc:
            next(stream)
        assert "dataset 'tiny'" in str(exc.value)
        assert f"partition '{partition}'" in str(exc.value)
        assert f"skip_examples={skip}" in str(exc.value)
        assert "larger source or reduce the skip offset" in str(exc.value)
    assert stream.restart_count == 0
    assert len(loads) == 1


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("partial", [False, True])
def test_validation_skips_tiny_holdout_and_discards_partial_results(monkeypatch, capsys, backend, partial):
    from types import SimpleNamespace

    loaded = []

    def load_dataset(path, **kwargs):
        loaded.append(path)
        documents = [{"text": _partition_text(path == "good")}]
        return IterableDataset.from_generator(lambda: iter(documents))

    class Tokenizer:
        calls = 0

        def encode(self, text, **kwargs):
            assert text == _partition_text(True)  # Never fall back to training text.
            self.calls += 1
            return [0, 1, 2]

    tokenizer = Tokenizer()
    args = SimpleNamespace(
        validation_batches=2, seed=17, shuffle_buffer_size=5,
        validation_offset_examples=0, sequence_length=2, max_document_tokens=32,
        micro_batch_size=1,
        val_mixture=("good||train|text=1," if partial else "") + "tiny||train|text=1",
    )
    monkeypatch.setattr(streams, "load_dataset", load_dataset)

    def build_stream(*args, **kwargs):
        stream = streams.build_batch_stream(*args, **kwargs)
        if partial:
            order = iter([0, 1])  # One complete good batch, then an empty source.
            stream.sequence_stream.text_stream.rng.choices = lambda *a, **kw: [next(order)]
        return stream

    if backend == "mlx":
        pytest.importorskip("mlx.core")
        import mlx_train
        monkeypatch.setattr(mlx_train, "build_batch_stream", build_stream)
        assert mlx_train.build_validation_batches(tokenizer, args) == []
    else:
        import torch
        from config import TernaryConfig
        from data.presets import parse_mixture
        from layers.infini_attention import InfiniAttention
        import training.runtime as runtime

        class Runner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.memory = InfiniAttention(TernaryConfig(
                    hidden_size=16, num_attention_heads=2, head_dim=8,
                    intermediate_size=32, use_hadamard=False,
                ))
                self.memory._ensure_memory_batch(1)
                self.memory.memory_m.fill_(7)
                self.memory.memory_z.fill_(9)
                self.memory.memory_initialized.fill_(True)

            def forward(self, inputs, **kwargs):
                assert not self.training
                assert not self.memory.memory_m.any()
                self.memory.memory_m.fill_(3)
                return torch.zeros(*inputs.shape, 3)

        runner = Runner().train(partial)
        saved_memory = runner.memory.get_memory_state()
        monkeypatch.setattr(runtime, "build_batch_stream", build_stream)
        assert runtime.evaluate(runner, tokenizer, parse_mixture(args.val_mixture), args,
                                torch.device("cpu"), False, None) == {}
        assert runner.training == partial
        assert all(torch.equal(saved_memory[key], value)
                   for key, value in runner.memory.get_memory_state().items())

    assert tokenizer.calls == int(partial)
    assert loaded == (["good", "tiny"] if partial else ["tiny"])
    warning = capsys.readouterr().out
    assert "Warning: validation skipped" in warning
    assert "dataset 'tiny'" in warning
