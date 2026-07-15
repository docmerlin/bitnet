"""Restartable text streams, sequence packing, batching, and prefetch."""

from __future__ import annotations

import queue
import random
import sys
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from data.presets import COMMON_TEXT_FIELDS, DatasetSource

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for training only.
    load_dataset = None


if load_dataset is not None and sys.version_info >= (3, 14):
    import pyarrow as pa
    import pyarrow.dataset as ds
    from datasets.builder import Key
    from datasets.packaged_modules.parquet import parquet as parquet_module
    from datasets.packaged_modules.parquet.parquet import Parquet
    from datasets.utils.file_utils import xopen

    def _generate_tables_without_arrow_threads(self, files, row_groups_list):
        # Arrow workers calling Python-backed HTTP files can deadlock during
        # CPython 3.14 finalization. Synchronous reads leave no worker behind.
        if self.config.features is not None and self.config.columns is not None:
            if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load parquet data with columns '{self.config.columns}' "
                    f"with mismatching features '{self.info.features}'"
                )
        filter_expr = (
            parquet_module.pq.filters_to_expression(self.config.filters)
            if isinstance(self.config.filters, list)
            else self.config.filters
        )
        file_format = ds.ParquetFileFormat(default_fragment_scan_options=self.config.fragment_scan_options)
        for file_index, (file, row_groups) in enumerate(zip(files, row_groups_list)):
            open_file = getattr(parquet_module, "open", xopen)
            try:
                with open_file(file, "rb") as handle:
                    fragment = file_format.make_fragment(handle)
                    if row_groups is not None:
                        fragment = fragment.subset(row_group_ids=row_groups)
                    if fragment.row_groups:
                        batch_size = self.config.batch_size or fragment.row_groups[0].num_rows
                        for batch_index, batch in enumerate(
                            fragment.to_batches(
                                batch_size=batch_size,
                                columns=self.config.columns,
                                filter=filter_expr,
                                batch_readahead=0,
                                fragment_readahead=0,
                                use_threads=False,
                            )
                        ):
                            yield Key(file_index, batch_index), self._cast_table(pa.Table.from_batches([batch]))
            except (pa.ArrowInvalid, ValueError) as error:
                message = f"Skipping bad file '{file}'. {type(error).__name__}: {error}"
                if self.config.on_bad_files == "error":
                    raise
                if self.config.on_bad_files == "warn":
                    parquet_module.logger.warning(message)
                else:
                    parquet_module.logger.debug(message)

    Parquet._generate_tables = _generate_tables_without_arrow_threads

try:
    from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
except ImportError:  # pragma: no cover
    HierarchicalTokenizer = Any  # type: ignore[misc, assignment]


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value

__all__ = [
    "BatchStream",
    "PackedSequenceStream",
    "PrefetchStream",
    "TextDatasetStream",
    "WeightedMixtureStream",
    "build_batch_stream",
    "build_text_stream",
]


class PrefetchStream:
    """Background-thread prefetcher that overlaps CPU work with GPU compute.

    Daemon worker pulls from an iterator, optionally pins tensor dicts for async
    H2D, and pushes through a bounded queue (backpressure). Lazy-starts on first
    ``next`` so unused streams hold no handles. Non-dict batch objects are not
    pinned (pass plain tensor dicts if pin_memory matters).
    """

    def __init__(
        self,
        stream: Iterator[Any],
        *,
        buffer_size: int = 2,
        pin_memory: bool = False,
    ) -> None:
        self.stream = stream
        self.pin_memory = pin_memory
        self.queue: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=max(1, buffer_size))
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False
        self._exhausted = False

    def _maybe_pin(self, item: Any) -> Any:
        if not self.pin_memory:
            return item
        if isinstance(item, dict):
            return {
                key: value.pin_memory() if torch.is_tensor(value) else value
                for key, value in item.items()
            }
        if torch.is_tensor(item):
            return item.pin_memory()
        return item

    def _worker(self) -> None:
        try:
            for item in self.stream:
                self.queue.put(self._maybe_pin(item))
        except BaseException as exc:
            self._error = exc
        finally:
            self.queue.put(None)

    def __iter__(self) -> "PrefetchStream":
        return self

    def __next__(self) -> Any:
        if not self._started:
            self._thread.start()
            self._started = True
        if self._exhausted:
            raise self._error if self._error is not None else StopIteration
        item = self.queue.get()
        if item is None:
            self._exhausted = True
            raise self._error if self._error is not None else StopIteration
        return item


class TextDatasetStream:
    """Restartable streaming Hugging Face text source."""

    def __init__(
        self,
        source: DatasetSource,
        *,
        seed: int,
        shuffle: bool,
        shuffle_buffer_size: int,
        skip_examples: int,
        restart_on_eof: bool,
    ) -> None:
        self.source = source
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        if shuffle and shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive when shuffle is enabled")
        self.skip_examples = skip_examples
        self.restart_on_eof = restart_on_eof
        self.restart_count = 0
        self.yielded_this_pass = False
        self.shuffle_rng = random.Random(seed)
        self.shuffle_buffer: List[str] = []
        self.source_exhausted = False
        self.dataset = None
        self.iterator = self._build_iterator()

    def _build_iterator(self) -> Iterator[Dict[str, Any]]:
        if load_dataset is None:
            raise ImportError(
                "train.py requires the `datasets` package. Install it with "
                "`python3 -m pip install -r requirements.txt`."
            )

        self.dataset = load_dataset(
            self.source.path,
            name=self.source.config_name,
            split=self.source.split,
            streaming=True,
        )
        if self.skip_examples > 0:
            self.dataset = self.dataset.skip(self.skip_examples)
        return iter(self.dataset)

    def _extract_text(self, example: Dict[str, Any]) -> Optional[str]:
        if self.source.text_field in example and isinstance(example[self.source.text_field], str):
            return example[self.source.text_field]

        for field in COMMON_TEXT_FIELDS:
            value = example.get(field)
            if isinstance(value, str):
                return value
        return None

    def __iter__(self) -> "TextDatasetStream":
        return self

    def _next_source_text(self) -> str:
        while True:
            example = next(self.iterator)
            text = self._extract_text(example)
            if text is None:
                continue
            text = text.strip()
            if text:
                self.yielded_this_pass = True
                return text

    def _restart(self) -> None:
        if not self.restart_on_eof:
            raise StopIteration
        if not self.yielded_this_pass:
            raise ValueError(f"dataset {self.source.path!r} produced no non-empty text records")
        self.restart_count += 1
        self.yielded_this_pass = False
        if getattr(self, "shuffle", False):
            self.source_exhausted = False
            self.shuffle_buffer.clear()
            self.shuffle_rng.seed(self.seed + self.restart_count)
        self.iterator = self._build_iterator()

    def __next__(self) -> str:
        if not getattr(self, "shuffle", False):
            while True:
                try:
                    return self._next_source_text()
                except StopIteration:
                    self._restart()

        while True:
            while len(self.shuffle_buffer) < self.shuffle_buffer_size and not self.source_exhausted:
                try:
                    self.shuffle_buffer.append(self._next_source_text())
                except StopIteration:
                    self.source_exhausted = True
            if self.shuffle_buffer:
                index = self.shuffle_rng.randrange(len(self.shuffle_buffer))
                value = self.shuffle_buffer[index]
                if self.source_exhausted:
                    self.shuffle_buffer.pop(index)
                else:
                    try:
                        self.shuffle_buffer[index] = self._next_source_text()
                    except StopIteration:
                        self.source_exhausted = True
                        self.shuffle_buffer.pop(index)
                return value
            self._restart()

    def state_dict(self) -> Dict[str, Any]:
        if self.dataset is None or not hasattr(self.dataset, "state_dict"):
            raise RuntimeError("dataset does not support resumable streaming state")
        return {
            "restart_count": self.restart_count,
            "yielded_this_pass": self.yielded_this_pass,
            "source_exhausted": self.source_exhausted,
            "shuffle_rng": self.shuffle_rng.getstate(),
            "shuffle_buffer": self.shuffle_buffer,
            "dataset": self.dataset.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.restart_count = int(state["restart_count"])
        self.yielded_this_pass = bool(state["yielded_this_pass"])
        self.source_exhausted = bool(state["source_exhausted"])
        self.shuffle_buffer = list(state["shuffle_buffer"])
        self.shuffle_rng.setstate(_nested_tuple(state["shuffle_rng"]))
        self.iterator = self._build_iterator()
        if self.dataset is None or not hasattr(self.dataset, "load_state_dict"):
            raise RuntimeError("dataset does not support resumable streaming state")
        self.dataset.load_state_dict(state["dataset"])
        self.iterator = iter(self.dataset)


class WeightedMixtureStream:
    """Draw documents from multiple restartable text streams according to weights."""

    def __init__(self, streams: Sequence[TextDatasetStream], weights: Sequence[float], seed: int) -> None:
        self.streams = list(streams)
        self.weights = list(weights)
        self.rng = random.Random(seed)

    def __iter__(self) -> "WeightedMixtureStream":
        return self

    def __next__(self) -> str:
        index = self.rng.choices(range(len(self.streams)), weights=self.weights, k=1)[0]
        return next(self.streams[index])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "streams": [stream.state_dict() for stream in self.streams],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if len(state["streams"]) != len(self.streams):
            raise ValueError("checkpoint dataset mixture does not match current mixture")
        self.rng.setstate(_nested_tuple(state["rng"]))
        for stream, stream_state in zip(self.streams, state["streams"]):
            stream.load_state_dict(stream_state)


class PackedSequenceStream:
    """Tokenize documents and pack them into fixed-length autoregressive windows."""

    def __init__(
        self,
        text_stream: Iterator[str],
        tokenizer: HierarchicalTokenizer,
        *,
        sequence_length: int,
        max_document_tokens: int,
    ) -> None:
        self.text_stream = text_stream
        self.tokenizer = tokenizer
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        if max_document_tokens < 2:
            raise ValueError("max_document_tokens must be >= 2")
        self.max_document_tokens = max_document_tokens
        self.buffer: List[int] = []
        # Parallel per-token document ids so packed windows can mask cross-document
        # attention. The counter only needs to be locally unique within a window;
        # ids are re-based to start at zero on emit.
        self.segment_buffer: List[int] = []
        self.next_segment_id = 0

    def __iter__(self) -> "PackedSequenceStream":
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        while len(self.buffer) < self.sequence_length + 1:
            text = next(self.text_stream)
            token_ids = self.tokenizer.encode(
                text[: self.max_document_tokens],
                max_length=self.max_document_tokens,
                add_special_tokens=True,
            )
            if len(token_ids) < 2:
                continue
            self.buffer.extend(token_ids)
            self.segment_buffer.extend([self.next_segment_id] * len(token_ids))
            self.next_segment_id += 1

        window = self.buffer[: self.sequence_length + 1]
        del self.buffer[: self.sequence_length]
        segment_window = self.segment_buffer[: self.sequence_length + 1]
        del self.segment_buffer[: self.sequence_length]

        base_id = segment_window[0]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)
        segment_ids = torch.tensor([s - base_id for s in segment_window[:-1]], dtype=torch.long)
        label_segment_ids = torch.tensor([s - base_id for s in segment_window[1:]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
            "label_segment_ids": label_segment_ids,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "text_stream": self.text_stream.state_dict(),
            "buffer": self.buffer,
            "segment_buffer": self.segment_buffer,
            "next_segment_id": self.next_segment_id,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.text_stream.load_state_dict(state["text_stream"])
        self.buffer = list(state["buffer"])
        self.segment_buffer = list(state["segment_buffer"])
        self.next_segment_id = int(state["next_segment_id"])


class BatchStream:
    """Batch fixed-length packed sequences without a DataLoader."""

    def __init__(self, sequence_stream: PackedSequenceStream, micro_batch_size: int) -> None:
        self.sequence_stream = sequence_stream
        self.micro_batch_size = micro_batch_size

    def __iter__(self) -> "BatchStream":
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        batch = [next(self.sequence_stream) for _ in range(self.micro_batch_size)]
        return {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in ("input_ids", "labels", "segment_ids", "label_segment_ids")
        }

    def state_dict(self) -> Dict[str, Any]:
        return {"sequence_stream": self.sequence_stream.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.sequence_stream.load_state_dict(state["sequence_stream"])


def build_text_stream(
    mixture: List[Tuple[DatasetSource, float]],
    *,
    seed: int,
    shuffle: bool,
    shuffle_buffer_size: int,
    skip_examples: int,
    restart_on_eof: bool,
) -> WeightedMixtureStream:
    streams = [
        TextDatasetStream(
            source=source,
            seed=seed + index * 1000,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            skip_examples=skip_examples,
            restart_on_eof=restart_on_eof,
        )
        for index, (source, _) in enumerate(mixture)
    ]
    weights = [weight for _, weight in mixture]
    return WeightedMixtureStream(streams, weights, seed=seed)


def build_batch_stream(
    mixture: List[Tuple[DatasetSource, float]],
    tokenizer: HierarchicalTokenizer,
    *,
    seed: int,
    shuffle: bool,
    shuffle_buffer_size: int,
    skip_examples: int,
    restart_on_eof: bool,
    sequence_length: int,
    max_document_tokens: int,
    micro_batch_size: int,
) -> BatchStream:
    text_stream = build_text_stream(
        mixture,
        seed=seed,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        skip_examples=skip_examples,
        restart_on_eof=restart_on_eof,
    )
    packed_stream = PackedSequenceStream(
        text_stream,
        tokenizer,
        sequence_length=sequence_length,
        max_document_tokens=max_document_tokens,
    )
    return BatchStream(packed_stream, micro_batch_size=micro_batch_size)
