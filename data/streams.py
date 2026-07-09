"""Restartable text streams, sequence packing, batching, and prefetch."""

from __future__ import annotations

import queue
import random
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from data.presets import COMMON_TEXT_FIELDS, DatasetSource

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for training only.
    load_dataset = None

try:
    from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
except ImportError:  # pragma: no cover
    HierarchicalTokenizer = Any  # type: ignore[misc, assignment]

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
        self.skip_examples = skip_examples
        self.restart_on_eof = restart_on_eof
        self.restart_count = 0
        self.iterator = self._build_iterator()

    def _build_iterator(self) -> Iterator[Dict[str, Any]]:
        if load_dataset is None:
            raise ImportError(
                "train.py requires the `datasets` package. Install it with "
                "`python3 -m pip install -r requirements.txt`."
            )

        dataset = load_dataset(
            self.source.path,
            name=self.source.config_name,
            split=self.source.split,
            streaming=True,
        )
        if self.skip_examples > 0:
            dataset = dataset.skip(self.skip_examples)
        if self.shuffle:
            dataset = dataset.shuffle(
                seed=self.seed + self.restart_count,
                buffer_size=self.shuffle_buffer_size,
            )
        return iter(dataset)

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

    def __next__(self) -> str:
        while True:
            try:
                example = next(self.iterator)
            except StopIteration:
                if not self.restart_on_eof:
                    raise
                self.restart_count += 1
                self.iterator = self._build_iterator()
                continue

            text = self._extract_text(example)
            if text is None:
                continue
            text = text.strip()
            if text:
                return text


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
        self.sequence_length = sequence_length
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
                text,
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
        segment_window = self.segment_buffer[: self.sequence_length]
        del self.segment_buffer[: self.sequence_length]

        base_id = segment_window[0]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)
        segment_ids = torch.tensor([s - base_id for s in segment_window], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }


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
            for key in ("input_ids", "labels", "segment_ids")
        }


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
