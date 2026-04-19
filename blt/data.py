"""Raw-byte data path for the ternary BLT stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import torch

from blt.config import TernaryBLTConfig

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional runtime dependency.
    load_dataset = None


class ByteVocabulary:
    def __init__(self, config: TernaryBLTConfig) -> None:
        self.config = config

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.config.bos_id)
        tokens.extend(self.config.byte_to_token_id(value) for value in text.encode("utf-8"))
        if add_eos:
            tokens.append(self.config.eos_id)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
            if add_eos and tokens:
                tokens[-1] = self.config.eos_id
        return tokens

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        bytes_out: List[int] = []
        for token_id in token_ids:
            if token_id == self.config.pad_id or token_id < self.config.offset:
                if skip_special_tokens:
                    continue
                raise ValueError(f"cannot decode special token id {token_id}")
            byte_value = token_id - self.config.offset
            if byte_value < 0 or byte_value >= self.config.byte_vocab_size:
                raise ValueError(f"cannot decode non-byte token id {token_id}")
            bytes_out.append(byte_value)
        return bytes(bytes_out).decode("utf-8", errors="replace")


class PackedByteSequenceStream:
    """Tokenize text to Meta-BLT-style byte IDs and pack autoregressive windows."""

    def __init__(
        self,
        text_stream: Iterator[str],
        vocabulary: ByteVocabulary,
        *,
        sequence_length: int,
        max_document_bytes: int,
    ) -> None:
        self.text_stream = text_stream
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.max_document_bytes = max_document_bytes
        self.buffer: List[int] = []

    def __iter__(self) -> "PackedByteSequenceStream":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while len(self.buffer) < self.sequence_length + 1:
            text = next(self.text_stream)
            token_ids = self.vocabulary.encode(text, max_length=self.max_document_bytes, add_bos=True, add_eos=True)
            if len(token_ids) < 2:
                continue
            self.buffer.extend(token_ids)

        window = self.buffer[: self.sequence_length + 1]
        del self.buffer[: self.sequence_length]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)
        attention_mask = torch.ones(self.sequence_length, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


@dataclass(slots=True)
class ByteBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


class BatchByteStream:
    """Group packed byte windows into fixed-size batches."""

    def __init__(self, sequence_stream: PackedByteSequenceStream, batch_size: int) -> None:
        self.sequence_stream = sequence_stream
        self.batch_size = batch_size

    def __iter__(self) -> "BatchByteStream":
        return self

    def __next__(self) -> ByteBatch:
        return collate_byte_batch([next(self.sequence_stream) for _ in range(self.batch_size)])


def collate_byte_batch(samples: Sequence[dict[str, torch.Tensor]]) -> ByteBatch:
    return ByteBatch(
        input_ids=torch.stack([sample["input_ids"] for sample in samples], dim=0),
        labels=torch.stack([sample["labels"] for sample in samples], dim=0),
        attention_mask=torch.stack([sample["attention_mask"] for sample in samples], dim=0),
    )


def iter_texts(texts: Iterable[str], *, restart_on_eof: bool = True) -> Iterator[str]:
    cached = [text for text in texts if text.strip()]
    if not cached:
        raise ValueError("text source must contain at least one non-empty string")

    while True:
        for text in cached:
            yield text
        if not restart_on_eof:
            return


def iter_text_file(path: str | Path, *, restart_on_eof: bool = True) -> Iterator[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"text file not found: {file_path}")

    while True:
        yielded = False
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yielded = True
                    yield text
        if not yielded:
            raise ValueError(f"text file is empty or only whitespace: {file_path}")
        if not restart_on_eof:
            return


def iter_hf_dataset(
    dataset_path: str,
    *,
    config_name: str | None,
    split: str,
    text_field: str,
    restart_on_eof: bool = True,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1000,
    seed: int = 0,
) -> Iterator[str]:
    if load_dataset is None:
        raise ImportError(
            "The `datasets` package is required for Hugging Face dataset streaming. "
            "Install it with `python3 -m pip install -r requirements.txt`."
        )

    current_seed = seed
    while True:
        dataset = load_dataset(dataset_path, name=config_name, split=split, streaming=True)
        if shuffle:
            dataset = dataset.shuffle(seed=current_seed, buffer_size=shuffle_buffer_size)
        yielded = False
        for example in dataset:
            text = example.get(text_field)
            if isinstance(text, str):
                text = text.strip()
                if text:
                    yielded = True
                    yield text
        if not yielded:
            raise ValueError(
                f"dataset '{dataset_path}' split '{split}' did not yield non-empty strings from field '{text_field}'"
            )
        if not restart_on_eof:
            return
        current_seed += 1
