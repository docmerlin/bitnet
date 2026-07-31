"""Raw byte corpora for from-scratch BLT training.

The distillation path reads a :class:`blt.teacher_cache.TeacherCache`; this reads
the bytes themselves. Same access shape -- ``len()`` and ``batch(indices)`` -- so
the trainer takes either, and the only difference is that a corpus carries no
teacher fields, which forces the KL term off.

Files are memory-mapped and chunked on the fly rather than loaded, so a corpus
larger than RAM costs nothing extra. Sequences never straddle a file boundary:
the tail of each file that will not fill a whole sequence is dropped, because a
sequence spanning two unrelated documents teaches a transition that does not
exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ByteCorpus:
    """Fixed-length byte sequences drawn from one or more files.

    Bytes map to token ids by adding ``offset`` (4 in the BLT vocabulary, which
    reserves 0-3 for BOE/BOS/EOS/BPE). Special tokens are the caller's business:
    a corpus is raw content, and wrapping documents is a decision about the data,
    not about the loader.
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        *,
        seq_len: int,
        offset: int = 4,
        stride: int | None = None,
    ) -> None:
        if seq_len <= 1:
            raise ValueError("seq_len must exceed 1 -- a sequence needs at least one target")
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if not paths:
            raise ValueError("ByteCorpus needs at least one file")

        self.seq_len = seq_len
        self.offset = offset
        # Non-overlapping by default. A shorter stride multiplies the number of
        # sequences without adding information, so it is opt-in.
        self.stride = seq_len if stride is None else stride
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self._files: list[np.memmap] = []
        self._starts: list[np.ndarray] = []
        self._file_index: list[np.ndarray] = []
        for path in paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"corpus file not found: {path}")
            data = np.memmap(path, dtype=np.uint8, mode="r")
            if data.size < seq_len:
                continue  # too short to yield even one sequence
            starts = np.arange(0, data.size - seq_len + 1, self.stride, dtype=np.int64)
            self._files.append(data)
            self._starts.append(starts)
            # Index into self._files, which skips files that were too short.
            self._file_index.append(np.full(starts.size, len(self._files) - 1))

        if not self._files:
            raise ValueError(f"no file in the corpus holds {seq_len} bytes")

        self._all_starts = np.concatenate(self._starts)
        self._all_files = np.concatenate(self._file_index)

    def __len__(self) -> int:
        return int(self._all_starts.size)

    @property
    def total_bytes(self) -> int:
        return int(sum(data.size for data in self._files))

    def batch(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        """Materialise one batch. Keys match the teacher cache's, minus the teacher."""
        indices = np.asarray(indices)
        tokens = np.empty((indices.size, self.seq_len), dtype=np.int32)
        for row, index in enumerate(indices):
            data = self._files[self._all_files[index]]
            start = int(self._all_starts[index])
            tokens[row] = data[start : start + self.seq_len].astype(np.int32) + self.offset
        return {"tokens": tokens, "mask": np.ones_like(tokens, dtype=bool)}


def write_byte_corpus(path: str | Path, text: str | bytes) -> Path:
    """Write raw bytes to ``path``. Convenience for tests and small corpora."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return path
