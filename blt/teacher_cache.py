"""On-disk teacher outputs, so an MLX student can distil without a torch teacher.

``facebook/blt-1b`` is torch-only and the upstream repo has to be importable to
build it, which makes a live teacher in an MLX training loop impractical. Instead
the teacher runs once under torch and its outputs land here; training then reads
only numpy.

Why top-k rather than full logits. BLT's vocabulary is 260, so a full fp16 logit
row is 520 bytes for a single *byte* of training text -- a 520x blowup over the
corpus itself. Top-32 with uint16 indices and fp16 values costs 128 bytes and,
for a byte-level model, covers essentially all the probability mass. The full-row
log-sum-exp is stored alongside so the covered mass is measurable rather than
assumed: :meth:`TeacherCache.coverage` reports it, and the smoke test asserts it.

The KL this supports is the truncated one: teacher probabilities renormalised
over the retained k, student log-probabilities still taken over the *full*
vocabulary. Only the teacher side is approximated; the student's normaliser stays
exact, which is what keeps the gradient right.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CACHE_VERSION = 1

_ARRAYS = {
    "tokens": np.int32,
    "mask": np.bool_,
    "patch_lengths": np.int32,
    "topk_indices": np.uint16,
    # Both shifted by the row maximum before storage; see TeacherCacheWriter.add.
    "topk_logits": np.float16,
    "logsumexp": np.float32,
}


@dataclass
class TeacherCacheMeta:
    version: int
    num_sequences: int
    seq_len: int
    max_patches: int
    top_k: int
    vocab_size: int
    teacher: str = "unknown"

    def __post_init__(self) -> None:
        if self.vocab_size > np.iinfo(np.uint16).max + 1:
            # Indices are stored as uint16; BLT's 260-entry vocabulary is nowhere
            # near this, but a caller reusing the format elsewhere might be.
            raise ValueError("vocab_size exceeds what uint16 token indices can address")
        if not 0 < self.top_k <= self.vocab_size:
            raise ValueError("top_k must be in (0, vocab_size]")


def _path(directory: Path, name: str) -> Path:
    return directory / f"{name}.npy"


class TeacherCacheWriter:
    """Streams teacher outputs into memory-mapped arrays.

    Sequence count and shapes are fixed up front so every array can be allocated
    once and filled in place -- a dump over a real corpus is far larger than RAM.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        num_sequences: int,
        seq_len: int,
        max_patches: int,
        top_k: int,
        vocab_size: int,
        teacher: str = "unknown",
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta = TeacherCacheMeta(
            version=CACHE_VERSION,
            num_sequences=num_sequences,
            seq_len=seq_len,
            max_patches=max_patches,
            top_k=top_k,
            vocab_size=vocab_size,
            teacher=teacher,
        )
        shapes = {
            "tokens": (num_sequences, seq_len),
            "mask": (num_sequences, seq_len),
            "patch_lengths": (num_sequences, max_patches),
            "topk_indices": (num_sequences, seq_len, top_k),
            "topk_logits": (num_sequences, seq_len, top_k),
            "logsumexp": (num_sequences, seq_len),
        }
        self._arrays = {
            name: np.lib.format.open_memmap(
                _path(self.directory, name), mode="w+", dtype=_ARRAYS[name], shape=shape
            )
            for name, shape in shapes.items()
        }
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    def add(
        self,
        *,
        tokens: np.ndarray,
        mask: np.ndarray,
        patch_lengths: np.ndarray,
        logits: np.ndarray,
    ) -> None:
        """Append one batch. ``logits`` is [batch, seq, vocab], full precision in."""
        tokens = np.asarray(tokens)
        batch = tokens.shape[0]
        if self._written + batch > self.meta.num_sequences:
            raise ValueError("teacher cache is full; declared num_sequences is too small")
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape[:2] != (batch, self.meta.seq_len):
            raise ValueError("logits must be shaped [batch, seq_len, vocab]")

        # Store everything shifted by the row maximum. Raw logits reach +-16 here,
        # where fp16 resolves to ~0.016 and costs over a percent of probability
        # mass; shifted, the entries that carry the mass sit near 0.0 where fp16
        # resolves to ~5e-4. Softmax is shift-invariant, so the reader is
        # unaffected -- it subtracts a maximum anyway.
        row_max = logits.max(axis=-1, keepdims=True)
        shifted = logits - row_max
        logsumexp = np.log(np.exp(shifted).sum(axis=-1))

        top_k = self.meta.top_k
        # argpartition gives the k largest unordered; sort them so the reader can
        # assume descending order and slice a smaller k without re-sorting.
        cut = np.argpartition(-shifted, top_k - 1, axis=-1)[..., :top_k]
        cut_values = np.take_along_axis(shifted, cut, axis=-1)
        order = np.argsort(-cut_values, axis=-1)
        indices = np.take_along_axis(cut, order, axis=-1)
        values = np.take_along_axis(cut_values, order, axis=-1)

        window = slice(self._written, self._written + batch)
        self._arrays["tokens"][window] = tokens
        self._arrays["mask"][window] = np.asarray(mask, dtype=bool)
        self._arrays["patch_lengths"][window] = self._pad_patches(np.asarray(patch_lengths))
        self._arrays["topk_indices"][window] = indices.astype(np.uint16)
        self._arrays["topk_logits"][window] = values.astype(np.float16)
        self._arrays["logsumexp"][window] = logsumexp.astype(np.float32)
        self._written += batch

    def _pad_patches(self, patch_lengths: np.ndarray) -> np.ndarray:
        width = patch_lengths.shape[1]
        if width > self.meta.max_patches:
            raise ValueError("patch_lengths wider than the declared max_patches")
        if width == self.meta.max_patches:
            return patch_lengths
        pad = np.zeros((patch_lengths.shape[0], self.meta.max_patches - width), patch_lengths.dtype)
        return np.concatenate([patch_lengths, pad], axis=1)

    def close(self) -> None:
        """Flush and record metadata, truncating to what was actually written."""
        for array in self._arrays.values():
            array.flush()
        self._arrays.clear()
        if self._written < self.meta.num_sequences:
            # A short dump leaves zero-filled tails that would train on garbage.
            for name in _ARRAYS:
                path = _path(self.directory, name)
                trimmed = np.array(np.load(path, mmap_mode="r")[: self._written])
                np.save(path, trimmed)
            self.meta.num_sequences = self._written
        (self.directory / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))

    def __enter__(self) -> TeacherCacheWriter:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class TeacherCache:
    """Read-only view over a dumped cache. Arrays stay memory-mapped."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no teacher cache at {self.directory} (missing meta.json)")
        self.meta = TeacherCacheMeta(**json.loads(meta_path.read_text()))
        if self.meta.version != CACHE_VERSION:
            raise ValueError(f"teacher cache version {self.meta.version} != expected {CACHE_VERSION}")
        self._arrays = {name: np.load(_path(self.directory, name), mmap_mode="r") for name in _ARRAYS}

    def __len__(self) -> int:
        return self.meta.num_sequences

    def batch(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        """Materialise one batch out of the memory map."""
        indices = np.asarray(indices)
        batch = {name: np.asarray(array[indices]) for name, array in self._arrays.items()}
        batch["tokens"] = batch["tokens"].astype(np.int32)
        batch["patch_lengths"] = batch["patch_lengths"].astype(np.int32)
        batch["topk_indices"] = batch["topk_indices"].astype(np.int32)
        batch["topk_logits"] = batch["topk_logits"].astype(np.float32)
        return batch

    def coverage(self, indices: np.ndarray | list[int]) -> float:
        """Mean probability mass the retained top-k actually covers.

        The one number that says whether ``top_k`` was set high enough. Below
        ~0.99 the truncated KL is teaching the student a visibly different
        distribution than the teacher produced.
        """
        batch = self.batch(indices)
        covered = np.exp(batch["topk_logits"] - batch["logsumexp"][..., None]).sum(axis=-1)
        mask = batch["mask"]
        if not mask.any():
            return 1.0
        return float(covered[mask].mean())


def teacher_probabilities(
    topk_logits: np.ndarray, *, temperature: float = 1.0
) -> np.ndarray:
    """Teacher distribution over the retained indices, renormalised.

    Renormalising over the kept k is the approximation: the tail mass is
    redistributed proportionally rather than dropped, which keeps the target a
    genuine distribution so the KL stays non-negative.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = topk_logits.astype(np.float32) / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=-1, keepdims=True)
