"""Rolling polynomial hash for byte n-gram embeddings.

BLT's local encoder sees one byte at a time, so a multi-byte unit like ``the``
arrives as three unrelated embedding lookups with no shared identity -- something
a BPE tokenizer gets for free. Meta's BLT restores it without a vocabulary by
adding, at every position, an embedding of each n-gram *ending* at that position
(arXiv:2412.09871, eqs. 2-4). The table is indexed by a hash, so the parameter
budget is fixed no matter how many distinct n-grams exist; collisions are
accepted and the frequent n-grams dominate their bucket.

Two properties the construction depends on:

*Causal.* ``g_{i,n}`` ends at ``i`` and reads only bytes at or before it, so
nothing has to be masked for autoregression.

*Rolling.* The size-``n+1`` hash is the size-``n`` hash plus one more term, so
every n-gram size at a position comes out of a single accumulation rather than
one pass per size. The construction would also support O(1) per appended byte at
decode, but nothing implements that: generation re-embeds the whole prefix on
every drafted byte, so the hash is recomputed over all of it. Worth doing when
generation length rather than model calls becomes the cost.

The constants below are load-bearing: the torch and MLX stacks index the same
tables, so both must produce bit-identical indices. ``reference_ngram_hashes``
is the executable definition, and both implementations are tested against it.
"""

from __future__ import annotations

import numpy as np

#: Mersenne prime modulus. Keeps ``byte * P**lag`` inside int64 at every step.
HASH_MODULUS = 2147483647
#: Polynomial base.
HASH_BASE = 1000003
#: Stand-in byte for positions before the start of the sequence. Those positions
#: are masked out anyway; this only keeps the arithmetic well defined.
HASH_PAD = 0


def hash_bases(max_ngram_size: int) -> list[int]:
    """``P**lag mod M`` for each lag, precomputed."""
    bases = []
    power = 1
    for _ in range(max_ngram_size):
        bases.append(power)
        power = (power * HASH_BASE) % HASH_MODULUS
    return bases


def reference_ngram_hashes(
    input_ids: np.ndarray, ngram_sizes: tuple[int, ...], vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(indices, valid)`` for each n-gram size, shaped ``[len(sizes), B, L]``.

    ``valid`` is False where the n-gram would reach past the start of the
    sequence. Callers additionally drop positions their attention mask rejects.
    """
    ids = np.asarray(input_ids, dtype=np.int64)
    batch, length = ids.shape
    bases = hash_bases(max(ngram_sizes))
    running = np.zeros((batch, length), dtype=np.int64)
    indices = np.zeros((len(ngram_sizes), batch, length), dtype=np.int64)
    valid = np.zeros((len(ngram_sizes), batch, length), dtype=bool)
    wanted = {size: position for position, size in enumerate(ngram_sizes)}
    for lag in range(max(ngram_sizes)):
        shifted = np.full((batch, length), HASH_PAD, dtype=np.int64)
        if lag < length:
            shifted[:, lag:] = ids[:, : length - lag]
        running = (running + shifted * bases[lag]) % HASH_MODULUS
        size = lag + 1
        if size in wanted:
            slot = wanted[size]
            indices[slot] = running % vocab_size
            valid[slot] = np.arange(length)[None, :] >= size - 1
    return indices, valid
