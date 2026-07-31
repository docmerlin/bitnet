"""Patch bookkeeping for the MLX BLT student.

Mirrors :mod:`blt.patching.teacher_patcher` exactly -- the tests assert equality
against it -- but without the per-row Python loops. Those loops are why the torch
originals are slow: ``normalize_patch_lengths`` walks every row and then every
patch inside it, and ``pool_patch_representations`` runs a Python loop of
``index_add_`` calls over the batch. Both cost real time at training batch sizes
and neither needs to be a loop.

Two rewrites carry that:

``normalize_patch_lengths``
    Trimming a patch list down to ``target_length`` is just clipping each patch
    to whatever budget survives the patches before it:
    ``keep_i = clip(target - offset_i, 0, len_i)``. One cumsum, no branches.

``pool_patch_representations``
    Segment-mean becomes a membership-matrix matmul. Costs an O(B*P*L) mask, but
    it is a single GPU kernel instead of B scatter calls.
"""

from __future__ import annotations

import mlx.core as mx


def build_uniform_patch_lengths(batch_size: int, seq_len: int, patch_size: int) -> mx.array:
    """Fixed-width patches, the last one holding whatever remains."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    num_patches = (seq_len + patch_size - 1) // patch_size
    if num_patches == 0:
        return mx.zeros((batch_size, 0), dtype=mx.int32)
    lengths = mx.full((num_patches,), patch_size, dtype=mx.int32)
    tail = seq_len - patch_size * (num_patches - 1)
    lengths = mx.concatenate([lengths[:-1], mx.array([tail], dtype=mx.int32)])
    return mx.broadcast_to(lengths[None], (batch_size, num_patches))


def patch_ids_from_lengths(patch_lengths: mx.array, seq_len: int) -> mx.array:
    """Which patch each byte position belongs to."""
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")
    cumulative = mx.cumsum(patch_lengths, axis=-1)
    positions = mx.arange(seq_len, dtype=patch_lengths.dtype).reshape(1, 1, seq_len)
    return mx.sum(positions >= cumulative[..., None], axis=1)


def patch_presence_mask(patch_lengths: mx.array) -> mx.array:
    return patch_lengths > 0


def patch_membership_mask(patch_ids: mx.array, num_patches: int, *, patches_as_queries: bool) -> mx.array:
    patch_range = mx.arange(num_patches, dtype=patch_ids.dtype)
    if patches_as_queries:
        return patch_range.reshape(1, num_patches, 1) == patch_ids[:, None, :]
    return patch_ids[..., None] == patch_range.reshape(1, 1, num_patches)


def normalize_patch_lengths(patch_lengths: mx.array, target_length: int) -> mx.array:
    """Trim or extend trailing patches so each row sums to ``target_length``.

    Trimming falls out of a cumsum: patch ``i`` may keep only the budget left
    after the patches before it, so ``clip(target - offset_i, 0, len_i)`` removes
    exactly the trailing bytes the loop version removed. Extending adds the
    shortfall to the last non-empty patch, or to patch 0 when the row is empty.
    """
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")

    lengths = patch_lengths.astype(mx.int32)
    if lengths.shape[1] == 0 or target_length == 0:
        return mx.zeros_like(lengths)

    offsets = mx.cumsum(lengths, axis=-1) - lengths
    trimmed = mx.clip(target_length - offsets, 0, lengths)

    # Extend: whatever the row still lacks goes on its last non-empty patch.
    shortfall = target_length - mx.sum(trimmed, axis=-1, keepdims=True)
    columns = mx.arange(lengths.shape[1], dtype=mx.int32).reshape(1, -1)
    occupied = mx.where(trimmed > 0, columns, mx.array(-1, dtype=mx.int32))
    last = mx.max(occupied, axis=-1, keepdims=True)
    # An all-empty row has no last patch; the loop version seeds patch 0.
    target_column = mx.maximum(last, 0)
    return trimmed + mx.where(columns == target_column, shortfall, 0)


def normalize_patch_lengths_to_targets(patch_lengths: mx.array, target_lengths: mx.array) -> mx.array:
    """Per-row :func:`normalize_patch_lengths`, one row per entry of ``target_lengths``."""
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")
    if target_lengths.ndim != 1 or target_lengths.shape[0] != patch_lengths.shape[0]:
        raise ValueError("target_lengths must be shaped [batch]")

    lengths = patch_lengths.astype(mx.int32)
    targets = target_lengths.astype(mx.int32).reshape(-1, 1)
    # No value check on `targets` here. Reading it forces a GPU sync and makes
    # every caller uncompilable; the lengths come from an attention mask's row
    # sums, which cannot be negative. Callers that build targets by hand should
    # check before they get here.
    if lengths.shape[1] == 0:
        return mx.zeros_like(lengths)

    offsets = mx.cumsum(lengths, axis=-1) - lengths
    trimmed = mx.clip(targets - offsets, 0, lengths)

    shortfall = targets - mx.sum(trimmed, axis=-1, keepdims=True)
    columns = mx.arange(lengths.shape[1], dtype=mx.int32).reshape(1, -1)
    occupied = mx.where(trimmed > 0, columns, mx.array(-1, dtype=mx.int32))
    last = mx.max(occupied, axis=-1, keepdims=True)
    extended = trimmed + mx.where(columns == mx.maximum(last, 0), shortfall, 0)
    return mx.where(targets == 0, mx.zeros_like(extended), extended)


def pad_patch_lengths_to_bucket(patch_lengths: mx.array, bucket: int) -> mx.array:
    """Zero-pad the patch axis up to the next multiple of ``bucket``.

    Entropy patching yields a different patch count almost every batch -- 25
    distinct widths in 40 batches on real text -- and ``mx.compile`` keys its
    cache on shape, so an uncompiled graph would be rebuilt continuously.
    Rounding the width up collapses that to a handful of shapes.

    Zero-length patches are already inert everywhere they are consumed:
    ``patch_presence_mask`` masks them out of global attention, membership
    matching never assigns a byte to one, and pooling divides by a clamped
    count. The cost is the padding's share of the global transformer.
    """
    if bucket <= 0:
        raise ValueError("bucket must be positive")
    width = patch_lengths.shape[1]
    padded = ((width + bucket - 1) // bucket) * bucket
    if padded == width:
        return patch_lengths
    return mx.concatenate(
        [patch_lengths, mx.zeros((patch_lengths.shape[0], padded - width), dtype=patch_lengths.dtype)],
        axis=1,
    )


def pool_patch_representations(
    hidden_states: mx.array,
    patch_lengths: mx.array,
    *,
    patch_ids: mx.array | None = None,
    token_mask: mx.array | None = None,
    pooling: str = "mean",
) -> mx.array:
    """Pool byte states into one vector per patch.

    A membership matmul rather than a scatter loop: ``(B, P, L) @ (B, L, D)``.
    The mask costs memory the scatter did not, but it is one kernel instead of
    one Python-level call per batch row.
    """
    if pooling not in {"mean", "sum"}:
        raise ValueError(f"unsupported pooling mode: {pooling}")

    batch_size, seq_len, _ = hidden_states.shape
    num_patches = patch_lengths.shape[1]
    if patch_ids is None:
        patch_ids = patch_ids_from_lengths(patch_lengths, seq_len)

    membership = patch_membership_mask(patch_ids, num_patches, patches_as_queries=True)
    if token_mask is not None:
        membership = membership & token_mask[:, None, :]

    weights = membership.astype(hidden_states.dtype)
    pooled = weights @ hidden_states
    if pooling == "mean":
        counts = mx.sum(weights, axis=-1, keepdims=True)
        pooled = pooled / mx.maximum(counts, 1.0)
    return pooled


def patch_start_mask_from_lengths(patch_lengths: mx.array, seq_len: int) -> mx.array:
    """Byte positions that begin a patch."""
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")
    if seq_len == 0:
        return mx.zeros((patch_lengths.shape[0], 0), dtype=mx.bool_)

    starts = mx.cumsum(patch_lengths, axis=-1) - patch_lengths
    valid = patch_lengths > 0
    if bool(mx.any(valid & (starts >= seq_len))):
        raise ValueError("valid patch starts must be within the sequence length")
    positions = mx.arange(seq_len, dtype=patch_lengths.dtype).reshape(1, seq_len, 1)
    return mx.any((positions == starts[:, None, :]) & valid[:, None, :], axis=-1)
