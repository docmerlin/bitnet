"""Teacher-forced patch utilities for BLT distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def build_uniform_patch_lengths(
    batch_size: int,
    seq_len: int,
    patch_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    num_patches = (seq_len + patch_size - 1) // patch_size
    patch_lengths = torch.zeros(batch_size, num_patches, dtype=torch.long, device=device)
    patch_lengths[:, :-1] = patch_size
    patch_lengths[:, -1] = seq_len - patch_size * (num_patches - 1)
    return patch_lengths


def patch_ids_from_lengths(patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")
    cumulative = patch_lengths.cumsum(dim=-1)
    positions = torch.arange(seq_len, device=patch_lengths.device).view(1, 1, seq_len)
    return (positions >= cumulative.unsqueeze(-1)).sum(dim=1)


def normalize_patch_lengths_to_targets(
    patch_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")
    if target_lengths.ndim != 1 or target_lengths.size(0) != patch_lengths.size(0):
        raise ValueError("target_lengths must be shaped [batch]")

    normalized = patch_lengths.clone().to(dtype=torch.long)
    for row, target_length in zip(normalized, target_lengths.tolist()):
        if target_length < 0:
            raise ValueError("target lengths must be non-negative")
        if target_length == 0:
            row.zero_()
            continue
        row.copy_(normalize_patch_lengths(row.unsqueeze(0), target_length)[0])
    return normalized


def patch_presence_mask(patch_lengths: torch.Tensor) -> torch.Tensor:
    return patch_lengths > 0


def patch_start_mask_from_lengths(patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    if patch_lengths.ndim != 2:
        raise ValueError("patch_lengths must be shaped [batch, num_patches]")
    mask = torch.zeros(patch_lengths.size(0), seq_len, dtype=torch.bool, device=patch_lengths.device)
    if seq_len == 0:
        return mask

    starts = torch.cat(
        [
            torch.zeros(patch_lengths.size(0), 1, dtype=torch.long, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1)[:, :-1],
        ],
        dim=1,
    )
    valid = patch_lengths > 0
    if torch.any(valid):
        batch_indices, patch_indices = torch.nonzero(valid, as_tuple=True)
        start_indices = starts[batch_indices, patch_indices]
        if torch.any(start_indices >= seq_len):
            raise ValueError("valid patch starts must be within the sequence length")
        mask[batch_indices, start_indices] = True
    return mask


def patch_membership_mask(
    patch_ids: torch.Tensor,
    num_patches: int,
    *,
    patches_as_queries: bool,
) -> torch.Tensor:
    patch_range = torch.arange(num_patches, device=patch_ids.device)
    if patches_as_queries:
        return patch_range.view(1, num_patches, 1) == patch_ids.unsqueeze(1)
    return patch_ids.unsqueeze(-1) == patch_range.view(1, 1, num_patches)


def normalize_patch_lengths(patch_lengths: torch.Tensor, target_length: int) -> torch.Tensor:
    """Trim or extend trailing patches so lengths sum to ``target_length``."""

    normalized = patch_lengths.clone().to(dtype=torch.long)
    for row in normalized:
        total = int(row.sum().item())
        if total == target_length:
            continue

        if target_length < 0:
            raise ValueError("target_length must be non-negative")
        if target_length == 0:
            row.zero_()
            continue

        valid = torch.nonzero(row > 0, as_tuple=False).flatten()
        if valid.numel() == 0:
            row[0] = target_length
            continue

        if total > target_length:
            overflow = total - target_length
            for index in reversed(valid.tolist()):
                if overflow <= 0:
                    break
                removable = min(int(row[index].item()), overflow)
                row[index] -= removable
                overflow -= removable
            if overflow != 0:
                raise ValueError("could not normalize patch lengths down to the target length")
        else:
            row[valid[-1]] += target_length - total

    return normalized


def pool_patch_representations(
    hidden_states: torch.Tensor,
    patch_lengths: torch.Tensor,
    *,
    patch_ids: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    pooling: str = "mean",
) -> torch.Tensor:
    if pooling not in {"mean", "sum"}:
        raise ValueError(f"unsupported pooling mode: {pooling}")

    batch_size, seq_len, hidden_dim = hidden_states.shape
    if patch_ids is None:
        patch_ids = patch_ids_from_lengths(patch_lengths, seq_len)

    if token_mask is None:
        token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
    else:
        token_mask = token_mask.to(device=hidden_states.device, dtype=torch.bool)

    max_num_patches = patch_lengths.size(1)
    pooled = hidden_states.new_zeros(batch_size, max_num_patches, hidden_dim)
    counts = hidden_states.new_zeros(batch_size, max_num_patches, 1)
    ones = hidden_states.new_ones(seq_len, 1)

    for batch_index in range(batch_size):
        valid = token_mask[batch_index]
        if not torch.any(valid):
            continue
        pooled[batch_index].index_add_(0, patch_ids[batch_index][valid], hidden_states[batch_index][valid])
        counts[batch_index].index_add_(0, patch_ids[batch_index][valid], ones[: valid.sum()])

    if pooling == "mean":
        pooled = pooled / counts.clamp_min(1.0)
    return pooled


@dataclass(slots=True)
class UniformPatcher:
    patch_size: int

    def patch(self, tokens: torch.Tensor) -> torch.Tensor:
        return build_uniform_patch_lengths(tokens.size(0), tokens.size(1), self.patch_size, device=tokens.device)
