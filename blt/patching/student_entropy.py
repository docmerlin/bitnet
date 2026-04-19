"""Student-side boundary model for future standalone BLT patching."""

from __future__ import annotations

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.patching.teacher_patcher import normalize_patch_lengths


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def _split_segment_length(length: int, max_patch_length: int | None) -> list[int]:
    if max_patch_length is None or max_patch_length <= 0:
        return [length]
    chunks: list[int] = []
    remaining = length
    while remaining > max_patch_length:
        chunks.append(max_patch_length)
        remaining -= max_patch_length
    chunks.append(remaining)
    return chunks


def _patch_lengths_from_boundary_mask(boundary_mask: torch.Tensor, *, max_patch_length: int | None = None) -> torch.Tensor:
    batch_lengths: list[list[int]] = []
    max_patches = 0
    for row in boundary_mask.to(dtype=torch.bool):
        starts = torch.nonzero(row, as_tuple=False).flatten().tolist()
        if not starts or starts[0] != 0:
            starts = [0] + starts
        starts.append(row.numel())
        lengths: list[int] = []
        for start_index, end_index in zip(starts, starts[1:]):
            lengths.extend(_split_segment_length(end_index - start_index, max_patch_length))
        batch_lengths.append(lengths)
        max_patches = max(max_patches, len(lengths))

    patch_lengths = boundary_mask.new_zeros((boundary_mask.size(0), max_patches), dtype=torch.long)
    for index, lengths in enumerate(batch_lengths):
        patch_lengths[index, : len(lengths)] = torch.tensor(lengths, dtype=torch.long, device=boundary_mask.device)
    return patch_lengths


class StudentEntropyModel(nn.Module):
    """Small full-precision boundary model for later student-driven patching.

    The first distillation path uses teacher patching directly. This module exists
    so the separate BLT package already has a place for future boundary learning.
    """

    def __init__(
        self,
        config: TernaryBLTConfig,
        *,
        dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("StudentEntropyModel dim must be divisible by num_heads")
        self.max_patch_length = config.max_patch_length
        self.embedding = nn.Embedding(config.vocab_size, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.RMSNorm(dim)
        self.boundary_head = nn.Linear(dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        hidden = self.encoder(hidden, mask=_causal_mask(input_ids.size(1), input_ids.device))
        hidden = self.norm(hidden)
        return self.boundary_head(hidden).squeeze(-1)

    def predict_patch_lengths_from_logits(
        self,
        boundary_logits: torch.Tensor,
        *,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        boundary_mask = boundary_logits > threshold
        boundary_mask[:, 0] = True
        patch_lengths = _patch_lengths_from_boundary_mask(boundary_mask, max_patch_length=self.max_patch_length)
        return normalize_patch_lengths(patch_lengths, boundary_logits.size(1))

    def predict_patch_lengths(self, input_ids: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
        boundary_logits = self.forward(input_ids)
        return self.predict_patch_lengths_from_logits(boundary_logits, threshold=threshold)
