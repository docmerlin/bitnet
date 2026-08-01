"""Hashed byte n-gram embeddings for the ternary BLT student."""

from __future__ import annotations

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.ngram_hash import HASH_MODULUS, HASH_PAD, hash_bases
from layers.h_bitlinear import HBitLinear


class HashNgramEmbedding(nn.Module):
    """Hashed byte n-gram embeddings summed into the byte embedding.

    Meta's BLT, eq. 3: ``e_i = (x_i + sum_n E_n[Hash(g_{i,n})]) / (|sizes| + 1)``.
    See :mod:`blt.ngram_hash` for why this exists and what the hash guarantees.
    Structural mirror of :class:`blt.mlx_layers.MLXHashNgramEmbedding`; both index
    the same tables, so the hash must agree bit for bit.
    """

    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.sizes = tuple(config.ngram_sizes)
        self.vocab_size = int(config.ngram_vocab_size)
        self.dim = int(config.ngram_dim)
        self.embedding = nn.Embedding(len(self.sizes) * self.vocab_size, self.dim)
        self.proj = (
            HBitLinear(self.dim, config.local_dim, config=config)
            if self.dim != config.local_dim
            else None
        )
        # Derived constants, deliberately not buffers: they would otherwise be
        # absent from this state dict but present in the MLX parameter tree,
        # breaking the name-for-name parity both stacks are built on.
        self.bases = hash_bases(max(self.sizes))

    def hashes(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(indices, valid)`` per n-gram size, both ``[len(sizes), B, L]``."""
        batch, length = input_ids.shape
        ids = input_ids.to(torch.long)
        running = torch.zeros((batch, length), dtype=torch.long, device=ids.device)
        wanted = {size: slot for slot, size in enumerate(self.sizes)}
        indices: list[torch.Tensor] = [None] * len(self.sizes)
        valid: list[torch.Tensor] = [None] * len(self.sizes)
        positions = torch.arange(length, device=ids.device)
        for lag in range(max(self.sizes)):
            if lag == 0:
                shifted = ids
            elif lag < length:
                pad = ids.new_full((batch, lag), HASH_PAD)
                shifted = torch.cat((pad, ids[:, : length - lag]), dim=1)
            else:
                shifted = ids.new_full((batch, length), HASH_PAD)
            running = (running + shifted * self.bases[lag]) % HASH_MODULUS
            size = lag + 1
            if size in wanted:
                slot = wanted[size]
                indices[slot] = running % self.vocab_size
                valid[slot] = (positions.unsqueeze(0) >= size - 1).expand(batch, length)
        return torch.stack(indices), torch.stack(valid)

    def forward(
        self,
        byte_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        indices, valid = self.hashes(input_ids)
        if attention_mask is not None:
            # Padding is a suffix, so a position being real implies its whole
            # backward window is real; masking the position is enough.
            valid = valid & attention_mask.to(torch.bool).unsqueeze(0)
        offsets = torch.arange(len(self.sizes), device=indices.device) * self.vocab_size
        gathered = self.embedding(indices + offsets.view(-1, 1, 1))
        gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0.0)
        total = gathered.sum(dim=0).to(byte_embeddings.dtype)
        if self.proj is not None:
            total = self.proj(total)
        # Eq. 3 is a plain sum. An earlier draft averaged over |sizes|+1, which
        # measured as dividing the byte embedding by 7 -- std 0.994 -> 0.204 --
        # because the projected n-gram term is much smaller than the byte term,
        # so the mean mostly just shrinks the latter.
        return byte_embeddings + total
