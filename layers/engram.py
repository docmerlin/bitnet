"""DeepSeek Engram-style conditional N-gram memory."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig


class Engram(nn.Module):
    """Retrieve hashed token N-grams and gate them into hidden states."""

    def __init__(self, config: TernaryConfig, layer_id: int):
        super().__init__()
        self.pad_id = config.engram_pad_id
        self.max_ngram_size = config.engram_max_ngram_size
        self.num_heads = config.engram_num_heads
        self.table_size = config.engram_vocab_size
        self.kernel_size = config.engram_kernel_size

        num_ngrams = self.max_ngram_size - 1
        num_tables = num_ngrams * self.num_heads
        self.embedding = nn.Embedding(
            num_tables * self.table_size,
            config.engram_head_dim,
        )
        self.register_buffer(
            "offsets",
            torch.arange(num_tables, dtype=torch.long) * self.table_size,
        )

        generator = torch.Generator().manual_seed(config.engram_seed + 10_007 * layer_id)
        multipliers = torch.randint(
            1,
            2**31,
            (self.max_ngram_size, self.num_heads),
            generator=generator,
            dtype=torch.long,
        )
        self.register_buffer("multipliers", multipliers | 1)

        memory_size = num_tables * config.engram_head_dim
        self.key_proj = nn.Linear(memory_size, config.hidden_size, bias=False)
        self.value_proj = nn.Linear(memory_size, config.hidden_size, bias=False)
        self.key_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.query_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.short_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=self.kernel_size,
            groups=config.hidden_size,
            bias=False,
        )

    @staticmethod
    def _valid_tokens(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            return attention_mask.to(device=input_ids.device, dtype=torch.bool)
        return torch.ones_like(input_ids, dtype=torch.bool)

    def hash_ids(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return deterministic table-local IDs with shape ``(B, T, tables)``."""
        input_ids = input_ids.to(dtype=torch.long)
        valid = self._valid_tokens(input_ids, attention_mask)
        batch_size, seq_len = input_ids.shape
        hashes = []

        for ngram_size in range(2, self.max_ngram_size + 1):
            mixed = input_ids.unsqueeze(-1) * self.multipliers[0]
            for lag in range(1, ngram_size):
                shifted = input_ids.new_full((batch_size, seq_len), self.pad_id)
                shifted[:, lag:] = input_ids[:, :-lag]

                keep = torch.zeros_like(valid)
                keep[:, lag:] = valid[:, lag:] & valid[:, :-lag]
                if segment_ids is not None:
                    keep[:, lag:] &= segment_ids[:, lag:] == segment_ids[:, :-lag]
                shifted = torch.where(keep, shifted, self.pad_id)
                mixed = torch.bitwise_xor(
                    mixed,
                    shifted.unsqueeze(-1) * self.multipliers[lag],
                )
            hashes.append(torch.remainder(mixed, self.table_size))

        return torch.cat(hashes, dim=-1)

    def _causal_short_conv(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.conv_norm(x)
        if segment_ids is None and bool(valid.all()):
            y = F.conv1d(
                x.transpose(1, 2),
                self.short_conv.weight,
                padding=self.kernel_size - 1,
                groups=x.size(-1),
            )[..., : x.size(1)]
            return F.silu(y.transpose(1, 2))

        output = torch.zeros_like(x)
        for lag in range(self.kernel_size):
            shifted = torch.zeros_like(x)
            if lag == 0:
                shifted = x
                keep = valid
            else:
                shifted[:, lag:] = x[:, :-lag]
                keep = torch.zeros_like(valid)
                keep[:, lag:] = valid[:, lag:] & valid[:, :-lag]
                if segment_ids is not None:
                    keep[:, lag:] &= segment_ids[:, lag:] == segment_ids[:, :-lag]
            weight = self.short_conv.weight[:, 0, self.kernel_size - 1 - lag]
            output = output + shifted * keep.unsqueeze(-1) * weight
        return F.silu(output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hash_ids = self.hash_ids(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        memory = self.embedding(hash_ids + self.offsets).flatten(start_dim=-2)

        key = self.key_norm(self.key_proj(memory))
        query = self.query_norm(hidden_states)
        score = (key * query).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
        score = score.sign() * score.abs().clamp_min(1e-6).sqrt()
        value = score.sigmoid().unsqueeze(-1) * self.value_proj(memory)

        valid = self._valid_tokens(input_ids, attention_mask)
        output = value + self._causal_short_conv(value, valid, segment_ids)
        return output * valid.unsqueeze(-1)
