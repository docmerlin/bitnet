"""Cross-attention modules for byte/patch communication."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.config import TernaryBLTConfig
from layers.h_bitlinear import HBitLinear


class TernaryCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        *,
        hidden_dim: int,
        num_heads: int,
        config: TernaryBLTConfig,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (hidden_dim // num_heads) % 2 != 0:
            raise ValueError("cross-attention head_dim must be even")

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim or query_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = config.dropout

        self.query_norm = nn.RMSNorm(query_dim)
        self.kv_norm = nn.RMSNorm(kv_dim)
        self.q_proj = HBitLinear(query_dim, hidden_dim, config=config)
        self.k_proj = HBitLinear(kv_dim, hidden_dim, config=config)
        self.v_proj = HBitLinear(kv_dim, hidden_dim, config=config)
        self.out_proj = HBitLinear(hidden_dim, self.output_dim, config=config)
        self.residual_proj = None
        if query_dim != self.output_dim:
            self.residual_proj = HBitLinear(query_dim, self.output_dim, config=config)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        kv_len = key_value.size(1)

        q = self.q_proj(self.query_norm(query)).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.kv_norm(key_value)).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.kv_norm(key_value)).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_floor = torch.finfo(scores.dtype).min
        query_valid = None

        if mask is not None:
            if mask.dtype == torch.bool:
                bool_mask = mask[:, None, :, :]
                query_valid = bool_mask.any(dim=-1, keepdim=True)
                scores = scores.masked_fill(~bool_mask, mask_floor)
            elif mask.ndim == 3:
                additive_mask = mask[:, None, :, :].to(dtype=scores.dtype)
                scores = scores + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            elif mask.ndim == 4:
                additive_mask = mask[:, :, :, :].to(dtype=scores.dtype)
                scores = scores + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            else:
                raise ValueError("cross-attention mask must be bool, 3D additive, or 4D additive")

        if query_valid is not None:
            scores = scores.masked_fill(~query_valid, 0.0)

        probs = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            probs = F.dropout(probs, p=self.dropout, training=self.training)
        if query_valid is not None:
            probs = probs.masked_fill(~query_valid, 0.0)

        context = torch.matmul(probs, v)
        if query_valid is not None:
            context = context.masked_fill(~query_valid, 0.0)

        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        residual = self.residual_proj(query) if self.residual_proj is not None else query
        return residual + self.out_proj(context)
