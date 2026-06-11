"""Transformer primitives for the ternary BLT student."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.config import TernaryBLTConfig
from layers.h_bitlinear import HBitLinear
from utils import apply_rotary_emb, build_rope_cache


class TernarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        config: TernaryBLTConfig,
        local_window: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_window = local_window
        self.causal = causal
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta

        self.q_proj = HBitLinear(dim, dim, config=config)
        self.k_proj = HBitLinear(dim, dim, config=config)
        self.v_proj = HBitLinear(dim, dim, config=config)
        self.o_proj = HBitLinear(dim, dim, config=config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(seq_len, self.head_dim, theta=self.rope_theta, device=x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        mask_floor = torch.finfo(q.dtype).min
        attn_bias = None
        if self.causal:
            q_positions = torch.arange(seq_len, device=x.device).view(seq_len, 1)
            k_positions = torch.arange(seq_len, device=x.device).view(1, seq_len)
            invalid = k_positions > q_positions
            if self.local_window is not None:
                invalid |= (q_positions - k_positions) >= self.local_window
            attn_bias = torch.zeros(seq_len, seq_len, dtype=q.dtype, device=x.device)
            attn_bias = attn_bias.masked_fill(invalid, mask_floor).view(1, 1, seq_len, seq_len)

        query_valid = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                key_valid = attention_mask[:, None, None, :seq_len].to(torch.bool)
                if attn_bias is None:
                    attn_bias = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=q.dtype, device=x.device)
                else:
                    attn_bias = attn_bias.expand(batch_size, 1, seq_len, seq_len).clone()
                attn_bias.masked_fill_(~key_valid, mask_floor)
                query_valid = attention_mask[:, None, :seq_len, None].to(torch.bool)
            elif attention_mask.ndim == 3:
                additive_mask = attention_mask[:, None, :seq_len, :seq_len].to(dtype=q.dtype)
                attn_bias = additive_mask if attn_bias is None else attn_bias + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            elif attention_mask.ndim == 4:
                additive_mask = attention_mask[:, :, :seq_len, :seq_len].to(dtype=q.dtype)
                attn_bias = additive_mask if attn_bias is None else attn_bias + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            else:
                raise ValueError("attention_mask must be 2D, 3D, or 4D")

        dropout_p = self.dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        if query_valid is not None:
            context = context.masked_fill(~query_valid, 0.0)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.o_proj(context)


class TernaryMLP(nn.Module):
    def __init__(self, dim: int, multiplier: float, *, config: TernaryBLTConfig) -> None:
        super().__init__()
        hidden_dim = max(int(dim * multiplier), dim)
        self.gate_proj = HBitLinear(dim, hidden_dim, config=config)
        self.up_proj = HBitLinear(dim, hidden_dim, config=config)
        self.down_proj = HBitLinear(hidden_dim, dim, config=config)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.dropout > 0.0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        config: TernaryBLTConfig,
        ffn_multiplier: float,
        local_window: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)
        self.attn = TernarySelfAttention(
            dim,
            num_heads,
            config=config,
            local_window=local_window,
            causal=causal,
        )
        self.mlp = TernaryMLP(dim, ffn_multiplier, config=config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
