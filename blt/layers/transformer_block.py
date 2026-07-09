"""Transformer primitives for the ternary BLT student."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.config import TernaryBLTConfig
from layers.h_bitlinear import HBitLinear
from utils import apply_rotary_emb, build_rope_cache, causal_window_attention_bias, combine_attention_bias


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

        base_bias = (
            causal_window_attention_bias(seq_len, self.local_window, dtype=q.dtype, device=x.device)
            if self.causal
            else None
        )
        attn_bias, query_valid = combine_attention_bias(
            attention_mask,
            base_bias=base_bias,
            batch_size=batch_size,
            q_len=seq_len,
            k_len=seq_len,
            dtype=q.dtype,
            device=x.device,
        )

        dropout_p = self.dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        if query_valid is not None:
            context = context.masked_fill(~query_valid, 0.0)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.o_proj(context)


class TernaryMLP(nn.Module):
    """SwiGLU expand → mid (silu) → down; 4 HBitLinears / 3 stages.

    Shared by local-encoder, global, and local-decoder ``TransformerBlock``s.
    ``gate_proj``/``up_proj`` are separate (unlike hybrid dense which fuses them
    into one ``ffn_up`` of width ``2*inter``).
    """

    def __init__(self, dim: int, multiplier: float, *, config: TernaryBLTConfig) -> None:
        super().__init__()
        hidden_dim = max(int(dim * multiplier), dim)
        self.hidden_dim = hidden_dim
        self.gate_proj = HBitLinear(dim, hidden_dim, config=config)
        self.up_proj = HBitLinear(dim, hidden_dim, config=config)
        self.mid_proj = HBitLinear(hidden_dim, hidden_dim, config=config)
        self.down_proj = HBitLinear(hidden_dim, dim, config=config)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        hidden = F.silu(self.mid_proj(hidden))
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
