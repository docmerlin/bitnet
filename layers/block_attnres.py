"""Block Attention Residuals (STTNRes) for deep ternary models.

Splits the sequence into local blocks and computes attention only within each
block. This improves gradient flow in very deep (64+) networks. Supports
runtime change of `num_blocks` for progressive block growth during training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.h_bitlinear import HBitLinear
from utils import apply_rotary_emb, build_rope_cache


class BlockAttentionResidual(nn.Module):
    """Causal block-wise attention with residual and MLP sublayers."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 32,
        block_size: int = 4,
        config: Optional[object] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        self.num_blocks = max(1, block_size)
        self.config = config

        intermediate_size = getattr(config, "intermediate_size", hidden_size * 2)
        norm_eps = getattr(config, "rms_norm_eps", 1e-5)

        self.attn_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.qkv = HBitLinear(hidden_size, hidden_size * 3, bias=False, config=config)
        self.o_proj = HBitLinear(hidden_size, hidden_size, bias=False, config=config)
        self.ffn_up = HBitLinear(hidden_size, intermediate_size * 2, bias=False, config=config)
        self.ffn_down = HBitLinear(intermediate_size, hidden_size, bias=False, config=config)

    def _rope_scaling_factor(self, seq_len: int) -> float:
        """Simple YaRN-style scaling factor."""
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if not rope_scaling:
            return 1.0
        original = rope_scaling.get("original_max_position_embeddings", seq_len)
        factor = rope_scaling.get("factor", 1.0)
        return float(factor if seq_len > original else 1.0)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Block-wise causal attention + FFN with progressive block support."""
        residual = x
        x = self.attn_norm(x)

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        rope_theta = float(getattr(self.config, "rope_theta", 10000.0))
        rope_scale = self._rope_scaling_factor(seq_len)
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            dim=self.head_dim,
            theta=rope_theta,
            scaling_factor=rope_scale,
            device=x.device,
        )
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask for local blocks
        num_blocks = max(1, min(self.num_blocks, seq_len))
        block_size = math.ceil(seq_len / num_blocks)
        positions = torch.arange(seq_len, device=x.device)
        block_ids = positions.div(block_size, rounding_mode="floor")
        block_mask = block_ids[:, None] != block_ids[None, :]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        invalid_attn = causal_mask | block_mask

        attn = attn.masked_fill(
            invalid_attn.view(1, 1, seq_len, seq_len),
            torch.finfo(attn.dtype).min,
        )

        mask_floor = torch.finfo(attn.dtype).min
        query_valid = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                query_valid = attention_mask[:, None, :seq_len, None].to(torch.bool)
                key_valid = attention_mask[:, None, None, :seq_len].to(torch.bool)
                attn = attn.masked_fill(~key_valid, mask_floor)
            elif attention_mask.ndim == 3:
                additive_mask = attention_mask[:, None, :seq_len, :seq_len].to(dtype=attn.dtype)
                attn = attn + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            elif attention_mask.ndim == 4:
                additive_mask = attention_mask[:, :, :seq_len, :seq_len].to(dtype=attn.dtype)
                attn = attn + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            else:
                raise ValueError("attention_mask must be 2D, 3D, or 4D")

        if query_valid is not None:
            # Rows that are fully masked need explicit zeroing so softmax does not
            # turn them into a uniform distribution.
            attn = attn.masked_fill(~query_valid, 0.0)

        attn = F.softmax(attn, dim=-1)
        if query_valid is not None:
            attn = attn.masked_fill(~query_valid, 0.0)

        context = torch.matmul(attn, v)
        if query_valid is not None:
            context = context.masked_fill(~query_valid, 0.0)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        x = residual + self.o_proj(context)

        # FFN (using HBitLinear for down projection)
        residual = x
        x = self.ffn_norm(x)
        gate, value = self.ffn_up(x).chunk(2, dim=-1)
        x = self.ffn_down(F.silu(gate) * value)
        return residual + x


if __name__ == "__main__":
    block = BlockAttentionResidual()
    x = torch.randn(2, 64, 1024)
    y = block(x)
    print(f"BlockAttentionResidual test passed. Output shape: {y.shape}")
