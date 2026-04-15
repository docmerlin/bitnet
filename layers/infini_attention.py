"""Infini-Attention with per-head memory gating.

The current model uses Block Attention Residuals in all 64 layers, but this
module remains available for future hybrid experiments. The implementation keeps
local token attention and memory attention separate, then blends their context
vectors with a learned per-head gate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.h_bitlinear import HBitLinear
from utils import apply_rotary_emb, build_rope_cache


class InfiniAttention(nn.Module):
    """Attention with compressive memory and per-head gating."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 32,
        memory_dim: int = 64,
        config: Optional[object] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.memory_dim = memory_dim
        self.config = config

        norm_eps = getattr(config, "rms_norm_eps", 1e-5)
        self.qkv = HBitLinear(hidden_size, hidden_size * 3, bias=False, config=config)
        self.o_proj = HBitLinear(hidden_size, hidden_size, bias=False, config=config)
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.gate = nn.Parameter(torch.zeros(num_heads))

        self.register_buffer("memory_k", torch.zeros(num_heads, memory_dim, self.head_dim))
        self.register_buffer("memory_v", torch.zeros(num_heads, memory_dim, self.head_dim))

    def _rope_scaling_factor(self, seq_len: int) -> float:
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if not rope_scaling:
            return 1.0

        original = rope_scaling.get("original_max_position_embeddings", seq_len)
        factor = rope_scaling.get("factor", 1.0)
        return float(factor if seq_len > original else 1.0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Compress the current sequence into the fixed-size memory buffers."""
        batch_size, num_heads, seq_len, head_dim = k.shape

        pooled_k = F.adaptive_avg_pool1d(
            k.permute(0, 1, 3, 2).reshape(batch_size * num_heads, head_dim, seq_len),
            self.memory_dim,
        )
        pooled_v = F.adaptive_avg_pool1d(
            v.permute(0, 1, 3, 2).reshape(batch_size * num_heads, head_dim, seq_len),
            self.memory_dim,
        )

        pooled_k = pooled_k.transpose(1, 2).reshape(batch_size, num_heads, self.memory_dim, head_dim).mean(dim=0)
        pooled_v = pooled_v.transpose(1, 2).reshape(batch_size, num_heads, self.memory_dim, head_dim).mean(dim=0)

        self.memory_k.mul_(0.99).add_(0.01 * pooled_k)
        self.memory_v.mul_(0.99).add_(0.01 * pooled_v)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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

        scale = self.head_dim ** -0.5
        local_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        local_scores = local_scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), torch.finfo(local_scores.dtype).min)

        if attention_mask is not None and attention_mask.ndim == 2:
            key_valid = attention_mask[:, None, None, :seq_len].to(torch.bool)
            local_scores = local_scores.masked_fill(~key_valid, torch.finfo(local_scores.dtype).min)

        local_probs = torch.softmax(local_scores, dim=-1)
        local_context = torch.matmul(local_probs, v)

        mem_k = self.memory_k.unsqueeze(0).expand(batch_size, -1, -1, -1).to(dtype=q.dtype)
        mem_v = self.memory_v.unsqueeze(0).expand(batch_size, -1, -1, -1).to(dtype=q.dtype)
        memory_scores = torch.matmul(q, mem_k.transpose(-2, -1)) * scale
        memory_probs = torch.softmax(memory_scores, dim=-1)
        memory_context = torch.matmul(memory_probs, mem_v)

        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        context = (1.0 - gate) * local_context + gate * memory_context
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)

        with torch.no_grad():
            self._update_memory(k, v)

        return residual + output


if __name__ == "__main__":
    layer = InfiniAttention()
    x = torch.randn(2, 128, 1024)
    y = layer(x)
    print(f"InfiniAttention test passed. Output shape: {y.shape}")
