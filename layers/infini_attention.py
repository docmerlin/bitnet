"""Infini-Attention with per-head memory gating.

This module is the pure attention sublayer used inside ``HybridTransformerBlock``.
The block owns pre-norm and AttnRes residual handling; this module only produces
the attention output after mixing local block-causal attention with compressive
memory attention.
"""

from __future__ import annotations

import contextlib
import math
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
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        self.memory_dim = memory_dim
        self.config = config
        self.num_blocks = max(1, getattr(config, "block_size", 1))

        self.qkv = HBitLinear(hidden_size, hidden_size * 3, bias=False, config=config)
        self.o_proj = HBitLinear(hidden_size, hidden_size, bias=False, config=config)
        self.gate = nn.Parameter(torch.zeros(num_heads))
        self.update_memory_buffers = True

        # This is transient recurrence state, not learned model state.
        self.register_buffer("memory_k", torch.zeros(num_heads, memory_dim, self.head_dim), persistent=False)
        self.register_buffer("memory_v", torch.zeros(num_heads, memory_dim, self.head_dim), persistent=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Older checkpoints may still contain serialized memory buffers.
        state_dict.pop(f"{prefix}memory_k", None)
        state_dict.pop(f"{prefix}memory_v", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_memory(self) -> None:
        self.memory_k.zero_()
        self.memory_v.zero_()

    def get_memory_state(self) -> dict[str, torch.Tensor]:
        return {
            "memory_k": self.memory_k.detach().clone(),
            "memory_v": self.memory_v.detach().clone(),
        }

    def load_memory_state(self, state: dict[str, torch.Tensor]) -> None:
        self.memory_k.copy_(state["memory_k"].to(device=self.memory_k.device, dtype=self.memory_k.dtype))
        self.memory_v.copy_(state["memory_v"].to(device=self.memory_v.device, dtype=self.memory_v.dtype))

    @contextlib.contextmanager
    def no_memory_updates(self):
        previous = self.update_memory_buffers
        self.update_memory_buffers = False
        try:
            yield
        finally:
            self.update_memory_buffers = previous

    @contextlib.contextmanager
    def use_memory_state(self, state: dict[str, torch.Tensor], *, update_memory_buffers: bool = True):
        previous_state = self.get_memory_state()
        previous_update = self.update_memory_buffers
        self.load_memory_state(state)
        self.update_memory_buffers = update_memory_buffers
        try:
            yield
        finally:
            self.update_memory_buffers = previous_update
            self.load_memory_state(previous_state)

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

        num_blocks = max(1, min(self.num_blocks, seq_len))
        block_size = math.ceil(seq_len / num_blocks)
        positions = torch.arange(seq_len, device=x.device)
        block_ids = positions.div(block_size, rounding_mode="floor")
        block_mask = block_ids[:, None] != block_ids[None, :]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        invalid_local = causal_mask | block_mask
        local_scores = local_scores.masked_fill(
            invalid_local.view(1, 1, seq_len, seq_len),
            torch.finfo(local_scores.dtype).min,
        )

        mask_floor = torch.finfo(local_scores.dtype).min
        query_valid = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                query_valid = attention_mask[:, None, :seq_len, None].to(torch.bool)
                key_valid = attention_mask[:, None, None, :seq_len].to(torch.bool)
                local_scores = local_scores.masked_fill(~key_valid, mask_floor)
            elif attention_mask.ndim == 3:
                additive_mask = attention_mask[:, None, :seq_len, :seq_len].to(dtype=local_scores.dtype)
                local_scores = local_scores + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            elif attention_mask.ndim == 4:
                additive_mask = attention_mask[:, :, :seq_len, :seq_len].to(dtype=local_scores.dtype)
                local_scores = local_scores + additive_mask
                query_valid = additive_mask.amax(dim=-1, keepdim=True) >= 0
            else:
                raise ValueError("attention_mask must be 2D, 3D, or 4D")

        if query_valid is not None:
            # Fully masked query rows must stay zero instead of normalizing into
            # a synthetic attention distribution.
            local_scores = local_scores.masked_fill(~query_valid, 0.0)

        local_probs = torch.softmax(local_scores, dim=-1)
        if query_valid is not None:
            local_probs = local_probs.masked_fill(~query_valid, 0.0)

        local_context = torch.matmul(local_probs, v)
        if query_valid is not None:
            local_context = local_context.masked_fill(~query_valid, 0.0)

        mem_k = self.memory_k.unsqueeze(0).expand(batch_size, -1, -1, -1).to(dtype=q.dtype)
        mem_v = self.memory_v.unsqueeze(0).expand(batch_size, -1, -1, -1).to(dtype=q.dtype)
        memory_scores = torch.matmul(q, mem_k.transpose(-2, -1)) * scale
        memory_probs = torch.softmax(memory_scores, dim=-1)
        memory_context = torch.matmul(memory_probs, mem_v)
        if query_valid is not None:
            memory_context = memory_context.masked_fill(~query_valid, 0.0)

        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        context = (1.0 - gate) * local_context + gate * memory_context
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)

        if self.training and self.update_memory_buffers:
            with torch.no_grad():
                self._update_memory(k, v)

        return output


if __name__ == "__main__":
    layer = InfiniAttention()
    x = torch.randn(2, 128, 1024)
    y = layer(x)
    print(f"InfiniAttention test passed. Output shape: {y.shape}")
