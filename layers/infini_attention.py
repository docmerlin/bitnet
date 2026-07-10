"""Infini-Attention with per-head memory gating.

Pure attention sublayer inside ``HybridTransformerBlock``. The block owns
pre-norm and AttnRes; this module mixes local block-causal attention with
compressive memory attention.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig
from layers.h_bitlinear import HBitLinear
from utils import (
    apply_rotary_emb,
    build_rope_cache,
    causal_block_attention_bias,
    combine_attention_bias,
)


class InfiniAttention(nn.Module):
    """Attention with compressive memory and per-head gating."""

    def __init__(self, config: TernaryConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        self.memory_dim = config.infini_memory_dim
        self.config = config
        self.num_blocks = max(1, config.block_size)

        self.qkv = HBitLinear(hidden_size, hidden_size * 3, bias=False, config=config)
        self.o_proj = HBitLinear(hidden_size, hidden_size, bias=False, config=config)
        self.gate = nn.Parameter(torch.zeros(num_heads))
        self.update_memory_buffers = True

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.register_buffer(
            "memory_k",
            torch.zeros(num_heads, self.memory_dim, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_v",
            torch.zeros(num_heads, self.memory_dim, self.head_dim),
            persistent=False,
        )
        # RoPE cache: (seq_len, device, dtype, scale_factor) -> (cos, sin)
        self._rope_cache_key: Optional[tuple] = None
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

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
    def use_memory_state(self, state: dict[str, torch.Tensor], *, update_memory_buffers: bool = True):
        """Temporarily load ``state`` (and optionally freeze writes).

        Intended for gradient-checkpoint **recompute** only: on exit, restores the
        prior buffers/flag so a no-write recompute cannot clobber post-forward memory.
        """
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
        rope_scaling = self.config.rope_scaling
        if not rope_scaling:
            return 1.0
        original = rope_scaling.get("original_max_position_embeddings", seq_len)
        factor = rope_scaling.get("factor", 1.0)
        return float(factor if seq_len > original else 1.0)

    def _get_rope(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._rope_scaling_factor(seq_len)
        key = (seq_len, device.type, device.index, str(dtype), scale)
        if (
            self._rope_cache_key != key
            or self._rope_cos is None
            or self._rope_sin is None
            or self._rope_cos.device != device
        ):
            cos, sin = build_rope_cache(
                seq_len=seq_len,
                dim=self.head_dim,
                theta=self.config.rope_theta,
                scaling_factor=scale,
                device=device,
            )
            self._rope_cache_key = key
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos, self._rope_sin

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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        update_memory: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self._get_rope(seq_len, device=x.device, dtype=q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        scale = self.head_dim ** -0.5

        # Shared bias from BitNetDeep.forward when available; else build here.
        if attn_bias is None:
            base_bias = causal_block_attention_bias(
                seq_len, self.num_blocks, dtype=q.dtype, device=x.device
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
        else:
            attn_bias = attn_bias.to(dtype=q.dtype)

        local_context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
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

        # Always readable. Writes when requested AND module flag allows (ckpt recompute
        # freezes the flag). Allowed in eval too so multi-segment decode can accumulate
        # with reset_memory=False.
        requested = self.update_memory_buffers if update_memory is None else bool(update_memory)
        do_update = requested and self.update_memory_buffers
        if do_update:
            with torch.no_grad():
                self._update_memory(k, v)

        return output
