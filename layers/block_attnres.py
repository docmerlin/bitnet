"""Block Attention Residuals for the deep ternary transformer.

The sequence is split into a configurable number of local blocks. Each block
performs causal self-attention only within its own span, which keeps compute
bounded and stabilizes very deep stacks. A standard feed-forward sublayer is
included so one ``BlockAttentionResidual`` instance acts as a full transformer
block.
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
        self.num_blocks = max(1, block_size)
        self.config = config

        intermediate_size = getattr(config, "intermediate_size", hidden_size * 2)
        norm_eps = getattr(config, "rms_norm_eps", 1e-5)

        self.attn_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.qkv = HBitLinear(hidden_size, hidden_size * 3, bias=False, config=config)
        self.o_proj = HBitLinear(hidden_size, hidden_size, bias=False, config=config)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.ffn_down = HBitLinear(intermediate_size, hidden_size, bias=False, config=config)

    def _rope_scaling_factor(self, seq_len: int) -> float:
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if not rope_scaling:
            return 1.0

        original = rope_scaling.get("original_max_position_embeddings", seq_len)
        factor = rope_scaling.get("factor", 1.0)
        return float(factor if seq_len > original else 1.0)

    def _pad_sequence_dimension(self, tensor: torch.Tensor, padded_seq_len: int) -> torch.Tensor:
        pad_len = padded_seq_len - tensor.size(2)
        if pad_len <= 0:
            return tensor
        return F.pad(tensor, (0, 0, 0, pad_len))

    def _build_valid_mask(
        self,
        batch_size: int,
        seq_len: int,
        padded_seq_len: int,
        block_tokens: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        valid = torch.arange(padded_seq_len, device=device).unsqueeze(0) < seq_len
        valid = valid.expand(batch_size, -1).clone()

        if attention_mask is not None and attention_mask.ndim == 2:
            valid[:, :seq_len] &= attention_mask[:, :seq_len].to(torch.bool)

        return valid.view(batch_size, self.num_blocks, block_tokens)

    def _build_local_additive_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        block_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        padded_seq_len: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None or attention_mask.ndim != 4:
            return None

        if attention_mask.size(-1) < padded_seq_len:
            pad = padded_seq_len - attention_mask.size(-1)
            attention_mask = F.pad(attention_mask, (0, pad, 0, pad))

        local_blocks = []
        for block_index in range(self.num_blocks):
            start = block_index * block_tokens
            end = start + block_tokens
            local_blocks.append(attention_mask[:, :, start:end, start:end])

        return torch.stack(local_blocks, dim=1).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_residual = x
        x = self.attn_norm(x)

        batch_size, seq_len, _ = x.shape
        block_tokens = math.ceil(seq_len / self.num_blocks)
        padded_seq_len = self.num_blocks * block_tokens

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self._pad_sequence_dimension(q, padded_seq_len)
        k = self._pad_sequence_dimension(k, padded_seq_len)
        v = self._pad_sequence_dimension(v, padded_seq_len)

        rope_theta = float(getattr(self.config, "rope_theta", 10000.0))
        rope_scale = self._rope_scaling_factor(padded_seq_len)
        cos, sin = build_rope_cache(
            seq_len=padded_seq_len,
            dim=self.head_dim,
            theta=rope_theta,
            scaling_factor=rope_scale,
            device=x.device,
        )
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q.view(batch_size, self.num_heads, self.num_blocks, block_tokens, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_heads, self.num_blocks, block_tokens, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_heads, self.num_blocks, block_tokens, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.ones(block_tokens, block_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.view(1, 1, 1, block_tokens, block_tokens), torch.finfo(scores.dtype).min)

        valid_mask = self._build_valid_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            padded_seq_len=padded_seq_len,
            block_tokens=block_tokens,
            device=x.device,
            attention_mask=attention_mask,
        )
        query_valid = valid_mask[:, :, None, :, None]
        key_valid = valid_mask[:, :, None, None, :]
        scores = scores.masked_fill(~key_valid, torch.finfo(scores.dtype).min)

        local_additive_mask = self._build_local_additive_mask(
            attention_mask=attention_mask,
            block_tokens=block_tokens,
            dtype=scores.dtype,
            device=x.device,
            padded_seq_len=padded_seq_len,
        )
        if local_additive_mask is not None:
            scores = scores + local_additive_mask

        # Avoid NaNs for padded query rows by zeroing them before softmax and again after.
        scores = scores.masked_fill(~query_valid, 0.0)

        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~query_valid, 0.0)
        context = torch.matmul(attn, v)
        context = context.masked_fill(~query_valid, 0.0)

        context = context.transpose(1, 2).contiguous().view(batch_size, self.num_heads, padded_seq_len, self.head_dim)
        context = context[:, :, :seq_len, :].transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        x = attn_residual + self.o_proj(context)

        ffn_residual = x
        ffn_input = self.ffn_norm(x)
        gate, value = self.ffn_up(ffn_input).chunk(2, dim=-1)
        ffn_output = self.ffn_down(F.silu(gate) * value)
        return ffn_residual + ffn_output


if __name__ == "__main__":
    block = BlockAttentionResidual()
    x = torch.randn(2, 64, 1024)
    y = block(x)
    print(f"BlockAttentionResidual test passed. Output shape: {y.shape}")
