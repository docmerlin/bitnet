"""Cross-attention modules for byte/patch communication.

Two directions, and only one of them is attention. Encoder-side, a patch queries
every byte it contains, so :class:`TernaryCrossAttention` does real work. Decoder
-side, a byte reads the single patch it belongs to; softmax over one permitted
key is 1 whatever the query and key say, so that direction is a gather and gets
:class:`TernaryPatchGather` instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.config import TernaryBLTConfig
from layers.h_bitlinear import HBitLinear
from utils import combine_attention_bias


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

        attn_bias, query_valid = combine_attention_bias(
            mask,
            base_bias=None,
            batch_size=batch_size,
            q_len=query_len,
            k_len=kv_len,
            dtype=q.dtype,
            device=query.device,
        )

        dropout_p = self.dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        if query_valid is not None:
            context = context.masked_fill(~query_valid, 0.0)

        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        residual = self.residual_proj(query) if self.residual_proj is not None else query
        return residual + self.out_proj(context)


class TernaryPatchGather(nn.Module):
    """Byte reads the value of the one patch it belongs to.

    This replaces a cross-attention whose mask was one-hot by construction. With
    exactly one permitted key the attention weights are a constant 1, so the
    query and key projections could not influence the output and received no
    gradient at all -- 0.5M parameters at production width that never learned
    anything. Dropping them changes no output, only the parameter count.

    Retained: ``kv_norm``, ``v_proj``, ``out_proj`` and the residual projection.
    Retired: ``query_norm``, ``q_proj``, ``k_proj`` -- see
    ``training.arch_upgrade.RETIRED_DECODER_CROSS_ATTN_TOKENS`` for the resume
    path that lets older checkpoints load past them.

    If bytes should ever attend across *several* patches -- a causal window over
    patch latents rather than only their own -- this is the module to replace
    with real attention again, and the query/key path would earn its keep.
    """

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

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim or query_dim
        self.hidden_dim = hidden_dim

        self.kv_norm = nn.RMSNorm(kv_dim)
        self.v_proj = HBitLinear(kv_dim, hidden_dim, config=config)
        self.out_proj = HBitLinear(hidden_dim, self.output_dim, config=config)
        self.residual_proj = None
        if query_dim != self.output_dim:
            self.residual_proj = HBitLinear(query_dim, self.output_dim, config=config)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        patch_ids: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``patch_ids`` is [batch, queries]; ``valid`` marks queries with a patch."""
        values = self.v_proj(self.kv_norm(key_value))
        # patch_ids is -1 on padded bytes; clamp so the gather stays in range and
        # zero those rows afterwards, matching what a fully masked attention row did.
        gathered = values.gather(
            1, patch_ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, values.size(-1))
        )
        if valid is not None:
            gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0.0)
        residual = self.residual_proj(query) if self.residual_proj is not None else query
        return residual + self.out_proj(gathered)
