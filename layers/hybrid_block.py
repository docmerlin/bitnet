"""Hybrid Transformer Block: Infini-Attention + AttnRes in every layer."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig
from layers.h_bitlinear import HBitLinear
from layers.infini_attention import InfiniAttention
from layers.rfmoe import RFMoE


class AttentionResidual(nn.Module):
    """Depth-weighted residual: x_in + scale * norm(sublayer_out)."""

    def __init__(self, hidden_size: int, init_scale: float = 0.1):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x_in: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return x_in + self.scale * self.norm(sublayer_out)


class HybridTransformerBlock(nn.Module):
    """One transformer block. Constructed only from ``TernaryConfig``."""

    def __init__(self, config: TernaryConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.intermediate_size = config.intermediate_size
        self.num_blocks = max(1, config.block_size)

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)

        self.infini_attn = InfiniAttention(config)
        self.attn_res = AttentionResidual(config.hidden_size, init_scale=config.attn_res_init_scale)
        self.mlp_res = AttentionResidual(config.hidden_size, init_scale=config.attn_res_init_scale)

        self.use_rfmoe = config.use_rfmoe
        if self.use_rfmoe:
            expert_dim = config.rfmoe_expert_dim or config.intermediate_size // 4
            self.moe = RFMoE(
                config.hidden_size,
                expert_dim=expert_dim,
                num_experts=config.rfmoe_num_experts,
                rank=config.rfmoe_rank,
                theta=config.rfmoe_theta,
                residual=False,
            )
        else:
            self.ffn_up = HBitLinear(
                config.hidden_size, config.intermediate_size * 2, bias=False, config=config
            )
            self.ffn_down = HBitLinear(
                config.intermediate_size, config.hidden_size, bias=False, config=config
            )

        # sigmoid(0)=0.5 at init: attention path starts damped.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.attn_norm(x)
        self.infini_attn.num_blocks = self.num_blocks
        infini_out = self.infini_attn(
            x_norm, attention_mask, attn_bias=attn_bias, query_valid=query_valid
        )
        x = self.attn_res(residual, torch.sigmoid(self.gate) * infini_out)

        residual = x
        x_norm = self.mlp_norm(x)
        if self.use_rfmoe:
            mlp_out = self.moe(x_norm)
        else:
            gate_up, value = self.ffn_up(x_norm).chunk(2, dim=-1)
            mlp_out = self.ffn_down(F.silu(gate_up) * value)
        return self.mlp_res(residual, mlp_out)
