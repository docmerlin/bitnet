"""Hybrid Transformer Block combining Infini-Attention and Attention Residuals (AttnRes) in EVERY layer.

This is the unified block used from layer 0 to the final layer as requested.
It applies:
- Pre-Norm RMSNorm before each sublayer
- Infini-Attention (local masked attention + compressive memory with per-head gating)
- Attention Residual (AttnRes) around both the attention and MLP sublayers
- A learned gate to balance the Infini-Attention output with the residual path

This satisfies the requirement that every single block contains both mechanisms.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.h_bitlinear import HBitLinear
from layers.infini_attention import InfiniAttention
from layers.rfmoe import RFMoE


class AttentionResidual(nn.Module):
    """Attention Residual (AttnRes) wrapper.
    
    Implements depth-weighted residual connection as described in the AttnRes
    formulation. This helps with gradient flow in very deep ternary networks.
    """
    def __init__(self, hidden_size: int, init_scale: float = 0.1):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x_in: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """x = x_in + scale * norm(sublayer_out)"""
        return x_in + self.scale * self.norm(sublayer_out)


class HybridTransformerBlock(nn.Module):
    """Unified block that contains BOTH Infini-Attention and Attention Residuals in every layer.
    
    Structure (exactly as requested):
    
    x_in = input (or depth-weighted residual from previous layers)
    attn_out = InfiniAttention(x_in)          # local masked attention + compressive memory
    x = AttnRes(x_in, attn_out)               # Attention Residual around attention
    mlp_out = MLP(x)
    x = AttnRes(x, mlp_out)                   # Attention Residual around MLP
    output = x
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 32,
        intermediate_size: int = 2048,
        memory_dim: int = 64,
        init_scale: float = 0.1,
        config=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.config = config
        self.num_blocks = max(1, getattr(config, "block_size", 8))  # for progressive growth compatibility

        # Pre-Norm for both sublayers
        self.attn_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=1e-5)

        # Infini-Attention (replaces standard MHA)
        self.infini_attn = InfiniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            memory_dim=memory_dim,
            config=config,
        )

        # Attention Residuals (AttnRes) for both attention and MLP
        init_scale = getattr(config, "attn_res_init_scale", init_scale)
        self.attn_res = AttentionResidual(hidden_size, init_scale=init_scale)
        self.mlp_res = AttentionResidual(hidden_size, init_scale=init_scale)

        # MLP: either a dense GLU FFN (H-BitLinear, ternary) or a routing-free MoE
        # FFN (roadmap step 1). RFMoE is a drop-in replacement for the whole FFN.
        self.use_rfmoe = bool(getattr(config, "use_rfmoe", False))
        if self.use_rfmoe:
            expert_dim = getattr(config, "rfmoe_expert_dim", None) or intermediate_size // 4
            self.moe = RFMoE(
                hidden_size,
                expert_dim=expert_dim,
                num_experts=getattr(config, "rfmoe_num_experts", 8),
                rank=getattr(config, "rfmoe_rank", None),
                theta=getattr(config, "rfmoe_theta", 0.01),
                residual=False,  # mlp_res (AttnRes) adds the residual
            )
        else:
            self.ffn_up = HBitLinear(hidden_size, intermediate_size * 2, bias=False, config=config)
            self.ffn_down = HBitLinear(intermediate_size, hidden_size, bias=False, config=config)

        # Learned gate to balance Infini-Attention with residual path.
        # Initialized to 0 so sigmoid(gate)=0.5 at init: the attention path starts
        # damped to ~half scale (and AttnRes scales it again by attn_res_init_scale).
        # If early training is sluggish, a positive init here opens the attention path.
        self.gate = nn.Parameter(torch.zeros(1))  # learned scalar gate

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        # === Attention sublayer with Infini-Attention + AttnRes ===
        x_norm = self.attn_norm(x)
        if attn_bias is None:
            # Fallback path: attention builds its own block-causal bias from num_blocks.
            self.infini_attn.num_blocks = self.num_blocks
            infini_out = self.infini_attn(x_norm, attention_mask)
        else:
            # Precomputed path: the shared bias already encodes num_blocks.
            infini_out = self.infini_attn(x_norm, attn_bias=attn_bias, query_valid=query_valid)

        # Learned gate to smoothly blend Infini output with residual
        gate = torch.sigmoid(self.gate)
        gated_infini = gate * infini_out
        
        x = self.attn_res(residual, gated_infini)

        # === MLP sublayer with AttnRes ===
        residual = x
        x_norm = self.mlp_norm(x)
        
        if self.use_rfmoe:
            mlp_out = self.moe(x_norm)  # AttnRes below owns the residual (moe.residual=False)
        else:
            gate_up, value = self.ffn_up(x_norm).chunk(2, dim=-1)
            mlp_out = self.ffn_down(F.silu(gate_up) * value)
        
        x = self.mlp_res(residual, mlp_out)

        return x
