"""Hybrid Transformer Block: Infini-Attention + sandwich residual every layer.

Sandwich residual (critical under recurrent-depth / looped unrolls)::

    x ← post_norm(x + scale * sublayer(pre_norm(x)))

Pre-norm stabilizes each sublayer input; post-norm rebounds the residual stream
so magnitudes stay controlled when the same block stack is applied many times.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig
from layers.engram import Engram
from layers.h_bitlinear import HBitLinear
from layers.infini_attention import InfiniAttention
from layers.rfmoe import RFMoE

# Near-identity residual under deep unrolls; still nonzero for grad flow.
SUBLAYER_OUT_INIT_SCALE = 0.01


class AttentionResidual(nn.Module):
    """Sandwich residual post-norm: ``post_norm(x + scale * sublayer_out)``.

    Name is historical (AttnRes scale); paired with pre-norm on the block this is
    full sandwich RMSNorm around attention / FFN.
    """

    def __init__(self, hidden_size: int, init_scale: float = 0.1, eps: float = 1e-5):
        super().__init__()
        # Kept as ``norm`` so older checkpoints still soft-load residual scales/norms.
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x_in: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return self.norm(x_in + self.scale * sublayer_out)


class HybridTransformerBlock(nn.Module):
    """One transformer block. Constructed only from ``TernaryConfig``.

    Each sublayer is sandwich-normalized::

        h = pre_norm(x)
        y = Attn/FFN(h)
        x = post_norm(x + scale * y)   # AttentionResidual
    """

    def __init__(self, config: TernaryConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.intermediate_size = config.intermediate_size
        self.num_blocks = max(1, config.block_size)
        eps = config.rms_norm_eps

        self.engram = (
            Engram(config, layer_id)
            if config.use_engram and layer_id in config.engram_layer_ids
            else None
        )

        # Pre-norms (sandwich front half).
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=eps)

        self.infini_attn = InfiniAttention(config)
        # Post-norms live inside AttentionResidual (sandwich back half).
        self.attn_res = AttentionResidual(
            config.hidden_size, init_scale=config.attn_res_init_scale, eps=eps
        )
        self.mlp_res = AttentionResidual(
            config.hidden_size, init_scale=config.attn_res_init_scale, eps=eps
        )

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
                config=config,
            )
        else:
            # SwiGLU expand (fused gate+value) -> mid (silu) -> down. Three HBitLinears /
            # three stages; mid stays at intermediate width.
            inter = config.intermediate_size
            self.ffn_up = HBitLinear(config.hidden_size, inter * 2, bias=False, config=config)
            self.ffn_mid = HBitLinear(inter, inter, bias=False, config=config)
            # Cold start: identity mid ≈ classic 2-mat path (silu pass-through on expand).
            with torch.no_grad():
                self.ffn_mid.weight.copy_(
                    torch.eye(inter, device=self.ffn_mid.weight.device, dtype=self.ffn_mid.weight.dtype)
                )
            self.ffn_down = HBitLinear(inter, config.hidden_size, bias=False, config=config)

        # sigmoid(0)=0.5 at init: attention path starts damped.
        self.gate = nn.Parameter(torch.zeros(1))

        # Deep / looped residual: shrink last proj so sandwich starts near pass-through.
        self._scale_sublayer_outputs(SUBLAYER_OUT_INIT_SCALE)

    def _scale_sublayer_outputs(self, scale: float) -> None:
        with torch.no_grad():
            self.infini_attn.o_proj.weight.mul_(scale)
            if not self.use_rfmoe:
                self.ffn_down.weight.mul_(scale)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        update_memory: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.engram is not None:
            if input_ids is None:
                raise ValueError("input_ids are required by Engram layers")
            x = x + self.engram(
                x,
                input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )

        # Attention sandwich: post(x + scale * gate * Attn(pre(x)))
        residual = x
        x_norm = self.attn_norm(x)
        self.infini_attn.num_blocks = self.num_blocks
        infini_out = self.infini_attn(
            x_norm,
            attention_mask,
            attn_bias=attn_bias,
            query_valid=query_valid,
            segment_ids=segment_ids,
            update_memory=update_memory,
        )
        x = self.attn_res(residual, torch.sigmoid(self.gate) * infini_out)

        # FFN sandwich: post(x + scale * FFN(pre(x)))
        residual = x
        x_norm = self.mlp_norm(x)
        if self.use_rfmoe:
            mlp_out = self.moe(x_norm)
        else:
            gate_up, value = self.ffn_up(x_norm).chunk(2, dim=-1)
            hidden = F.silu(gate_up) * value
            hidden = F.silu(self.ffn_mid(hidden))
            mlp_out = self.ffn_down(hidden)
        return self.mlp_res(residual, mlp_out)
