"""Hybrid Transformer Block: Infini-Attention + residual every layer.

Default residual path is **Kimi Block AttnRes** (arXiv:2603.15031): depth softmax
over completed residual blocks + partial sum. Optional legacy sandwich::

    x ← post_norm(x + scale * sublayer(pre_norm(x)))

Engram, PaTH/Infini, and dense/RFMoE FFN are unchanged.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig
from layers.attn_res import AttentionResidual, AttnResStream, SandwichResidual
from layers.engram import Engram
from layers.h_bitlinear import HBitLinear
from layers.infini_attention import InfiniAttention
from layers.rfmoe import RFMoE

# Near-identity residual under deep unrolls; still nonzero for grad flow.
SUBLAYER_OUT_INIT_SCALE = 0.01


class HybridTransformerBlock(nn.Module):
    """One transformer block. Constructed only from ``TernaryConfig``.

    Kimi mode expects an ``AttnResStream`` and returns the updated stream.
    Sandwich mode takes/returns a plain hidden tensor (legacy).
    """

    def __init__(self, config: TernaryConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.intermediate_size = config.intermediate_size
        self.num_blocks = max(1, config.block_size)
        self.attn_res_mode = str(getattr(config, "attn_res_mode", "kimi"))
        eps = config.rms_norm_eps

        self.engram = (
            Engram(config, layer_id)
            if config.use_engram and layer_id is not None and layer_id in config.engram_layer_ids
            else None
        )

        # Pre-norms (paper: norm before attn / mlp on mixed residual).
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=eps)

        self.infini_attn = InfiniAttention(config)

        if self.attn_res_mode == "sandwich":
            self.attn_res = SandwichResidual(
                config.hidden_size, init_scale=config.attn_res_init_scale, eps=eps
            )
            self.mlp_res = SandwichResidual(
                config.hidden_size, init_scale=config.attn_res_init_scale, eps=eps
            )
            self.attn_res_mix = None
            self.mlp_res_mix = None
        elif self.attn_res_mode == "kimi":
            # Per-layer pseudo-queries (paper assigns w_l per layer / branch).
            from layers.attn_res import DepthAttnMix

            self.attn_res_mix = DepthAttnMix(config.hidden_size, eps=eps)
            self.mlp_res_mix = DepthAttnMix(config.hidden_size, eps=eps)
            # Keep attributes for old metric code / soft-load probes.
            self.attn_res = None
            self.mlp_res = None
        else:
            raise ValueError(
                f"attn_res_mode must be 'kimi' or 'sandwich', got {self.attn_res_mode!r}"
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
            # SwiGLU expand → square mid → down. Mid cold-starts as identity.
            inter = config.intermediate_size
            self.ffn_up = HBitLinear(config.hidden_size, inter * 2, bias=False, config=config)
            self.ffn_mid = HBitLinear(inter, inter, bias=False, config=config)
            with torch.no_grad():
                self.ffn_mid.weight.copy_(
                    torch.eye(inter, device=self.ffn_mid.weight.device, dtype=self.ffn_mid.weight.dtype)
                )
            self.ffn_down = HBitLinear(inter, config.hidden_size, bias=False, config=config)

        # sigmoid(0)=0.5 at init: attention path starts damped.
        self.gate = nn.Parameter(torch.zeros(1))

        # Deep / looped residual: shrink last proj so stack starts near pass-through.
        self._scale_sublayer_outputs(SUBLAYER_OUT_INIT_SCALE)

    def _scale_sublayer_outputs(self, scale: float) -> None:
        with torch.no_grad():
            self.infini_attn.o_proj.weight.mul_(scale)
            if not self.use_rfmoe:
                self.ffn_down.weight.mul_(scale)

    def _dense_mlp(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, value = self.ffn_up(x).chunk(2, dim=-1)
        hidden = F.silu(gate_up) * value
        hidden = F.silu(self.ffn_mid(hidden))
        return self.ffn_down(hidden)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rfmoe:
            return self.moe(x)
        return self._dense_mlp(x)

    def forward_sandwich(
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
        """Legacy sandwich residual path."""
        if self.engram is not None:
            if input_ids is None:
                raise ValueError("input_ids are required by Engram layers")
            x = x + self.engram(
                x,
                input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )

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

        residual = x
        x_norm = self.mlp_norm(x)
        mlp_out = self._mlp(x_norm)
        return self.mlp_res(residual, mlp_out)

    def forward_kimi(
        self,
        stream: AttnResStream,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        update_memory: Optional[bool] = None,
    ) -> AttnResStream:
        """Kimi Block AttnRes: mix depth → pre-norm sublayer → accumulate delta."""
        # Use this layer's mix modules (per-layer w_l).
        stream.attn_mix = self.attn_res_mix
        stream.mlp_mix = self.mlp_res_mix

        h = stream.mix_attn()
        if self.engram is not None:
            if input_ids is None:
                raise ValueError("input_ids are required by Engram layers")
            h = h + self.engram(
                h,
                input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )

        self.infini_attn.num_blocks = self.num_blocks
        attn_out = self.infini_attn(
            self.attn_norm(h),
            attention_mask,
            attn_bias=attn_bias,
            query_valid=query_valid,
            segment_ids=segment_ids,
            update_memory=update_memory,
        )
        stream.add_sublayer(torch.sigmoid(self.gate) * attn_out)

        h = stream.mix_mlp()
        mlp_out = self._mlp(self.mlp_norm(h))
        stream.add_sublayer(mlp_out)
        stream.close_layer()
        return stream

    def forward(
        self,
        x: Union[torch.Tensor, AttnResStream],
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        update_memory: Optional[bool] = None,
    ) -> Union[torch.Tensor, AttnResStream]:
        if self.attn_res_mode == "kimi":
            if not isinstance(x, AttnResStream):
                raise TypeError("kimi AttnRes mode requires AttnResStream input")
            return self.forward_kimi(
                x,
                attention_mask,
                attn_bias=attn_bias,
                query_valid=query_valid,
                segment_ids=segment_ids,
                input_ids=input_ids,
                update_memory=update_memory,
            )
        if isinstance(x, AttnResStream):
            raise TypeError("sandwich mode expects a hidden tensor, not AttnResStream")
        return self.forward_sandwich(
            x,
            attention_mask,
            attn_bias=attn_bias,
            query_valid=query_valid,
            segment_ids=segment_ids,
            input_ids=input_ids,
            update_memory=update_memory,
        )
