"""Local byte decoder conditioned on latent patch states."""

from __future__ import annotations

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.layers.cross_attention import TernaryCrossAttention
from blt.layers.transformer_block import TransformerBlock
from blt.patching.teacher_patcher import patch_membership_mask
from layers.h_bitlinear import HBitLinear


class LocalDecoder(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.byte_state_proj = None
        if config.local_dim != config.decoder_dim:
            self.byte_state_proj = HBitLinear(config.local_dim, config.decoder_dim, config=config)

        self.patch_state_proj = None
        if config.global_dim != config.decoder_dim:
            self.patch_state_proj = HBitLinear(config.global_dim, config.decoder_dim, config=config)

        self.cross_attn_layers = nn.ModuleList(
            [
                TernaryCrossAttention(
                    config.decoder_dim,
                    config.decoder_dim,
                    hidden_dim=config.decoder_dim,
                    num_heads=config.n_heads_cross,
                    config=config,
                )
                for _ in range(config.n_layers_local_decoder)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.decoder_dim,
                    config.n_heads_local_decoder,
                    config=config,
                    ffn_multiplier=config.ffn_multiplier_decoder,
                    local_window=config.local_window,
                    causal=True,
                )
                for _ in range(config.n_layers_local_decoder)
            ]
        )
        self.output_norm = nn.RMSNorm(config.decoder_dim)

    def forward(
        self,
        byte_states: torch.Tensor,
        patch_states: torch.Tensor,
        patch_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.byte_state_proj(byte_states) if self.byte_state_proj is not None else byte_states
        latent = self.patch_state_proj(patch_states) if self.patch_state_proj is not None else patch_states
        cross_mask = patch_membership_mask(patch_ids, latent.size(1), patches_as_queries=False)
        byte_mask = None
        if attention_mask is not None:
            byte_mask = attention_mask[:, : hidden.size(1)].to(torch.bool)
            cross_mask = cross_mask & byte_mask.unsqueeze(-1)

        for cross_attn, block in zip(self.cross_attn_layers, self.blocks):
            hidden = cross_attn(hidden, latent, mask=cross_mask)
            if byte_mask is not None:
                hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
            hidden = block(hidden, attention_mask=attention_mask)
            if byte_mask is not None:
                hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)

        hidden = self.output_norm(hidden)
        if byte_mask is not None:
            hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
        return hidden
