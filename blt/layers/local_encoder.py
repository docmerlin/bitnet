"""Local byte encoder for the ternary BLT student."""

from __future__ import annotations

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.layers.cross_attention import TernaryCrossAttention
from blt.layers.transformer_block import TransformerBlock
from blt.patching.teacher_patcher import patch_ids_from_lengths, patch_membership_mask, pool_patch_representations
from layers.h_bitlinear import HBitLinear


class LocalEncoder(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.local_dim,
                    config.n_heads_local_encoder,
                    config=config,
                    ffn_multiplier=config.ffn_multiplier_local,
                    local_window=config.local_window,
                    causal=True,
                )
                for _ in range(config.n_layers_local_encoder)
            ]
        )
        self.output_norm = nn.RMSNorm(config.local_dim)
        self.patch_init_proj = None
        if config.local_dim != config.global_dim:
            self.patch_init_proj = HBitLinear(config.local_dim, config.global_dim, config=config)
        self.patch_cross_attn = TernaryCrossAttention(
            config.global_dim,
            config.local_dim,
            hidden_dim=config.global_dim,
            num_heads=config.n_heads_cross,
            config=config,
        )

    def forward(
        self,
        byte_embeddings: torch.Tensor,
        patch_lengths: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = byte_embeddings
        byte_mask = None
        if attention_mask is not None:
            byte_mask = attention_mask[:, : byte_embeddings.size(1)].to(torch.bool)

        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)

        hidden = self.output_norm(hidden)
        if byte_mask is not None:
            hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
        patch_ids = patch_ids_from_lengths(patch_lengths, hidden.size(1))
        if byte_mask is not None:
            patch_ids = patch_ids.masked_fill(~byte_mask, -1)
        patch_states = pool_patch_representations(
            hidden,
            patch_lengths,
            patch_ids=patch_ids,
            token_mask=byte_mask,
            pooling="mean",
        )
        if self.patch_init_proj is not None:
            patch_states = self.patch_init_proj(patch_states)
        patch_mask = patch_membership_mask(patch_ids, patch_states.size(1), patches_as_queries=True)
        if byte_mask is not None:
            patch_mask = patch_mask & byte_mask.unsqueeze(1)
        patch_states = self.patch_cross_attn(patch_states, hidden, mask=patch_mask)
        return hidden, patch_states, patch_ids
