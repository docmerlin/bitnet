"""Latent global transformer over patch representations."""

from __future__ import annotations

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.layers.transformer_block import TransformerBlock


class GlobalTransformer(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.global_dim,
                    config.n_heads_global,
                    config=config,
                    ffn_multiplier=config.ffn_multiplier_global,
                    local_window=None,
                    causal=True,
                )
                for _ in range(config.n_layers_global)
            ]
        )
        self.output_norm = nn.RMSNorm(config.global_dim)

    def forward(
        self,
        patch_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = patch_states
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.output_norm(hidden)
