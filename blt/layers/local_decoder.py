"""Local byte decoder conditioned on latent patch states."""

from __future__ import annotations

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.layers.cross_attention import TernaryCrossAttention, TernaryPatchGather
from blt.layers.transformer_block import TransformerBlock
from blt.patching.teacher_patcher import patch_membership_mask
from layers.h_bitlinear import HBitLinear


class LocalDecoder(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.byte_state_proj = None
        if config.local_dim != config.decoder_dim:
            self.byte_state_proj = HBitLinear(config.local_dim, config.decoder_dim, config=config)

        # At k > 1 the projection also fans each patch latent out into k slots,
        # so it is needed even when the widths already agree.
        self.cross_attn_k = config.cross_attn_k
        self.patch_state_proj = None
        if config.global_dim != config.decoder_dim or self.cross_attn_k > 1:
            self.patch_state_proj = HBitLinear(
                config.global_dim, config.decoder_dim * self.cross_attn_k, config=config
            )

        # k == 1 leaves a one-hot mask, where attention is exactly a gather and
        # the query/key projections are provably inert; k > 1 gives the byte a
        # real choice among its patch's slots, so use actual attention.
        cross_attention = TernaryCrossAttention if self.cross_attn_k > 1 else TernaryPatchGather
        self.cross_attn_layers = nn.ModuleList(
            [
                cross_attention(
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
        byte_mask = None
        if attention_mask is not None:
            byte_mask = attention_mask[:, : hidden.size(1)].to(torch.bool)

        # patch_ids is -1 on padded bytes, which is exactly where the membership
        # mask is empty, so the two notions of "no patch" coincide.
        cross_valid = patch_ids >= 0
        if byte_mask is not None:
            cross_valid = cross_valid & byte_mask

        cross_mask = None
        if self.cross_attn_k > 1:
            batch_size, num_patches = patch_states.size(0), patch_states.size(1)
            # [B, P, D*k] -> [B, P*k, D], patch-major so slot i of patch p lands
            # at p*k + i, which is the order repeat_interleave gives the mask.
            latent = latent.reshape(batch_size, num_patches * self.cross_attn_k, -1)
            cross_mask = patch_membership_mask(
                patch_ids.clamp_min(0), num_patches, patches_as_queries=False
            ).repeat_interleave(self.cross_attn_k, dim=-1)
            cross_mask = cross_mask & cross_valid.unsqueeze(-1)

        for cross_attn, block in zip(self.cross_attn_layers, self.blocks):
            hidden = (
                cross_attn(hidden, latent, mask=cross_mask)
                if self.cross_attn_k > 1
                else cross_attn(hidden, latent, patch_ids, valid=cross_valid)
            )
            if byte_mask is not None:
                hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
            hidden = block(hidden, attention_mask=attention_mask)
            if byte_mask is not None:
                hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)

        hidden = self.output_norm(hidden)
        if byte_mask is not None:
            hidden = hidden.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
        return hidden

    def _prepare_latents(self, patch_states: torch.Tensor) -> torch.Tensor:
        latent = self.patch_state_proj(patch_states) if self.patch_state_proj is not None else patch_states
        if self.cross_attn_k > 1:
            batch_size, num_patches = patch_states.size(0), patch_states.size(1)
            latent = latent.reshape(batch_size, num_patches * self.cross_attn_k, -1)
        return latent

    def prepare_cross_cache(self, patch_states: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Prepared latents plus per-layer K/V (or values at k=1) for frozen draft."""
        latent = self._prepare_latents(patch_states)
        if self.cross_attn_k > 1:
            projected = [layer.project_kv(latent) for layer in self.cross_attn_layers]
        else:
            projected = [layer.project_values(latent) for layer in self.cross_attn_layers]
        return latent, projected

    def prefill(
        self,
        byte_states: torch.Tensor,
        patch_states: torch.Tensor,
        patch_ids: torch.Tensor,
        *,
        latent: torch.Tensor | None = None,
        cross_projected: list | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Full-prefix decoder for generation; returns last-pos-ready state + self-attn caches."""
        hidden = self.byte_state_proj(byte_states) if self.byte_state_proj is not None else byte_states
        if latent is None:
            latent = self._prepare_latents(patch_states)
        cross_valid = patch_ids >= 0
        cross_mask = None
        if self.cross_attn_k > 1:
            membership = patch_membership_mask(
                patch_ids.clamp_min(0), patch_states.size(1), patches_as_queries=False
            )
            cross_mask = membership.repeat_interleave(self.cross_attn_k, dim=-1) & cross_valid.unsqueeze(-1)

        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, (cross_attn, block) in enumerate(zip(self.cross_attn_layers, self.blocks)):
            projected = None if cross_projected is None else cross_projected[index]
            hidden = (
                cross_attn(hidden, latent, mask=cross_mask, projected_kv=projected)
                if self.cross_attn_k > 1
                else cross_attn(hidden, latent, patch_ids, valid=cross_valid, values=projected)
            )
            hidden, cache = block.prefill(hidden)
            caches.append(cache)
        return self.output_norm(hidden), caches

    def extend(
        self,
        byte_states: torch.Tensor,
        patch_states: torch.Tensor,
        patch_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        offset: int,
        latent: torch.Tensor | None = None,
        cross_projected: list | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Decode only new positions with self-attn K/V cache."""
        hidden = self.byte_state_proj(byte_states) if self.byte_state_proj is not None else byte_states
        if latent is None:
            latent = self._prepare_latents(patch_states)
        cross_valid = patch_ids >= 0
        cross_mask = None
        if self.cross_attn_k > 1:
            membership = patch_membership_mask(
                patch_ids.clamp_min(0), patch_states.size(1), patches_as_queries=False
            )
            cross_mask = membership.repeat_interleave(self.cross_attn_k, dim=-1) & cross_valid.unsqueeze(-1)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, (cross_attn, block, cache) in enumerate(
            zip(self.cross_attn_layers, self.blocks, caches)
        ):
            projected = None if cross_projected is None else cross_projected[index]
            if self.cross_attn_k > 1:
                hidden = cross_attn(hidden, latent, mask=cross_mask, projected_kv=projected)
            else:
                hidden = cross_attn(hidden, latent, patch_ids, valid=cross_valid, values=projected)
            hidden, cache = block.extend(hidden, cache, offset=offset)
            new_caches.append(cache)
        return self.output_norm(hidden), new_caches
