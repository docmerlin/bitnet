"""Full ternary BLT student model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.layers.global_transformer import GlobalTransformer
from blt.layers.local_decoder import LocalDecoder
from blt.layers.local_encoder import LocalEncoder
from blt.patching.teacher_patcher import UniformPatcher, normalize_patch_lengths_to_targets, patch_presence_mask
from layers.h_bitlinear import HBitLinear
from utils import validate_suffix_padded_mask


@dataclass(slots=True)
class TernaryBLTOutput:
    logits: torch.Tensor
    patch_lengths: torch.Tensor
    patch_ids: torch.Tensor
    encoder_hidden: torch.Tensor
    encoder_patches: torch.Tensor
    global_hidden: torch.Tensor
    decoder_hidden: torch.Tensor


class TernaryBLTModel(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.config = config
        padding_idx = config.pad_id if config.pad_id >= 0 else None
        self.byte_embeddings = nn.Embedding(config.vocab_size, config.local_dim, padding_idx=padding_idx)
        self.local_encoder = LocalEncoder(config)
        self.global_transformer = GlobalTransformer(config)
        self.local_decoder = LocalDecoder(config)
        self.output_head = HBitLinear(config.decoder_dim, config.vocab_size, config=config)
        self.fallback_patcher = UniformPatcher(config.patch_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        patch_lengths: torch.Tensor | None = None,
    ) -> TernaryBLTOutput:
        if attention_mask is None:
            if self.config.pad_id >= 0:
                attention_mask = input_ids.ne(self.config.pad_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)

        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        validate_suffix_padded_mask(attention_mask)

        valid_lengths = attention_mask.sum(dim=1)

        if patch_lengths is None:
            patch_lengths = self.fallback_patcher.patch(input_ids)
        patch_lengths = normalize_patch_lengths_to_targets(patch_lengths, valid_lengths)

        byte_embeddings = self.byte_embeddings(input_ids)
        encoder_hidden, encoder_patches, patch_ids = self.local_encoder(
            byte_embeddings,
            patch_lengths,
            attention_mask=attention_mask,
        )
        global_mask = patch_presence_mask(patch_lengths).long()
        global_hidden = self.global_transformer(encoder_patches, attention_mask=global_mask)
        decoder_hidden = self.local_decoder(
            encoder_hidden,
            global_hidden,
            patch_ids,
            attention_mask=attention_mask,
        )
        logits = self.output_head(decoder_hidden)
        return TernaryBLTOutput(
            logits=logits,
            patch_lengths=patch_lengths,
            patch_ids=patch_ids,
            encoder_hidden=encoder_hidden,
            encoder_patches=encoder_patches,
            global_hidden=global_hidden,
            decoder_hidden=decoder_hidden,
        )
