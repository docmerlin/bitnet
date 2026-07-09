"""Main BitNetDeep model.

Every Transformer layer contains BOTH Infini-Attention and Attention Residuals.
"""
from __future__ import annotations

import contextlib
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import TernaryConfig
from layers.hybrid_block import HybridTransformerBlock
from utils import (
    causal_block_attention_bias,
    combine_attention_bias,
    document_attention_keep_mask,
)


class BitNetDeep(nn.Module):
    """Deep ternary LLM: hybrid blocks, ternary projections, tied embeddings.

    Training knobs on the model:
    - ``gradient_checkpointing`` freezes Infini memory for recompute-safe backward
    - ``return_mtp=True`` also returns multi-token-prediction logits

    Always: full-precision lm_head, tied embed/unembed, per-head QK norm.
    """

    def __init__(self, config: Optional[TernaryConfig] = None):
        super().__init__()
        self.config = config or TernaryConfig()
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.subln = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.layers = nn.ModuleList(
            HybridTransformerBlock(self.config)
            for _ in range(self.config.num_hidden_layers)
        )

        # Full-precision output projection; tied to embeddings.
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.mtp_depth = int(self.config.mtp_depth)
        if self.mtp_depth > 0:
            self.mtp_transforms = nn.ModuleList(
                nn.Sequential(
                    nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
                )
                for _ in range(self.mtp_depth)
            )

        self.apply(self._init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def mtp_logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Per-depth logits from the final hidden. Depth i predicts offset i+2."""
        if self.mtp_depth <= 0:
            return []
        return [self.lm_head(transform(hidden)) for transform in self.mtp_transforms]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _build_shared_attention_bias(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        device_type = x.device.type
        compute_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else x.dtype
        )
        num_blocks = self.layers[0].num_blocks if len(self.layers) else 1
        base_bias = causal_block_attention_bias(
            seq_len, num_blocks, dtype=compute_dtype, device=x.device
        )
        return combine_attention_bias(
            attention_mask,
            base_bias=base_bias,
            batch_size=batch_size,
            q_len=seq_len,
            k_len=seq_len,
            dtype=compute_dtype,
            device=x.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
        segment_ids: Optional[torch.Tensor] = None,
        return_mtp: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        if reset_memory:
            for layer in self.layers:
                layer.infini_attn.reset_memory()

        if segment_ids is not None:
            attention_mask = document_attention_keep_mask(segment_ids)

        x = self.embed_tokens(input_ids)
        x = self.subln(x)
        attn_bias, query_valid = self._build_shared_attention_bias(x, attention_mask)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # All layers are HybridTransformerBlock; freeze Infini memory for recompute.
                layer_memory_state = layer.infini_attn.get_memory_state()
                x = checkpoint(
                    lambda hidden_states, layer=layer, attn_bias=attn_bias, query_valid=query_valid: layer(
                        hidden_states, attn_bias=attn_bias, query_valid=query_valid
                    ),
                    x,
                    use_reentrant=False,
                    context_fn=lambda layer=layer, layer_memory_state=layer_memory_state: (
                        contextlib.nullcontext(),
                        layer.infini_attn.use_memory_state(
                            layer_memory_state, update_memory_buffers=False
                        ),
                    ),
                )
            else:
                x = layer(x, attn_bias=attn_bias, query_valid=query_valid)

        x = self.norm(x)
        logits = self.lm_head(x)
        if return_mtp:
            return logits, self.mtp_logits(x)
        return logits
