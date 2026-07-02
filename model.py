"""Main BitNetDeep model.

Every Transformer layer now contains BOTH:
- Infini-Attention (local masked attention + compressive memory with per-head gating)
- Attention Residuals (AttnRes) around both the attention and MLP sublayers

This matches the exact requirement: no layer splitting — full combination in every block from layer 0 to the final layer.
"""
import torch
import torch.nn as nn
from typing import Optional

from config import TernaryConfig, config as default_config
from layers.h_bitlinear import HBitLinear
from layers.hybrid_block import HybridTransformerBlock
from utils import (
    causal_block_attention_bias,
    combine_attention_bias,
    document_attention_keep_mask,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class BitNetDeep(nn.Module):
    """Deep Ternary LLM based on BitNet b1.58 with modern techniques.

    The model uses:
    - hybrid transformer layers that combine Infini-Attention and AttnRes in every block
    - ternary H-BitLinear projections
    - RMSNorm plus an extra SubLN before the stack
    - tied embeddings / output projection weights
    """

    def __init__(self, config: Optional[TernaryConfig] = None):
        super().__init__()
        self.config = config or default_config

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

        # SubLN for extra ternary stability (as per project spec)
        self.subln = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

        self.layers = nn.ModuleList()

        # Every single layer now contains BOTH Infini-Attention and Attention Residuals (AttnRes)
        # as explicitly requested. No layer splitting.
        for _ in range(self.config.num_hidden_layers):
            layer = HybridTransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
                memory_dim=self.config.infini_memory_dim,
                init_scale=self.config.attn_res_init_scale,
                config=self.config,
            )
            self.layers.append(layer)

        # The output projection defaults to full precision (better quality than a
        # ternary head); set quantize_lm_head=True to keep it ternary.
        if self.config.quantize_lm_head:
            self.lm_head = HBitLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
                config=self.config,
            )
        else:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie weights after initialization so the shared tensor is not overwritten.
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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
        """Block-causal + optional document bias, computed once for all layers.

        All layers share a single ``num_blocks`` (progressive growth updates every
        block to the same value), so the structural bias is identical per layer.
        Built directly in the attention compute dtype (the autocast dtype under
        AMP, else ``x``'s dtype) so each attention sublayer's cast to its query
        dtype is a no-op instead of allocating a fresh copy every layer.
        """
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
    ) -> torch.Tensor:
        if reset_memory:
            for layer in self.layers:
                layer.infini_attn.reset_memory()

        if segment_ids is not None:
            attention_mask = document_attention_keep_mask(segment_ids)

        x = self.embed_tokens(input_ids)
        x = self.subln(x)  # Extra stability for deep ternary net

        # Build the block-causal + document attention bias once and share it across
        # every layer instead of recomputing it in each attention sublayer.
        attn_bias, query_valid = self._build_shared_attention_bias(x, attention_mask)

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, query_valid=query_valid)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer

    model = BitNetDeep()
    tokenizer = HierarchicalTokenizer()

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = torch.tensor([tokenizer.encode(text)[:64]])  # short sequence for test

    with torch.no_grad():
        logits = model(input_ids)

    print("BitNetDeep forward pass successful!")
    print(f"Model params: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Output logits shape: {logits.shape}")
