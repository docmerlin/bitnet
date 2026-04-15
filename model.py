"""Main BitNetDeep model.

The current architecture uses Block Attention Residuals in all 64 layers,
matching the updated requirement to keep STTNRes-style block attention as the
main residual mechanism throughout the network.
"""
import torch
import torch.nn as nn
from typing import Optional

from config import TernaryConfig, config as default_config
from layers.h_bitlinear import HBitLinear
from layers.block_attnres import BlockAttentionResidual


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
    - 64 Block Attention Residual layers
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

        for _ in range(self.config.num_hidden_layers):
            layer = BlockAttentionResidual(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                block_size=self.config.block_size,
                config=self.config
            )
            self.layers.append(layer)

        self.lm_head = HBitLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            config=self.config
        )

        self.apply(self._init_weights)

        # Tie weights after initialization so the shared tensor is not overwritten.
        self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = self.subln(x)  # Extra stability for deep ternary net

        for layer in self.layers:
            x = layer(x, attention_mask)

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
