"""Equivalence check for the shared-attention-bias rewrite.

Sharing a precomputed attention bias across layers must produce the same
output as letting each attention sublayer build the bias itself.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TernaryConfig
from layers.infini_attention import InfiniAttention
from utils import (
    causal_block_attention_bias,
    combine_attention_bias,
    document_attention_keep_mask,
)


def _config() -> TernaryConfig:
    return TernaryConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=32,
        intermediate_size=256,
        block_size=4,
        infini_memory_dim=8,
        attn_res_init_scale=0.1,
    )


def test_precomputed_bias_matches_fallback() -> bool:
    torch.manual_seed(0)
    config = _config()
    attn = InfiniAttention(config.hidden_size, config.num_attention_heads, config.infini_memory_dim, config)
    attn.eval()  # eval so memory buffers don't mutate between the two calls
    x = torch.randn(2, 12, config.hidden_size)
    segment_ids = torch.tensor([[0] * 6 + [1] * 6, [0] * 4 + [1] * 8])

    # Fallback path: attention builds its own bias from a keep-mask.
    keep_mask = document_attention_keep_mask(segment_ids)
    attn.reset_memory()
    fallback = attn(x, keep_mask)

    # Precomputed path: caller folds the same mask once and shares it.
    base_bias = causal_block_attention_bias(x.size(1), attn.num_blocks, dtype=x.dtype, device=x.device)
    attn_bias, query_valid = combine_attention_bias(
        keep_mask, base_bias=base_bias, batch_size=x.size(0),
        q_len=x.size(1), k_len=x.size(1), dtype=x.dtype, device=x.device,
    )
    attn.reset_memory()
    shared = attn(x, attn_bias=attn_bias, query_valid=query_valid)

    assert torch.allclose(fallback, shared, atol=1e-6, rtol=1e-5), "Shared bias path diverged from fallback"
    print("Precomputed attention bias matches per-layer fallback")
    return True


if __name__ == "__main__":
    test_precomputed_bias_matches_fallback()
