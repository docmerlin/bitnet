"""Validation tests for model configuration edge cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TernaryConfig
from layers.block_attnres import BlockAttentionResidual
from layers.infini_attention import InfiniAttention


def test_odd_head_dim_is_rejected() -> bool:
    try:
        TernaryConfig(
            vocab_size=32,
            hidden_size=15,
            num_hidden_layers=1,
            num_attention_heads=3,
            head_dim=5,
            intermediate_size=30,
        )
    except ValueError as exc:
        assert "head_dim must be even" in str(exc)
    else:
        raise AssertionError("TernaryConfig should reject odd head dimensions for rotary embeddings")

    for constructor in (
        lambda: BlockAttentionResidual(hidden_size=15, num_heads=3, block_size=1),
        lambda: InfiniAttention(hidden_size=15, num_heads=3, memory_dim=4),
    ):
        try:
            constructor()
        except ValueError as exc:
            assert "head dimension must be even" in str(exc)
        else:
            raise AssertionError("Attention layers should reject odd head dimensions for rotary embeddings")

    print("Configuration validation tests passed")
    return True


if __name__ == "__main__":
    test_odd_head_dim_is_rejected()
