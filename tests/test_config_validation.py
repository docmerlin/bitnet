"""Validation tests for model configuration edge cases."""

from config import TernaryConfig


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

    try:
        TernaryConfig(
            vocab_size=32,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=3,
            head_dim=16,
            intermediate_size=128,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("TernaryConfig should reject hidden_size not divisible by num_heads")

    print("Configuration validation tests passed")
    return True


if __name__ == "__main__":
    test_odd_head_dim_is_rejected()
