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


def test_loop_structure_validation() -> bool:
    try:
        TernaryConfig(
            vocab_size=32,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            num_prelude_layers=1,
            num_recurrent_layers=0,
            num_coda_layers=1,
            num_loops=2,
        )
    except ValueError as exc:
        assert "num_recurrent_layers" in str(exc)
    else:
        raise AssertionError("num_loops > 1 with empty recurrent core should fail")

    try:
        TernaryConfig(
            vocab_size=32,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            num_prelude_layers=1,
            num_recurrent_layers=1,
            num_coda_layers=1,
            num_loops=0,
        )
    except ValueError as exc:
        assert "num_loops" in str(exc)
    else:
        raise AssertionError("num_loops < 1 should fail")

    try:
        TernaryConfig(
            vocab_size=32,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            num_hidden_layers=10,
            num_prelude_layers=1,
            num_recurrent_layers=2,
            num_coda_layers=1,
        )
    except ValueError as exc:
        assert "num_hidden_layers" in str(exc)
    else:
        raise AssertionError("mismatched num_hidden_layers vs structure should fail")

    print("Loop structure validation tests passed")
    return True


if __name__ == "__main__":
    test_odd_head_dim_is_rejected()
    test_loop_structure_validation()
