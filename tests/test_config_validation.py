"""Validation tests for model configuration edge cases."""

from config import TernaryConfig
from blt.config import TernaryBLTConfig
from train import build_arg_parser


def test_head_dimensions_are_validated() -> bool:
    config = TernaryConfig(
        vocab_size=32,
        hidden_size=15,
        num_hidden_layers=1,
        num_attention_heads=3,
        head_dim=5,
        intermediate_size=30,
    )
    assert config.head_dim == 5  # PaTH does not require paired rotary dimensions.

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


def test_blt_decoder_cross_attention_dimension_is_validated() -> None:
    try:
        TernaryBLTConfig(decoder_dim=40, n_heads_cross=8)
    except ValueError as exc:
        assert "decoder cross-attention head_dim" in str(exc)
    else:
        raise AssertionError("BLT config should reject odd decoder cross-attention head dimensions")


def test_train_sequence_length_must_match_path_window() -> None:
    from train import _validate_sequence_path_window

    parser = build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.sequence_length == 1024
    assert defaults.path_window_size == 1024
    assert defaults.mud_block_size == 32

    bad = parser.parse_args(["--sequence-length", "512", "--path-window-size", "1024"])
    try:
        _validate_sequence_path_window(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("train CLI should reject seq not divisible by path_window")

    ok = parser.parse_args(["--sequence-length", "2048", "--path-window-size", "1024"])
    _validate_sequence_path_window(ok)
    assert ok.sequence_length == 2048


if __name__ == "__main__":
    test_head_dimensions_are_validated()
    test_loop_structure_validation()
