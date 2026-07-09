"""Round-trip tests for the hierarchical tokenizer."""

from __future__ import annotations

from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer


def test_encode_decode_roundtrip_ascii() -> bool:
    tokenizer = HierarchicalTokenizer(max_patch_size=8, vocab_size_target=4096)
    sample = "Hello, hierarchical tokenizer round-trip!"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample, (sample, decoded)
    print("Tokenizer ASCII round-trip tests passed")
    return True


def test_encode_decode_roundtrip_unicode() -> bool:
    tokenizer = HierarchicalTokenizer(max_patch_size=8, vocab_size_target=4096)
    sample = "Byte models can handle emoji: \U0001f9ee and accents: café"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample, (sample, decoded)
    print("Tokenizer Unicode round-trip tests passed")
    return True


def test_encode_truncates_at_max_length() -> bool:
    tokenizer = HierarchicalTokenizer(max_patch_size=8, vocab_size_target=4096)
    sample = "A" * 64
    encoded = tokenizer.encode(sample, max_length=16)
    assert len(encoded) == 16
    print("Tokenizer max_length truncation tests passed")
    return True


if __name__ == "__main__":
    test_encode_decode_roundtrip_ascii()
    test_encode_decode_roundtrip_unicode()
    test_encode_truncates_at_max_length()