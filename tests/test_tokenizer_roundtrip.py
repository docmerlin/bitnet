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


def test_corpus_special_token_literals_and_actual_vocabulary() -> None:
    tokenizer = HierarchicalTokenizer(max_patch_size=8, vocab_size_target=32768)
    sample = "Corpus literals: <|endoftext|> and <|fim_prefix|>."
    encoded = tokenizer.encode(sample, add_special_tokens=True)
    assert tokenizer.decode(encoded) == sample
    assert encoded[0] == tokenizer.bos_id and encoded[-1] == tokenizer.eos_id
    assert len(tokenizer) == tokenizer.next_token_id == 260 + len(tokenizer.merges)
    assert len(tokenizer) < tokenizer.vocab_size_target
    assert all(0 <= token < len(tokenizer) for token in encoded)
    assert all(0 <= byte < 256 for token in range(len(tokenizer))
               for byte in tokenizer._expand_token(token))


def test_second_stage_patch_cache_copies_and_hits() -> None:
    tokenizer = HierarchicalTokenizer(max_patch_size=8, vocab_size_target=4096)
    sample = "Hello, hierarchical tokenizer round-trip!"
    first = tokenizer.encode_patches(sample)
    assert tokenizer._patch_cache
    first[0].append(999)
    second = tokenizer.encode_patches(sample)
    assert 999 not in second[0]
    assert first[1:] == second[1:]
    assert tokenizer.encode(sample) == tokenizer.encode(sample)
    assert tokenizer.decode(tokenizer.encode(sample)) == sample


if __name__ == "__main__":
    test_encode_decode_roundtrip_ascii()
    test_encode_decode_roundtrip_unicode()
    test_encode_truncates_at_max_length()
