"""Checks for exact greedy MTP generation."""

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_generate import greedy_generate, model_callbacks, speculative_greedy_generate
from mlx_model import MLXBitNet, MLXBitNetConfig


def test_speculative_greedy_matches_vanilla_on_acceptance_and_rejection() -> None:
    sequence = [1, 2, 3, 4, 5, 6, 7]

    def vanilla_propose(tokens):
        return [sequence[len(tokens)]]

    def speculative_propose(tokens):
        start = len(tokens)
        proposals = sequence[start : start + 3]
        if start == 1:
            proposals[1] = 9
        return proposals

    def verify(prefix, candidates):
        start = len(prefix)
        verified = sequence[start + 1 : start + len(candidates) + 1]
        next_drafts = sequence[start + len(candidates) + 1 : start + len(candidates) + 3]
        return verified, next_drafts

    expected = greedy_generate([1], 6, vanilla_propose)
    actual = speculative_greedy_generate([1], 6, speculative_propose, verify)
    assert actual == expected


def test_speculative_greedy_reuses_verified_drafts() -> None:
    proposal_calls = 0

    def propose(tokens):
        nonlocal proposal_calls
        proposal_calls += 1
        return [len(tokens), len(tokens) + 1]

    def verify(prefix, candidates):
        return [candidates[1], candidates[1] + 1], [candidates[1] + 2]

    tokens = speculative_greedy_generate([0], 4, propose, verify)
    assert tokens == [0, 1, 2, 3, 4]
    assert proposal_calls == 1


def test_mlx_speculative_generation_matches_model_greedy() -> None:
    model = MLXBitNet(
        MLXBitNetConfig(
            vocab_size=16,
            hidden_size=8,
            num_attention_heads=2,
            intermediate_size=16,
            num_prelude_layers=1,
            num_recurrent_layers=1,
            num_coda_layers=1,
            num_loops=2,
            block_size=2,
            path_window_size=4,
            infini_memory_dim=2,
            use_engram=True,
            engram_layer_ids=(0, 1, 2),
            engram_vocab_size=17,
            engram_num_heads=2,
            engram_head_dim=2,
            mtp_depth=2,
        )
    )
    model.set_dtype(mx.bfloat16)
    model.set_quantization_state(1.0, 1.0, 4)
    model.set_inference_block_width(2)
    mx.eval(model.parameters())
    propose, verify = model_callbacks(model)

    expected = greedy_generate([1, 2], 6, propose)
    actual = speculative_greedy_generate([1, 2], 6, propose, verify)

    assert actual == expected


def test_fixed_inference_chunks_preserve_prefix_hidden_states() -> None:
    model = MLXBitNet(
        MLXBitNetConfig(
            vocab_size=16,
            hidden_size=8,
            num_attention_heads=2,
            intermediate_size=16,
            num_prelude_layers=1,
            num_recurrent_layers=0,
            num_coda_layers=0,
            block_size=4,
            path_window_size=4,
            infini_memory_dim=2,
            use_engram=False,
        )
    )
    model.set_inference_block_width(2)
    prefix = mx.array([[1, 2, 3]])
    extended = mx.array([[1, 2, 3, 4, 5]])

    prefix_hidden = model.hidden_states(prefix)
    extended_hidden = model.hidden_states(extended)
    mx.eval(prefix_hidden, extended_hidden)

    assert mx.allclose(prefix_hidden, extended_hidden[:, :3], rtol=1e-5, atol=1e-5).item()


@pytest.mark.parametrize("use_full_stack", [False, True])
def test_incremental_infini_cache_matches_fixed_prefixes(use_full_stack: bool) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=1 if use_full_stack else 0,
        num_coda_layers=1 if use_full_stack else 0,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=use_full_stack,
        engram_layer_ids=(0, 1, 2),
        engram_vocab_size=17,
        engram_num_heads=2,
        engram_head_dim=2,
    )
    model = MLXBitNet(config)
    model.set_inference_block_width(2)
    cache = model.new_inference_cache(num_loops=2)
    tokens = [1, 2, 3, 4, 5]

    for position, token in enumerate(tokens, start=1):
        incremental = model.inference_step(mx.array([[token]]), cache)
        prefix = mx.array([tokens[:position]])
        expected = model.hidden_states(prefix, num_loops=2)[:, -1:]
        mx.eval(incremental, expected, cache.arrays())
        assert mx.allclose(incremental, expected, rtol=1e-4, atol=1e-5).item(), position


@pytest.mark.parametrize("prompt_length", [3, 4])
def test_batched_prefill_matches_prefix_and_continuation(prompt_length: int) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=True,
        engram_layer_ids=(0, 1, 2),
        engram_vocab_size=17,
        engram_num_heads=2,
        engram_head_dim=2,
    )
    model = MLXBitNet(config)
    model.set_inference_block_width(2)
    tokens = [1, 2, 3, 4, 5, 6]
    prompt = mx.array([tokens[:prompt_length]])
    cache = model.new_inference_cache(num_loops=2)

    actual = model.prefill(prompt, cache)
    expected = model.hidden_states(prompt, num_loops=2)
    mx.eval(actual, expected, cache.arrays())
    assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item()

    for position in range(prompt_length, len(tokens)):
        actual = model.inference_step(mx.array([[tokens[position]]]), cache)
        expected = model.hidden_states(mx.array([tokens[: position + 1]]), num_loops=2)[:, -1:]
        mx.eval(actual, expected, cache.arrays())
        assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item(), position


def test_batched_cache_extension_matches_sequential_steps() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=True,
        engram_layer_ids=(0, 1, 2),
        engram_vocab_size=17,
        engram_num_heads=2,
        engram_head_dim=2,
    )
    model = MLXBitNet(config)
    model.set_dtype(mx.bfloat16)
    model.set_quantization_state(1.0, 1.0, 4)
    model.set_inference_block_width(2)
    cache = model.new_inference_cache(num_loops=2)
    model.prefill(mx.array([[1, 2, 3]]), cache)
    sequential_cache = cache.clone()

    batched = model.inference_extend(mx.array([[4, 5, 6]]), cache)
    sequential = mx.concatenate(
        [model.inference_step(mx.array([[token]]), sequential_cache) for token in (4, 5, 6)],
        axis=1,
    )
    mx.eval(batched, sequential, cache.arrays(), sequential_cache.arrays())
    assert mx.allclose(batched, sequential, rtol=1e-3, atol=1e-4).item()

    batched_next = model.inference_step(mx.array([[7]]), cache)
    sequential_next = model.inference_step(mx.array([[7]]), sequential_cache)
    mx.eval(batched_next, sequential_next, cache.arrays(), sequential_cache.arrays())
    assert mx.allclose(batched_next, sequential_next, rtol=1e-3, atol=1e-4).item()
