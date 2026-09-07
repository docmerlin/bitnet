"""Checks for exact greedy MTP generation."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from mlx_generate import greedy_generate, model_callbacks, speculative_greedy_generate
from mlx_model import MLXBitNet, MLXBitNetConfig


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("valid_vocab_size", [None, 7])
def test_callbacks_mask_all_heads_and_skip_unused_drafts(monkeypatch, speculative, valid_vocab_size):
    model = MLXBitNet(MLXBitNetConfig(
        vocab_size=16, hidden_size=8, num_attention_heads=2, intermediate_size=16,
        num_prelude_layers=0, num_recurrent_layers=0, num_coda_layers=0,
        use_engram=False, mtp_depth=2,
    ))
    draft_calls = []

    def draft_logits(hidden):
        assert speculative, "non-speculative generation must not evaluate draft heads"
        draft_calls.append(True)
        return mx.broadcast_to(mx.arange(16), (1, 2, 16))

    monkeypatch.setattr(model, "draft_logits", draft_logits)
    monkeypatch.setattr(
        model, "logits_from",
        lambda hidden: mx.broadcast_to(mx.arange(16), (*hidden.shape[:-1], 16)),
    )
    propose, verify = model_callbacks(
        model, speculative=speculative, valid_vocab_size=valid_vocab_size,
    )
    expected = (valid_vocab_size or 16) - 1
    proposals = propose([1, 2])
    assert proposals == [expected] * (3 if speculative else 1)
    verified, drafts = verify([1, 2], proposals)
    assert verified == [expected] * len(proposals)
    assert drafts == ([expected, expected] if speculative else [])
    if speculative:
        assert len(draft_calls) == 2
        tokens = speculative_greedy_generate([1, 2], 6, propose, verify)
    else:
        tokens = greedy_generate([1, 2], 6, propose)
    assert tokens == [1, 2] + [expected] * 6
    assert model.embedding.weight.shape[0] == 16


@pytest.mark.parametrize("valid_vocab_size", [0, -1, 17])
def test_generation_rejects_invalid_vocabulary_bound(valid_vocab_size):
    from mlx_generate import _generation_argmax

    with pytest.raises(ValueError, match="valid_vocab_size"):
        _generation_argmax(mx.zeros((1, 16)), valid_vocab_size)


@pytest.mark.parametrize("mode,speculative", [("ar", False), ("ar", True), ("dblock", False)])
def test_cli_uses_defined_vocab_and_effective_speculation(monkeypatch, mode, speculative):
    import sys
    import mlx_generate

    model = MLXBitNet(MLXBitNetConfig(
        vocab_size=16, hidden_size=8, num_attention_heads=2, intermediate_size=16,
        num_prelude_layers=1, num_recurrent_layers=0, num_coda_layers=0,
        num_loops=1, use_engram=False, mtp_depth=2 if mode == "ar" else 0,
        train_mode=mode,
    ))
    model.inference_num_loops = 1
    draft_calls = []

    class Tokenizer:
        bos_id, eos_id = 1, 5

        def __init__(self, *, max_patch_size, vocab_size_target):
            assert vocab_size_target == 16  # Rebuild IDs using the checkpoint's target.

        def __len__(self):
            return 7

        def encode(self, prompt):
            return [1, 2]

        def decode(self, tokens):
            assert tokens == [1, 2, 6, 6, 6]
            return "decoded"

    def draft_logits(hidden):
        assert speculative
        draft_calls.append(True)
        return mx.broadcast_to(mx.arange(16), (1, 2, 16))

    monkeypatch.setattr(model, "draft_logits", draft_logits)
    monkeypatch.setattr(
        model, "logits_from",
        lambda hidden: mx.broadcast_to(mx.arange(16), (*hidden.shape[:-1], 16)),
    )
    monkeypatch.setattr(mlx_generate, "load_model", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(mlx_generate, "HierarchicalTokenizer", Tokenizer)
    monkeypatch.setattr(sys, "argv", [
        "mlx_generate.py", "unused.safetensors", "--prompt", "x", "--max-new-tokens", "3",
        "--dblock-euler-steps", "1", "--speculative" if speculative else "--no-speculative",
    ])
    mlx_generate.main()
    assert bool(draft_calls) == speculative
    assert model.config.vocab_size == model.embedding.weight.shape[0] == 16


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
    model.set_inference_block_width(2)
    cache = model.new_inference_cache(num_loops=2)
    model.prefill(mx.array([[1, 2, 3]]), cache)
    cached_weights = {key: id(value) for key, value in cache.weight_cache.items()}
    assert cached_weights
    sequential_cache = cache.clone()
    assert sequential_cache.weight_cache is cache.weight_cache

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
    assert {key: id(value) for key, value in cache.weight_cache.items()} == cached_weights


def test_packed_inference_cache_matches_dense_and_reuses_weights() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=1024,
        num_prelude_layers=0,
        num_recurrent_layers=1,
        num_coda_layers=0,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=False,
    )
    dense = MLXBitNet(config)
    packed = MLXBitNet(config, recurrent_quantized_matmul=True)
    packed.load_weights(list(tree_flatten(dense.parameters())))
    for model in (dense, packed):
        model.set_dtype(mx.bfloat16)

    dense_cache = dense.new_inference_cache()
    packed_cache = packed.new_inference_cache()
    expected = dense.prefill(mx.array([[1, 2, 3]]), dense_cache)
    actual = packed.prefill(mx.array([[1, 2, 3]]), packed_cache)
    mx.eval(expected, actual, dense_cache.arrays(), packed_cache.arrays())
    cached_weights = {key: id(value) for key, value in packed_cache.weight_cache.items()}

    expected_next = dense.inference_step(mx.array([[4]]), dense_cache)
    actual_next = packed.inference_step(mx.array([[4]]), packed_cache)
    mx.eval(expected_next, actual_next, dense_cache.arrays(), packed_cache.arrays())

    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-2).item()
    assert mx.allclose(actual_next, expected_next, rtol=1e-2, atol=1e-2).item()
    assert any(key[1].startswith("packed-") for key in packed_cache.weight_cache)
    # Prefill-populated packed entries must be reused (same object ids) during decode.
    after = {key: id(value) for key, value in packed_cache.weight_cache.items()}
    for key, value_id in cached_weights.items():
        assert after.get(key) == value_id


def test_path_border_update_t_matches_full_solve() -> None:
    """Running T is O(L^2) border-updated and matches a full open-chunk solve."""
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        block_size=4,
        path_window_size=8,
        infini_memory_dim=4,
        use_engram=False,
        use_path_kernel=False,
    )
    attn = MLXBitNet(config).blocks[0].attn
    mx.random.seed(7)
    batch, heads, max_len, dim = 2, 4, 6, 4
    w_all = mx.random.normal((batch, max_len, heads, dim))
    beta_all = mx.random.uniform(0.25, 1.5, (batch, max_len, heads))
    mx.eval(w_all, beta_all)
    t_run = None
    for length in range(1, max_len + 1):
        w = w_all[:, :length]
        beta = beta_all[:, :length]
        t_run = attn.path_border_update_t(t_run if length > 1 else None, w, beta)
        t_full = attn.path_system_t_inverse(w, beta)
        mx.eval(t_run, t_full)
        assert mx.allclose(t_run, t_full, rtol=1e-4, atol=1e-5).item(), length
    # Last-query path with a fixed T matches path_chunk's last position.
    length = max_len
    w, beta = w_all, beta_all
    q = mx.random.normal((batch, heads, length, dim))
    k = mx.random.normal((batch, heads, length, dim))
    v = mx.random.normal((batch, heads, length, dim))
    log_forget = mx.random.normal((batch, length, heads)) * 0.05
    mx.eval(q, k, v, log_forget)
    t_full = attn.path_system_t_inverse(w, beta)
    last = attn.path_chunk_last_with_t(q, k, v, w, beta, log_forget, t_full, None)
    full = attn.path_chunk(q, k, v, w, beta, log_forget, None)
    mx.eval(last, full)
    assert mx.allclose(last, full[:, :, length - 1 : length], rtol=1e-4, atol=1e-5).item()


def test_path_chunk_last_matches_full_path_chunk() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        block_size=4,
        path_window_size=8,
        infini_memory_dim=4,
        use_engram=False,
        use_path_kernel=False,
    )
    model = MLXBitNet(config)
    attn = model.blocks[0].attn
    mx.random.seed(0)
    batch, heads, length, dim = 2, config.num_attention_heads, 5, attn.head_dim
    q = mx.random.normal((batch, heads, length, dim))
    k = mx.random.normal((batch, heads, length, dim))
    v = mx.random.normal((batch, heads, length, dim))
    w = mx.random.normal((batch, length, heads, dim))
    beta = mx.random.uniform(0.2, 1.5, (batch, length, heads))
    log_forget = mx.random.normal((batch, length, heads)) * 0.05
    mx.eval(q, k, v, w, beta, log_forget)
    full = attn.path_chunk(q, k, v, w, beta, log_forget, None)
    last = attn.path_chunk_last(q, k, v, w, beta, log_forget, None)
    mx.eval(full, last)
    assert mx.allclose(full[:, :, length - 1 : length], last, rtol=1e-4, atol=1e-5).item()


def test_path_decode_recompute_matches_last_mode() -> None:
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
        use_engram=False,
    )
    last_model = MLXBitNet(config)
    recompute = MLXBitNet(config)
    recompute.load_weights(list(tree_flatten(last_model.parameters())))
    last_model.set_path_decode_mode("last")
    recompute.set_path_decode_mode("recompute")
    for model in (last_model, recompute):
        model.set_inference_block_width(2)
    tokens = [1, 2, 3, 4, 5]
    cache_last = last_model.new_inference_cache(num_loops=2)
    cache_re = recompute.new_inference_cache(num_loops=2)
    for token in tokens:
        a = last_model.inference_step(mx.array([[token]]), cache_last)
        b = recompute.inference_step(mx.array([[token]]), cache_re)
        mx.eval(a, b, *cache_last.arrays(), *cache_re.arrays())
        assert mx.allclose(a, b, rtol=1e-4, atol=1e-5).item()


def test_pin_inference_weights_reuses_dense_or_packed() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=False,
    )
    model = MLXBitNet(config, recurrent_quantized_matmul=True)
    model.set_dtype(mx.bfloat16)
    model.pin_inference_weights(mx.bfloat16, prefer_packed=False)
    linear = model.blocks[0].up
    assert linear._pinned_dense is not None
    first_id = id(linear._pinned_dense)
    x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
    y1 = linear(x)
    y2 = linear(x)
    mx.eval(y1, y2)
    assert id(linear._pinned_dense) == first_id
    assert linear._packed_weight(x) is None


def test_num_loops_override_changes_cache_depth() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=2,
        num_coda_layers=1,
        num_loops=4,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=False,
    )
    model = MLXBitNet(config)
    model.inference_num_loops = 1
    cache = model.new_inference_cache(num_loops=1)
    assert len(cache.layers) == 1 + 2 * 1 + 1
    out = model.inference_step(mx.array([[1]]), cache)
    mx.eval(out, *cache.arrays())
    assert mx.all(mx.isfinite(out)).item()
    model.inference_num_loops = 3
    cache3 = model.new_inference_cache(num_loops=3)
    assert len(cache3.layers) == 1 + 2 * 3 + 1


def test_compiled_inference_step_matches_eager(monkeypatch) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=False,
    )
    eager = MLXBitNet(config)
    compiled = MLXBitNet(config)
    compiled.load_weights(list(tree_flatten(eager.parameters())))
    for model in (eager, compiled):
        model.set_dtype(mx.bfloat16)
        model.set_inference_block_width(2)
        model.inference_num_loops = 2
        model.path_decode_mode = "last"
    compiled.pin_inference_weights(mx.bfloat16, prefer_packed=False)
    compile_calls = 0
    original_compile = mx.compile

    def tracked_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(mx, "compile", tracked_compile)
    assert compiled.enable_compiled_inference()
    assert compile_calls == 0
    tokens = [1, 2, 3, 4, 5, 6]
    cache_e = eager.new_inference_cache(num_loops=2)
    cache_c = compiled.new_inference_cache(num_loops=2)
    pref = mx.array([tokens[:2]], dtype=mx.int32)
    pe = eager.prefill(pref, cache_e)
    pc = compiled.prefill(pref, cache_c)
    mx.eval(pe, pc, *cache_e.arrays(), *cache_c.arrays())
    assert mx.allclose(pe, pc, rtol=1e-2, atol=1e-2).item()
    for token in tokens[2:]:
        a = eager.inference_step(mx.array([[token]]), cache_e)
        b = compiled.inference_step(mx.array([[token]]), cache_c)
        mx.eval(a, b, *cache_e.arrays(), *cache_c.arrays())
        assert mx.allclose(a, b, rtol=1e-2, atol=1e-2).item(), token
    assert compile_calls > 0
    assert compiled._compiled_inference_step is not None
    assert compiled._compiled_by_open_len


def test_lazy_compiled_inference_failure_falls_back_to_eager(monkeypatch) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        num_loops=1,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=False,
    )
    eager = MLXBitNet(config)
    fallback = MLXBitNet(config)
    fallback.load_weights(list(tree_flatten(eager.parameters())))
    for model in (eager, fallback):
        model.set_inference_block_width(2)

    compile_calls = 0

    def fail_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        raise RuntimeError("compile unavailable")

    monkeypatch.setattr(mx, "compile", fail_compile)
    assert fallback.enable_compiled_inference()
    eager_cache = eager.new_inference_cache()
    fallback_cache = fallback.new_inference_cache()

    for token in (1, 2):
        expected = eager.inference_step(mx.array([[token]]), eager_cache)
        actual = fallback.inference_step(mx.array([[token]]), fallback_cache)
        mx.eval(expected, actual, *eager_cache.arrays(), *fallback_cache.arrays())
        assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-4).item()

    assert compile_calls == 1
    assert fallback._compiled_inference_step is None


def test_compiled_inference_does_not_dummy_prefill(monkeypatch) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        infini_memory_dim=2,
        use_engram=False,
    )
    model = MLXBitNet(config)
    model.set_dtype(mx.bfloat16)
    model.set_inference_block_width(2)
    model.inference_num_loops = 2
    model.path_decode_mode = "last"
    model.pin_inference_weights(mx.bfloat16, prefer_packed=False)

    prefill_lengths: list[int] = []
    original_prefill = MLXBitNet._prefill

    def tracked_prefill(self, tokens, cache):
        prefill_lengths.append(int(tokens.shape[1]))
        return original_prefill(self, tokens, cache)

    monkeypatch.setattr(MLXBitNet, "_prefill", tracked_prefill)
    assert model.enable_compiled_inference()
    cache = model.new_inference_cache(num_loops=2)
    # Width 2, prompt 3: one leftover token, so the first decode compiles at
    # open_len=1. The old warmup prefills a dummy cache of that length.
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    model.prefill(prompt, cache)
    mx.eval(*cache.arrays())
    for token in (4, 5):
        hidden = model.inference_step(mx.array([[token]], dtype=mx.int32), cache)
        mx.eval(hidden, *cache.arrays())
    assert prefill_lengths == [3]
    assert model._compiled_by_open_len
