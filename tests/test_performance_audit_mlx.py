"""PaTH cache tests with aligned vectors, chunk boundaries, and rollback."""

import mlx.core as mx
import pytest

from mlx_model import MLXBitNet, MLXBitNetConfig, MLXPaTHAttention


def _attention():
    config = MLXBitNetConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_prelude_layers=1, num_recurrent_layers=0, num_coda_layers=0,
        path_window_size=8, infini_memory_dim=4, use_engram=False, use_path_kernel=True,
    )
    return MLXBitNet(config).blocks[0].attn


@pytest.mark.parametrize("aligned", [False, True])
def test_path_cached_products_and_suffix_queries(aligned):
    mx.random.seed(4)
    attn = _attention()
    length = 32
    w = mx.random.normal((2, length, 4, 8))
    if aligned:
        w = mx.broadcast_to(w[:, :1], w.shape) + 0.005 * w
    w = w / mx.sqrt(mx.sum(w * w, axis=-1, keepdims=True))
    beta = mx.full((2, length, 4), 1.95)
    q, k, v = [mx.random.normal((2, 4, length, 8)) for _ in range(3)]
    forget = mx.full((2, length, 4), -0.01)
    t, wk = None, None
    for end in range(1, length + 1):
        t = attn.path_border_update_t(t, w[:, :end], beta[:, :end])
        wk = attn.path_wk_extend(wk, w[:, :end], k[:, :, :end])
    full, exact_t, exact_wk = attn.path_chunk_with_state(q, k, v, w, beta, forget, None)
    last = attn.path_chunk_last_with_t(q, k, v, w, beta, forget, t, wk=wk)
    suffix, _, _ = attn.path_chunk_with_state(q, k, v, w, beta, forget, None, query_start=25)
    mx.eval(t, wk, full, exact_t, exact_wk, last, suffix)
    for actual, expected in ((t, exact_t), (wk, exact_wk), (last, full[:, :, -1:]),
                             (suffix, full[:, :, 25:])):
        assert mx.allclose(actual, expected, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("update_memory", [False, True])
@pytest.mark.parametrize("prompt_len", [3, 16, 19])
@pytest.mark.parametrize("cache_products", [False, True])
def test_prefill_extend_and_rollback_match_serial(monkeypatch, update_memory, prompt_len, cache_products):
    attn = _attention()
    attn.cache_path_products = cache_products
    mx.random.seed(17)
    x = mx.random.normal((1, prompt_len + 11, 32))
    calls = []
    original = MLXPaTHAttention._batched_path_chunks

    def tracked(self, *args):
        calls.append(args[-1])
        return original(self, *args)

    monkeypatch.setattr(MLXPaTHAttention, "_batched_path_chunks", tracked)
    cache = attn.new_inference_cache(1)
    prefill = attn.prefill(x[:, :prompt_len], cache, update_memory)
    assert calls == ([2] if prompt_len >= 16 else [])
    serial = attn.new_inference_cache(1)
    expected = mx.concatenate([attn.incremental(x[:, i:i+1], serial, update_memory)
                               for i in range(prompt_len)], axis=1)
    mx.eval(prefill, expected)
    assert mx.allclose(prefill, expected, atol=2e-4, rtol=2e-4).item()
    saved = cache.clone()
    extension = attn.extend(x[:, prompt_len:prompt_len+10], cache, update_memory)
    expected = mx.concatenate([attn.incremental(x[:, i:i+1], serial, update_memory)
                               for i in range(prompt_len, prompt_len+10)], axis=1)
    mx.eval(extension, expected)
    assert mx.allclose(extension, expected, atol=2e-4, rtol=2e-4).item()
    a = attn.incremental(x[:, -1:], cache, update_memory)
    b = attn.incremental(x[:, -1:], serial, update_memory)
    assert mx.allclose(a, b, atol=2e-4, rtol=2e-4).item()
    # Reject the speculative branch and accept only its first two positions.
    branch = saved.clone()
    a = attn.extend(x[:, prompt_len:prompt_len+2], branch, update_memory)
    b = mx.concatenate([attn.incremental(x[:, i:i+1], saved, update_memory)
                        for i in range(prompt_len, prompt_len+2)], axis=1)
    assert mx.allclose(a, b, atol=2e-4, rtol=2e-4).item()
    for left, right in zip(branch.arrays(), saved.arrays()):
        assert mx.allclose(left, right, atol=2e-4, rtol=2e-4).item()
