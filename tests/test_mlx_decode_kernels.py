import mlx.core as mx
import pytest

from mlx_model import MLXBitNetConfig, MLXDepthAttnMix


@pytest.mark.parametrize("dim", [1024, 2048])  # 2048: two elements per thread
@pytest.mark.parametrize("n_completed", [1, 3, 8])
@pytest.mark.parametrize("dtype,tol", [(mx.float32, 1e-5), (mx.bfloat16, 5e-3)])
def test_depth_attn_mix_m1_matches_reference(dim: int, n_completed: int, dtype, tol: float) -> None:
    mx.random.seed(0)
    mix = MLXDepthAttnMix(dim, config=MLXBitNetConfig())
    # Non-trivial norm/proj weights so the softmax is not uniform.
    mix.norm.weight = mx.random.uniform(0.5, 1.5, (dim,))
    mix.proj.weight = mx.random.normal((1, dim)) * 0.2
    mix.set_dtype(dtype)
    completed = [(mx.random.normal((1, 1, dim)) * (j + 1)).astype(dtype) for j in range(n_completed)]
    partial = mx.random.normal((1, 1, dim)).astype(dtype)
    stacked = mx.concatenate([c.reshape(1, -1) for c in completed], axis=0)

    actual = mix(completed, partial, stacked)
    assert actual.shape == partial.shape and actual.dtype == dtype
    # fp32 ground truth: the bf16 op chain rounds logits and softmax to bf16 and drifts
    # up to ~3.5e-2 at N=8; the kernel computes in fp32 and rounds only the output.
    mix.set_dtype(mx.float32)
    expected = mix([c.astype(mx.float32) for c in completed], partial.astype(mx.float32))
    mx.eval(expected, actual)
    rel = (mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)).max() / mx.abs(expected).max()).item()
    assert rel < tol, rel


@pytest.mark.parametrize("feature_map", ["elu", "favor"])
@pytest.mark.parametrize("hidden,heads", [(128, 2), (128, 4)])
# bf16: fused and op-by-op both sit ~12% from an fp32 run (bf16 projections), so
# they may disagree by a few percent; the kernel itself is no less accurate.
@pytest.mark.parametrize("dtype,tol", [(mx.float32, 1e-4), (mx.bfloat16, 5e-2)])
def test_path_decode_step_m1_matches_op_by_op(feature_map: str, hidden: int, heads: int, dtype, tol: float) -> None:
    from mlx.utils import tree_flatten

    from mlx_model import MLXPaTHAttention

    config = MLXBitNetConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        path_window_size=4,
        infini_feature_map=feature_map,
        use_engram=False,
    )
    mx.random.seed(0)
    fused = MLXPaTHAttention(config)
    fused.memory_gate = mx.random.normal((heads,))  # non-trivial gate
    reference = MLXPaTHAttention(config)
    reference.load_weights(list(tree_flatten(fused.parameters())))
    reference.fused_decode = False
    for module in (fused, reference):
        module.set_dtype(dtype)
    fused_cache, reference_cache = fused.new_inference_cache(1), reference.new_inference_cache(1)

    # 10 tokens with window 4: crosses chunk boundaries, writes then reads memory.
    for step in range(10):
        x = mx.random.normal((1, 1, hidden)).astype(dtype)
        actual = fused.incremental(x, fused_cache, True)
        expected = reference.incremental(x, reference_cache, True)
        mx.eval(actual, expected, *fused_cache.arrays(), *reference_cache.arrays())
        rel = (mx.abs(actual - expected).max() / mx.abs(expected).max()).item()
        assert rel < tol, (step, rel)
        # The kernel builds the token's cache entries from the raw projections in
        # fp32; the op chain rounds intermediates (e.g. path_beta's output, which
        # seeds T) to the activation dtype.
        cache_tol = 1e-5 if dtype == mx.float32 else 2e-2
        if reference_cache.t_inverse is not None:
            assert mx.allclose(
                fused_cache.t_inverse, reference_cache.t_inverse, rtol=max(cache_tol, 1e-4), atol=cache_tol
            ).item(), step
        assert fused_cache.open_len == reference_cache.open_len, step
        for name in ("q", "k", "v", "w", "beta", "log_forget", "path_projected"):
            got, want = getattr(fused_cache, name), getattr(reference_cache, name)
            assert (got is None) == (want is None), (step, name)
            if want is not None:
                assert got.shape == want.shape and got.dtype == want.dtype, (step, name)
                assert mx.allclose(got, want, rtol=cache_tol, atol=cache_tol).item(), (step, name)


@pytest.mark.parametrize("compiled", [False, True])
def test_fused_decode_matches_dense_decode(compiled: bool, monkeypatch) -> None:
    """Packed pins route decode through every fused kernel (merged qkv/path_down,
    norm+up+swiglu, mid+silu, PaTH step, depth mix); dense pins through none of the
    ternary ones. Same model, same tokens: outputs must agree."""
    from mlx.utils import tree_flatten

    import mlx_model
    from mlx_model import MLXBitNet

    # Outputs agree whether or not a fusion fires, so also record which ones ran.
    variants = set()
    original = mlx_model.ternary_fused_linear_m1

    def recording(*args, **kwargs):
        variants.add((kwargs.get("norm_weight") is not None, kwargs.get("epilogue")))
        return original(*args, **kwargs)

    monkeypatch.setattr(mlx_model, "ternary_fused_linear_m1", recording)
    config = MLXBitNetConfig(
        vocab_size=64,
        hidden_size=64,
        num_attention_heads=2,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=2,
        num_coda_layers=1,
        num_loops=2,
        path_window_size=4,
        use_engram=False,
    )
    mx.random.seed(0)
    fused = MLXBitNet(config)
    dense = MLXBitNet(config)
    dense.load_weights(list(tree_flatten(fused.parameters())))
    for model in (fused, dense):
        model.recurrent_quantized_matmul = True
        model.eval()
    fused.pin_inference_weights(mx.float32, prefer_packed=True)
    dense.pin_inference_weights(mx.float32, prefer_packed=False)
    if compiled:
        assert fused.enable_compiled_inference()
    assert fused.blocks[0].attn._merged_projection is not None

    fused_cache, dense_cache = fused.new_inference_cache(), dense.new_inference_cache()
    for token in [3, 17, 5, 60, 1, 8, 33, 2, 9, 41]:  # crosses chunk boundaries (width 4)
        tokens = mx.array([[token]], dtype=mx.int32)
        actual = fused.inference_step(tokens, fused_cache)
        expected = dense.inference_step(tokens, dense_cache)
        mx.eval(actual, expected)
        rel = (mx.abs(actual - expected).max() / mx.abs(expected).max()).item()
        # Dense keeps the fp8 STE (x + (q - x)), fused uses q directly: fp32 rounding only.
        assert rel < 1e-3, (token, rel)
    assert {(True, "swiglu"), (False, "silu"), (False, None)} <= variants, variants
    if compiled:
        assert fused._compiled_inference_step is not None, "compiled step fell back to eager"


@pytest.mark.parametrize("hidden,heads", [(128, 2), (128, 4)])
def test_path_decode_step_m1_long_chunk(hidden: int, heads: int) -> None:
    """Window 80 > the 64-row suffix tile (head_dim 64) and > one simdgroup of rows."""
    from mlx.utils import tree_flatten

    from mlx_model import MLXPaTHAttention

    config = MLXBitNetConfig(hidden_size=hidden, num_attention_heads=heads, path_window_size=80, use_engram=False)
    mx.random.seed(1)
    fused = MLXPaTHAttention(config)
    fused.memory_gate = mx.random.normal((heads,))
    reference = MLXPaTHAttention(config)
    reference.load_weights(list(tree_flatten(fused.parameters())))
    reference.fused_decode = False
    fused_cache, reference_cache = fused.new_inference_cache(1), reference.new_inference_cache(1)
    for step in range(85):  # fills one chunk past both tile edges, then starts the next
        x = mx.random.normal((1, 1, hidden))
        actual = fused.incremental(x, fused_cache, True)
        expected = reference.incremental(x, reference_cache, True)
        mx.eval(actual, expected)
        # The attention state agrees to ~1e-7; the output goes through `out`, whose
        # fp8 input rounding turns that into an occasional one-step flip (~3e-4).
        rel = (mx.abs(actual - expected).max() / mx.abs(expected).max()).item()
        assert rel < 1e-3, (step, rel)
        if reference_cache.t_inverse is not None:
            t_rel = mx.abs(fused_cache.t_inverse - reference_cache.t_inverse).max() / mx.abs(reference_cache.t_inverse).max()
            assert t_rel.item() < 1e-5, (step, t_rel.item())
