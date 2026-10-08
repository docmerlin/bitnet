"""DiffusionBlocks Huginn path: AR identity, finite denoise CE, slice grads."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_train import _masked_ce, create_dblock_gradient_step, dblock_block_gradients


def _ar_config(**overrides) -> MLXBitNetConfig:
    base = dict(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_prelude_layers=1,
        num_recurrent_layers=2,
        num_coda_layers=1,
        num_loops=2,
        block_size=2,
        path_window_size=4,
        use_engram=False,
        use_hadamard=False,
        mtp_depth=0,
        train_mode="ar",
    )
    base.update(overrides)
    return MLXBitNetConfig(**base)


def _dblock_config(**overrides) -> MLXBitNetConfig:
    settings = dict(_ar_config().__dict__)
    settings.update(train_mode="dblock", dblock_blocks=1, num_loops=2)
    settings.update(overrides)
    return MLXBitNetConfig(**settings)


def test_dblock_disables_engram_and_rejects_mtp() -> None:
    config = _dblock_config(use_engram=True, engram_layer_ids=(0,))
    assert config.use_engram is False
    assert config.engram_layer_ids == ()
    with pytest.raises(ValueError, match="MTP is AR-only"):
        _dblock_config(mtp_depth=2)


def test_ar_model_has_no_dblock_modules() -> None:
    model = MLXBitNet(_ar_config())
    names = dict(tree_flatten(model.parameters()))
    assert all("ada_" not in name and "sigma_embed" not in name for name in names)


def test_dblock_model_allocates_adarms() -> None:
    model = MLXBitNet(_dblock_config())
    names = dict(tree_flatten(model.parameters()))
    assert any(name.startswith("sigma_embed.") for name in names)
    assert any(".ada_attn." in name or name.endswith("ada_attn.proj.weight") for name in names)


def test_ar_hidden_states_unchanged_by_dblock_kwargs() -> None:
    mx.random.seed(0)
    model = MLXBitNet(_ar_config())
    tokens = mx.random.randint(0, 32, (1, 4))
    mx.eval(model.parameters())
    a = model.hidden_states(tokens)
    b = model.hidden_states(tokens, cond=None, apply_input_norm=True, flatten_loops=False)
    mx.eval(a, b)
    assert mx.allclose(a, b, rtol=0, atol=0).item()


def test_dblock_logits_are_finite() -> None:
    mx.random.seed(1)
    model = MLXBitNet(_dblock_config())
    tokens = mx.random.randint(0, 32, (2, 4))
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    sigma = mx.array(1.0)
    eps = mx.random.normal((2, 4, 16))
    mx.eval(model.parameters())
    logits = model.dblock_logits(tokens, tokens, segments, sigma, eps)
    mx.eval(logits)
    assert logits.shape == (2, 4, 32)
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_denoised_is_expected_clean_embedding() -> None:
    model = MLXBitNet(_dblock_config())
    logits = mx.full((1, 1, 32), -1e9).at[..., 5].add(1e9)
    expected = model.dblock_clean_embeddings(mx.array([[5]])).astype(mx.float32)
    assert mx.allclose(model.dblock_denoised(logits), expected, atol=1e-5).item()


def test_dblock_logits_never_see_clean_target_or_future() -> None:
    """Position i: clean context ≤ i, noisy target i only. Paper App. E.4 causality."""
    mx.random.seed(7)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4]])
    targets = mx.array([[2, 3, 4, 5]])
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    sigma = mx.array([0.5])
    eps = mx.random.normal((1, 4, 16))
    base = model.dblock_logits(tokens, targets, segments, sigma, eps)
    # Change everything after position 1 (future context and future noisy targets).
    later = model.dblock_logits(
        mx.array([[1, 2, 9, 9]]), mx.array([[2, 3, 9, 9]]), segments, sigma, eps
    )
    mx.eval(base, later)
    assert mx.allclose(base[:, :2], later[:, :2], atol=1e-5).item()


def test_dblock_loss_is_weighted_next_token_ce() -> None:
    mx.random.seed(7)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    step = create_dblock_gradient_step(model, compile_step=False, block_id=0)
    tokens = mx.random.randint(0, 32, (2, 4))
    targets = (tokens + 3) % 32
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    sigma = mx.array([0.8, 0.3])
    eps = mx.random.normal((2, 4, 16))
    weight = mx.array([2.0, 0.5])
    rest = (segments, segments, mx.array(0.0), mx.array(1.0), mx.array(0.1), sigma, eps)
    loss, _ = step(tokens, targets, *rest, weight)
    logits = model.dblock_logits(tokens, targets, segments, sigma, eps)
    valid = mx.ones(tokens.shape, dtype=mx.bool_)
    per_row = [_masked_ce(logits[i : i + 1], targets[i : i + 1], valid[i : i + 1]) for i in range(2)]
    expected = (2.0 * per_row[0] + 0.5 * per_row[1]) / 2
    mx.eval(loss, expected)
    assert mx.allclose(loss, expected, rtol=1e-4, atol=1e-4).item()


def test_dblock_train_step_is_finite() -> None:
    mx.random.seed(2)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    step = create_dblock_gradient_step(model, compile_step=False, block_id=0)
    tokens = mx.random.randint(0, 32, (1, 4))
    targets = mx.random.randint(0, 32, (1, 4))
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    sigma = mx.array([0.8])
    eps = mx.random.normal((1, 4, 16))
    loss, grads = step(
        tokens,
        targets,
        segments,
        segments,
        mx.array(0.0),
        mx.array(1.0),
        mx.array(0.1),
        sigma,
        eps,
        mx.array([1.0]),
    )
    mx.eval(loss, grads)
    assert mx.isfinite(loss).item()
    assert float(loss.item()) > 0


def test_stage2_slice_grads_skip_idle_blocks() -> None:
    mx.random.seed(3)
    config = _dblock_config(dblock_blocks=4, num_prelude_layers=1, num_recurrent_layers=2, num_coda_layers=1)
    assert config.dblock_layer_ranges() == [(0, 1), (1, 2), (2, 3), (3, 4)]
    model = MLXBitNet(config)
    mx.eval(model.parameters())
    active = 1
    step = create_dblock_gradient_step(model, compile_step=False, block_id=active)
    tokens = mx.random.randint(0, 32, (1, 4))
    targets = mx.random.randint(0, 32, (1, 4))
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    loss, grads = step(
        tokens,
        targets,
        segments,
        segments,
        mx.array(0.0),
        mx.array(1.0),
        mx.array(0.1),
        mx.array([1.2]),
        mx.random.normal((1, 4, 16)),
        mx.array([1.0]),
    )
    mx.eval(loss, grads)
    flat = dict(tree_flatten(grads))
    def _block_grad(index: int) -> float:
        total = 0.0
        prefix = f"blocks.{index}."
        for name, value in flat.items():
            if name.startswith(prefix) and value is not None:
                total += float(mx.sum(mx.abs(value)).item())
        return total

    assert _block_grad(active) > 0.0
    for idle in (0, 2, 3):
        assert _block_grad(idle) == 0.0
    # Shared readout still trains.
    assert float(mx.sum(mx.abs(flat["embedding.weight"])).item()) > 0.0
    # Idle blocks are dropped, not zeroed: optimizer momentum/decay must skip them.
    pruned = dblock_block_gradients(grads, config.dblock_layer_ranges()[active])
    assert [bool(entry) for entry in pruned["blocks"]] == [False, True, False, False]


def test_pruned_gradients_leave_idle_blocks_untouched() -> None:
    from mlx_optim import CMUD

    mx.random.seed(4)
    config = _dblock_config(dblock_blocks=2)
    model = MLXBitNet(config)
    mx.eval(model.parameters())
    optimizer = CMUD(mud_learning_rate=0.02, fallback_learning_rate=1e-3, weight_decay=0.1)
    ones = lambda tree: {k: ones(v) if isinstance(v, (dict, list)) else mx.ones_like(v) for k, v in tree.items()} if isinstance(tree, dict) else [ones(v) for v in tree]
    grads = ones(model.trainable_parameters())
    optimizer.update(model, grads)  # warm momentum everywhere
    mx.eval(model.parameters(), optimizer.state)
    idle_before = dict(tree_flatten(model.blocks[0].parameters()))
    for _ in range(2):
        optimizer.update(model, dblock_block_gradients(grads, config.dblock_layer_ranges()[1]))
        mx.eval(model.parameters(), optimizer.state)
    idle_after = dict(tree_flatten(model.blocks[0].parameters()))
    assert all(mx.array_equal(idle_before[k], idle_after[k]).item() for k in idle_before)


def test_old_checkpoint_defaults_euler_steps_to_50() -> None:
    from mlx_train import config_from_saved

    saved = dict(_dblock_config(num_loops=4).__dict__)
    saved.pop("dblock_euler_steps")
    restored = config_from_saved(saved)
    assert restored.dblock_euler_steps == 50
    assert restored.dblock_sample_steps() == 50
    assert restored.num_loops == 4


def test_dblock_euler_steps_default_is_paper_50_not_num_loops() -> None:
    config = _dblock_config(num_loops=4)
    assert config.dblock_euler_steps == 50
    assert config.dblock_sample_steps() == 50
    assert config.dblock_sample_steps(8) == 8
    assert config.dblock_sample_steps() != config.num_loops
    split = _dblock_config(dblock_blocks=4, num_prelude_layers=2, num_recurrent_layers=4, num_coda_layers=2)
    assert split.dblock_sample_steps() == 4
    assert split.dblock_sample_steps(50) == 4
    with pytest.raises(ValueError, match="dblock_euler_steps"):
        _dblock_config(dblock_euler_steps=0)


def test_dblock_greedy_generate_returns_prompt_plus_suffix() -> None:
    from mlx_generate import dblock_greedy_generate

    mx.random.seed(5)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    prompt = [1, 2]
    out = dblock_greedy_generate(
        model, prompt, max_new_tokens=3, eos_token_id=None, euler_steps=2
    )
    assert out[:2] == prompt
    assert len(out) == 5
    assert all(0 <= token < 32 for token in out)


def test_dblock_generate_is_autoregressive_one_block_per_eval(monkeypatch) -> None:
    """B blocks → B evals per token, σ_max→σ_min, growing clean context."""
    from mlx_generate import dblock_greedy_generate

    model = MLXBitNet(_dblock_config(dblock_blocks=4))
    mx.eval(model.parameters())
    calls = []
    real = model.dblock_decode

    def spy(z, sigma, tokens, *args, **kwargs):
        calls.append((tokens.shape[1], z.shape[1], kwargs["block_id"], float(sigma.item())))
        return real(z, sigma, tokens, *args, **kwargs)

    monkeypatch.setattr(model, "dblock_decode", spy)
    dblock_greedy_generate(model, [1, 2], max_new_tokens=2, compile_step=False)
    # Per token: commit the previous position per block, then B cached queries
    # (one new position each). Nothing commits after the last token.
    queries = [c for c in calls if c[0] == 1 and c[1] == 1]
    assert [c[2] for c in calls[:4]] == [0, 1, 2, 3]
    assert [c[2] for c in calls] == [0, 1, 2, 3] * 4
    assert all(c[0] == c[1] == 1 for c in calls)
    assert len(queries) == len(calls)
    assert calls[4][3] == pytest.approx(80.0) and calls[7][3] == pytest.approx(0.002)


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("attn_res_mode", ["kimi", "sandwich"])
def test_dblock_cached_decode_matches_full_prefix(attn_res_mode: str, folded: bool, monkeypatch) -> None:
    """Cached per-block decode == full-prefix denoiser forward with the same z.

    folded: AdaRMS folded per σ (``dblock_decode_cond``) and packed pins, so the
    MLP runs the fused norm+bias kernels as in generate."""
    import mlx_model

    biased = []
    original = mlx_model.ternary_fused_linear_m1
    monkeypatch.setattr(
        mlx_model,
        "ternary_fused_linear_m1",
        lambda *a, **k: biased.append(k.get("norm_bias") is not None) or original(*a, **k),
    )
    mx.random.seed(7)
    config = _dblock_config(
        dblock_blocks=2, attn_res_mode=attn_res_mode, hidden_size=64, num_attention_heads=2, intermediate_size=128
    )
    model = MLXBitNet(config)
    for block in model.blocks:  # non-zero AdaRMS (zero-init is the identity)
        for ada in (block.ada_attn, block.ada_mlp):
            ada.proj.weight = mx.random.normal(ada.proj.weight.shape) * 0.1
            ada.proj.bias = mx.random.normal(ada.proj.bias.shape) * 0.1
    mx.eval(model.parameters())
    model.set_inference_block_width(4)  # train chunks = decode windows, as generate pins
    if folded:
        model.recurrent_quantized_matmul = True
        model.pin_inference_weights(mx.float32, prefer_packed=True)
    tokens = mx.random.randint(0, 32, (1, 11))
    length = tokens.shape[1]
    for block_id, value in enumerate((3.0, 0.05)):
        sigma = mx.array([value])
        cond = model.dblock_decode_cond(sigma, block_id) if folded else None
        clean = model.dblock_clean_embeddings(tokens[:, 1:])
        noise = mx.random.normal(clean.shape)
        z = mx.concatenate([clean + value * noise, mx.random.normal((1, 1, config.hidden_size))], axis=1)
        expected = model.dblock_forward_from_z(z, sigma, tokens, block_id=block_id)
        cache = model.new_dblock_cache(block_id)
        # prefill 3, extend 2, then one at a time: crosses the width-4 chunk edges.
        pieces = [(0, 3), (3, 5)] + [(i, i + 1) for i in range(5, length - 1)]
        for lo, hi in pieces:
            got = model.dblock_decode(z[:, lo:hi], sigma, tokens[:, lo:hi], cache, block_id=block_id, cond=cond)
            assert mx.allclose(got, expected[:, lo:hi], atol=1e-4).item(), (block_id, lo)
        for _ in range(2):  # queries on clones leave the cache alone
            got = model.dblock_decode(z[:, -1:], sigma, tokens[:, -1:], cache.clone(), block_id=block_id, cond=cond)
            assert mx.allclose(got, expected[:, -1:], atol=1e-4).item(), block_id
    assert any(biased) == folded  # folded decode runs the fused norm+shift MLP kernel


@pytest.mark.parametrize("infer", ["euler", "loops"])
def test_dblock_generation_excludes_undefined_ids(monkeypatch, infer) -> None:
    from mlx_generate import dblock_greedy_generate

    model = MLXBitNet(_dblock_config(dblock_infer=infer))
    monkeypatch.setattr(
        model, "logits_from",
        lambda hidden: mx.broadcast_to(mx.arange(32), (*hidden.shape[:-1], 32)),
    )
    assert dblock_greedy_generate(
        model, [1, 2], 3, euler_steps=1, valid_vocab_size=7,
    ) == [1, 2, 6, 6, 6]
    assert model.embedding.weight.shape[0] == 32


def test_dblock_generate_honors_inference_num_loops_and_loops_mode() -> None:
    from mlx_generate import dblock_greedy_generate

    mx.random.seed(6)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    model.inference_num_loops = 1
    prompt = [1, 2]
    euler = dblock_greedy_generate(
        model, prompt, max_new_tokens=2, eos_token_id=None, euler_steps=2
    )
    assert euler[:2] == prompt
    assert len(euler) == 4
    object.__setattr__(model.config, "dblock_infer", "loops")
    looped = dblock_greedy_generate(model, prompt, max_new_tokens=2, eos_token_id=None)
    assert looped[:2] == prompt
    assert len(looped) == 4
    object.__setattr__(model.config, "dblock_blocks", 4)
    with pytest.raises(ValueError, match="dblock_infer='loops' requires dblock_blocks=1"):
        dblock_greedy_generate(model, prompt, max_new_tokens=2)


def test_dblock_compiled_generate_matches_eager(monkeypatch) -> None:
    """Compiled token step (commit + B queries) == eager, across chunk edges."""
    import mlx_generate
    from mlx_generate import dblock_greedy_generate

    mx.random.seed(8)
    model = MLXBitNet(_dblock_config(dblock_blocks=2, hidden_size=64, num_attention_heads=2, intermediate_size=128))
    mx.eval(model.parameters())
    model.set_inference_block_width(4)
    model.recurrent_quantized_matmul = True
    model.pin_inference_weights(mx.float32, prefer_packed=True)

    logits = {False: [], True: []}
    mode = [False]
    argmax = mlx_generate._generation_argmax
    monkeypatch.setattr(
        mlx_generate, "_generation_argmax", lambda x, v: logits[mode[0]].append(x) or argmax(x, v)
    )
    ran = []
    compile_ = mx.compile

    def counting_compile(fn):
        compiled = compile_(fn)

        def call(*args):
            result = compiled(*args)
            ran.append(1)  # only after a successful compiled call
            return result

        return call

    monkeypatch.setattr(mx, "compile", counting_compile)
    out = {}
    for compiled in (False, True):
        mode[0] = compiled
        mx.random.seed(1)
        out[compiled] = dblock_greedy_generate(model, [3, 5, 7], 10, compile_step=compiled)
    assert out[True] == out[False]
    assert len(ran) == 10  # every token compiled (prompt > 2: all have a commit)
    for eager, fused in zip(logits[False], logits[True]):
        assert mx.allclose(eager, fused, atol=1e-4).item()


def _grouped_model(**overrides):
    mx.random.seed(9)
    config = _dblock_config(
        dblock_blocks=2, hidden_size=64, num_attention_heads=2, intermediate_size=128, **overrides
    )
    model = MLXBitNet(config)
    for block in model.blocks:  # non-zero AdaRMS and gates
        for ada in (block.ada_attn, block.ada_mlp):
            ada.proj.weight = mx.random.normal(ada.proj.weight.shape) * 0.1
            ada.proj.bias = mx.random.normal(ada.proj.bias.shape) * 0.1
        block.attn_gate = mx.random.normal((1,))
        for mix in (block.attn_res_mix, block.mlp_res_mix):
            mix.proj.weight = mx.random.normal(mix.proj.weight.shape) * 0.2
    mx.eval(model.parameters())
    model.set_inference_block_width(4)
    model.recurrent_quantized_matmul = True
    model.pin_inference_weights(mx.float32, prefer_packed=True)
    return model


def test_dblock_grouped_commit_matches_per_block() -> None:
    """One grouped commit chain == B per-block commits (caches and later queries)."""
    model = _grouped_model()
    hidden = model.config.hidden_size
    values = (3.0, 0.05)
    sigmas = [mx.array([v]) for v in values]
    conds = [model.dblock_decode_cond(s, k) for k, s in enumerate(sigmas)]
    group = model.dblock_commit_group([0, 1], conds)
    assert group is not None
    caches = [model.new_dblock_cache(k) for k in range(2)]
    stacked = model.dblock_stack_caches([model.new_dblock_cache(k) for k in range(2)])
    tokens = mx.random.randint(0, 32, (1, 10))
    for i in range(9):  # crosses the width-4 chunk edges twice (memory folds)
        clean = model.dblock_clean_embeddings(tokens[:, i + 1 : i + 2])
        noise = mx.random.normal((2, 1, hidden))
        for k in range(2):
            model.dblock_decode(clean + values[k] * noise[k], sigmas[k], tokens[:, i : i + 1], caches[k],
                                block_id=k, cond=conds[k])
        model.dblock_grouped_commit(clean + mx.array(values).reshape(2, 1, 1) * noise,
                                    mx.array(values), tokens[:, i : i + 1], group, stacked)
        z = mx.random.normal((1, 1, hidden))
        for k in range(2):
            want = model.dblock_decode(z, sigmas[k], tokens[:, i + 1 : i + 2], caches[k].clone(), block_id=k, cond=conds[k])
            got = model.dblock_decode(z, sigmas[k], tokens[:, i + 1 : i + 2],
                                      model.dblock_row_cache(stacked, k, i + 1), block_id=k, cond=conds[k])
            assert mx.allclose(got, want, atol=1e-4).item(), (i, k, mx.abs(got - want).max().item())


@pytest.mark.parametrize("prompt", [[3], [3, 5], [3, 5, 7, 9, 11]])
def test_dblock_grouped_generate_matches_per_block(monkeypatch, prompt) -> None:
    import mlx_generate
    from mlx_generate import dblock_greedy_generate

    model = _grouped_model()
    logits = {False: [], True: []}
    mode = [False]
    argmax = mlx_generate._generation_argmax
    monkeypatch.setattr(mlx_generate, "_generation_argmax", lambda x, v: logits[mode[0]].append(x) or argmax(x, v))
    grouped_calls = []
    real_commit = model.dblock_grouped_commit
    monkeypatch.setattr(model, "dblock_grouped_commit", lambda *a: grouped_calls.append(1) or real_commit(*a))
    real_group = model.dblock_commit_group
    out = {}
    for grouped in (False, True):
        mode[0] = grouped
        monkeypatch.setattr(model, "dblock_commit_group", real_group if grouped else lambda *a: None)
        mx.random.seed(1)
        out[grouped] = dblock_greedy_generate(model, prompt, 9)
    assert out[True] == out[False]
    assert len(grouped_calls) == 9 - (len(prompt) == 1)  # every token after the first has a commit
    for want, got in zip(logits[False], logits[True]):
        assert mx.allclose(want, got, atol=1e-4).item()
