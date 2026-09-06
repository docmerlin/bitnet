"""DiffusionBlocks Huginn path: AR identity, finite denoise CE, slice grads."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_train import _masked_ce, create_dblock_gradient_step


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
    logits = model.dblock_logits(tokens, segments, sigma, eps)
    mx.eval(logits)
    assert logits.shape == (2, 4, 32)
    assert bool(mx.all(mx.isfinite(logits)).item())


@pytest.mark.parametrize("sigma_out", [0.0, 1.0, 2.0])
def test_euler_perfect_denoiser_reduces_noise(monkeypatch, sigma_out) -> None:
    model = MLXBitNet(_dblock_config())
    clean = mx.eye(model.config.hidden_size)[:1][None]
    monkeypatch.setattr(model, "dblock_forward_from_z", lambda *args, **kwargs: clean)
    noise = mx.ones_like(clean)
    actual = model.dblock_euler_step(clean + 2.0 * noise, mx.array(2.0), mx.array(sigma_out))
    assert mx.allclose(actual, clean + sigma_out * noise).item()


def test_dblock_ce_ignores_shifted_targets() -> None:
    mx.random.seed(7)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    step = create_dblock_gradient_step(model, compile_step=False, block_id=0)
    tokens = mx.random.randint(0, 32, (1, 4))
    other = (tokens + 3) % 32
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    sigma = mx.array(0.8)
    eps = mx.random.normal((1, 4, 16))
    args = (
        tokens,
        other,
        segments,
        segments,
        mx.array(0.0),
        mx.array(1.0),
        mx.array(0.1),
        sigma,
        eps,
        mx.array(1.0),
    )
    loss_other, _ = step(*args)
    loss_same, _ = step(tokens, tokens, *args[2:])
    logits = model.dblock_logits(tokens, segments, sigma, eps)
    expected = _masked_ce(logits, tokens, mx.ones(tokens.shape, dtype=mx.bool_))
    mx.eval(loss_other, loss_same, expected)
    assert mx.allclose(loss_other, loss_same, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(loss_same, expected, rtol=1e-5, atol=1e-5).item()


def test_dblock_train_step_is_finite() -> None:
    mx.random.seed(2)
    model = MLXBitNet(_dblock_config())
    mx.eval(model.parameters())
    step = create_dblock_gradient_step(model, compile_step=False, block_id=0)
    tokens = mx.random.randint(0, 32, (1, 4))
    targets = mx.random.randint(0, 32, (1, 4))
    segments = mx.zeros(tokens.shape, dtype=mx.int32)
    sigma = mx.array(0.8)
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
        mx.array(1.0),
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
        mx.array(1.2),
        mx.random.normal((1, 4, 16)),
        mx.array(1.0),
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
