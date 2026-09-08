"""MLX generation must match torch generation, byte for byte.

Two contracts stacked. Within MLX, ``speculation_window`` is a pure speed knob:
greedy self-speculation rejects anything the full model disagrees with, so every
window must produce identical output. Across frameworks, the MLX port must
reproduce the torch reference -- otherwise a model trained in MLX and sampled in
torch (or the reverse) silently generates different text.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_unflatten

from blt.config import TernaryBLTConfig
from blt.generate import generate as torch_generate
from blt.mlx_entropy_model import MLXByteEntropyModel, calibrate_threshold
from blt.mlx_generate import (
    GenerationStats,
    _Patching,
    _embed_last_byte,
    _ngram_embed_window,
    generate,
)
from blt.mlx_model import MLXTernaryBLTModel
from blt.model import TernaryBLTModel
from blt.patching.teacher_patcher import UniformPatcher


def _config(**overrides) -> TernaryBLTConfig:
    base = dict(
        local_dim=32,
        global_dim=32,
        decoder_dim=32,
        n_layers_local_encoder=1,
        n_layers_global=1,
        n_layers_local_decoder=1,
        n_heads_local_encoder=4,
        n_heads_global=4,
        n_heads_local_decoder=4,
        n_heads_cross=4,
        local_window=None,
        patch_size=4,
    )
    base.update(overrides)
    return TernaryBLTConfig(**base)


def _pair(config):
    torch.manual_seed(0)
    torch_model = TernaryBLTModel(config).eval()
    mlx_model = MLXTernaryBLTModel(config)
    mlx_model.update(
        tree_unflatten(
            [(name, mx.array(t.detach().numpy())) for name, t in torch_model.state_dict().items()]
        )
    )
    mx.eval(mlx_model.parameters())
    return torch_model, mlx_model


def _prompt(length=8, seed=1):
    return np.random.default_rng(seed).integers(4, 260, size=(1, length))


@pytest.mark.parametrize("cross_attn_k", [1, 2])
@pytest.mark.parametrize("speculation_window", [0, 1, 4, 8])
def test_matches_torch_generation(cross_attn_k, speculation_window):
    config = _config(cross_attn_k=cross_attn_k)
    torch_model, mlx_model = _pair(config)
    prompt = _prompt()

    expected, _ = torch_generate(
        torch_model, torch.from_numpy(prompt), max_new_bytes=16, speculation_window=speculation_window
    )
    actual, _ = generate(
        mlx_model, mx.array(prompt), max_new_bytes=16, speculation_window=speculation_window
    )
    assert np.array_equal(np.asarray(actual), expected.numpy())


@pytest.mark.parametrize("speculation_window", [1, 2, 4, 8, 16])
def test_speculation_is_a_pure_speed_knob(speculation_window):
    config = _config()
    _, mlx_model = _pair(config)
    prompt = mx.array(_prompt())

    baseline, _ = generate(mlx_model, prompt, max_new_bytes=24, speculation_window=0)
    speculative, _ = generate(
        mlx_model, prompt, max_new_bytes=24, speculation_window=speculation_window
    )
    assert np.array_equal(np.asarray(baseline), np.asarray(speculative))


def test_speculation_runs_one_global_pass_per_round():
    # The verification pass doubles as the next round's encoder/global pass. If
    # that regresses, BLT-S pays twice for the model it exists to skip.
    config = _config()
    _, mlx_model = _pair(config)
    _, stats = generate(mlx_model, mx.array(_prompt()), max_new_bytes=24, speculation_window=8)
    rounds = stats.global_model - 1
    assert 1 <= rounds <= stats.committed
    assert stats.committed == 24
    assert stats.bytes_per_global_pass > 1.0


def test_budget_is_respected():
    config = _config()
    _, mlx_model = _pair(config)
    prompt = mx.array(_prompt())
    for window in (0, 8):
        tokens, stats = generate(mlx_model, prompt, max_new_bytes=10, speculation_window=window)
        assert tokens.shape[1] == prompt.shape[1] + 10
        assert stats.committed == 10


@pytest.mark.parametrize("speculation_window", [0, 4, 8])
def test_generation_stops_at_eos(speculation_window):
    config = _config()
    _, mlx_model = _pair(config)
    prompt = mx.array(_prompt())
    unbounded, _ = generate(mlx_model, prompt, max_new_bytes=16, speculation_window=0)
    eos_id = int(unbounded[0, prompt.shape[1] + 4].item())

    baseline, _ = generate(mlx_model, prompt, max_new_bytes=16, speculation_window=0, eos_id=eos_id)
    tokens, _ = generate(
        mlx_model, prompt, max_new_bytes=16, speculation_window=speculation_window, eos_id=eos_id
    )
    assert int(baseline[0, -1].item()) == eos_id
    assert np.array_equal(np.asarray(tokens), np.asarray(baseline))


@pytest.mark.parametrize("cap_only", [False, True])
def test_entropy_patcher_drives_generation(cap_only):
    config = _config(max_patch_length=3 if cap_only else 32)
    _, mlx_model = _pair(config)
    mx.random.seed(0)
    patcher = MLXByteEntropyModel(config, dim=32, num_layers=1, num_heads=4, max_seq_len=128)
    mx.eval(patcher.parameters())
    calibrate_threshold(patcher, mx.array(_prompt(64, seed=5)), target_patch_size=4.0)
    if cap_only:
        patcher.set_threshold(100.0)

    reference = mx.array(_prompt())
    for _ in range(20):
        output = mlx_model(reference, patch_lengths=patcher.predict_patch_lengths(reference))
        reference = mx.concatenate([reference, mx.argmax(output.logits[:, -1:], axis=-1)], axis=1)
        mx.eval(reference)
    for window in (0, 4, 8):
        tokens, stats = generate(
            mlx_model,
            mx.array(_prompt()),
            max_new_bytes=20,
            patcher=patcher,
            speculation_window=window,
            eos_id=-1,
        )
        # An entropy patcher predicts, so even the baseline drifts past bytes
        # rather than re-patching each one.
        assert stats.bytes_per_global_pass > 1.0
        assert np.array_equal(np.asarray(tokens), np.asarray(reference))


def test_uniform_patcher_override_changes_the_pace():
    config = _config()
    _, mlx_model = _pair(config)
    prompt = mx.array(_prompt())
    _, fine = generate(
        mlx_model, prompt, max_new_bytes=16, patcher=UniformPatcher(2), speculation_window=0
    )
    _, coarse = generate(
        mlx_model, prompt, max_new_bytes=16, patcher=UniformPatcher(8), speculation_window=0
    )
    assert coarse.global_model < fine.global_model


def test_batched_generation_is_refused():
    config = _config()
    _, mlx_model = _pair(config)
    with pytest.raises(ValueError, match=r"\[1, seq_len\]"):
        generate(mlx_model, mx.array(_prompt().repeat(2, axis=0)), max_new_bytes=4)


def test_padded_prompt_is_refused():
    config = _config(pad_id=300)
    _, mlx_model = _pair(config)
    prompt = np.concatenate([_prompt(4), np.full((1, 2), 300)], axis=1)
    with pytest.raises(ValueError, match="padded prompt"):
        generate(mlx_model, mx.array(prompt), max_new_bytes=4)


def test_unsupported_patcher_is_refused():
    config = _config()
    _, mlx_model = _pair(config)
    with pytest.raises(TypeError, match="unsupported patcher"):
        generate(mlx_model, mx.array(_prompt()), max_new_bytes=4, patcher=object())


def test_stats_bytes_per_global_pass_is_zero_without_passes():
    assert GenerationStats().bytes_per_global_pass == 0.0
    assert GenerationStats().acceptance_rate == 1.0


def test_both_patchers_are_predictive():
    config = _config()
    mx.random.seed(0)
    entropy = MLXByteEntropyModel(config, dim=32, num_layers=1, num_heads=4)
    assert _Patching(UniformPatcher(4)).positional
    assert _Patching(entropy).positional


def test_patcher_is_consulted_once_per_byte():
    # The boundary decision was being asked twice for identical tokens -- once
    # to end the drafting run, once to compute the next patch id. With an
    # entropy patcher each ask is a full forward of the patcher.
    import blt.mlx_generate as module

    config = _config()
    _, mlx_model = _pair(config)
    mx.random.seed(0)
    patcher = MLXByteEntropyModel(config, dim=32, num_layers=1, num_heads=4, max_seq_len=128)
    mx.eval(patcher.parameters())
    calibrate_threshold(patcher, mx.array(_prompt(64, seed=5)), target_patch_size=4.0)

    calls = []
    original = module._Patching.opens_new_patch
    module._Patching.opens_new_patch = lambda self, tokens: (
        calls.append(tokens.shape[1]),
        original(self, tokens),
    )[1]
    try:
        _, stats = module.generate(
            mlx_model, mx.array(_prompt()), max_new_bytes=24, patcher=patcher, speculation_window=0
        )
    finally:
        module._Patching.opens_new_patch = original

    # One decision per generated byte, plus the one that seeds the first round.
    assert len(calls) <= stats.committed + 1
    assert len(calls) == len(set(enumerate(calls)))  # no repeated call at the same length


@pytest.mark.parametrize(
    "use_ngrams,sizes",
    [(True, (3, 4, 5, 6, 7, 8)), (True, (2, 4)), (False, (3, 4, 5, 6, 7, 8))],
)
def test_last_byte_embed_matches_full_prefix(use_ngrams, sizes):
    config = _config(use_ngram_embeddings=use_ngrams, ngram_sizes=sizes)
    _, model = _pair(config)
    window = _ngram_embed_window(model)
    assert window == (1 if not use_ngrams else max(sizes))
    for length in (1, 2, window, window + 5, 20):
        tokens = mx.array(_prompt(length, seed=length + 3))
        expected = model.embed_bytes(tokens, mx.ones(tokens.shape, dtype=mx.bool_))[:, -1:]
        actual = _embed_last_byte(model, tokens)
        assert np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-5), length


@pytest.mark.parametrize("cross_attn_k", [1, 2])
def test_generation_matches_torch_with_custom_ngrams(cross_attn_k):
    config = _config(cross_attn_k=cross_attn_k, ngram_sizes=(2, 5))
    torch_model, mlx_model = _pair(config)
    prompt = _prompt()
    expected, _ = torch_generate(
        torch_model, torch.from_numpy(prompt), max_new_bytes=12, speculation_window=4
    )
    actual, _ = generate(
        mlx_model, mx.array(prompt), max_new_bytes=12, speculation_window=4
    )
    assert np.array_equal(np.asarray(actual), expected.numpy())
