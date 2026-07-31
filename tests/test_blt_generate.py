"""BLT-S self-speculation must be a pure speed knob.

The contract from arXiv:2605.08044 is that greedy self-speculative decoding is
byte-identical to greedy autoregressive decoding -- verification rejects any
drafted byte the full model disagrees with. If that equality ever breaks, the
speculation window has stopped being free and these tests are the tripwire.
"""

import pytest
import torch

from blt.config import TernaryBLTConfig
from blt.generate import GenerationStats, _draft, _Patching, _run_global, _verify, generate
from blt.model import TernaryBLTModel
from blt.patching.student_entropy import StudentEntropyModel
from blt.patching.teacher_patcher import UniformPatcher, patch_ids_from_lengths


def _model(patch_size: int = 2, pad_id: int = -1) -> TernaryBLTModel:
    config = TernaryBLTConfig(
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
        patch_size=patch_size,
        local_window=None,
        pad_id=pad_id,
    )
    torch.manual_seed(0)
    model = TernaryBLTModel(config)
    model.eval()
    return model


def _prompt(model: TernaryBLTModel, length: int = 5) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(model.config.offset, model.config.offset + 256, (1, length))


def _entropy_patcher(model: TernaryBLTModel) -> StudentEntropyModel:
    torch.manual_seed(3)
    patcher = StudentEntropyModel(model.config, dim=32, num_layers=1, num_heads=4)
    patcher.eval()
    return patcher


@torch.no_grad()
def _reference_greedy(model, prompt, count, patcher):
    """Ground-truth greedy decode: re-patch the whole prefix for every byte.

    Deliberately naive. Both ``generate`` strategies are optimisations of this
    loop and must reproduce it exactly, whatever the patcher.
    """
    patching = _Patching(patcher)
    tokens = prompt
    for _ in range(count):
        output = model(
            tokens,
            patch_lengths=patching.patch_lengths(tokens),
            attention_mask=torch.ones_like(tokens, dtype=torch.bool),
        )
        tokens = torch.cat([tokens, output.logits[:, -1:].argmax(dim=-1)], dim=1)
    return tokens


@pytest.mark.parametrize("speculation_window", [1, 2, 4, 8])
def test_self_speculation_matches_autoregressive(speculation_window):
    model = _model()
    prompt = _prompt(model)

    baseline, _ = generate(model, prompt, max_new_bytes=16, speculation_window=0)
    speculative, _ = generate(model, prompt, max_new_bytes=16, speculation_window=speculation_window)

    assert torch.equal(baseline, speculative)


def test_generation_respects_the_byte_budget():
    # Verification hands back a free byte on a full-length match, which is the
    # one path that can run past the requested budget.
    model = _model()
    prompt = _prompt(model)
    for window in (0, 8):
        tokens, _ = generate(model, prompt, max_new_bytes=10, speculation_window=window)
        assert tokens.size(1) == prompt.size(1) + 10
        assert torch.equal(tokens[:, : prompt.size(1)], prompt)


def test_speculation_trades_global_calls_for_decoder_calls():
    model = _model()
    prompt = _prompt(model)

    _, baseline = generate(model, prompt, max_new_bytes=24, speculation_window=0)
    _, speculative = generate(model, prompt, max_new_bytes=24, speculation_window=8)

    assert speculative.decoder > baseline.decoder
    assert 0.0 <= speculative.acceptance_rate <= 1.0
    assert speculative.accepted <= speculative.drafted


@pytest.mark.parametrize("speculation_window", [2, 4, 8])
def test_speculation_runs_one_global_pass_per_round(speculation_window):
    # The verification pass doubles as the next round's encoder/global pass. If
    # that reuse regresses, BLT-S pays twice for the model it exists to skip and
    # every speedup in the paper evaporates.
    model = _model()
    prompt = _prompt(model)
    _, stats = generate(model, prompt, max_new_bytes=24, speculation_window=speculation_window)

    # One seeding pass for the prompt, then one per round; each round commits at
    # least the verified byte, so rounds can never exceed the bytes produced.
    rounds = stats.global_model - 1
    assert 1 <= rounds <= stats.committed
    assert stats.committed == 24
    assert stats.bytes_per_global_pass > 1.0


def test_baseline_runs_the_global_model_once_per_patch():
    model = _model(patch_size=4)
    prompt = _prompt(model, length=4)
    _, stats = generate(model, prompt, max_new_bytes=16, speculation_window=0)
    # 16 new bytes at 4 bytes per patch, plus the pass that opens the first one.
    assert stats.global_model == pytest.approx(16 / 4, abs=1)
    assert stats.drafted == 0


def test_batched_generation_is_refused():
    model = _model()
    with pytest.raises(ValueError, match=r"\[1, seq_len\]"):
        generate(model, _prompt(model).repeat(2, 1), max_new_bytes=4)


def test_unsupported_patcher_is_refused():
    model = _model()
    with pytest.raises(TypeError, match="unsupported patcher"):
        generate(model, _prompt(model), max_new_bytes=4, patcher=object())


def test_generation_leaves_training_mode_untouched():
    model = _model()
    model.train()
    generate(model, _prompt(model), max_new_bytes=4, speculation_window=4)
    assert model.training


def test_memory_bandwidth_tracks_the_component_counts():
    model = _model()
    stats = GenerationStats(encoder=1, global_model=1, decoder=1)
    single = stats.memory_bandwidth_gb(model)
    assert single > 0.0
    doubled = GenerationStats(encoder=2, global_model=2, decoder=2).memory_bandwidth_gb(model)
    assert doubled == pytest.approx(2 * single)


@pytest.mark.parametrize("speculation_window", [0, 1, 4, 8])
def test_content_scored_patcher_still_decodes_greedily(speculation_window):
    # A StudentEntropyModel scores position t from context including byte t, so
    # nothing can be said about a byte that does not exist yet. Reading the last
    # committed byte's score as if it described the next one made the baseline
    # disagree with its own patch_lengths and decode against the wrong latent.
    model = _model()
    prompt = _prompt(model)
    patcher = _entropy_patcher(model)

    tokens, _ = generate(
        model, prompt, max_new_bytes=12, patcher=patcher, speculation_window=speculation_window
    )
    assert torch.equal(tokens, _reference_greedy(model, prompt, 12, patcher))


def test_content_scored_baseline_reports_its_true_cost():
    # It cannot drift to the next boundary, so it re-patches every byte. That is
    # correct but is not the paper's once-per-patch baseline, and the stats have
    # to show it rather than flattering a BLT-S comparison.
    model = _model()
    _, stats = generate(
        model, _prompt(model), max_new_bytes=12, patcher=_entropy_patcher(model), speculation_window=0
    )
    assert stats.bytes_per_global_pass == pytest.approx(1.0, abs=0.1)


def test_opens_new_patch_refuses_a_retrospective_patcher():
    # StudentEntropyModel scores position t using byte t, so it cannot be asked
    # about a byte that does not exist yet. ByteEntropyModel can -- see
    # tests/test_blt_entropy_model.py.
    model = _model()
    patching = _Patching(_entropy_patcher(model))
    assert not patching.positional
    with pytest.raises(RuntimeError, match="undefined for a retrospective patcher"):
        patching.opens_new_patch(_prompt(model))


@pytest.mark.parametrize("speculation_window", [1, 2, 4, 8])
def test_speculation_stops_at_eos(speculation_window):
    # _draft runs through EOS and _verify commits a free byte on top, so without
    # an explicit trim the speculative path returns bytes the baseline never emits.
    model = _model()
    prompt = _prompt(model)
    unbounded, _ = generate(model, prompt, max_new_bytes=16, speculation_window=0)
    eos_id = int(unbounded[0, prompt.size(1) + 4].item())

    baseline, _ = generate(model, prompt, max_new_bytes=16, speculation_window=0, eos_id=eos_id)
    speculative, _ = generate(
        model, prompt, max_new_bytes=16, speculation_window=speculation_window, eos_id=eos_id
    )

    assert int(baseline[0, -1].item()) == eos_id
    assert torch.equal(baseline, speculative)


@pytest.mark.parametrize("patch_size", [1, 2, 3, 4, 6])
def test_verify_hands_back_the_patch_id_the_model_will_actually_use(patch_size):
    # _verify decides how many of its latents survive the rollback. Keeping one
    # too few is invisible in the output -- verification still rejects whatever
    # the stale latent got wrong -- but it makes the next round draft against the
    # wrong patch and eat the rejection. Pin the invariant directly: the next_id
    # handed back must be the patch the full model assigns to the resumed byte.
    model = _model(patch_size)
    patching = _Patching(UniformPatcher(patch_size))
    stats = GenerationStats()
    prompt = _prompt(model, length=7)

    latents, patch_ids = _run_global(model, prompt, patching, stats)
    candidate = _draft(
        model, prompt, latents, patch_ids, next_id=int(patch_ids[0, -1].item()), count=6, stats=stats
    )
    tokens, latents, _, next_id = _verify(model, prompt, candidate, patching, stats)

    truth = patch_ids_from_lengths(patching.patch_lengths(tokens), tokens.size(1))
    assert next_id == int(truth[0, -1].item())
    # The latent that byte reads must exist: a patch id of i indexes latents[i-1].
    assert latents.size(1) >= next_id


def test_padded_prompt_is_refused():
    # The draft path treats every byte as real while model.forward derives a mask
    # from pad_id; the two only agree when there is no padding. Batch is 1, so
    # padding is never needed anyway.
    model = _model(pad_id=300)
    prompt = torch.cat([_prompt(model, 4), torch.full((1, 2), 300)], dim=1)
    with pytest.raises(ValueError, match="padded prompt"):
        generate(model, prompt, max_new_bytes=4, speculation_window=4)


def test_generating_the_pad_byte_does_not_break_verification():
    # model.forward would otherwise infer a mask with a hole in it and trip
    # validate_suffix_padded_mask from inside _verify.
    model = _model(pad_id=260)
    prompt = _prompt(model)
    baseline, _ = generate(model, prompt, max_new_bytes=8, speculation_window=0)
    speculative, _ = generate(model, prompt, max_new_bytes=8, speculation_window=4)
    assert torch.equal(baseline, speculative)


def test_draft_passes_are_charged_less_than_full_encoder_passes():
    # encode_bytes skips patch_init_proj and patch_cross_attn. Charging drafts the
    # whole encoder overstates BLT-S bandwidth, and that number is what decides
    # whether speculation is worth enabling.
    model = _model()
    full = GenerationStats(encoder=1).memory_bandwidth_gb(model)
    draft = GenerationStats(draft_encoder=1).memory_bandwidth_gb(model)
    assert 0.0 < draft < full


def test_patcher_override_is_honoured():
    model = _model(patch_size=2)
    prompt = _prompt(model)
    _, fine = generate(model, prompt, max_new_bytes=12, speculation_window=0)
    _, coarse = generate(
        model, prompt, max_new_bytes=12, speculation_window=0, patcher=UniformPatcher(6)
    )
    assert coarse.global_model < fine.global_model
