"""Entropy patching from a byte LM, and the property that makes it usable.

``StudentEntropyModel`` scores position ``t`` using byte ``t``, so it cannot say
anything about a byte that has not been generated -- which forced
``blt.generate`` onto a re-patch-every-byte baseline. ``ByteEntropyModel``
predicts the next byte instead, so its boundary decision depends only on
committed context. The tests below pin that offset, because getting it wrong
gives a model that looks fine on training data and silently decodes against the
wrong latent.
"""

import math

import numpy as np
import pytest
import torch

from blt.config import TernaryBLTConfig
from blt.generate import _Patching, generate
from blt.model import TernaryBLTModel
from blt.patching.entropy_model import (
    ByteEntropyModel,
    boundaries_from_entropy,
    calibrate_threshold,
    cap_patch_lengths,
    next_byte_entropy,
    patch_lengths_from_starts,
)
from blt.patching.student_entropy import StudentEntropyModel


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


def _entropy_model(config=None, **kwargs):
    torch.manual_seed(0)
    settings = dict(dim=32, num_layers=1, num_heads=4)
    settings.update(kwargs)
    return ByteEntropyModel(config or _config(), **settings).eval()


def test_entropy_of_a_uniform_distribution_is_log_vocab():
    logits = torch.zeros(1, 3, 256)
    assert torch.allclose(next_byte_entropy(logits), torch.full((1, 3), math.log(256)), atol=1e-5)


def test_entropy_of_a_certain_distribution_is_zero():
    logits = torch.full((1, 3, 256), -1e4)
    logits[..., 7] = 1e4
    assert float(next_byte_entropy(logits).max()) < 1e-4


def test_boundaries_read_the_previous_position_not_the_current_one():
    # The offset that makes this patcher causally usable. entropy[j-1] is the
    # prediction made before byte j existed, so it -- and only it -- may decide
    # whether byte j opens a patch.
    entropy = torch.tensor([[0.0, 9.0, 0.0, 0.0]])
    starts = boundaries_from_entropy(entropy, threshold=1.0)
    # entropy[1] is high, so position 2 begins a patch. Position 1 does not.
    assert starts.tolist() == [[True, False, True, False]]


def test_position_zero_always_begins_a_patch():
    entropy = torch.zeros(2, 5)
    assert boundaries_from_entropy(entropy, threshold=1.0)[:, 0].all()


def test_relative_rule_fires_on_a_rise_not_a_level():
    # Uniformly high entropy is not a boundary under the monotonic rule; a jump is.
    level = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    assert not boundaries_from_entropy(level, relative_threshold=0.5)[:, 1:].any()
    rise = torch.tensor([[1.0, 1.0, 4.0, 1.0]])
    assert boundaries_from_entropy(rise, relative_threshold=0.5).tolist() == [
        [True, False, False, True]
    ]


def test_both_rules_together_require_both():
    entropy = torch.tensor([[1.0, 9.0, 9.2, 0.0]])
    # Position 2 clears the global bar, and position 3 shows only a small rise.
    both = boundaries_from_entropy(entropy, threshold=5.0, relative_threshold=1.0)
    globally = boundaries_from_entropy(entropy, threshold=5.0)
    assert globally[0, 3]
    assert not both[0, 3]


@pytest.mark.parametrize("max_patch_length", [1, 2, 3, 8])
def test_cap_patch_lengths_bounds_every_run(max_patch_length):
    starts = torch.zeros(3, 20, dtype=torch.bool)
    starts[:, 0] = True
    starts[1, 7] = True
    capped = cap_patch_lengths(starts, max_patch_length)
    lengths = patch_lengths_from_starts(capped)
    assert int(lengths.max()) <= max_patch_length
    # Capping only adds boundaries; it never removes one.
    assert bool((capped | starts == capped).all())
    assert int(lengths.sum(dim=1).max()) == 20


def test_patch_lengths_sum_to_the_sequence():
    torch.manual_seed(1)
    starts = torch.rand(4, 32) < 0.3
    starts[:, 0] = True
    lengths = patch_lengths_from_starts(starts)
    assert torch.equal(lengths.sum(dim=1), torch.full((4,), 32))


def test_model_trains_down_on_a_memorisable_batch():
    config = _config()
    model = ByteEntropyModel(config, dim=32, num_layers=1, num_heads=4)
    tokens = torch.randint(config.offset, config.offset + 16, (4, 24))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = float(model.loss(tokens))
    for _ in range(40):
        optimizer.zero_grad()
        loss = model.loss(tokens)
        loss.backward()
        optimizer.step()
    assert float(loss) < first


def test_entropy_at_a_position_ignores_later_bytes():
    # The causality guarantee, tested directly: changing byte t cannot move the
    # entropy the model reported for any position before t.
    model = _entropy_model()
    tokens = torch.randint(4, 260, (1, 16))
    before = model.entropy(tokens)
    changed = tokens.clone()
    changed[0, 10] = (int(changed[0, 10]) + 40) % 256 + 4
    after = model.entropy(changed)
    assert torch.allclose(before[:, :10], after[:, :10], atol=1e-5)


def test_opens_new_patch_agrees_with_predict_patch_lengths():
    # The two APIs must not disagree: whether generation drifts past a byte and
    # how the sequence is later segmented have to be the same decision.
    from blt.patching.entropy_model import patch_lengths_from_starts as _lengths

    model = _entropy_model()
    tokens = torch.randint(4, 260, (1, 24))
    threshold = model.default_threshold

    for cut in range(4, 20):
        prefix = tokens[:, :cut]
        predicted = bool(model.opens_new_patch(prefix, threshold=threshold)[0])
        starts = boundaries_from_entropy(model.entropy(tokens[:, : cut + 1]), threshold=threshold)
        assert predicted == bool(starts[0, cut])


def test_calibrate_threshold_hits_the_requested_patch_size():
    model = _entropy_model()
    tokens = torch.randint(4, 260, (4, 256))
    for target in (2.0, 4.0, 8.0):
        calibrate_threshold(model, tokens, target_patch_size=target)
        lengths = model.predict_patch_lengths(tokens)
        mean_width = float(lengths.sum()) / float((lengths > 0).sum())
        assert mean_width == pytest.approx(target, rel=0.35), f"target {target}, got {mean_width}"


def test_calibrate_rejects_a_degenerate_target():
    with pytest.raises(ValueError, match="target_patch_size must exceed"):
        calibrate_threshold(_entropy_model(), torch.randint(4, 260, (1, 32)), target_patch_size=1.0)


def test_threshold_rides_along_in_the_state_dict():
    model = _entropy_model()
    model.set_threshold(3.25)
    restored = _entropy_model()
    restored.load_state_dict(model.state_dict())
    assert restored.default_threshold == pytest.approx(3.25)


def test_generation_treats_the_entropy_model_as_predictive():
    # The payoff. A retrospective classifier forces the baseline to re-patch
    # every byte (bytes_per_global_pass == 1.0); this one lets it drift.
    config = _config()
    torch.manual_seed(0)
    model = TernaryBLTModel(config).eval()
    entropy = _entropy_model(config)
    calibrate_threshold(entropy, torch.randint(4, 260, (2, 128)), target_patch_size=4.0)
    prompt = torch.randint(config.offset, config.offset + 256, (1, 8))

    assert _Patching(entropy).positional
    assert not _Patching(StudentEntropyModel(config, dim=32, num_layers=1, num_heads=4)).positional

    _, predictive = generate(model, prompt, max_new_bytes=24, patcher=entropy, speculation_window=0)
    _, retrospective = generate(
        model,
        prompt,
        max_new_bytes=24,
        patcher=StudentEntropyModel(config, dim=32, num_layers=1, num_heads=4),
        speculation_window=0,
    )
    assert retrospective.bytes_per_global_pass == pytest.approx(1.0, abs=0.1)
    assert predictive.bytes_per_global_pass > 1.5


@pytest.mark.parametrize("speculation_window", [0, 4, 8])
def test_speculation_still_matches_the_reference_under_entropy_patching(speculation_window):
    config = _config()
    torch.manual_seed(0)
    model = TernaryBLTModel(config).eval()
    entropy = _entropy_model(config)
    calibrate_threshold(entropy, torch.randint(4, 260, (2, 128)), target_patch_size=4.0)
    prompt = torch.randint(config.offset, config.offset + 256, (1, 8))

    patching = _Patching(entropy)
    reference = prompt
    with torch.no_grad():
        for _ in range(20):
            output = model(
                reference,
                patch_lengths=patching.patch_lengths(reference),
                attention_mask=torch.ones_like(reference, dtype=torch.bool),
            )
            reference = torch.cat([reference, output.logits[:, -1:].argmax(dim=-1)], dim=1)

    tokens, _ = generate(
        model, prompt, max_new_bytes=20, patcher=entropy, speculation_window=speculation_window
    )
    assert torch.equal(tokens, reference)


def test_oversized_sequence_is_refused():
    model = _entropy_model(max_seq_len=16)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_no_threshold_at_all_is_refused():
    with pytest.raises(ValueError, match="give threshold"):
        boundaries_from_entropy(torch.zeros(1, 4))


def test_patcher_is_consulted_once_per_byte():
    # Same dedup as the MLX path: asking the entropy model twice for the same
    # tokens is a wasted forward pass per byte.
    import blt.generate as module

    config = _config()
    torch.manual_seed(0)
    model = TernaryBLTModel(config).eval()
    entropy = _entropy_model(config)
    calibrate_threshold(entropy, torch.randint(4, 260, (2, 128)), target_patch_size=4.0)

    calls = []
    original = module._Patching.opens_new_patch
    module._Patching.opens_new_patch = lambda self, tokens: (
        calls.append(tokens.size(1)),
        original(self, tokens),
    )[1]
    try:
        _, stats = generate(
            model,
            torch.randint(config.offset, config.offset + 256, (1, 8)),
            max_new_bytes=24,
            patcher=entropy,
            speculation_window=0,
        )
    finally:
        module._Patching.opens_new_patch = original

    assert len(calls) <= stats.committed + 1
    assert len(calls) == len(set(calls))
