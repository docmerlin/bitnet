"""Logit softcap: train-time smooth bound on LM logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.losses import language_modeling_loss, softcap_logits


def test_softcap_identity_near_zero() -> None:
    x = torch.linspace(-1.0, 1.0, 21)
    y = softcap_logits(x, 30.0)
    assert torch.allclose(y, x, atol=2e-3)


def test_softcap_bounds_extremes() -> None:
    x = torch.tensor([-100.0, 0.0, 100.0])
    y = softcap_logits(x, 15.0)
    assert float(y[0]) > -15.0 and float(y[0]) < -14.9
    assert abs(float(y[1])) < 1e-6
    assert float(y[2]) < 15.0 and float(y[2]) > 14.9


def test_softcap_disabled_is_noop() -> None:
    x = torch.randn(4, 8, 16)
    assert softcap_logits(x, 0.0) is x
    assert softcap_logits(x, -1.0) is x


def test_softcap_changes_ce_on_large_logits() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 32) * 40.0  # deliberately huge
    labels = torch.randint(0, 32, (2, 5))
    plain = language_modeling_loss(logits, labels, logit_softcap=0.0)
    capped = language_modeling_loss(logits, labels, logit_softcap=30.0)
    # Same targets; softcap changes the effective distribution → different CE.
    assert not torch.allclose(plain, capped, atol=1e-4)


def test_softcap_zero_matches_plain_ce() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 32)
    labels = torch.randint(0, 32, (2, 5))
    plain = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
    via = language_modeling_loss(logits, labels, logit_softcap=0.0)
    assert torch.allclose(plain, via)
