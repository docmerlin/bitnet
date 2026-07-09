"""Unit tests for composed training loss (CE + MTP + RFMoE aux)."""

from __future__ import annotations


import torch
import torch.nn.functional as F

from config import TernaryConfig
from model import BitNetDeep
from training.losses import compute_train_loss, language_modeling_loss


def _tiny_config(**kwargs) -> TernaryConfig:
    base = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        use_hadamard=False,
        use_4bit_activations=False,
        mtp_depth=0,
        use_rfmoe=False,
    )
    base.update(kwargs)
    return TernaryConfig(**base)


def test_compute_train_loss_matches_plain_ce() -> bool:
    torch.manual_seed(0)
    model = BitNetDeep(_tiny_config())
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    plain = language_modeling_loss(logits, labels, z_loss_coef=0.0)
    composed = compute_train_loss(model, logits, labels, z_loss_coef=0.0)
    assert torch.allclose(plain, composed), (float(plain), float(composed))
    print("compute_train_loss dense CE match passed")
    return True


def test_compute_train_loss_adds_z_loss() -> bool:
    torch.manual_seed(1)
    model = BitNetDeep(_tiny_config())
    logits = torch.randn(2, 8, 64)
    labels = torch.randint(0, 64, (2, 8))
    base = compute_train_loss(model, logits, labels, z_loss_coef=0.0)
    with_z = compute_train_loss(model, logits, labels, z_loss_coef=1e-2)
    assert float(with_z) > float(base), (float(base), float(with_z))
    print("compute_train_loss z-loss term passed")
    return True


def test_rfmoe_aux_terms_are_independent() -> bool:
    """Locality/diversity apply when RFMoE is present even without density lam≠0."""
    torch.manual_seed(2)
    config = _tiny_config(
        use_rfmoe=True,
        rfmoe_num_experts=4,
        rfmoe_expert_dim=16,
        rfmoe_theta=0.01,
    )
    model = BitNetDeep(config)
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)

    ce_only = compute_train_loss(model, logits, labels, density_lam=0.0)
    with_locality = compute_train_loss(
        model, logits, labels, density_lam=0.0, locality_coef=0.5, rfmoe_s=1.0, rfmoe_alpha=0.1
    )
    # Locality KL is non-negative; when usage is not already the staircase it is > 0.
    assert float(with_locality.detach()) >= float(ce_only.detach()) - 1e-5, (
        float(ce_only.detach()),
        float(with_locality.detach()),
    )
    # Gradients should flow through locality without needing density_lam > 0.
    with_locality.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print("compute_train_loss RFMoE independent aux terms passed")
    return True


def test_bitnet_checkpointing_flag_on_model() -> bool:
    torch.manual_seed(3)
    config = _tiny_config(num_hidden_layers=2)
    model = BitNetDeep(config)
    model.gradient_checkpointing = True
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("BitNetDeep.gradient_checkpointing path passed")
    return True


if __name__ == "__main__":
    test_compute_train_loss_matches_plain_ce()
    test_compute_train_loss_adds_z_loss()
    test_rfmoe_aux_terms_are_independent()
    test_bitnet_checkpointing_flag_on_model()
