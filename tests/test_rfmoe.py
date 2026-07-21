"""RFMoE forward, density controller, locality, and diversity checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from layers.rfmoe import (
    DensityController,
    RFMoE,
    add_rfmoe_experts,
    rfmoe_aux_activity,
    rfmoe_density,
    rfmoe_diversity_loss,
    rfmoe_locality_loss,
    staircase_target,
)
from layers.h_bitlinear import HBitLinear


def test_rfmoe_forward_and_grad() -> None:
    torch.manual_seed(0)
    hidden = 64
    moe = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
    # Each expert body is 3-stage (up/mid/down).
    expert = moe.experts[0]
    assert all(
        isinstance(layer, HBitLinear)
        for layer in (expert.a_gate, expert.b_gate, expert.w_up, expert.w_mid, expert.w_down)
    )
    assert hasattr(expert, "w_mid")
    assert expert.w_mid.weight.shape == (32, 32)
    assert expert.w_up.weight.shape == (32, hidden)
    assert expert.w_down.weight.shape == (hidden, 32)
    eye = torch.eye(32, dtype=expert.w_mid.weight.dtype)
    assert torch.allclose(expert.w_mid.weight.detach().cpu(), eye, atol=1e-5), (
        "w_mid must cold-start as identity"
    )
    x = torch.randn(4, 10, hidden)
    y = moe(x)
    assert y.shape == x.shape

    target = torch.randn_like(y)
    opt = torch.optim.SGD(moe.parameters(), lr=0.1)
    loss0 = F.mse_loss(moe(x), target)
    opt.zero_grad()
    loss0.backward()
    assert expert.w_mid.weight.grad is not None, "w_mid must participate in backward"
    assert float(expert.w_mid.weight.grad.abs().sum()) > 0.0, "w_mid grad should be nonzero"
    opt.step()
    loss1 = F.mse_loss(moe(x), target)
    assert loss1 < loss0


def test_grouped_forward_matches_expert_loop() -> None:
    torch.manual_seed(4)
    moe = RFMoE(16, expert_dim=12, num_experts=4, rank=3, theta=0.2, residual=False)
    with torch.no_grad():
        for expert, bias in zip(moe.experts, (-0.5, 0.2, 0.6, 100.0)):
            expert.bias.fill_(bias)
    x = torch.randn(2, 5, 16)

    expected = torch.zeros_like(x.reshape(-1, 16))
    for expert in moe.experts:
        z, score = expert.score(x.reshape(-1, 16))
        gate = F.relu(score - expert.bias)
        idx = (gate >= moe.theta).nonzero(as_tuple=True)[0]
        if idx.numel():
            expected.index_add_(0, idx, gate[idx, None] * expert.expert(x.reshape(-1, 16)[idx], z[idx]))

    torch.testing.assert_close(moe(x), expected.reshape_as(x))


def test_add_expert_preserves_output_and_can_train_new_only() -> None:
    dense = torch.nn.Linear(2, 2)
    assert add_rfmoe_experts(dense) == []
    assert all(parameter.requires_grad for parameter in dense.parameters())

    torch.manual_seed(5)
    moe = RFMoE(16, expert_dim=12, num_experts=3, rank=3, theta=0.2, residual=False)
    x = torch.randn(2, 5, 16)
    before = moe(x).detach()

    cold = moe.add_expert(bias=1e6)
    torch.testing.assert_close(moe(x), before)
    assert len(moe.experts) == 4
    assert moe.usage_ema.shape == (4,)
    assert cold.a_gate.weight.device == moe.experts[0].a_gate.weight.device

    added = add_rfmoe_experts(moe, bias=1e-6, freeze_existing=True)
    assert added == [moe.experts[-1]]
    assert moe.config.rfmoe_num_experts == 5
    assert all(not parameter.requires_grad for expert in moe.experts[:-1] for parameter in expert.parameters())
    assert all(parameter.requires_grad for parameter in added[0].parameters())
    moe(x).square().mean().backward()
    assert any(parameter.grad is not None for parameter in added[0].parameters())

    restored = RFMoE(
        16, expert_dim=12, num_experts=moe.config.rfmoe_num_experts, rank=3, theta=moe.theta, residual=False
    )
    restored.load_state_dict(moe.state_dict())
    torch.testing.assert_close(restored(x), moe(x))


def test_theta_controls_density() -> None:
    torch.manual_seed(0)
    moe = RFMoE(64, expert_dim=32, num_experts=8, rank=4, theta=0.01)
    moe.train()
    x = torch.randn(4, 10, 64)
    moe.theta = 0.01
    moe(x)
    density_low = moe._last_density
    moe.theta = 5.0
    moe(x)
    density_high = moe._last_density
    assert density_high < density_low


def test_density_controller_and_penalty() -> None:
    ctrl = DensityController(target=0.3, eta=0.1, lam=1.0)
    ctrl.update(0.9)
    assert ctrl.lam > 1.0
    ctrl.update(0.1)
    ctrl.update(0.1)
    assert ctrl.lam < 1.0

    hidden, x = 64, torch.randn(4, 10, 64)

    def train_density(lam: float, steps: int = 150) -> float:
        torch.manual_seed(1)
        moe = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
        opt = torch.optim.SGD(moe.parameters(), lr=0.1)
        tgt = torch.randn(4, 10, hidden)
        for _ in range(steps):
            loss = F.mse_loss(moe(x), tgt) + lam * rfmoe_aux_activity(moe)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return rfmoe_density(moe)

    d_free = train_density(0.0)
    d_pen = train_density(2.0)
    assert d_free > 0.9
    assert d_pen < d_free


def test_staircase_and_locality() -> None:
    pi = staircase_target(8, s=1.0, alpha=0.1)
    assert abs(float(pi.sum()) - 1.0) < 1e-5
    assert torch.all(pi[:-1] >= pi[1:])
    assert float(pi.min()) >= 0.1 / 8 - 1e-6

    torch.manual_seed(2)
    hidden, x = 64, torch.randn(4, 10, 64)
    moe = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
    opt = torch.optim.SGD(moe.parameters(), lr=0.1)
    tgt = torch.randn(4, 10, hidden)
    moe(x)
    kl_start = float(rfmoe_locality_loss(moe, s=1.0, alpha=0.1).detach())
    for _ in range(200):
        loss = F.mse_loss(moe(x), tgt) + 0.5 * rfmoe_locality_loss(moe, s=1.0, alpha=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    moe(x)
    kl_end = float(rfmoe_locality_loss(moe, s=1.0, alpha=0.1).detach())
    p_end = (moe._last_usage / moe._last_usage.sum()).detach().sort(descending=True).values
    assert kl_end < kl_start
    assert float(p_end.min()) > 0.0


def test_diversity_loss() -> None:
    moe = RFMoE(64, expert_dim=32, num_experts=3, rank=4, theta=0.01)
    moe._last_gate = torch.tensor([[1.0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]])
    same = float(rfmoe_diversity_loss(moe))
    moe._last_gate = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    disjoint = float(rfmoe_diversity_loss(moe))
    assert same > 0.5 and disjoint < same

    torch.manual_seed(3)
    hidden, x = 64, torch.randn(4, 10, 64)
    moe = RFMoE(hidden, expert_dim=32, num_experts=6, rank=4, theta=0.01)
    opt = torch.optim.SGD(moe.parameters(), lr=0.2)
    tgt = torch.randn(4, 10, hidden)
    moe(x)
    div_start = float(rfmoe_diversity_loss(moe).detach())
    for _ in range(200):
        loss = F.mse_loss(moe(x), tgt) + 1.0 * rfmoe_diversity_loss(moe)
        opt.zero_grad()
        loss.backward()
        opt.step()
    moe(x)
    div_end = float(rfmoe_diversity_loss(moe).detach())
    assert div_end < div_start
