"""RFMoE forward, density controller, locality, and diversity checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from layers.rfmoe import (
    DensityController,
    RFMoE,
    rfmoe_aux_activity,
    rfmoe_density,
    rfmoe_diversity_loss,
    rfmoe_locality_loss,
    staircase_target,
)


def test_rfmoe_forward_and_grad() -> None:
    torch.manual_seed(0)
    hidden = 64
    moe = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
    x = torch.randn(4, 10, hidden)
    y = moe(x)
    assert y.shape == x.shape

    target = torch.randn_like(y)
    opt = torch.optim.SGD(moe.parameters(), lr=0.1)
    loss0 = F.mse_loss(moe(x), target)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    loss1 = F.mse_loss(moe(x), target)
    assert loss1 < loss0


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
