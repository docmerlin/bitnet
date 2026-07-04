"""Routing-free Mixture-of-Experts (RFMoE) FFN — self-gating experts.

Step 1 of the MoE roadmap in ``todo.md``. Each expert decides its OWN activation
from an internal score, so there is no centralized router: adding or removing an
expert never renormalizes the others (that is what makes the design extensible).

This is the minimal cut — dense float experts, per-expert skip path, score-
weighted residual combine, and a single global threshold ``theta`` as the
compute/quality knob. Deferred to later roadmap steps: ternary HBitLinear
experts, adaptive-lambda density control + global density target, and the
locality/staircase usage target.

Reference: RFMoE (arXiv 2604.00801), building on AoE (2501.13074) / ReMoE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFMoEExpert(nn.Module):
    """One self-gating GLU expert.

    ``FFN(x) = [sigmoid(x A_gate B_gate) ⊙ (x W_up)] W_down``

    The rank-r projection ``z = x A_gate`` is computed ONCE: its L2 norm is the
    self-score that makes the fire decision, and the same ``z`` feeds the gate
    branch ``B_gate``. So a token that doesn't fire pays only the cheap D×r
    projection and skips ``W_up``/``B_gate``/``W_down`` — that is the FLOP saving.
    """

    def __init__(self, hidden_size: int, expert_dim: int, rank: int) -> None:
        super().__init__()
        self.a_gate = nn.Linear(hidden_size, rank, bias=False)        # D -> r  (score + gate, dual use)
        self.b_gate = nn.Linear(rank, expert_dim, bias=False)         # r -> D_act
        self.w_up = nn.Linear(hidden_size, expert_dim, bias=False)    # D -> D_act
        self.w_down = nn.Linear(expert_dim, hidden_size, bias=False)  # D_act -> D
        # Per-expert fire bias. Warm-started ~0 so every expert fires early
        # (explore/specialize); a density loss ramps it up later to enforce
        # sparsity (roadmap step 2). This scalar is the only decision-dedicated
        # parameter — appending an expert adds just {this bias + the 4 matrices}.
        self.bias = nn.Parameter(torch.full((1,), 1e-6))

    def score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.a_gate(x)                       # (T, r)
        return z, torch.linalg.vector_norm(z, dim=-1)  # reuse z; score = ‖z‖₂

    def expert(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.b_gate(z))     # (n, D_act)
        return self.w_down(gate * self.w_up(x))  # (n, D)


class RFMoE(nn.Module):
    """Self-gating MoE FFN sublayer. Drop-in for a dense GLU FFN.

    ``theta`` is the inference-time compute knob: higher -> fewer experts clear
    the threshold -> fewer activations, lower FLOPs, some quality lost.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_dim: int,
        num_experts: int,
        rank: int | None = None,
        theta: float = 0.01,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        rank = rank or max(1, hidden_size // 16)  # paper sizing: r ≈ D/16
        self.experts = nn.ModuleList(
            RFMoEExpert(hidden_size, expert_dim, rank) for _ in range(num_experts)
        )
        self.theta = theta
        # Add x back inside forward (todo spec h = x + Σ G_i E_i). Set False when
        # an outer residual wrapper (e.g. AttnRes) already owns the residual.
        self.residual = residual
        self._last_density = 0.0   # mean fire fraction over (token, expert) pairs (float, for logging)
        # Differentiable density proxy: mean pre-threshold gate activity over ALL
        # tokens/experts. Minimizing it raises biases / shrinks projections ->
        # fewer fire. The adaptive-lambda controller closes the loop on real density.
        self._last_activity: torch.Tensor = torch.zeros(())
        # Per-expert differentiable usage (mean gate activity per expert) for the
        # locality/staircase loss, plus a detached EMA used only to RANK experts
        # (which one is hot) so the permutation is stable batch-to-batch while
        # gradients still flow through the current usage values.
        self._last_usage: torch.Tensor = torch.full((num_experts,), 1.0 / num_experts)
        self._last_gate: torch.Tensor = torch.zeros(num_experts, 1)  # (N, T) firing patterns, set in forward
        self.usage_decay = 0.99
        self.register_buffer("usage_ema", torch.full((num_experts,), 1.0 / num_experts), persistent=False)

    def add_expert(self, expert_dim: int | None = None, rank: int | None = None, bias_init: float = 3.0) -> RFMoEExpert:
        """Append capacity post-training (roadmap part 2).

        The new expert starts COLD (high ``bias_init``) so it rarely fires, lands
        in the offload tier, and doesn't perturb existing routing — the other
        experts' fire decisions never depended on the expert count.
        """
        ref = self.experts[0]
        expert_dim = expert_dim or ref.w_up.out_features
        rank = rank or ref.a_gate.out_features
        expert = RFMoEExpert(self.hidden_size, expert_dim, rank).to(ref.a_gate.weight.device)
        with torch.no_grad():
            expert.bias.fill_(bias_init)
        self.experts.append(expert)
        # Extend the ranking EMA; the cold newcomer starts at the current floor so
        # it ranks in the tail (consistent with the locality staircase).
        floor = self.usage_ema.min().item()
        self.usage_ema = torch.cat([self.usage_ema, self.usage_ema.new_full((1,), floor)])
        return expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)   # (T, D)
        out = torch.zeros_like(flat)
        fired = 0
        gate_rows = []                           # per-expert (T,) firing pattern (differentiable)
        # ponytail: per-expert Python loop with boolean-indexed skip path. Real
        # per-token FLOP saving, but O(num_experts) launches — fine for step-1
        # validation. Batch across experts (grouped GEMM / one padded matmul)
        # when this moves to a real training run.
        for expert in self.experts:
            z, s = expert.score(flat)
            gate_value = F.relu(s - expert.bias)      # (T,) 0 below the bias
            if self.training:
                gate_rows.append(gate_value)          # keep full pattern for diversity
            fire = gate_value >= self.theta           # (T,) skip-path mask
            if not bool(fire.any()):
                continue
            idx = fire.nonzero(as_tuple=True)[0]
            contrib = expert.expert(flat[idx], z[idx])            # only fired tokens run the FFN
            out.index_add_(0, idx, gate_value[idx, None] * contrib)  # score-weighted, NO divide-by-count
            fired += idx.numel()
        # Aux stats feed the density/locality/diversity losses — training only.
        # Inference (the theta compute knob) skips the (N, T) stack + EMA update.
        if self.training:
            num_experts = len(self.experts)
            gate_stack = torch.stack(gate_rows)          # (N, T) firing patterns
            self._last_gate = gate_stack                 # for the diversity loss
            usage = gate_stack.mean(dim=1)               # (N,) mean gate activity per expert
            self._last_usage = usage
            self._last_activity = usage.mean()           # step-2 density proxy = mean over experts
            self._last_density = fired / max(flat.size(0) * num_experts, 1)
            # Detached EMA for stable ranking (rare experts are noisy per batch).
            with torch.no_grad():
                self.usage_ema.mul_(self.usage_decay).add_(usage.detach(), alpha=1.0 - self.usage_decay)
        out = out.reshape(shape)
        return out + x if self.residual else out      # residual combine (or outer wrapper's job)


def iter_rfmoe(model: nn.Module):
    """Yield every RFMoE sublayer in a model."""
    for module in model.modules():
        if isinstance(module, RFMoE):
            yield module


def rfmoe_aux_activity(model: nn.Module) -> torch.Tensor:
    """Sum the differentiable gate-activity proxy across all RFMoE layers.

    One summed term + one lambda => only the GLOBAL average density is pinned;
    individual layers stay free (early layers may go sparse, late layers dense).
    Returns a 0-dim tensor (0.0 if the model has no RFMoE layers).
    """
    acts = [m._last_activity for m in iter_rfmoe(model)]
    if not acts:
        return torch.zeros(())
    return torch.stack(acts).sum()


def rfmoe_density(model: nn.Module) -> float:
    """Mean fire fraction across all RFMoE layers (global empirical density rho)."""
    densities = [m._last_density for m in iter_rfmoe(model)]
    return sum(densities) / len(densities) if densities else 0.0


def staircase_target(n: int, s: float = 1.0, alpha: float = 0.1,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """Tier-matched usage target: Zipf head + uniform tail (roadmap step 3).

    ``pi = (1-alpha)*Zipf(s) + alpha*Uniform(1/n)`` — a descending, sums-to-1
    distribution. The Zipf head gives a fine hot-set ordering for tier placement;
    the uniform component puts a floor ``>= alpha/n`` under every expert so the
    cold tail stays trained and alive. ``s`` sets head skew (hot-set size),
    ``alpha`` sets the tail floor height.
    """
    ranks = torch.arange(1, n + 1, device=device, dtype=dtype)
    zipf = ranks.pow(-s)
    zipf = zipf / zipf.sum()
    uniform = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    return (1.0 - alpha) * zipf + alpha * uniform


def rfmoe_locality_loss(model: nn.Module, s: float = 1.0, alpha: float = 0.1,
                        eps: float = 1e-8) -> torch.Tensor:
    """Sum of KL(pi || sorted usage) over RFMoE layers — shape usage to a staircase.

    ``p`` = current per-expert usage normalized to a distribution (differentiable).
    Experts are ranked by the detached usage EMA, so WHICH expert is hot emerges
    from data and is stable batch-to-batch, while gradients flow through the usage
    values. ``KL(pi || p)`` diverges as any ``p_i -> 0`` while ``pi_i > 0``, so it
    both concentrates mass on the head AND forbids any expert going fully dead.
    Orthogonal to the density controller: this shapes the relative distribution,
    the controller pins the absolute activation rate.
    """
    total = None
    for m in iter_rfmoe(model):
        usage = m._last_usage
        p = usage / usage.sum().clamp(min=eps)                  # (N,) distribution, differentiable
        order = torch.argsort(m.usage_ema, descending=True)     # detached permutation (stop-grad)
        p_sorted = p[order]
        pi = staircase_target(p.numel(), s, alpha, device=p.device, dtype=p.dtype)
        kl = (pi * (pi.clamp(min=eps).log() - p_sorted.clamp(min=eps).log())).sum()
        total = kl if total is None else total + kl
    return total if total is not None else torch.zeros(())


def rfmoe_diversity_loss(model: nn.Module, eps: float = 1e-8) -> torch.Tensor:
    """Sum over RFMoE layers of the mean positive pairwise firing-pattern overlap.

    Equal usage != equal function: two experts that fire on the SAME tokens claim
    the same niche and one is redundant cold weight (roadmap step 5, cf. R2MoE).
    Each expert's per-token gate pattern is its functional signature; this
    penalizes the positive off-diagonal correlations between those signatures so
    experts spread onto distinct token subsets. Anti-correlated experts (already
    distinct niches) are not penalized. Differentiable, reuses the gate patterns
    already computed in forward.
    """
    total = None
    for m in iter_rfmoe(model):
        g = m._last_gate                                   # (N, T) differentiable firing patterns
        n = g.size(0)
        if n < 2 or g.size(1) < 2:
            continue
        gc = g - g.mean(dim=1, keepdim=True)               # center each expert over tokens
        gn = gc / gc.norm(dim=1, keepdim=True).clamp(min=eps)
        corr = gn @ gn.t()                                 # (N, N) cosine of firing patterns
        off = corr - torch.diag(torch.diagonal(corr))      # zero the diagonal
        # ponytail: uniform over all pairs. Weight by inverse usage if the tail
        # needs diversity more than the (unavoidably-overlapping) hot set.
        overlap = off.clamp(min=0.0).pow(2).sum() / (n * (n - 1))  # only redundancy, not anti-corr
        total = overlap if total is None else total + overlap
    return total if total is not None else torch.zeros(())


class DensityController:
    """Adaptive-lambda controller: drive global activation density to a target.

    Multiplicative update ``lam <- lam * (1+eta)^sign(rho - target)`` — raise the
    penalty when the MoE fires too much, relax it when too sparse, so the training
    equilibrium (density penalty vs. the LM loss wanting useful experts on) sits at
    ``target``. Biases warm-start ~0 (all experts fire) and a small ``lam`` ramps
    up to carve out sparsity without collapsing any expert.

    Usage per optimizer step:
        loss = lm_loss + controller.lam * rfmoe_aux_activity(model)
        ...backward/step...
        controller.update(rfmoe_density(model))
    """

    def __init__(self, target: float = 0.25, eta: float = 0.01, lam: float = 1e-3,
                 lam_min: float = 1e-6, lam_max: float = 1e3) -> None:
        self.target = target
        self.eta = eta
        self.lam = lam
        self.lam_min = lam_min
        self.lam_max = lam_max

    def update(self, density: float) -> float:
        if density > self.target:
            self.lam *= (1.0 + self.eta)
        elif density < self.target:
            self.lam /= (1.0 + self.eta)
        self.lam = min(max(self.lam, self.lam_min), self.lam_max)
        return self.lam


if __name__ == "__main__":
    torch.manual_seed(0)
    hidden = 64
    moe = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
    x = torch.randn(4, 10, hidden)

    y = moe(x)
    assert y.shape == x.shape, y.shape

    # Grad flows through the self-gating + experts: one SGD step lowers the loss.
    target = torch.randn_like(y)
    opt = torch.optim.SGD(moe.parameters(), lr=0.1)
    loss0 = F.mse_loss(moe(x), target)
    opt.zero_grad(); loss0.backward(); opt.step()
    loss1 = F.mse_loss(moe(x), target)
    assert loss1 < loss0, (float(loss0), float(loss1))

    # theta knob: raising the threshold fires fewer experts.
    moe.theta = 0.01; moe(x); density_low = moe._last_density
    moe.theta = 5.0;  moe(x); density_high = moe._last_density
    assert density_high < density_low, (density_low, density_high)

    # Append a cold expert: forward stays valid, expert count grows.
    moe.theta = 0.01
    moe.add_expert()
    assert moe(x).shape == x.shape
    assert len(moe.experts) == 9

    print(f"RFMoE ok. density theta=0.01 -> {density_low:.3f}, theta=5 -> {density_high:.3f}, experts={len(moe.experts)}")

    # Controller update() sign logic: over target -> raise lambda, under -> lower.
    ctrl = DensityController(target=0.3, eta=0.1, lam=1.0)
    ctrl.update(0.9); assert ctrl.lam > 1.0, ctrl.lam   # too dense -> more penalty
    ctrl.update(0.1); ctrl.update(0.1); assert ctrl.lam < 1.0, ctrl.lam  # too sparse -> relax

    # Density is controllable by the penalty: lambda=0 stays dense, strong lambda
    # drives sparsity (the differentiable activity proxy raises biases / shrinks z).
    def train_density(lam: float, steps: int = 150) -> float:
        torch.manual_seed(1)
        moe2 = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
        opt2 = torch.optim.SGD(moe2.parameters(), lr=0.1)
        tgt = torch.randn(4, 10, hidden)
        for _ in range(steps):
            loss = F.mse_loss(moe2(x), tgt) + lam * rfmoe_aux_activity(moe2)
            opt2.zero_grad(); loss.backward(); opt2.step()
        return rfmoe_density(moe2)

    d_free = train_density(0.0)
    d_pen = train_density(2.0)
    assert d_free > 0.9, d_free            # no penalty -> stays dense
    assert d_pen < d_free, (d_free, d_pen)  # penalty carves out sparsity
    print(f"density control: lam=0 -> rho {d_free:.3f}, lam=2 -> rho {d_pen:.3f}")

    # Staircase target: descending, sums to 1, floored (no zero).
    pi = staircase_target(8, s=1.0, alpha=0.1)
    assert abs(float(pi.sum()) - 1.0) < 1e-5, float(pi.sum())
    assert torch.all(pi[:-1] >= pi[1:]), pi          # descending
    assert float(pi.min()) >= 0.1 / 8 - 1e-6, pi     # uniform floor alpha/n

    # Locality loss shapes usage toward the staircase: KL drops, tail stays alive.
    torch.manual_seed(2)
    moe3 = RFMoE(hidden, expert_dim=32, num_experts=8, rank=4, theta=0.01)
    opt3 = torch.optim.SGD(moe3.parameters(), lr=0.1)
    tgt3 = torch.randn(4, 10, hidden)
    moe3(x); kl_start = float(rfmoe_locality_loss(moe3, s=1.0, alpha=0.1).detach())
    for _ in range(200):
        loss = F.mse_loss(moe3(x), tgt3) + 0.5 * rfmoe_locality_loss(moe3, s=1.0, alpha=0.1)
        opt3.zero_grad(); loss.backward(); opt3.step()
    moe3(x); kl_end = float(rfmoe_locality_loss(moe3, s=1.0, alpha=0.1).detach())
    p_end = (moe3._last_usage / moe3._last_usage.sum()).detach().sort(descending=True).values
    assert kl_end < kl_start, (kl_start, kl_end)     # usage moved toward the staircase
    assert float(p_end.min()) > 0.0, p_end           # no expert went dead (KL floor works)
    print(f"locality: KL(pi||p) {kl_start:.3f} -> {kl_end:.3f}, "
          f"hot={float(p_end[0]):.3f} tail={float(p_end[-1]):.3f}")

    # Diversity loss math: identical firing patterns penalized, orthogonal not.
    moe4 = RFMoE(hidden, expert_dim=32, num_experts=3, rank=4, theta=0.01)
    moe4._last_gate = torch.tensor([[1.,0,1,0,1,0],[1,0,1,0,1,0],[1,0,1,0,1,0]])  # all identical
    same = float(rfmoe_diversity_loss(moe4))
    moe4._last_gate = torch.tensor([[1.,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])  # disjoint niches
    disjoint = float(rfmoe_diversity_loss(moe4))
    assert same > 0.5 and disjoint < same, (same, disjoint)

    # Training with the diversity term decorrelates expert firing patterns.
    torch.manual_seed(3)
    moe5 = RFMoE(hidden, expert_dim=32, num_experts=6, rank=4, theta=0.01)
    opt5 = torch.optim.SGD(moe5.parameters(), lr=0.2)
    tgt5 = torch.randn(4, 10, hidden)
    moe5(x); div_start = float(rfmoe_diversity_loss(moe5).detach())
    for _ in range(200):
        loss = F.mse_loss(moe5(x), tgt5) + 1.0 * rfmoe_diversity_loss(moe5)
        opt5.zero_grad(); loss.backward(); opt5.step()
    moe5(x); div_end = float(rfmoe_diversity_loss(moe5).detach())
    assert div_end < div_start, (div_start, div_end)
    print(f"diversity: identical={same:.3f} disjoint={disjoint:.3f}, overlap {div_start:.3f} -> {div_end:.3f}")
