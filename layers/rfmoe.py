"""Routing-free Mixture-of-Experts (RFMoE) FFN — self-gating experts.

Step 1 of the MoE roadmap in ``todo.md``. Each expert decides its OWN activation
from an internal score, so there is no centralized router: adding or removing an
expert never renormalizes the others (that is what makes the design extensible).

Dense float experts, per-expert skip path, score-weighted residual combine,
and a single global threshold ``theta`` as the compute/quality knob. Aux losses
(density / locality / diversity) live as free functions. Deferred: ternary
experts, grouped GEMM, extensible ``add_expert`` (see todo.md).

Reference: RFMoE (arXiv 2604.00801), building on AoE (2501.13074) / ReMoE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFMoEExpert(nn.Module):
    """One self-gating GLU expert.

    ``FFN(x) = W_down(silu(W_mid([sigmoid(x A_gate B_gate) ⊙ (x W_up)])))``

    Body depth matches dense hybrid (up → mid → down), but the expand gate stays
    paper-style sigmoid GLU — not SwiGLU (dense hybrid / BLT use silu(gate)*up).
    The rank-r projection ``z = x A_gate`` is computed ONCE: its L2 norm is the
    self-score that makes the fire decision, and the same ``z`` feeds the gate
    branch ``B_gate``. So a token that doesn't fire pays only the cheap D×r
    projection and skips the expert body — that is the FLOP saving.
    """

    def __init__(self, hidden_size: int, expert_dim: int, rank: int) -> None:
        super().__init__()
        self.a_gate = nn.Linear(hidden_size, rank, bias=False)        # D -> r  (score + gate, dual use)
        self.b_gate = nn.Linear(rank, expert_dim, bias=False)         # r -> D_act
        self.w_up = nn.Linear(hidden_size, expert_dim, bias=False)    # D -> D_act
        self.w_mid = nn.Linear(expert_dim, expert_dim, bias=False)    # D_act -> D_act
        self.w_down = nn.Linear(expert_dim, hidden_size, bias=False)  # D_act -> D
        # Per-expert fire bias. Warm-started ~0 so every expert fires early
        # (explore/specialize); a density loss ramps it up later to enforce
        # sparsity. This scalar is the only decision-dedicated parameter.
        self.bias = nn.Parameter(torch.full((1,), 1e-6))

    def score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.a_gate(x)                       # (T, r)
        return z, torch.linalg.vector_norm(z, dim=-1)  # reuse z; score = ‖z‖₂

    def expert(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.b_gate(z))     # (n, D_act)
        hidden = gate * self.w_up(x)
        return self.w_down(F.silu(self.w_mid(hidden)))


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
        # Per-expert differentiable usage (mean gate activity per expert) for the
        # locality/staircase loss, plus a detached EMA used only to RANK experts
        # (which one is hot) so the permutation is stable batch-to-batch while
        # gradients still flow through the current usage values.
        self._last_usage: torch.Tensor = torch.full((num_experts,), 1.0 / num_experts)
        self._last_gate: torch.Tensor = torch.zeros(num_experts, 1)  # (N, T) firing patterns, set in forward
        self.usage_decay = 0.99
        self.register_buffer("usage_ema", torch.full((num_experts,), 1.0 / num_experts), persistent=False)

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
    acts = [m._last_usage.mean() for m in iter_rfmoe(model)]
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
