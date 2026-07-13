"""Routing-free Mixture-of-Experts (RFMoE) FFN — self-gating experts.

Step 1 of the MoE roadmap in ``todo.md``. Each expert decides its OWN activation
from an internal score, so there is no centralized router: adding or removing an
expert never renormalizes the others (that is what makes the design extensible).

Ternary experts, grouped sparse execution, score-weighted residual combine, and
a single global threshold ``theta`` as the compute/quality knob. Aux losses
(density / locality / diversity) and expert-append helpers live as free functions.
Domain-specific append experiments remain deferred (see todo.md).

Reference: RFMoE (arXiv 2604.00801), building on AoE (2501.13074) / ReMoE.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.h_bitlinear import HBitLinear


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

    def __init__(self, hidden_size: int, expert_dim: int, rank: int, config) -> None:
        super().__init__()
        self.a_gate = HBitLinear(hidden_size, rank, bias=False, config=config)        # D -> r
        self.b_gate = HBitLinear(rank, expert_dim, bias=False, config=config)         # r -> D_act
        self.w_up = HBitLinear(hidden_size, expert_dim, bias=False, config=config)    # D -> D_act
        self.w_mid = HBitLinear(expert_dim, expert_dim, bias=False, config=config)    # D_act -> D_act
        self.w_down = HBitLinear(expert_dim, hidden_size, bias=False, config=config)  # D_act -> D
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
        config=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim
        rank = rank or max(1, hidden_size // 16)  # paper sizing: r ≈ D/16
        config = config or SimpleNamespace(use_hadamard=True, use_4bit_activations=True)
        self.rank = rank
        self.config = config
        self.experts = nn.ModuleList(
            RFMoEExpert(hidden_size, expert_dim, rank, config) for _ in range(num_experts)
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

    @staticmethod
    def _grouped_linear_input(x: torch.Tensor, layers: list[HBitLinear]) -> torch.Tensor:
        return layers[0].prepare_input(x)

    @staticmethod
    def _grouped_weight(layers: list[HBitLinear], dtype: torch.dtype) -> torch.Tensor:
        weights = torch.stack([layer.weight for layer in layers])
        return layers[0].effective_weight(dtype, weights)

    def add_expert(self, bias: float = 10.0) -> RFMoEExpert:
        """Append a cold expert without changing existing expert parameters or keys."""
        reference = self.experts[0]
        parameter = next(reference.parameters())
        expert = RFMoEExpert(self.hidden_size, self.expert_dim, self.rank, self.config).to(
            device=parameter.device, dtype=parameter.dtype
        )
        expert.train(self.training)
        with torch.no_grad():
            expert.bias.fill_(bias)
        for source, target in zip(reference.modules(), expert.modules()):
            if isinstance(source, HBitLinear) and isinstance(target, HBitLinear):
                target.set_quantization_state(
                    weight_mix=source.weight_quantization_mix,
                    activation_mix=source.activation_quantization_mix,
                    activation_bits=source.activation_bits,
                    enable_weight_quantization=source.enable_weight_quantization,
                    enable_activation_quantization=source.enable_activation_quantization,
                )
        self.experts.append(expert)
        device = parameter.device
        self._last_usage = torch.cat((self._last_usage.to(device), torch.zeros(1, device=device)))
        self._last_gate = torch.cat(
            (self._last_gate.to(device), torch.zeros(1, self._last_gate.size(1), device=device))
        )
        self.usage_ema = torch.cat((self.usage_ema, self.usage_ema.new_zeros(1)))
        return expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)   # (T, D)
        out = torch.zeros_like(flat)

        # Score every expert in one contraction, then pack only fired token/expert
        # pairs for batched expert-body GEMMs. ModuleList storage keeps checkpoint
        # keys stable while removing one launch sequence per expert.
        a_gates = [expert.a_gate for expert in self.experts]
        score_input = self._grouped_linear_input(flat, a_gates)
        a_gate = self._grouped_weight(a_gates, score_input.dtype)
        z = torch.einsum("td,nrd->ntr", score_input, a_gate)
        scores = torch.linalg.vector_norm(z, dim=-1)
        biases = torch.stack([expert.bias for expert in self.experts])
        gate_stack = F.relu(scores - biases)
        fire = gate_stack >= self.theta
        active = fire.nonzero()                       # (K, 2): expert, token

        if active.numel():
            counts = fire.sum(dim=1)
            max_count = int(counts.max())
            expert_idx, token_idx = active.unbind(dim=1)
            offsets = counts.cumsum(dim=0) - counts
            positions = torch.arange(active.size(0), device=x.device) - torch.repeat_interleave(
                offsets, counts, output_size=active.size(0)
            )

            padded_x = flat.new_zeros(len(self.experts), max_count, self.hidden_size)
            padded_z = z.new_zeros(len(self.experts), max_count, z.size(-1))
            padded_x[expert_idx, positions] = flat[token_idx]
            padded_z[expert_idx, positions] = z[expert_idx, token_idx]

            b_gates = [expert.b_gate for expert in self.experts]
            w_ups = [expert.w_up for expert in self.experts]
            w_mids = [expert.w_mid for expert in self.experts]
            w_downs = [expert.w_down for expert in self.experts]
            gate_input = self._grouped_linear_input(padded_z, b_gates)
            up_input = self._grouped_linear_input(padded_x, w_ups)
            b_gate = self._grouped_weight(b_gates, gate_input.dtype)
            w_up = self._grouped_weight(w_ups, up_input.dtype)
            gate = torch.sigmoid(torch.bmm(gate_input, b_gate.transpose(1, 2)))
            hidden = gate * torch.bmm(up_input, w_up.transpose(1, 2))
            mid_input = self._grouped_linear_input(hidden, w_mids)
            w_mid = self._grouped_weight(w_mids, mid_input.dtype)
            hidden = F.silu(torch.bmm(mid_input, w_mid.transpose(1, 2)))
            down_input = self._grouped_linear_input(hidden, w_downs)
            w_down = self._grouped_weight(w_downs, down_input.dtype)
            contrib = torch.bmm(down_input, w_down.transpose(1, 2))
            out.index_add_(
                0,
                token_idx,
                gate_stack[expert_idx, token_idx, None] * contrib[expert_idx, positions],
            )
        # Aux stats feed the density/locality/diversity losses — training only.
        # Inference (the theta compute knob) skips the (N, T) stack + EMA update.
        if self.training:
            num_experts = len(self.experts)
            self._last_gate = gate_stack                 # for the diversity loss
            usage = gate_stack.mean(dim=1)               # (N,) mean gate activity per expert
            self._last_usage = usage
            self._last_density = active.size(0) / max(flat.size(0) * num_experts, 1)
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


def add_rfmoe_experts(
    model: nn.Module,
    bias: float = 1e-6,
    freeze_existing: bool = True,
) -> list[RFMoEExpert]:
    """Append one expert per RFMoE layer, optionally training only new experts.

    Rebuild the optimizer after calling this function so it owns the appended
    parameters. Existing diversity loss supplies the niche-finding objective.
    """
    moes = list(iter_rfmoe(model))
    if not moes:
        return []
    if freeze_existing:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    added = [moe.add_expert(bias=bias) for moe in moes]
    if freeze_existing:
        for expert in added:
            expert.requires_grad_(True)
    config = getattr(model, "config", None)
    if config is not None:
        config.rfmoe_num_experts = len(moes[0].experts)
    return added


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
