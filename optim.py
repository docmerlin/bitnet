"""C-MUD optimizer: cautious MomentUm Decorrelation with an 8-bit C-Lion fallback.

This implements the optimizer plan documented in the README:

- **C-MUD** for 2D matrix weights (the bulk of the model). MUD (MomentUm
  Decorrelation, Southworth & Thomas 2026) *decorrelates* heavy-ball momentum with a
  triangular whitening surrogate rather than Muon's polar / Newton-Schulz iteration:
  each pass row-normalizes the matrix, forms the row Gram ``G = Q Qᵀ``, takes its
  lower triangle ``T = tril(G)`` as a cheap Cholesky-like factor, applies a forward
  triangular solve ``Q <- T⁻¹ Q``, and re-normalizes. One pass (MUD1) is the default
  and costs a single ``k×k`` triangular solve (k = the smaller matrix dimension) —
  roughly 12x fewer FLOPs than Muon's repeated full matmuls. The ``C-`` prefix is the
  cautious-optimizer mask from *Cautious Optimizers: Improving Training with One Line
  of Code*, which zeroes any per-coordinate update whose sign disagrees with the
  current gradient and rescales the survivors to preserve the average step size.

- **8-bit C-Lion** for every other parameter (embeddings, norms, biases, scalar
  gates, AttnRes scales). This is the cautious variant of Lion; its single momentum
  buffer is stored as block-wise int8 to keep optimizer memory low. Lion's
  sign-based update is robust to the quantization noise, which is what makes 8-bit
  Lion practical.

The two paths live in one :class:`Optimizer` so they compose with ``LambdaLR`` and
``torch.cuda.amp.GradScaler`` exactly like a normal optimizer: each parameter group
carries a ``kind`` of ``"mud"`` or ``"clion"`` and :meth:`CMUD.step` dispatches on it.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

_QUANT_BLOCK_SIZE = 2048


def mud_decorrelate(update: torch.Tensor, passes: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """MUD triangular whitening of a momentum matrix (Algorithm 2 of the MUD paper).

    Decorrelates ``update`` toward a row-orthonormal matrix (``Q Qᵀ ≈ I_k`` along the
    smaller dimension ``k = min(n, m)``) using a lower-triangular Gram surrogate and a
    forward triangular solve instead of Muon's polar iteration. Each of ``passes``
    iterations:

    1. row-normalize ``Q = diag((r + ε)⁻¹) M``     (``r`` = per-row L2 norms),
    2. form the row Gram ``G = Q Qᵀ``,
    3. take its lower triangle ``T = tril(G)``,
    4. forward-solve ``Q = T⁻¹ Q``,
    5. row-normalize again from ``Q``'s recomputed norms.

    Work is done in fp32 on the ``k×d`` orientation (transposing tall matrices so the
    triangular solve is over the smaller dimension), then transposed back.
    """
    if update.ndim != 2:
        raise ValueError("mud_decorrelate expects a 2D matrix")
    if passes < 1:
        raise ValueError("passes must be >= 1")

    q = update.to(torch.float32)
    transposed = q.size(0) > q.size(1)
    if transposed:
        q = q.t()  # now k x d with k = min(n, m) <= d

    for _ in range(passes):
        row_norm = q.norm(dim=1, keepdim=True)
        q = q / (row_norm + eps)
        gram = q @ q.t()
        tri = torch.tril(gram)
        q = torch.linalg.solve_triangular(tri, q, upper=False)
        row_norm = q.norm(dim=1, keepdim=True)
        q = q / (row_norm + eps)

    if transposed:
        q = q.t()
    return q.to(update.dtype)


def cautious_mask(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Apply the cautious-optimizer mask to a proposed (pre-``lr``) update.

    Coordinates where the update and gradient share a sign are kept; the rest are
    zeroed. Survivors are rescaled by ``numel / kept`` so the mean step magnitude is
    preserved. The descent step is ``param -= lr * update``, so sign agreement means
    the step moves against the gradient.
    """
    mask = (update * grad > 0).to(update.dtype)
    scale = mask.numel() / mask.sum().clamp_min(1.0)
    return update.mul(mask).mul_(scale)


def _quantize_blockwise(tensor: torch.Tensor, block_size: int = _QUANT_BLOCK_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric block-wise int8 quantization with a per-block fp32 scale."""
    flat = tensor.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.clamp(torch.round(blocks / scale), -127, 127).to(torch.int8)
    return quantized, scale.squeeze(1)


def _dequantize_blockwise(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    blocks = quantized.to(torch.float32) * scale.unsqueeze(1)
    return blocks.reshape(-1)[: shape.numel()].view(shape)


def split_parameters_for_cmud(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Route parameters into ``(mud_params, fallback_params)``.

    2D weights take the MUD path, except embedding tables (which are handled by the
    C-Lion fallback, matching Muon recipes that keep embeddings on an Adam/Lion-style
    optimizer). Tied weights are de-duplicated by identity so a shared embedding /
    output projection is only optimized once.
    """
    embedding_param_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    }

    mud_params: List[nn.Parameter] = []
    fallback_params: List[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim == 2 and id(parameter) not in embedding_param_ids:
            mud_params.append(parameter)
        else:
            fallback_params.append(parameter)
    return mud_params, fallback_params


class CMUD(Optimizer):
    """Cautious MUD with an 8-bit Cautious-Lion fallback, dispatched per group.

    Each parameter group must set ``kind`` to ``"mud"`` or ``"clion"``. MUD groups use
    ``momentum``/``nesterov``/``passes``; C-Lion groups use ``betas`` and ``eight_bit``.
    ``cautious`` and ``weight_decay`` apply to both.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        *,
        momentum: float = 0.95,
        nesterov: bool = True,
        passes: int = 1,
        betas: Tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 0.0,
        cautious: bool = True,
        eight_bit: bool = True,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("C-MUD learning rate must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("C-MUD momentum must be in [0, 1)")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("C-Lion betas must be in [0, 1)")
        if passes < 1:
            raise ValueError("passes must be >= 1")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "passes": passes,
            "betas": betas,
            "weight_decay": weight_decay,
            "cautious": cautious,
            "eight_bit": eight_bit,
            "kind": "clion",
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("kind", "clion") == "mud":
                self._step_mud(group)
            else:
                self._step_clion(group)

        return loss

    def _step_mud(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        passes = group["passes"]
        weight_decay = group["weight_decay"]
        cautious = group["cautious"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("C-MUD does not support sparse gradients")
            if grad.ndim != 2:
                raise RuntimeError("MUD groups only accept 2D matrix parameters")

            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(grad)

            direction = grad.add(buffer, alpha=momentum) if nesterov else buffer
            update = mud_decorrelate(direction, passes=passes)
            # s(W) in the MUD weight update (Eq. 10) is left generic in the body
            # ("shape-dependent scaling"); the concrete 0.2 * sqrt(max(n, m)) is the
            # Appendix A code constant, from Liu et al. 2025 (Muon RMS-matching scale).
            update = update * (0.2 * (max(param.size(0), param.size(1)) ** 0.5))

            if cautious:
                update = cautious_mask(update, grad)
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update, alpha=-lr)

    def _step_clion(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        cautious = group["cautious"]
        eight_bit = group["eight_bit"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("C-Lion does not support sparse gradients")

            state = self.state[param]
            use_8bit = eight_bit and grad.numel() >= _QUANT_BLOCK_SIZE
            exp_avg = self._load_exp_avg(state, grad, use_8bit)

            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)

            # ``exp_avg.mul(beta1)`` allocates a fresh tensor, so the momentum buffer
            # is left intact for the beta2 EMA update below.
            update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
            if cautious:
                update = cautious_mask(update, grad)
            param.add_(update, alpha=-lr)

            exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
            self._store_exp_avg(state, exp_avg, use_8bit)

    @staticmethod
    def _load_exp_avg(state: dict, grad: torch.Tensor, use_8bit: bool) -> torch.Tensor:
        if use_8bit:
            if "exp_avg_q" not in state:
                return torch.zeros_like(grad)
            return _dequantize_blockwise(state["exp_avg_q"], state["exp_avg_scale"], grad.shape).to(
                device=grad.device, dtype=grad.dtype
            )
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
        return state["exp_avg"]

    @staticmethod
    def _store_exp_avg(state: dict, exp_avg: torch.Tensor, use_8bit: bool) -> None:
        if use_8bit:
            quantized, scale = _quantize_blockwise(exp_avg)
            state["exp_avg_q"] = quantized
            state["exp_avg_scale"] = scale
        else:
            state["exp_avg"] = exp_avg


def build_cmud(
    model: nn.Module,
    *,
    lr: float,
    fallback_lr: float,
    weight_decay: float,
    momentum: float = 0.95,
    passes: int = 1,
    betas: Tuple[float, float] = (0.95, 0.98),
    cautious: bool = True,
    eight_bit: bool = True,
) -> CMUD:
    """Build a :class:`CMUD` with MUD and C-Lion parameter groups for ``model``."""
    mud_params, fallback_params = split_parameters_for_cmud(model)
    groups = [
        {
            "params": mud_params,
            "kind": "mud",
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": fallback_params,
            "kind": "clion",
            "lr": fallback_lr,
            "weight_decay": 0.0,
        },
    ]
    return CMUD(
        groups,
        lr=lr,
        momentum=momentum,
        passes=passes,
        betas=betas,
        weight_decay=weight_decay,
        cautious=cautious,
        eight_bit=eight_bit,
    )
