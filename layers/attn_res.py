"""Kimi-style Attention Residuals (AttnRes) — Block AttnRes residual stream.

Paper: arXiv:2603.15031 (Moonshot / Kimi Team). Official Block AttnRes mix::

    V = stack(completed_blocks + [partial])   # [N+1, B, T, D]
    K = RMSNorm(V)
    logits[n,b,t] = w · K[n,b,t]              # pseudo-query w ∈ R^d
    h = softmax_n(logits) · V

Intra-block: standard sum of sublayer deltas into ``partial``.
Inter-block: depth softmax over completed block reps + current partial.

**Loops:** BitNetDeep resets the AttnRes stream at each Hyperloop boundary.
Cross-loop mixing is owned by ``LoopHyperConnection``; AttnRes only sees depth
within the current prelude / recurrent pass / coda segment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAttnMix(nn.Module):
    """Single pseudo-query depth attention over residual block states."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        # Paper: Linear(d, 1) without bias; squeeze weight → w ∈ R^d.
        self.proj = nn.Linear(hidden_size, 1, bias=False)
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        completed: List[torch.Tensor],
        partial: torch.Tensor,
    ) -> torch.Tensor:
        """Mix completed block states + partial into one residual vector."""
        if not completed:
            # Only partial (should not happen if stream seeds with embed).
            return partial
        # [N+1, B, T, D]
        v = torch.stack([*completed, partial], dim=0)
        k = self.norm(v)
        # logits: [N+1, B, T]
        w = self.proj.weight.view(-1)  # Linear(d→1) weight is [1, D] → [D]
        logits = torch.einsum("d,nbtd->nbt", w, k)
        weights = torch.softmax(logits, dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, v)


class SandwichResidual(nn.Module):
    """Legacy sandwich: ``post_norm(x + scale * sublayer_out)``.

    Kept for ``attn_res_mode="sandwich"`` ablations and old checkpoints.
    """

    def __init__(self, hidden_size: int, init_scale: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x_in: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return self.norm(x_in + self.scale * sublayer_out)


# Back-compat alias (historical name for sandwich).
AttentionResidual = SandwichResidual


@dataclass
class AttnResStream:
    """Mutable Block AttnRes bookkeeping for one depth segment.

    Attributes:
        completed: Finished depth-block residual sums (includes seed embed).
        partial: Running sum of sublayer outputs in the open depth-block.
        layers_in_block: Transformer layers closed in the open depth-block.
        group_size: Transformer layers per AttnRes depth-block.
        last_hidden: Last depth-mixed vector (for handoff / HC / lm_head).
    """

    completed: List[torch.Tensor]
    partial: Optional[torch.Tensor]
    layers_in_block: int
    group_size: int
    last_hidden: torch.Tensor
    attn_mix: DepthAttnMix
    mlp_mix: DepthAttnMix

    @classmethod
    def start(
        cls,
        seed: torch.Tensor,
        *,
        group_size: int,
        attn_mix: DepthAttnMix,
        mlp_mix: DepthAttnMix,
    ) -> "AttnResStream":
        """Begin a segment with token residual ``seed`` as the first completed block."""
        if group_size < 1:
            raise ValueError("attn_res_group_size must be >= 1")
        # Paper: blocks already include token embedding; partial starts as seed.
        return cls(
            completed=[seed],
            partial=seed,
            layers_in_block=0,
            group_size=int(group_size),
            last_hidden=seed,
            attn_mix=attn_mix,
            mlp_mix=mlp_mix,
        )

    def mix_attn(self) -> torch.Tensor:
        partial = self.partial if self.partial is not None else self.completed[-1]
        h = self.attn_mix(self.completed, partial)
        self.last_hidden = h
        return h

    def mix_mlp(self) -> torch.Tensor:
        partial = self.partial if self.partial is not None else self.completed[-1]
        h = self.mlp_mix(self.completed, partial)
        self.last_hidden = h
        return h

    def add_sublayer(self, delta: torch.Tensor) -> None:
        """Intra-block residual: accumulate sublayer output into partial."""
        if self.partial is None:
            self.partial = delta
        else:
            self.partial = self.partial + delta

    def close_layer(self) -> None:
        """Call after a full transformer layer (attn + mlp). May seal a depth-block."""
        self.layers_in_block += 1
        if self.layers_in_block < self.group_size:
            return
        if self.partial is None:
            raise RuntimeError("AttnResStream.close_layer with empty partial")
        self.completed.append(self.partial)
        self.partial = None
        self.layers_in_block = 0

    def hidden(self) -> torch.Tensor:
        """Best single residual vector for HC / coda handoff / final norm."""
        if self.partial is not None:
            return self.mlp_mix(self.completed, self.partial)
        return self.completed[-1]
