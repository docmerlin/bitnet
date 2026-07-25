"""Pure-PyTorch Mamba-3-style selective SSM mixer.

Reference design follows the public Mamba-3 module API (Dao AI Lab / Goombalab):
heavy-tail data-dependent A, softplus Δt, trapezoidal mix, multi-head SSM, gated
output. This port is **training-friendly on CPU/MPS** via an explicit scan — not
the fused Triton/TileLang kernels from ``mamba_ssm``.

Used as a drop-in **attention replacement** inside ``HybridTransformerBlock``
(FFN / AttnRes / Engram unchanged).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig
from layers.h_bitlinear import HBitLinear


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """Positive heavy-tail map used for data-dependent A (Mamba-3).

    f(x) = 1 + x          if x >= 0
         = 1 / (1 - x)    if x < 0
    """
    neg = x.clamp_max(0.0)
    pos = x.clamp_min(0.0)
    return pos + torch.reciprocal((1.0 - neg).clamp(min=1e-4))


def is_mamba3_layer(layer_id: Optional[int], period: int = 3) -> bool:
    """True for layers 0, period, 2*period, ... (≈1/period of the stack)."""
    if layer_id is None or period < 1:
        return False
    return int(layer_id) % int(period) == 0


class Mamba3Mixer(nn.Module):
    """Selective multi-head SSM (Mamba-3-inspired SISO reference).

    Input/output: ``(batch, seq, d_model)``. Causal over the sequence axis.
    """

    def __init__(self, config: TernaryConfig) -> None:
        super().__init__()
        d_model = int(config.hidden_size)
        d_state = int(getattr(config, "mamba_d_state", 64))
        expand = int(getattr(config, "mamba_expand", 2))
        headdim = int(getattr(config, "mamba_headdim", 32))
        d_conv = int(getattr(config, "mamba_d_conv", 4))
        dt_min = float(getattr(config, "mamba_dt_min", 0.001))
        dt_max = float(getattr(config, "mamba_dt_max", 0.1))
        a_floor = float(getattr(config, "mamba_a_floor", 1e-4))

        d_inner = int(expand * d_model)
        if d_inner % headdim != 0:
            # Snap headdim down to a divisor of d_inner.
            for candidate in (headdim, 64, 32, 16, 8, 4):
                if candidate > 0 and d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 1
        nheads = d_inner // headdim

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.headdim = headdim
        self.nheads = nheads
        self.d_conv = d_conv
        self.a_floor = a_floor
        self.rms_eps = float(config.rms_norm_eps)

        # z, x, B, C, dd_dt, dd_A, trap
        d_in = 2 * d_inner + 2 * d_state + 3 * nheads
        self.in_proj = HBitLinear(d_model, d_in, bias=False, config=config)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        # dt bias in softplus-inverse space (Mamba-style).
        dt = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.B_bias = nn.Parameter(torch.ones(nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(nheads, d_state))
        self.D = nn.Parameter(torch.ones(nheads))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = HBitLinear(d_inner, d_model, bias=False, config=config)

    def _norm_last(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm over the last dim (B/C channels)."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.rms_eps)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: ``(batch, seq, d_model)``
        Returns:
            Same shape as ``u``.
        """
        batch, seq_len, _ = u.shape
        projected = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.d_state,
                self.d_state,
                self.nheads,
                self.nheads,
                self.nheads,
            ],
            dim=-1,
        )

        # Depthwise causal conv on x (Mamba block convention).
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = F.silu(x_conv)

        # Heads: (B, L, H, P)
        x = x.view(batch, seq_len, self.nheads, self.headdim)
        z = z.view(batch, seq_len, self.nheads, self.headdim)

        # Data-dependent A (negative) and Δt > 0.
        a = -heavy_tail_activation(dd_A.float())
        a = a.clamp(max=-self.a_floor)
        dt = F.softplus(dd_dt.float() + self.dt_bias)  # (B, L, H)
        adt = a * dt  # log of decay scale (negative)
        decay = torch.exp(adt)  # (B, L, H) in (0, 1)

        # B, C: (B, L, H, N) with shared input projected to state, broadcast over heads.
        B = self._norm_last(B.float()).unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C = self._norm_last(C.float()).unsqueeze(2).expand(-1, -1, self.nheads, -1)
        B = B + self.B_bias.view(1, 1, self.nheads, self.d_state)
        C = C + self.C_bias.view(1, 1, self.nheads, self.d_state)

        trap = torch.sigmoid(trap.float())  # (B, L, H) ∈ (0, 1)

        # Selective scan with exponential-trapezoidal input mix (Mamba-3 style).
        # State h: (B, H, P, N)
        y = self._selective_scan(
            x.float(),
            decay,
            dt,
            B,
            C,
            trap,
        )
        y = y.to(dtype=u.dtype)
        # Skip + gate
        y = y + self.D.view(1, 1, self.nheads, 1).to(dtype=y.dtype) * x
        y = y * F.silu(z)
        y = y.reshape(batch, seq_len, self.d_inner)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        decay: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        trap: torch.Tensor,
    ) -> torch.Tensor:
        """Causal SSM scan.

        Args:
            x: (B, L, H, P)
            decay: (B, L, H)  — exp(A Δt)
            dt: (B, L, H)
            B, C: (B, L, H, N)
            trap: (B, L, H) trapezoidal weight toward current input
        Returns:
            y: (B, L, H, P)
        """
        batch, seq_len, nheads, headdim = x.shape
        device = x.device
        dtype = x.dtype

        # h[b,h,p,n]
        state = torch.zeros(
            batch, nheads, headdim, self.d_state, device=device, dtype=dtype
        )
        # Previous Bx term for trapezoid: (B, H, P, N)
        prev_bu = torch.zeros_like(state)
        outputs = []

        for t in range(seq_len):
            dec_t = decay[:, t]  # (B, H)
            dt_t = dt[:, t]
            trap_t = trap[:, t]
            b_t = B[:, t]  # (B, H, N)
            c_t = C[:, t]
            x_t = x[:, t]  # (B, H, P)

            # B⊗x → (B, H, P, N); Δt scales the input injection.
            bu = b_t.unsqueeze(2) * x_t.unsqueeze(-1) * dt_t.view(batch, nheads, 1, 1)
            # Trapezoidal blend of previous and current forcing terms.
            trap_v = trap_t.view(batch, nheads, 1, 1)
            force = (1.0 - trap_v) * prev_bu + trap_v * bu

            dec_v = dec_t.view(batch, nheads, 1, 1)
            state = dec_v * state + force
            # y = C · h over state dim
            y_t = (state * c_t.unsqueeze(2)).sum(dim=-1)  # (B, H, P)
            outputs.append(y_t)
            prev_bu = bu

        return torch.stack(outputs, dim=1)
