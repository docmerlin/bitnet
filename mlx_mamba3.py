"""MLX Mamba-3-style selective SSM mixer (parity with ``layers/mamba3.py``).

Explicit causal scan for train/eval on Apple GPU — no CUDA ``mamba_ssm`` kernels.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from layers.mamba3 import is_mamba3_layer  # noqa: F401 — re-export schedule helper


def heavy_tail_activation(x: mx.array) -> mx.array:
    """Positive heavy-tail map for data-dependent A (Mamba-3)."""
    neg = mx.minimum(x, 0.0)
    pos = mx.maximum(x, 0.0)
    return pos + mx.reciprocal(mx.maximum(1.0 - neg, 1e-4))


class MLXMamba3Mixer(nn.Module):
    """Selective multi-head SSM. I/O ``(batch, seq, d_model)``."""

    def __init__(self, config):
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
            for candidate in (headdim, 64, 32, 16, 8, 4):
                if candidate > 0 and d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 1
        nheads = d_inner // headdim

        self.config = config
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.headdim = headdim
        self.nheads = nheads
        self.d_conv = d_conv
        self.a_floor = a_floor
        self.rms_eps = float(config.rms_norm_eps)

        from mlx_model import MLXHBitLinear

        d_in = 2 * d_inner + 2 * d_state + 3 * nheads
        self.in_proj = MLXHBitLinear(d_model, d_in, config)
        # Depthwise causal conv (channels-last after transpose in forward).
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        # softplus-inverse init for dt bias
        u = mx.random.uniform(shape=(nheads,))
        dt = mx.exp(u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = mx.maximum(dt, 1e-4)
        # inv_softplus(dt) = dt + log(-expm1(-dt)) ≈ log(exp(dt)-1) for large; use stable form
        inv_dt = dt + mx.log(-mx.expm1(-dt))
        self.dt_bias = inv_dt
        self.B_bias = mx.ones((nheads, d_state))
        self.C_bias = mx.ones((nheads, d_state))
        self.D = mx.ones((nheads,))
        self.out_proj = MLXHBitLinear(d_inner, d_model, config)
        self.out_proj.weight = self.out_proj.weight * 0.01

    @staticmethod
    def _norm_last(x: mx.array, eps: float) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + eps)

    def _softplus(self, x: mx.array) -> mx.array:
        return mx.logaddexp(x, mx.zeros_like(x))

    def forward_arrays(self, u: mx.array) -> mx.array:
        batch, seq_len, _ = u.shape
        projected = self.in_proj(u)
        # Manual split along last dim
        sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state,
            self.d_state,
            self.nheads,
            self.nheads,
            self.nheads,
        ]
        parts = []
        offset = 0
        for size in sizes:
            parts.append(projected[..., offset : offset + size])
            offset += size
        z, x, B, C, dd_dt, dd_A, trap = parts

        # Conv1d expects (N, L, C) in MLX nn.Conv1d
        x_conv = self.conv1d(x)
        if x_conv.shape[1] > seq_len:
            x_conv = x_conv[:, :seq_len, :]
        x = nn.silu(x_conv)

        x = x.reshape(batch, seq_len, self.nheads, self.headdim)
        z = z.reshape(batch, seq_len, self.nheads, self.headdim)

        a = -heavy_tail_activation(dd_A.astype(mx.float32))
        a = mx.minimum(a, -self.a_floor)
        dt = self._softplus(dd_dt.astype(mx.float32) + self.dt_bias)
        decay = mx.exp(a * dt)

        B = self._norm_last(B.astype(mx.float32), self.rms_eps)
        C = self._norm_last(C.astype(mx.float32), self.rms_eps)
        # (B, L, 1, N) → broadcast heads
        B = mx.broadcast_to(B[:, :, None, :], (batch, seq_len, self.nheads, self.d_state))
        C = mx.broadcast_to(C[:, :, None, :], (batch, seq_len, self.nheads, self.d_state))
        B = B + self.B_bias.reshape(1, 1, self.nheads, self.d_state)
        C = C + self.C_bias.reshape(1, 1, self.nheads, self.d_state)
        trap = mx.sigmoid(trap.astype(mx.float32))

        y = self._selective_scan(
            x.astype(mx.float32),
            decay,
            dt,
            B,
            C,
            trap,
            use_kernel=bool(getattr(self.config, "use_mamba_scan_kernel", True)),
        )
        y = y.astype(u.dtype)
        y = y + self.D.reshape(1, 1, self.nheads, 1).astype(y.dtype) * x.astype(y.dtype)
        y = y * nn.silu(z)
        y = y.reshape(batch, seq_len, self.d_inner)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: mx.array,
        decay: mx.array,
        dt: mx.array,
        B: mx.array,
        C: mx.array,
        trap: mx.array,
        *,
        use_kernel: bool = True,
    ) -> mx.array:
        if use_kernel and self.d_state <= 128:
            try:
                from mlx_mamba_scan_kernel import selective_scan

                return selective_scan(x, decay, dt, B, C, trap)
            except Exception:
                pass
        from mlx_mamba_scan_kernel import selective_scan_reference

        return selective_scan_reference(x, decay, dt, B, C, trap)

    def __call__(self, u: mx.array) -> mx.array:
        return self.forward_arrays(u)

    def prefill(self, u: mx.array, cache: "MLXMamba3InferenceCache") -> mx.array:
        """Full-sequence prefill; store final SSM state for decode."""
        y = self.forward_arrays(u)
        # Cache last-token state by one extra step bookkeeping is heavy; recompute on
        # incremental from scratch for short contexts via stored last inputs.
        cache.last_hidden = u
        cache.ready = True
        return y

    def extend(self, u: mx.array, cache: "MLXMamba3InferenceCache") -> mx.array:
        if cache.last_hidden is None:
            return self.prefill(u, cache)
        combined = mx.concatenate([cache.last_hidden, u], axis=1)
        y = self.forward_arrays(combined)
        cache.last_hidden = combined
        return y[:, -u.shape[1] :, :]

    def incremental(self, u: mx.array, cache: "MLXMamba3InferenceCache") -> mx.array:
        """Single-token (or short) step: recompute over cached prefix + new tokens."""
        return self.extend(u, cache)


class MLXMamba3InferenceCache:
    """Lightweight cache: keep prefix activations for recompute-based decode."""

    def __init__(self):
        self.last_hidden: mx.array | None = None
        self.ready: bool = False

    def clone(self) -> "MLXMamba3InferenceCache":
        out = MLXMamba3InferenceCache()
        out.last_hidden = self.last_hidden
        out.ready = self.ready
        return out
