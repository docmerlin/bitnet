"""Fused Metal selective scan for MLX Mamba-3 mixer (forward + backward).

Forward: one thread per (batch, head, headdim), state in registers.
Backward:
  1) Metal recompute writes state history (B,L,H,P,N)
  2) Metal reverse: one threadgroup per (batch, head) with P threads;
     no atomics — reduce dB/dC/ddecay/ddt/dtrap in threadgroup memory.
"""

from __future__ import annotations

import mlx.core as mx


_SCAN_FWD = mx.fast.metal_kernel(
    name="mamba3_selective_scan_fwd",
    input_names=["x", "decay", "dt", "Bmat", "Cmat", "trap"],
    output_names=["y"],
    source=r"""
        uint p = thread_position_in_grid.x;
        uint h = thread_position_in_grid.y;
        uint b = thread_position_in_grid.z;

        uint L = x_shape[1];
        uint H = x_shape[2];
        uint P = x_shape[3];
        uint N = Bmat_shape[3];
        if (p >= P || h >= H || b >= x_shape[0]) return;

        float state[NMAX];
        float prev_bu[NMAX];
        for (uint n = 0; n < NMAX; ++n) {
            state[n] = 0.0f;
            prev_bu[n] = 0.0f;
        }

        for (uint t = 0; t < L; ++t) {
            ulong th = ulong(t) * H + h;
            float dec = float(decay[ulong(b) * L * H + th]);
            float dtt = float(dt[ulong(b) * L * H + th]);
            float tr = float(trap[ulong(b) * L * H + th]);
            float xv = float(x[ulong(b) * L * H * P + ulong(t) * H * P + ulong(h) * P + p]);

            float yv = 0.0f;
            for (uint n = 0; n < NMAX; ++n) {
                float Bv = float(Bmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n]);
                float Cv = float(Cmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n]);
                float bu = Bv * xv * dtt;
                float force = (1.0f - tr) * prev_bu[n] + tr * bu;
                float st = dec * state[n] + force;
                state[n] = st;
                prev_bu[n] = bu;
                yv += st * Cv;
            }
            y[ulong(b) * L * H * P + ulong(t) * H * P + ulong(h) * P + p] = T(yv);
        }
    """,
)


_SCAN_RECOMPUTE = mx.fast.metal_kernel(
    name="mamba3_selective_scan_recompute",
    input_names=["x", "decay", "dt", "Bmat", "trap"],
    output_names=["states"],
    source=r"""
        uint p = thread_position_in_grid.x;
        uint h = thread_position_in_grid.y;
        uint b = thread_position_in_grid.z;

        uint L = x_shape[1];
        uint H = x_shape[2];
        uint P = x_shape[3];
        uint N = Bmat_shape[3];
        if (p >= P || h >= H || b >= x_shape[0]) return;

        float state[NMAX];
        float prev_bu[NMAX];
        for (uint n = 0; n < NMAX; ++n) {
            state[n] = 0.0f;
            prev_bu[n] = 0.0f;
        }

        ulong bStride = ulong(L) * H * P * N;
        for (uint t = 0; t < L; ++t) {
            ulong th = ulong(t) * H + h;
            float dec = float(decay[ulong(b) * L * H + th]);
            float dtt = float(dt[ulong(b) * L * H + th]);
            float tr = float(trap[ulong(b) * L * H + th]);
            float xv = float(x[ulong(b) * L * H * P + ulong(t) * H * P + ulong(h) * P + p]);

            for (uint n = 0; n < NMAX; ++n) {
                float Bv = float(Bmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n]);
                float bu = Bv * xv * dtt;
                float force = (1.0f - tr) * prev_bu[n] + tr * bu;
                float st = dec * state[n] + force;
                state[n] = st;
                prev_bu[n] = bu;
                states[ulong(b) * bStride + ulong(t) * H * P * N + ulong(h) * P * N + ulong(p) * N + n] = T(st);
            }
        }
    """,
)


# Threadgroup (P,1,1) per (h,b). Shared reduce for scalars + dB/dC strips.
# sh_dB/sh_dC: [P][N] with P<=64, N<=64 → 16KB each if N=64; use N<=64 path in shared.
#
# Every cross-thread reduction here is split across the P threads. The obvious
# way to write it -- park the sums on thread 0 -- makes the dB/dC reduction
# 2*N*P serial float ops per timestep with P-1 threads stalled at a barrier,
# which at L=1024, N=64, P=32 is 4096 serial ops x 1024 steps and dominated the
# whole training backward (measured: the scan was 77% of it). Striping n over
# the threads makes it 2*N ops each.
_SCAN_BWD = mx.fast.metal_kernel(
    name="mamba3_selective_scan_bwd",
    input_names=["x", "decay", "dt", "Bmat", "Cmat", "trap", "dy", "states"],
    output_names=["dx", "ddecay", "ddt", "dBmat", "dCmat", "dtrap"],
    source=r"""
        uint p = thread_position_in_threadgroup.x;
        uint h = thread_position_in_grid.y;
        uint b = thread_position_in_grid.z;

        uint L = x_shape[1];
        uint H = x_shape[2];
        uint P = x_shape[3];
        uint N = Bmat_shape[3];
        if (h >= H || b >= x_shape[0] || p >= P) return;

        float dstate[NMAX];
        float dprev_bu[NMAX];
        for (uint n = 0; n < NMAX; ++n) {
            dstate[n] = 0.0f;
            dprev_bu[n] = 0.0f;
        }

        // One shared array per scalar so the three reduce in parallel on
        // threads 0/1/2 behind a single barrier pair.
        threadgroup float sh_dec[64];
        threadgroup float sh_dt[64];
        threadgroup float sh_tr[64];
        // dB/dC: thread p writes its strip to sh_dBN[p*N + n], then the threads
        // split the column sums between them. Host side rejects P*N > 2048.
        threadgroup float sh_dBN[2048];
        threadgroup float sh_dCN[2048];

        ulong histStride = ulong(L) * H * P * N;

        for (int ti = int(L) - 1; ti >= 0; --ti) {
            uint t = uint(ti);
            ulong th = ulong(t) * H + h;
            float dec = float(decay[ulong(b) * L * H + th]);
            float dtt = float(dt[ulong(b) * L * H + th]);
            float tr = float(trap[ulong(b) * L * H + th]);
            float xv = float(x[ulong(b) * L * H * P + ulong(t) * H * P + ulong(h) * P + p]);
            float dyv = float(dy[ulong(b) * L * H * P + ulong(t) * H * P + ulong(h) * P + p]);

            float d_dec_local = 0.0f;
            float d_dt_local = 0.0f;
            float d_tr_local = 0.0f;
            float d_x_local = 0.0f;

            for (uint n = 0; n < NMAX; ++n) {
                ulong hist = ulong(b) * histStride
                    + ulong(t) * H * P * N + ulong(h) * P * N + ulong(p) * N + n;
                float st = float(states[hist]);
                float stm1 = 0.0f;
                if (t > 0) {
                    ulong hist_m1 = ulong(b) * histStride
                        + ulong(t - 1) * H * P * N + ulong(h) * P * N + ulong(p) * N + n;
                    stm1 = float(states[hist_m1]);
                }
                float Bv = float(Bmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n]);
                float Cv = float(Cmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n]);
                float bu = Bv * xv * dtt;
                float bum1 = 0.0f;
                if (t > 0) {
                    float xv_m = float(x[ulong(b) * L * H * P + ulong(t - 1) * H * P + ulong(h) * P + p]);
                    float dtt_m = float(dt[ulong(b) * L * H + ulong(t - 1) * H + h]);
                    float Bv_m = float(Bmat[ulong(b) * L * H * N + ulong(t - 1) * H * N + ulong(h) * N + n]);
                    bum1 = Bv_m * xv_m * dtt_m;
                }

                dstate[n] += dyv * Cv;
                float dC = dyv * st;
                float d_force = dstate[n];
                d_dec_local += d_force * stm1;
                float d_bu = d_force * tr + dprev_bu[n];
                float d_bum1 = d_force * (1.0f - tr);
                d_tr_local += d_force * (bu - bum1);
                d_x_local += d_bu * Bv * dtt;
                float dB = d_bu * xv * dtt;
                d_dt_local += d_bu * Bv * xv;
                dstate[n] = d_force * dec;
                dprev_bu[n] = d_bum1;

                sh_dBN[p * N + n] = dB;
                sh_dCN[p * N + n] = dC;
            }

            dx[ulong(b) * L * H * P + ulong(t) * H * P + ulong(h) * P + p] = T(d_x_local);

            sh_dec[p] = d_dec_local;
            sh_dt[p] = d_dt_local;
            sh_tr[p] = d_tr_local;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Striped, not one slot per thread: the threadgroup is P threads
            // wide and P can be smaller than 3, in which case a fixed
            // p==0/1/2 assignment leaves ddt and dtrap never written and the
            // output buffer is returned uninitialised. Measured at headdim 2:
            // dtrap error 7.7e-03 against 7.5e-09 at headdim 32.
            for (uint slot = p; slot < 3; slot += P) {
                float s = 0.0f;
                if (slot == 0) {
                    for (uint i = 0; i < P; ++i) s += sh_dec[i];
                    ddecay[ulong(b) * L * H + th] = T(s);
                } else if (slot == 1) {
                    for (uint i = 0; i < P; ++i) s += sh_dt[i];
                    ddt[ulong(b) * L * H + th] = T(s);
                } else {
                    for (uint i = 0; i < P; ++i) s += sh_tr[i];
                    dtrap[ulong(b) * L * H + th] = T(s);
                }
            }

            // Column sums of sh_dBN/sh_dCN, n striped over the P threads.
            for (uint n = p; n < N; n += P) {
                float sB = 0.0f, sC = 0.0f;
                for (uint i = 0; i < P; ++i) {
                    sB += sh_dBN[i * N + n];
                    sC += sh_dCN[i * N + n];
                }
                dBmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n] = T(sB);
                dCmat[ulong(b) * L * H * N + ulong(t) * H * N + ulong(h) * N + n] = T(sC);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
)


def _f32(*arrays: mx.array) -> tuple[mx.array, ...]:
    return tuple(a.astype(mx.float32) for a in arrays)


def selective_scan_fwd_metal(
    x: mx.array,
    decay: mx.array,
    dt: mx.array,
    Bmat: mx.array,
    Cmat: mx.array,
    trap: mx.array,
) -> mx.array:
    x, decay, dt, Bmat, Cmat, trap = _f32(x, decay, dt, Bmat, Cmat, trap)
    bsz, _l, nheads, headdim = x.shape
    if Bmat.shape[-1] > 128:
        raise ValueError("Metal scan supports d_state <= 128")
    if headdim > 64:
        raise ValueError("Metal bwd threadgroup expects headdim (P) <= 64")
    return _SCAN_FWD(
        inputs=[x, decay, dt, Bmat, Cmat, trap],
        # d_state as a template constant, not a loop bound read from the shape:
        # the per-thread state/prev_bu arrays then size to the real d_state
        # instead of the 128 worst case, which is the difference between fitting
        # in registers and spilling.
        template=[("T", mx.float32), ("NMAX", int(Bmat.shape[-1]))],
        grid=(headdim, nheads, bsz),
        threadgroup=(min(headdim, 32), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]


def selective_scan_bwd_metal(
    x: mx.array,
    decay: mx.array,
    dt: mx.array,
    Bmat: mx.array,
    Cmat: mx.array,
    trap: mx.array,
    dy: mx.array,
) -> tuple[mx.array, ...]:
    x, decay, dt, Bmat, Cmat, trap, dy = _f32(x, decay, dt, Bmat, Cmat, trap, dy)
    bsz, seq_len, nheads, headdim = x.shape
    n_state = Bmat.shape[-1]
    if n_state > 128:
        raise ValueError("Metal scan supports d_state <= 128")
    if headdim > 64:
        raise ValueError("Metal bwd threadgroup expects headdim (P) <= 64")
    if headdim * n_state > 2048:
        # sh_dBN/sh_dCN are [P][N]; past this the kernel would read uninitialised
        # threadgroup memory and hand back silently wrong dB/dC.
        raise ValueError("Metal scan bwd needs headdim * d_state <= 2048")

    hist_shape = (bsz, seq_len, nheads, headdim, n_state)
    (states,) = _SCAN_RECOMPUTE(
        inputs=[x, decay, dt, Bmat, trap],
        template=[("T", mx.float32), ("NMAX", n_state)],
        grid=(headdim, nheads, bsz),
        threadgroup=(min(headdim, 32), 1, 1),
        output_shapes=[hist_shape],
        output_dtypes=[mx.float32],
    )

    # One threadgroup per (h,b) with P threads for reduction.
    outs = _SCAN_BWD(
        inputs=[x, decay, dt, Bmat, Cmat, trap, dy, states],
        template=[("T", mx.float32), ("NMAX", n_state)],
        grid=(headdim, nheads, bsz),
        threadgroup=(headdim, 1, 1),
        output_shapes=[
            x.shape,
            decay.shape,
            dt.shape,
            Bmat.shape,
            Cmat.shape,
            trap.shape,
        ],
        output_dtypes=[mx.float32] * 6,
    )
    return outs


def selective_scan_reference(
    x: mx.array,
    decay: mx.array,
    dt: mx.array,
    Bmat: mx.array,
    Cmat: mx.array,
    trap: mx.array,
) -> mx.array:
    x, decay, dt, Bmat, Cmat, trap = _f32(x, decay, dt, Bmat, Cmat, trap)
    batch, seq_len, nheads, headdim = x.shape
    n_state = Bmat.shape[-1]
    state = mx.zeros((batch, nheads, headdim, n_state), dtype=mx.float32)
    prev_bu = mx.zeros_like(state)
    outputs = []
    for t in range(seq_len):
        bu = (
            Bmat[:, t][:, :, None, :]
            * x[:, t][:, :, :, None]
            * dt[:, t].reshape(batch, nheads, 1, 1)
        )
        trap_v = trap[:, t].reshape(batch, nheads, 1, 1)
        force = (1.0 - trap_v) * prev_bu + trap_v * bu
        dec_v = decay[:, t].reshape(batch, nheads, 1, 1)
        state = dec_v * state + force
        y_t = mx.sum(state * Cmat[:, t][:, :, None, :], axis=-1)
        outputs.append(y_t)
        prev_bu = bu
    return mx.stack(outputs, axis=1)


def _scan_vjp_mlx(primals, dy):
    x, decay, dt, Bmat, Cmat, trap = [a.astype(mx.float32) for a in primals]
    dy = dy.astype(mx.float32)
    batch, seq_len, nheads, headdim = x.shape
    n_state = Bmat.shape[-1]

    state = mx.zeros((batch, nheads, headdim, n_state), dtype=mx.float32)
    prev_bu = mx.zeros_like(state)
    states, bus, prev_bus = [], [], []
    for t in range(seq_len):
        prev_bus.append(prev_bu)
        bu = (
            Bmat[:, t][:, :, None, :]
            * x[:, t][:, :, :, None]
            * dt[:, t].reshape(batch, nheads, 1, 1)
        )
        trap_v = trap[:, t].reshape(batch, nheads, 1, 1)
        force = (1.0 - trap_v) * prev_bu + trap_v * bu
        dec_v = decay[:, t].reshape(batch, nheads, 1, 1)
        state = dec_v * state + force
        states.append(state)
        bus.append(bu)
        prev_bu = bu

    dx_l, ddecay_l, ddt_l, dB_l, dC_l, dtrap_l = [], [], [], [], [], []
    dstate = mx.zeros((batch, nheads, headdim, n_state), dtype=mx.float32)
    dprev_bu = mx.zeros_like(dstate)
    for t in range(seq_len - 1, -1, -1):
        st = states[t]
        bu = bus[t]
        bu_tm1 = prev_bus[t]
        state_tm1 = states[t - 1] if t > 0 else mx.zeros_like(st)
        tr = trap[:, t].reshape(batch, nheads, 1, 1)
        dec = decay[:, t].reshape(batch, nheads, 1, 1)
        dtt = dt[:, t].reshape(batch, nheads, 1, 1)
        xt = x[:, t][:, :, :, None]
        Bt = Bmat[:, t][:, :, None, :]

        dstate = dstate + dy[:, t][:, :, :, None] * Cmat[:, t][:, :, None, :]
        dC_l.append(mx.sum(dy[:, t][:, :, :, None] * st, axis=2))
        d_force = dstate
        ddecay_l.append(mx.sum(dstate * state_tm1, axis=(2, 3)))
        d_bu = d_force * tr + dprev_bu
        d_bu_tm1 = d_force * (1.0 - tr)
        dtrap_l.append(mx.sum(d_force * (bu - bu_tm1), axis=(2, 3)))
        dB_l.append(mx.sum(d_bu * xt * dtt, axis=2))
        dx_l.append(mx.sum(d_bu * Bt * dtt, axis=3))
        ddt_l.append(mx.sum(d_bu * Bt * xt, axis=(2, 3)))
        dstate = dstate * dec
        dprev_bu = d_bu_tm1

    return (
        mx.stack(dx_l[::-1], axis=1),
        mx.stack(ddecay_l[::-1], axis=1),
        mx.stack(ddt_l[::-1], axis=1),
        mx.stack(dB_l[::-1], axis=1),
        mx.stack(dC_l[::-1], axis=1),
        mx.stack(dtrap_l[::-1], axis=1),
    )


@mx.custom_function
def selective_scan(
    x: mx.array,
    decay: mx.array,
    dt: mx.array,
    Bmat: mx.array,
    Cmat: mx.array,
    trap: mx.array,
) -> mx.array:
    """Trainable scan: Metal forward + Metal reverse VJP."""
    return selective_scan_fwd_metal(x, decay, dt, Bmat, Cmat, trap)


@selective_scan.vjp
def _selective_scan_vjp(primals, cotangent, _output):
    try:
        return selective_scan_bwd_metal(*primals, cotangent)
    except Exception:
        return _scan_vjp_mlx(primals, cotangent)
