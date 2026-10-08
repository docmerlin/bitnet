"""Single-token (decode) Metal kernels that collapse chains of small MLX ops.

Decode at M=1 is bound by kernel count, not FLOPs, so each kernel here replaces a
handful of dispatches with one. None has a VJP: callers use them only for
single-token inputs, which only occur at inference.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# Bound at import: caching a kernel's dispatch is separate from compiling a decode
# step, which callers (and tests) may patch mx.compile to control.
_compile_dispatch = mx.compile

# Kimi AttnRes depth mix for one token: h = softmax_j(w . RMSNorm(v_j)) . v over the
# N completed blocks plus the partial block. RMSNorm folds into the logit:
# w . (v * r * g) == r * (v . (g * w)) with r = rsqrt(mean(v^2) + eps).
# Decode is latency-bound: one thread per hidden element loads its slice of every
# block at once (no serial load chains) and keeps it in registers for the mix.
_DEPTH_MIX = mx.fast.metal_kernel(
    name="depth_attn_mix_m1",
    input_names=["completed", "partial", "norm_weight", "proj_weight", "eps"],
    output_names=["y"],
    source=r"""
        constexpr uint EPT = (D + THREADS - 1) / THREADS;
        constexpr uint SIMDGROUPS = THREADS / 32;
        uint t = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        // Weight group (grid y, 1 unless grouped): G layers' mixes in one dispatch.
        uint gi = threadgroup_position_in_grid.y;
        auto cg = completed + gi * N * D;
        auto pg = partial + gi * D;
        auto nw = norm_weight + gi * D;
        auto pw = proj_weight + gi * D;
        auto yg = y + gi * D;
        threadgroup float red[SIMDGROUPS][2 * (N + 1)];
        threadgroup float logits[N + 1];

        float v[N + 1][EPT];
        float ss[N + 1];
        float dot[N + 1];
        for (uint j = 0; j <= N; ++j) {
            ss[j] = 0.0f;
            dot[j] = 0.0f;
        }
        for (uint e = 0; e < EPT; ++e) {
            uint d = t + e * THREADS;
            bool in = d < D;
            float gw = in ? float(nw[d]) * float(pw[d]) : 0.0f;
            for (uint j = 0; j <= N; ++j) {
                float x = !in ? 0.0f : (j < N ? float(cg[j * D + d]) : float(pg[d]));
                v[j][e] = x;
                ss[j] += x * x;
                dot[j] += x * gw;
            }
        }
        for (uint j = 0; j <= N; ++j) {
            float a = simd_sum(ss[j]);
            float c = simd_sum(dot[j]);
            if (lane == 0) {
                red[sg][2 * j] = a;
                red[sg][2 * j + 1] = c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t <= N) {
            float a = 0.0f;
            float c = 0.0f;
            for (uint s = 0; s < SIMDGROUPS; ++s) {
                a += red[s][2 * t];
                c += red[s][2 * t + 1];
            }
            logits[t] = rsqrt(a / float(D) + eps[0]) * c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m = logits[0];
        for (uint j = 1; j <= N; ++j) {
            m = max(m, logits[j]);
        }
        float p[N + 1];
        float z = 0.0f;
        for (uint j = 0; j <= N; ++j) {
            p[j] = exp(logits[j] - m);
            z += p[j];
        }
        for (uint e = 0; e < EPT; ++e) {
            uint d = t + e * THREADS;
            if (d < D) {
                float acc = 0.0f;
                for (uint j = 0; j <= N; ++j) {
                    acc += p[j] * v[j][e];
                }
                yg[d] = T(acc / z);
            }
        }
    """,
)


@lru_cache(maxsize=None)
def _compiled_depth_mix(n: int, d: int, dtype, groups: int = 1):
    threads = min(1024, (d + 31) // 32 * 32)
    def dispatch(completed, partial, norm_weight, proj_weight, eps):
        return _DEPTH_MIX(
            inputs=[completed, partial, norm_weight, proj_weight, eps],
            template=[("T", dtype), ("N", n), ("D", d), ("THREADS", threads)],
            grid=(threads, groups, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(groups * d,)],
            output_dtypes=[dtype],
        )[0]

    # Compiled replay avoids metal_kernel's ~10us Python cost per call.
    return _compile_dispatch(dispatch)


@lru_cache(maxsize=None)
def _scalar(value: float) -> mx.array:
    return mx.array([value], dtype=mx.float32)


def depth_attn_mix_m1(
    completed: mx.array,
    partial: mx.array,
    norm_weight: mx.array,
    proj_weight: mx.array,
    eps: float,
) -> mx.array:
    """Depth mix of ``completed`` ``(N, D)`` and ``partial`` (``D`` elements, any shape).

    Grouped: ``completed`` ``(G, N, D)``, ``partial`` ``G·D`` elements and weights
    ``(G, D)`` run G layers' mixes in one dispatch.
    """
    groups = completed.shape[0] if completed.ndim == 3 else 1
    n, d = completed.shape[-2:]
    if partial.size != groups * d:
        raise ValueError("depth_attn_mix_m1 requires a single-token partial (per group)")
    kernel = _compiled_depth_mix(n, d, partial.dtype, groups)
    y = kernel(completed, partial.reshape(-1), norm_weight, proj_weight.reshape(-1), _scalar(eps))
    return y.reshape(partial.shape)


# One PaTH decode step (last query of the open chunk) fused with the Infini memory
# read and gate, per (batch, head) on one simdgroup. Replaces path_border_update_t,
# path_chunk_last_with_t, _retrieve_memory and the gate mix (~45 small kernels).
#   T row:      s_j = beta_new (w_new . w_j),  t_row = -s^T T_prev
#   u = qw T,   corrected_j = sum_{i>j} u_i (w_i . k_j) = k_j . R_j,  R_j = sum_{i>j} u_i w_i
# R_j is a suffix sum, so corrected costs O(L D) instead of O(L^2 D).
# It also builds the token's own inputs from the raw projections (per-head q/k
# RMSNorm, causal conv + silu + normalize for w, beta and forget heads) and appends
# them to the open-chunk caches and path history: each threadgroup writes only its
# own (batch, head) slice.
_PATH_DECODE = mx.fast.metal_kernel(
    name="path_decode_step_m1",
    input_names=[
        "qkv", "projected", "history", "conv_w", "x_norm", "head_w", "head_small",
        "q_prev", "k_prev", "v_prev", "w_prev", "beta_prev", "lf_prev", "t_prev",
        "memory_m", "memory_z", "memory_initialized", "params",
    ],
    output_names=["y", "t_new", "q_cat", "k_cat", "v_cat", "w_cat", "beta_cat", "lf_cat", "hist_new"],
    source=r"""
        // Decode is latency-bound: 256 threads per (batch, head), no serial per-lane
        // chains over L. Phases are separated by barriers; each does O(1) dependent
        // loads per thread (copies batch loads before stores, dots split D four ways).
        constexpr uint THREADS = 256;
        constexpr uint SG = THREADS / 32;
        constexpr uint DPL = D / 32;          // lane-over-D elements in simdgroup work
        constexpr uint JP = 4;                // threads per row in the dot phases
        constexpr uint DJ = D / JP;
        constexpr uint JSTEP = THREADS / JP;
        constexpr uint PARTS = THREADS / D;   // threads per output element at the end
        constexpr uint TILE = 4096 / D;       // rows of the suffix-sum tile (16KB)
        constexpr uint HD = H * D;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint bh = threadgroup_position_in_grid.x;
        uint b = bh / H;
        uint h = bh % H;
        // GROUPED: batch row b runs layer b's weights (G layers' steps in one dispatch).
        auto cw = conv_w + (GROUPED ? b * H * D * 3 : 0);
        auto hw = head_w + (GROUPED ? b * 2 * H * H * D : 0);
        auto hs = head_small + (GROUPED ? b * (3 * H + 2 * D) : 0);
        uint L = uint(params[0]);
        uint Lp = L - 1;
        uint n_hist = uint(params[1]);
        float eps = params[2];
        threadgroup float qk[LMAX];
        threadgroup float qw[LMAX];
        threadgroup float s[LMAX];
        threadgroup float u[LMAX];
        threadgroup float corr[LMAX];
        threadgroup float pf[LMAX];
        threadgroup float trow[LMAX];
        threadgroup float sq[D];
        threadgroup float rt[TILE * D];
        threadgroup float red[2 * SG];
        threadgroup float den_shared[1];

        // Layouts: qkv (B,3*H*D); projected, x_norm (B,H*D); history (B,n_hist,H*D);
        // conv_w (H*D,3); head_w (2H,H*D) = [path_beta W; path_forget W]; head_small
        // fp32 = [beta b (H), forget b (H), memory gate (H), q_norm (D), k_norm (D)];
        // q, k, v caches (B,H,L,D); w (B,L,H,D); beta, forget (B,L,H); "_prev" caches
        // hold L - 1 tokens (unread when L == 1); t_prev (B,H,L-1,L-1);
        // memory_m (B,H,D,D); memory_z (B,H,D).
        auto wrow = [&](uint j) { return ((b * L + j) * H + h) * D; };
        uint kbase = bh * L * D;
        uint newrow = kbase + Lp * D;
        uint pbase = bh * Lp * D;
        uint tpbase = bh * Lp * Lp;
        uint tnbase = bh * L * L;

        // ---- P: copy the open chunk, build the new token's row ----
        for (uint base = tid; base < Lp * D; base += 4 * THREADS) {
            T qa[4];
            T ka[4];
            T va[4];
            float wa[4];
            for (uint r = 0; r < 4; ++r) {
                uint idx = base + r * THREADS;
                bool ok = idx < Lp * D;
                uint j = idx / D;
                uint d = idx % D;
                qa[r] = ok ? q_prev[pbase + idx] : T(0);
                ka[r] = ok ? k_prev[pbase + idx] : T(0);
                va[r] = ok ? v_prev[pbase + idx] : T(0);
                wa[r] = ok ? w_prev[((b * Lp + j) * H + h) * D + d] : 0.0f;
            }
            for (uint r = 0; r < 4; ++r) {
                uint idx = base + r * THREADS;
                if (idx < Lp * D) {
                    q_cat[kbase + idx] = qa[r];
                    k_cat[kbase + idx] = ka[r];
                    v_cat[kbase + idx] = va[r];
                    w_cat[wrow(idx / D) + idx % D] = wa[r];
                }
            }
        }
        for (uint j = tid; j < Lp; j += THREADS) {
            float bv = beta_prev[(b * Lp + j) * H + h];
            float fv = lf_prev[(b * Lp + j) * H + h];
            beta_cat[(b * L + j) * H + h] = bv;
            lf_cat[(b * L + j) * H + h] = fv;
        }
        for (uint base = tid; base < Lp * L; base += 4 * THREADS) {
            float ta[4];
            for (uint r = 0; r < 4; ++r) {
                uint idx = base + r * THREADS;
                uint row = idx / L;
                uint col = idx % L;
                ta[r] = (idx < Lp * L && col < Lp) ? t_prev[tpbase + row * Lp + col] : 0.0f;
            }
            for (uint r = 0; r < 4; ++r) {
                uint idx = base + r * THREADS;
                if (idx < Lp * L) {
                    t_new[tnbase + idx] = ta[r];
                }
            }
        }
        if (sg == 0) {
            // q, k: per-head RMSNorm of the raw projection (as q_norm/k_norm, rounded to
            // the cache dtype); v as projected. Also the Infini read features phi(q).
            uint qkv_base = b * 3 * HD + h * D;
            float qr[DPL];
            float kr[DPL];
            float qss = 0.0f;
            float kss = 0.0f;
            for (uint t = 0; t < DPL; ++t) {
                uint d = lane + 32 * t;
                qr[t] = float(qkv[qkv_base + d]);
                kr[t] = float(qkv[qkv_base + HD + d]);
                qss += qr[t] * qr[t];
                kss += kr[t] * kr[t];
            }
            float qi = rsqrt(simd_sum(qss) / float(D) + eps);
            float ki = rsqrt(simd_sum(kss) / float(D) + eps);
            float qn = 0.0f;
            float qm = -INFINITY;
            for (uint t = 0; t < DPL; ++t) {
                uint d = lane + 32 * t;
                T qv = T(qr[t] * qi * hs[3 * H + d]);
                q_cat[newrow + d] = qv;
                k_cat[newrow + d] = T(kr[t] * ki * hs[3 * H + D + d]);
                v_cat[newrow + d] = qkv[qkv_base + 2 * HD + d];
                qr[t] = float(qv);
                qn += qr[t] * qr[t];
                qm = max(qm, qr[t]);
            }
            qn = simd_sum(qn) * 0.5f;
            qm = simd_max(qm);
            float den = 0.0f;
            for (uint t = 0; t < DPL; ++t) {
                uint d = lane + 32 * t;
                float f = FAVOR ? exp(qr[t] - qn - qm) : (qr[t] > 0.0f ? qr[t] + 1.0f : exp(qr[t]));
                sq[d] = f;
                den += f * float(memory_z[bh * D + d]);
            }
            den = simd_sum(den);
            if (lane == 0) {
                den_shared[0] = max(den, 1e-6f);
            }
        } else if (sg == 1) {
            // w: causal conv over (history, projected), silu, then unit norm per head
            // (sqrt(sum + 1e-12) floored at 1e-6, as _safe_normalize).
            float cs[DPL];
            float wss = 0.0f;
            for (uint t = 0; t < DPL; ++t) {
                uint idx = h * D + lane + 32 * t;
                float c = float(projected[b * HD + idx]) * float(cw[idx * 3 + 2]);
                if (n_hist >= 1) {
                    c += float(history[(b * n_hist + n_hist - 1) * HD + idx]) * float(cw[idx * 3 + 1]);
                }
                if (n_hist >= 2) {
                    c += float(history[(b * n_hist + n_hist - 2) * HD + idx]) * float(cw[idx * 3]);
                }
                cs[t] = c / (1.0f + exp(-c));
                wss += cs[t] * cs[t];
            }
            float winv = 1.0f / max(sqrt(simd_sum(wss) + 1e-12f), 1e-6f);
            for (uint t = 0; t < DPL; ++t) {
                w_cat[wrow(Lp) + lane + 32 * t] = cs[t] * winv;
            }
            // Path history keeps the last two projections of (history, projected).
            uint n_out = min(n_hist + 1, 2u);
            uint skip = n_hist + 1 - n_out;
            for (uint r = 0; r < n_out; ++r) {
                uint src = r + skip;
                for (uint t = 0; t < DPL; ++t) {
                    uint idx = h * D + lane + 32 * t;
                    hist_new[(b * n_out + r) * HD + idx] =
                        src < n_hist ? history[(b * n_hist + src) * HD + idx] : projected[b * HD + idx];
                }
            }
        }
        // beta = 2 sigmoid(path_beta(x)); log_forget = log sigmoid(path_forget(x)):
        // every thread takes a slice of the H*D dot products.
        {
            float bd = 0.0f;
            float fd = 0.0f;
            for (uint i = tid; i < HD; i += THREADS) {
                float xv = float(x_norm[b * HD + i]);
                bd += xv * float(hw[h * HD + i]);
                fd += xv * float(hw[(H + h) * HD + i]);
            }
            bd = simd_sum(bd);
            fd = simd_sum(fd);
            if (lane == 0) {
                red[sg] = bd;
                red[SG + sg] = fd;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float bd = hs[h];
        float fd = hs[H + h];
        for (uint i = 0; i < SG; ++i) {
            bd += red[i];
            fd += red[SG + i];
        }
        float beta_new = 2.0f / (1.0f + exp(-bd));
        if (tid == 0) {
            beta_cat[(b * L + Lp) * H + h] = beta_new;
            lf_cat[(b * L + Lp) * H + h] = min(fd, 0.0f) - log1p(exp(-abs(fd)));
        }

        // ---- A: qk_j = q.k_j, qw_j = q.w_j, s_j = beta_new (w_new.w_j) ----
        uint part = tid % JP;
        for (uint j = tid / JP; j < L; j += JSTEP) {
            float a = 0.0f;
            float c = 0.0f;
            float e = 0.0f;
            for (uint dd = 0; dd < DJ; ++dd) {
                uint d = part * DJ + dd;
                float qd = float(q_cat[newrow + d]);
                float wd = float(w_cat[wrow(j) + d]);
                a += qd * float(k_cat[kbase + j * D + d]);
                c += qd * wd;
                e += float(w_cat[wrow(Lp) + d]) * wd;
            }
            for (ushort m = 1; m < JP; m <<= 1) {
                a += simd_shuffle_xor(a, m);
                c += simd_shuffle_xor(c, m);
                e += simd_shuffle_xor(e, m);
            }
            if (part == 0) {
                qk[j] = a;
                qw[j] = c;
                s[j] = beta_new * e;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- B: T row, t_row_c = -sum_{i >= c} s_i T_prev[i][c] ----
        for (uint c = tid / JP; c < Lp; c += JSTEP) {
            float acc = 0.0f;
            for (uint i = c + part; i < Lp; i += JP) {
                acc += s[i] * float(t_prev[tpbase + i * Lp + c]);
            }
            for (ushort m = 1; m < JP; m <<= 1) {
                acc += simd_shuffle_xor(acc, m);
            }
            if (part == 0) {
                trow[c] = -acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- C: u = qw T (T lower triangular) and T's new last row ----
        for (uint i = tid / JP; i < L; i += JSTEP) {
            float acc = 0.0f;
            for (uint m = i + part; m < Lp; m += JP) {
                acc += qw[m] * float(t_prev[tpbase + m * Lp + i]);
            }
            for (ushort m = 1; m < JP; m <<= 1) {
                acc += simd_shuffle_xor(acc, m);
            }
            if (part == 0) {
                u[i] = acc + qw[Lp] * (i < Lp ? trow[i] : beta_new);
            }
        }
        for (uint c = tid; c < L; c += THREADS) {
            t_new[tnbase + Lp * L + c] = c < Lp ? trow[c] : beta_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- D: corrected_j = k_j . R_j, R_j = sum_{i > j} u_i w_i, by TILE-row tiles
        // from the end; thread d carries the running suffix sum across tiles ----
        float carry = 0.0f;
        for (int t0 = int((L - 1) / TILE * TILE); t0 >= 0; t0 -= int(TILE)) {
            uint start = uint(t0);
            uint rows = min(L - start, TILE);
            for (uint idx = tid; idx < rows * D; idx += THREADS) {
                uint r = start + idx / D;
                rt[idx] = u[r] * w_cat[wrow(r) + idx % D];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < D) {
                for (int r = int(rows) - 1; r >= 0; --r) {
                    float pv = rt[uint(r) * D + tid];
                    rt[uint(r) * D + tid] = carry;
                    carry += pv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = tid / JP; jj < rows; jj += JSTEP) {
                uint j = start + jj;
                float acc = 0.0f;
                for (uint dd = 0; dd < DJ; ++dd) {
                    uint d = part * DJ + dd;
                    acc += float(k_cat[kbase + j * D + d]) * rt[jj * D + d];
                }
                for (ushort m = 1; m < JP; m <<= 1) {
                    acc += simd_shuffle_xor(acc, m);
                }
                if (part == 0) {
                    corr[j] = acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- E: forget prefix, logits, softmax (simdgroup 0) ----
        if (sg == 0) {
            float run = 0.0f;
            for (uint c = 0; c < L; c += 32) {
                uint j = c + lane;
                float lf = j < L ? float(lf_cat[(b * L + j) * H + h]) : 0.0f;
                float ps = simd_prefix_inclusive_sum(lf) + run;
                if (j < L) {
                    pf[j] = ps;
                }
                run = simd_broadcast(ps, 31);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float scale = rsqrt(float(D));
            float m_max = -INFINITY;
            for (uint j = lane; j < L; j += 32) {
                float logit = (qk[j] - corr[j]) * scale + pf[Lp] - pf[j];
                qk[j] = logit;
                m_max = max(m_max, logit);
            }
            m_max = simd_max(m_max);
            float z = 0.0f;
            for (uint j = lane; j < L; j += 32) {
                float pj = exp(qk[j] - m_max);
                qk[j] = pj;
                z += pj;
            }
            z = simd_sum(z);
            for (uint j = lane; j < L; j += 32) {
                qk[j] /= z;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- G: local = p . V and the Infini read phi(q) M / den, gated ----
        uint d = tid / PARTS;
        uint q4 = tid % PARTS;
        float local = 0.0f;
        float num = 0.0f;
        for (uint j = q4; j < L; j += PARTS) {
            local += qk[j] * float(v_cat[kbase + j * D + d]);
        }
        for (uint e = q4; e < D; e += PARTS) {
            num += sq[e] * float(memory_m[(bh * D + e) * D + d]);
        }
        for (ushort m = 1; m < PARTS; m <<= 1) {
            local += simd_shuffle_xor(local, m);
            num += simd_shuffle_xor(num, m);
        }
        if (q4 == 0) {
            float out = local;
            if (memory_initialized[b]) {
                float g = 1.0f / (1.0f + exp(-hs[2 * H + h]));
                out = (1.0f - g) * local + g * (num / den_shared[0]);
            }
            y[bh * D + d] = T(out);
        }
    """,
)

# Threadgroup scratch is 7 * LMAX floats plus a 16KB tile; 256 keeps it at 23KB.
PATH_DECODE_MAX_WIDTH = 256


@lru_cache(maxsize=None)
def _step_params(length: int, n_hist: int, eps: float) -> mx.array:
    return mx.array([length, n_hist, eps], dtype=mx.float32)


@lru_cache(maxsize=None)
def _compiled_path_step(
    batch: int,
    heads: int,
    length: int,
    dim: int,
    n_hist: int,
    lmax: int,
    dtype,
    hist_dtype,
    favor: bool,
    grouped: bool = False,
):
    cache_shape = (batch, heads, length, dim)

    def dispatch(*inputs):
        return _PATH_DECODE(
            inputs=list(inputs),
            template=[("T", dtype), ("D", dim), ("H", heads), ("LMAX", lmax), ("FAVOR", favor), ("GROUPED", grouped)],
            grid=(256 * batch * heads, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (batch, heads, 1, dim),
                (batch, heads, length, length),
                cache_shape,
                cache_shape,
                cache_shape,
                (batch, length, heads, dim),
                (batch, length, heads),
                (batch, length, heads),
                (batch, min(n_hist + 1, 2), heads * dim),
            ],
            output_dtypes=[dtype, mx.float32, dtype, dtype, dtype, mx.float32, mx.float32, mx.float32, hist_dtype],
        )

    return _compile_dispatch(dispatch)


def pack_path_head_weights(
    beta_weight: mx.array,
    beta_bias: mx.array,
    forget_weight: mx.array,
    forget_bias: mx.array,
    memory_gate: mx.array,
    q_norm_weight: mx.array,
    k_norm_weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """(head_w, head_small) for path_decode_step_m1; Metal caps a kernel at 31 buffers."""
    small = [t.astype(mx.float32) for t in (beta_bias, forget_bias, memory_gate, q_norm_weight, k_norm_weight)]
    return mx.concatenate([beta_weight, forget_weight]), mx.concatenate(small)


def path_decode_step_m1(
    qkv: mx.array,
    projected: mx.array,
    history: mx.array | None,
    x_norm: mx.array,
    prev: tuple[mx.array, ...] | None,
    t_prev: mx.array | None,
    memory: tuple[mx.array, mx.array, mx.array],
    weights: tuple[mx.array, mx.array, mx.array],
    *,
    heads: int,
    norm_eps: float,
    chunk_width: int,
    favor: bool,
) -> tuple[mx.array, ...]:
    """One PaTH decode step from the raw projections of one token.

    ``qkv`` (B,1,3*H*D), ``projected`` (path_up output) and ``x_norm`` (B,1,H*D);
    ``history`` the last <= 2 path projections (B,n,H*D) or None. ``prev`` is the
    open chunk (q, k, v (B,H,L-1,D); w (B,L-1,H,D); beta, log_forget (B,L-1,H)) or None
    to start one, ``t_prev`` its running T. ``memory`` is (M, z, initialized);
    ``weights`` is (conv (H*D,3), head_w, head_small) from pack_path_head_weights;
    stacked over B layers (leading axis) it is grouped: batch row b runs layer b.

    Returns (context with the gated Infini memory read, T, q, k, v, w, beta,
    log_forget, history): the grown caches and the new path history.
    """
    batch = qkv.shape[0]
    hidden = projected.shape[-1]
    dim = hidden // heads
    length = 1 if prev is None else prev[1].shape[2] + 1
    n_hist = 0 if history is None else history.shape[1]
    if dim not in (32, 64, 128, 256) or dim * heads != hidden:
        raise ValueError("path_decode_step_m1 requires head_dim in {32, 64, 128, 256}")
    if length > chunk_width or chunk_width > PATH_DECODE_MAX_WIDTH:
        raise ValueError(f"path_decode_step_m1 supports chunk widths <= {PATH_DECODE_MAX_WIDTH}")
    if (t_prev is None) != (prev is None):
        raise ValueError("t_prev must be given exactly when continuing a chunk")
    if n_hist > 2:
        raise ValueError("path history holds at most two projections")
    grouped = weights[0].ndim == 3
    if grouped and weights[0].shape[0] != batch:
        raise ValueError("grouped weights need one layer per batch row")
    kernel = _compiled_path_step(
        batch, heads, length, dim, n_hist, chunk_width, qkv.dtype, projected.dtype, favor, grouped
    )
    # Unread slots (no open chunk, no history) still need an array.
    prev = (qkv,) * 6 if prev is None else prev
    return tuple(kernel(
        qkv.reshape(batch, 3 * hidden),
        projected.reshape(batch, hidden),
        projected if history is None else history,
        weights[0],
        x_norm.reshape(batch, hidden),
        weights[1],
        weights[2],
        *prev,
        qkv if t_prev is None else t_prev,
        *memory,
        _step_params(length, n_hist, norm_eps),
    ))
