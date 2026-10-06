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
            float gw = in ? float(norm_weight[d]) * float(proj_weight[d]) : 0.0f;
            for (uint j = 0; j <= N; ++j) {
                float x = !in ? 0.0f : (j < N ? float(completed[j * D + d]) : float(partial[d]));
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
                y[d] = T(acc / z);
            }
        }
    """,
)


@lru_cache(maxsize=None)
def _compiled_depth_mix(n: int, d: int, dtype):
    threads = min(1024, (d + 31) // 32 * 32)
    def dispatch(completed, partial, norm_weight, proj_weight, eps):
        return _DEPTH_MIX(
            inputs=[completed, partial, norm_weight, proj_weight, eps],
            template=[("T", dtype), ("N", n), ("D", d), ("THREADS", threads)],
            grid=(threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(d,)],
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
    """Depth mix of ``completed`` ``(N, D)`` and ``partial`` (``D`` elements, any shape)."""
    n, d = completed.shape
    if partial.size != d:
        raise ValueError("depth_attn_mix_m1 requires a single-token partial")
    kernel = _compiled_depth_mix(n, d, partial.dtype)
    return kernel(completed, partial.reshape(d), norm_weight, proj_weight.reshape(d), _scalar(eps)).reshape(partial.shape)


# One PaTH decode step (last query of the open chunk) fused with the Infini memory
# read and gate, per (batch, head) on one simdgroup. Replaces path_border_update_t,
# path_chunk_last_with_t, _retrieve_memory and the gate mix (~45 small kernels).
#   T row:      s_j = beta_new (w_new . w_j),  t_row = -s^T T_prev
#   u = qw T,   corrected_j = sum_{i>j} u_i (w_i . k_j) = k_j . R_j,  R_j = sum_{i>j} u_i w_i
# R_j is a suffix sum, so corrected costs O(L D) instead of O(L^2 D).
# It also appends the new token to the open-chunk caches (q, k, v, w, beta, forget),
# replacing six concatenations: each threadgroup copies its own (batch, head) slice.
_PATH_DECODE = mx.fast.metal_kernel(
    name="path_decode_step_m1",
    input_names=[
        "q_new", "k_new", "v_new", "w_new", "beta_new_in", "lf_new",
        "q_prev", "k_prev", "v_prev", "w_prev", "beta_prev", "lf_prev", "t_prev",
        "memory_m", "memory_z", "memory_initialized", "memory_gate", "params",
    ],
    output_names=["y", "t_new", "q_cat", "k_cat", "v_cat", "w_cat", "beta_cat", "lf_cat"],
    source=r"""
        constexpr uint DPL = D / 32;
        uint lane = thread_index_in_simdgroup;
        uint bh = threadgroup_position_in_grid.x;
        uint b = bh / H;
        uint h = bh % H;
        uint L = uint(params[0]);
        uint Lp = L - 1;
        threadgroup float qk[LMAX];
        threadgroup float qw[LMAX];
        threadgroup float s[LMAX];
        threadgroup float u[LMAX];
        threadgroup float corr[LMAX];
        threadgroup float pf[LMAX];
        threadgroup float trow[LMAX];
        threadgroup float sq[D];

        // Layouts: q, k, v (B,H,L,D); w (B,L,H,D); beta, forget (B,L,H); "_new" inputs
        // are the same with L = 1 and "_prev" with L - 1 (unread when L == 1);
        // t_prev (B,H,L-1,L-1); memory_m (B,H,D,D); memory_z (B,H,D).
        auto wrow = [&](uint j) { return ((b * L + j) * H + h) * D; };
        uint qbase = bh * D;
        uint kbase = bh * L * D;
        for (uint idx = lane; idx < L * D; idx += 32) {
            uint j = idx / D;
            uint d = idx % D;
            bool old = j < Lp;
            q_cat[kbase + idx] = old ? q_prev[bh * Lp * D + idx] : q_new[qbase + d];
            k_cat[kbase + idx] = old ? k_prev[bh * Lp * D + idx] : k_new[qbase + d];
            v_cat[kbase + idx] = old ? v_prev[bh * Lp * D + idx] : v_new[qbase + d];
            w_cat[wrow(j) + d] = old ? w_prev[((b * Lp + j) * H + h) * D + d] : w_new[(b * H + h) * D + d];
        }
        for (uint j = lane; j < L; j += 32) {
            bool old = j < Lp;
            beta_cat[(b * L + j) * H + h] = old ? beta_prev[(b * Lp + j) * H + h] : beta_new_in[b * H + h];
            lf_cat[(b * L + j) * H + h] = old ? lf_prev[(b * Lp + j) * H + h] : lf_new[b * H + h];
        }
        // This threadgroup reads back only the slice it just wrote.
        threadgroup_barrier(mem_flags::mem_device);
        float beta_new = float(beta_cat[(b * L + Lp) * H + h]);

        for (uint j = lane; j < L; j += 32) {
            float a = 0.0f, c = 0.0f, e = 0.0f;
            uint wj = wrow(j);
            uint wn = wrow(Lp);
            for (uint d = 0; d < D; ++d) {
                float qd = float(q_new[qbase + d]);
                float wd = float(w_cat[wj + d]);
                a += qd * float(k_cat[kbase + j * D + d]);
                c += qd * wd;
                e += float(w_cat[wn + d]) * wd;
            }
            qk[j] = a;
            qw[j] = c;
            s[j] = beta_new * e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tpbase = bh * Lp * Lp;
        for (uint c = lane; c < Lp; c += 32) {
            float acc = 0.0f;
            for (uint i = c; i < Lp; ++i) {
                acc += s[i] * float(t_prev[tpbase + i * Lp + c]);
            }
            trow[c] = -acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tnbase = bh * L * L;
        for (uint idx = lane; idx < L * L; idx += 32) {
            uint r = idx / L;
            uint c = idx % L;
            float val;
            if (r < Lp) {
                val = c < Lp ? float(t_prev[tpbase + r * Lp + c]) : 0.0f;
            } else {
                val = c < Lp ? trow[c] : beta_new;
            }
            t_new[tnbase + idx] = val;
        }
        for (uint i = lane; i < L; i += 32) {
            float acc = 0.0f;
            for (uint m = i; m < Lp; ++m) {
                acc += qw[m] * float(t_prev[tpbase + m * Lp + i]);
            }
            u[i] = acc + qw[Lp] * (i < Lp ? trow[i] : beta_new);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float r[DPL];
        for (uint t = 0; t < DPL; ++t) {
            r[t] = 0.0f;
        }
        for (int j = int(Lp); j >= 0; --j) {
            float part = 0.0f;
            uint wj = wrow(uint(j));
            for (uint t = 0; t < DPL; ++t) {
                part += float(k_cat[kbase + uint(j) * D + lane + 32 * t]) * r[t];
            }
            part = simd_sum(part);
            if (lane == 0) {
                corr[j] = part;
            }
            float uj = u[j];
            for (uint t = 0; t < DPL; ++t) {
                r[t] += uj * float(w_cat[wj + lane + 32 * t]);
            }
        }
        if (lane == 0) {
            float acc = 0.0f;
            for (uint j = 0; j < L; ++j) {
                acc += float(lf_cat[(b * L + j) * H + h]);
                pf[j] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
            float p = exp(qk[j] - m_max);
            qk[j] = p;
            z += p;
        }
        z = simd_sum(z);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local[DPL];
        for (uint t = 0; t < DPL; ++t) {
            local[t] = 0.0f;
        }
        for (uint j = 0; j < L; ++j) {
            float p = qk[j] / z;
            for (uint t = 0; t < DPL; ++t) {
                local[t] += p * float(v_cat[kbase + j * D + lane + 32 * t]);
            }
        }

        // Infini memory read: A_mem = phi(q) M / (phi(q) . z).
        float qn = 0.0f;
        float qm = -INFINITY;
        for (uint t = 0; t < DPL; ++t) {
            float qd = float(q_new[qbase + lane + 32 * t]);
            qn += qd * qd;
            qm = max(qm, qd);
        }
        qn = simd_sum(qn) * 0.5f;
        qm = simd_max(qm);
        for (uint t = 0; t < DPL; ++t) {
            uint d = lane + 32 * t;
            float qd = float(q_new[qbase + d]);
            sq[d] = FAVOR ? exp(qd - qn - qm) : (qd > 0.0f ? qd + 1.0f : exp(qd));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float den = 0.0f;
        for (uint t = 0; t < DPL; ++t) {
            uint d = lane + 32 * t;
            den += sq[d] * float(memory_z[bh * D + d]);
        }
        den = max(simd_sum(den), 1e-6f);
        float g = 1.0f / (1.0f + exp(-float(memory_gate[h])));
        bool init = memory_initialized[b];
        for (uint t = 0; t < DPL; ++t) {
            uint d = lane + 32 * t;
            float out = local[t];
            if (init) {
                float num = 0.0f;
                for (uint e = 0; e < D; ++e) {
                    num += sq[e] * float(memory_m[(bh * D + e) * D + d]);
                }
                out = (1.0f - g) * out + g * (num / den);
            }
            y[qbase + d] = T(out);
        }
    """,
)

# Threadgroup scratch is 7 * LMAX floats; 512 keeps it at 14KB.
PATH_DECODE_MAX_WIDTH = 512


@lru_cache(maxsize=None)
def _length_param(length: int) -> mx.array:
    return mx.array([length], dtype=mx.int32)


@lru_cache(maxsize=None)
def _compiled_path_step(batch: int, heads: int, length: int, dim: int, lmax: int, dtypes: tuple, favor: bool):
    q_dtype, k_dtype, v_dtype = dtypes
    cache_shape = (batch, heads, length, dim)

    def dispatch(*inputs):
        return _PATH_DECODE(
            inputs=list(inputs),
            template=[("T", v_dtype), ("D", dim), ("H", heads), ("LMAX", lmax), ("FAVOR", favor)],
            grid=(32 * batch * heads, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (batch, heads, 1, dim),
                (batch, heads, length, length),
                cache_shape,
                cache_shape,
                cache_shape,
                (batch, length, heads, dim),
                (batch, length, heads),
                (batch, length, heads),
            ],
            output_dtypes=[v_dtype, mx.float32, q_dtype, k_dtype, v_dtype, mx.float32, mx.float32, mx.float32],
        )

    return _compile_dispatch(dispatch)


def path_decode_step_m1(
    new: tuple[mx.array, ...],
    prev: tuple[mx.array, ...] | None,
    t_prev: mx.array | None,
    memory_m: mx.array,
    memory_z: mx.array,
    memory_initialized: mx.array,
    memory_gate: mx.array,
    *,
    chunk_width: int,
    favor: bool,
) -> tuple[mx.array, ...]:
    """One PaTH decode step: appends the token to the open chunk and attends with it.

    ``new`` is (q, k, v, w, beta, log_forget) for the token: q, k, v (B,H,1,D),
    w (B,1,H,D), beta and log_forget (B,1,H). ``prev`` is the open chunk in the same
    layouts with L - 1 tokens, or None to start a chunk; ``t_prev`` its running T
    (B,H,L-1,L-1). Returns (context with the gated Infini memory read, T, q, k, v, w,
    beta, log_forget), the last six being the grown caches.
    """
    batch, heads, _, dim = new[1].shape
    length = 1 if prev is None else prev[1].shape[2] + 1
    if dim % 32:
        raise ValueError("path_decode_step_m1 requires head_dim divisible by 32")
    if length > chunk_width or chunk_width > PATH_DECODE_MAX_WIDTH:
        raise ValueError(f"path_decode_step_m1 supports chunk widths <= {PATH_DECODE_MAX_WIDTH}")
    if (t_prev is None) != (prev is None):
        raise ValueError("t_prev must be given exactly when continuing a chunk")
    dtypes = (new[0].dtype, new[1].dtype, new[2].dtype)
    kernel = _compiled_path_step(batch, heads, length, dim, chunk_width, dtypes, favor)
    # Starting a chunk, the kernel never reads the "_prev" slots; fill them with any array.
    prev = new if prev is None else prev
    return tuple(kernel(
        *new, *prev, new[4] if t_prev is None else t_prev,
        memory_m, memory_z, memory_initialized, memory_gate, _length_param(length),
    ))
