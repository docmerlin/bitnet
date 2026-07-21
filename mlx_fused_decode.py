"""Fast M=1 GEMV and sequential GEMV-chain Metal kernels for decode."""

from __future__ import annotations

import mlx.core as mx

# One row-parallel GEMV: y[o] = sum_i x[i] * W[o, i]  for M=1
_GEMV = mx.fast.metal_kernel(
    name="fast_gemv_m1",
    input_names=["x", "weight"],
    output_names=["y"],
    source=r"""
        uint o = thread_position_in_grid.x;
        uint out_d = weight_shape[0];
        uint in_d = weight_shape[1];
        if (o >= out_d) return;
        float sum = 0.0f;
        // Vectorize by 4 when possible
        uint i = 0;
        for (; i + 4u <= in_d; i += 4u) {
            sum += float(x[i])     * float(weight[o * in_d + i]);
            sum += float(x[i+1u]) * float(weight[o * in_d + i + 1u]);
            sum += float(x[i+2u]) * float(weight[o * in_d + i + 2u]);
            sum += float(x[i+3u]) * float(weight[o * in_d + i + 3u]);
        }
        for (; i < in_d; ++i) {
            sum += float(x[i]) * float(weight[o * in_d + i]);
        }
        y[o] = T(sum);
    """,
)

# Full sequential chain in one dispatch (shared memory activation)
_GEMV_CHAIN = mx.fast.metal_kernel(
    name="fast_gemv_chain_m1",
    input_names=["x", "weights", "offsets", "in_dims", "out_dims"],
    output_names=["y"],
    source=r"""
        threadgroup float act_a[2048];
        threadgroup float act_b[2048];
        uint lane = thread_position_in_threadgroup.x;
        uint tg = threads_per_threadgroup.x;
        uint n_layers = offsets_shape[0];
        uint in0 = in_dims[0];
        for (uint i = lane; i < in0; i += tg) {
            act_a[i] = float(x[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint layer = 0; layer < n_layers; ++layer) {
            uint in_d = in_dims[layer];
            uint out_d = out_dims[layer];
            uint woff = offsets[layer];
            // Each thread owns a set of output rows
            for (uint o = lane; o < out_d; o += tg) {
                float sum = 0.0f;
                uint base = woff + o * in_d;
                uint i = 0;
                for (; i + 4u <= in_d; i += 4u) {
                    sum += act_a[i]     * float(weights[base + i]);
                    sum += act_a[i+1u] * float(weights[base + i + 1u]);
                    sum += act_a[i+2u] * float(weights[base + i + 2u]);
                    sum += act_a[i+3u] * float(weights[base + i + 3u]);
                }
                for (; i < in_d; ++i) {
                    sum += act_a[i] * float(weights[base + i]);
                }
                act_b[o] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint o = lane; o < out_d; o += tg) {
                act_a[o] = act_b[o];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint last_out = out_dims[n_layers - 1];
        for (uint o = lane; o < last_out; o += tg) {
            y[o] = T(act_a[o]);
        }
    """,
)


def fast_gemv(x: mx.array, weight: mx.array) -> mx.array:
    """M=1 GEMV: x (..., in) @ weight.T (out, in) -> (..., out)."""
    out_d, in_d = map(int, weight.shape)
    orig = x.shape
    xf = x.reshape(-1, in_d)
    if int(xf.shape[0]) != 1:
        return x @ weight.astype(x.dtype).T
    y = _GEMV(
        inputs=[xf[0].astype(mx.float32), weight.astype(mx.float32)],
        template=[("T", mx.float32)],
        grid=(out_d, 1, 1),
        threadgroup=(min(out_d, 256), 1, 1),
        output_shapes=[(out_d,)],
        output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*orig[:-1], out_d).astype(x.dtype)


def fused_gemv_chain(
    x: mx.array,
    weight_list: list[mx.array],
    *,
    silu_layers: set[int] | None = None,
) -> mx.array:
    """Sequential GEMVs; silu_layers ignored in chain kernel (apply outside if needed)."""
    if not weight_list:
        return x
    if silu_layers:
        # Fall back when silu interleave required
        y = x
        for i, w in enumerate(weight_list):
            y = fast_gemv(y, w)
            if i in silu_layers:
                y = y * mx.sigmoid(y)
        return y
    orig = x.shape
    in0 = int(weight_list[0].shape[1])
    xf = x.reshape(-1, in0)
    if int(xf.shape[0]) != 1:
        y = x
        for w in weight_list:
            y = y @ w.astype(y.dtype).T
        return y
    offsets, in_dims, out_dims, pieces = [], [], [], []
    offset = 0
    for w in weight_list:
        w32 = w.astype(mx.float32)
        out_d, in_d = map(int, w32.shape)
        pieces.append(w32.reshape(-1))
        offsets.append(offset)
        in_dims.append(in_d)
        out_dims.append(out_d)
        offset += out_d * in_d
        if out_d > 2048 or in_d > 2048:
            # shared mem limit — sequential fast_gemv
            y = x
            for ww in weight_list:
                y = fast_gemv(y, ww)
            return y
    y = _GEMV_CHAIN(
        inputs=[
            xf[0].astype(mx.float32),
            mx.concatenate(pieces),
            mx.array(offsets, dtype=mx.uint32),
            mx.array(in_dims, dtype=mx.uint32),
            mx.array(out_dims, dtype=mx.uint32),
        ],
        template=[("T", mx.float32)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(out_dims[-1],)],
        output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*orig[:-1], out_dims[-1]).astype(x.dtype)
