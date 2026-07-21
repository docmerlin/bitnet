"""Packed 2-bit ternary linear primitive backed by MLX Metal kernels."""

from __future__ import annotations

import mlx.core as mx


_PACK_TERNARY_WEIGHT = mx.fast.metal_kernel(
    name="pack_ternary_weight",
    input_names=["weight"],
    output_names=["packed", "scales"],
    source=r"""
        uint lane = thread_position_in_threadgroup.x;
        uint row = thread_position_in_grid.y;
        uint input_size = weight_shape[1];
        uint group_size = input_size % 64 == 0 ? 64 : 32;
        threadgroup float partial[256];
        float sum = 0.0f;
        for (uint index = lane; index < input_size; index += 256) {
            sum += abs(float(weight[ulong(row) * input_size + index]));
        }
        partial[lane] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float scale = max(partial[0] / float(input_size), 1e-5f);
        for (uint group = lane; group < input_size / group_size; group += 256) {
            scales[ulong(row) * (input_size / group_size) + group] = T(scale);
        }
        for (uint word_index = lane; word_index < input_size / 16; word_index += 256) {
            uint word = 0;
            uint start = word_index * 16;
            for (uint offset = 0; offset < 16; ++offset) {
                float value = float(weight[ulong(row) * input_size + start + offset]);
                uint code = value > 0.5f * scale ? 2 : (value < -0.5f * scale ? 0 : 1);
                word |= code << (2 * offset);
            }
            packed[ulong(row) * (input_size / 16) + word_index] = word;
        }
    """,
)


def pack_ternary_weight(weight: mx.array):
    if weight.ndim != 2:
        raise ValueError("ternary quantized matmul requires rank-2 weights")
    input_dims = weight.shape[-1]
    group_size = 64 if input_dims % 64 == 0 else 32
    if input_dims % group_size:
        raise ValueError("ternary quantized matmul requires input dimensions divisible by 32")
    rows = weight.shape[0]
    packed, scales = _PACK_TERNARY_WEIGHT(
        inputs=[weight],
        template=[("T", weight.dtype)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, input_dims // 16), (rows, input_dims // group_size)],
        output_dtypes=[mx.uint32, weight.dtype],
    )
    return mx.stop_gradient(packed), mx.stop_gradient(scales), group_size


@mx.custom_function
def ternary_quantized_linear(
    x: mx.array,
    weight: mx.array,
    packed: mx.array,
    scales: mx.array,
) -> mx.array:
    group_size = 64 if weight.shape[-1] % 64 == 0 else 32
    return mx.quantized_matmul(
        x,
        packed,
        scales,
        -scales,
        group_size=group_size,
        bits=2,
    )


@ternary_quantized_linear.vjp
def _ternary_quantized_linear_vjp(primals, cotangent, _output):
    x, weight, packed, scales = primals
    cotangent = cotangent.astype(x.dtype)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_cotangent = cotangent.reshape(-1, cotangent.shape[-1])
    group_size = 64 if weight.shape[-1] % 64 == 0 else 32
    grad_x = mx.quantized_matmul(
        cotangent,
        packed,
        scales,
        -scales,
        transpose=False,
        group_size=group_size,
        bits=2,
    )
    grad_weight = flat_cotangent.T @ flat_x
    return grad_x, grad_weight, mx.zeros_like(packed), mx.zeros_like(scales)


# ---------------------------------------------------------------------------
# Decode-optimized M=1 path: optional absmax act quant + ternary add/sub GEMV
# Pack layout matches pack_ternary_weight: 2 bits/weight, 16 codes per uint32,
# code 0 -> -1, 1 -> 0, 2 -> +1, times per-row (replicated per-group) scale.
# ---------------------------------------------------------------------------

_TERNARY_FUSED_M1 = mx.fast.metal_kernel(
    name="ternary_fused_linear_m1",
    input_names=["x", "packed", "scales", "params"],
    output_names=["y"],
    source=r"""
        // params: [in_dim, out_dim, group_size, words_per_row, groups_per_row,
        //          quantize_acts (0/1), act_levels, neg_levels]
        // Threadgroup cooperatively loads/quantizes x once; each thread owns output rows.
        uint out_dim = uint(params[1]);
        uint in_dim = uint(params[0]);
        uint words_per_row = uint(params[3]);
        uint groups_per_row = uint(params[4]);
        uint quantize_acts = uint(params[5]);
        float act_levels = params[6];
        float neg_levels = params[7];

        uint lane = thread_position_in_threadgroup.x;
        uint tg = threads_per_threadgroup.x;
        uint o0 = thread_position_in_grid.x; // one thread per output when tg maps 1:1

        threadgroup float xq[2048];
        threadgroup float partial[256];
        if (in_dim > 2048u) {
            if (o0 < out_dim) y[o0] = T(0);
            return;
        }

        // --- shared absmax + quantize (or plain load) of x ---
        float local_max = 0.0f;
        for (uint i = lane; i < in_dim; i += tg) {
            local_max = max(local_max, abs(float(x[i])));
        }
        partial[lane] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride && lane + stride < tg) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float amax = max(partial[0], 1e-5f);
        float ascale = amax / max(act_levels, 1.0f);
        if (quantize_acts != 0u) {
            for (uint i = lane; i < in_dim; i += tg) {
                float v = float(x[i]) / ascale;
                v = round(v);
                v = clamp(v, -neg_levels, act_levels);
                xq[i] = v * ascale;
            }
        } else {
            for (uint i = lane; i < in_dim; i += tg) {
                xq[i] = float(x[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- ternary GEMV for this thread's output row ---
        if (o0 >= out_dim) {
            return;
        }
        float wscale = float(scales[o0 * groups_per_row]);
        float sum = 0.0f;
        uint row_base = o0 * words_per_row;
        for (uint word_index = 0; word_index < words_per_row; ++word_index) {
            uint word = packed[row_base + word_index];
            uint base_i = word_index * 16u;
            // Unroll 16 codes
            for (uint offset = 0; offset < 16u; ++offset) {
                uint i = base_i + offset;
                if (i >= in_dim) break;
                uint code = (word >> (2u * offset)) & 3u;
                float xv = xq[i];
                if (code == 2u) sum += xv * wscale;
                else if (code == 0u) sum -= xv * wscale;
            }
        }
        y[o0] = T(sum);
    """,
)


def ternary_fused_linear_m1(
    x: mx.array,
    packed: mx.array,
    scales: mx.array,
    *,
    in_dim: int,
    out_dim: int,
    group_size: int,
    quantize_acts: bool = False,
    act_levels: float = 7.0,
    dtype=None,
) -> mx.array:
    """Fused decode linear for a single token (M=1): optional act quant + ternary GEMV.

    Matches dense ``x @ effective_ternary.T`` for the pack layout of ``pack_ternary_weight``.
    ``x`` may be rank-1 ``(in_dim,)`` or rank-2/3 with leading size 1.
    """
    if dtype is None:
        dtype = x.dtype
    orig_shape = x.shape
    flat = x.reshape(-1, in_dim)
    if int(flat.shape[0]) != 1:
        raise ValueError("ternary_fused_linear_m1 requires batch*seq == 1")
    if in_dim > 4096:
        raise ValueError("ternary_fused_linear_m1 supports in_dim <= 4096")
    if in_dim % 32:
        raise ValueError("in_dim must be divisible by 32")
    words_per_row = in_dim // 16
    groups_per_row = in_dim // group_size
    neg_levels = act_levels + 1.0
    params = mx.array(
        [
            float(in_dim),
            float(out_dim),
            float(group_size),
            float(words_per_row),
            float(groups_per_row),
            1.0 if quantize_acts else 0.0,
            float(act_levels),
            float(neg_levels),
        ],
        dtype=mx.float32,
    )
    tg = 256
    # Pad grid to a multiple of the threadgroup so shared-memory reductions are well-defined.
    grid_x = ((out_dim + tg - 1) // tg) * tg
    y = _TERNARY_FUSED_M1(
        inputs=[flat[0].astype(mx.float32), packed, scales.astype(mx.float32), params],
        template=[("T", mx.float32)],
        grid=(grid_x, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(out_dim,)],
        output_dtypes=[mx.float32],
    )[0]
    # Restore leading singleton dims of x (e.g. (1,1,H) -> (1,1,out))
    out_shape = list(orig_shape[:-1]) + [out_dim]
    return y.reshape(out_shape).astype(dtype)


def ternary_effective_weight(weight: mx.array) -> mx.array:
    """Dense {-s,0,+s} materialization matching pack_ternary_weight thresholds."""
    scale = mx.maximum(mx.mean(mx.abs(weight), axis=-1, keepdims=True), 1e-5)
    normalized = weight / scale
    return mx.where(normalized > 0.5, scale, mx.where(normalized < -0.5, -scale, 0.0))


_TERNARY_FFN_M1 = mx.fast.metal_kernel(
    name="ternary_fused_ffn_m1",
    input_names=[
        "x",
        "up_packed",
        "up_scales",
        "mid_packed",
        "mid_scales",
        "down_packed",
        "down_scales",
        "params",
    ],
    output_names=["y"],
    source=r"""
        // BitNet dense FFN for M=1:
        //   u = tern(up, quant(x)); (g,v)=split(u); h = silu(g)*v;
        //   h = tern(mid, quant(h)); h = silu(h);
        //   y = tern(down, quant(h));
        // params: [hidden, inter, up_words, mid_words, down_words, up_groups, mid_groups, down_groups,
        //          quantize_acts, act_levels, neg_levels]
        uint hidden = uint(params[0]);
        uint inter = uint(params[1]);
        uint up_words = uint(params[2]);
        uint mid_words = uint(params[3]);
        uint down_words = uint(params[4]);
        uint up_groups = uint(params[5]);
        uint mid_groups = uint(params[6]);
        uint down_groups = uint(params[7]);
        uint quantize_acts = uint(params[8]);
        float act_levels = params[9];
        float neg_levels = params[10];

        uint lane = thread_position_in_threadgroup.x;
        uint tg = threads_per_threadgroup.x;

        threadgroup float buf_a[2048];
        threadgroup float buf_b[4096]; // holds up to 2*inter for up output
        threadgroup float partial[256];

        // --- load / quant x into buf_a[0:hidden] ---
        float local_max = 0.0f;
        for (uint i = lane; i < hidden; i += tg) {
            local_max = max(local_max, abs(float(x[i])));
        }
        partial[lane] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float amax = max(partial[0], 1e-5f);
        float ascale = amax / max(act_levels, 1.0f);
        for (uint i = lane; i < hidden; i += tg) {
            float v = float(x[i]);
            if (quantize_acts != 0u) {
                v = round(v / ascale);
                v = clamp(v, -neg_levels, act_levels) * ascale;
            }
            buf_a[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- up GEMV: hidden -> 2*inter into buf_b ---
        uint up_out = inter * 2u;
        for (uint o = lane; o < up_out; o += tg) {
            float wscale = float(up_scales[o * up_groups]);
            float sum = 0.0f;
            uint row_base = o * up_words;
            for (uint wi = 0; wi < up_words; ++wi) {
                uint word = up_packed[row_base + wi];
                uint base_i = wi * 16u;
                for (uint off = 0; off < 16u; ++off) {
                    uint i = base_i + off;
                    if (i >= hidden) break;
                    uint code = (word >> (2u * off)) & 3u;
                    float xv = buf_a[i];
                    if (code == 2u) sum += xv * wscale;
                    else if (code == 0u) sum -= xv * wscale;
                }
            }
            buf_b[o] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- silu(gate)*value into buf_a[0:inter] ---
        for (uint i = lane; i < inter; i += tg) {
            float g = buf_b[i];
            float v = buf_b[i + inter];
            float sig = 1.0f / (1.0f + exp(-g));
            buf_a[i] = (g * sig) * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // quant mid input
        if (quantize_acts != 0u) {
            local_max = 0.0f;
            for (uint i = lane; i < inter; i += tg) local_max = max(local_max, abs(buf_a[i]));
            partial[lane] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride > 0u; stride >>= 1u) {
                if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            amax = max(partial[0], 1e-5f);
            ascale = amax / max(act_levels, 1.0f);
            for (uint i = lane; i < inter; i += tg) {
                float v = buf_a[i] / ascale;
                v = round(v);
                buf_a[i] = clamp(v, -neg_levels, act_levels) * ascale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- mid GEMV: inter -> inter into buf_b ---
        for (uint o = lane; o < inter; o += tg) {
            float wscale = float(mid_scales[o * mid_groups]);
            float sum = 0.0f;
            uint row_base = o * mid_words;
            for (uint wi = 0; wi < mid_words; ++wi) {
                uint word = mid_packed[row_base + wi];
                uint base_i = wi * 16u;
                for (uint off = 0; off < 16u; ++off) {
                    uint i = base_i + off;
                    if (i >= inter) break;
                    uint code = (word >> (2u * off)) & 3u;
                    float xv = buf_a[i];
                    if (code == 2u) sum += xv * wscale;
                    else if (code == 0u) sum -= xv * wscale;
                }
            }
            // silu
            float sig = 1.0f / (1.0f + exp(-sum));
            buf_b[o] = sum * sig;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // quant down input
        if (quantize_acts != 0u) {
            local_max = 0.0f;
            for (uint i = lane; i < inter; i += tg) local_max = max(local_max, abs(buf_b[i]));
            partial[lane] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride > 0u; stride >>= 1u) {
                if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            amax = max(partial[0], 1e-5f);
            ascale = amax / max(act_levels, 1.0f);
            for (uint i = lane; i < inter; i += tg) {
                float v = buf_b[i] / ascale;
                v = round(v);
                buf_b[i] = clamp(v, -neg_levels, act_levels) * ascale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- down GEMV: inter -> hidden (each lane owns a strided set of outputs) ---
        for (uint o = lane; o < hidden; o += tg) {
            float wscale = float(down_scales[o * down_groups]);
            float sum = 0.0f;
            uint row_base = o * down_words;
            for (uint wi = 0; wi < down_words; ++wi) {
                uint word = down_packed[row_base + wi];
                uint base_i = wi * 16u;
                for (uint off = 0; off < 16u; ++off) {
                    uint i = base_i + off;
                    if (i >= inter) break;
                    uint code = (word >> (2u * off)) & 3u;
                    float xv = buf_b[i];
                    if (code == 2u) sum += xv * wscale;
                    else if (code == 0u) sum -= xv * wscale;
                }
            }
            y[o] = T(sum);
        }
    """,
)


def ternary_fused_ffn_m1(
    x: mx.array,
    up_packed,
    up_scales,
    mid_packed,
    mid_scales,
    down_packed,
    down_scales,
    *,
    hidden: int,
    intermediate: int,
    quantize_acts: bool = True,
    act_levels: float = 7.0,
    dtype=None,
) -> mx.array:
    """Fused ternary SwiGLU-mid FFN for a single token (M=1)."""
    if dtype is None:
        dtype = x.dtype
    orig = x.shape
    flat = x.reshape(-1, hidden)
    if int(flat.shape[0]) != 1:
        raise ValueError("ternary_fused_ffn_m1 requires M=1")
    if hidden > 2048 or intermediate * 2 > 4096:
        raise ValueError("FFN fused kernel buffer limit exceeded")
    up_gs = 64 if hidden % 64 == 0 else 32
    mid_gs = 64 if intermediate % 64 == 0 else 32
    down_gs = mid_gs
    params = mx.array(
        [
            float(hidden),
            float(intermediate),
            float(hidden // 16),
            float(intermediate // 16),
            float(intermediate // 16),
            float(hidden // up_gs),
            float(intermediate // mid_gs),
            float(intermediate // down_gs),
            1.0 if quantize_acts else 0.0,
            float(act_levels),
            float(act_levels + 1.0),
        ],
        dtype=mx.float32,
    )
    # Single threadgroup so up/mid shared buffers are computed once.
    tg = 256
    y = _TERNARY_FFN_M1(
        inputs=[
            flat[0].astype(mx.float32),
            up_packed,
            up_scales.astype(mx.float32),
            mid_packed,
            mid_scales.astype(mx.float32),
            down_packed,
            down_scales.astype(mx.float32),
            params,
        ],
        template=[("T", mx.float32)],
        grid=(tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*orig[:-1], hidden).astype(dtype)
