"""Packed 2-bit ternary linear primitive backed by MLX Metal kernels."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# Bound at import: caching a kernel's dispatch is separate from compiling a decode
# step, which callers (and tests) may patch mx.compile to control.
_compile_dispatch = mx.compile


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
# Decode-optimized M=1 path: ternary GEMV, optionally fused with MLXHBitLinear's
# input preparation (Hadamard + fp8-e4m3 round trip) so a linear is one dispatch.
# Pack layout matches pack_ternary_weight: 2 bits/weight, 16 codes per uint32,
# code 0 -> -1, 1 -> 0, 2 -> +1, times per-row (replicated per-group) scale.
# ---------------------------------------------------------------------------

# Each simdgroup owns _M1_ROWS output rows; its 32 lanes stride over a row's packed
# words (coalesced loads) and combine partial sums with simd_sum. Shapes are
# template constants, so each (in_dim, out_dim) compiles once with fixed loop bounds.
# Eight simdgroups per threadgroup: fused prep is redone per threadgroup, so fewer,
# wider threadgroups cut that redundancy and spread each copy over 256 threads.
_M1_ROWS = 8
_M1_SIMDGROUPS = 8
# Fused prep stages x in threadgroup memory: 4096 floats = 16KB of the 32KB budget.
M1_PREP_MAX_DIM = 4096

_M1_HEADER = r"""
// MLX's fp8_e4m3 (PyTorch Float8_e4m3fn, saturating) encode then decode, as
// mx.from_fp8(mx.to_fp8(x)) computes it.
inline float fp8_e4m3_roundtrip(float f) {
    uint32_t f_bits = as_type<uint32_t>(f);
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    uint8_t bits;
    if (f_bits >= (543u << 21)) {
        bits = 0x7E;
    } else if (f_bits < (121u << 23)) {
        uint32_t denorm_mask = 141u << 23;
        f_bits = as_type<uint32_t>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = uint8_t(f_bits - denorm_mask);
    } else {
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = uint8_t(f_bits >> 20);
    }
    half converted = as_type<half>(uint16_t((bits & 127) << 7)) * half(256.0);
    return sign ? -float(converted) : float(converted);
}
"""

_TERNARY_FUSED_M1 = mx.fast.metal_kernel(
    name="ternary_fused_linear_m1",
    input_names=["x", "packed", "scales", "norm_weight", "norm_eps"],
    output_names=["y"],
    header=_M1_HEADER,
    source=r"""
        constexpr uint WORDS = IN_DIM / 16;
        constexpr uint GROUPS = IN_DIM / GROUP_SIZE;
        constexpr uint THREADS = 32 * SIMDGROUPS;
        // EPILOGUE 2 (swiglu) writes silu(row p) * row (p + P) for p < P = OUT_DIM / 2,
        // so each simdgroup computes ROWS / 2 gate rows and their value rows.
        constexpr uint P = EPILOGUE == 2 ? OUT_DIM / 2 : OUT_DIM;
        constexpr uint UNITS = EPILOGUE == 2 ? ROWS / 2 : ROWS;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint tid = sg * 32 + lane;
        // No early exit for rows past the end: every thread must reach the prep
        // barriers. Those rows clamp their loads and skip their writes instead.
        uint unit0 = (threadgroup_position_in_grid.x * SIMDGROUPS + sg) * UNITS;
        uint rows[ROWS];
        for (uint r = 0; r < ROWS; ++r) {
            rows[r] = EPILOGUE == 2
                ? (r < UNITS ? min(unit0 + r, P - 1) : P + min(unit0 + r - UNITS, P - 1))
                : min(unit0 + r, uint(OUT_DIM - 1));
        }

        // Every threadgroup redoes the prep of x: a few thousand flops, far cheaper
        // than the separate dispatches it replaces.
        threadgroup float xs[PREP ? IN_DIM : 1];
        threadgroup float red[SIMDGROUPS];
        if (PREP) {
            for (uint i = tid; i < IN_DIM; i += THREADS) {
                xs[i] = float(x[i]);
            }
            if (NORM) {
                // RMSNorm prologue, as mx.fast.rms_norm: fp32 math, output in XT.
                float ss = 0.0f;
                for (uint i = tid; i < IN_DIM; i += THREADS) {
                    ss += xs[i] * xs[i];
                }
                ss = simd_sum(ss);
                if (lane == 0) {
                    red[sg] = ss;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float total = 0.0f;
                for (uint j = 0; j < SIMDGROUPS; ++j) {
                    total += red[j];
                }
                float inv = rsqrt(total / float(IN_DIM) + norm_eps[0]);
                for (uint i = tid; i < IN_DIM; i += THREADS) {
                    xs[i] = float(XT(xs[i] * inv * float(norm_weight[i])));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (HADAMARD) {
                // Sylvester-order FWHT, as mx.hadamard_transform (scale 1/sqrt(n)).
                for (uint h = 1; h < IN_DIM; h <<= 1) {
                    for (uint b = tid; b < IN_DIM / 2; b += THREADS) {
                        uint k = b & (h - 1);
                        uint i = ((b - k) << 1) + k;
                        float a = xs[i];
                        float c = xs[i + h];
                        xs[i] = a + c;
                        xs[i + h] = a - c;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint i = tid; i < IN_DIM; i += THREADS) {
                float v = HADAMARD ? xs[i] * rsqrt(float(IN_DIM)) : xs[i];
                // Round to the activation dtype first, like hadamard_transform's output.
                xs[i] = fp8_e4m3_roundtrip(float(XT(v)));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // sum((code - 1) * x) == sum(code * x) - sum(x): the -1 leaves the inner loop.
        // xv[i] is pre-scaled by 4^-i so (word & (3 << 2i)) * xv[i] == code_i * x_i
        // without a shift per weight (exact: power-of-two scaling).
        float acc[ROWS] = {0.0f};
        float xsum = 0.0f;
        for (uint w = lane; w < WORDS; w += 32) {
            float xv[16];
            for (uint i = 0; i < 16; ++i) {
                float v = PREP ? xs[w * 16 + i] : float(x[w * 16 + i]);
                xsum += v;
                xv[i] = v * (1.0f / float(1u << (2 * i)));
            }
            for (uint r = 0; r < ROWS; ++r) {
                uint word = packed[rows[r] * WORDS + w];
                float s = 0.0f;
                for (uint i = 0; i < 16; ++i) {
                    s = fma(xv[i], float(word & (3u << (2 * i))), s);
                }
                acc[r] += s;
            }
        }
        xsum = simd_sum(xsum);
        float val[ROWS];
        for (uint r = 0; r < ROWS; ++r) {
            val[r] = (simd_sum(acc[r]) - xsum) * float(scales[rows[r] * GROUPS]);
        }
        if (lane == 0) {
            for (uint r = 0; r < UNITS; ++r) {
                uint o = unit0 + r;
                if (o < P) {
                    float v = val[r];
                    if (EPILOGUE == 1) {
                        v = v / (1.0f + exp(-v));
                    } else if (EPILOGUE == 2) {
                        v = v / (1.0f + exp(-v)) * val[r + UNITS];
                    }
                    y[o] = T(v);
                }
            }
        }
    """,
)

_EPILOGUES = {None: 0, "silu": 1, "swiglu": 2}


@lru_cache(maxsize=None)
def _scalar(value: float) -> mx.array:
    return mx.array([value], dtype=mx.float32)


# A raw metal_kernel call costs ~10us of Python per dispatch; replaying a compiled
# graph costs ~1.5us. One entry per distinct layer shape, so the cache stays tiny.
@lru_cache(maxsize=None)
def _compiled_m1(
    in_dim: int,
    out_dim: int,
    group_size: int,
    x_dtype,
    dtype,
    prepare: bool,
    hadamard: bool,
    norm: bool,
    epilogue: int,
):
    outputs = out_dim // 2 if epilogue == 2 else out_dim
    units_per_tg = (_M1_ROWS // 2 if epilogue == 2 else _M1_ROWS) * _M1_SIMDGROUPS
    threadgroups = (outputs + units_per_tg - 1) // units_per_tg

    def dispatch(x, packed, scales, norm_weight, norm_eps):
        return _TERNARY_FUSED_M1(
            inputs=[x, packed, scales, norm_weight, norm_eps],
            template=[
                ("T", dtype),
                ("XT", x_dtype),
                ("IN_DIM", in_dim),
                ("OUT_DIM", out_dim),
                ("GROUP_SIZE", group_size),
                ("ROWS", _M1_ROWS),
                ("SIMDGROUPS", _M1_SIMDGROUPS),
                ("PREP", prepare),
                ("HADAMARD", hadamard),
                ("NORM", norm),
                ("EPILOGUE", epilogue),
            ],
            grid=(threadgroups * 32 * _M1_SIMDGROUPS, 1, 1),
            threadgroup=(32 * _M1_SIMDGROUPS, 1, 1),
            output_shapes=[(outputs,)],
            output_dtypes=[dtype],
        )[0]

    return _compile_dispatch(dispatch)


def ternary_fused_linear_m1(
    x: mx.array,
    packed: mx.array,
    scales: mx.array,
    *,
    in_dim: int,
    out_dim: int,
    group_size: int,
    dtype=None,
    prepare: bool = False,
    hadamard: bool = False,
    norm_weight: mx.array | None = None,
    norm_eps: float = 0.0,
    epilogue: str | None = None,
) -> mx.array:
    """Fused decode linear for a single token (M=1): ternary GEMV.

    Matches dense ``x @ effective_ternary.T`` for the pack layout of ``pack_ternary_weight``.
    ``x`` may be rank-1 ``(in_dim,)`` or rank-2/3 with leading size 1.

    ``prepare=True`` takes raw activations and applies MLXHBitLinear's input prep in
    the kernel: Hadamard (if ``hadamard``, in_dim a power of two) then the fp8-e4m3
    round trip. Inference only: the kernel has no VJP, so no STE is needed.
    ``norm_weight`` adds an RMSNorm (weight, ``norm_eps``) before that prep.
    ``epilogue``: ``"silu"`` applies silu to the output; ``"swiglu"`` returns
    ``silu(y[:P]) * y[P:]`` with ``P = out_dim // 2``.
    """
    if dtype is None:
        dtype = x.dtype
    orig_shape = x.shape
    flat = x.reshape(-1, in_dim)
    if int(flat.shape[0]) != 1:
        raise ValueError("ternary_fused_linear_m1 requires batch*seq == 1")
    if in_dim % 32:
        raise ValueError("in_dim must be divisible by 32")
    if prepare and in_dim > M1_PREP_MAX_DIM:
        raise ValueError(f"fused prepare supports in_dim <= {M1_PREP_MAX_DIM}")
    if hadamard and (not prepare or in_dim & (in_dim - 1)):
        raise ValueError("hadamard requires prepare=True and a power-of-two in_dim")
    if norm_weight is not None and not prepare:
        raise ValueError("norm_weight requires prepare=True")
    if epilogue == "swiglu" and out_dim % 2:
        raise ValueError("swiglu epilogue requires an even out_dim")
    norm = norm_weight is not None
    kernel = _compiled_m1(in_dim, out_dim, group_size, x.dtype, dtype, prepare, hadamard, norm, _EPILOGUES[epilogue])
    # Unused inputs still need an array in their slot.
    y = kernel(flat, packed, scales, norm_weight if norm else scales, _scalar(norm_eps))
    outputs = out_dim // 2 if epilogue == "swiglu" else out_dim
    # Restore leading singleton dims of x (e.g. (1,1,H) -> (1,1,out))
    return y.reshape(*orig_shape[:-1], outputs)


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
        //   u = tern(up, x); (g,v)=split(u); h = silu(g)*v;
        //   h = silu(tern(mid, h));
        //   y = tern(down, h);
        // params: [hidden, inter, up_words, mid_words, down_words, up_groups, mid_groups, down_groups]
        uint hidden = uint(params[0]);
        uint inter = uint(params[1]);
        uint up_words = uint(params[2]);
        uint mid_words = uint(params[3]);
        uint down_words = uint(params[4]);
        uint up_groups = uint(params[5]);
        uint mid_groups = uint(params[6]);
        uint down_groups = uint(params[7]);

        uint lane = thread_position_in_threadgroup.x;
        uint tg = threads_per_threadgroup.x;

        threadgroup float buf_a[2048];
        threadgroup float buf_b[4096]; // holds up to 2*inter for up output

        for (uint i = lane; i < hidden; i += tg) {
            buf_a[i] = float(x[i]);
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
                    float w = float((word >> (2u * off)) & 3u) - 1.0f;
                    sum = fma(buf_a[base_i + off], w, sum);
                }
            }
            buf_b[o] = sum * wscale;
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

        // --- mid GEMV: inter -> inter into buf_b ---
        for (uint o = lane; o < inter; o += tg) {
            float wscale = float(mid_scales[o * mid_groups]);
            float sum = 0.0f;
            uint row_base = o * mid_words;
            for (uint wi = 0; wi < mid_words; ++wi) {
                uint word = mid_packed[row_base + wi];
                uint base_i = wi * 16u;
                for (uint off = 0; off < 16u; ++off) {
                    float w = float((word >> (2u * off)) & 3u) - 1.0f;
                    sum = fma(buf_a[base_i + off], w, sum);
                }
            }
            sum *= wscale;
            // silu
            float sig = 1.0f / (1.0f + exp(-sum));
            buf_b[o] = sum * sig;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- down GEMV: inter -> hidden (each lane owns a strided set of outputs) ---
        for (uint o = lane; o < hidden; o += tg) {
            float wscale = float(down_scales[o * down_groups]);
            float sum = 0.0f;
            uint row_base = o * down_words;
            for (uint wi = 0; wi < down_words; ++wi) {
                uint word = down_packed[row_base + wi];
                uint base_i = wi * 16u;
                for (uint off = 0; off < 16u; ++off) {
                    float w = float((word >> (2u * off)) & 3u) - 1.0f;
                    sum = fma(buf_b[base_i + off], w, sum);
                }
            }
            y[o] = T(sum * wscale);
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
