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
