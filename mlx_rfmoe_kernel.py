"""Differentiable sparse grouped linear Metal kernel for MLX RFMoE."""

from __future__ import annotations

import mlx.core as mx


_MASKED_GROUPED_LINEAR = mx.fast.metal_kernel(
    name="rfmoe_masked_grouped_linear",
    input_names=["x", "weight", "active"],
    output_names=["output"],
    source=r"""
        uint output_index = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        uint expert = thread_position_in_grid.z;
        uint tokens = x_shape[1];
        uint input_size = x_shape[2];
        uint output_size = weight_shape[1];
        ulong row = ulong(expert) * tokens + token;
        ulong output_offset = row * output_size + output_index;
        if (!active[row]) {
            output[output_offset] = T(0);
            return;
        }
        ulong input_row = x_shape[0] == 1 ? token : row;
        ulong input_offset = input_row * input_size;
        ulong weight_offset = (ulong(expert) * output_size + output_index) * input_size;
        float value = 0.0f;
        for (uint index = 0; index < input_size; ++index) {
            value += float(x[input_offset + index]) * float(weight[weight_offset + index]);
        }
        output[output_offset] = T(value);
    """,
)

_MASKED_GROUPED_INPUT_GRAD = mx.fast.metal_kernel(
    name="rfmoe_masked_grouped_input_grad",
    input_names=["cotangent", "weight", "active"],
    output_names=["grad_x"],
    source=r"""
        uint input_index = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        uint expert = thread_position_in_grid.z;
        uint tokens = cotangent_shape[1];
        uint output_size = cotangent_shape[2];
        uint input_size = weight_shape[2];
        ulong row = ulong(expert) * tokens + token;
        ulong grad_offset = row * input_size + input_index;
        if (!active[row]) {
            grad_x[grad_offset] = T(0);
            return;
        }
        float value = 0.0f;
        for (uint output_index = 0; output_index < output_size; ++output_index) {
            ulong weight_offset = (ulong(expert) * output_size + output_index) * input_size;
            value += float(cotangent[row * output_size + output_index])
                * float(weight[weight_offset + input_index]);
        }
        grad_x[grad_offset] = T(value);
    """,
)

_MASKED_GROUPED_WEIGHT_GRAD = mx.fast.metal_kernel(
    name="rfmoe_masked_grouped_weight_grad",
    input_names=["x", "cotangent", "active"],
    output_names=["grad_weight"],
    source=r"""
        uint input_index = thread_position_in_grid.x;
        uint output_index = thread_position_in_grid.y;
        uint expert = thread_position_in_grid.z;
        uint tokens = cotangent_shape[1];
        uint output_size = cotangent_shape[2];
        uint input_size = x_shape[2];
        float value = 0.0f;
        for (uint token = 0; token < tokens; ++token) {
            ulong row = ulong(expert) * tokens + token;
            if (active[row]) {
                ulong input_row = x_shape[0] == 1 ? token : row;
                value += float(cotangent[row * output_size + output_index])
                    * float(x[input_row * input_size + input_index]);
            }
        }
        ulong offset = (ulong(expert) * output_size + output_index) * input_size + input_index;
        grad_weight[offset] = T(value);
    """,
)

_COMPACTED_GROUPED_WEIGHT_GRAD = mx.fast.metal_kernel(
    name="rfmoe_compacted_grouped_weight_grad",
    input_names=["x", "cotangent", "expert_offsets", "token_indices"],
    output_names=["grad_weight"],
    source=r"""
        uint input_index = thread_position_in_grid.x;
        uint output_index = thread_position_in_grid.y;
        uint expert = thread_position_in_grid.z;
        uint tokens = cotangent_shape[1];
        uint output_size = cotangent_shape[2];
        uint input_size = x_shape[2];
        float value = 0.0f;
        for (uint route = expert_offsets[expert]; route < expert_offsets[expert + 1]; ++route) {
            uint token = token_indices[route];
            ulong input_row = x_shape[0] == 1 ? token : ulong(expert) * tokens + token;
            ulong cotangent_offset = (ulong(expert) * tokens + token) * output_size + output_index;
            value += float(cotangent[cotangent_offset])
                * float(x[input_row * input_size + input_index]);
        }
        ulong offset = (ulong(expert) * output_size + output_index) * input_size + input_index;
        grad_weight[offset] = T(value);
    """,
)


def _run_masked_grouped_linear(x: mx.array, weight: mx.array, active: mx.array) -> mx.array:
    if x.ndim != 3 or weight.ndim != 3 or active.ndim != 2:
        raise ValueError("expected x [E,T,I], weight [E,O,I], and active [E,T]")
    if x.shape[0] not in (1, weight.shape[0]) or x.shape[1] != active.shape[1]:
        raise ValueError("masked grouped linear shapes do not match")
    if weight.shape[0] != active.shape[0] or x.shape[2] != weight.shape[2]:
        raise ValueError("masked grouped linear shapes do not match")
    if x.dtype != weight.dtype or active.dtype != mx.bool_:
        raise ValueError("inputs and weights must share dtype and active must be bool")
    experts, tokens = active.shape
    outputs = weight.shape[1]
    return _MASKED_GROUPED_LINEAR(
        inputs=[x, weight, active],
        template=[("T", x.dtype)],
        grid=(outputs, tokens, experts),
        threadgroup=(min(outputs, 256), 1, 1),
        output_shapes=[(experts, tokens, outputs)],
        output_dtypes=[x.dtype],
    )[0]


def _run_input_grad(cotangent: mx.array, weight: mx.array, active: mx.array) -> mx.array:
    experts, tokens, outputs = cotangent.shape
    inputs = weight.shape[2]
    return _MASKED_GROUPED_INPUT_GRAD(
        inputs=[cotangent, weight, active],
        template=[("T", cotangent.dtype)],
        grid=(inputs, tokens, experts),
        threadgroup=(min(inputs, 256), 1, 1),
        output_shapes=[(experts, tokens, inputs)],
        output_dtypes=[cotangent.dtype],
    )[0]


def _run_weight_grad(x: mx.array, cotangent: mx.array, active: mx.array) -> mx.array:
    experts, _, outputs = cotangent.shape
    inputs = x.shape[2]
    return _MASKED_GROUPED_WEIGHT_GRAD(
        inputs=[x, cotangent, active],
        template=[("T", x.dtype)],
        grid=(inputs, outputs, experts),
        threadgroup=(min(inputs, 256), 1, 1),
        output_shapes=[(experts, outputs, inputs)],
        output_dtypes=[x.dtype],
    )[0]


def _run_compacted_weight_grad(
    x: mx.array,
    cotangent: mx.array,
    expert_offsets: mx.array,
    token_indices: mx.array,
) -> mx.array:
    experts, _, outputs = cotangent.shape
    inputs = x.shape[2]
    return _COMPACTED_GROUPED_WEIGHT_GRAD(
        inputs=[x, cotangent, expert_offsets, token_indices],
        template=[("T", x.dtype)],
        grid=(inputs, outputs, experts),
        threadgroup=(min(inputs, 256), 1, 1),
        output_shapes=[(experts, outputs, inputs)],
        output_dtypes=[x.dtype],
    )[0]


@mx.custom_function
def masked_grouped_linear(x: mx.array, weight: mx.array, active: mx.array) -> mx.array:
    return _run_masked_grouped_linear(x, weight, active)


@masked_grouped_linear.vjp
def _masked_grouped_linear_vjp(primals, cotangent, _output):
    x, weight, active = primals
    cotangent = cotangent.astype(x.dtype)
    grad_x = _run_input_grad(cotangent, weight, active)
    if x.shape[0] == 1:
        grad_x = mx.sum(grad_x, axis=0, keepdims=True)
    grad_weight = _run_weight_grad(x, cotangent, active)
    return grad_x, grad_weight, mx.zeros_like(active)


def _run_compacted_grouped_linear(
    x: mx.array,
    weight: mx.array,
    active: mx.array,
    expert_indices: mx.array,
    token_indices: mx.array,
) -> mx.array:
    experts, tokens = active.shape
    outputs = weight.shape[1]
    if expert_indices.size == 0:
        return mx.zeros((experts, tokens, outputs), dtype=x.dtype)
    selected_x = x[0, token_indices] if x.shape[0] == 1 else x[expert_indices, token_indices]
    selected = mx.squeeze(
        mx.gather_mm(
            selected_x[:, None, :],
            weight.swapaxes(-1, -2),
            rhs_indices=expert_indices,
        ),
        axis=1,
    )
    return mx.zeros((experts, tokens, outputs), dtype=x.dtype).at[expert_indices, token_indices].add(selected)


@mx.custom_function
def compacted_grouped_linear(
    x: mx.array,
    weight: mx.array,
    active: mx.array,
    expert_indices: mx.array,
    token_indices: mx.array,
    expert_offsets: mx.array,
) -> mx.array:
    """Compact host-selected rows forward; use sparse Metal kernels backward."""
    return _run_compacted_grouped_linear(x, weight, active, expert_indices, token_indices)


@compacted_grouped_linear.vjp
def _compacted_grouped_linear_vjp(primals, cotangent, _output):
    x, weight, active, expert_indices, token_indices, expert_offsets = primals
    if expert_indices.size == 0:
        return tuple(mx.zeros_like(value) for value in primals)
    cotangent = cotangent.astype(x.dtype)
    selected_cotangent = cotangent[expert_indices, token_indices]
    selected_grad_x = mx.squeeze(
        mx.gather_mm(
            selected_cotangent[:, None, :],
            weight,
            rhs_indices=expert_indices,
        ),
        axis=1,
    )
    if x.shape[0] == 1:
        grad_x = mx.zeros_like(x[0]).at[token_indices].add(selected_grad_x)[None]
    else:
        grad_x = mx.zeros_like(x).at[expert_indices, token_indices].add(selected_grad_x)
    grad_weight = _run_compacted_weight_grad(x, cotangent, expert_offsets, token_indices)
    return (
        grad_x,
        grad_weight,
        mx.zeros_like(active),
        mx.zeros_like(expert_indices),
        mx.zeros_like(token_indices),
        mx.zeros_like(expert_offsets),
    )
