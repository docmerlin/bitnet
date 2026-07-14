"""Custom MLX Metal triangular solve used by PaTH-FoX."""

from __future__ import annotations

import mlx.core as mx


_LOWER_SOLVE = mx.fast.metal_kernel(
    name="path_lower_triangular_solve",
    input_names=["matrix", "rhs"],
    output_names=["solution"],
    source=r"""
        uint column = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint size = matrix_shape[matrix_ndim - 1];
        if (column >= size) {
            return;
        }
        ulong base = ulong(batch) * size * size;
        for (uint row = 0; row < size; ++row) {
            float value = float(rhs[base + row * size + column]);
            for (uint inner = 0; inner < row; ++inner) {
                value -= float(matrix[base + row * size + inner])
                    * float(solution[base + inner * size + column]);
            }
            value /= float(matrix[base + row * size + row]);
            solution[base + row * size + column] = T(value);
        }
    """,
)

_UPPER_TRANSPOSE_SOLVE = mx.fast.metal_kernel(
    name="path_upper_transpose_triangular_solve",
    input_names=["matrix", "rhs"],
    output_names=["solution"],
    source=r"""
        uint column = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint size = matrix_shape[matrix_ndim - 1];
        if (column >= size) {
            return;
        }
        ulong base = ulong(batch) * size * size;
        for (int row = int(size) - 1; row >= 0; --row) {
            float value = float(rhs[base + uint(row) * size + column]);
            for (uint inner = uint(row) + 1; inner < size; ++inner) {
                value -= float(matrix[base + inner * size + uint(row)])
                    * float(solution[base + inner * size + column]);
            }
            value /= float(matrix[base + uint(row) * size + uint(row)]);
            solution[base + uint(row) * size + column] = T(value);
        }
    """,
)


def _run_kernel(kernel, matrix: mx.array, rhs: mx.array) -> mx.array:
    if matrix.dtype != mx.float32 or rhs.dtype != mx.float32:
        raise ValueError("PaTH Metal solve requires float32 inputs")
    if matrix.shape != rhs.shape or matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix and rhs must have matching [..., N, N] shapes")
    size = matrix.shape[-1]
    batches = matrix.size // (size * size)
    return kernel(
        inputs=[matrix, rhs],
        template=[("T", mx.float32)],
        grid=(size, batches, 1),
        threadgroup=(min(size, 256), 1, 1),
        output_shapes=[matrix.shape],
        output_dtypes=[mx.float32],
    )[0]


@mx.custom_function
def path_triangular_solve(matrix: mx.array, rhs: mx.array) -> mx.array:
    return _run_kernel(_LOWER_SOLVE, matrix, rhs)


@path_triangular_solve.vjp
def _path_triangular_solve_vjp(primals, cotangent, output):
    matrix, _ = primals
    grad_rhs = _run_kernel(_UPPER_TRANSPOSE_SOLVE, matrix, cotangent)
    grad_matrix = mx.tril(-(grad_rhs @ output.swapaxes(-1, -2)))
    return grad_matrix, grad_rhs


def reference_triangular_solve(matrix: mx.array, rhs: mx.array) -> mx.array:
    """Differentiable MLX fallback used for parity tests."""
    rows = []
    for index in range(matrix.shape[-1]):
        row = rhs[..., index : index + 1, :]
        if rows:
            previous = mx.stack(rows, axis=-2)
            row = row - matrix[..., index : index + 1, :index] @ previous
        row = row / matrix[..., index : index + 1, index : index + 1]
        rows.append(mx.squeeze(row, axis=-2))
    return mx.stack(rows, axis=-2)
