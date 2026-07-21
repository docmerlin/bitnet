"""MLX implementation of cautious MUD with 8-bit cautious Lion fallback."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.optimizers as optim


QUANT_BLOCK_SIZE = 2048
MASTER_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}

_RECTANGULAR_LOWER_SOLVE = mx.fast.metal_kernel(
    name="mud_rectangular_lower_solve",
    input_names=["matrix", "rhs"],
    output_names=["solution"],
    source=r"""
        uint column = thread_position_in_grid.x;
        uint rows = matrix_shape[0];
        uint columns = rhs_shape[1];
        if (column >= columns) {
            return;
        }
        for (uint row = 0; row < rows; ++row) {
            float value = float(rhs[row * columns + column]);
            for (uint inner = 0; inner < row; ++inner) {
                value -= float(matrix[row * rows + inner])
                    * float(solution[inner * columns + column]);
            }
            value /= float(matrix[row * rows + row]);
            solution[row * columns + column] = T(value);
        }
    """,
)

_BATCHED_RECTANGULAR_LOWER_SOLVE = mx.fast.metal_kernel(
    name="mud_batched_rectangular_lower_solve",
    input_names=["matrix", "rhs"],
    output_names=["solution"],
    source=r"""
        uint column = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint rows = matrix_shape[1];
        uint columns = rhs_shape[2];
        if (column >= columns || batch >= rhs_shape[0]) {
            return;
        }
        uint matrix_offset = batch * rows * rows;
        uint rhs_offset = batch * rows * columns;
        for (uint row = 0; row < rows; ++row) {
            float value = float(rhs[rhs_offset + row * columns + column]);
            for (uint inner = 0; inner < row; ++inner) {
                value -= float(matrix[matrix_offset + row * rows + inner])
                    * float(solution[rhs_offset + inner * columns + column]);
            }
            value /= float(matrix[matrix_offset + row * rows + row]);
            solution[rhs_offset + row * columns + column] = T(value);
        }
    """,
)


def lower_solve(matrix: mx.array, rhs: mx.array) -> mx.array:
    if rhs.ndim == 3:
        batches, _, columns = rhs.shape
        return _BATCHED_RECTANGULAR_LOWER_SOLVE(
            inputs=[matrix.astype(mx.float32), rhs.astype(mx.float32)],
            template=[("T", mx.float32)],
            grid=(columns, batches, 1),
            threadgroup=(min(columns, 256), 1, 1),
            output_shapes=[rhs.shape],
            output_dtypes=[mx.float32],
        )[0]
    columns = rhs.shape[1]
    return _RECTANGULAR_LOWER_SOLVE(
        inputs=[matrix.astype(mx.float32), rhs.astype(mx.float32)],
        template=[("T", mx.float32)],
        grid=(columns, 1, 1),
        threadgroup=(min(columns, 256), 1, 1),
        output_shapes=[rhs.shape],
        output_dtypes=[mx.float32],
    )[0]


def mud_decorrelate(
    update: mx.array,
    passes: int = 1,
    eps: float = 1e-8,
    block_size: int | None = None,
) -> mx.array:
    if update.ndim != 2:
        raise ValueError("mud_decorrelate expects a 2D matrix")
    if passes < 1:
        raise ValueError("passes must be positive")
    if block_size is not None and block_size < 1:
        raise ValueError("block_size must be positive")
    original_dtype = update.dtype
    q = update.astype(mx.float32)
    transposed = q.shape[0] > q.shape[1]
    if transposed:
        q = q.T
    block_size = q.shape[0] if block_size is None else block_size
    if q.shape[0] > block_size and q.shape[0] % block_size == 0:
        blocks = q.reshape(-1, block_size, q.shape[1])
        for _ in range(passes):
            blocks = blocks / (mx.linalg.norm(blocks, axis=2, keepdims=True) + eps)
            triangle = mx.tril(blocks @ blocks.swapaxes(-1, -2)) + eps * mx.eye(block_size)
            blocks = lower_solve(triangle, blocks)
            blocks = blocks / (mx.linalg.norm(blocks, axis=2, keepdims=True) + eps)
        q = blocks.reshape(q.shape)
        return (q.T if transposed else q).astype(original_dtype)
    blocks = []
    for start in range(0, q.shape[0], block_size):
        block = q[start : start + block_size]
        for _ in range(passes):
            block = block / (mx.linalg.norm(block, axis=1, keepdims=True) + eps)
            triangle = mx.tril(block @ block.T) + eps * mx.eye(block.shape[0])
            block = lower_solve(triangle, block)
            block = block / (mx.linalg.norm(block, axis=1, keepdims=True) + eps)
        blocks.append(block)
    q = mx.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
    return (q.T if transposed else q).astype(original_dtype)


def cautious_mask(update: mx.array, gradient: mx.array) -> mx.array:
    mask = (update * gradient > 0).astype(update.dtype)
    scale = update.size / mx.maximum(mx.sum(mask), 1.0)
    return update * mask * scale


def quantize_blockwise(tensor: mx.array, block_size: int = QUANT_BLOCK_SIZE):
    flat = tensor.astype(mx.float32).reshape(-1)
    padding = (-flat.size) % block_size
    if padding:
        flat = mx.concatenate((flat, mx.zeros((padding,), dtype=flat.dtype)))
    blocks = flat.reshape(-1, block_size)
    scale = mx.maximum(mx.max(mx.abs(blocks), axis=1, keepdims=True), 1e-8) / 127.0
    quantized = mx.clip(mx.round(blocks / scale), -127, 127).astype(mx.int8)
    return quantized, mx.squeeze(scale, axis=1)


def dequantize_blockwise(quantized: mx.array, scale: mx.array, shape: tuple[int, ...]):
    size = math.prod(shape)
    return (quantized.astype(mx.float32) * scale[:, None]).reshape(-1)[:size].reshape(shape)


class MUD(optim.Optimizer):
    def __init__(
        self,
        learning_rate: float,
        momentum: float = 0.95,
        passes: int = 1,
        weight_decay: float = 0.0,
        block_size: int | None = None,
        eight_bit: bool = False,
        master_dtype: str = "float32",
    ):
        super().__init__()
        if master_dtype not in MASTER_DTYPES:
            raise ValueError(f"Unsupported master dtype: {master_dtype}")
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.passes = passes
        self.weight_decay = weight_decay
        self.block_size = block_size
        self.eight_bit = eight_bit
        self.master_dtype = master_dtype

    def init_single(self, parameter: mx.array, state: dict):
        if self.eight_bit and parameter.size >= QUANT_BLOCK_SIZE:
            blocks = (parameter.size + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
            state["momentum_buffer_q"] = mx.zeros((blocks, QUANT_BLOCK_SIZE), dtype=mx.int8)
            state["momentum_buffer_scale"] = mx.zeros((blocks,), dtype=mx.float32)
        else:
            state["momentum_buffer"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["master_parameter"] = parameter.astype(MASTER_DTYPES[self.master_dtype])

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        parameter_dtype = parameter.dtype
        gradient = gradient.astype(mx.float32)
        use_eight_bit = self.eight_bit and parameter.size >= QUANT_BLOCK_SIZE
        if "momentum_buffer_q" in state:
            previous = dequantize_blockwise(
                state["momentum_buffer_q"],
                state["momentum_buffer_scale"],
                gradient.shape,
            )
        elif "momentum_buffer" in state:
            previous = state["momentum_buffer"]
        else:
            previous = mx.zeros(parameter.shape, dtype=mx.float32)
        momentum_buffer = self.momentum * previous + gradient
        direction = gradient + self.momentum * momentum_buffer
        update = mud_decorrelate(direction, self.passes, block_size=self.block_size)
        update = update * (0.2 * math.sqrt(max(parameter.shape)))
        update = cautious_mask(update, gradient)
        if use_eight_bit:
            state["momentum_buffer_q"], state["momentum_buffer_scale"] = quantize_blockwise(momentum_buffer)
            state.pop("momentum_buffer", None)
        else:
            state["momentum_buffer"] = momentum_buffer
            state.pop("momentum_buffer_q", None)
            state.pop("momentum_buffer_scale", None)
        learning_rate = self.learning_rate.astype(mx.float32)
        master_parameter = state.get("master_parameter", parameter.astype(MASTER_DTYPES[self.master_dtype]))
        master_parameter = (
            master_parameter * (1.0 - learning_rate * self.weight_decay) - learning_rate * update
        ).astype(MASTER_DTYPES[self.master_dtype])
        state["master_parameter"] = master_parameter
        return master_parameter.astype(parameter_dtype)


class CLion(optim.Optimizer):
    def __init__(
        self,
        learning_rate: float,
        betas: tuple[float, float] = (0.95, 0.98),
        eight_bit: bool = True,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.beta1, self.beta2 = betas
        self.eight_bit = eight_bit

    def init_single(self, parameter: mx.array, state: dict):
        state["master_parameter"] = parameter.astype(mx.float32)
        if self.eight_bit and parameter.size >= QUANT_BLOCK_SIZE:
            blocks = (parameter.size + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
            state["exp_avg_q"] = mx.zeros((blocks, QUANT_BLOCK_SIZE), dtype=mx.int8)
            state["exp_avg_scale"] = mx.zeros((blocks,), dtype=mx.float32)
        else:
            state["exp_avg"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        gradient = gradient.astype(mx.float32)
        use_eight_bit = self.eight_bit and parameter.size >= QUANT_BLOCK_SIZE
        if "exp_avg_q" in state:
            momentum = dequantize_blockwise(
                state["exp_avg_q"],
                state["exp_avg_scale"],
                gradient.shape,
            )
        elif "exp_avg" in state:
            momentum = state["exp_avg"]
        else:
            momentum = mx.zeros(parameter.shape, dtype=mx.float32)
        update = mx.sign(self.beta1 * momentum + (1.0 - self.beta1) * gradient)
        update = cautious_mask(update, gradient)
        momentum = self.beta2 * momentum + (1.0 - self.beta2) * gradient
        if use_eight_bit:
            state["exp_avg_q"], state["exp_avg_scale"] = quantize_blockwise(momentum)
            state.pop("exp_avg", None)
        else:
            state["exp_avg"] = momentum
            state.pop("exp_avg_q", None)
            state.pop("exp_avg_scale", None)
        master_parameter = state.get("master_parameter", parameter.astype(mx.float32))
        master_parameter = (
            master_parameter - self.learning_rate.astype(mx.float32) * update
        ).astype(mx.float32)
        state["master_parameter"] = master_parameter
        return master_parameter.astype(parameter.dtype)


class CMUD(optim.MultiOptimizer):
    def __init__(
        self,
        *,
        mud_learning_rate: float,
        fallback_learning_rate: float,
        weight_decay: float,
        momentum: float = 0.95,
        passes: int = 1,
        betas: tuple[float, float] = (0.95, 0.98),
        eight_bit: bool = True,
        mud_eight_bit: bool = False,
        block_size: int | None = None,
        mud_master_dtype: str = "float32",
    ):
        self.mud_learning_rate = mud_learning_rate
        self.fallback_learning_rate = fallback_learning_rate
        mud = MUD(
            mud_learning_rate,
            momentum,
            passes,
            weight_decay,
            block_size,
            mud_eight_bit,
            mud_master_dtype,
        )
        clion = CLion(fallback_learning_rate, betas, eight_bit)
        super().__init__([mud, clion], [self._is_mud_parameter])

    @staticmethod
    def _is_mud_parameter(path: str, parameter: mx.array) -> bool:
        embedding = path.endswith("embedding.weight") or path.endswith("loop_embed.weight")
        depthwise_conv = path.endswith("short_conv_weight")
        return parameter.ndim == 2 and not embedding and not depthwise_conv

    def set_lr_multiplier(self, multiplier: float) -> None:
        self.optimizers[0].learning_rate = self.mud_learning_rate * multiplier
        self.optimizers[1].learning_rate = self.fallback_learning_rate * multiplier

    def checkpoint_config(self) -> dict:
        mud, clion = self.optimizers
        return {
            "mud_learning_rate": self.mud_learning_rate,
            "fallback_learning_rate": self.fallback_learning_rate,
            "weight_decay": mud.weight_decay,
            "momentum": mud.momentum,
            "passes": mud.passes,
            "block_size": mud.block_size,
            "betas": [clion.beta1, clion.beta2],
            "eight_bit": clion.eight_bit,
            "mud_eight_bit": mud.eight_bit,
            "mud_master_dtype": mud.master_dtype,
        }
