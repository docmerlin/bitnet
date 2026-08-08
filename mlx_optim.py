"""MLX implementation of cautious MUD with 8-bit cautious Lion fallback."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.optimizers as optim


QUANT_BLOCK_SIZE = 2048
MASTER_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}

_RECTANGULAR_LOWER_SOLVE = mx.fast.metal_kernel(
    name="mud_rectangular_lower_solve",
    input_names=["matrix", "rhs", "diagonal_epsilon"],
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
            value /= float(matrix[row * rows + row]) + float(diagonal_epsilon);
            solution[row * columns + column] = T(value);
        }
    """,
)

_BATCHED_RECTANGULAR_LOWER_SOLVE = mx.fast.metal_kernel(
    name="mud_batched_rectangular_lower_solve",
    input_names=["matrix", "rhs", "diagonal_epsilon"],
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
            value /= float(matrix[matrix_offset + row * rows + row]) + float(diagonal_epsilon);
            solution[rhs_offset + row * columns + column] = T(value);
        }
    """,
)

_FUSED_MUD_MOMENTUM = mx.fast.metal_kernel(
    name="fused_mud_momentum",
    input_names=["gradient", "previous_q", "previous_scale", "momentum_value"],
    output_names=["direction", "next_q", "next_scale"],
    source=r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = thread_position_in_grid.y;
        uint size = gradient_shape[0] * gradient_shape[1];
        float momentum = float(momentum_value);
        ulong base = ulong(block) * BLOCK_SIZE;

        threadgroup float values[BLOCK_SIZE];
        threadgroup float maxima[256];
        float local_max = 0.0f;
        for (uint offset = lane; offset < BLOCK_SIZE; offset += 256) {
            ulong index = base + offset;
            // Volatile preserves separate-operation rounding instead of contracting
            // into FMAs, keeping optimizer state bit-identical to the native path.
            volatile float previous = index < size
                ? float(previous_q[index]) * previous_scale[block]
                : 0.0f;
            volatile float value = momentum * previous;
            value = index < size ? value + float(gradient[index]) : 0.0f;
            values[offset] = value;
            local_max = max(local_max, abs(value));
        }
        maxima[lane] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) maxima[lane] = max(maxima[lane], maxima[lane + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float scale = max(maxima[0], 1e-8f) / 127.0f;
        if (lane == 0) next_scale[block] = scale;
        for (uint offset = lane; offset < BLOCK_SIZE; offset += 256) {
            ulong index = base + offset;
            float value = values[offset];
            next_q[index] = char(clamp(rint(value / scale), -127.0f, 127.0f));
            volatile float decayed = momentum * value;
            if (index < size) direction[index] = float(gradient[index]) + decayed;
        }
    """,
)


def lower_solve(matrix: mx.array, rhs: mx.array, diagonal_epsilon: float = 0.0) -> mx.array:
    epsilon = mx.array(diagonal_epsilon, dtype=mx.float32)
    if rhs.ndim == 3:
        batches, _, columns = rhs.shape
        return _BATCHED_RECTANGULAR_LOWER_SOLVE(
            inputs=[matrix.astype(mx.float32), rhs.astype(mx.float32), epsilon],
            template=[("T", mx.float32)],
            grid=(columns, batches, 1),
            threadgroup=(min(columns, 256), 1, 1),
            output_shapes=[rhs.shape],
            output_dtypes=[mx.float32],
        )[0]
    columns = rhs.shape[1]
    return _RECTANGULAR_LOWER_SOLVE(
        inputs=[matrix.astype(mx.float32), rhs.astype(mx.float32), epsilon],
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
    """Triangular whitening of a matrix-valued momentum update.

    Tall matrices are transposed first so the cheaper axis is whitened. That
    leaves the neuron (output) axis uneven for ``[out, in]`` with ``out > in``;
    a stateless per-neuron re-normalise was tried (``mud-neuron-norm``) and
    regressed val CE on a small A/B, so it was removed rather than left as a
    dead flag.
    """
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

    def finish(whitened: mx.array) -> mx.array:
        result = whitened.T if transposed else whitened
        return result.astype(original_dtype)

    block_size = q.shape[0] if block_size is None else block_size
    if q.shape[0] > block_size and q.shape[0] % block_size == 0:
        blocks = q.reshape(-1, block_size, q.shape[1])
        for _ in range(passes):
            blocks = blocks / (mx.linalg.norm(blocks, axis=2, keepdims=True) + eps)
            gram = blocks @ blocks.swapaxes(-1, -2)
            blocks = lower_solve(gram, blocks, eps)
            blocks = blocks / (mx.linalg.norm(blocks, axis=2, keepdims=True) + eps)
        return finish(blocks.reshape(q.shape))
    blocks = []
    for start in range(0, q.shape[0], block_size):
        block = q[start : start + block_size]
        for _ in range(passes):
            block = block / (mx.linalg.norm(block, axis=1, keepdims=True) + eps)
            gram = block @ block.T
            block = lower_solve(gram, block, eps)
            block = block / (mx.linalg.norm(block, axis=1, keepdims=True) + eps)
        blocks.append(block)
    return finish(mx.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0])


def cautious_mask(update: mx.array, gradient: mx.array) -> mx.array:
    mask = (update * gradient > 0).astype(update.dtype)
    scale = update.size / mx.maximum(mx.sum(mask), 1.0)
    return update * mask * scale


def cautious_decay(update: mx.array, parameter: mx.array) -> mx.array:
    """Coordinates where decay would fight the update, as a 0/1 mask.

    Cautious Weight Decay (modded-nanogpt records 43 and 50): only shrink a
    coordinate when the step is already moving it toward zero. Note the mask is
    update-vs-*parameter*, not the update-vs-gradient sign test in
    :func:`cautious_mask` -- different question, so it needs its own mask.

    ``update`` is the descent direction as applied, i.e. the parameter moves by
    ``-lr * update``, so ``update * parameter > 0`` means the step is already
    shrinking that coordinate.
    """
    return (update * parameter > 0).astype(update.dtype)


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


def _fused_mud_momentum(
    gradient: mx.array,
    previous_q: mx.array,
    previous_scale: mx.array,
    momentum: float,
):
    blocks = previous_q.shape[0]
    return _FUSED_MUD_MOMENTUM(
        inputs=[gradient, previous_q, previous_scale, mx.array(momentum, dtype=mx.float32)],
        template=[("BLOCK_SIZE", QUANT_BLOCK_SIZE)],
        grid=(256, blocks, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gradient.shape, previous_q.shape, previous_scale.shape],
        output_dtypes=[mx.float32, mx.int8, mx.float32],
    )


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
        cautious_weight_decay: bool = True,
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
        self.cautious_weight_decay = cautious_weight_decay

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
        fused_state = None
        if use_eight_bit and "momentum_buffer_q" in state:
            direction, next_q, next_scale = _fused_mud_momentum(
                gradient,
                state["momentum_buffer_q"],
                state["momentum_buffer_scale"],
                self.momentum,
            )
            fused_state = next_q, next_scale
        elif "momentum_buffer_q" in state:
            previous = dequantize_blockwise(
                state["momentum_buffer_q"],
                state["momentum_buffer_scale"],
                gradient.shape,
            )
        elif "momentum_buffer" in state:
            previous = state["momentum_buffer"]
        else:
            previous = mx.zeros(parameter.shape, dtype=mx.float32)
        if fused_state is None:
            momentum_buffer = self.momentum * previous + gradient
            direction = gradient + self.momentum * momentum_buffer
        update = mud_decorrelate(direction, self.passes, block_size=self.block_size)
        update = update * (0.2 * math.sqrt(max(parameter.shape)))
        update = cautious_mask(update, gradient)
        if fused_state is not None:
            state["momentum_buffer_q"], state["momentum_buffer_scale"] = fused_state
            state.pop("momentum_buffer", None)
        elif use_eight_bit:
            state["momentum_buffer_q"], state["momentum_buffer_scale"] = quantize_blockwise(momentum_buffer)
            state.pop("momentum_buffer", None)
        else:
            state["momentum_buffer"] = momentum_buffer
            state.pop("momentum_buffer_q", None)
            state.pop("momentum_buffer_scale", None)
        learning_rate = self.learning_rate.astype(mx.float32)
        master_parameter = state.get("master_parameter", parameter.astype(MASTER_DTYPES[self.master_dtype]))
        decay = learning_rate * self.weight_decay
        if self.cautious_weight_decay:
            decay = decay * cautious_decay(update, master_parameter.astype(mx.float32))
        master_parameter = (
            master_parameter * (1.0 - decay) - learning_rate * update
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
    """MUD on the body, cautious Lion on the embeddings and everything else.

    Three groups, not two: modded-nanogpt's largest win after Muon was untying
    the output head from the input embedding and running the embedding at a much
    higher rate than the body. Both of those tables are lookup-shaped rather than
    matrix-shaped, so neither belongs in MUD's whitening; giving them their own
    optimizer is what makes a separate rate expressible.
    """

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
        embedding_learning_rate: float | None = None,
        cautious_weight_decay: bool = True,
    ):
        self.mud_learning_rate = mud_learning_rate
        self.fallback_learning_rate = fallback_learning_rate
        # None keeps embeddings on the fallback rate, i.e. the pre-split behaviour.
        self.embedding_learning_rate = (
            fallback_learning_rate if embedding_learning_rate is None else embedding_learning_rate
        )
        mud = MUD(
            mud_learning_rate,
            momentum,
            passes,
            weight_decay,
            block_size,
            mud_eight_bit,
            mud_master_dtype,
            cautious_weight_decay,
        )
        embedding = CLion(self.embedding_learning_rate, betas, eight_bit)
        clion = CLion(fallback_learning_rate, betas, eight_bit)
        super().__init__(
            [mud, embedding, clion],
            [self._is_mud_parameter, self._is_embedding_parameter],
        )

    def _split_dictionary(self, gradients: dict):
        # tree_unflatten turns an empty group into [], and Optimizer.init then
        # indexes into it and raises. Any model without an embedding-shaped
        # parameter -- a bare nn.Linear in a test, say -- hits that.
        return [part if part else {} for part in super()._split_dictionary(gradients)]

    @staticmethod
    def _is_embedding_parameter(path: str, parameter: mx.array) -> bool:
        """Token-indexed tables: the input embedding and an untied output head."""
        return (
            path.endswith("embedding.weight")
            or path.endswith("loop_embed.weight")
            or path.endswith("lm_head.weight")
        )

    @classmethod
    def _is_mud_parameter(cls, path: str, parameter: mx.array) -> bool:
        depthwise_conv = path.endswith("short_conv_weight")
        return (
            parameter.ndim == 2
            and not cls._is_embedding_parameter(path, parameter)
            and not depthwise_conv
        )

    def set_lr_multiplier(self, multiplier: float) -> None:
        self.optimizers[0].learning_rate = self.mud_learning_rate * multiplier
        self.optimizers[1].learning_rate = self.embedding_learning_rate * multiplier
        self.optimizers[2].learning_rate = self.fallback_learning_rate * multiplier

    def checkpoint_config(self) -> dict:
        mud, _embedding, clion = self.optimizers
        return {
            "mud_learning_rate": self.mud_learning_rate,
            "fallback_learning_rate": self.fallback_learning_rate,
            "embedding_learning_rate": self.embedding_learning_rate,
            "weight_decay": mud.weight_decay,
            "momentum": mud.momentum,
            "passes": mud.passes,
            "block_size": mud.block_size,
            "betas": [clion.beta1, clion.beta2],
            "eight_bit": clion.eight_bit,
            "mud_eight_bit": mud.eight_bit,
            "mud_master_dtype": mud.master_dtype,
            "cautious_weight_decay": mud.cautious_weight_decay,
        }
