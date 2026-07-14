"""Experimental MLX port of the dense BitNet PaTH-FoX model path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_path_kernel import path_triangular_solve, reference_triangular_solve


@dataclass(frozen=True)
class MLXBitNetConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    num_attention_heads: int = 16
    intermediate_size: int = 1024
    num_prelude_layers: int = 2
    num_recurrent_layers: int = 4
    num_coda_layers: int = 2
    num_loops: int = 4
    block_size: int = 8
    path_window_size: int = 64
    infini_memory_dim: int = 64
    activation_bits: int = 4
    use_4bit_activations: bool = True
    use_hadamard: bool = True
    use_path_kernel: bool = True
    rms_norm_eps: float = 1e-5
    use_engram: bool = True
    engram_layer_ids: tuple[int, ...] = (1, 15)
    engram_vocab_size: int = 4093
    engram_max_ngram_size: int = 3
    engram_num_heads: int = 4
    engram_head_dim: int = 16
    engram_kernel_size: int = 4
    engram_pad_id: int = 257
    engram_seed: int = 0
    use_rfmoe: bool = False
    rfmoe_num_experts: int = 8
    rfmoe_expert_dim: int | None = None
    rfmoe_rank: int | None = None
    rfmoe_theta: float = 0.01
    mtp_depth: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.path_window_size < 1:
            raise ValueError("path_window_size must be positive")
        if min(self.num_prelude_layers, self.num_recurrent_layers, self.num_coda_layers) < 0:
            raise ValueError("layer counts must be non-negative")
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        if self.block_size < 1 or self.infini_memory_dim < 1:
            raise ValueError("block_size and infini_memory_dim must be positive")
        if self.activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if self.engram_vocab_size < 1 or self.engram_max_ngram_size < 2:
            raise ValueError("invalid Engram table or N-gram size")
        if self.rfmoe_num_experts < 1:
            raise ValueError("rfmoe_num_experts must be positive")

    @property
    def num_hidden_layers(self) -> int:
        return self.num_prelude_layers + self.num_recurrent_layers + self.num_coda_layers

    @property
    def effective_depth(self) -> int:
        return self.num_prelude_layers + self.num_recurrent_layers * self.num_loops + self.num_coda_layers


class MLXHBitLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, config: MLXBitNetConfig):
        super().__init__()
        self.input_dims = input_dims
        self.config = config
        self.weight = mx.random.uniform(
            low=-input_dims**-0.5,
            high=input_dims**-0.5,
            shape=(output_dims, input_dims),
        )
        self.weight_mix = mx.array(1.0)
        self.activation_mix = mx.array(1.0)
        self.activation_levels = mx.array(float((2 ** (config.activation_bits - 1)) - 1))
        self.freeze(keys=["weight_mix", "activation_mix", "activation_levels"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.config.use_hadamard and self.input_dims & (self.input_dims - 1) == 0:
            x = mx.hadamard_transform(x)
        if self.config.use_4bit_activations:
            levels = self.activation_levels
            negative_levels = levels + 1
            activation_scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5) / levels
            quantized_x = mx.clip(mx.round(x / activation_scale), -negative_levels, levels) * activation_scale
            x = x + self.activation_mix * mx.stop_gradient(quantized_x - x)

        weight_scale = mx.maximum(mx.mean(mx.abs(self.weight), axis=-1, keepdims=True), 1e-5)
        normalized = self.weight / weight_scale
        ternary = mx.where(normalized > 0.5, 1.0, mx.where(normalized < -0.5, -1.0, 0.0))
        quantized_weight = ternary * weight_scale
        weight = self.weight + self.weight_mix * mx.stop_gradient(quantized_weight - self.weight)
        return x @ weight.astype(x.dtype).T

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        self.weight_mix = mx.array(weight_mix)
        self.activation_mix = mx.array(activation_mix)
        self.activation_levels = mx.array(float((2 ** (max(bits, 2) - 1)) - 1))


class MLXEngram(nn.Module):
    def __init__(self, config: MLXBitNetConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.num_tables = (config.engram_max_ngram_size - 1) * config.engram_num_heads
        self.embedding = nn.Embedding(
            self.num_tables * config.engram_vocab_size,
            config.engram_head_dim,
        )
        self.offsets = mx.arange(self.num_tables, dtype=mx.int64) * config.engram_vocab_size
        rng = np.random.default_rng(config.engram_seed + 10_007 * layer_id)
        multipliers = rng.integers(
            1,
            2**31,
            size=(config.engram_max_ngram_size, config.engram_num_heads),
            dtype=np.int64,
        )
        self.multipliers = mx.array(multipliers | 1)
        memory_size = self.num_tables * config.engram_head_dim
        self.key_proj = nn.Linear(memory_size, config.hidden_size, bias=False)
        self.value_proj = nn.Linear(memory_size, config.hidden_size, bias=False)
        self.key_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.query_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.short_conv_weight = mx.random.uniform(
            low=-config.engram_kernel_size**-0.5,
            high=config.engram_kernel_size**-0.5,
            shape=(config.hidden_size, config.engram_kernel_size),
        )
        self.freeze(keys=["offsets", "multipliers"], recurse=False)

    def hash_ids(self, input_ids: mx.array, segment_ids: mx.array | None = None) -> mx.array:
        hashes = []
        for ngram_size in range(2, self.config.engram_max_ngram_size + 1):
            mixed = input_ids[..., None] * self.multipliers[0]
            for lag in range(1, ngram_size):
                shifted = mx.full(input_ids.shape, self.config.engram_pad_id, dtype=input_ids.dtype)
                shifted = mx.concatenate((shifted[:, :lag], input_ids[:, :-lag]), axis=1)
                keep = mx.ones(input_ids.shape, dtype=mx.bool_)
                keep = mx.concatenate((mx.zeros((input_ids.shape[0], lag), dtype=mx.bool_), keep[:, lag:]), axis=1)
                if segment_ids is not None:
                    same = segment_ids[:, lag:] == segment_ids[:, :-lag]
                    same = mx.concatenate((mx.zeros((input_ids.shape[0], lag), dtype=mx.bool_), same), axis=1)
                    keep = keep & same
                shifted = mx.where(keep, shifted, self.config.engram_pad_id)
                mixed = mx.bitwise_xor(mixed, shifted[..., None] * self.multipliers[lag])
            hashes.append(mx.remainder(mixed, self.config.engram_vocab_size))
        return mx.concatenate(hashes, axis=-1)

    def __call__(self, hidden_states: mx.array, input_ids: mx.array, segment_ids: mx.array | None) -> mx.array:
        memory = self.embedding(self.hash_ids(input_ids, segment_ids) + self.offsets)
        memory = memory.reshape(*memory.shape[:-2], -1)
        key = self.key_norm(self.key_proj(memory))
        query = self.query_norm(hidden_states)
        score = mx.sum(key * query, axis=-1) / math.sqrt(hidden_states.shape[-1])
        score = mx.sign(score) * mx.sqrt(mx.maximum(mx.abs(score), 1e-6))
        value = mx.sigmoid(score)[..., None] * self.value_proj(memory)
        normalized = self.conv_norm(value)
        convolved = mx.zeros_like(normalized)
        for lag in range(self.config.engram_kernel_size):
            shifted = normalized if lag == 0 else mx.concatenate(
                (mx.zeros_like(normalized[:, :lag]), normalized[:, :-lag]), axis=1
            )
            if segment_ids is not None and lag > 0:
                same = segment_ids[:, lag:] == segment_ids[:, :-lag]
                same = mx.concatenate((mx.zeros((segment_ids.shape[0], lag), dtype=mx.bool_), same), axis=1)
                shifted = shifted * same[..., None]
            weight = self.short_conv_weight[:, self.config.engram_kernel_size - 1 - lag]
            convolved = convolved + shifted * weight
        return value + nn.silu(convolved)


class MLXRFMoEExpert(nn.Module):
    def __init__(self, config: MLXBitNetConfig, expert_dim: int, rank: int):
        super().__init__()
        hidden = config.hidden_size
        self.a_gate = MLXHBitLinear(hidden, rank, config)
        self.b_gate = MLXHBitLinear(rank, expert_dim, config)
        self.w_up = MLXHBitLinear(hidden, expert_dim, config)
        self.w_mid = MLXHBitLinear(expert_dim, expert_dim, config)
        self.w_down = MLXHBitLinear(expert_dim, hidden, config)
        self.bias = mx.array([1e-6])

    def __call__(self, x: mx.array):
        z = self.a_gate(x)
        gate = mx.maximum(mx.linalg.norm(z, axis=-1) - self.bias, 0.0)
        hidden = mx.sigmoid(self.b_gate(z)) * self.w_up(x)
        contribution = self.w_down(nn.silu(self.w_mid(hidden)))
        return gate, contribution


class MLXRFMoE(nn.Module):
    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        expert_dim = config.rfmoe_expert_dim or config.intermediate_size // 4
        rank = config.rfmoe_rank or max(1, config.hidden_size // 16)
        self.theta = config.rfmoe_theta
        self.experts = [MLXRFMoEExpert(config, expert_dim, rank) for _ in range(config.rfmoe_num_experts)]
        self.usage_ema = mx.full((config.rfmoe_num_experts,), 1.0 / config.rfmoe_num_experts)
        self.last_usage = mx.zeros((config.rfmoe_num_experts,))
        self.last_gate = mx.zeros((config.rfmoe_num_experts, 1))
        self.last_density = mx.array(0.0)
        self.freeze(keys=["usage_ema", "last_usage", "last_gate", "last_density"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        output = mx.zeros_like(flat)
        gates = []
        fires = []
        for expert in self.experts:
            gate, contribution = expert(flat)
            fire = gate >= self.theta
            output = output + mx.where(fire[:, None], gate[:, None] * contribution, 0.0)
            gates.append(gate)
            fires.append(fire)
        gate_stack = mx.stack(gates)
        fire_stack = mx.stack(fires)
        if self.training:
            self.last_gate = gate_stack
            self.last_usage = mx.mean(gate_stack, axis=1)
            self.last_density = mx.mean(fire_stack)
            self.usage_ema = 0.99 * self.usage_ema + 0.01 * mx.stop_gradient(self.last_usage)
        return output.reshape(x.shape)

    def aux_losses(self, s: float, alpha: float):
        density = mx.mean(self.last_usage)
        usage = self.last_usage / mx.maximum(mx.sum(self.last_usage), 1e-8)
        order = mx.argsort(self.usage_ema)[::-1]
        ranked = usage[order]
        ranks = mx.arange(1, len(self.experts) + 1, dtype=mx.float32)
        target = ranks ** (-s)
        target = (1.0 - alpha) * target / mx.sum(target) + alpha / len(self.experts)
        locality = mx.sum(target * (mx.log(mx.maximum(target, 1e-8)) - mx.log(mx.maximum(ranked, 1e-8))))
        centered = self.last_gate - mx.mean(self.last_gate, axis=1, keepdims=True)
        normalized = centered / mx.maximum(mx.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
        correlation = normalized @ normalized.T
        off_diagonal = correlation - mx.diag(mx.diag(correlation))
        diversity = mx.sum(mx.square(mx.maximum(off_diagonal, 0.0))) / max(
            len(self.experts) * (len(self.experts) - 1), 1
        )
        return density, locality, diversity


class MLXLoopHyperConnection(nn.Module):
    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        flat = 4 * config.hidden_size
        self.norm = nn.RMSNorm(flat, eps=config.rms_norm_eps)
        self.w_pre = nn.Linear(flat, 4)
        self.w_post = nn.Linear(flat, 4)
        self.w_res = nn.Linear(flat, 4)
        self.alpha_pre = mx.array([1.0])
        self.alpha_post = mx.array([1.0])
        self.alpha_res = mx.array([1.0])
        self.loop_embed = nn.Embedding(64, config.hidden_size)
        self.w_pre.weight = mx.zeros_like(self.w_pre.weight)
        self.w_pre.bias = mx.full(self.w_pre.bias.shape, -math.log(3.0))
        self.w_post.weight = mx.zeros_like(self.w_post.weight)
        self.w_post.bias = mx.full(self.w_post.bias.shape, -3.0)
        self.w_res.weight = mx.zeros_like(self.w_res.weight)
        self.w_res.bias = mx.full(self.w_res.bias.shape, 3.0)
        self.loop_embed.weight = mx.zeros_like(self.loop_embed.weight)

    @staticmethod
    def expand(x: mx.array) -> mx.array:
        return mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], 4, x.shape[-1]))

    @staticmethod
    def fold(y: mx.array) -> mx.array:
        return mx.mean(y, axis=2)

    def project_in(self, y: mx.array):
        z = self.norm(y.reshape(*y.shape[:2], -1))
        pre = mx.sigmoid(self.alpha_pre * self.w_pre(z))
        post = 2.0 * mx.sigmoid(self.alpha_post * self.w_post(z))
        residual = mx.sigmoid(self.alpha_res * self.w_res(z))
        return mx.sum(pre[..., None] * y, axis=2), post, residual

    @staticmethod
    def write_back(y: mx.array, output: mx.array, post: mx.array, residual: mx.array) -> mx.array:
        return residual[..., None] * y + post[..., None] * output[:, :, None, :]


class MLXPaTHAttention(nn.Module):
    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        self.config = config
        self.num_blocks = config.block_size
        hidden = config.hidden_size
        self.head_dim = hidden // config.num_attention_heads
        self.qkv = MLXHBitLinear(hidden, hidden * 3, config)
        self.out = MLXHBitLinear(hidden, hidden, config)
        path_rank = min(32, hidden)
        self.path_down = MLXHBitLinear(hidden, path_rank, config)
        self.path_up = MLXHBitLinear(path_rank, hidden, config)
        self.path_conv_weight = mx.random.normal((hidden, 3)) * 0.02
        self.path_beta = nn.Linear(hidden, config.num_attention_heads)
        self.path_forget = nn.Linear(hidden, config.num_attention_heads)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.memory_gate = mx.zeros((config.num_attention_heads,))
        self.out.weight = self.out.weight * 0.01
        self.memory_k = mx.zeros((0, config.num_attention_heads, config.infini_memory_dim, self.head_dim))
        self.memory_v = mx.zeros_like(self.memory_k)
        self.memory_initialized = mx.zeros((0,), dtype=mx.bool_)
        self.freeze(keys=["memory_k", "memory_v", "memory_initialized"], recurse=False)

    def reset_memory(self, batch_size: int | None = None) -> None:
        batch_size = self.memory_k.shape[0] if batch_size is None else batch_size
        shape = (
            batch_size,
            self.config.num_attention_heads,
            self.config.infini_memory_dim,
            self.head_dim,
        )
        self.memory_k = mx.zeros(shape, dtype=mx.float32)
        self.memory_v = mx.zeros(shape, dtype=mx.float32)
        self.memory_initialized = mx.zeros((batch_size,), dtype=mx.bool_)

    def _update_memory(self, keys: mx.array, values: mx.array, rows: mx.array) -> None:
        length = keys.shape[2]
        pooled_keys = []
        pooled_values = []
        for index in range(self.config.infini_memory_dim):
            start = index * length // self.config.infini_memory_dim
            end = max(start + 1, ((index + 1) * length + self.config.infini_memory_dim - 1) // self.config.infini_memory_dim)
            pooled_keys.append(mx.mean(keys[:, :, start:end], axis=2))
            pooled_values.append(mx.mean(values[:, :, start:end], axis=2))
        key_update = mx.stack(pooled_keys, axis=2).astype(mx.float32)
        value_update = mx.stack(pooled_values, axis=2).astype(mx.float32)
        rows = rows[:, None, None, None]
        next_k = 0.99 * self.memory_k + 0.01 * mx.stop_gradient(key_update)
        next_v = 0.99 * self.memory_v + 0.01 * mx.stop_gradient(value_update)
        self.memory_k = mx.where(rows, next_k, self.memory_k)
        self.memory_v = mx.where(rows, next_v, self.memory_v)
        self.memory_initialized = self.memory_initialized | rows[:, 0, 0, 0]

    @staticmethod
    def _shift(projected: mx.array, segment_ids: mx.array | None, offset: int) -> mx.array:
        padding = mx.zeros_like(projected[:, :offset])
        shifted = mx.concatenate((padding, projected[:, :-offset]), axis=1)
        if segment_ids is not None:
            same = segment_ids[:, offset:] == segment_ids[:, :-offset]
            leading = mx.zeros(segment_ids[:, :offset].shape, dtype=mx.bool_)
            same = mx.concatenate((leading, same), axis=1)
            shifted = shifted * same[..., None]
        return shifted

    def path_vectors(self, x: mx.array, segment_ids: mx.array | None) -> mx.array:
        projected = self.path_up(self.path_down(x))
        convolved = (
            projected * self.path_conv_weight[:, 2]
            + self._shift(projected, segment_ids, 1) * self.path_conv_weight[:, 1]
            + self._shift(projected, segment_ids, 2) * self.path_conv_weight[:, 0]
        )
        convolved = nn.silu(convolved.astype(mx.float32)).reshape(
            x.shape[0], x.shape[1], self.config.num_attention_heads, self.head_dim
        )
        return convolved / mx.maximum(mx.linalg.norm(convolved, axis=-1, keepdims=True), 1e-6)

    def path_chunk(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        w: mx.array,
        beta: mx.array,
        log_forget: mx.array,
        segment_ids: mx.array | None,
    ) -> mx.array:
        length = q.shape[2]
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        wf = w.transpose(0, 2, 1, 3)
        beta = beta.transpose(0, 2, 1)
        gram = wf @ wf.transpose(0, 1, 3, 2)
        eye = mx.eye(length, dtype=mx.float32)
        system = eye + mx.tril(beta[..., None] * gram, k=-1)
        diagonal = eye * beta[..., :, None]
        solve = path_triangular_solve if self.config.use_path_kernel else reference_triangular_solve
        t_inverse = solve(system, diagonal)
        qk = qf @ kf.transpose(0, 1, 3, 2)
        qw = mx.tril(qf @ wf.transpose(0, 1, 3, 2))
        wk = mx.tril(wf @ kf.transpose(0, 1, 3, 2), k=-1)
        logits = (qk - (qw @ t_inverse) @ wk) * self.head_dim**-0.5
        prefix = mx.cumsum(log_forget.transpose(0, 2, 1), axis=-1)
        logits = logits + prefix[..., :, None] - prefix[..., None, :]
        keep = mx.tril(mx.ones((length, length), dtype=mx.bool_))
        if segment_ids is not None:
            keep = keep & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
        logits = mx.where(keep, logits, -1e9)
        return mx.softmax(logits, axis=-1).astype(v.dtype) @ v

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
    ) -> mx.array:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            length,
            3,
            self.config.num_attention_heads,
            self.head_dim,
        )
        q = self.q_norm(qkv[:, :, 0].transpose(0, 2, 1, 3))
        k = self.k_norm(qkv[:, :, 1].transpose(0, 2, 1, 3))
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        w = self.path_vectors(x, segment_ids)
        beta = 2.0 * mx.sigmoid(self.path_beta(x).astype(mx.float32))
        forget_logits = self.path_forget(x).astype(mx.float32)
        log_forget = -mx.logaddexp(mx.zeros_like(forget_logits), -forget_logits)
        chunks = []
        block_width = (length + self.num_blocks - 1) // self.num_blocks
        memory_safe = (
            mx.ones((batch,), dtype=mx.bool_)
            if segment_ids is None
            else mx.all(segment_ids == segment_ids[:, :1], axis=1)
        )
        for block_start in range(0, length, block_width):
            block_end = min(block_start + block_width, length)
            for start in range(block_start, block_end, self.config.path_window_size):
                end = min(start + self.config.path_window_size, block_end)
                chunk_segments = segment_ids[:, start:end] if segment_ids is not None else None
                local = self.path_chunk(
                    q[:, :, start:end],
                    k[:, :, start:end],
                    v[:, :, start:end],
                    w[:, start:end],
                    beta[:, start:end],
                    log_forget[:, start:end],
                    chunk_segments,
                )
                scores = q[:, :, start:end].astype(mx.float32) @ self.memory_k.swapaxes(-1, -2)
                scores = scores * self.head_dim**-0.5
                memory_context = mx.softmax(scores, axis=-1).astype(v.dtype) @ self.memory_v.astype(v.dtype)
                gate = mx.sigmoid(self.memory_gate)[None, :, None, None]
                mixed = (1.0 - gate) * local + gate * memory_context
                use_memory = self.memory_initialized & memory_safe
                local = mx.where(use_memory[:, None, None, None], mixed, local)
                if update_memory:
                    self._update_memory(k[:, :, start:end], v[:, :, start:end], memory_safe)
                chunks.append(local)
        context = mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.out(context)


class MLXHybridBlock(nn.Module):
    def __init__(self, config: MLXBitNetConfig, layer_id: int):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        self.engram = MLXEngram(config, layer_id) if config.use_engram and layer_id in config.engram_layer_ids else None
        self.attn_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.attn = MLXPaTHAttention(config)
        self.attn_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.attn_scale = mx.array([0.1])
        self.attn_gate = mx.array([0.0])
        self.mlp_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.moe = MLXRFMoE(config) if config.use_rfmoe else None
        if self.moe is None:
            self.up = MLXHBitLinear(hidden, intermediate * 2, config)
            self.mid = MLXHBitLinear(intermediate, intermediate, config)
            self.down = MLXHBitLinear(intermediate, hidden, config)
            self.down.weight = self.down.weight * 0.01
        self.mlp_post = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.mlp_scale = mx.array([0.1])

    def __call__(
        self,
        x: mx.array,
        input_ids: mx.array,
        segment_ids: mx.array | None = None,
        update_memory: bool = True,
    ) -> mx.array:
        if self.engram is not None:
            x = x + self.engram(x, input_ids, segment_ids)
        attention = self.attn(self.attn_norm(x), segment_ids, update_memory)
        x = self.attn_post(x + self.attn_scale * mx.sigmoid(self.attn_gate) * attention)
        normalized = self.mlp_norm(x)
        if self.moe is not None:
            output = self.moe(normalized)
        else:
            gate, value = mx.split(self.up(normalized), 2, axis=-1)
            hidden = nn.silu(gate) * value
            hidden = nn.silu(self.mid(hidden))
            output = self.down(hidden)
        return self.mlp_post(x + self.mlp_scale * output)


class MLXBitNet(nn.Module):
    """MLX BitNet with PaTH/Infini, Engram, RFMoE, Hyperloop, and MTP."""

    def __init__(self, config: MLXBitNetConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.subln = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.blocks = [MLXHybridBlock(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        self.loop_hc = MLXLoopHyperConnection(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_transforms = [
            nn.Sequential(
                nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            )
            for _ in range(config.mtp_depth)
        ]

    def reset_memory(self, batch_size: int) -> None:
        for block in self.blocks:
            block.attn.reset_memory(batch_size)

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        def update(_, module):
            if isinstance(module, MLXHBitLinear):
                module.set_quantization_state(weight_mix, activation_mix, bits)

        self.apply_to_modules(update)

    def set_active_blocks(self, blocks: int) -> None:
        for block in self.blocks:
            block.attn.num_blocks = max(1, blocks)

    def rfmoe_aux_losses(self, s: float, alpha: float):
        density = mx.array(0.0)
        locality = mx.array(0.0)
        diversity = mx.array(0.0)
        hard_densities = []
        for block in self.blocks:
            if block.moe is not None:
                layer_density, layer_locality, layer_diversity = block.moe.aux_losses(s, alpha)
                density = density + layer_density
                locality = locality + layer_locality
                diversity = diversity + layer_diversity
                hard_densities.append(block.moe.last_density)
        hard = mx.mean(mx.stack(hard_densities)) if hard_densities else mx.array(0.0)
        return density, locality, diversity, hard

    def __call__(
        self,
        tokens: mx.array,
        segment_ids: mx.array | None = None,
        num_loops: int | None = None,
        return_mtp: bool = False,
        reset_memory: bool = True,
    ):
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        if reset_memory:
            self.reset_memory(tokens.shape[0])
        x = self.subln(self.embedding(tokens))
        prelude_end = self.config.num_prelude_layers
        recurrent_end = prelude_end + self.config.num_recurrent_layers
        for block in self.blocks[:prelude_end]:
            x = block(x, tokens, segment_ids, True)
        if self.config.num_recurrent_layers:
            streams = self.loop_hc.expand(x)
            for loop_index in range(loops):
                x, post, residual = self.loop_hc.project_in(streams)
                for block in self.blocks[prelude_end:recurrent_end]:
                    x = block(x, tokens, segment_ids, loop_index == loops - 1)
                embedding_index = min(loop_index, 63)
                output = x + self.loop_hc.loop_embed.weight[embedding_index].astype(x.dtype)
                streams = self.loop_hc.write_back(streams, output, post, residual)
            x = self.loop_hc.fold(streams)
        for block in self.blocks[recurrent_end:]:
            x = block(x, tokens, segment_ids, True)
        hidden = self.norm(x)
        logits = hidden @ self.embedding.weight.T
        if return_mtp:
            return logits, [transform(hidden) @ self.embedding.weight.T for transform in self.mtp_transforms]
        return logits
