"""Local PaTH-FoX attention with paper-style Infini compressive memory.

Pure attention sublayer inside ``HybridTransformerBlock``. Local path is
PaTH-FoX over path windows; long-range path is Infini-attention as in
Munkhdalai et al. 2024 (associative matrix memory + linear attention retrieve,
optional delta-rule update), mixed by a per-head gate.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TernaryConfig
from layers.h_bitlinear import HBitLinear
from utils import combine_attention_bias


class InfiniAttention(nn.Module):
    """PaTH-FoX local attention + paper Infini associative memory."""

    def __init__(self, config: TernaryConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.config = config
        self.num_blocks = max(1, config.block_size)
        self.path_window_size = config.path_window_size
        # Paper Linear+Delta by default (arXiv:2404.07143).
        self.use_delta_rule = bool(getattr(config, "infini_delta_rule", True))
        self._memory_eps = 1e-6

        self.qkv = HBitLinear(hidden_size, hidden_size * 3, bias=False, config=config)
        self.o_proj = HBitLinear(hidden_size, hidden_size, bias=False, config=config)
        path_rank = min(32, hidden_size)
        self.path_w_down = HBitLinear(hidden_size, path_rank, bias=False, config=config)
        self.path_w_up = HBitLinear(path_rank, hidden_size, bias=False, config=config)
        self.path_conv_weight = nn.Parameter(torch.empty(hidden_size, 3))
        self.path_beta = nn.Linear(hidden_size, num_heads)
        self.path_forget = nn.Linear(hidden_size, num_heads)
        nn.init.normal_(self.path_conv_weight, std=config.initializer_range)
        self.gate = nn.Parameter(torch.zeros(num_heads))
        self.use_topk_blocks = bool(getattr(config, "use_topk_blocks", False))
        if self.use_topk_blocks:
            # -2.0 (sigmoid ~0.12): start retrieval as a nudge, not half the output.
            # Only registered when enabled, so checkpoints without the branch stay
            # loadable and no dead parameter shows up in gradient checks.
            self.topk_gate = nn.Parameter(torch.full((num_heads,), -2.0))
        self.update_memory_buffers = True

        # Infini memory capacity is memory_dim * head_dim per head; memory_dim is the
        # only knob that changes how much it can hold. Shared across heads on purpose.
        self.memory_dim = int(getattr(config, "infini_memory_expand", 0)) or self.head_dim
        if self.memory_dim != self.head_dim:
            self.memory_proj = nn.Linear(self.head_dim, self.memory_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Paper memory: M ∈ R^{d×d} associative bindings, z ∈ R^{d} key normalizer.
        self.register_buffer(
            "memory_m",
            torch.zeros(0, num_heads, self.memory_dim, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_z",
            torch.zeros(0, num_heads, self.memory_dim),
            persistent=False,
        )
        self.register_buffer("memory_initialized", torch.zeros(0, dtype=torch.bool), persistent=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Transient memory and legacy slot-bank keys are never checkpointed.
        for name in (
            "memory_m",
            "memory_z",
            "memory_k",
            "memory_v",
            "memory_initialized",
            "branch_gates",
        ):
            state_dict.pop(f"{prefix}{name}", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_memory(self) -> None:
        self.memory_m.zero_()
        self.memory_z.zero_()
        self.memory_initialized.fill_(False)

    def _ensure_memory_batch(self, batch_size: int) -> None:
        if self.memory_m.size(0) == batch_size:
            return
        device = self.qkv.weight.device
        d = self.head_dim
        h = self.num_heads
        self.memory_m = torch.zeros(batch_size, h, self.memory_dim, d, device=device)
        self.memory_z = torch.zeros(batch_size, h, self.memory_dim, device=device)
        self.memory_initialized = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def get_memory_state(self) -> dict[str, torch.Tensor]:
        return {
            "memory_m": self.memory_m.detach().clone(),
            "memory_z": self.memory_z.detach().clone(),
            "memory_initialized": self.memory_initialized.detach().clone(),
        }

    def load_memory_state(self, state: dict[str, torch.Tensor]) -> None:
        device = self.qkv.weight.device
        if "memory_m" in state and "memory_z" in state:
            self.memory_m = state["memory_m"].to(device=device, dtype=self.memory_m.dtype).clone()
            self.memory_z = state["memory_z"].to(device=device, dtype=self.memory_z.dtype).clone()
        else:
            # Legacy slot-bank state: drop and re-init empty paper memory.
            batch = int(state.get("memory_k", state.get("memory_v", self.memory_m)).size(0))
            self._ensure_memory_batch(max(batch, 1))
            self.reset_memory()
            if batch != self.memory_m.size(0):
                self._ensure_memory_batch(batch)
            return
        initialized = state.get("memory_initialized")
        if initialized is not None:
            self.memory_initialized = initialized.to(device=device)
        else:
            self.memory_initialized = self.memory_z.flatten(1).count_nonzero(dim=1).bool()

    @contextlib.contextmanager
    def use_memory_state(self, state: dict[str, torch.Tensor], *, update_memory_buffers: bool = True):
        """Temporarily load ``state`` (and optionally freeze writes).

        Intended for gradient-checkpoint **recompute** only: on exit, restores the
        prior buffers/flag so a no-write recompute cannot clobber post-forward memory.
        """
        previous_state = self.get_memory_state()
        previous_update = self.update_memory_buffers
        self.load_memory_state(state)
        self.update_memory_buffers = update_memory_buffers
        try:
            yield
        finally:
            self.update_memory_buffers = previous_update
            self.load_memory_state(previous_state)

    def _sigma(self, x: torch.Tensor) -> torch.Tensor:
        """phi() of the key/query features; see mlx_model.MLXPaTHAttention._memory_features."""
        projected = self.memory_proj(x) if self.memory_dim != self.head_dim else x
        if getattr(self.config, "infini_feature_map", "elu") == "favor":
            norm = x.pow(2).sum(dim=-1, keepdim=True) / 2
            shift = projected.max(dim=-1, keepdim=True).values
            return torch.exp(projected - norm - shift)
        return F.elu(projected) + 1.0

    def _retrieve_memory(
        self,
        q: torch.Tensor,
        memory_m: torch.Tensor,
        memory_z: torch.Tensor,
    ) -> torch.Tensor:
        """A_mem = σ(Q) M / (σ(Q) z). q: (B,H,T,D); M: (B,H,D,D); z: (B,H,D)."""
        sq = self._sigma(q.float())
        numerator = torch.matmul(sq, memory_m)
        denominator = torch.einsum("bhtd,bhd->bht", sq, memory_z).unsqueeze(-1)
        denominator = denominator.clamp_min(self._memory_eps)
        return numerator / denominator

    def _update_memory_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_m: torch.Tensor,
        memory_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear or Linear+Delta associative update (paper eqs. 8–9).

        Truncated BPTT, window 1: history is detached, this chunk's K/V stay live.
        Keeps gradient on the write path while matching the MLX port, which cannot
        afford full BPTT here (see ``mlx_model.MLXPaTHAttention._next_memory``).
        """
        memory_m = memory_m.detach()
        memory_z = memory_z.detach()
        sk = self._sigma(k.float())
        vf = v.float()
        if self.use_delta_rule:
            denominator = torch.einsum("bhtd,bhd->bht", sk, memory_z).unsqueeze(-1)
            denominator = denominator.clamp_min(self._memory_eps)
            retrieved = torch.matmul(sk, memory_m) / denominator
            delta_v = vf - retrieved
            memory_m = memory_m + torch.matmul(sk.transpose(-2, -1), delta_v)
        else:
            memory_m = memory_m + torch.matmul(sk.transpose(-2, -1), vf)
        memory_z = memory_z + sk.sum(dim=2)
        return memory_m, memory_z

    def _topk_context(
        self, q_chunk: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> Optional[torch.Tensor]:
        """Softmax attention over the top-k most similar whole blocks before ``start``.

        Mirrors ``mlx_model.MLXPaTHAttention._topk_context``: past tokens are split
        into ``topk_block_size`` blocks, scored by mean-key against the chunk's mean
        query, and the best ``topk_blocks`` are attended in full. Selected tokens all
        precede the chunk, so causality needs no mask.
        """
        block = int(self.config.topk_block_size)
        num_blocks = start // block
        if num_blocks == 0:
            return None
        take = min(int(self.config.topk_blocks), num_blocks)
        batch, heads, _, dim = k.shape
        shape = (batch, heads, num_blocks, block, dim)
        key_blocks = k[:, :, : num_blocks * block].reshape(shape)
        value_blocks = v[:, :, : num_blocks * block].reshape(shape)

        summary = key_blocks.float().mean(dim=3)
        probe = q_chunk.float().mean(dim=2, keepdim=True)
        scores = (summary * probe).sum(dim=-1)
        chosen = scores.topk(take, dim=-1).indices[..., None, None]
        chosen = chosen.expand(-1, -1, -1, block, dim)
        selected_k = key_blocks.gather(2, chosen).reshape(batch, heads, take * block, dim)
        selected_v = value_blocks.gather(2, chosen).reshape(batch, heads, take * block, dim)
        logits = torch.matmul(q_chunk, selected_k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=v.dtype)
        return torch.matmul(weights, selected_v)

    def _path_vectors(
        self,
        x: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
        prepared_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Paper PaTH low-rank projection, causal width-3 depthwise conv, L2 norm."""
        down = self.path_w_down(x) if prepared_x is None else self.path_w_down.forward_prepared(prepared_x)
        projected = self.path_w_up(down)
        weight = self.path_conv_weight.to(dtype=projected.dtype)
        convolved = projected * weight[:, 2]
        for offset, kernel_index in ((1, 1), (2, 0)):
            shifted = F.pad(projected[:, :-offset], (0, 0, offset, 0))
            if segment_ids is not None:
                same_segment = F.pad(
                    segment_ids[:, offset:].eq(segment_ids[:, :-offset]),
                    (offset, 0),
                )
                shifted = shifted * same_segment.unsqueeze(-1)
            convolved = convolved + shifted * weight[:, kernel_index]
        convolved = F.silu(convolved).float().view(
            x.size(0), x.size(1), self.num_heads, self.head_dim
        )
        return F.normalize(convolved, dim=-1, eps=1e-6)

    def _chunk_ranges(self, seq_len: int):
        block_width = (seq_len + self.num_blocks - 1) // self.num_blocks
        for block_start in range(0, seq_len, block_width):
            block_end = min(block_start + block_width, seq_len)
            for start in range(block_start, block_end, self.path_window_size):
                yield start, min(start + self.path_window_size, block_end)

    @staticmethod
    def _slice_mask(mask: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if mask.ndim == 2:
            return mask[:, start:end]
        if mask.ndim == 3:
            return mask[:, start:end, start:end]
        if mask.ndim == 4:
            return mask[:, :, start:end, start:end]
        raise ValueError("attention_mask must be a bool, 2D, 3D, or 4D tensor")

    def _path_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        beta: torch.Tensor,
        log_forget: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Paper-exact PaTH-FoX logits through compact UT Householder products."""
        batch_size, _, chunk_len, _ = q.shape
        qf, kf = q.float(), k.float()
        wf = w.transpose(1, 2)  # (B, H, T, D)
        beta = beta.transpose(1, 2)

        gram = torch.matmul(wf, wf.transpose(-2, -1))
        system = torch.eye(chunk_len, device=q.device, dtype=torch.float32).view(1, 1, chunk_len, chunk_len)
        system = system + torch.tril(beta.unsqueeze(-1) * gram, diagonal=-1)
        t_inv = torch.linalg.solve_triangular(
            system,
            torch.diag_embed(beta),
            upper=False,
            unitriangular=True,
        )
        qk = torch.matmul(qf, kf.transpose(-2, -1))
        qw = torch.tril(torch.matmul(qf, wf.transpose(-2, -1)))
        wk = torch.tril(torch.matmul(wf, kf.transpose(-2, -1)), diagonal=-1)
        logits = (qk - torch.matmul(torch.matmul(qw, t_inv), wk)) * (self.head_dim ** -0.5)

        prefix = log_forget.transpose(1, 2).cumsum(dim=-1)
        logits = logits + prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
        causal = torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=q.device).tril()
        base_bias = torch.zeros(1, 1, chunk_len, chunk_len, dtype=torch.float32, device=q.device)
        base_bias.masked_fill_(~causal, torch.finfo(torch.float32).min)
        attn_bias, query_valid = combine_attention_bias(
            attention_mask,
            base_bias=base_bias,
            batch_size=batch_size,
            q_len=chunk_len,
            k_len=chunk_len,
            dtype=torch.float32,
            device=q.device,
        )
        probabilities = torch.softmax(logits + attn_bias, dim=-1).to(dtype=v.dtype)
        output = torch.matmul(probabilities, v)
        if query_valid is not None:
            output = output.masked_fill(~query_valid, 0.0)
        return output

    def _local_path_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        beta: torch.Tensor,
        log_forget: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
        update_memory: bool,
        memory_safe: Optional[bool] = None,
    ) -> torch.Tensor:
        if memory_safe is None:
            memory_safe = True
            if attention_mask is not None:
                memory_safe = memory_safe and attention_mask.ndim == 2 and bool(attention_mask.all())
            if segment_ids is not None:
                memory_safe = memory_safe and bool((segment_ids == segment_ids[:, :1]).all())

        ranges = list(self._chunk_ranges(q.size(2)))
        chunk_len = ranges[0][1] - ranges[0][0]
        regular_chunks = all(
            start == index * chunk_len and end == start + chunk_len
            for index, (start, end) in enumerate(ranges)
        )
        if not memory_safe and attention_mask is None and regular_chunks:
            batch_size, heads, seq_len, dim = q.shape
            num_chunks = len(ranges)

            def batch_chunks(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape(batch_size, heads, num_chunks, chunk_len, dim).transpose(1, 2).reshape(
                    batch_size * num_chunks, heads, chunk_len, dim
                )

            chunk_mask = None
            if segment_ids is not None:
                ids = segment_ids.reshape(batch_size, num_chunks, chunk_len)
                chunk_mask = ids.unsqueeze(-1).eq(ids.unsqueeze(-2)).reshape(
                    batch_size * num_chunks, chunk_len, chunk_len
                )
            batched = self._path_chunk(
                batch_chunks(q),
                batch_chunks(k),
                batch_chunks(v),
                w.reshape(batch_size * num_chunks, chunk_len, heads, dim),
                beta.reshape(batch_size * num_chunks, chunk_len, heads),
                log_forget.reshape(batch_size * num_chunks, chunk_len, heads),
                chunk_mask,
            )
            return batched.reshape(batch_size, num_chunks, heads, chunk_len, dim).transpose(1, 2).reshape(
                batch_size, heads, seq_len, dim
            )

        # Carry M,z through path windows (BPTT within the forward; detach on store).
        memory_m = self.memory_m
        memory_z = self.memory_z
        memory_initialized = self.memory_initialized

        chunks = []
        for start, end in ranges:
            chunk_mask = self._slice_mask(attention_mask, start, end) if attention_mask is not None else None
            if segment_ids is not None:
                ids = segment_ids[:, start:end]
                document_mask = ids[:, :, None].eq(ids[:, None, :])
                if chunk_mask is None:
                    chunk_mask = document_mask
                elif chunk_mask.ndim == 2:
                    valid = chunk_mask.bool()
                    chunk_mask = document_mask & valid[:, :, None] & valid[:, None, :]
                elif chunk_mask.dtype == torch.bool:
                    chunk_mask = chunk_mask & document_mask
                else:
                    chunk_mask = chunk_mask + torch.zeros_like(document_mask, dtype=chunk_mask.dtype).masked_fill(
                        ~document_mask, torch.finfo(chunk_mask.dtype).min
                    )
            local_context = self._path_chunk(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                w[:, start:end],
                beta[:, start:end],
                log_forget[:, start:end],
                chunk_mask,
            )
            if memory_safe:
                memory_context = self._retrieve_memory(
                    q[:, :, start:end], memory_m, memory_z
                ).to(dtype=v.dtype)
                gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
                mixed_context = (1.0 - gate) * local_context + gate * memory_context
                initialized = memory_initialized.view(-1, 1, 1, 1)
                local_context = torch.where(initialized, mixed_context, local_context)
            if self.use_topk_blocks and memory_safe:
                # memory_safe already rules out packed documents, which cross-block
                # retrieval would leak across.
                retrieved = self._topk_context(q[:, :, start:end], k, v, start)
                if retrieved is not None:
                    topk_gate = torch.sigmoid(self.topk_gate).view(1, self.num_heads, 1, 1)
                    local_context = (
                        1.0 - topk_gate
                    ) * local_context + topk_gate * retrieved.to(dtype=local_context.dtype)
            chunks.append(local_context)
            if memory_safe and update_memory:
                memory_m, memory_z = self._update_memory_state(
                    k[:, :, start:end],
                    v[:, :, start:end],
                    memory_m,
                    memory_z,
                )
                memory_initialized = torch.ones_like(memory_initialized)

        if update_memory and memory_safe:
            # Detach stored state so the next top-level call does not keep this graph.
            self.memory_m = memory_m.detach()
            self.memory_z = memory_z.detach()
            self.memory_initialized = memory_initialized.detach()
        return torch.cat(chunks, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        update_memory: Optional[bool] = None,
        memory_safe: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        self._ensure_memory_batch(batch_size)
        prepared_x = self.qkv.prepare_input(x)
        qkv = self.qkv.forward_prepared(prepared_x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        same_preparation = (
            self.qkv.hadamard_size == self.path_w_down.hadamard_size
            and self.qkv.enable_activation_quantization == self.path_w_down.enable_activation_quantization
            and self.qkv.activation_bits == self.path_w_down.activation_bits
            and self.qkv.activation_quantization_mix == self.path_w_down.activation_quantization_mix
        )
        w = self._path_vectors(x, segment_ids, prepared_x if same_preparation else None)
        beta = 2.0 * torch.sigmoid(self.path_beta(x).float())
        log_forget = F.logsigmoid(self.path_forget(x).float())
        local_mask = attn_bias if attn_bias is not None else attention_mask
        requested = self.update_memory_buffers if update_memory is None else bool(update_memory)
        do_update = requested and self.update_memory_buffers
        context = self._local_path_attention(
            q, k, v, w, beta, log_forget, local_mask, segment_ids, do_update, memory_safe
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)

        return output
