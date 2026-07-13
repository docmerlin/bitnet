"""Local PaTH-FoX attention with fixed-size Infini memory.

Pure attention sublayer inside ``HybridTransformerBlock``. The block owns
pre-norm and AttnRes; this module mixes local block-causal attention with
compressive memory attention.
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
    """Attention with compressive memory and per-head gating."""

    def __init__(self, config: TernaryConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.memory_dim = config.infini_memory_dim
        self.config = config
        self.num_blocks = max(1, config.block_size)
        self.path_window_size = config.path_window_size

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
        self.update_memory_buffers = True

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.register_buffer(
            "memory_k",
            torch.zeros(0, num_heads, self.memory_dim, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_v",
            torch.zeros(0, num_heads, self.memory_dim, self.head_dim),
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
        # Older checkpoints may still contain serialized memory buffers.
        state_dict.pop(f"{prefix}memory_k", None)
        state_dict.pop(f"{prefix}memory_v", None)
        state_dict.pop(f"{prefix}memory_initialized", None)
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
        self.memory_k.zero_()
        self.memory_v.zero_()
        self.memory_initialized.fill_(False)

    def _ensure_memory_batch(self, batch_size: int) -> None:
        if self.memory_k.size(0) == batch_size:
            return
        shape = (batch_size, self.num_heads, self.memory_dim, self.head_dim)
        self.memory_k = torch.zeros(shape, device=self.qkv.weight.device)
        self.memory_v = torch.zeros_like(self.memory_k)
        self.memory_initialized = torch.zeros(batch_size, dtype=torch.bool, device=self.qkv.weight.device)

    def get_memory_state(self) -> dict[str, torch.Tensor]:
        return {
            "memory_k": self.memory_k.detach().clone(),
            "memory_v": self.memory_v.detach().clone(),
            "memory_initialized": self.memory_initialized.detach().clone(),
        }

    def load_memory_state(self, state: dict[str, torch.Tensor]) -> None:
        self.memory_k = state["memory_k"].to(device=self.qkv.weight.device, dtype=self.memory_k.dtype).clone()
        self.memory_v = state["memory_v"].to(device=self.qkv.weight.device, dtype=self.memory_v.dtype).clone()
        initialized = state.get("memory_initialized")
        self.memory_initialized = (
            initialized.to(device=self.memory_initialized.device)
            if initialized is not None
            else self.memory_v.flatten(1).count_nonzero(dim=1).bool()
        )

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

    def _path_vectors(self, x: torch.Tensor, segment_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Paper PaTH low-rank projection, causal width-3 depthwise conv, L2 norm."""
        projected = self.path_w_up(self.path_w_down(x))
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
    ) -> torch.Tensor:
        chunks = []
        memory_safe = True
        if attention_mask is not None:
            memory_safe = memory_safe and attention_mask.ndim == 2 and bool(attention_mask.all())
        if segment_ids is not None:
            memory_safe = memory_safe and bool((segment_ids == segment_ids[:, :1]).all())
        for start, end in self._chunk_ranges(q.size(2)):
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
                memory_k = self.memory_k.detach().clone()
                memory_v = self.memory_v.detach().clone()
                memory_scores = torch.matmul(
                    q[:, :, start:end], memory_k.transpose(-2, -1).to(dtype=q.dtype)
                ) * (self.head_dim ** -0.5)
                memory_context = torch.matmul(
                    torch.softmax(memory_scores, dim=-1),
                    memory_v.to(dtype=v.dtype),
                )
                gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
                mixed_context = (1.0 - gate) * local_context + gate * memory_context
                initialized = self.memory_initialized.clone().view(-1, 1, 1, 1)
                local_context = torch.where(initialized, mixed_context, local_context)
            chunks.append(local_context)
            if memory_safe and update_memory:
                with torch.no_grad():
                    self._update_memory(k[:, :, start:end], v[:, :, start:end])
        return torch.cat(chunks, dim=2)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Compress the current sequence into the fixed-size memory buffers."""
        batch_size, num_heads, seq_len, head_dim = k.shape

        def pool(values: torch.Tensor) -> torch.Tensor:
            bins = []
            for index in range(self.memory_dim):
                start = index * seq_len // self.memory_dim
                end = max(start + 1, ((index + 1) * seq_len + self.memory_dim - 1) // self.memory_dim)
                bins.append(values[:, :, start:end].mean(dim=2))
            return torch.stack(bins, dim=2)

        pooled_k = pool(k)
        pooled_v = pool(v)

        self.memory_k.mul_(0.99).add_(0.01 * pooled_k)
        self.memory_v.mul_(0.99).add_(0.01 * pooled_v)
        self.memory_initialized.fill_(True)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        update_memory: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        self._ensure_memory_batch(batch_size)
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        w = self._path_vectors(x, segment_ids)
        beta = 2.0 * torch.sigmoid(self.path_beta(x).float())
        log_forget = F.logsigmoid(self.path_forget(x).float())
        local_mask = attn_bias if attn_bias is not None else attention_mask
        requested = self.update_memory_buffers if update_memory is None else bool(update_memory)
        do_update = requested and self.update_memory_buffers
        context = self._local_path_attention(
            q, k, v, w, beta, log_forget, local_mask, segment_ids, do_update
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)

        return output
