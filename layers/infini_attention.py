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
        self.update_memory_buffers = True

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Paper memory: M ∈ R^{d×d} associative bindings, z ∈ R^{d} key normalizer.
        self.register_buffer(
            "memory_m",
            torch.zeros(0, num_heads, self.head_dim, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_z",
            torch.zeros(0, num_heads, self.head_dim),
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
        self.memory_m = torch.zeros(batch_size, h, d, d, device=device)
        self.memory_z = torch.zeros(batch_size, h, d, device=device)
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

    @staticmethod
    def _sigma(x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map from Linear Transformer / Infini-attention paper."""
        return F.elu(x) + 1.0

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
        """Linear or Linear+Delta associative update (paper eqs. 8–9)."""
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

        # Carry M,z through path windows (BPTT within the forward; detach on store).
        memory_m = self.memory_m
        memory_z = self.memory_z
        memory_initialized = self.memory_initialized

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
                memory_context = self._retrieve_memory(
                    q[:, :, start:end], memory_m, memory_z
                ).to(dtype=v.dtype)
                gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
                mixed_context = (1.0 - gate) * local_context + gate * memory_context
                initialized = memory_initialized.view(-1, 1, 1, 1)
                local_context = torch.where(initialized, mixed_context, local_context)
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
