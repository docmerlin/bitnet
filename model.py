"""Main BitNetDeep model.

Every Transformer layer contains BOTH Infini-Attention and Attention Residuals.
Depth is looped: unique prelude once, recurrent stack × R, unique coda once.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import TernaryConfig
from layers.hybrid_block import HybridTransformerBlock
from layers.loop_mhc import LoopHyperConnection
from utils import (
    causal_block_attention_bias,
    combine_attention_bias,
    document_attention_keep_mask,
)

# Checkpoint granularity:
#   layer — max VRAM save (recompute each block); slowest
#   loop  — recompute each full mid-stack pass (default); fewer segments, higher peak
#           acts within a pass than layer mode
CheckpointGranularity = str  # "layer" | "loop"


class BitNetDeep(nn.Module):
    """Deep ternary LLM: hybrid blocks, ternary projections, tied embeddings.

    Control flow::

        embed → subln
        → prelude layers (once; Infini may write)
        → for r in 1..R:
              HC.project_in → recurrent stack → +e_r → HC.write_back
              (Infini: read every pass; write only on last r)
        → mean streams → coda → norm → lm_head

    Knobs: ``gradient_checkpointing`` + ``checkpoint_granularity`` (loop|layer),
    ``num_loops`` override on forward, ``return_mtp``.
    """

    def __init__(self, config: Optional[TernaryConfig] = None):
        super().__init__()
        self.config = config or TernaryConfig()
        self.gradient_checkpointing = False
        self.checkpoint_granularity: CheckpointGranularity = "loop"

        self.num_prelude = int(self.config.num_prelude_layers)
        self.num_recurrent = int(self.config.num_recurrent_layers)
        self.num_coda = int(self.config.num_coda_layers)

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.subln = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.layers = nn.ModuleList(
            HybridTransformerBlock(self.config)
            for _ in range(self.config.num_hidden_layers)
        )

        # Hyperloop loop-boundary HC (hardcoded 4 streams, diagonal H_res).
        self.loop_hc = LoopHyperConnection(
            hidden_size=self.config.hidden_size,
            rms_norm_eps=self.config.rms_norm_eps,
        )

        # Full-precision output projection; tied to embeddings.
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.mtp_depth = int(self.config.mtp_depth)
        if self.mtp_depth > 0:
            self.mtp_transforms = nn.ModuleList(
                nn.Sequential(
                    nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
                )
                for _ in range(self.mtp_depth)
            )

        self.apply(self._init_weights)
        self.lm_head.weight = self.embed_tokens.weight
        # Re-apply HC-friendly biases after global init (do not wipe with N(0, σ)).
        self.loop_hc._init_identity_friendly()

    def mtp_logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Per-depth logits from the final hidden. Depth i predicts offset i+2."""
        if self.mtp_depth <= 0:
            return []
        return [self.lm_head(transform(hidden)) for transform in self.mtp_transforms]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _build_shared_attention_bias(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        device_type = x.device.type
        compute_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else x.dtype
        )
        num_blocks = self.layers[0].num_blocks if len(self.layers) else 1
        base_bias = causal_block_attention_bias(
            seq_len, num_blocks, dtype=compute_dtype, device=x.device
        )
        return combine_attention_bias(
            attention_mask,
            base_bias=base_bias,
            batch_size=batch_size,
            q_len=seq_len,
            k_len=seq_len,
            dtype=compute_dtype,
            device=x.device,
        )

    @staticmethod
    def _snapshot_infini_states(
        layers: Sequence[HybridTransformerBlock],
    ) -> List[dict[str, torch.Tensor]]:
        return [layer.infini_attn.get_memory_state() for layer in layers]

    @staticmethod
    @contextlib.contextmanager
    def _freeze_infini_states(
        layers: Sequence[HybridTransformerBlock],
        states: Sequence[dict[str, torch.Tensor]],
    ):
        """On recompute: restore Infini banks and disable writes (no double-update)."""
        with contextlib.ExitStack() as stack:
            for layer, state in zip(layers, states):
                stack.enter_context(
                    layer.infini_attn.use_memory_state(state, update_memory_buffers=False)
                )
            yield

    def _run_layer(
        self,
        layer: HybridTransformerBlock,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        query_valid: Optional[torch.Tensor],
        *,
        update_memory: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run one block. Checkpoint when enabled and granularity is ``layer``."""
        do_ckpt = (
            self.gradient_checkpointing
            and self.training
            and self.checkpoint_granularity == "layer"
        )
        if do_ckpt:
            layer_memory_state = layer.infini_attn.get_memory_state()
            return checkpoint(
                lambda hidden_states, layer=layer, attn_bias=attn_bias, query_valid=query_valid, update_memory=update_memory: layer(
                    hidden_states,
                    attn_bias=attn_bias,
                    query_valid=query_valid,
                    update_memory=update_memory,
                ),
                x,
                use_reentrant=False,
                context_fn=lambda layer=layer, layer_memory_state=layer_memory_state: (
                    contextlib.nullcontext(),
                    layer.infini_attn.use_memory_state(
                        layer_memory_state, update_memory_buffers=False
                    ),
                ),
            )
        return layer(
            x,
            attn_bias=attn_bias,
            query_valid=query_valid,
            update_memory=update_memory,
        )

    def _run_stack(
        self,
        layers: Iterable[HybridTransformerBlock],
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        query_valid: Optional[torch.Tensor],
        *,
        update_memory: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run a sequence of blocks without per-layer checkpointing."""
        for layer in layers:
            x = layer(
                x,
                attn_bias=attn_bias,
                query_valid=query_valid,
                update_memory=update_memory,
            )
        return x

    def _run_recurrent_iteration(
        self,
        recurrent: Sequence[HybridTransformerBlock],
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        query_valid: Optional[torch.Tensor],
        *,
        update_memory: bool,
    ) -> torch.Tensor:
        """One full middle-stack pass; optional loop-granularity checkpoint."""
        use_loop_ckpt = (
            self.gradient_checkpointing
            and self.training
            and self.checkpoint_granularity == "loop"
            and len(recurrent) > 0
        )
        if not use_loop_ckpt:
            # Layer granularity (or ckpt off): each block handled in _run_layer.
            for layer in recurrent:
                x = self._run_layer(
                    layer,
                    x,
                    attn_bias,
                    query_valid,
                    update_memory=update_memory,
                )
            return x

        # Checkpoint the entire recurrent stack as one segment (R checkpoints total).
        layers_list = list(recurrent)
        mem_states = self._snapshot_infini_states(layers_list)

        def run_stack(
            hidden_states: torch.Tensor,
            layers_list=layers_list,
            attn_bias=attn_bias,
            query_valid=query_valid,
            update_memory=update_memory,
        ) -> torch.Tensor:
            return self._run_stack(
                layers_list,
                hidden_states,
                attn_bias,
                query_valid,
                update_memory=update_memory,
            )

        return checkpoint(
            run_stack,
            x,
            use_reentrant=False,
            context_fn=lambda layers_list=layers_list, mem_states=mem_states: (
                contextlib.nullcontext(),
                self._freeze_infini_states(layers_list, mem_states),
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
        segment_ids: Optional[torch.Tensor] = None,
        return_mtp: bool = False,
        num_loops: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        if reset_memory:
            for layer in self.layers:
                layer.infini_attn.reset_memory()

        if segment_ids is not None:
            attention_mask = document_attention_keep_mask(segment_ids)

        x = self.embed_tokens(input_ids)
        x = self.subln(x)
        attn_bias, query_valid = self._build_shared_attention_bias(x, attention_mask)

        p = self.num_prelude
        r = self.num_recurrent
        loops = int(self.config.num_loops if num_loops is None else num_loops)
        if loops < 1:
            raise ValueError("num_loops must be >= 1")

        # Prelude once (may seed Infini memory; recurrent early loops can read it).
        for layer in self.layers[:p]:
            x = self._run_layer(layer, x, attn_bias, query_valid)

        # Recurrent × R + Hyperloop HC. Infini B: read all loops, write last only.
        recurrent = self.layers[p : p + r]
        if r > 0:
            y = self.loop_hc.expand(x)  # (B, T, n=4, C)
            for loop_i in range(loops):
                write_memory = loop_i == loops - 1
                x_in, _h_pre, h_post, h_res = self.loop_hc.project_in(y)
                x_in = self._run_recurrent_iteration(
                    recurrent,
                    x_in,
                    attn_bias,
                    query_valid,
                    update_memory=write_memory,
                )
                e_l = self.loop_hc.loop_embedding(
                    loop_i, device=x_in.device, dtype=x_in.dtype
                )
                u = x_in + e_l
                y = self.loop_hc.write_back(y, u, h_post, h_res)
            x = self.loop_hc.fold(y)
        # else: no recurrent core — prelude output flows straight to coda

        # Coda once
        for layer in self.layers[p + r :]:
            x = self._run_layer(layer, x, attn_bias, query_valid)

        x = self.norm(x)
        logits = self.lm_head(x)
        if return_mtp:
            return logits, self.mtp_logits(x)
        return logits
