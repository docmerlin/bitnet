"""Main BitNetDeep model.

Every Transformer layer contains BOTH Infini-Attention and residual path
(default: Kimi Block AttnRes). Depth is looped: unique prelude once,
recurrent stack × R, unique coda once.

**AttnRes + loops:** the depth-attention residual stream is **reset at each
Hyperloop boundary**. Cross-loop mixing is owned by ``LoopHyperConnection``;
AttnRes only sees depth within the current prelude / one recurrent pass / coda.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import TernaryConfig
from layers.attn_res import AttnResStream
from layers.hybrid_block import HybridTransformerBlock
from layers.loop_mhc import LoopHyperConnection

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

    Selected unique blocks inject Engram memory before attention.

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
            HybridTransformerBlock(self.config, layer_id=layer_id)
            for layer_id in range(self.config.num_hidden_layers)
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

    @staticmethod
    def _snapshot_infini_states(
        layers: Sequence[HybridTransformerBlock],
    ) -> List[dict[str, torch.Tensor]]:
        return [layer.infini_attn.get_memory_state() for layer in layers]

    @staticmethod
    @contextlib.contextmanager
    def _recompute_infini_states(
        layers: Sequence[HybridTransformerBlock],
        states: Sequence[dict[str, torch.Tensor]],
    ):
        """Recompute from original banks, then restore post-forward runtime state."""
        with contextlib.ExitStack() as stack:
            for layer, state in zip(layers, states):
                stack.enter_context(
                    layer.infini_attn.use_memory_state(state, update_memory_buffers=True)
                )
            yield

    @property
    def _kimi_mode(self) -> bool:
        return str(self.config.attn_res_mode) == "kimi"

    def _new_stream(self, seed: torch.Tensor) -> AttnResStream:
        """Start a fresh AttnRes segment (used at embed, each loop, coda seed).

        Uses first layer's mix modules as placeholders; each block swaps its own
        per-layer pseudo-query modules in ``forward_kimi``.
        """
        for layer in self.layers:
            if layer.attn_res_mix is not None and layer.mlp_res_mix is not None:
                return AttnResStream.start(
                    seed,
                    group_size=int(self.config.attn_res_group_size),
                    attn_mix=layer.attn_res_mix,
                    mlp_mix=layer.mlp_res_mix,
                )
        raise RuntimeError("kimi AttnRes requires at least one hybrid layer with depth mixes")

    def _run_layer(
        self,
        layer: HybridTransformerBlock,
        state: Union[torch.Tensor, AttnResStream],
        attention_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        *,
        update_memory: Optional[bool] = None,
    ) -> Union[torch.Tensor, AttnResStream]:
        """Run one block. Checkpoint when enabled and granularity is ``layer``."""
        do_ckpt = (
            self.gradient_checkpointing
            and self.training
            and self.checkpoint_granularity == "layer"
        )
        if do_ckpt:
            if self._kimi_mode:
                # Stream carries Python lists — layer ckpt not supported; fall through.
                do_ckpt = False
        if do_ckpt:
            layer_memory_state = layer.infini_attn.get_memory_state()
            return checkpoint(
                lambda hidden_states, layer=layer, attention_mask=attention_mask, segment_ids=segment_ids, input_ids=input_ids, update_memory=update_memory: layer(
                    hidden_states,
                    attention_mask,
                    segment_ids=segment_ids,
                    input_ids=input_ids,
                    update_memory=update_memory,
                ),
                state,
                use_reentrant=False,
                context_fn=lambda layer=layer, layer_memory_state=layer_memory_state: (
                    contextlib.nullcontext(),
                    layer.infini_attn.use_memory_state(
                        layer_memory_state, update_memory_buffers=True
                    ),
                ),
            )
        return layer(
            state,
            attention_mask,
            segment_ids=segment_ids,
            input_ids=input_ids,
            update_memory=update_memory,
        )

    def _run_stack(
        self,
        layers: Iterable[HybridTransformerBlock],
        state: Union[torch.Tensor, AttnResStream],
        attention_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        *,
        update_memory: Optional[bool] = None,
    ) -> Union[torch.Tensor, AttnResStream]:
        """Run a sequence of blocks without per-layer checkpointing."""
        for layer in layers:
            state = layer(
                state,
                attention_mask,
                segment_ids=segment_ids,
                input_ids=input_ids,
                update_memory=update_memory,
            )
        return state

    def _run_recurrent_iteration(
        self,
        recurrent: Sequence[HybridTransformerBlock],
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        *,
        update_memory: bool,
    ) -> torch.Tensor:
        """One full middle-stack pass; optional loop-granularity checkpoint.

        Kimi mode: **new AttnResStream from ``x``** so depth history does not
        cross Hyperloop iterations (HC owns cross-loop mixing).
        """
        use_loop_ckpt = (
            self.gradient_checkpointing
            and self.training
            and self.checkpoint_granularity == "loop"
            and len(recurrent) > 0
            and not self._kimi_mode  # stream state is not pure Tensor for ckpt
        )
        if self._kimi_mode:
            stream = self._new_stream(x)
            for layer in recurrent:
                stream = self._run_layer(
                    layer,
                    stream,
                    attention_mask,
                    segment_ids,
                    input_ids,
                    update_memory=update_memory,
                )
            return stream.hidden()

        if not use_loop_ckpt:
            for layer in recurrent:
                x = self._run_layer(
                    layer,
                    x,
                    attention_mask,
                    segment_ids,
                    input_ids,
                    update_memory=update_memory,
                )
            return x

        layers_list = list(recurrent)
        mem_states = self._snapshot_infini_states(layers_list)

        def run_stack(
            hidden_states: torch.Tensor,
            layers_list=layers_list,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            input_ids=input_ids,
            update_memory=update_memory,
        ) -> torch.Tensor:
            return self._run_stack(
                layers_list,
                hidden_states,
                attention_mask,
                segment_ids,
                input_ids,
                update_memory=update_memory,
            )

        return checkpoint(
            run_stack,
            x,
            use_reentrant=False,
            context_fn=lambda layers_list=layers_list, mem_states=mem_states: (
                contextlib.nullcontext(),
                self._recompute_infini_states(layers_list, mem_states),
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

        x = self.embed_tokens(input_ids)
        x = self.subln(x)

        p = self.num_prelude
        r = self.num_recurrent
        loops = int(self.config.num_loops if num_loops is None else num_loops)
        if loops < 1:
            raise ValueError("num_loops must be >= 1")

        kimi = self._kimi_mode
        # Prelude once (may seed Infini memory; recurrent early loops can read it).
        if kimi:
            stream = self._new_stream(x)
            for layer in self.layers[:p]:
                stream = self._run_layer(layer, stream, attention_mask, segment_ids, input_ids)
            x = stream.hidden()
        else:
            for layer in self.layers[:p]:
                x = self._run_layer(layer, x, attention_mask, segment_ids, input_ids)

        # Recurrent × R + Hyperloop HC. Infini B: read all loops, write last only.
        # Each loop iteration **resets** AttnRes depth history (see module docstring).
        recurrent = self.layers[p : p + r]
        if r > 0:
            y = self.loop_hc.expand(x)  # (B, T, n=4, C)
            for loop_i in range(loops):
                write_memory = loop_i == loops - 1
                x_in, _h_pre, h_post, h_res = self.loop_hc.project_in(y)
                x_in = self._run_recurrent_iteration(
                    recurrent,
                    x_in,
                    attention_mask,
                    segment_ids,
                    input_ids,
                    update_memory=write_memory,
                )
                e_l = self.loop_hc.loop_embedding(
                    loop_i, device=x_in.device, dtype=x_in.dtype
                )
                u = x_in + e_l
                y = self.loop_hc.write_back(y, u, h_post, h_res)
            x = self.loop_hc.fold(y)

        # Coda once (fresh AttnRes segment after HC fold).
        if kimi:
            stream = self._new_stream(x)
            for layer in self.layers[p + r :]:
                stream = self._run_layer(layer, stream, attention_mask, segment_ids, input_ids)
            x = stream.hidden()
        else:
            for layer in self.layers[p + r :]:
                x = self._run_layer(layer, x, attention_mask, segment_ids, input_ids)

        x = self.norm(x)
        logits = self.lm_head(x)
        if return_mtp:
            return logits, self.mtp_logits(x)
        return logits
