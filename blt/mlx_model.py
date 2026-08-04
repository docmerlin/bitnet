"""MLX ternary BLT student.

Structural mirror of :mod:`blt.model`. Module and parameter names match the torch
stack exactly so a state dict transfers with no renaming, which is what the
parity tests rely on and what checkpoint conversion will rely on later.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from blt.config import TernaryBLTConfig
from blt.mlx_layers import (
    MLXHashNgramEmbedding,
    MLXHBitLinear,
    MLXTernaryCrossAttention,
    MLXTernaryPatchGather,
    MLXTransformerBlock,
)
from blt.mlx_patching import (
    build_uniform_patch_lengths,
    normalize_patch_lengths_to_targets,
    patch_ids_from_lengths,
    patch_membership_mask,
    patch_presence_mask,
    pool_patch_representations,
)


@dataclass
class MLXTernaryBLTOutput:
    logits: mx.array
    patch_lengths: mx.array
    patch_ids: mx.array
    encoder_hidden: mx.array
    encoder_patches: mx.array
    global_hidden: mx.array
    decoder_hidden: mx.array


def _validate_suffix_padded_mask(attention_mask: mx.array) -> None:
    """Padding must be a suffix; a hole in the middle breaks the patch arithmetic.

    Reading the result forces a GPU sync, which is why it is skippable -- see
    :attr:`MLXTernaryBLTModel.validate_inputs`.
    """
    if attention_mask.shape[1] == 0:
        return
    # Once a position is dropped every later position must be dropped too.
    dropped = ~attention_mask
    if bool(mx.any(attention_mask[:, 1:] & dropped[:, :-1])):
        raise ValueError("attention_mask must use suffix-padded attention masks")


class MLXLocalEncoder(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.blocks = [
            MLXTransformerBlock(
                config.local_dim,
                config.n_heads_local_encoder,
                config=config,
                ffn_multiplier=config.ffn_multiplier_local,
                local_window=config.local_window,
                causal=True,
            )
            for _ in range(config.n_layers_local_encoder)
        ]
        self.output_norm = nn.RMSNorm(config.local_dim, eps=1.1920928955078125e-07)
        self.patch_init_proj = (
            MLXHBitLinear(config.local_dim, config.global_dim, config=config)
            if config.local_dim != config.global_dim
            else None
        )
        self.patch_cross_attn = MLXTernaryCrossAttention(
            config.global_dim,
            config.local_dim,
            hidden_dim=config.global_dim,
            num_heads=config.n_heads_cross,
            config=config,
        )

    def encode_bytes(self, byte_embeddings: mx.array, *, attention_mask: mx.array | None = None) -> mx.array:
        """Byte-level hidden states only, skipping patch pooling and cross-attention."""
        hidden = byte_embeddings
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = self.output_norm(hidden)
        if attention_mask is not None:
            hidden = mx.where(attention_mask[..., None], hidden, 0.0)
        return hidden

    def encode_bytes_prefill(
        self, byte_embeddings: mx.array
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Generation prefill: full-prefix encode + per-block self-attn K/V."""
        hidden = byte_embeddings
        caches: list[tuple[mx.array, mx.array]] = []
        for block in self.blocks:
            hidden, cache = block.prefill(hidden)
            caches.append(cache)
        return self.output_norm(hidden), caches

    def encode_bytes_extend(
        self,
        byte_embeddings: mx.array,
        caches: list[tuple[mx.array, mx.array]],
        *,
        offset: int,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Generation step: encode only new byte positions against cached K/V."""
        hidden = byte_embeddings
        new_caches: list[tuple[mx.array, mx.array]] = []
        for block, cache in zip(self.blocks, caches):
            hidden, cache = block.extend(hidden, cache, offset=offset)
            new_caches.append(cache)
        return self.output_norm(hidden), new_caches

    def __call__(
        self,
        byte_embeddings: mx.array,
        patch_lengths: mx.array,
        *,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        byte_mask = None if attention_mask is None else attention_mask.astype(mx.bool_)
        hidden = self.encode_bytes(byte_embeddings, attention_mask=byte_mask)

        patch_ids = patch_ids_from_lengths(patch_lengths, hidden.shape[1])
        if byte_mask is not None:
            patch_ids = mx.where(byte_mask, patch_ids, -1)
        patch_states = pool_patch_representations(
            hidden, patch_lengths, patch_ids=patch_ids, token_mask=byte_mask, pooling="mean"
        )
        if self.patch_init_proj is not None:
            patch_states = self.patch_init_proj(patch_states)

        patch_mask = patch_membership_mask(patch_ids, patch_states.shape[1], patches_as_queries=True)
        if byte_mask is not None:
            patch_mask = patch_mask & byte_mask[:, None, :]
        patch_states = self.patch_cross_attn(patch_states, hidden, mask=patch_mask)
        return hidden, patch_states, patch_ids


class MLXGlobalTransformer(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.blocks = [
            MLXTransformerBlock(
                config.global_dim,
                config.n_heads_global,
                config=config,
                ffn_multiplier=config.ffn_multiplier_global,
                local_window=None,
                causal=True,
            )
            for _ in range(config.n_layers_global)
        ]
        self.output_norm = nn.RMSNorm(config.global_dim, eps=1.1920928955078125e-07)

    def __call__(self, patch_states: mx.array, *, attention_mask: mx.array | None = None) -> mx.array:
        hidden = patch_states
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.output_norm(hidden)


class MLXLocalDecoder(nn.Module):
    def __init__(self, config: TernaryBLTConfig) -> None:
        super().__init__()
        self.byte_state_proj = (
            MLXHBitLinear(config.local_dim, config.decoder_dim, config=config)
            if config.local_dim != config.decoder_dim
            else None
        )
        # At k > 1 the projection also fans each patch latent out into k slots,
        # so it is needed even when the widths already agree.
        self.cross_attn_k = config.cross_attn_k
        self.patch_state_proj = (
            MLXHBitLinear(config.global_dim, config.decoder_dim * self.cross_attn_k, config=config)
            if config.global_dim != config.decoder_dim or self.cross_attn_k > 1
            else None
        )
        # k == 1 leaves a one-hot mask, where attention is exactly a gather and
        # the query/key projections are provably inert; k > 1 gives the byte a
        # real choice among its patch's slots, so use actual attention.
        cross_attention = MLXTernaryCrossAttention if self.cross_attn_k > 1 else MLXTernaryPatchGather
        self.cross_attn_layers = [
            cross_attention(
                config.decoder_dim,
                config.decoder_dim,
                hidden_dim=config.decoder_dim,
                num_heads=config.n_heads_cross,
                config=config,
            )
            for _ in range(config.n_layers_local_decoder)
        ]
        self.blocks = [
            MLXTransformerBlock(
                config.decoder_dim,
                config.n_heads_local_decoder,
                config=config,
                ffn_multiplier=config.ffn_multiplier_decoder,
                local_window=config.local_window,
                causal=True,
            )
            for _ in range(config.n_layers_local_decoder)
        ]
        self.output_norm = nn.RMSNorm(config.decoder_dim, eps=1.1920928955078125e-07)

    def __call__(
        self,
        byte_states: mx.array,
        patch_states: mx.array,
        patch_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        hidden = self.byte_state_proj(byte_states) if self.byte_state_proj is not None else byte_states
        latent = self.patch_state_proj(patch_states) if self.patch_state_proj is not None else patch_states

        byte_mask = None
        if attention_mask is not None:
            byte_mask = attention_mask[:, : hidden.shape[1]].astype(mx.bool_)

        # patch_ids is -1 on padded bytes, which is exactly where the membership
        # mask is empty, so the two notions of "no patch" coincide.
        cross_valid = patch_ids >= 0
        if byte_mask is not None:
            cross_valid = cross_valid & byte_mask

        cross_mask = None
        if self.cross_attn_k > 1:
            batch_size, num_patches = patch_states.shape[0], patch_states.shape[1]
            # [B, P, D*k] -> [B, P*k, D], patch-major so slot i of patch p lands
            # at p*k + i, which is the order the repeated mask expects.
            latent = latent.reshape(batch_size, num_patches * self.cross_attn_k, -1)
            membership = patch_membership_mask(
                mx.maximum(patch_ids, 0), num_patches, patches_as_queries=False
            )
            cross_mask = mx.repeat(membership, self.cross_attn_k, axis=-1) & cross_valid[..., None]

        for cross_attn, block in zip(self.cross_attn_layers, self.blocks):
            hidden = (
                cross_attn(hidden, latent, mask=cross_mask)
                if self.cross_attn_k > 1
                else cross_attn(hidden, latent, patch_ids, valid=cross_valid)
            )
            if byte_mask is not None:
                hidden = mx.where(byte_mask[..., None], hidden, 0.0)
            hidden = block(hidden, attention_mask=attention_mask)
            if byte_mask is not None:
                hidden = mx.where(byte_mask[..., None], hidden, 0.0)

        hidden = self.output_norm(hidden)
        if byte_mask is not None:
            hidden = mx.where(byte_mask[..., None], hidden, 0.0)
        return hidden


    def _prepare_latents(self, patch_states: mx.array) -> mx.array:
        latent = self.patch_state_proj(patch_states) if self.patch_state_proj is not None else patch_states
        if self.cross_attn_k > 1:
            batch_size, num_patches = patch_states.shape[0], patch_states.shape[1]
            latent = latent.reshape(batch_size, num_patches * self.cross_attn_k, -1)
        return latent

    def prefill(
        self,
        byte_states: mx.array,
        patch_states: mx.array,
        patch_ids: mx.array,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Full-prefix decoder for generation; returns last-pos-ready state + self-attn caches."""
        hidden = self.byte_state_proj(byte_states) if self.byte_state_proj is not None else byte_states
        latent = self._prepare_latents(patch_states)
        cross_valid = patch_ids >= 0
        cross_mask = None
        if self.cross_attn_k > 1:
            membership = patch_membership_mask(
                mx.maximum(patch_ids, 0), patch_states.shape[1], patches_as_queries=False
            )
            cross_mask = mx.repeat(membership, self.cross_attn_k, axis=-1) & cross_valid[..., None]

        caches: list[tuple[mx.array, mx.array]] = []
        for cross_attn, block in zip(self.cross_attn_layers, self.blocks):
            hidden = (
                cross_attn(hidden, latent, mask=cross_mask)
                if self.cross_attn_k > 1
                else cross_attn(hidden, latent, patch_ids, valid=cross_valid)
            )
            hidden, cache = block.prefill(hidden)
            caches.append(cache)
        return self.output_norm(hidden), caches

    def extend(
        self,
        byte_states: mx.array,
        patch_states: mx.array,
        patch_ids: mx.array,
        caches: list[tuple[mx.array, mx.array]],
        *,
        offset: int,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Decode only new positions (typically one byte) with self-attn K/V cache."""
        hidden = self.byte_state_proj(byte_states) if self.byte_state_proj is not None else byte_states
        latent = self._prepare_latents(patch_states)
        cross_valid = patch_ids >= 0
        cross_mask = None
        if self.cross_attn_k > 1:
            membership = patch_membership_mask(
                mx.maximum(patch_ids, 0), patch_states.shape[1], patches_as_queries=False
            )
            cross_mask = mx.repeat(membership, self.cross_attn_k, axis=-1) & cross_valid[..., None]

        new_caches: list[tuple[mx.array, mx.array]] = []
        for cross_attn, block, cache in zip(self.cross_attn_layers, self.blocks, caches):
            # Cross-attn over frozen latents is pointwise in query length; run only
            # on the new rows. For k=1 gather, only the new patch_ids matter.
            hidden = (
                cross_attn(hidden, latent, mask=cross_mask)
                if self.cross_attn_k > 1
                else cross_attn(hidden, latent, patch_ids, valid=cross_valid)
            )
            hidden, cache = block.extend(hidden, cache, offset=offset)
            new_caches.append(cache)
        return self.output_norm(hidden), new_caches


class MLXTernaryBLTModel(nn.Module):
    def __init__(self, config: TernaryBLTConfig, *, global_transformer: nn.Module | None = None) -> None:
        """``global_transformer`` swaps the patch-level backbone.

        Defaults to BLT's own plain stack. Pass
        :class:`blt.mlx_global.MLXBitNetGlobalTransformer` to run the BitNet
        model -- PaTH, Infini, RFMoE -- over the patch latents instead.
        Anything with a ``(patch_states, attention_mask=...) -> latents``
        signature works.
        """
        super().__init__()
        self.config = config
        self.byte_embeddings = nn.Embedding(config.vocab_size, config.local_dim)
        self.ngram_embeddings = MLXHashNgramEmbedding(config) if config.use_ngram_embeddings else None
        self.local_encoder = MLXLocalEncoder(config)
        self.global_transformer = global_transformer or MLXGlobalTransformer(config)
        self.local_decoder = MLXLocalDecoder(config)
        self.output_head = MLXHBitLinear(config.decoder_dim, config.vocab_size, config=config)
        self.patch_size = config.patch_size
        # Input validation reads array contents, which forces a GPU sync and
        # makes the forward uncompilable ("Attempting to eval an array during
        # function transformations"). On by default so a stray bad mask is
        # caught; a training loop that validates its own batches -- cheaply, in
        # numpy, before they ever reach the GPU -- turns it off to compile.
        self.validate_inputs = True

    def set_quantization_state(self, weight_mix: float, activation_mix: float, bits: int) -> None:
        """Ramp quantisation across every ternary projection, backbone included.

        Starting at full 4-bit activations diverges -- see MLXHBitLinear -- so
        training ramps rather than fixing them at 1.0.
        """
        from blt.mlx_layers import MLXHBitLinear as _BLTHBitLinear

        def update(_, module):
            if isinstance(module, _BLTHBitLinear):
                module.set_quantization_state(weight_mix, activation_mix, bits)

        self.apply_to_modules(update)
        backbone = getattr(self.global_transformer, "backbone", None)
        if backbone is not None and hasattr(backbone, "set_quantization_state"):
            backbone.set_quantization_state(weight_mix, activation_mix, bits)

    def pin_inference_weights(self) -> None:
        """Pin effective ternary weights for generation; skip per-token rematerialize.

        Also pins the BitNet global backbone when present (packed path when enabled).
        """
        from blt.mlx_layers import MLXHBitLinear as _BLTHBitLinear

        def pin(_, module):
            if isinstance(module, _BLTHBitLinear):
                module.pin_inference_weight()

        self.apply_to_modules(pin)
        pinned = [
            module._pinned_weight
            for _, module in self.named_modules()
            if isinstance(module, _BLTHBitLinear) and module._pinned_weight is not None
        ]
        if pinned:
            mx.eval(*pinned)
        backbone = getattr(self.global_transformer, "backbone", None)
        if backbone is not None and hasattr(backbone, "pin_inference_weights"):
            backbone.pin_inference_weights()

    def clear_pinned_inference_weights(self) -> None:
        from blt.mlx_layers import MLXHBitLinear as _BLTHBitLinear

        def clear(_, module):
            if isinstance(module, _BLTHBitLinear):
                module.clear_pinned_inference_weight()

        self.apply_to_modules(clear)
        backbone = getattr(self.global_transformer, "backbone", None)
        if backbone is not None and hasattr(backbone, "clear_pinned_inference_weights"):
            backbone.clear_pinned_inference_weights()
        elif backbone is not None:

            def clear_backbone(_, module):
                if hasattr(module, "clear_pinned_inference_weight"):
                    module.clear_pinned_inference_weight()

            backbone.apply_to_modules(clear_backbone)

    def embed_bytes(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """Byte embeddings, plus hashed n-gram embeddings when enabled.

        Every path into the model goes through here -- training and both
        generation paths -- so the n-grams cannot be silently skipped at decode.
        """
        embeddings = self.byte_embeddings(input_ids)
        if self.ngram_embeddings is None:
            return embeddings
        return self.ngram_embeddings(embeddings, input_ids, attention_mask)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        patch_lengths: mx.array | None = None,
    ) -> MLXTernaryBLTOutput:
        if attention_mask is None:
            if self.config.pad_id >= 0:
                attention_mask = input_ids != self.config.pad_id
            else:
                attention_mask = mx.ones(input_ids.shape, dtype=mx.bool_)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        attention_mask = attention_mask.astype(mx.bool_)
        if self.validate_inputs:
            _validate_suffix_padded_mask(attention_mask)
        valid_lengths = mx.sum(attention_mask.astype(mx.int32), axis=1)

        if patch_lengths is None:
            patch_lengths = build_uniform_patch_lengths(
                input_ids.shape[0], input_ids.shape[1], self.patch_size
            )
        patch_lengths = normalize_patch_lengths_to_targets(patch_lengths, valid_lengths)

        byte_embeddings = self.embed_bytes(input_ids, attention_mask)
        encoder_hidden, encoder_patches, patch_ids = self.local_encoder(
            byte_embeddings, patch_lengths, attention_mask=attention_mask
        )
        global_mask = patch_presence_mask(patch_lengths)
        global_hidden = self.global_transformer(encoder_patches, attention_mask=global_mask)
        decoder_patches = mx.concatenate(
            [mx.zeros_like(global_hidden[:, :1]), global_hidden[:, :-1]], axis=1
        )
        decoder_hidden = self.local_decoder(
            encoder_hidden, decoder_patches, patch_ids, attention_mask=attention_mask
        )
        return MLXTernaryBLTOutput(
            logits=self.output_head(decoder_hidden),
            patch_lengths=patch_lengths,
            patch_ids=patch_ids,
            encoder_hidden=encoder_hidden,
            encoder_patches=encoder_patches,
            global_hidden=global_hidden,
            decoder_hidden=decoder_hidden,
        )
