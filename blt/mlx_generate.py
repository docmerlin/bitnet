"""Greedy byte-level generation for the MLX BLT student.

Port of :mod:`blt.generate`, including BLT-S self-speculation (Kallini et al.,
*Fast Byte Latent Transformer*, arXiv:2605.08044, Algorithm 2). Same two
strategies, same contract: under greedy decoding every ``speculation_window``
produces byte-identical output, so the window is purely a speed knob that trades
global-model passes for local-decoder ones.

Every correctness trap the torch version hit is carried over rather than
rediscovered:

- The verification pass doubles as the next round's encoder/global pass, so a
  speculative round costs one global forward, not two.
- ``_verify`` re-patches the *committed* tokens to decide which latents survive
  the rollback. The candidate's trailing patch is cut off by the end of the
  sequence and looks closed whether or not it is.
- Drafting runs through EOS and verification appends a free byte on top, so the
  result is trimmed at the first generated EOS.
- Boundaries are only knowable ahead of the byte for a patcher that predicts;
  see :attr:`_Patching.positional`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from blt.mlx_entropy_model import MLXByteEntropyModel, EntropyGenerationCache
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_patching import (
    build_uniform_patch_lengths,
    normalize_patch_lengths,
    patch_ids_from_lengths,
    patch_presence_mask,
)
from blt.patching.teacher_patcher import UniformPatcher


@dataclass
class GenerationStats:
    """Forward passes per component, plus draft acceptance."""

    encoder: int = 0
    draft_encoder: int = 0
    global_model: int = 0
    decoder: int = 0
    drafted: int = 0
    accepted: int = 0
    committed: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 1.0

    @property
    def bytes_per_global_pass(self) -> float:
        """The number BLT-S exists to raise. Baseline sits at the patch length."""
        return self.committed / self.global_model if self.global_model else 0.0


class _Patching:
    """Adapts a patcher to the questions generation actually asks it."""

    def __init__(self, patcher: object, *, threshold: float | None = None) -> None:
        if not isinstance(patcher, (UniformPatcher, MLXByteEntropyModel)):
            raise TypeError(f"unsupported patcher for generation: {type(patcher).__name__}")
        self.patcher = patcher
        self.threshold = threshold
        self.entropy_cache = EntropyGenerationCache(patcher, threshold) if isinstance(patcher, MLXByteEntropyModel) else None

    @property
    def uniform_patch_size(self) -> int | None:
        return self.patcher.patch_size if isinstance(self.patcher, UniformPatcher) else None

    @property
    def positional(self) -> bool:
        """Can the next byte's patch be known before that byte exists?

        Both supported patchers qualify -- ``UniformPatcher`` because boundaries
        depend only on position, ``MLXByteEntropyModel`` because it scores the
        *next* byte from bytes already committed. A retrospective classifier
        would not, and is refused outright rather than silently decoding against
        the wrong latent.
        """
        return True

    def patch_lengths(self, tokens: mx.array) -> mx.array:
        if isinstance(self.patcher, UniformPatcher):
            lengths = build_uniform_patch_lengths(
                tokens.shape[0], tokens.shape[1], self.patcher.patch_size
            )
        else:
            lengths = self.entropy_cache.patch_lengths(tokens)
        return normalize_patch_lengths(lengths, tokens.shape[1])

    def opens_new_patch(self, tokens: mx.array) -> bool:
        if isinstance(self.patcher, UniformPatcher):
            return tokens.shape[1] % self.patcher.patch_size == 0
        return self.entropy_cache.opens_new_patch(tokens)


def _ones_mask(tokens: mx.array) -> mx.array:
    """Every committed byte is real; generation never pads."""
    return mx.ones(tokens.shape, dtype=mx.bool_)


def _ngram_embed_window(model: MLXTernaryBLTModel) -> int:
    """Bytes the last position's hash n-grams can see, or 1 when n-grams are off."""
    ngrams = model.ngram_embeddings
    return 1 if ngrams is None else max(ngrams.sizes)


def _embed_last_byte(model: MLXTernaryBLTModel, tokens: mx.array) -> mx.array:
    """Embedding of the last byte, using only the n-gram suffix.

    Hash n-grams of size ``n`` need ``n - 1`` previous bytes; a lone tail token
    hashes as if padded. Transformer KV caches already hold the prefix, so the
    rest of the sequence does not need to be re-embedded.
    """
    window = min(tokens.shape[1], _ngram_embed_window(model))
    suffix = tokens[:, -window:]
    return model.embed_bytes(suffix, _ones_mask(suffix))[:, -1:]


def _run_global(
    model: MLXTernaryBLTModel, tokens: mx.array, patching: _Patching, stats: GenerationStats,
    draft_cache: _DraftCache | None = None,
) -> tuple[mx.array, mx.array]:
    """Full encoder + global pass over the committed prefix."""
    patch_lengths = patching.patch_lengths(tokens)
    embeds = model.embed_bytes(tokens, _ones_mask(tokens))
    prefilled = model.local_encoder.encode_bytes_prefill(embeds)
    _, patch_states, patch_ids = model.local_encoder(
        embeds, patch_lengths, attention_mask=_ones_mask(tokens), unpadded=True,
        prefilled=prefilled, uniform_patch_size=patching.uniform_patch_size,
    )
    if draft_cache is not None:
        draft_cache.encoder_hidden, draft_cache.encoder = prefilled
        draft_cache.length = tokens.shape[1]
    latents = model.global_transformer(
        patch_states, attention_mask=patch_presence_mask(patch_lengths)
    )
    stats.encoder += 1
    stats.global_model += 1
    return latents, patch_ids


@dataclass
class _DraftCache:
    """Self-attn K/V for encoder and decoder while drafting against frozen latents.

    Invalidated after every global pass. ``last_hidden`` holds the decoder
    hidden at the last committed position for the next argmax. Cross-attn
    projections of the frozen latents are reused for the cache lifetime.
    """

    encoder: list[tuple[mx.array, mx.array]] | None = None
    decoder: list[tuple[mx.array, mx.array]] | None = None
    length: int = 0
    last_hidden: mx.array | None = None
    prepared_latents: mx.array | None = None
    cross_projected: list | None = None
    encoder_hidden: mx.array | None = None


def _decoder_patches(latents: mx.array) -> mx.array:
    # Shift by one so a byte in patch i reads latents[i-1], matching the model's
    # forward. Shifting rather than truncating leaves room for patch index
    # len(latents) -- the not-yet-encoded patch freshly drafted bytes belong to.
    leading = mx.zeros((latents.shape[0], 1, latents.shape[2]), dtype=latents.dtype)
    return mx.concatenate([leading, latents], axis=1)


def _ensure_draft_prefill(
    model: MLXTernaryBLTModel,
    tokens: mx.array,
    latents: mx.array,
    patch_ids: mx.array,
    stats: GenerationStats,
    draft_cache: _DraftCache,
) -> None:
    """Prefill encoder+decoder caches when missing or desynced from ``tokens``."""
    length = tokens.shape[1]
    if (
        draft_cache.encoder is not None
        and draft_cache.decoder is not None
        and draft_cache.last_hidden is not None
        and draft_cache.prepared_latents is not None
        and draft_cache.length == length
    ):
        return
    if draft_cache.encoder_hidden is not None and draft_cache.length == length:
        hidden, enc_caches = draft_cache.encoder_hidden, draft_cache.encoder
    else:
        embeds = model.embed_bytes(tokens, _ones_mask(tokens))
        hidden, enc_caches = model.local_encoder.encode_bytes_prefill(embeds)
        stats.draft_encoder += 1
    decoder_patches = _decoder_patches(latents)
    latent, projected = model.local_decoder.prepare_cross_cache(decoder_patches)
    decoded, dec_caches = model.local_decoder.prefill(
        hidden, decoder_patches, patch_ids, latent=latent, cross_projected=projected
    )
    stats.decoder += 1
    draft_cache.encoder_hidden = None
    draft_cache.encoder = enc_caches
    draft_cache.decoder = dec_caches
    draft_cache.length = length
    draft_cache.last_hidden = decoded[:, -1:]
    draft_cache.prepared_latents = latent
    draft_cache.cross_projected = projected


def _draft(
    model: MLXTernaryBLTModel,
    tokens: mx.array,
    latents: mx.array,
    patch_ids: mx.array,
    *,
    next_id: int,
    count: int,
    stats: GenerationStats,
    draft_cache: _DraftCache | None = None,
) -> mx.array:
    """Extend ``tokens`` by ``count`` bytes, all charged to patch ``next_id``.

    Reuses self-attn K/V across drafted bytes. Pass a persistent ``draft_cache``
    from the outer generate loop so single-byte drafts still hit the fast path.
    """
    if draft_cache is None:
        draft_cache = _DraftCache()
    decoder_patches = _decoder_patches(latents)

    for _ in range(count):
        _ensure_draft_prefill(model, tokens, latents, patch_ids, stats, draft_cache)
        assert draft_cache.last_hidden is not None
        byte = mx.argmax(model.output_head(draft_cache.last_hidden), axis=-1)
        tokens = mx.concatenate([tokens, byte.astype(tokens.dtype)], axis=1)
        patch_ids = mx.concatenate(
            [patch_ids, mx.full((1, 1), next_id, dtype=patch_ids.dtype)], axis=1
        )
        # Fold the newly committed byte into the caches and refresh last_hidden
        # for the next prediction. Hash n-grams only need max(ngram_sizes)
        # trailing bytes; the self-attn cache already holds the prefix.
        offset = tokens.shape[1] - 1
        new_embed = _embed_last_byte(model, tokens)
        new_hidden, enc_caches = model.local_encoder.encode_bytes_extend(
            new_embed, draft_cache.encoder, offset=offset
        )
        stats.draft_encoder += 1
        if draft_cache.prepared_latents is None:
            latent, projected = model.local_decoder.prepare_cross_cache(decoder_patches)
            draft_cache.prepared_latents = latent
            draft_cache.cross_projected = projected
        decoded, dec_caches = model.local_decoder.extend(
            new_hidden,
            decoder_patches,
            patch_ids[:, -1:],
            draft_cache.decoder,
            offset=offset,
            latent=draft_cache.prepared_latents,
            cross_projected=draft_cache.cross_projected,
        )
        stats.decoder += 1
        draft_cache.encoder = enc_caches
        draft_cache.decoder = dec_caches
        draft_cache.length = tokens.shape[1]
        draft_cache.last_hidden = decoded

    return tokens


def _verify(
    model: MLXTernaryBLTModel,
    committed: mx.array,
    candidate: mx.array,
    patching: _Patching,
    stats: GenerationStats,
    draft_cache: _DraftCache | None = None,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Algorithm 2: accept drafted bytes up to the first mismatch."""
    prefilled = model.local_encoder.encode_bytes_prefill(model.embed_bytes(candidate, _ones_mask(candidate)))
    output = model(
        candidate,
        patch_lengths=patching.patch_lengths(candidate),
        attention_mask=_ones_mask(candidate),
        unpadded=True,
        uniform_patch_size=patching.uniform_patch_size,
        encoder_prefilled=prefilled,
    )
    stats.encoder += 1
    stats.global_model += 1
    stats.decoder += 1

    predictions = mx.argmax(output.logits, axis=-1)[0]
    start = committed.shape[1]
    drafted = candidate.shape[1] - start
    stats.drafted += drafted

    # One sync for the whole comparison rather than one per drafted byte.
    candidate_row = candidate[0].tolist()
    predicted_row = predictions.tolist()
    accepted = drafted
    for offset in range(drafted):
        if candidate_row[start + offset] != predicted_row[start + offset - 1]:
            accepted = offset
            break
    stats.accepted += accepted

    cut = start + accepted
    tokens = mx.concatenate(
        [candidate[:, :cut], mx.array([[predicted_row[cut - 1]]], dtype=candidate.dtype)], axis=1
    )
    if draft_cache is not None:
        # Prefill retains full K/V, so any rejected suffix can be removed.
        caches = [(k[:, :, :cut], v[:, :, :cut]) for k, v in prefilled[1]]
        hidden, caches = model.local_encoder.encode_bytes_extend(
            _embed_last_byte(model, tokens), caches, offset=cut
        )
        stats.draft_encoder += 1
        draft_cache.encoder_hidden = mx.concatenate([prefilled[0][:, :cut], hidden], axis=1)
        draft_cache.encoder = caches
        draft_cache.length = tokens.shape[1]

    # Ask the patcher where the resumed byte lands rather than inferring it from
    # the candidate's segmentation: the candidate's last patch is cut off by the
    # end of the sequence and so looks closed whether or not it is.
    next_id = int(patch_ids_from_lengths(patching.patch_lengths(tokens), tokens.shape[1])[0, -1].item())
    latents = output.global_hidden[:, :next_id]
    patch_ids = mx.concatenate(
        [output.patch_ids[:, :cut], mx.full((1, 1), next_id, dtype=output.patch_ids.dtype)], axis=1
    )
    return tokens, latents, patch_ids, next_id


def _trim_at_eos(tokens: mx.array, prompt_length: int, eos_id: int) -> tuple[mx.array, bool]:
    """Cut immediately after the first generated ``eos_id``."""
    generated = tokens[0, prompt_length:].tolist()
    if eos_id not in generated:
        return tokens, False
    return tokens[:, : prompt_length + generated.index(eos_id) + 1], True


def generate(
    model: MLXTernaryBLTModel,
    input_ids: mx.array,
    *,
    max_new_bytes: int,
    patcher: object | None = None,
    speculation_window: int = 0,
    boundary_threshold: float | None = None,
    eos_id: int | None = None,
) -> tuple[mx.array, GenerationStats]:
    """Greedily continue ``input_ids``; see the module docstring for the strategies.

    ponytail: batch size 1 only. Verification accepts a different number of bytes
    per row, so batching means ragged sequences and per-row bookkeeping.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("generation expects input_ids shaped [1, seq_len]")
    if input_ids.shape[1] == 0:
        raise ValueError("generation needs at least one prompt byte")
    if max_new_bytes < 0:
        raise ValueError("max_new_bytes must be non-negative")
    if speculation_window < 0:
        raise ValueError("speculation_window must be non-negative")
    if model.config.pad_id >= 0 and bool(mx.any(input_ids == model.config.pad_id)):
        raise ValueError("generation does not accept a padded prompt")

    if patcher is None:
        patcher = UniformPatcher(model.config.patch_size)
    patching = _Patching(patcher, threshold=boundary_threshold)
    if eos_id is None:
        eos_id = model.config.eos_id

    stats = GenerationStats()
    prompt_length = input_ids.shape[1]
    tokens = input_ids

    # Materialize ternarized weights once for the whole generate call. Without
    # this every draft byte re-scales and thresholds every HBitLinear.
    model.pin_inference_weights()
    try:
        return _generate_pinned(
            model,
            tokens,
            patching=patching,
            max_new_bytes=max_new_bytes,
            speculation_window=speculation_window,
            eos_id=eos_id,
            prompt_length=prompt_length,
            stats=stats,
        )
    finally:
        model.clear_pinned_inference_weights()


def _generate_pinned(
    model: MLXTernaryBLTModel,
    tokens: mx.array,
    *,
    patching: _Patching,
    max_new_bytes: int,
    speculation_window: int,
    eos_id: int,
    prompt_length: int,
    stats: GenerationStats,
) -> tuple[mx.array, GenerationStats]:
    draft_cache = _DraftCache()
    latents, patch_ids = _run_global(model, tokens, patching, stats, draft_cache)
    next_id = int(patch_ids[0, -1].item()) + int(patching.opens_new_patch(tokens))
    # Self-attn K/V live across drafted bytes; drop after each global pass.

    while tokens.shape[1] - prompt_length < max_new_bytes:
        budget = max_new_bytes - (tokens.shape[1] - prompt_length)

        opened = False
        if speculation_window == 0:
            while budget > 0:
                tokens = _draft(
                    model,
                    tokens,
                    latents,
                    patch_ids,
                    next_id=next_id,
                    count=1,
                    stats=stats,
                    draft_cache=draft_cache,
                )
                patch_ids = mx.concatenate(
                    [patch_ids, mx.full((1, 1), next_id, dtype=patch_ids.dtype)], axis=1
                )
                budget -= 1
                if int(tokens[0, -1].item()) == eos_id:
                    break
                # Remembered rather than asked again below: with an entropy
                # patcher this is a full forward of the patcher, and the answer
                # for the same tokens cannot differ.
                opened = patching.opens_new_patch(tokens)
                if opened:
                    break
        else:
            candidate = _draft(
                model,
                tokens,
                latents,
                patch_ids,
                next_id=next_id,
                count=min(speculation_window, budget),
                stats=stats,
                draft_cache=draft_cache,
            )
            draft_cache = _DraftCache()
            tokens, latents, patch_ids, next_id = _verify(model, tokens, candidate, patching, stats, draft_cache)

        tokens, finished = _trim_at_eos(tokens, prompt_length, eos_id)
        if finished:
            break

        if speculation_window == 0 and tokens.shape[1] - prompt_length < max_new_bytes:
            draft_cache = _DraftCache()
            latents, patch_ids = _run_global(model, tokens, patching, stats, draft_cache)
            next_id = int(patch_ids[0, -1].item()) + int(opened)

    tokens = tokens[:, : prompt_length + max_new_bytes]
    stats.committed = tokens.shape[1] - prompt_length
    return tokens, stats
