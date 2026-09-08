"""Greedy byte-level generation for the ternary BLT student.

Two strategies, selected by ``speculation_window`` (default ``4``, BLT-S):

``0``
    Standard BLT autoregressive decoding. The local decoder emits bytes against
    the last available patch latent until the patcher opens a new patch, at which
    point the encoder and global model run again.

``k > 0``
    BLT-S self-speculation (Kallini et al., *Fast Byte Latent Transformer*,
    arXiv:2605.08044, Algorithm 2). The decoder ignores patch boundaries and
    drafts ``k`` bytes against the last available latent; one full forward pass
    then re-patches the candidate and accepts drafted bytes up to the first
    mismatch, replacing that byte with the model's own prediction.

Under greedy decoding both produce byte-identical output -- ``speculation_window``
is purely a speed knob. Pass ``0`` to restore the AR baseline. It buys that speed by trading global-model forward passes
for local-decoder ones, so it only pays off when the patcher is fine-grained
enough that the global model would otherwise run every couple of bytes. Check
``GenerationStats`` rather than assuming; past roughly ``1.14 * (k + 1)`` bytes
per patch the trade runs backwards and no acceptance rate rescues it.

A caveat on the ``0`` baseline. Skipping global passes between boundaries is only
sound when the next byte's patch is knowable before that byte exists.
``UniformPatcher`` qualifies -- boundaries are a function of position alone -- and
so does ``ByteEntropyModel``, which scores the *next* byte from bytes already
committed. ``StudentEntropyModel`` does not: it scores position ``t`` from
context *including* byte ``t``, so nothing can be said about a byte that has not
been emitted. With that patcher the baseline re-patches after every byte, which
is the correct greedy reference but runs the global model once per byte rather
than once per patch; ``bytes_per_global_pass`` reads 1.0 to make it visible. Do
not read a BLT-S speedup against that reference as the paper's speedup -- compare
against ``ByteEntropyModel``, which patches predictively as the paper's does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from blt.model import TernaryBLTModel
from blt.patching.entropy_model import ByteEntropyModel
from blt.patching.student_entropy import StudentEntropyModel
from blt.patching.teacher_patcher import (
    UniformPatcher,
    normalize_patch_lengths,
    patch_ids_from_lengths,
    patch_presence_mask,
)


@dataclass(slots=True)
class GenerationStats:
    """Forward passes per component, plus draft acceptance.

    Counted separately rather than folded into the paper's
    ``N_dec*P_dec + N_enc*(P_enc + P_glob)`` bandwidth formula: that formula
    assumes the encoder runs only as often as the global model, which holds when
    the encoder is rounding error next to the global model (19M vs 1.28B in the
    paper) but not for a student where it is an eighth of the parameters.
    """

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

    def memory_bandwidth_gb(self, model: TernaryBLTModel, *, bytes_per_param: float = 2.0) -> float:
        """Weight traffic for this run, the paper's efficiency metric.

        ``draft_encoder`` is charged less than ``encoder``: drafting calls
        ``encode_bytes``, which touches the encoder blocks but not the patch
        pooling projection or cross-attention. Charging draft passes the full
        encoder overstates BLT-S by ~10% on the default config, and this number
        is what decides whether speculation pays.

        ``bytes_per_param`` defaults to 2.0 (fp16) to stay comparable with the
        paper. A deployed ternary student moves far less; the ratio between
        settings is what carries over, not the absolute number.
        """

        def count(module: torch.nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        embeddings = count(model.byte_embeddings)
        byte_path = count(model.local_encoder.blocks) + count(model.local_encoder.output_norm)
        return (
            bytes_per_param
            * (
                self.encoder * (count(model.local_encoder) + embeddings)
                + self.draft_encoder * (byte_path + embeddings)
                + self.global_model * count(model.global_transformer)
                + self.decoder * (count(model.local_decoder) + count(model.output_head))
            )
            / 1e9
        )


class _Patching:
    """Adapts a patcher to the questions generation actually asks it."""

    def __init__(self, patcher: object, *, threshold: float | None = None) -> None:
        if not isinstance(patcher, (UniformPatcher, ByteEntropyModel, StudentEntropyModel)):
            raise TypeError(f"unsupported patcher for generation: {type(patcher).__name__}")
        self.patcher = patcher
        # None means "whatever the patcher considers its own". A ByteEntropyModel
        # carries a calibrated entropy cutoff in nats; forcing 0.0 on it would
        # clear every position and put a boundary on every byte.
        if threshold is None and isinstance(patcher, StudentEntropyModel):
            threshold = 0.0
        self.threshold = threshold

    @property
    def positional(self) -> bool:
        """Can the next byte's patch be known before that byte exists?

        True for ``UniformPatcher``, whose boundaries depend only on position,
        and for ``ByteEntropyModel``, which scores the *next* byte from the bytes
        already committed. False for ``StudentEntropyModel``, a retrospective
        classifier that scores position ``t`` using byte ``t`` -- it has to see
        the byte first, so the caller must re-patch after every byte instead of
        drifting to the next boundary.
        """
        return isinstance(self.patcher, (UniformPatcher, ByteEntropyModel))

    def patch_lengths(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.patcher, UniformPatcher):
            lengths = self.patcher.patch(tokens)
        else:
            lengths = self.patcher.predict_patch_lengths(tokens, threshold=self.threshold)
        return normalize_patch_lengths(lengths, tokens.size(1))

    def opens_new_patch(self, tokens: torch.Tensor) -> bool:
        """Does the byte after ``tokens`` start a new patch?

        Defined only where :attr:`positional` holds. Answering it for a
        retrospective classifier would mean reading the score of the last
        *committed* byte as though it described the next one, which disagrees
        with the segmentation ``patch_lengths`` produces and silently decodes
        against the wrong latent.
        """
        if not self.positional:
            raise RuntimeError("opens_new_patch is undefined for a retrospective patcher")
        if isinstance(self.patcher, UniformPatcher):
            return tokens.size(1) % self.patcher.patch_size == 0
        return bool(self.patcher.opens_new_patch(tokens, threshold=self.threshold)[0].item())


def _ones_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Every committed byte is real; generation never pads.

    Passed explicitly everywhere, including into ``model.forward``, so the draft
    and verify paths cannot disagree about which bytes count. Left implicit,
    ``forward`` would derive a mask from ``pad_id`` and reject a generated byte
    that happens to equal the pad token.
    """
    return torch.ones_like(tokens, dtype=torch.bool)


def _run_global(
    model: TernaryBLTModel,
    tokens: torch.Tensor,
    patching: _Patching,
    stats: GenerationStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full encoder + global pass over the committed prefix."""
    patch_lengths = patching.patch_lengths(tokens)
    byte_embeddings = model.embed_bytes(tokens, _ones_mask(tokens))
    _, patch_states, patch_ids = model.local_encoder(
        byte_embeddings, patch_lengths, attention_mask=_ones_mask(tokens)
    )
    latents = model.global_transformer(patch_states, attention_mask=patch_presence_mask(patch_lengths).long())
    stats.encoder += 1
    stats.global_model += 1
    return latents, patch_ids


def _ngram_embed_window(model: TernaryBLTModel) -> int:
    """Bytes the last position's hash n-grams can see, or 1 when n-grams are off."""
    ngrams = model.ngram_embeddings
    return 1 if ngrams is None else max(ngrams.sizes)


def _embed_last_byte(model: TernaryBLTModel, tokens: torch.Tensor) -> torch.Tensor:
    """Embedding of the last byte, using only the n-gram suffix."""
    window = min(tokens.size(1), _ngram_embed_window(model))
    suffix = tokens[:, -window:]
    return model.embed_bytes(suffix, _ones_mask(suffix))[:, -1:]


@dataclass(slots=True)
class _DraftCache:
    """Self-attn K/V for encoder and decoder while drafting against frozen latents.

    Invalidated after every global pass. Cross-attn projections of the frozen
    latents are reused for the cache lifetime.
    """

    encoder: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    decoder: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    length: int = 0
    last_hidden: torch.Tensor | None = None
    prepared_latents: torch.Tensor | None = None
    cross_projected: list | None = None


def _decoder_patches(latents: torch.Tensor) -> torch.Tensor:
    # new_zeros, not zeros_like(latents[:, :1]): after a rollback ``latents`` can
    # legitimately hold zero patches, and slicing an empty tensor stays empty.
    leading = latents.new_zeros((latents.size(0), 1, latents.size(2)))
    return torch.cat([leading, latents], dim=1)


def _ensure_draft_prefill(
    model: TernaryBLTModel,
    tokens: torch.Tensor,
    latents: torch.Tensor,
    patch_ids: torch.Tensor,
    stats: GenerationStats,
    draft_cache: _DraftCache,
) -> None:
    length = tokens.size(1)
    if (
        draft_cache.encoder is not None
        and draft_cache.decoder is not None
        and draft_cache.last_hidden is not None
        and draft_cache.prepared_latents is not None
        and draft_cache.length == length
    ):
        return
    attention_mask = _ones_mask(tokens)
    hidden, enc_caches = model.local_encoder.encode_bytes_prefill(
        model.embed_bytes(tokens, attention_mask)
    )
    stats.draft_encoder += 1
    decoder_patches = _decoder_patches(latents)
    latent, projected = model.local_decoder.prepare_cross_cache(decoder_patches)
    decoded, dec_caches = model.local_decoder.prefill(
        hidden, decoder_patches, patch_ids, latent=latent, cross_projected=projected
    )
    stats.decoder += 1
    draft_cache.encoder = enc_caches
    draft_cache.decoder = dec_caches
    draft_cache.length = length
    draft_cache.last_hidden = decoded[:, -1:]
    draft_cache.prepared_latents = latent
    draft_cache.cross_projected = projected


def _draft(
    model: TernaryBLTModel,
    tokens: torch.Tensor,
    latents: torch.Tensor,
    patch_ids: torch.Tensor,
    *,
    next_id: int,
    count: int,
    stats: GenerationStats,
    draft_cache: _DraftCache | None = None,
) -> torch.Tensor:
    """Extend ``tokens`` by ``count`` bytes, all charged to patch ``next_id``.

    Every drafted byte shares one patch id by design: BLT-S drafts *through*
    boundaries, so a byte that really belongs to a later patch is conditioned on
    a latent that predates it. Verification is what catches those.
    """
    if draft_cache is None:
        draft_cache = _DraftCache()
    decoder_patches = _decoder_patches(latents)

    for _ in range(count):
        _ensure_draft_prefill(model, tokens, latents, patch_ids, stats, draft_cache)
        assert draft_cache.last_hidden is not None
        byte = model.output_head(draft_cache.last_hidden).argmax(dim=-1)
        tokens = torch.cat([tokens, byte], dim=1)
        patch_ids = torch.cat([patch_ids, patch_ids.new_full((1, 1), next_id)], dim=1)
        offset = tokens.size(1) - 1
        new_hidden, enc_caches = model.local_encoder.encode_bytes_extend(
            _embed_last_byte(model, tokens), draft_cache.encoder, offset=offset
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
        draft_cache.length = tokens.size(1)
        draft_cache.last_hidden = decoded
    return tokens


def _verify(
    model: TernaryBLTModel,
    committed: torch.Tensor,
    candidate: torch.Tensor,
    patching: _Patching,
    stats: GenerationStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Algorithm 2: accept drafted bytes up to the first mismatch.

    Also hands back the latents this pass computed, so the next round drafts
    against them instead of paying for a second encoder/global pass. That reuse
    is what keeps BLT-S at one global pass per round -- the entire saving.

    A latent survives the rollback when its patch lies wholly inside the accepted
    prefix: those bytes are unchanged and every component is causal. The patch
    holding the replaced byte does not, so it is dropped. Keep one too few and
    the next round drafts a patch stale and eats the rejection; keep one too many
    and it reads a latent contaminated by bytes that were rolled back.

    Always returns at least one more byte than ``committed`` -- on a full-length
    match the verification pass hands back a free extra byte.
    """
    patch_lengths = patching.patch_lengths(candidate)
    output = model(candidate, patch_lengths=patch_lengths, attention_mask=_ones_mask(candidate))
    stats.encoder += 1
    stats.global_model += 1
    stats.decoder += 1

    # predictions[j] is the model's byte for position j + 1.
    predictions = output.logits.argmax(dim=-1)[0]
    start = committed.size(1)
    drafted = candidate.size(1) - start
    stats.drafted += drafted

    accepted = drafted
    for offset in range(drafted):
        if candidate[0, start + offset] != predictions[start + offset - 1]:
            accepted = offset
            break
    stats.accepted += accepted

    # One byte past the accepted run: the verified replacement, or the free byte.
    cut = start + accepted
    tokens = torch.cat([candidate[:, :cut], predictions[cut - 1].view(1, 1)], dim=1)

    # Ask the patcher where the resumed byte actually lands rather than inferring
    # it from the candidate's segmentation. The candidate's last patch is cut off
    # by the end of the sequence, so it looks closed whether or not it is, and a
    # content-scored patcher cannot place the replaced byte until it has seen it.
    # This is a second patcher call per round, but the patcher is small next to
    # the global model this whole routine exists to avoid.
    next_id = int(patch_ids_from_lengths(patching.patch_lengths(tokens), tokens.size(1))[0, -1].item())

    # A patch id of i reads latents[i-1], so next_id patches must survive. They
    # do: patch next_id - 1 lies wholly inside the unchanged prefix, and every
    # component is causal. patch_ids are non-decreasing, so the accepted bytes
    # all sit at or below next_id and stay indexable too.
    latents = output.global_hidden[:, :next_id]
    patch_ids = torch.cat(
        [output.patch_ids[:, :cut], output.patch_ids.new_full((1, 1), next_id)],
        dim=1,
    )
    return tokens, latents, patch_ids, next_id


def _trim_at_eos(tokens: torch.Tensor, prompt_length: int, eos_id: int) -> tuple[torch.Tensor, bool]:
    """Cut immediately after the first generated ``eos_id``.

    Speculation drafts through EOS and verification commits a free byte on top,
    so without this the speculative path returns bytes past the end of sequence
    that the baseline never emits -- breaking the one invariant that makes
    ``speculation_window`` a pure speed knob.
    """
    generated = tokens[0, prompt_length:].tolist()
    if eos_id not in generated:
        return tokens, False
    return tokens[:, : prompt_length + generated.index(eos_id) + 1], True


@torch.no_grad()
def generate(
    model: TernaryBLTModel,
    input_ids: torch.Tensor,
    *,
    max_new_bytes: int,
    patcher: object | None = None,
    speculation_window: int = 4,
    boundary_threshold: float | None = None,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, GenerationStats]:
    """Greedily continue ``input_ids``; see the module docstring for the strategies.

    ponytail: batch size 1 only. Verification accepts a different number of bytes
    per row, so batching means ragged sequences and per-row bookkeeping. Add it
    when batched eval throughput matters.
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("generation expects input_ids shaped [1, seq_len]")
    if input_ids.size(1) == 0:
        raise ValueError("generation needs at least one prompt byte")
    if max_new_bytes < 0:
        raise ValueError("max_new_bytes must be non-negative")
    if speculation_window < 0:
        raise ValueError("speculation_window must be non-negative")
    if model.config.pad_id >= 0 and bool((input_ids == model.config.pad_id).any()):
        # Batch size is 1, so padding is never needed; a padded prompt would be
        # encoded as content here and as padding inside model.forward.
        raise ValueError("generation does not accept a padded prompt")

    patching = _Patching(patcher if patcher is not None else model.fallback_patcher, threshold=boundary_threshold)
    if eos_id is None:
        eos_id = model.config.eos_id

    was_training = model.training
    model.eval()
    stats = GenerationStats()
    prompt_length = input_ids.size(1)
    tokens = input_ids

    try:
        latents, patch_ids = _run_global(model, tokens, patching, stats)
        next_id = int(patch_ids[0, -1].item()) + int(patching.positional and patching.opens_new_patch(tokens))
        draft_cache = _DraftCache()

        while tokens.size(1) - prompt_length < max_new_bytes:
            budget = max_new_bytes - (tokens.size(1) - prompt_length)

            opened = False
            if speculation_window == 0:
                # Emit against the current latent until the patcher opens a patch.
                # A retrospective patcher cannot say where the next boundary
                # falls, so it re-patches every byte instead of drifting.
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
                    patch_ids = torch.cat([patch_ids, patch_ids.new_full((1, 1), next_id)], dim=1)
                    budget -= 1
                    if tokens[0, -1].item() == eos_id:
                        break
                    if not patching.positional:
                        break
                    # Remembered rather than asked again below: with an entropy
                    # patcher this is a full forward of the patcher, and the
                    # answer for the same tokens cannot differ.
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
                tokens, latents, patch_ids, next_id = _verify(model, tokens, candidate, patching, stats)
                draft_cache = _DraftCache()

            tokens, finished = _trim_at_eos(tokens, prompt_length, eos_id)
            if finished:
                break

            if speculation_window == 0 and tokens.size(1) - prompt_length < max_new_bytes:
                latents, patch_ids = _run_global(model, tokens, patching, stats)
                next_id = int(patch_ids[0, -1].item()) + int(opened)
                draft_cache = _DraftCache()
    finally:
        model.train(was_training)

    # Verification's free byte can overshoot the budget.
    tokens = tokens[:, : prompt_length + max_new_bytes]
    stats.committed = tokens.size(1) - prompt_length
    return tokens, stats
