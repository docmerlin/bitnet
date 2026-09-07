"""Entropy-based patching from a small byte-level language model.

This is the patcher BLT actually calls for, and it differs from
:class:`blt.patching.student_entropy.StudentEntropyModel` in the way that
matters. That one is a boundary *classifier* distilled from a teacher's
segmentation: it scores position ``t`` from context including byte ``t``, so it
can say nothing about a byte that has not been generated, and it needs a teacher
to learn from at all.

This one is a *language model*. It predicts the next byte, and a boundary is
declared where that prediction is uncertain -- which is computable from context
strictly before the byte in question. Two consequences:

- It trains on raw bytes with plain cross-entropy. No teacher.
- It is causally usable at generation time, so the decoder can know whether the
  byte it is about to emit opens a patch. The retrospective classifier could
  not, which is why ``blt.generate`` has to re-patch after every byte with it.

Two boundary rules, both from the BLT paper. The global rule declares a boundary
wherever entropy exceeds a threshold. The monotonic rule declares one wherever
entropy *rises* by more than a threshold, which adapts to text whose overall
predictability drifts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.config import TernaryBLTConfig


def next_byte_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy, in nats, of each next-byte distribution.

    ``logits[:, t]`` scores byte ``t + 1``, so the returned ``entropy[:, t]``
    describes how uncertain the model is about the byte that has not arrived
    yet. That offset is the whole reason this is usable during generation.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def boundaries_from_entropy(
    entropy: torch.Tensor,
    *,
    threshold: float | None = None,
    relative_threshold: float | None = None,
) -> torch.Tensor:
    """Boolean [batch, seq] mask of positions that begin a patch.

    Position 0 always begins one. Position ``j`` begins one when the model was
    uncertain about it, which is ``entropy[j - 1]`` -- the prediction made
    *before* byte ``j`` existed.

    Supplying both thresholds requires both to fire.
    """
    if threshold is None and relative_threshold is None:
        raise ValueError("give threshold, relative_threshold, or both")

    batch, seq_len = entropy.shape
    starts = torch.zeros(batch, seq_len, dtype=torch.bool, device=entropy.device)
    if seq_len == 0:
        return starts
    starts[:, 0] = True
    if seq_len == 1:
        return starts

    predicted = entropy[:, :-1]  # entropy[j - 1] aligned to position j
    fires = torch.ones_like(predicted, dtype=torch.bool)
    if threshold is not None:
        fires &= predicted > threshold
    if relative_threshold is not None:
        # Rise relative to the previous prediction. The first predicted position
        # has no predecessor, so it cannot show a rise.
        rise = torch.full_like(predicted, float("-inf"))
        rise[:, 1:] = predicted[:, 1:] - predicted[:, :-1]
        fires &= rise > relative_threshold

    starts[:, 1:] = fires
    return starts


def cap_patch_lengths(starts: torch.Tensor, max_patch_length: int) -> torch.Tensor:
    """Force a boundary wherever a run would exceed ``max_patch_length``.

    One prefix scan over the original starts, then every positive multiple of
    the cap inside each original interval. Same positions as the old fixpoint
    that marked only the byte exactly ``max_patch_length`` past the last
    boundary each pass.
    """
    if max_patch_length <= 0 or starts.shape[1] == 0:
        return starts

    seq_len = starts.shape[1]
    positions = torch.arange(seq_len, device=starts.device).unsqueeze(0)
    last_start = torch.cummax(torch.where(starts, positions, -1), dim=1).values
    distance = positions - last_start
    return starts | ((distance > 0) & (distance % max_patch_length == 0))


def patch_lengths_from_starts(starts: torch.Tensor) -> torch.Tensor:
    """Patch widths [batch, max_patches] from a boundary mask, zero-padded."""
    batch, seq_len = starts.shape
    if seq_len == 0:
        return torch.zeros(batch, 0, dtype=torch.long, device=starts.device)

    patch_ids = starts.long().cumsum(dim=1) - 1
    num_patches = int(patch_ids.max().item()) + 1
    lengths = torch.zeros(batch, num_patches, dtype=torch.long, device=starts.device)
    lengths.scatter_add_(1, patch_ids, torch.ones_like(patch_ids))
    return lengths


ENTROPY_MODEL_EPS = 1e-5


class _EntropyBlock(nn.Module):
    """Pre-norm causal self-attention + GELU MLP.

    Written out rather than using ``nn.TransformerEncoderLayer`` so the MLX
    mirror in :mod:`blt.mlx_entropy_model` can match it exactly. The two
    frameworks' stock encoder layers differ in bias placement and norm details,
    which makes a checkpoint non-portable in ways that only show up as slightly
    wrong patch boundaries.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_norm = nn.RMSNorm(dim, eps=ENTROPY_MODEL_EPS)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.mlp_norm = nn.RMSNorm(dim, eps=ENTROPY_MODEL_EPS)
        self.up_proj = nn.Linear(dim, dim * 4)
        self.down_proj = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, causal_bias: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        normed = self.attn_norm(x)

        def heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            heads(self.q_proj(normed)),
            heads(self.k_proj(normed)),
            heads(self.v_proj(normed)),
            attn_mask=causal_bias,
        )
        x = x + self.o_proj(context.transpose(1, 2).reshape(batch, seq_len, dim))
        return x + self.down_proj(F.gelu(self.up_proj(self.mlp_norm(x))))


class ByteEntropyModel(nn.Module):
    """Small full-precision byte LM whose next-byte entropy drives patching.

    Kept deliberately separate from the ternary student: it is a fixed piece of
    preprocessing, it is tiny next to the global model, and quantising it would
    put noise straight into the patch boundaries.
    """

    def __init__(
        self,
        config: TernaryBLTConfig,
        *,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("ByteEntropyModel dim must be divisible by num_heads")
        self.config = config
        self.max_patch_length = config.max_patch_length
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(config.vocab_size, dim)
        self.position = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([_EntropyBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim, eps=ENTROPY_MODEL_EPS)
        self.output_head = nn.Linear(dim, config.vocab_size)
        # A buffer, not a flag, so a threshold calibrated against a corpus rides
        # along in the checkpoint instead of living in a caller's argument.
        # ln(256) ~ 5.55 nats is a uniform byte; half of it keeps a freshly built
        # model from declaring either one patch or all of them.
        self.register_buffer("_threshold", torch.tensor(math.log(256) / 2.0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-byte logits [batch, seq, vocab]."""
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence of {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.embedding(input_ids) + self.position(positions).unsqueeze(0)
        causal_bias = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(hidden.dtype).min, device=input_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, causal_bias)
        return self.output_head(self.norm(hidden))

    def loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-byte cross-entropy. This is the entire training objective."""
        logits = self.forward(input_ids)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
        )

    @torch.no_grad()
    def entropy(self, input_ids: torch.Tensor) -> torch.Tensor:
        return next_byte_entropy(self.forward(input_ids))

    @torch.no_grad()
    def predict_patch_lengths(
        self,
        input_ids: torch.Tensor,
        *,
        threshold: float | None = None,
        relative_threshold: float | None = None,
    ) -> torch.Tensor:
        threshold = self.default_threshold if threshold is None else threshold
        starts = boundaries_from_entropy(
            self.entropy(input_ids),
            threshold=threshold,
            relative_threshold=relative_threshold,
        )
        starts = cap_patch_lengths(starts, self.max_patch_length)
        return patch_lengths_from_starts(starts)

    @torch.no_grad()
    def opens_new_patch(self, input_ids: torch.Tensor, *, threshold: float | None = None) -> torch.Tensor:
        """Would the byte *after* ``input_ids`` begin a patch?

        Uses the same entropy threshold and length cap as full segmentation,
        depending only on bytes already committed.
        """
        threshold = self.default_threshold if threshold is None else threshold
        entropy = self.entropy(input_ids)
        # The dummy prediction exposes the next position, but is never read.
        starts = boundaries_from_entropy(
            torch.cat([entropy, entropy.new_zeros((entropy.size(0), 1))], dim=1),
            threshold=threshold,
        )
        return cap_patch_lengths(starts, self.max_patch_length)[:, -1]

    @property
    def default_threshold(self) -> float:
        """Entropy above which a byte is deemed unpredictable enough to split."""
        return float(self._threshold)

    def set_threshold(self, threshold: float) -> None:
        self._threshold.fill_(float(threshold))


@torch.no_grad()
def calibrate_threshold(
    model: ByteEntropyModel, input_ids: torch.Tensor, *, target_patch_size: float
) -> float:
    """Pick the entropy threshold that yields ``target_patch_size`` bytes per patch.

    Patch size is what actually governs cost -- it sets how often the global
    model runs -- but the knob is an entropy cutoff, and the mapping between
    them depends entirely on the corpus and how well trained the model is.
    Rather than have callers guess nats, solve for the quantile that produces
    the requested average width.
    """
    if target_patch_size <= 1.0:
        raise ValueError("target_patch_size must exceed 1.0")
    entropy = model.entropy(input_ids)[:, :-1].reshape(-1)
    if entropy.numel() == 0:
        return model.default_threshold
    # One boundary per `target_patch_size` bytes means the top 1/target of
    # predictions should clear the bar.
    quantile = 1.0 - 1.0 / target_patch_size
    threshold = float(torch.quantile(entropy.float(), quantile))
    model.set_threshold(threshold)
    return threshold
