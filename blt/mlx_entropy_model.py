"""MLX byte entropy model: the patcher, ported.

Numerical mirror of :mod:`blt.patching.entropy_model`. Parameter names and
``[out, in]`` weight layout match, so a model trained in either framework runs in
the other -- which matters here more than it does for the student, because the
patcher is shared between MLX training and torch generation.

The boundary rules are the same two from the BLT paper: a global entropy
threshold, and a monotonic rule that fires on a *rise* in entropy. ``entropy[t]``
describes byte ``t + 1``, so a boundary at ``j`` is decided by ``entropy[j - 1]``
-- a prediction made before byte ``j`` existed. That offset is what lets
generation ask about a byte it has not emitted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from blt.config import TernaryBLTConfig

ENTROPY_MODEL_EPS = 1e-5
_MASK_FLOOR = -3.4028234663852886e38


def next_byte_entropy(logits: mx.array) -> mx.array:
    """Shannon entropy, in nats, of each next-byte distribution."""
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    return -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)


def boundaries_from_entropy(
    entropy: mx.array,
    *,
    threshold: float | None = None,
    relative_threshold: float | None = None,
) -> mx.array:
    """Boolean [batch, seq] mask of positions that begin a patch."""
    if threshold is None and relative_threshold is None:
        raise ValueError("give threshold, relative_threshold, or both")

    batch, seq_len = entropy.shape
    if seq_len == 0:
        return mx.zeros((batch, 0), dtype=mx.bool_)
    leading = mx.ones((batch, 1), dtype=mx.bool_)
    if seq_len == 1:
        return leading

    predicted = entropy[:, :-1]  # entropy[j - 1] aligned to position j
    fires = mx.ones(predicted.shape, dtype=mx.bool_)
    if threshold is not None:
        fires = fires & (predicted > threshold)
    if relative_threshold is not None:
        # The first predicted position has no predecessor and cannot show a rise.
        rise = mx.concatenate(
            [mx.full((batch, 1), -mx.inf), predicted[:, 1:] - predicted[:, :-1]], axis=1
        )
        fires = fires & (rise > relative_threshold)
    return mx.concatenate([leading, fires], axis=1)


def cap_patch_lengths(starts: mx.array, max_patch_length: int) -> mx.array:
    """Force a boundary wherever a run would exceed ``max_patch_length``.

    Fixpoint rather than a per-row loop; each pass marks only positions sitting
    exactly ``max_patch_length`` past the last boundary, because marking every
    over-long position at once would turn the tail of a long run into
    all-boundaries.
    """
    if max_patch_length <= 0 or starts.shape[1] == 0:
        return starts

    seq_len = starts.shape[1]
    positions = mx.arange(seq_len, dtype=mx.int32).reshape(1, -1)
    for _ in range(seq_len // max_patch_length + 1):
        last_start = mx.cummax(mx.where(starts, positions, mx.array(-1, dtype=mx.int32)), axis=1)
        forced = (positions - last_start) == max_patch_length
        if not bool(mx.any(forced)):
            break
        starts = starts | forced
    return starts


def patch_lengths_from_starts(starts: mx.array) -> mx.array:
    """Patch widths [batch, max_patches] from a boundary mask, zero-padded."""
    batch, seq_len = starts.shape
    if seq_len == 0:
        return mx.zeros((batch, 0), dtype=mx.int32)

    patch_ids = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
    num_patches = int(mx.max(patch_ids).item()) + 1
    # Count membership by comparison rather than scatter: one kernel, no loop.
    lanes = mx.arange(num_patches, dtype=mx.int32).reshape(1, num_patches, 1)
    return mx.sum((patch_ids[:, None, :] == lanes).astype(mx.int32), axis=-1)


class _MLXEntropyBlock(nn.Module):
    """Pre-norm causal self-attention + GELU MLP, matching ``_EntropyBlock``."""

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

    def __call__(self, x: mx.array, causal_bias: mx.array) -> mx.array:
        batch, seq_len, dim = x.shape
        normed = self.attn_norm(x)

        def heads(t):
            return t.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        context = mx.fast.scaled_dot_product_attention(
            heads(self.q_proj(normed)),
            heads(self.k_proj(normed)),
            heads(self.v_proj(normed)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=causal_bias,
        )
        x = x + self.o_proj(context.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim))
        return x + self.down_proj(nn.gelu(self.up_proj(self.mlp_norm(x))))


class MLXByteEntropyModel(nn.Module):
    """Small full-precision byte LM whose next-byte entropy drives patching."""

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
            raise ValueError("MLXByteEntropyModel dim must be divisible by num_heads")
        self.config = config
        self.max_patch_length = config.max_patch_length
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(config.vocab_size, dim)
        self.position = nn.Embedding(max_seq_len, dim)
        self.blocks = [_MLXEntropyBlock(dim, num_heads) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(dim, eps=ENTROPY_MODEL_EPS)
        self.output_head = nn.Linear(dim, config.vocab_size)
        self._threshold = mx.array(math.log(256) / 2.0)
        self.freeze(keys=["_threshold"], recurse=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence of {seq_len} exceeds max_seq_len {self.max_seq_len}")
        hidden = self.embedding(input_ids) + self.position(mx.arange(seq_len))[None]
        rows = mx.arange(seq_len).reshape(-1, 1)
        causal_bias = mx.where(
            mx.arange(seq_len).reshape(1, -1) > rows, mx.array(_MASK_FLOOR), mx.array(0.0)
        )
        for block in self.blocks:
            hidden = block(hidden, causal_bias)
        return self.output_head(self.norm(hidden))

    def loss(self, input_ids: mx.array) -> mx.array:
        """Next-byte cross-entropy. This is the entire training objective."""
        logits = self(input_ids)
        return nn.losses.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
            reduction="mean",
        )

    def entropy(self, input_ids: mx.array) -> mx.array:
        return next_byte_entropy(self(input_ids))

    def predict_patch_lengths(
        self,
        input_ids: mx.array,
        *,
        threshold: float | None = None,
        relative_threshold: float | None = None,
    ) -> mx.array:
        threshold = self.default_threshold if threshold is None else threshold
        starts = boundaries_from_entropy(
            self.entropy(input_ids), threshold=threshold, relative_threshold=relative_threshold
        )
        return patch_lengths_from_starts(cap_patch_lengths(starts, self.max_patch_length))

    def opens_new_patch(self, input_ids: mx.array, *, threshold: float | None = None) -> mx.array:
        """Would the byte *after* ``input_ids`` begin a patch?"""
        threshold = self.default_threshold if threshold is None else threshold
        return self.entropy(input_ids)[:, -1] > threshold

    @property
    def default_threshold(self) -> float:
        return float(self._threshold)

    def set_threshold(self, threshold: float) -> None:
        self._threshold = mx.array(float(threshold))


def load_entropy_model(path: str | Path, config: TernaryBLTConfig) -> MLXByteEntropyModel:
    """Rebuild a saved entropy model from its weights and sidecar metadata.

    The sidecar carries the architecture and the calibrated threshold, so a
    caller does not have to remember dims that must match or supply a cutoff in
    nats alongside the file. A threshold that does not travel with the weights is
    a threshold that eventually gets paired with the wrong ones.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"entropy model at {path} has no {sidecar.name}; it carries the "
            "architecture and calibrated threshold and cannot be inferred"
        )
    meta = json.loads(sidecar.read_text())
    model = MLXByteEntropyModel(
        config,
        dim=meta["dim"],
        num_layers=meta["layers"],
        num_heads=meta["heads"],
        max_seq_len=meta.get("max_seq_len", 4096),
    )
    model.load_weights(str(path))
    model.set_threshold(meta["threshold"])
    model.eval()
    mx.eval(model.parameters())
    return model


def calibrate_threshold(
    model: MLXByteEntropyModel, input_ids: mx.array, *, target_patch_size: float
) -> float:
    """Pick the entropy threshold yielding ``target_patch_size`` bytes per patch.

    Patch size is what governs cost -- it sets how often the global model runs --
    but the knob is a cutoff in nats, and the mapping between them depends on the
    corpus and how well trained the model is. Solve for it rather than guess.
    """
    if target_patch_size <= 1.0:
        raise ValueError("target_patch_size must exceed 1.0")
    entropy = model.entropy(input_ids)[:, :-1].reshape(-1)
    if entropy.size == 0:
        return model.default_threshold
    # One boundary per `target_patch_size` bytes means the top 1/target of
    # predictions should clear the bar. MLX has no quantile, and sorting a
    # calibration sample is cheap next to the forward pass that produced it.
    ordered = mx.sort(entropy)
    index = min(int((1.0 - 1.0 / target_patch_size) * ordered.size), ordered.size - 1)
    threshold = float(ordered[index])
    model.set_threshold(threshold)
    return threshold
