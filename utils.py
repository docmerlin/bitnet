"""Utility helpers for the deep ternary BitNet model."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by pairs for RoPE."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


@lru_cache(maxsize=128)
def _build_rope_cache_cached(
    seq_len: int,
    dim: int,
    theta: float,
    scaling_factor: float,
    device_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_key)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32) / scaling_factor
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def clear_rope_cache() -> None:
    """Clear cached RoPE and attention-bias tables. Useful in tests that vary device or shape often."""
    _build_rope_cache_cached.cache_clear()
    _causal_block_bias_cached.cache_clear()
    _causal_window_bias_cached.cache_clear()


def build_rope_cache(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine caches for rotary embeddings.

    ``scaling_factor`` provides a simple NTK/YaRN-style position stretch.
    Results are cached by shape/device/scaling parameters.
    """
    device_key = "cpu" if device is None else str(device)
    return _build_rope_cache_cached(seq_len, dim, theta, scaling_factor, device_key)


@lru_cache(maxsize=128)
def _causal_block_bias_cached(
    seq_len: int, num_blocks: int, dtype: torch.dtype, device_key: str
) -> torch.Tensor:
    device = torch.device(device_key)
    positions = torch.arange(seq_len, device=device)
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    if num_blocks > 1:
        block_size = (seq_len + num_blocks - 1) // num_blocks
        block_ids = positions.div(block_size, rounding_mode="floor")
        causal = causal | (block_ids[:, None] != block_ids[None, :])
    bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device).masked_fill(causal, torch.finfo(dtype).min)
    return bias.view(1, 1, seq_len, seq_len)


@lru_cache(maxsize=128)
def _causal_window_bias_cached(
    seq_len: int, window: int, dtype: torch.dtype, device_key: str
) -> torch.Tensor:
    device = torch.device(device_key)
    q_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    k_pos = torch.arange(seq_len, device=device).view(1, seq_len)
    invalid = k_pos > q_pos
    if window >= 0:
        invalid = invalid | ((q_pos - k_pos) >= window)
    bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device).masked_fill(invalid, torch.finfo(dtype).min)
    return bias.view(1, 1, seq_len, seq_len)


def causal_block_attention_bias(
    seq_len: int, num_blocks: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Cached additive bias for block-local causal attention ([1, 1, S, S])."""
    num_blocks = max(1, min(num_blocks, seq_len))
    return _causal_block_bias_cached(seq_len, num_blocks, dtype, str(device))


def causal_window_attention_bias(
    seq_len: int, window: int | None, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Cached additive bias for (optionally sliding-window) causal attention."""
    return _causal_window_bias_cached(seq_len, window if window is not None else -1, dtype, str(device))


def combine_attention_bias(
    attention_mask: torch.Tensor | None,
    *,
    base_bias: torch.Tensor | None,
    batch_size: int,
    q_len: int,
    k_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Fold a caller mask into a structural ``base_bias`` for one SDPA call.

    Returns ``(attn_bias, query_valid)``. ``query_valid`` is ``None`` when no mask
    is supplied; otherwise it marks rows the caller must zero after attention (fully
    masked query rows otherwise softmax into a synthetic distribution). Accepts a
    boolean keep-mask ``[B, q, k]``, a 2D suffix-padding mask ``[B, k]``, or a 3D/4D
    additive mask. ``base_bias`` is never mutated, so cached biases can be passed in.
    """
    if attention_mask is None:
        return base_bias, None

    mask_floor = torch.finfo(dtype).min
    if attention_mask.ndim == 2:
        # 2D key-padding mask [B, k] (bool or int), suffix-padding contract.
        key_valid = attention_mask[:, None, None, :k_len].to(torch.bool)
        if base_bias is None:
            attn_bias = torch.zeros(batch_size, 1, q_len, k_len, dtype=dtype, device=device)
        else:
            attn_bias = base_bias.expand(batch_size, 1, q_len, k_len).clone()
        attn_bias.masked_fill_(~key_valid, mask_floor)
        query_valid = attention_mask[:, None, :q_len, None].to(torch.bool)
    elif attention_mask.dtype == torch.bool:
        # Boolean keep-mask [B, q, k] (cross-attention).
        keep = attention_mask[:, None, :q_len, :k_len]
        query_valid = keep.any(dim=-1, keepdim=True)
        bias = torch.zeros(batch_size, 1, q_len, k_len, dtype=dtype, device=device).masked_fill(~keep, mask_floor)
        attn_bias = bias if base_bias is None else base_bias + bias
    elif attention_mask.ndim == 3:
        additive = attention_mask[:, None, :q_len, :k_len].to(dtype=dtype)
        attn_bias = additive if base_bias is None else base_bias + additive
        query_valid = additive.amax(dim=-1, keepdim=True) >= 0
    elif attention_mask.ndim == 4:
        additive = attention_mask[:, :, :q_len, :k_len].to(dtype=dtype)
        attn_bias = additive if base_bias is None else base_bias + additive
        query_valid = additive.amax(dim=-1, keepdim=True) >= 0
    else:
        raise ValueError("attention_mask must be a bool, 2D, 3D, or 4D tensor")
    return attn_bias, query_valid


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy (if installed), and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict:
    """Load a training checkpoint with ``weights_only=True`` when supported."""
    path = Path(checkpoint_path).expanduser().resolve()
    load_kwargs: dict = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def validate_suffix_padded_mask(mask: torch.Tensor, *, name: str = "attention_mask") -> None:
    """Reject masks where valid tokens are not a single left-aligned prefix."""
    if mask.numel() == 0:
        return
    if mask.ndim == 1:
        shifted = mask[1:].to(dtype=torch.bool) > mask[:-1].to(dtype=torch.bool)
        if torch.any(shifted):
            raise ValueError(f"{name} must use suffix-padded attention masks")
        return
    shifted = mask[:, 1:].to(dtype=torch.bool) > mask[:, :-1].to(dtype=torch.bool)
    if torch.any(shifted):
        raise ValueError(f"{name} must use suffix-padded attention masks")


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to ``x``.

    ``x`` is expected to be shaped ``[batch, heads, seq, head_dim]``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def ternary_quantize(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dense weights to ternary values with abs-mean scaling."""
    scale = weights.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    ternary = torch.where(
        weights > 0.5 * scale,
        torch.ones_like(weights),
        torch.where(weights < -0.5 * scale, -torch.ones_like(weights), torch.zeros_like(weights)),
    )
    return ternary.to(torch.int8), scale.squeeze(-1)
