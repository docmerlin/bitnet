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
    """Clear cached RoPE tables. Useful in tests that vary device or shape often."""
    _build_rope_cache_cached.cache_clear()


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
