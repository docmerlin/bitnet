"""Progress-driven training schedules (quantization, blocks, RFMoE, LR)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from data.presets import DatasetSource
from layers.h_bitlinear import HBitLinear
from layers.hybrid_block import HybridTransformerBlock


def choose_stage_mixture(
    progress_ratio: float,
    *,
    early_mixture: List[Tuple[DatasetSource, float]],
    late_mixture: Optional[List[Tuple[DatasetSource, float]]],
    switch_ratio: float,
) -> Tuple[str, List[Tuple[DatasetSource, float]]]:
    """Return the active curriculum stage and mixture for the current progress."""
    progress_ratio = min(max(progress_ratio, 0.0), 1.0)
    if late_mixture is not None and progress_ratio >= switch_ratio:
        return "late", late_mixture
    return "early", early_mixture


def lr_schedule_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0

    if step < warmup_steps:
        return float(step + 1) / max(warmup_steps, 1)

    main_steps = max(total_steps - warmup_steps - cooldown_steps, 1)
    if step < warmup_steps + main_steps:
        progress = (step - warmup_steps) / main_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    cooldown_progress = (step - warmup_steps - main_steps) / max(cooldown_steps, 1)
    return max(min_lr_ratio * (1.0 - cooldown_progress), 0.0)


def update_quantization_schedule(model: nn.Module, token_progress: float, args) -> Dict[str, float]:
    token_progress = min(max(token_progress, 0.0), 1.0)

    if token_progress < args.stage1_ratio:
        stage_progress = token_progress / max(args.stage1_ratio, 1e-8)
        weight_mix = args.stage1_weight_mix_start + stage_progress * (1.0 - args.stage1_weight_mix_start)
        activation_mix = args.stage1_activation_mix_start + stage_progress * (1.0 - args.stage1_activation_mix_start)
        activation_bits = int(round(
            args.stage1_activation_bits
            - stage_progress * (args.stage1_activation_bits - args.final_activation_bits)
        ))
    else:
        weight_mix = 1.0
        activation_mix = 1.0
        activation_bits = args.final_activation_bits

    for module in model.modules():
        if isinstance(module, HBitLinear):
            module.set_quantization_state(
                weight_mix=weight_mix,
                activation_mix=activation_mix,
                activation_bits=activation_bits,
                enable_weight_quantization=True,
                enable_activation_quantization=True,
            )

    return {
        "quant_weight_mix": weight_mix,
        "quant_activation_mix": activation_mix,
        "quant_activation_bits": float(activation_bits),
    }


def update_block_growth(model: nn.Module, token_progress: float, args) -> int:
    token_progress = min(max(token_progress, 0.0), 1.0)
    if token_progress < args.block_growth_ratio:
        growth_progress = token_progress / max(args.block_growth_ratio, 1e-8)
        blocks = round(args.initial_blocks + growth_progress * (args.final_blocks - args.initial_blocks))
    else:
        blocks = args.final_blocks

    blocks = max(1, min(blocks, args.sequence_length))
    for module in model.modules():
        if isinstance(module, HybridTransformerBlock):
            module.num_blocks = blocks
    return blocks


def rfmoe_staircase_schedule(token_progress: float, args) -> Tuple[float, float]:
    """Flat->skew curriculum for the RFMoE locality target (roadmap step 4).

    Anneal from a FLAT target (s=0, alpha=1 -> uniform, so the whole expert
    population trains up and the tail learns to be competent) to the configured
    skew (s, alpha) over the first ``rfmoe_curriculum_ratio`` of training, then
    hold. With ratio<=0 there is no curriculum: the configured target applies
    from step 0. Returns ``(s, alpha)``.
    """
    s_end, alpha_end = args.rfmoe_zipf_s, args.rfmoe_uniform_alpha
    ratio = args.rfmoe_curriculum_ratio
    if ratio <= 0.0:
        return s_end, alpha_end
    frac = min(max(token_progress, 0.0) / ratio, 1.0)
    s = frac * s_end                      # 0 -> s_end   (uniform head -> skewed)
    alpha = 1.0 + frac * (alpha_end - 1.0)  # 1 -> alpha_end (full floor -> configured floor)
    return s, alpha
