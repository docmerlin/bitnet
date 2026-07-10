"""Device/AMP helpers, logging, optimizer wiring, evaluation."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from config import TernaryConfig
from data.presets import DatasetSource
from data.streams import build_batch_stream
from model import BitNetDeep
from optim import build_cmud
from training.memory import (
    capture_infini_memory_state,
    reset_infini_memory,
    restore_infini_memory_state,
)

try:
    from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
except ImportError:  # pragma: no cover
    HierarchicalTokenizer = Any  # type: ignore[misc, assignment]


def choose_device(device_arg: str = "auto") -> torch.device:
    """Resolve ``auto`` / explicit device strings for BitNet and BLT trainers."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class JsonlLogger:
    """JSONL metrics file + stdout. No TensorBoard/WandB."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        payload = {"step": step, "time": time.time(), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        summary_parts = [f"step={step}"]
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)):
                summary_parts.append(f"{key}={value:.6g}")
        print(" | ".join(summary_parts), flush=True)

    def close(self) -> None:
        return


def configure_mixed_precision(device: torch.device, precision: str) -> Tuple[bool, Optional[torch.dtype]]:
    if device.type == "mps":
        # MPS mixed precision is still less stable for long-running training.
        return False, None

    if precision == "fp32":
        return False, None
    if precision == "bf16":
        return device.type in {"cuda", "cpu"}, torch.bfloat16
    if precision == "fp16":
        return device.type == "cuda", torch.float16

    if device.type == "cuda":
        return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return False, None


def autocast_context(
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> contextlib.AbstractContextManager[Any]:
    if not amp_enabled or amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def build_model_config(args: argparse.Namespace, tokenizer: HierarchicalTokenizer) -> TernaryConfig:
    defaults = TernaryConfig()
    kwargs: dict = {
        "vocab_size": len(tokenizer),
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_heads,
        "head_dim": args.hidden_size // args.num_heads,
        "intermediate_size": args.intermediate_size,
        "rms_norm_eps": defaults.rms_norm_eps,
        "rope_theta": defaults.rope_theta,
        "rope_scaling": {
            "type": "yarn",
            "factor": args.rope_scaling_factor,
            "original_max_position_embeddings": defaults.rope_scaling["original_max_position_embeddings"],
        },
        "initializer_range": defaults.initializer_range,
        "block_size": args.final_blocks,
        "infini_memory_dim": defaults.infini_memory_dim,
        "use_hadamard": not args.disable_hadamard,
        "use_4bit_activations": True,
        "use_rfmoe": args.use_rfmoe,
        "rfmoe_num_experts": args.rfmoe_num_experts,
        "rfmoe_expert_dim": args.rfmoe_expert_dim,
        "rfmoe_theta": args.rfmoe_theta,
        "mtp_depth": args.mtp_depth,
    }

    prelude = getattr(args, "num_prelude_layers", None)
    recurrent = getattr(args, "num_recurrent_layers", None)
    coda = getattr(args, "num_coda_layers", None)
    loops = getattr(args, "num_loops", None)
    structure_set = any(v is not None for v in (prelude, recurrent, coda))

    if structure_set:
        kwargs["num_prelude_layers"] = prelude
        kwargs["num_recurrent_layers"] = recurrent
        kwargs["num_coda_layers"] = coda
        kwargs["num_loops"] = loops
    elif getattr(args, "num_layers", None) is not None:
        kwargs["num_hidden_layers"] = args.num_layers
        kwargs["num_loops"] = loops
    else:
        # Production looped default (8 / 48 / 8, R=4) via TernaryConfig resolution.
        kwargs["num_loops"] = loops

    return TernaryConfig(**kwargs)


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> Optimizer:
    return build_cmud(
        model,
        lr=args.mud_learning_rate,
        fallback_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.mud_momentum,
        passes=args.mud_passes,
        betas=(args.lion_beta1, args.lion_beta2),
        eight_bit=not args.no_optimizer_8bit,
    )


@torch.no_grad()
def evaluate(
    runner: nn.Module,
    tokenizer: HierarchicalTokenizer,
    mixture: List[Tuple[DatasetSource, float]],
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    if args.validation_batches <= 0:
        return {}

    was_training = runner.training
    memory_state = capture_infini_memory_state(runner)
    reset_infini_memory(runner)
    runner.eval()
    eval_stream = build_batch_stream(
        mixture,
        tokenizer,
        seed=args.seed + 999,
        shuffle=False,
        shuffle_buffer_size=args.shuffle_buffer_size,
        skip_examples=args.validation_offset_examples,
        restart_on_eof=True,
        sequence_length=args.sequence_length,
        max_document_tokens=args.max_document_tokens,
        micro_batch_size=args.micro_batch_size,
    )

    try:
        losses: List[float] = []
        for _ in range(args.validation_batches):
            batch = next(eval_stream)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            segment_ids = batch["segment_ids"].to(device)

            with autocast_context(device, amp_enabled, amp_dtype):
                logits = runner(input_ids, segment_ids=segment_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            losses.append(float(loss.item()))

        mean_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(mean_loss, 20.0))
        return {"val_loss": mean_loss, "val_perplexity": perplexity}
    finally:
        restore_infini_memory_state(runner, memory_state)
        runner.train(was_training)


__all__ = [
    "JsonlLogger",
    "autocast_context",
    "build_model_config",
    "choose_device",
    "configure_mixed_precision",
    "create_optimizer",
    "evaluate",
]
