"""Production training entrypoint for the deep ternary BitNet model.

Features:
- streaming Hugging Face mixtures (FineWeb-Edu + DCLM by default)
- built-in programming dataset presets via CodeSearchNet language shards and an all-language alias
- built-in math dataset presets via FineMath and OpenWebMath
- hierarchical tokenization with sequence packing
- two-stage quantization schedule for ternary warmup -> full QAT
- progressive Block Attention Residual growth
- Lion optimizer with decoupled weight decay
- cosine LR schedule with warmup and cooldown
- optional gradient checkpointing, mixed precision, torch.compile, validation,
  checkpointing, TensorBoard, and WandB logging

Example:
    python3 train.py \
        --output-dir runs/bitnet \
        --early-train-mixture fineweb_edu=0.60,dclm=0.25,code_search_net_all=0.10,finemath_3plus=0.05 \
        --late-train-mixture fineweb_edu=0.35,dclm=0.15,code_search_net_all=0.20,finemath_3plus=0.30 \
        --mixture-switch-ratio 0.70 \
        --val-mixture fineweb_edu=0.40,code_search_net_all=0.20,finemath_3plus=0.40 \
        --sequence-length 1024 \
        --micro-batch-size 1 \
        --grad-accumulation-steps 16 \
        --total-tokens 50000000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint

from config import TernaryConfig, config as default_config
from layers.hybrid_block import HybridTransformerBlock
from layers.h_bitlinear import HBitLinear
from layers.infini_attention import InfiniAttention
from model import BitNetDeep

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for training only.
    load_dataset = None

try:
    from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
    _TOKENIZER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional dependency for training only.
    HierarchicalTokenizer = Any
    _TOKENIZER_IMPORT_ERROR = exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency.
    SummaryWriter = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency.
    wandb = None


COMMON_TEXT_FIELDS = ("text", "content", "raw_content", "document", "body")


@dataclass(frozen=True)
class DatasetSource:
    alias: str
    path: str
    config_name: Optional[str]
    split: str
    text_field: str


DATASET_PRESETS: Dict[str, DatasetSource] = {
    "fineweb_edu": DatasetSource(
        alias="fineweb_edu",
        path="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        text_field="text",
    ),
    "dclm": DatasetSource(
        alias="dclm",
        path="mlfoundations/dclm-baseline-1.0",
        config_name=None,
        split="train",
        text_field="text",
    ),
    "c4": DatasetSource(
        alias="c4",
        path="allenai/c4",
        config_name="en",
        split="train",
        text_field="text",
    ),
    "finemath_3plus": DatasetSource(
        alias="finemath_3plus",
        path="HuggingFaceTB/finemath",
        config_name="finemath-3plus",
        split="train",
        text_field="text",
    ),
    "open_web_math": DatasetSource(
        alias="open_web_math",
        path="open-web-math/open-web-math",
        config_name=None,
        split="train",
        text_field="text",
    ),
    "code_search_net_python": DatasetSource(
        alias="code_search_net_python",
        path="code-search-net/code_search_net",
        config_name="python",
        split="train",
        text_field="whole_func_string",
    ),
    "code_search_net_go": DatasetSource(
        alias="code_search_net_go",
        path="code-search-net/code_search_net",
        config_name="go",
        split="train",
        text_field="whole_func_string",
    ),
    "code_search_net_javascript": DatasetSource(
        alias="code_search_net_javascript",
        path="code-search-net/code_search_net",
        config_name="javascript",
        split="train",
        text_field="whole_func_string",
    ),
    "code_search_net_java": DatasetSource(
        alias="code_search_net_java",
        path="code-search-net/code_search_net",
        config_name="java",
        split="train",
        text_field="whole_func_string",
    ),
    "code_search_net_php": DatasetSource(
        alias="code_search_net_php",
        path="code-search-net/code_search_net",
        config_name="php",
        split="train",
        text_field="whole_func_string",
    ),
    "code_search_net_ruby": DatasetSource(
        alias="code_search_net_ruby",
        path="code-search-net/code_search_net",
        config_name="ruby",
        split="train",
        text_field="whole_func_string",
    ),
}


MIXTURE_GROUP_PRESETS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    # Weighted toward the most broadly useful languages while still covering
    # every CodeSearchNet shard the trainer can stream without gated access.
    "code_search_net_all": (
        ("code_search_net_python", 0.30),
        ("code_search_net_javascript", 0.22),
        ("code_search_net_java", 0.18),
        ("code_search_net_go", 0.15),
        ("code_search_net_php", 0.08),
        ("code_search_net_ruby", 0.07),
    ),
}


@dataclass
class TrainerState:
    step: int = 0
    tokens_processed: int = 0
    samples_processed: int = 0
    best_val_loss: float = float("inf")


class Lion(Optimizer):
    """Lion optimizer with decoupled weight decay.

    Lion is a good fit for ternary/QAT setups because its sign-based update is
    robust to noisy gradients while still tracking full-precision shadow weights.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Lion learning rate must be positive")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Lion betas must be in [0, 1)")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["exp_avg"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                param.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


class TrainingWrapper(nn.Module):
    """Wrapper around ``BitNetDeep`` that adds optional activation checkpointing."""

    def __init__(self, model: BitNetDeep, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.model = model
        self.gradient_checkpointing = gradient_checkpointing

    @staticmethod
    def _checkpoint_context(layer: nn.Module, memory_state: Optional[Dict[str, torch.Tensor]] = None):
        infini_attn = getattr(layer, "infini_attn", None)
        if isinstance(infini_attn, InfiniAttention) and memory_state is not None:
            return contextlib.nullcontext(), infini_attn.use_memory_state(memory_state, update_memory_buffers=False)
        return contextlib.nullcontext(), contextlib.nullcontext()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> torch.Tensor:
        if reset_memory:
            reset_infini_memory(self.model)

        x = self.model.embed_tokens(input_ids)
        x = self.model.subln(x)

        for layer in self.model.layers:
            if self.gradient_checkpointing and self.training:
                layer_memory_state = None
                infini_attn = getattr(layer, "infini_attn", None)
                if isinstance(infini_attn, InfiniAttention):
                    layer_memory_state = infini_attn.get_memory_state()
                x = checkpoint(
                    lambda hidden_states, layer=layer, attention_mask=attention_mask: layer(hidden_states, attention_mask),
                    x,
                    use_reentrant=False,
                    context_fn=lambda layer=layer, layer_memory_state=layer_memory_state: self._checkpoint_context(
                        layer,
                        layer_memory_state,
                    ),
                )
            else:
                x = layer(x, attention_mask)

        x = self.model.norm(x)
        return self.model.lm_head(x)


def iter_infini_attention_modules(module: nn.Module) -> Iterator[InfiniAttention]:
    for submodule in module.modules():
        if isinstance(submodule, InfiniAttention):
            yield submodule


def capture_infini_memory_state(module: nn.Module) -> List[Dict[str, torch.Tensor]]:
    return [submodule.get_memory_state() for submodule in iter_infini_attention_modules(module)]


def restore_infini_memory_state(
    module: nn.Module,
    state: Sequence[Dict[str, torch.Tensor]],
) -> None:
    infini_modules = list(iter_infini_attention_modules(module))
    if len(infini_modules) != len(state):
        raise ValueError("InfiniAttention state does not match the current model layout")
    for submodule, memory_state in zip(infini_modules, state):
        submodule.load_memory_state(memory_state)


def reset_infini_memory(module: nn.Module) -> None:
    for submodule in iter_infini_attention_modules(module):
        submodule.reset_memory()


class JsonlLogger:
    """Minimal logger that can also mirror metrics to TensorBoard or WandB."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.tensorboard_writer = None
        self.wandb_run = None

        if not args.disable_tensorboard and SummaryWriter is not None:
            self.tensorboard_writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        if args.wandb_project:
            if wandb is None:
                raise ImportError("wandb is not installed but --wandb-project was provided")
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                dir=str(self.output_dir),
            )

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

        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step)

        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


class TextDatasetStream:
    """Restartable streaming Hugging Face text source."""

    def __init__(
        self,
        source: DatasetSource,
        *,
        seed: int,
        shuffle: bool,
        shuffle_buffer_size: int,
        skip_examples: int,
        restart_on_eof: bool,
    ) -> None:
        self.source = source
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.restart_on_eof = restart_on_eof
        self.restart_count = 0
        self.iterator = self._build_iterator()

    def _build_iterator(self) -> Iterator[Dict[str, Any]]:
        if load_dataset is None:
            raise ImportError(
                "train.py requires the `datasets` package. Install it with "
                "`python3 -m pip install -r requirements.txt`."
            )

        dataset = load_dataset(
            self.source.path,
            name=self.source.config_name,
            split=self.source.split,
            streaming=True,
        )
        if self.skip_examples > 0:
            dataset = dataset.skip(self.skip_examples)
        if self.shuffle:
            dataset = dataset.shuffle(
                seed=self.seed + self.restart_count,
                buffer_size=self.shuffle_buffer_size,
            )
        return iter(dataset)

    def _extract_text(self, example: Dict[str, Any]) -> Optional[str]:
        if self.source.text_field in example and isinstance(example[self.source.text_field], str):
            return example[self.source.text_field]

        for field in COMMON_TEXT_FIELDS:
            value = example.get(field)
            if isinstance(value, str):
                return value
        return None

    def __iter__(self) -> "TextDatasetStream":
        return self

    def __next__(self) -> str:
        while True:
            try:
                example = next(self.iterator)
            except StopIteration:
                if not self.restart_on_eof:
                    raise
                self.restart_count += 1
                self.iterator = self._build_iterator()
                continue

            text = self._extract_text(example)
            if text is None:
                continue
            text = text.strip()
            if text:
                return text


class WeightedMixtureStream:
    """Draw documents from multiple restartable text streams according to weights."""

    def __init__(self, streams: Sequence[TextDatasetStream], weights: Sequence[float], seed: int) -> None:
        self.streams = list(streams)
        self.weights = list(weights)
        self.rng = random.Random(seed)

    def __iter__(self) -> "WeightedMixtureStream":
        return self

    def __next__(self) -> str:
        index = self.rng.choices(range(len(self.streams)), weights=self.weights, k=1)[0]
        return next(self.streams[index])


class PackedSequenceStream:
    """Tokenize documents and pack them into fixed-length autoregressive windows."""

    def __init__(
        self,
        text_stream: Iterator[str],
        tokenizer: HierarchicalTokenizer,
        *,
        sequence_length: int,
        max_document_tokens: int,
    ) -> None:
        self.text_stream = text_stream
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.max_document_tokens = max_document_tokens
        self.buffer: List[int] = []

    def __iter__(self) -> "PackedSequenceStream":
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        while len(self.buffer) < self.sequence_length + 1:
            text = next(self.text_stream)
            token_ids = self.tokenizer.encode(
                text,
                max_length=self.max_document_tokens,
                add_special_tokens=True,
            )
            if len(token_ids) < 2:
                continue
            self.buffer.extend(token_ids)

        window = self.buffer[: self.sequence_length + 1]
        del self.buffer[: self.sequence_length]

        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)
        attention_mask = torch.ones(self.sequence_length, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class BatchStream:
    """Batch fixed-length packed sequences without a DataLoader."""

    def __init__(self, sequence_stream: PackedSequenceStream, micro_batch_size: int) -> None:
        self.sequence_stream = sequence_stream
        self.micro_batch_size = micro_batch_size

    def __iter__(self) -> "BatchStream":
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        batch = [next(self.sequence_stream) for _ in range(self.micro_batch_size)]
        return {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in ("input_ids", "labels", "attention_mask")
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: Optional[torch.dtype]) -> contextlib.AbstractContextManager[Any]:
    if not amp_enabled or amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def parse_mixture_entry(entry: str) -> List[Tuple[DatasetSource, float]]:
    entry = entry.strip()
    if not entry:
        raise ValueError("Mixture entries must not be empty")
    if "=" not in entry:
        raise ValueError(f"Mixture entry '{entry}' is missing '=weight'")

    source_name, weight_text = entry.rsplit("=", 1)
    weight = float(weight_text)
    if weight <= 0:
        raise ValueError(f"Mixture weight must be positive: {entry}")

    source_name = source_name.strip()
    if source_name in DATASET_PRESETS:
        return [(DATASET_PRESETS[source_name], weight)]

    if source_name in MIXTURE_GROUP_PRESETS:
        return [
            (DATASET_PRESETS[member_name], weight * member_weight)
            for member_name, member_weight in MIXTURE_GROUP_PRESETS[source_name]
        ]

    parts = source_name.split("|")
    if len(parts) != 4:
        raise ValueError(
            "Custom mixture entries must use 'path|config|split|text_field=weight'"
        )

    path, config_name, split, text_field = parts
    return [(
        DatasetSource(
            alias=path,
            path=path,
            config_name=config_name or None,
            split=split,
            text_field=text_field,
        ),
        weight,
    )]


def parse_mixture(spec: str) -> List[Tuple[DatasetSource, float]]:
    expanded: List[Tuple[DatasetSource, float]] = []
    for entry in spec.split(","):
        if entry.strip():
            expanded.extend(parse_mixture_entry(entry))
    return expanded


def build_text_stream(
    mixture: List[Tuple[DatasetSource, float]],
    *,
    seed: int,
    shuffle: bool,
    shuffle_buffer_size: int,
    skip_examples: int,
    restart_on_eof: bool,
) -> WeightedMixtureStream:
    streams = [
        TextDatasetStream(
            source=source,
            seed=seed + index * 1000,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            skip_examples=skip_examples,
            restart_on_eof=restart_on_eof,
        )
        for index, (source, _) in enumerate(mixture)
    ]
    weights = [weight for _, weight in mixture]
    return WeightedMixtureStream(streams, weights, seed=seed)


def build_batch_stream(
    mixture: List[Tuple[DatasetSource, float]],
    tokenizer: HierarchicalTokenizer,
    *,
    seed: int,
    shuffle: bool,
    shuffle_buffer_size: int,
    skip_examples: int,
    restart_on_eof: bool,
    sequence_length: int,
    max_document_tokens: int,
    micro_batch_size: int,
) -> BatchStream:
    text_stream = build_text_stream(
        mixture,
        seed=seed,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        skip_examples=skip_examples,
        restart_on_eof=restart_on_eof,
    )
    packed_stream = PackedSequenceStream(
        text_stream,
        tokenizer,
        sequence_length=sequence_length,
        max_document_tokens=max_document_tokens,
    )
    return BatchStream(packed_stream, micro_batch_size=micro_batch_size)


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


def build_model_config(args: argparse.Namespace, tokenizer: HierarchicalTokenizer) -> TernaryConfig:
    return TernaryConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        head_dim=args.hidden_size // args.num_heads,
        intermediate_size=args.intermediate_size,
        rms_norm_eps=default_config.rms_norm_eps,
        rope_theta=default_config.rope_theta,
        rope_scaling={
            "type": "yarn",
            "factor": args.rope_scaling_factor,
            "original_max_position_embeddings": default_config.rope_scaling["original_max_position_embeddings"],
        },
        max_position_embeddings=args.sequence_length,
        initializer_range=default_config.initializer_range,
        block_size=args.final_blocks,
        infini_memory_dim=default_config.infini_memory_dim,
        use_hadamard=not args.disable_hadamard,
        use_4bit_activations=True,
        ternary_weight_bits=default_config.ternary_weight_bits,
    )


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> Lion:
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return Lion(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.lion_beta1, args.lion_beta2),
    )


def lr_schedule_multiplier(step: int, total_steps: int, warmup_steps: int, cooldown_steps: int, min_lr_ratio: float) -> float:
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


def update_quantization_schedule(model: nn.Module, token_progress: float, args: argparse.Namespace) -> Dict[str, float]:
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


def update_block_growth(model: nn.Module, token_progress: float, args: argparse.Namespace) -> int:
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


def save_checkpoint(
    output_dir: Path,
    model: BitNetDeep,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[Any],
    state: TrainerState,
    model_config: TernaryConfig,
    args: argparse.Namespace,
    checkpoint_name: str,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "trainer_state": asdict(state),
        "model_config": asdict(model_config),
        "args": vars(args),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: BitNetDeep,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[Any],
) -> TrainerState:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model"])
    reset_infini_memory(model)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return TrainerState(**payload["trainer_state"])


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
            attention_mask = batch["attention_mask"].to(device)

            with autocast_context(device, amp_enabled, amp_dtype):
                logits = runner(input_ids, attention_mask)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            losses.append(float(loss.item()))

        mean_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(mean_loss, 20.0))
        return {"val_loss": mean_loss, "val_perplexity": perplexity}
    finally:
        restore_infini_memory_state(runner, memory_state)
        runner.train(was_training)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the deep ternary BitNet model with streaming HF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output / reproducibility
    parser.add_argument("--output-dir", type=str, default="runs/bitnet")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume-from", type=str, default="")

    # Data
    parser.add_argument(
        "--train-mixture",
        type=str,
        default="fineweb_edu=0.55,dclm=0.20,code_search_net_all=0.10,finemath_3plus=0.15",
    )
    parser.add_argument(
        "--early-train-mixture",
        type=str,
        default="",
        help="Optional early-stage training mixture. Falls back to --train-mixture when unset.",
    )
    parser.add_argument(
        "--late-train-mixture",
        type=str,
        default="",
        help="Optional late-stage training mixture. Falls back to --train-mixture when unset.",
    )
    parser.add_argument(
        "--mixture-switch-ratio",
        type=float,
        default=0.7,
        help="Fraction of total tokens after which the trainer switches from the early to late mixture.",
    )
    parser.add_argument(
        "--val-mixture",
        type=str,
        default="fineweb_edu=0.45,code_search_net_all=0.20,finemath_3plus=0.35",
    )
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--validation-offset-examples", type=int, default=25_000)
    parser.add_argument("--max-document-tokens", type=int, default=32_768)
    parser.add_argument("--tokenizer-max-patch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=131_072)

    # Model shape
    parser.add_argument("--hidden-size", type=int, default=default_config.hidden_size)
    parser.add_argument("--num-layers", type=int, default=default_config.num_hidden_layers)
    parser.add_argument("--num-heads", type=int, default=default_config.num_attention_heads)
    parser.add_argument("--intermediate-size", type=int, default=default_config.intermediate_size)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--rope-scaling-factor", type=float, default=default_config.rope_scaling["factor"])
    parser.add_argument("--disable-hadamard", action="store_true")

    # Optimization
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=16)
    parser.add_argument("--total-tokens", type=int, default=50_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lion-beta1", type=float, default=0.95)
    parser.add_argument("--lion-beta2", type=float, default=0.98)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--cooldown-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cooldown-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Ternary / schedule controls
    parser.add_argument("--stage1-ratio", type=float, default=0.12)
    parser.add_argument("--stage1-weight-mix-start", type=float, default=0.25)
    parser.add_argument("--stage1-activation-mix-start", type=float, default=0.0)
    parser.add_argument("--stage1-activation-bits", type=int, default=8)
    parser.add_argument("--final-activation-bits", type=int, default=4)
    parser.add_argument("--initial-blocks", type=int, default=8)
    parser.add_argument("--final-blocks", type=int, default=16)
    parser.add_argument("--block-growth-ratio", type=float, default=0.6)

    # Runtime controls
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")

    # Logging / eval / saving
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.hidden_size % args.num_heads != 0:
        parser.error("--hidden-size must be divisible by --num-heads")
    if args.final_blocks < args.initial_blocks:
        parser.error("--final-blocks must be greater than or equal to --initial-blocks")
    if not 0.0 <= args.mixture_switch_ratio <= 1.0:
        parser.error("--mixture-switch-ratio must be between 0.0 and 1.0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = choose_device(args.device)
    amp_enabled, amp_dtype = configure_mixed_precision(device, args.precision)
    base_train_mixture = parse_mixture(args.train_mixture)
    early_train_mixture = parse_mixture(args.early_train_mixture) if args.early_train_mixture else base_train_mixture
    late_train_mixture = parse_mixture(args.late_train_mixture) if args.late_train_mixture else None
    val_mixture = parse_mixture(args.val_mixture)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if _TOKENIZER_IMPORT_ERROR is not None:
        raise ImportError(
            "train.py requires the tokenizer dependencies. Install them with "
            "`python3 -m pip install -r requirements.txt`."
        ) from _TOKENIZER_IMPORT_ERROR

    tokenizer = HierarchicalTokenizer(
        max_patch_size=args.tokenizer_max_patch_size,
        vocab_size_target=args.vocab_size,
    )
    model_config = build_model_config(args, tokenizer)
    base_model = BitNetDeep(model_config)
    base_model.to(device)

    runner = TrainingWrapper(base_model, gradient_checkpointing=args.gradient_checkpointing).to(device)
    if args.compile and hasattr(torch, "compile") and not args.gradient_checkpointing:
        runner = torch.compile(runner, mode=args.compile_mode)
    elif args.compile and args.gradient_checkpointing:
        print("Skipping torch.compile because gradient checkpointing is enabled.", flush=True)

    parameter_count = sum(param.numel() for param in base_model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {parameter_count / 1e6:.2f}M")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    if late_train_mixture is None:
        print(f"Training mixture: {args.train_mixture}")
    else:
        print(f"Early training mixture: {args.early_train_mixture or args.train_mixture}")
        print(f"Late training mixture: {args.late_train_mixture}")
        print(f"Mixture switch ratio: {args.mixture_switch_ratio:.2f}")
    print(f"Validation mixture: {args.val_mixture}")

    tokens_per_optimization_step = (
        args.micro_batch_size * args.sequence_length * args.grad_accumulation_steps
    )
    total_steps = max(1, math.ceil(args.total_tokens / tokens_per_optimization_step))
    warmup_steps = args.warmup_steps or math.ceil(total_steps * args.warmup_ratio)
    cooldown_steps = args.cooldown_steps or math.ceil(total_steps * args.cooldown_ratio)

    optimizer = create_optimizer(base_model, args)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_schedule_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            min_lr_ratio=args.min_lr_ratio,
        ),
    )

    scaler = None
    if amp_enabled and amp_dtype == torch.float16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    logger = JsonlLogger(args)
    early_train_batch_stream = build_batch_stream(
        early_train_mixture,
        tokenizer,
        seed=args.seed,
        shuffle=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        skip_examples=0,
        restart_on_eof=True,
        sequence_length=args.sequence_length,
        max_document_tokens=args.max_document_tokens,
        micro_batch_size=args.micro_batch_size,
    )
    late_train_batch_stream = None
    if late_train_mixture is not None:
        late_train_batch_stream = build_batch_stream(
            late_train_mixture,
            tokenizer,
            seed=args.seed + 17,
            shuffle=True,
            shuffle_buffer_size=args.shuffle_buffer_size,
            skip_examples=0,
            restart_on_eof=True,
            sequence_length=args.sequence_length,
            max_document_tokens=args.max_document_tokens,
            micro_batch_size=args.micro_batch_size,
        )

    state = TrainerState()
    if args.resume_from:
        state = load_checkpoint(Path(args.resume_from), base_model, optimizer, scheduler, scaler)
        print(f"Resumed from {args.resume_from} at step {state.step}")

    start_time = time.time()

    try:
        while state.step < total_steps and state.tokens_processed < args.total_tokens:
            train_progress = state.tokens_processed / max(args.total_tokens, 1)
            quant_metrics = update_quantization_schedule(base_model, train_progress, args)
            active_blocks = update_block_growth(base_model, train_progress, args)
            mixture_stage, _ = choose_stage_mixture(
                train_progress,
                early_mixture=early_train_mixture,
                late_mixture=late_train_mixture,
                switch_ratio=args.mixture_switch_ratio,
            )
            active_train_batch_stream = early_train_batch_stream if mixture_stage == "early" else late_train_batch_stream
            if active_train_batch_stream is None:
                active_train_batch_stream = early_train_batch_stream

            runner.train()
            optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0
            step_tokens = 0
            step_start = time.time()

            for _ in range(args.grad_accumulation_steps):
                batch = next(active_train_batch_stream)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast_context(device, amp_enabled, amp_dtype):
                    logits = runner(input_ids, attention_mask)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    scaled_loss = loss / args.grad_accumulation_steps

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accumulated_loss += float(loss.item())
                step_tokens += int(input_ids.numel())
                state.samples_processed += int(input_ids.size(0))

            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            state.step += 1
            state.tokens_processed += step_tokens

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - step_start
            tokens_per_second = step_tokens / max(elapsed, 1e-6)
            loss_value = accumulated_loss / args.grad_accumulation_steps

            if state.step == 1 or state.step % args.log_interval == 0:
                logger.log(
                    state.step,
                    {
                        "train_loss": loss_value,
                        "learning_rate": current_lr,
                        "grad_norm": float(grad_norm),
                        "tokens_processed": float(state.tokens_processed),
                        "tokens_per_second": tokens_per_second,
                        "active_blocks": float(active_blocks),
                        "mixture_stage_id": 0.0 if mixture_stage == "early" else 1.0,
                        "mixture_switch_ratio": args.mixture_switch_ratio,
                        **quant_metrics,
                    },
                )

            if args.eval_interval > 0 and state.step % args.eval_interval == 0:
                eval_metrics = evaluate(
                    runner,
                    tokenizer,
                    val_mixture,
                    args,
                    device,
                    amp_enabled,
                    amp_dtype,
                )
                if eval_metrics:
                    logger.log(state.step, eval_metrics)
                    if eval_metrics["val_loss"] < state.best_val_loss:
                        state.best_val_loss = eval_metrics["val_loss"]
                        best_path = save_checkpoint(
                            output_dir,
                            base_model,
                            optimizer,
                            scheduler,
                            scaler,
                            state,
                            model_config,
                            args,
                            checkpoint_name="best.pt",
                        )
                        print(f"Saved new best checkpoint to {best_path}")

            if args.save_interval > 0 and state.step % args.save_interval == 0:
                checkpoint_path = save_checkpoint(
                    output_dir,
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    model_config,
                    args,
                    checkpoint_name=f"step_{state.step:07d}.pt",
                )
                save_checkpoint(
                    output_dir,
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    model_config,
                    args,
                    checkpoint_name="last.pt",
                )
                print(f"Saved checkpoint to {checkpoint_path}")

        final_path = save_checkpoint(
            output_dir,
            base_model,
            optimizer,
            scheduler,
            scaler,
            state,
            model_config,
            args,
            checkpoint_name="final.pt",
        )
        logger.log(
            state.step,
            {
                "train_wall_time_sec": time.time() - start_time,
                "total_tokens_processed": float(state.tokens_processed),
            },
        )
        print(f"Training complete. Final checkpoint written to {final_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
