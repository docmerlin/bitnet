"""Experimental MLX trainer for the dense BitNet PaTH-FoX port."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from data.presets import parse_mixture
from data.streams import build_batch_stream
from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_optim import CMUD
from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer


_MEMORY_STATE_NAMES = (".memory_k", ".memory_v", ".memory_initialized")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/mlx_bitnet")
    parser.add_argument("--train-mixture", default="fineweb_edu=0.7,dclm=0.3")
    parser.add_argument("--early-train-mixture", default="")
    parser.add_argument("--late-train-mixture", default="")
    parser.add_argument("--mixture-switch-ratio", type=float, default=0.7)
    parser.add_argument("--val-mixture", default="fineweb_edu=0.5,dclm=0.5")
    parser.add_argument("--validation-offset-examples", type=int, default=25000)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument("--max-document-tokens", type=int, default=32768)
    parser.add_argument("--tokenizer-max-patch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-prelude-layers", type=int, default=2)
    parser.add_argument("--num-recurrent-layers", type=int, default=4)
    parser.add_argument("--num-coda-layers", type=int, default=2)
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--min-num-loops", type=int, default=1)
    parser.add_argument("--loop-curriculum-ratio", type=float, default=0.2)
    parser.add_argument("--initial-blocks", type=int, default=8)
    parser.add_argument("--final-blocks", type=int, default=16)
    parser.add_argument("--block-growth-ratio", type=float, default=0.6)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--path-window-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--total-tokens", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mud-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mud-momentum", type=float, default=0.95)
    parser.add_argument("--mud-passes", type=int, default=1)
    parser.add_argument("--lion-beta1", type=float, default=0.95)
    parser.add_argument("--lion-beta2", type=float, default=0.98)
    parser.add_argument("--no-optimizer-8bit", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--cooldown-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cooldown-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--z-loss-coef", type=float, default=1e-4)
    parser.add_argument("--mtp-depth", type=int, default=0)
    parser.add_argument("--mtp-loss-coef", type=float, default=0.3)
    parser.add_argument("--engram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--engram-layer-ids", default="1,15")
    parser.add_argument("--engram-vocab-size", type=int, default=4093)
    parser.add_argument("--use-rfmoe", action="store_true")
    parser.add_argument("--rfmoe-num-experts", type=int, default=8)
    parser.add_argument("--rfmoe-expert-dim", type=int, default=None)
    parser.add_argument("--rfmoe-rank", type=int, default=None)
    parser.add_argument("--rfmoe-theta", type=float, default=0.01)
    parser.add_argument("--rfmoe-density-target", type=float, default=0.25)
    parser.add_argument("--rfmoe-density-eta", type=float, default=0.01)
    parser.add_argument("--rfmoe-locality-coef", type=float, default=0.0)
    parser.add_argument("--rfmoe-diversity-coef", type=float, default=0.0)
    parser.add_argument("--rfmoe-zipf-s", type=float, default=1.0)
    parser.add_argument("--rfmoe-uniform-alpha", type=float, default=0.1)
    parser.add_argument("--rfmoe-curriculum-ratio", type=float, default=0.0)
    parser.add_argument("--stage1-ratio", type=float, default=0.12)
    parser.add_argument("--stage1-weight-mix-start", type=float, default=0.25)
    parser.add_argument("--stage1-activation-mix-start", type=float, default=0.0)
    parser.add_argument("--stage1-activation-bits", type=int, default=8)
    parser.add_argument("--final-activation-bits", type=int, default=4)
    parser.add_argument("--precision", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume-from", default="")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.hidden_size % args.num_heads:
        raise ValueError("hidden-size must be divisible by num-heads")
    if args.sequence_length % args.path_window_size:
        raise ValueError("sequence-length must be divisible by path-window-size")
    if min(args.micro_batch_size, args.grad_accumulation_steps, args.total_tokens) < 1:
        raise ValueError("batch, accumulation, and token counts must be positive")
    if min(args.stage1_activation_bits, args.final_activation_bits) < 2:
        raise ValueError("activation bits must be at least 2")
    if not 0 <= args.mixture_switch_ratio <= 1:
        raise ValueError("mixture-switch-ratio must be between zero and one")


def _masked_ce(logits, targets, valid):
    safe_targets = mx.where(valid, targets, 0)
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), safe_targets, reduction="none")
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), 1)


def create_gradient_step(
    model: MLXBitNet,
    *,
    compile_step: bool,
    num_loops: int,
    z_loss_coef: float = 0.0,
    mtp_loss_coef: float = 0.0,
    locality_coef: float = 0.0,
    diversity_coef: float = 0.0,
):
    def loss_fn(inputs, targets, segment_ids, label_segment_ids, density_lam, rfmoe_s, rfmoe_alpha):
        return_mtp = model.config.mtp_depth > 0
        output = model(inputs, segment_ids, num_loops=num_loops, return_mtp=return_mtp)
        logits, mtp_logits = output if return_mtp else (output, [])
        valid = segment_ids == label_segment_ids
        loss = _masked_ce(logits, targets, valid)
        if z_loss_coef > 0:
            log_z = mx.logsumexp(logits.astype(mx.float32), axis=-1)
            loss = loss + z_loss_coef * mx.sum(mx.square(log_z) * valid) / mx.maximum(mx.sum(valid), 1)
        mtp_losses = []
        for index, depth_logits in enumerate(mtp_logits):
            shift = index + 1
            if shift >= targets.shape[1]:
                continue
            depth_valid = segment_ids[:, : -shift] == label_segment_ids[:, shift:]
            mtp_losses.append(_masked_ce(depth_logits[:, :-shift], targets[:, shift:], depth_valid))
        if mtp_losses:
            loss = loss + mtp_loss_coef * mx.mean(mx.stack(mtp_losses))
        if model.config.use_rfmoe:
            density, locality, diversity, _ = model.rfmoe_aux_losses(rfmoe_s, rfmoe_alpha)
            loss = loss + density_lam * density
            loss = loss + locality_coef * locality + diversity_coef * diversity
        return loss

    gradient_step = nn.value_and_grad(model, loss_fn)
    if compile_step:
        gradient_step = partial(mx.compile, inputs=model.state, outputs=model.state)(gradient_step)
    return gradient_step


def create_train_step(model: MLXBitNet, optimizer: optim.Optimizer, *, compile_step: bool = True):
    optimizer.init(model.trainable_parameters())
    gradient_step = create_gradient_step(
        model,
        compile_step=compile_step,
        num_loops=model.config.num_loops,
    )
    state = [model.state, optimizer.state]

    def train_step(inputs, targets, segment_ids, label_segment_ids):
        loss, gradients = gradient_step(
            inputs,
            targets,
            segment_ids,
            label_segment_ids,
            mx.array(0.0),
            mx.array(1.0),
            mx.array(0.1),
        )
        optimizer.update(model, gradients)
        return loss

    if compile_step:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)
    return train_step, state


def convert_batch(batch) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    return tuple(
        mx.array(batch[key].numpy())
        for key in ("input_ids", "labels", "segment_ids", "label_segment_ids")
    )


def save_checkpoint(
    output_dir: Path,
    model: MLXBitNet,
    optimizer: optim.Optimizer,
    config: MLXBitNetConfig,
    trainer_state: dict,
    name: str,
    training_args: dict | None = None,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.safetensors"
    parameters = {
        key: value
        for key, value in tree_flatten(model.parameters())
        if not key.endswith(_MEMORY_STATE_NAMES)
    }
    mx.save_safetensors(str(path), parameters)
    optimizer_path = checkpoint_dir / f"{name}.optimizer.safetensors"
    mx.save_safetensors(str(optimizer_path), dict(tree_flatten(optimizer.state)))
    metadata = {
        "trainer_state": trainer_state,
        "model_config": asdict(config),
        "optimizer_config": optimizer.checkpoint_config() if isinstance(optimizer, CMUD) else None,
        "training_args": training_args,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def load_checkpoint(path: Path, model: MLXBitNet, optimizer: optim.Optimizer) -> dict:
    parameters = dict(tree_flatten(model.parameters()))
    expected = {key for key in parameters if not key.endswith(_MEMORY_STATE_NAMES)}
    loaded = mx.load(str(path))
    missing = expected - loaded.keys()
    unexpected = loaded.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Model checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    invalid_shapes = [key for key, value in loaded.items() if value.shape != parameters[key].shape]
    if invalid_shapes:
        raise ValueError(f"Model checkpoint tensor mismatch: {sorted(invalid_shapes)}")
    floating = {mx.float16, mx.bfloat16, mx.float32}
    weights = [
        (key, value.astype(parameters[key].dtype) if value.dtype in floating else value)
        for key, value in loaded.items()
    ]
    model.load_weights(weights, strict=False)
    optimizer_path = path.with_name(f"{path.stem}.optimizer.safetensors")
    if not optimizer_path.exists():
        raise FileNotFoundError(f"Missing optimizer checkpoint: {optimizer_path}")
    optimizer_state = mx.load(str(optimizer_path))
    expected_optimizer = dict(tree_flatten(optimizer.state))
    missing_optimizer = expected_optimizer.keys() - optimizer_state.keys()
    unexpected_optimizer = optimizer_state.keys() - expected_optimizer.keys()
    if missing_optimizer or unexpected_optimizer:
        raise ValueError(
            "Optimizer checkpoint mismatch: "
            f"missing={sorted(missing_optimizer)}, unexpected={sorted(unexpected_optimizer)}"
        )
    invalid_optimizer = [
        key
        for key, value in optimizer_state.items()
        if value.shape != expected_optimizer[key].shape or value.dtype != expected_optimizer[key].dtype
    ]
    if invalid_optimizer:
        raise ValueError(f"Optimizer checkpoint tensor mismatch: {sorted(invalid_optimizer)}")
    optimizer.state = tree_unflatten(list(optimizer_state.items()))
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return metadata["trainer_state"]


def scheduled_value(start: float, end: float, progress: float, ratio: float) -> float:
    if ratio <= 0:
        return end
    fraction = min(max(progress / ratio, 0.0), 1.0)
    return start + fraction * (end - start)


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, cooldown_steps: int, minimum: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    main_steps = max(total_steps - warmup_steps - cooldown_steps, 1)
    if step < warmup_steps + main_steps:
        progress = (step - warmup_steps) / main_steps
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    cooldown = (step - warmup_steps - main_steps) / max(cooldown_steps, 1)
    return max(minimum * (1.0 - cooldown), 0.0)


def evaluate(model: MLXBitNet, tokenizer, args) -> dict[str, float]:
    if args.validation_batches <= 0:
        return {}
    model.eval()
    stream = build_batch_stream(
        parse_mixture(args.val_mixture),
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
    losses = []
    for _ in range(args.validation_batches):
        inputs, targets, segments, label_segments = convert_batch(next(stream))
        logits = model(inputs, segments)
        loss = _masked_ce(logits, targets, segments == label_segments)
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"val_loss": mean_loss, "val_perplexity": math.exp(min(mean_loss, 20.0))}


def main() -> None:
    args = build_parser().parse_args()
    saved = None
    if args.resume_from:
        saved = json.loads(Path(args.resume_from).with_suffix(".json").read_text(encoding="utf-8"))
        protected = {"output_dir", "resume_from", "compile", "path_kernel", "precision"}
        for key, value in (saved.get("training_args") or {}).items():
            if key not in protected and hasattr(args, key):
                setattr(args, key, value)
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)

    if args.resume_from:
        saved_config = saved["model_config"]
        saved_config["engram_layer_ids"] = tuple(saved_config["engram_layer_ids"])
        config = MLXBitNetConfig(**saved_config)
        tokenizer_vocab_size = config.vocab_size
    else:
        tokenizer_vocab_size = args.vocab_size
        config = None
    tokenizer = HierarchicalTokenizer(
        max_patch_size=args.tokenizer_max_patch_size,
        vocab_size_target=tokenizer_vocab_size,
    )
    if config is None:
        config = MLXBitNetConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            num_prelude_layers=args.num_prelude_layers,
            num_recurrent_layers=args.num_recurrent_layers,
            num_coda_layers=args.num_coda_layers,
            num_loops=args.num_loops,
            block_size=args.initial_blocks,
            path_window_size=args.path_window_size,
            use_path_kernel=args.path_kernel,
            use_engram=args.engram,
            engram_layer_ids=tuple(int(value) for value in args.engram_layer_ids.split(",") if value),
            engram_vocab_size=args.engram_vocab_size,
            use_rfmoe=args.use_rfmoe,
            rfmoe_num_experts=args.rfmoe_num_experts,
            rfmoe_expert_dim=args.rfmoe_expert_dim,
            rfmoe_rank=args.rfmoe_rank,
            rfmoe_theta=args.rfmoe_theta,
            mtp_depth=args.mtp_depth,
        )
    model = MLXBitNet(config)
    dtype = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}[args.precision]
    model.set_dtype(dtype)
    if args.resume_from and saved.get("optimizer_config"):
        optimizer = CMUD(**saved["optimizer_config"])
    else:
        optimizer = CMUD(
            mud_learning_rate=args.mud_learning_rate,
            fallback_learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.mud_momentum,
            passes=args.mud_passes,
            betas=(args.lion_beta1, args.lion_beta2),
            eight_bit=not args.no_optimizer_8bit,
        )
    optimizer.init(model.trainable_parameters())
    trainer_state = {
        "step": 0,
        "tokens_processed": 0,
        "best_val_loss": float("inf"),
        "density_lambda": 1e-3,
    }
    if args.resume_from:
        trainer_state.update(load_checkpoint(Path(args.resume_from), model, optimizer))

    early_spec = args.early_train_mixture or args.train_mixture
    early_stream = build_batch_stream(
        parse_mixture(early_spec),
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
    late_stream = None
    if args.late_train_mixture:
        late_stream = build_batch_stream(
            parse_mixture(args.late_train_mixture),
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

    tokens_per_microbatch = args.micro_batch_size * args.sequence_length
    tokens_per_step = tokens_per_microbatch * args.grad_accumulation_steps
    total_steps = math.ceil(args.total_tokens / tokens_per_step)
    warmup_steps = args.warmup_steps or math.ceil(total_steps * args.warmup_ratio)
    cooldown_steps = args.cooldown_steps or math.ceil(total_steps * args.cooldown_ratio)
    parameters = sum(value.size for _, value in tree_flatten(model.parameters()))
    print(f"Device: {mx.device_info()['device_name']}")
    print(f"Model parameters: {parameters / 1e6:.2f}M")
    print(f"Effective depth: {config.effective_depth}")
    print(f"Early training mixture: {early_spec}")
    if late_stream is not None:
        print(f"Late training mixture: {args.late_train_mixture}")

    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    gradient_steps = {}
    state = [model.state, optimizer.state]
    for step in range(trainer_state["step"] + 1, total_steps + 1):
        progress = trainer_state["tokens_processed"] / max(args.total_tokens, 1)
        loop_fraction = min(progress / max(args.loop_curriculum_ratio, 1e-8), 1.0)
        active_loops = config.num_loops if args.loop_curriculum_ratio <= 0 else round(
            args.min_num_loops + loop_fraction * (config.num_loops - args.min_num_loops)
        )
        active_blocks = round(scheduled_value(
            args.initial_blocks,
            args.final_blocks,
            progress,
            args.block_growth_ratio,
        ))
        model.set_active_blocks(active_blocks)
        quant_fraction = 1.0 if args.stage1_ratio <= 0 else min(progress / args.stage1_ratio, 1.0)
        weight_mix = args.stage1_weight_mix_start + quant_fraction * (1.0 - args.stage1_weight_mix_start)
        activation_mix = args.stage1_activation_mix_start + quant_fraction * (1.0 - args.stage1_activation_mix_start)
        activation_bits = round(
            args.stage1_activation_bits
            - quant_fraction * (args.stage1_activation_bits - args.final_activation_bits)
        )
        model.set_quantization_state(weight_mix, activation_mix, activation_bits)
        rf_fraction = min(progress / max(args.rfmoe_curriculum_ratio, 1e-8), 1.0)
        if args.rfmoe_curriculum_ratio <= 0:
            rf_fraction = 1.0
        rf_s = rf_fraction * args.rfmoe_zipf_s
        rf_alpha = 1.0 + rf_fraction * (args.rfmoe_uniform_alpha - 1.0)
        key = (active_loops, active_blocks)
        if key not in gradient_steps:
            gradient_steps[key] = create_gradient_step(
                model,
                compile_step=args.compile,
                num_loops=active_loops,
                z_loss_coef=args.z_loss_coef,
                mtp_loss_coef=args.mtp_loss_coef,
                locality_coef=args.rfmoe_locality_coef,
                diversity_coef=args.rfmoe_diversity_coef,
            )
        gradient_step = gradient_steps[key]
        active_stream = late_stream if late_stream is not None and progress >= args.mixture_switch_ratio else early_stream
        step_started = time.perf_counter()
        accumulated_gradients = None
        losses = []
        hard_densities = []
        for _ in range(args.grad_accumulation_steps):
            batch = convert_batch(next(active_stream))
            loss, gradients = gradient_step(
                *batch,
                mx.array(trainer_state["density_lambda"]),
                mx.array(rf_s),
                mx.array(rf_alpha),
            )
            hard_density = model.rfmoe_aux_losses(rf_s, rf_alpha)[3]
            mx.eval(loss, gradients, hard_density, model.state)
            losses.append(float(loss.item()))
            hard_densities.append(float(hard_density.item()))
            accumulated_gradients = gradients if accumulated_gradients is None else tree_map(
                lambda total, current: total + current,
                accumulated_gradients,
                gradients,
            )
        accumulated_gradients = tree_map(
            lambda gradient: gradient / args.grad_accumulation_steps,
            accumulated_gradients,
        )
        accumulated_gradients, grad_norm = optim.clip_grad_norm(accumulated_gradients, args.grad_clip)
        multiplier = lr_multiplier(step - 1, total_steps, warmup_steps, cooldown_steps, args.min_lr_ratio)
        optimizer.set_lr_multiplier(multiplier)
        optimizer.update(model, accumulated_gradients)
        mx.eval(model.state, optimizer.state, grad_norm)
        elapsed = time.perf_counter() - step_started
        trainer_state["step"] = step
        trainer_state["tokens_processed"] += tokens_per_step
        if config.use_rfmoe and hard_densities:
            density = sum(hard_densities) / len(hard_densities)
            factor = 1.0 + args.rfmoe_density_eta
            if density > args.rfmoe_density_target:
                trainer_state["density_lambda"] = min(trainer_state["density_lambda"] * factor, 1e3)
            elif density < args.rfmoe_density_target:
                trainer_state["density_lambda"] = max(trainer_state["density_lambda"] / factor, 1e-6)
        if step == 1 or step % args.log_interval == 0:
            metrics = {
                "step": step,
                "loss": sum(losses) / len(losses),
                "tokens_processed": trainer_state["tokens_processed"],
                "tokens_per_second": tokens_per_step / elapsed,
                "learning_rate": optimizer.mud_learning_rate * multiplier,
                "grad_norm": float(grad_norm.item()),
                "active_loops": active_loops,
                "active_blocks": active_blocks,
                "quant_weight_mix": weight_mix,
                "quant_activation_mix": activation_mix,
                "quant_activation_bits": activation_bits,
                "time": time.time(),
            }
            if config.use_rfmoe:
                metrics["rfmoe_density"] = sum(hard_densities) / len(hard_densities)
                metrics["rfmoe_lambda"] = trainer_state["density_lambda"]
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(" | ".join(f"{key}={value}" for key, value in metrics.items()))
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            validation = evaluate(model, tokenizer, args)
            print(" | ".join(f"{key}={value}" for key, value in validation.items()))
            if validation and validation["val_loss"] < trainer_state["best_val_loss"]:
                trainer_state["best_val_loss"] = validation["val_loss"]
                save_checkpoint(output_dir, model, optimizer, config, trainer_state, "best", vars(args))
        if args.save_interval > 0 and step % args.save_interval == 0:
            path = save_checkpoint(
                output_dir, model, optimizer, config, trainer_state, f"step_{step:07d}", vars(args)
            )
            save_checkpoint(output_dir, model, optimizer, config, trainer_state, "last", vars(args))
            print(f"Saved checkpoint to {path}")

    final_path = save_checkpoint(output_dir, model, optimizer, config, trainer_state, "final", vars(args))
    wall_time = time.perf_counter() - started
    print(f"Training complete in {wall_time:.1f}s. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
