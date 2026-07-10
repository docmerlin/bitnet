"""Production training entrypoint for the deep ternary BitNet model.

Packages:
- ``data/`` — presets, mixtures, packing
- ``training/`` — losses, schedules, checkpoints, runtime
- ``model.py`` — single forward path (checkpointing + MTP)
- ``utils.py`` — RoPE / attention-mask helpers

Example:
    python3 train.py --output-dir runs/bitnet --total-tokens 50000000
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from config import TernaryConfig
from data.presets import parse_mixture
from data.streams import PrefetchStream, build_batch_stream
from layers.rfmoe import DensityController, iter_rfmoe, rfmoe_density, rfmoe_diversity_loss
from model import BitNetDeep
from training.checkpoint import TrainerState, load_checkpoint, save_checkpoint
from training.losses import compute_train_loss
from training.runtime import (
    JsonlLogger,
    autocast_context,
    build_model_config,
    choose_device,
    configure_mixed_precision,
    create_optimizer,
    evaluate,
)
from training.schedules import (
    choose_stage_mixture,
    collect_loop_train_metrics,
    loop_count_for_progress,
    lr_schedule_multiplier,
    rfmoe_staircase_schedule,
    update_block_growth,
    update_quantization_schedule,
)
from utils import load_checkpoint_payload, seed_everything

try:
    from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
    _TOKENIZER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    HierarchicalTokenizer = Any  # type: ignore[misc, assignment]
    _TOKENIZER_IMPORT_ERROR = exc


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TernaryConfig()
    parser = argparse.ArgumentParser(
        description="Train the deep ternary BitNet model with streaming HF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir", type=str, default="runs/bitnet")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume-from", type=str, default="")

    parser.add_argument(
        "--train-mixture",
        type=str,
        default="fineweb_edu=0.55,dclm=0.20,code_search_net_all=0.10,finemath_3plus=0.15",
    )
    parser.add_argument("--early-train-mixture", type=str, default="")
    parser.add_argument("--late-train-mixture", type=str, default="")
    parser.add_argument("--mixture-switch-ratio", type=float, default=0.7)
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

    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Flat unique layer count (prelude=0, recurrent=N, coda=0). "
        "Ignored when any of --num-prelude/recurrent/coda-layers is set. "
        "Default with no structure flags: 8/48/8 looped.",
    )
    parser.add_argument(
        "--num-prelude-layers",
        type=int,
        default=None,
        help="Unique layers run once before the recurrent core (default 8).",
    )
    parser.add_argument(
        "--num-recurrent-layers",
        type=int,
        default=None,
        help="Unique layers in the looped middle stack (default 48).",
    )
    parser.add_argument(
        "--num-coda-layers",
        type=int,
        default=None,
        help="Unique layers run once after the recurrent core (default 8).",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Max recurrent loops after curriculum (default 4 with structure, 1 if flat --num-layers).",
    )
    parser.add_argument(
        "--min-num-loops",
        type=int,
        default=1,
        help="Starting loop count for the depth curriculum (ramps up to --num-loops).",
    )
    parser.add_argument(
        "--loop-curriculum-ratio",
        type=float,
        default=0.2,
        help="Fraction of training over which active loops ramp min→max (0 = always max).",
    )
    parser.add_argument("--num-heads", type=int, default=defaults.num_attention_heads)
    parser.add_argument("--intermediate-size", type=int, default=defaults.intermediate_size)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--rope-scaling-factor", type=float, default=defaults.rope_scaling["factor"])
    parser.add_argument("--disable-hadamard", action="store_true")

    parser.add_argument("--mtp-depth", type=int, default=0)
    parser.add_argument("--mtp-loss-coef", type=float, default=0.3)

    parser.add_argument("--use-rfmoe", action="store_true")
    parser.add_argument("--rfmoe-num-experts", type=int, default=8)
    parser.add_argument("--rfmoe-expert-dim", type=int, default=None)
    parser.add_argument("--rfmoe-theta", type=float, default=0.01)
    parser.add_argument("--rfmoe-density-target", type=float, default=0.25)
    parser.add_argument("--rfmoe-density-eta", type=float, default=0.01)
    parser.add_argument("--rfmoe-locality-coef", type=float, default=0.0)
    parser.add_argument("--rfmoe-zipf-s", type=float, default=1.0)
    parser.add_argument("--rfmoe-uniform-alpha", type=float, default=0.1)
    parser.add_argument("--rfmoe-curriculum-ratio", type=float, default=0.0)
    parser.add_argument("--rfmoe-diversity-coef", type=float, default=0.0)

    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=16)
    parser.add_argument("--total-tokens", type=int, default=50_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--z-loss-coef", type=float, default=1e-4)
    parser.add_argument("--lion-beta1", type=float, default=0.95)
    parser.add_argument("--lion-beta2", type=float, default=0.98)
    parser.add_argument("--mud-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mud-momentum", type=float, default=0.95)
    parser.add_argument("--mud-passes", type=int, default=1)
    parser.add_argument("--no-optimizer-8bit", action="store_true")
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--cooldown-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cooldown-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--stage1-ratio", type=float, default=0.12)
    parser.add_argument("--stage1-weight-mix-start", type=float, default=0.25)
    parser.add_argument("--stage1-activation-mix-start", type=float, default=0.0)
    parser.add_argument("--stage1-activation-bits", type=int, default=8)
    parser.add_argument("--final-activation-bits", type=int, default=4)
    parser.add_argument("--initial-blocks", type=int, default=8)
    parser.add_argument("--final-blocks", type=int, default=16)
    parser.add_argument("--block-growth-ratio", type=float, default=0.6)

    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activation checkpointing (default on for deep/looped training). "
        "Mutually exclusive with --compile in practice.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the model (CUDA only). Default off — conflicts with "
        "gradient checkpointing; use --no-gradient-checkpointing --compile for speed.",
    )
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=20)
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
    if args.min_num_loops < 1:
        parser.error("--min-num-loops must be >= 1")
    if not 0.0 <= args.loop_curriculum_ratio <= 1.0:
        parser.error("--loop-curriculum-ratio must be between 0.0 and 1.0")
    if args.compile and args.gradient_checkpointing:
        print(
            "Note: --compile and --gradient-checkpointing both set; "
            "using gradient checkpointing and skipping torch.compile "
            "(deep looped training needs recompute more than compile). "
            "Pass --no-gradient-checkpointing to prefer compile.",
            flush=True,
        )

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
    if args.resume_from:
        saved_config = load_checkpoint_payload(Path(args.resume_from), map_location="cpu").get("model_config")
        if saved_config:
            valid_fields = {f.name for f in fields(TernaryConfig)}
            model_config = TernaryConfig(**{k: v for k, v in saved_config.items() if k in valid_fields})
    base_model = BitNetDeep(model_config)
    base_model.gradient_checkpointing = args.gradient_checkpointing
    base_model.to(device)

    density_controller = (
        DensityController(target=args.rfmoe_density_target, eta=args.rfmoe_density_eta)
        if any(iter_rfmoe(base_model)) else None
    )

    runner = base_model
    # Policy: checkpoint XOR compile. Checkpoint wins when both requested.
    should_compile = (
        bool(args.compile)
        and not bool(args.gradient_checkpointing)
        and hasattr(torch, "compile")
    )
    if args.compile and args.gradient_checkpointing:
        pass  # already warned at parse time
    elif should_compile and device.type != "cuda":
        print("Skipping torch.compile: only enabled on CUDA.", flush=True)
        should_compile = False
    if should_compile:
        print(f"torch.compile enabled (mode={args.compile_mode}).", flush=True)
        runner = torch.compile(runner, mode=args.compile_mode)
    elif args.gradient_checkpointing:
        print("Gradient checkpointing enabled (torch.compile off).", flush=True)

    max_loops = int(model_config.num_loops)
    min_loops = min(int(args.min_num_loops), max_loops)
    parameter_count = sum(param.numel() for param in base_model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {parameter_count / 1e6:.2f}M")
    print(
        f"Loop curriculum: R={min_loops}→{max_loops} over "
        f"{args.loop_curriculum_ratio:.0%} of tokens "
        f"(effective depth at max ≈ {model_config.effective_depth})"
    )
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
    pin_memory = device.type == "cuda"
    non_blocking = device.type == "cuda"
    early_train_batch_stream: Iterator[Dict[str, torch.Tensor]] = PrefetchStream(
        build_batch_stream(
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
        ),
        pin_memory=pin_memory,
    )
    late_train_batch_stream = None
    if late_train_mixture is not None:
        late_train_batch_stream = PrefetchStream(
            build_batch_stream(
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
            ),
            pin_memory=pin_memory,
        )

    state = TrainerState()
    if args.resume_from:
        state = load_checkpoint(Path(args.resume_from), base_model, optimizer, scheduler, scaler)
        print(f"Resumed from {args.resume_from} at step {state.step}")
        print(
            "Resume restores optimizer/scheduler state only. Training streams restart from the "
            "beginning of each dataset and RNG state is not checkpointed.",
            flush=True,
        )

    start_time = time.time()

    try:
        while state.step < total_steps and state.tokens_processed < args.total_tokens:
            train_progress = state.tokens_processed / max(args.total_tokens, 1)
            quant_metrics = update_quantization_schedule(base_model, train_progress, args)
            active_blocks = update_block_growth(base_model, train_progress, args)
            active_loops = loop_count_for_progress(
                train_progress,
                min_loops=min_loops,
                max_loops=max_loops,
                curriculum_ratio=args.loop_curriculum_ratio,
            )
            rfmoe_s, rfmoe_alpha = rfmoe_staircase_schedule(train_progress, args)
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

            accumulated_loss: Optional[torch.Tensor] = None
            density_sum = 0.0
            step_tokens = 0
            step_start = time.time()

            for _ in range(args.grad_accumulation_steps):
                batch = next(active_train_batch_stream)
                input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
                labels = batch["labels"].to(device, non_blocking=non_blocking)
                segment_ids = batch["segment_ids"].to(device, non_blocking=non_blocking)

                with autocast_context(device, amp_enabled, amp_dtype):
                    if args.mtp_depth > 0:
                        logits, mtp_logits = runner(
                            input_ids,
                            segment_ids=segment_ids,
                            return_mtp=True,
                            num_loops=active_loops,
                        )
                    else:
                        logits, mtp_logits = runner(
                            input_ids,
                            segment_ids=segment_ids,
                            num_loops=active_loops,
                        ), []
                    loss = compute_train_loss(
                        base_model,
                        logits,
                        labels,
                        mtp_logits=mtp_logits,
                        z_loss_coef=args.z_loss_coef,
                        mtp_loss_coef=args.mtp_loss_coef,
                        density_lam=density_controller.lam if density_controller is not None else 0.0,
                        locality_coef=args.rfmoe_locality_coef,
                        diversity_coef=args.rfmoe_diversity_coef,
                        rfmoe_s=rfmoe_s,
                        rfmoe_alpha=rfmoe_alpha,
                    )
                    scaled_loss = loss / args.grad_accumulation_steps

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                detached_loss = loss.detach()
                accumulated_loss = detached_loss if accumulated_loss is None else accumulated_loss + detached_loss
                if density_controller is not None:
                    density_sum += rfmoe_density(base_model)
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

            if density_controller is not None:
                density_controller.update(density_sum / args.grad_accumulation_steps)

            state.step += 1
            state.tokens_processed += step_tokens

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - step_start
            tokens_per_second = step_tokens / max(elapsed, 1e-6)

            if state.step == 1 or state.step % args.log_interval == 0:
                loss_value = float(accumulated_loss) / args.grad_accumulation_steps
                rfmoe_metrics = (
                    {
                        "rfmoe_density": rfmoe_density(base_model),
                        "rfmoe_lambda": density_controller.lam,
                        "rfmoe_zipf_s": rfmoe_s,
                        "rfmoe_alpha": rfmoe_alpha,
                        **({"rfmoe_diversity": float(rfmoe_diversity_loss(base_model).detach())}
                           if args.rfmoe_diversity_coef > 0.0 else {}),
                    }
                    if density_controller is not None else {}
                )
                loop_metrics = collect_loop_train_metrics(
                    base_model, active_loops=active_loops
                )
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
                        **loop_metrics,
                        **rfmoe_metrics,
                        **quant_metrics,
                    },
                )

            if args.eval_interval > 0 and state.step % args.eval_interval == 0:
                eval_metrics = evaluate(
                    runner, tokenizer, val_mixture, args, device, amp_enabled, amp_dtype,
                )
                if eval_metrics:
                    logger.log(state.step, eval_metrics)
                    if eval_metrics["val_loss"] < state.best_val_loss:
                        state.best_val_loss = eval_metrics["val_loss"]
                        best_path = save_checkpoint(
                            output_dir, base_model, optimizer, scheduler, scaler,
                            state, model_config, args, checkpoint_name="best.pt",
                        )
                        print(f"Saved new best checkpoint to {best_path}")

            if args.save_interval > 0 and state.step % args.save_interval == 0:
                checkpoint_path = save_checkpoint(
                    output_dir, base_model, optimizer, scheduler, scaler,
                    state, model_config, args, checkpoint_name=f"step_{state.step:07d}.pt",
                )
                save_checkpoint(
                    output_dir, base_model, optimizer, scheduler, scaler,
                    state, model_config, args, checkpoint_name="last.pt",
                )
                print(f"Saved checkpoint to {checkpoint_path}")

        final_path = save_checkpoint(
            output_dir, base_model, optimizer, scheduler, scaler,
            state, model_config, args, checkpoint_name="final.pt",
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
