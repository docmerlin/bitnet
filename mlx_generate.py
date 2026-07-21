"""Generate text from an MLX BitNet checkpoint, optionally using exact MTP speculation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_train import scheduled_value
from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
from training.schedules import loop_count_for_progress


Proposal = Callable[[list[int]], list[int]]
Verification = Callable[[list[int], list[int]], tuple[list[int], list[int]]]


@dataclass
class GenerationStats:
    proposal_calls: int = 0
    verification_calls: int = 0
    accepted_drafts: int = 0
    rejected_drafts: int = 0


@dataclass
class PhaseTimings:
    """Optional wall-time breakdown for generate (seconds)."""

    prefill_s: float = 0.0
    decode_s: float = 0.0
    eval_sync_s: float = 0.0
    decode_steps: int = 0
    prefill_tokens: int = 0

    def summary_lines(self, generated: int, elapsed: float) -> list[str]:
        decode_ms = (self.decode_s / max(self.decode_steps, 1)) * 1000
        lines = [
            (
                f"profile prefill_s={self.prefill_s:.4f} | decode_s={self.decode_s:.4f} | "
                f"eval_sync_s={self.eval_sync_s:.4f} | decode_steps={self.decode_steps} | "
                f"ms_per_decode_step={decode_ms:.3f}"
            ),
            (
                f"profile prefill_tokens={self.prefill_tokens} | generated_tokens={generated} | "
                f"e2e_tokens_per_second={generated / max(elapsed, 1e-9):.3f}"
            ),
        ]
        return lines


def greedy_generate(
    prompt: list[int],
    max_new_tokens: int,
    propose: Proposal,
    eos_token_id: int | None = None,
) -> list[int]:
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        token = propose(tokens)[0]
        tokens.append(token)
        if token == eos_token_id:
            break
    return tokens


def speculative_greedy_generate(
    prompt: list[int],
    max_new_tokens: int,
    propose: Proposal,
    verify: Verification,
    eos_token_id: int | None = None,
    stats: GenerationStats | None = None,
) -> list[int]:
    stats = GenerationStats() if stats is None else stats
    tokens = list(prompt)
    generated = 0
    proposals = None
    while generated < max_new_tokens:
        if proposals is None:
            proposals = propose(tokens)
            stats.proposal_calls += 1
        first = proposals[0]
        tokens.append(first)
        generated += 1
        if generated == max_new_tokens or first == eos_token_id:
            break

        verified, next_drafts = verify(tokens[:-1], proposals)
        stats.verification_calls += 1
        all_accepted = True
        for index in range(1, len(proposals)):
            target = verified[index - 1]
            accepted = proposals[index] == target
            token = proposals[index] if accepted else target
            if accepted:
                stats.accepted_drafts += 1
            else:
                stats.rejected_drafts += 1
            tokens.append(token)
            generated += 1
            if token == eos_token_id or generated == max_new_tokens:
                return tokens
            if not accepted:
                all_accepted = False
                break
        proposals = [verified[-1], *next_drafts] if all_accepted else None
    return tokens


def load_model(
    checkpoint: Path,
    *,
    num_loops: int | None = None,
    path_decode_mode: str = "last",
    pin_weights: bool = True,
    compile_step: bool = True,
) -> tuple[MLXBitNet, dict]:
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config_data = dict(metadata["model_config"])
    config_data["engram_layer_ids"] = tuple(config_data["engram_layer_ids"])
    model = MLXBitNet(MLXBitNetConfig(**config_data))
    training_args = metadata.get("training_args") or {}
    dtype = {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }[training_args.get("precision", "bfloat16")]
    model.set_dtype(dtype)

    loaded = mx.load(str(checkpoint))
    parameters = dict(tree_flatten(model.parameters()))
    expected = {
        key
        for key in parameters
        if not key.endswith((".memory_k", ".memory_v", ".memory_initialized"))
    }
    if expected != loaded.keys():
        raise ValueError(
            f"Model checkpoint mismatch: missing={sorted(expected - loaded.keys())}; "
            f"unexpected={sorted(loaded.keys() - expected)}"
        )
    model.load_weights(list(loaded.items()), strict=False)

    trainer_state = metadata.get("trainer_state") or {}
    progress = trainer_state.get("tokens_processed", 0) / max(training_args.get("total_tokens", 1), 1)
    blocks = round(
        scheduled_value(
            training_args.get("initial_blocks", config_data["block_size"]),
            training_args.get("final_blocks", config_data["block_size"]),
            progress,
            training_args.get("block_growth_ratio", 0.0),
        )
    )
    model.set_active_blocks(blocks)
    model.set_inference_block_width(math.ceil(training_args.get("sequence_length", 512) / blocks))
    scheduled_loops = loop_count_for_progress(
        progress,
        min_loops=training_args.get("min_num_loops", config_data["num_loops"]),
        max_loops=config_data["num_loops"],
        curriculum_ratio=training_args.get("loop_curriculum_ratio", 0.0),
        curriculum_start_ratio=training_args.get("loop_curriculum_start_ratio", 0.0),
    )
    if num_loops is not None:
        if num_loops < 1:
            raise ValueError("num_loops override must be >= 1")
        if num_loops > config_data["num_loops"]:
            raise ValueError(
                f"num_loops override {num_loops} exceeds model max {config_data['num_loops']}"
            )
        model.inference_num_loops = num_loops
    else:
        model.inference_num_loops = scheduled_loops
    model.scheduled_inference_num_loops = scheduled_loops
    stage_ratio = training_args.get("stage1_ratio", 0.0)
    fraction = 1.0 if stage_ratio <= 0 else min(progress / stage_ratio, 1.0)
    weight_mix = training_args.get("stage1_weight_mix_start", 0.25) + fraction * (
        1.0 - training_args.get("stage1_weight_mix_start", 0.25)
    )
    activation_mix = training_args.get("stage1_activation_mix_start", 0.0) + fraction * (
        1.0 - training_args.get("stage1_activation_mix_start", 0.0)
    )
    stage_bits = training_args.get("stage1_activation_bits", 8)
    final_bits = training_args.get("final_activation_bits", 4)
    model.set_quantization_state(weight_mix, activation_mix, round(stage_bits - fraction * (stage_bits - final_bits)))
    model.recurrent_quantized_matmul = weight_mix >= 1.0
    model.set_path_decode_mode(path_decode_mode)
    model.eval()
    mx.eval(model.parameters())
    if pin_weights:
        model.pin_inference_weights(dtype)
    if compile_step:
        model.enable_compiled_inference()
    return model, training_args


def model_callbacks(
    model: MLXBitNet,
    *,
    profile: PhaseTimings | None = None,
) -> tuple[Proposal, Verification]:
    depth = model.config.mtp_depth
    loops = getattr(model, "inference_num_loops", None)
    cache = model.new_inference_cache(num_loops=loops)
    cached_tokens: list[int] = []
    cached_state = None
    pending = {}
    # Compiled steps already mx.eval their full result; avoid a second full-cache sync.
    compiled_decode = getattr(model, "_compiled_inference_step", None) is not None

    def _eval(*arrays):
        if profile is None:
            mx.eval(*arrays)
            return
        started = time.perf_counter()
        mx.eval(*arrays)
        profile.eval_sync_s += time.perf_counter() - started

    def sync(token_ids: list[int]):
        nonlocal cache, cached_tokens, cached_state
        current_matches = cached_state is not None and token_ids[: len(cached_tokens)] == cached_tokens
        best_tokens = cached_tokens if current_matches else []
        best = None
        for candidate_tokens, candidate in pending.items():
            if len(candidate_tokens) > len(best_tokens) and token_ids[: len(candidate_tokens)] == list(candidate_tokens):
                best_tokens = list(candidate_tokens)
                best = candidate
        if best is not None:
            cache, cached_state = best[0].clone(), best[1]
            cached_tokens = best_tokens
        elif not current_matches:
            if not token_ids:
                raise ValueError("token_ids must not be empty")
            cache = model.new_inference_cache(num_loops=loops)
            cached_tokens = list(token_ids)
            started = time.perf_counter()
            states = model.prefill(mx.array([token_ids], dtype=mx.int32), cache)
            cached_state = states[:, -1:]
            # Pass cache.arrays() as one tree (not *splat) to cut Python eval overhead.
            _eval(cached_state, cache.arrays())
            if profile is not None:
                profile.prefill_s += time.perf_counter() - started
                profile.prefill_tokens = len(token_ids)
            return cached_state
        for token in token_ids[len(cached_tokens) :]:
            started = time.perf_counter()
            cached_state = model.inference_step(mx.array([[token]], dtype=mx.int32), cache)
            cached_tokens.append(token)
            if compiled_decode:
                _eval(cached_state)
            else:
                _eval(cached_state, cache.arrays())
            if profile is not None:
                profile.decode_s += time.perf_counter() - started
                profile.decode_steps += 1
        return cached_state

    def propose(token_ids: list[int]) -> list[int]:
        states = sync(token_ids)
        pending.clear()
        main = states @ model.embedding.weight.T
        if depth:
            logits = mx.concatenate((main, model.draft_logits(states)), axis=1)
        else:
            logits = main
        result = mx.argmax(logits, axis=-1)
        _eval(result)
        return result[0].tolist()

    def verify(prefix: list[int], candidates: list[int]) -> tuple[list[int], list[int]]:
        sync(prefix)
        branch_cache = cache.clone()
        pending.clear()
        started = time.perf_counter()
        states = model.inference_extend(mx.array([candidates], dtype=mx.int32), branch_cache)
        _eval(states, branch_cache.arrays())
        if profile is not None:
            profile.decode_s += time.perf_counter() - started
            profile.decode_steps += 1
        pending[tuple([*prefix, *candidates])] = (branch_cache, states[:, -1:])
        verifier = states @ model.embedding.weight.T
        verified = mx.argmax(verifier, axis=-1)
        drafts = mx.argmax(model.draft_logits(states), axis=-1) if depth else mx.zeros((1, 0), dtype=mx.int32)
        _eval(verified, drafts)
        return verified[0].tolist(), drafts[0].tolist()

    return propose, verify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--speculative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Eval-time recurrent loop override (default: checkpoint schedule).",
    )
    parser.add_argument(
        "--path-decode",
        choices=("last", "recompute"),
        default="last",
        help="PaTH decode mode: last-query incremental (default) or full open-chunk recompute baseline.",
    )
    parser.add_argument(
        "--pin-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin effective/packed ternary weights once per generation (default: on).",
    )
    parser.add_argument(
        "--compile-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try mx.compile on inference_step (default: on; falls back if unsupported).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print prefill/decode/eval phase timings alongside tokens/second.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    model, training_args = load_model(
        args.checkpoint,
        num_loops=args.num_loops,
        path_decode_mode=args.path_decode,
        pin_weights=args.pin_weights,
        compile_step=args.compile_step,
    )
    tokenizer = HierarchicalTokenizer(
        max_patch_size=training_args.get("tokenizer_max_patch_size", 8),
        vocab_size_target=model.config.vocab_size,
    )
    prompt = tokenizer.encode(args.prompt)
    if not prompt:
        prompt = [tokenizer.bos_id]
    profile = PhaseTimings() if args.profile else None
    propose, verify = model_callbacks(model, profile=profile)
    started = time.perf_counter()
    stats = GenerationStats()
    if args.speculative and model.config.mtp_depth:
        tokens = speculative_greedy_generate(
            prompt,
            args.max_new_tokens,
            propose,
            verify,
            tokenizer.eos_id,
            stats,
        )
    else:
        tokens = greedy_generate(prompt, args.max_new_tokens, propose, tokenizer.eos_id)
    elapsed = time.perf_counter() - started
    generated = len(tokens) - len(prompt)
    print(tokenizer.decode(tokens))
    print(
        f"generated_tokens={generated} | elapsed_sec={elapsed:.3f} | "
        f"tokens_per_second={generated / elapsed:.3f} | "
        f"num_loops={model.inference_num_loops} | path_decode={getattr(model, 'path_decode_mode', 'last')} | "
        f"compiled={getattr(model, '_compiled_inference_step', None) is not None}"
    )
    if stats.verification_calls:
        print(
            f"target_calls={stats.proposal_calls + stats.verification_calls} | "
            f"accepted_drafts={stats.accepted_drafts} | rejected_drafts={stats.rejected_drafts}"
        )
    if profile is not None:
        for line in profile.summary_lines(generated, elapsed):
            print(line)


if __name__ == "__main__":
    main()
