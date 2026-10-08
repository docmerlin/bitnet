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
from mlx_train import scheduled_value, config_from_saved
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


def _generation_argmax(logits: mx.array, valid_vocab_size: int | None) -> mx.array:
    """Exclude undefined trailing IDs without resizing padded checkpoint heads."""
    if valid_vocab_size is not None and not 1 <= valid_vocab_size <= logits.shape[-1]:
        raise ValueError("valid_vocab_size must be between 1 and the logit vocabulary size")
    return mx.argmax(logits[..., :valid_vocab_size], axis=-1)


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


def dblock_greedy_generate(
    model: MLXBitNet,
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    euler_steps: int | None = None,
    valid_vocab_size: int | None = None,
    compile_step: bool = False,
) -> list[int]:
    """AR DiffusionBlocks decode (paper App. E.4): greedy, one token at a time.

    Each new token starts as σ_max noise and is denoised by
    ``dblock_sample_steps`` evaluations (B>1: one per block, paper T=B;
    B=1: K evals of the Huginn stack) on equi-probability σ from σ_max to
    σ_min, conditioned on the clean prefix. Past positions are re-noised at the
    current σ, exactly as in training. ``dblock_infer='loops'`` is one σ_min
    eval that unrolls Huginn R instead.

    B>1 keeps one PaTH cache per eval: a past position's noise is drawn once,
    when its target is committed, rather than fresh per token (same marginal
    as training), so each eval only runs the new positions. ``compile_step``
    runs each token (commit + B queries) as one mx.compile per open-chunk
    length, falling back to eager if compile fails. Off by default: pipelined
    eager is already GPU-bound (1B: 28.5 vs 29.2 ms/token warm) and each new
    open length costs ~75 ms of tracing, paid again per call.
    ponytail: B=1 still re-runs the full prefix per eval (K Hyperloop caches would
    cost K× AR cache memory); cache it if B=1 decode speed matters.
    ``valid_vocab_size`` restricts selection to defined IDs in a padded vocabulary.
    """
    if model.config.train_mode != "dblock":
        raise ValueError("dblock_greedy_generate requires train_mode='dblock'")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    schedule = model.config.noise_schedule()
    loops = None
    if model.config.dblock_infer == "loops":
        if model.config.dblock_blocks != 1:
            raise ValueError("dblock_infer='loops' requires dblock_blocks=1")
        loops = int(getattr(model, "inference_num_loops", None) or model.config.num_loops)
        if loops < 1:
            raise ValueError("inference_num_loops must be positive")
        sigmas = (schedule.sigma_min,)
    else:
        sigmas = schedule.sample_sigmas(model.config.dblock_sample_steps(euler_steps))
    hidden_size = model.config.hidden_size
    if model.config.dblock_blocks > 1:
        return _dblock_cached_generate(
            model, prompt, max_new_tokens, eos_token_id, sigmas, valid_vocab_size, compile_step
        )
    table = model.dblock_denoise_table()
    out = list(prompt)
    for _ in range(max_new_tokens):
        context = mx.array([out], dtype=mx.int32)
        # Position i carries the noisy embedding of token i + 1.
        past = model.dblock_clean_embeddings(context[:, 1:]).astype(mx.float32)
        z = sigmas[0] * mx.random.normal((1, 1, hidden_size))
        for index, sigma in enumerate(sigmas):
            noisy_past = past + sigma * mx.random.normal(past.shape)
            hidden = model.dblock_forward_from_z(
                mx.concatenate([noisy_past, z], axis=1),
                mx.array([sigma], dtype=mx.float32),
                context,
                None,
                block_id=schedule.block_for_sigma(sigma),
                num_loops=loops,
            )
            logits = model.logits_from(hidden[:, -1:])
            if index + 1 < len(sigmas):
                # VE Euler (paper Eq. 5): z ← z + (σ' − σ)/σ · (z − D).
                denoised = model.dblock_denoised(logits, table)
                z = z + (sigmas[index + 1] - sigma) / sigma * (z - denoised)
            mx.eval(z, logits)
        token = int(_generation_argmax(logits, valid_vocab_size)[0, -1].item())
        out.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    return out


def _dblock_cached_generate(
    model: MLXBitNet,
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    sigmas: tuple[float, ...],
    valid_vocab_size: int | None,
    compile_step: bool = False,
) -> list[int]:
    schedule = model.config.noise_schedule()
    evals = [(sigma, mx.array([sigma], dtype=mx.float32), schedule.block_for_sigma(sigma)) for sigma in sigmas]
    caches = [model.new_dblock_cache(block) for _, _, block in evals]
    conds = [model.dblock_decode_cond(sigma, block) for _, sigma, block in evals]
    table = model.dblock_denoise_table()
    hidden_size = model.config.hidden_size
    mx.eval(table, conds)

    def commit(caches, context, targets, noise, pipeline: bool) -> None:
        # Positions with known next tokens: clean token i + noisy target i + 1.
        clean = model.dblock_clean_embeddings(targets).astype(mx.float32)
        for (value, sigma, block), cache, cond, eps in zip(evals, caches, conds, noise):
            model.dblock_decode(clean + value * eps, sigma, context, cache, block_id=block, cond=cond)
            if pipeline:  # GPU runs this block while Python builds the next (~11 ms of graph per token)
                mx.async_eval(cache.arrays())

    def token_step(caches, context, targets, query, noise, z, pipeline: bool = False) -> mx.array:
        """Commit the previous position (if any) to every cache, then denoise ``query``'s next token."""
        if context is not None:
            commit(caches, context, targets, noise, pipeline)
        for index, ((value, sigma, block), cache, cond) in enumerate(zip(evals, caches, conds)):
            # Query on a clone: the query position is committed only once its token is known.
            logits = model.logits_from(model.dblock_decode(z, sigma, query, cache.clone(), block_id=block, cond=cond))
            if index + 1 < len(evals):
                z = z + (sigmas[index + 1] - value) / value * (z - model.dblock_denoised(logits, table))
                if pipeline:
                    mx.async_eval(z)
        return logits

    # One mx.compile per open-chunk length (as enable_compiled_inference); every
    # cache has seen the same positions, so the first layer's length is the key.
    layouts = [model._cache_layout(cache) for cache in caches]
    compiled: dict[int, object] = {}

    def make_pure_step():
        def pure(context, targets, query, noise, z, *flat):
            views, offset = [], 0
            for cache, layout in zip(caches, layouts):
                count = len(model._flatten_inference_cache(cache))
                view = model._unflatten_inference_cache(
                    list(flat[offset : offset + count]), layout, 1, cache.weight_cache, position=1
                )
                views.append(view)
                offset += count
            logits = token_step(views, context, targets, query, noise, z)
            return (logits, *(a for view in views for a in model._flatten_inference_cache(view)))

        return pure

    def compiled_step(context, targets, query, noise, z) -> mx.array | None:
        attention = caches[0].layers[0].attention
        key = 0 if attention.q is None else int(attention.q.shape[2])
        flats = [model._flatten_inference_cache(cache) for cache in caches]
        try:
            fn = compiled.get(key)
            if fn is None:
                fn = compiled[key] = mx.compile(make_pure_step())
            result = fn(context, targets, query, noise, z, *(a for flat in flats for a in flat))
            mx.eval(result)
        except Exception:
            return None
        offset = 1
        for cache, flat in zip(caches, flats):
            model._apply_flat_to_inference_cache(cache, list(result[offset : offset + len(flat)]))
            cache.position += 1
            offset += len(flat)
        return result[0]

    out = list(prompt)
    if len(out) > 2:  # positions 0..P-3; P-2 is committed by the first token step
        context = mx.array([out[:-2]], dtype=mx.int32)
        targets = mx.array([out[1:-1]], dtype=mx.int32)
        noise = [mx.random.normal((1, len(out) - 2, hidden_size)) for _ in evals]
        commit(caches, context, targets, noise, pipeline=True)
    for _ in range(max_new_tokens):
        query = mx.array([out[-1:]], dtype=mx.int32)
        context = targets = noise = None
        if len(out) > 1:
            context = mx.array([out[-2:-1]], dtype=mx.int32)
            targets = query
            noise = mx.random.normal((len(evals), 1, 1, hidden_size))
        z = sigmas[0] * mx.random.normal((1, 1, hidden_size))
        logits = None
        if compile_step and context is not None and caches[0].position > 0:
            logits = compiled_step(context, targets, query, noise, z)
            if logits is None:
                compile_step = False  # unsupported here: eager from now on
        if logits is None:
            logits = token_step(caches, context, targets, query, noise, z, pipeline=True)
        token = int(_generation_argmax(logits, valid_vocab_size)[0, -1].item())
        out.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    return out


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
    checkpoint = checkpoint.resolve()
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config_data = dict(metadata["model_config"])
    config_data["engram_layer_ids"] = tuple(config_data["engram_layer_ids"])
    model = MLXBitNet(config_from_saved(config_data))
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
        if not key.endswith(
            (
                ".memory_m",
                ".memory_z",
                ".memory_initialized",
                ".memory_k",
                ".memory_v",
                # Retired activation-quant module state from older checkpoints.
                ".activation_levels",
                ".activation_level_pair",
                ".weight_mix_value",
                ".activation_mix_value",
            )
        )
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
    model.recurrent_quantized_matmul = True
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
    speculative: bool = True,
    valid_vocab_size: int | None = None,
) -> tuple[Proposal, Verification]:
    """Cached greedy callbacks; optionally skip MTP and exclude padded vocabulary IDs."""
    depth = model.config.mtp_depth if speculative else 0
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
        main = model.logits_from(states)
        if depth:
            logits = mx.concatenate((main, model.draft_logits(states)), axis=1)
        else:
            logits = main
        result = _generation_argmax(logits, valid_vocab_size)
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
        verifier = model.logits_from(states)
        verified = _generation_argmax(verifier, valid_vocab_size)
        drafts = (
            _generation_argmax(model.draft_logits(states), valid_vocab_size)
            if depth else mx.zeros((1, 0), dtype=mx.int32)
        )
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
        help="Eval-time recurrent loop override (default: checkpoint schedule). "
        "dblock Euler ignores this; use --dblock-euler-steps.",
    )
    parser.add_argument(
        "--dblock-euler-steps",
        type=int,
        default=None,
        help="B=1 dblock denoiser evaluations per token (default: checkpoint dblock_euler_steps, paper 50). "
        "Independent of --num-loops.",
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
        "--dblock-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mx.compile the dblock token step (default: off; no warm gain, ~75 ms trace per open length).",
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
    started = time.perf_counter()
    stats = GenerationStats()
    if model.config.train_mode == "dblock":
        tokens = dblock_greedy_generate(
            model,
            prompt,
            args.max_new_tokens,
            tokenizer.eos_id,
            euler_steps=args.dblock_euler_steps,
            valid_vocab_size=len(tokenizer),
            compile_step=args.dblock_compile,
        )
    else:
        speculative = args.speculative and model.config.mtp_depth > 0
        propose, verify = model_callbacks(
            model, profile=profile, speculative=speculative, valid_vocab_size=len(tokenizer)
        )
        if speculative:
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
        f"num_loops={model.inference_num_loops} | "
        f"dblock_euler_steps={model.config.dblock_sample_steps(args.dblock_euler_steps) if model.config.train_mode == 'dblock' else '-'} | "
        f"path_decode={getattr(model, 'path_decode_mode', 'last')} | "
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
