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

from config import effective_path_window
from data.presets import parse_mixture
from data.streams import build_batch_stream
from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_optim import CMUD
from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer


_MEMORY_STATE_NAMES = (".memory_m", ".memory_z", ".memory_initialized")
# Quantisation knobs that live as module state only so mx.compile treats them as
# graph inputs rather than baking them in as constants. They are derived from the
# training schedule, not learned, and set_quantization_state rebuilds them every
# step -- so saving them bloats every checkpoint and makes the strict key
# comparison below reject any file written by a build with a different set.
_RUNTIME_QUANT_NAMES = (
    ".activation_levels",
    ".activation_level_pair",
    ".weight_mix_value",
    ".activation_mix_value",
)
_EXCLUDED_STATE_NAMES = _MEMORY_STATE_NAMES + _RUNTIME_QUANT_NAMES


#: Config fields this repo has removed. A checkpoint written before the removal
#: still carries them, and passing one to the dataclass is a TypeError. Dropping
#: *known* removed names keeps old checkpoints loadable while an genuinely
#: unrecognised key still fails loudly rather than being silently ignored.
RETIRED_CONFIG_FIELDS = frozenset(
    {
        "use_ffn_mid",
        "use_mamba3_layers",
        "mamba_layer_period",
        "mamba_d_state",
        "mamba_expand",
        "mamba_headdim",
        "mamba_d_conv",
        "mamba_dt_min",
        "mamba_dt_max",
        "mamba_a_floor",
        "use_mamba_scan_kernel",
    }
)


def config_from_saved(saved_config: dict) -> MLXBitNetConfig:
    """Rebuild a model config from a checkpoint, dropping retired fields."""
    retired = RETIRED_CONFIG_FIELDS & saved_config.keys()
    if retired:
        print(
            f"checkpoint predates the removal of {sorted(retired)}; ignoring",
            flush=True,
        )
    settings = {k: v for k, v in saved_config.items() if k not in RETIRED_CONFIG_FIELDS}
    if "engram_layer_ids" in settings:
        settings["engram_layer_ids"] = tuple(settings["engram_layer_ids"])
    return MLXBitNetConfig(**settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/mlx_bitnet")
    parser.add_argument("--train-mixture", default="fineweb_edu=0.7,dclm=0.3")
    parser.add_argument("--early-train-mixture", default="")
    parser.add_argument("--late-train-mixture", default="")
    parser.add_argument("--mixture-switch-ratio", type=float, default=0.7)
    parser.add_argument("--val-mixture", default="fineweb_edu=0.5,dclm=0.5")
    parser.add_argument("--validation-offset-examples", type=int, default=25000)
    parser.add_argument("--validation-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument("--max-document-tokens", type=int, default=32768)
    parser.add_argument("--tokenizer-max-patch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument(
        "--attn-res-mode",
        choices=("kimi", "sandwich"),
        default="kimi",
        help="Residual path: kimi Block AttnRes (default) or legacy sandwich.",
    )
    parser.add_argument(
        "--attn-res-group-size",
        type=int,
        default=None,
        help="Transformer layers per AttnRes depth-block (default: unique_layers//8).",
    )
    parser.add_argument("--num-prelude-layers", type=int, default=2)
    parser.add_argument("--num-recurrent-layers", type=int, default=4)
    parser.add_argument("--num-coda-layers", type=int, default=2)
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--min-num-loops", type=int, default=1)
    parser.add_argument("--loop-curriculum-start-ratio", type=float, default=0.0)
    parser.add_argument("--loop-curriculum-ratio", type=float, default=0.2)
    parser.add_argument("--initial-blocks", type=int, default=8)
    parser.add_argument("--final-blocks", type=int, default=16)
    parser.add_argument("--block-growth-ratio", type=float, default=0.6)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--path-window-size", type=int, default=1024)
    parser.add_argument(
        "--topk-blocks-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a block-granular top-k retrieval branch alongside local PaTH and "
        "Infini memory. Training only — decode raises if this is on.",
    )
    parser.add_argument(
        "--infini-memory-expand",
        type=int,
        default=0,
        help="Infini memory key-feature width (0 = head_dim). Capacity is this times "
        "head_dim per head; useful ceiling is tokens-per-reset (one sequence).",
    )
    parser.add_argument(
        "--infini-feature-map",
        choices=("elu", "favor"),
        default="elu",
        help="Infini memory kernel: elu (paper) or favor (exp kernel, content-dependent reads).",
    )
    parser.add_argument("--topk-blocks", type=int, default=4)
    parser.add_argument("--topk-block-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation-steps", type=int, default=4)
    parser.add_argument("--total-tokens", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--embedding-learning-rate",
        type=float,
        default=None,
        help="Rate for the token embedding and the untied output head (default: "
        "--learning-rate). modded-nanogpt runs these well above the body rate; "
        "try 10-30x once --no-tie-word-embeddings is in play.",
    )
    parser.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Share the output projection with the input embedding. Off by default: "
        "an untied head plus a higher embedding rate was modded-nanogpt's largest "
        "win after Muon. Costs vocab*hidden extra parameters.",
    )
    parser.add_argument("--mud-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mud-momentum", type=float, default=0.95)
    parser.add_argument("--mud-passes", type=int, default=1)
    parser.add_argument("--mud-block-size", type=int, default=64)
    parser.add_argument(
        "--mud-neuron-norm",
        action="store_true",
        help="Re-normalise neuron rows after MUD whitening. MUD transposes tall "
        "matrices, so qkv and the FFN up-projection come out with a 0.072 "
        "coefficient of variation across neuron norms (Muon: 0.030). This is the "
        "stateless form of NorMuon's fix. Off by default -- unvalidated, needs an A/B.",
    )
    parser.add_argument("--lion-beta1", type=float, default=0.95)
    parser.add_argument("--lion-beta2", type=float, default=0.98)
    parser.add_argument("--no-optimizer-8bit", action="store_true")
    parser.add_argument("--cmud-momentum-8bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cmud-master-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--cautious-weight-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only decay coordinates the MUD step is already shrinking "
        "(modded-nanogpt records 43/50). --no-cautious-weight-decay decays everything.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "wsd"),
        default="cosine",
        help="wsd = warmup, flat at peak, then linear decay over --cooldown-ratio "
        "(set that to ~0.4) down to --min-lr-ratio, not to zero.",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--cooldown-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cooldown-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--z-loss-coef", type=float, default=1e-4)
    parser.add_argument("--mtp-depth", type=int, default=4)
    parser.add_argument("--mtp-loss-coef", type=float, default=0.3)
    parser.add_argument("--engram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--engram-layer-ids", default="1,15")
    parser.add_argument(
        "--engram-vocab-size",
        type=int,
        default=None,
        help="Engram table slots; default auto-sizes to ~engram-param-fraction of body.",
    )
    parser.add_argument(
        "--engram-param-fraction",
        type=float,
        default=0.05,
        help="When engram-vocab-size is omitted, target Engram/body params (default 0.05).",
    )
    parser.add_argument("--use-rfmoe", action="store_true")
    parser.add_argument("--rfmoe-num-experts", type=int, default=8)
    parser.add_argument("--rfmoe-expert-dim", type=int, default=None)
    parser.add_argument("--rfmoe-rank", type=int, default=None)
    parser.add_argument("--rfmoe-theta", type=float, default=0.01)
    parser.add_argument("--rfmoe-backend", choices=("auto", "metal", "hybrid", "host"), default="auto")
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
    parser.add_argument("--final-activation-bits", type=int, default=8)
    parser.add_argument("--precision", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpoint-scope", choices=("recurrent", "all"), default="recurrent")
    parser.add_argument("--path-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-quantized-matmul", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-phases", action="store_true")
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
    if min(args.initial_blocks, args.final_blocks) < 1:
        raise ValueError("initial-blocks and final-blocks must be positive")
    if args.mud_block_size is not None and args.mud_block_size < 1:
        raise ValueError("mud-block-size must be positive")
    if min(args.stage1_activation_bits, args.final_activation_bits) < 2:
        raise ValueError("activation bits must be at least 2")
    if not 0 <= args.mixture_switch_ratio <= 1:
        raise ValueError("mixture-switch-ratio must be between zero and one")
    if args.min_num_loops < 1:
        raise ValueError("min-num-loops must be positive")
    if not 0 <= args.loop_curriculum_start_ratio <= args.loop_curriculum_ratio <= 1:
        raise ValueError("loop curriculum must satisfy 0 <= start <= end <= 1")
    if args.lr_schedule == "wsd" and not args.cooldown_steps and args.cooldown_ratio < 0.1:
        # WSD is a plateau plus a long ramp down. Left at the cosine default the
        # ramp is 5% of the run, which is a constant-LR run with a cliff at the
        # end -- worse than the cosine it replaced, and silently so.
        raise ValueError(
            "--lr-schedule wsd needs a long decay: set --cooldown-ratio to ~0.4 "
            "(or --cooldown-steps explicitly)"
        )


def _compile_supported(config: MLXBitNetConfig) -> bool:
    """Whether mx.compile can build this model's graph at all.

    RFMoE's non-metal backends compact dynamically. The top-k retrieval branch
    emits a data-dependent gather per chunk over a growing history, which overflows
    the compiler ("unordered_map::at: key not found") once block_size >= 7 under
    kimi AttnRes — the same wall that caps the Infini carry to one-step BPTT.
    """
    if config.use_topk_blocks:
        return False
    return not config.use_rfmoe or config.rfmoe_backend == "metal"


def _gradient_compile_safe(
    config: MLXBitNetConfig,
    requested: bool,
    sequence_length: int,
    active_blocks: int,
) -> bool:
    return requested and sequence_length % active_blocks == 0 and _compile_supported(config)


def _masked_ce(logits, targets, valid):
    safe_targets = mx.where(valid, targets, 0)
    # Cross entropy at the logits' own dtype rather than a forced fp32 copy --
    # modded-nanogpt record 37. At vocab 32768, sequence 1024, batch 4 the copy
    # is 537 MB and buys nothing: the gradient reaching a bf16 hidden state is
    # bit-identical either way (measured, relative L2 error 0.0) and the reported
    # loss moves 5e-4 relative. An fp32 run still gets fp32 here. The per-token
    # losses are summed in fp32 because that reduction does need the range.
    losses = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    return mx.sum(losses.astype(mx.float32) * valid) / mx.maximum(mx.sum(valid), 1)


def mtp_head_index(step: int, microbatch_index: int, accumulation_steps: int, depth: int) -> int:
    return ((step - 1) * accumulation_steps + microbatch_index) % depth


def prepare_mtp_batch(targets, segment_ids, label_segment_ids, index: int, depth: int):
    shift = index + 1
    shifted_targets = mx.concatenate((targets[:, shift:], mx.zeros_like(targets[:, :shift])), axis=1)
    valid = segment_ids[:, :-shift] == label_segment_ids[:, shift:]
    valid = mx.concatenate((valid, mx.zeros(segment_ids[:, :shift].shape, dtype=mx.bool_)), axis=1)
    selector = mx.arange(depth) == index
    return shifted_targets, valid, selector


def create_gradient_step(
    model: MLXBitNet,
    *,
    compile_step: bool,
    num_loops: int,
    z_loss_coef: float = 0.0,
    mtp_loss_coef: float = 0.0,
    locality_coef: float = 0.0,
    diversity_coef: float = 0.0,
    gradient_checkpointing: bool | str = False,
):
    def loss_fn(
        inputs,
        targets,
        segment_ids,
        label_segment_ids,
        density_lam,
        rfmoe_s,
        rfmoe_alpha,
        mtp_targets=None,
        mtp_valid=None,
        mtp_selector=None,
    ):
        return_mtp = model.config.mtp_depth > 0
        if return_mtp and mtp_selector is not None:
            hidden = model.hidden_states(
                inputs,
                segment_ids,
                num_loops,
                checkpoint_activations=gradient_checkpointing,
            )
            logits = model.logits_from(hidden)
            selected_mtp_logits = model.selected_mtp_logits(hidden, mtp_selector)
            mtp_logits = []
        else:
            output = model(
                inputs,
                segment_ids,
                num_loops=num_loops,
                return_mtp=return_mtp,
                checkpoint_activations=gradient_checkpointing,
            )
            logits, mtp_logits = output if return_mtp else (output, [])
        valid = segment_ids == label_segment_ids
        loss = _masked_ce(logits, targets, valid)
        if z_loss_coef > 0:
            # float32 here, unlike _masked_ce above. The cross-entropy term was
            # measured to give bit-identical gradients at the logits' own dtype;
            # this one squares its result, and a bfloat16 logsumexp over 32k
            # logits carries ~0.06 absolute error, which the square turns into
            # ~1.8 on the term the regulariser is trying to control.
            log_z = mx.logsumexp(logits.astype(mx.float32), axis=-1)
            loss = loss + z_loss_coef * mx.sum(mx.square(log_z) * valid) / mx.maximum(mx.sum(valid), 1)
        if return_mtp and mtp_selector is not None:
            loss = loss + mtp_loss_coef * _masked_ce(selected_mtp_logits, mtp_targets, mtp_valid)
        else:
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
    if compile_step and _compile_supported(model.config):
        gradient_step = partial(mx.compile, inputs=model.state, outputs=model.state)(gradient_step)
    return gradient_step


def create_train_step(model: MLXBitNet, optimizer: optim.Optimizer, *, compile_step: bool = True):
    compile_step = compile_step and _compile_supported(model.config)
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


def create_apply_step(
    model: MLXBitNet,
    optimizer: CMUD,
    *,
    grad_clip: float,
    compile_step: bool = True,
):
    state = [model.state, optimizer.state]

    def apply_step(gradients, lr_scale):
        gradients, grad_norm = optim.clip_grad_norm(gradients, grad_clip)
        optimizer.set_lr_multiplier(lr_scale)
        optimizer.update(model, gradients)
        return grad_norm

    if compile_step:
        apply_step = partial(mx.compile, inputs=state, outputs=state)(apply_step)
    return apply_step, state


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
    stream_state: dict | None = None,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.safetensors"
    parameters = {
        key: value
        for key, value in tree_flatten(model.parameters())
        if not key.endswith(_EXCLUDED_STATE_NAMES)
    }
    mx.save_safetensors(str(path), parameters)
    optimizer_path = checkpoint_dir / f"{name}.optimizer.safetensors"
    mx.save_safetensors(str(optimizer_path), dict(tree_flatten(optimizer.state)))
    metadata = {
        "trainer_state": trainer_state,
        "model_config": asdict(config),
        "optimizer_config": optimizer.checkpoint_config() if isinstance(optimizer, CMUD) else None,
        "training_args": training_args,
        "stream_state": stream_state,
        "mlx_random_state": [value.tolist() for value in mx.random.state],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def migrate_two_group_optimizer_state(
    optimizer_state: dict, expected: dict, optimizer: optim.Optimizer
) -> dict:
    """Split a two-group CMUD optimizer checkpoint across the current three.

    CMUD used to be [MUD, C-Lion]; it is now [MUD, C-Lion on the token-indexed
    tables, C-Lion on everything else]. Only the numbering moved -- MUD is still
    ``states.0``, and every parameter the old ``states.1`` held is in exactly one
    of the new ``states.1``/``states.2`` -- so the state transfers by rewriting
    the prefix rather than being discarded.
    """
    if not isinstance(optimizer, CMUD) or len(optimizer.optimizers) != 3:
        return optimizer_state
    saved_groups = {key.split(".")[1] for key in optimizer_state if key.startswith("states.")}
    if saved_groups != {"0", "1"}:
        return optimizer_state

    migrated = {}
    for key, value in optimizer_state.items():
        if not key.startswith("states.1."):
            migrated[key] = value
            continue
        name = key[len("states.1."):]
        if "." not in name:
            # Per-optimizer scalars (step, learning_rate) belong to the group,
            # not to any parameter, so both new groups need their own copy
            # rather than one of them inheriting the old group's.
            migrated[f"states.1.{name}"] = value
            migrated[f"states.2.{name}"] = value
            continue
        # Drop the trailing slot (exp_avg, master_parameter, ...) to recover the
        # parameter path the group predicate is defined over.
        path = name.rsplit(".", 1)[0]
        group = 1 if CMUD._is_embedding_parameter(path, None) else 2
        migrated[f"states.{group}.{name}"] = value
    if migrated.keys() == expected.keys():
        print("migrated a two-group optimizer checkpoint to the three-group split", flush=True)
        return migrated
    return optimizer_state


def load_checkpoint(path: Path, model: MLXBitNet, optimizer: optim.Optimizer) -> dict:
    parameters = dict(tree_flatten(model.parameters()))
    expected = {key for key in parameters if not key.endswith(_EXCLUDED_STATE_NAMES)}
    loaded = {
        key: value
        for key, value in mx.load(str(path)).items()
        if not key.endswith(_EXCLUDED_STATE_NAMES)
    }
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
    optimizer_state = migrate_two_group_optimizer_state(
        optimizer_state, expected_optimizer, optimizer
    )
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
    if metadata.get("mlx_random_state") is not None:
        mx.random.state[:] = [mx.array(value, dtype=mx.uint32) for value in metadata["mlx_random_state"]]
    return metadata["trainer_state"]


def scheduled_value(start: float, end: float, progress: float, ratio: float) -> float:
    if ratio <= 0:
        return end
    fraction = min(max(progress / ratio, 0.0), 1.0)
    return start + fraction * (end - start)


def lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
    minimum: float,
    schedule: str = "cosine",
) -> float:
    """Warmup, then either a cosine decay or a WSD trapezoid.

    ``wsd`` holds the peak rate flat and then decays linearly to ``minimum``
    over the last ``cooldown_steps``. modded-nanogpt converged on this over
    cosine: at a fixed step budget the plateau spends far more of the run at
    full rate, and the linear tail lands the model rather than coasting through
    a long low-rate stretch that buys little. It wants a long decay (~30-45% of
    the run), not the short cosine cooldown -- ``validate_args`` enforces that.

    The tail stops at ``minimum`` rather than zero: their record 19 was "lr
    decay to 0.1 instead of 0.0" and record 72 raised the floor again, so the
    last steps are meant to still be learning.
    """
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    main_steps = max(total_steps - warmup_steps - cooldown_steps, 1)
    if schedule == "wsd":
        if step < warmup_steps + main_steps:
            return 1.0
        decay = min((step - warmup_steps - main_steps) / max(cooldown_steps, 1), 1.0)
        return 1.0 * (1.0 - decay) + minimum * decay
    if step < warmup_steps + main_steps:
        progress = (step - warmup_steps) / main_steps
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    cooldown = (step - warmup_steps - main_steps) / max(cooldown_steps, 1)
    return max(minimum * (1.0 - cooldown), 0.0)


def build_validation_batches(tokenizer, args) -> list[tuple[mx.array, mx.array, mx.array, mx.array]]:
    if args.validation_batches <= 0:
        return []
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
    return [convert_batch(next(stream)) for _ in range(args.validation_batches)]


def evaluate(model: MLXBitNet, batches) -> dict[str, float]:
    if not batches:
        return {}
    model.eval()
    losses = []
    mtp_losses = [[] for _ in range(model.config.mtp_depth)]
    mtp_correct = [0.0] * model.config.mtp_depth
    mtp_agreement = [0.0] * model.config.mtp_depth
    mtp_counts = [0.0] * model.config.mtp_depth
    for inputs, targets, segments, label_segments in batches:
        output = model(inputs, segments, return_mtp=model.config.mtp_depth > 0)
        logits, depth_logits = output if model.config.mtp_depth > 0 else (output, [])
        loss = _masked_ce(logits, targets, segments == label_segments)
        depth_metrics = []
        for index, values in enumerate(depth_logits):
            shift = index + 1
            valid = segments[:, :-shift] == label_segments[:, shift:]
            predictions = mx.argmax(values[:, :-shift], axis=-1)
            main_predictions = mx.argmax(logits[:, shift:], axis=-1)
            depth_metrics.append(
                (
                    _masked_ce(values[:, :-shift], targets[:, shift:], valid),
                    mx.sum((predictions == targets[:, shift:]) * valid),
                    mx.sum((predictions == main_predictions) * valid),
                    mx.sum(valid),
                )
            )
        mx.eval(loss, depth_metrics)
        losses.append(float(loss.item()))
        for index, (depth_loss, correct, agreement, count) in enumerate(depth_metrics):
            mtp_losses[index].append(float(depth_loss.item()))
            mtp_correct[index] += float(correct.item())
            mtp_agreement[index] += float(agreement.item())
            mtp_counts[index] += float(count.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    metrics = {"val_loss": mean_loss, "val_perplexity": math.exp(min(mean_loss, 20.0))}
    for index, values in enumerate(mtp_losses):
        depth = index + 2
        count = max(mtp_counts[index], 1.0)
        metrics[f"mtp_loss_depth_{depth}"] = sum(values) / len(values)
        metrics[f"mtp_accuracy_depth_{depth}"] = mtp_correct[index] / count
        metrics[f"mtp_agreement_depth_{depth}"] = mtp_agreement[index] / count
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    saved = None
    if args.resume_from:
        saved = json.loads(Path(args.resume_from).with_suffix(".json").read_text(encoding="utf-8"))
        protected = {"output_dir", "resume_from", "compile", "path_kernel", "precision", "profile_phases"}
        saved_args = saved.get("training_args") or {}
        for key, value in saved_args.items():
            if key not in protected and hasattr(args, key):
                setattr(args, key, value)
        if "recurrent_quantized_matmul" not in saved_args:
            args.recurrent_quantized_matmul = False
        if saved_args.get("gradient_checkpointing") and "gradient_checkpoint_scope" not in saved_args:
            args.gradient_checkpoint_scope = "all"
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)

    if args.resume_from:
        config = config_from_saved(saved["model_config"])
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
            infini_memory_expand=args.infini_memory_expand,
            infini_feature_map=args.infini_feature_map,
            use_topk_blocks=args.topk_blocks_branch,
            topk_blocks=args.topk_blocks,
            topk_block_size=args.topk_block_size,
            use_path_kernel=args.path_kernel,
            use_engram=args.engram,
            engram_layer_ids=tuple(int(value) for value in args.engram_layer_ids.split(",") if value),
            engram_vocab_size=args.engram_vocab_size,
            engram_param_fraction=args.engram_param_fraction,
            use_rfmoe=args.use_rfmoe,
            rfmoe_num_experts=args.rfmoe_num_experts,
            rfmoe_expert_dim=args.rfmoe_expert_dim,
            rfmoe_rank=args.rfmoe_rank,
            rfmoe_theta=args.rfmoe_theta,
            rfmoe_backend=args.rfmoe_backend,
            mtp_depth=args.mtp_depth,
            attn_res_mode=args.attn_res_mode,
            attn_res_group_size=args.attn_res_group_size,
            tie_word_embeddings=args.tie_word_embeddings,
        )
    if args.compile and config.use_rfmoe and config.rfmoe_backend != "metal":
        print(
            "RFMoE uses dynamic compaction; compiling optimizer updates but not gradients.",
            flush=True,
        )
    model = MLXBitNet(config)
    dtype = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}[args.precision]
    model.set_dtype(dtype)
    if args.resume_from and saved.get("optimizer_config"):
        optimizer = CMUD(**saved["optimizer_config"])
        args.mud_block_size = optimizer.optimizers[0].block_size
        args.cmud_master_dtype = optimizer.optimizers[0].master_dtype
    else:
        optimizer = CMUD(
            mud_learning_rate=args.mud_learning_rate,
            fallback_learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.mud_momentum,
            passes=args.mud_passes,
            block_size=args.mud_block_size,
            betas=(args.lion_beta1, args.lion_beta2),
            eight_bit=not args.no_optimizer_8bit,
            mud_eight_bit=args.cmud_momentum_8bit,
            mud_master_dtype=args.cmud_master_dtype,
            embedding_learning_rate=args.embedding_learning_rate,
            cautious_weight_decay=args.cautious_weight_decay,
            neuron_norm=args.mud_neuron_norm,
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
    if args.resume_from:
        stream_state = saved.get("stream_state")
        if stream_state is None:
            print("Warning: checkpoint has no dataset stream state; training data restarts.", flush=True)
        else:
            early_stream.load_state_dict(stream_state["early"])
            if late_stream is not None and stream_state.get("late") is not None:
                late_stream.load_state_dict(stream_state["late"])

    def current_stream_state() -> dict:
        return {
            "early": early_stream.state_dict(),
            "late": late_stream.state_dict() if late_stream is not None else None,
        }

    tokens_per_microbatch = args.micro_batch_size * args.sequence_length
    tokens_per_step = tokens_per_microbatch * args.grad_accumulation_steps
    total_steps = math.ceil(args.total_tokens / tokens_per_step)
    warmup_steps = args.warmup_steps or math.ceil(total_steps * args.warmup_ratio)
    cooldown_steps = args.cooldown_steps or math.ceil(total_steps * args.cooldown_ratio)
    parameters = sum(value.size for _, value in tree_flatten(model.parameters()))
    print(f"Device: {mx.device_info()['device_name']}")
    print(f"Model parameters: {parameters / 1e6:.2f}M")
    print(f"Effective depth: {config.effective_depth}")
    window_start, window_end = (
        effective_path_window(
            path_window_size=config.path_window_size,
            block_size=blocks,
            sequence_length=args.sequence_length,
        )
        for blocks in (args.initial_blocks, args.final_blocks)
    )
    print(
        f"PaTH local window: {window_start} -> {window_end} tokens "
        f"(path_window_size={config.path_window_size}, blocks {args.initial_blocks} -> {args.final_blocks})"
    )
    if config.path_window_size > max(window_start, window_end):
        print(
            f"Warning: path_window_size={config.path_window_size} never binds at "
            f"sequence-length={args.sequence_length}; block_size caps the window at "
            f"{max(window_start, window_end)}. Lower --final-blocks to widen it.",
            flush=True,
        )
    if config.use_topk_blocks:
        print(
            f"Top-k retrieval branch: on ({config.topk_blocks} blocks of "
            f"{config.topk_block_size} tokens). mx.compile is disabled for this run, "
            "so compare wall-clock only against another top-k run.",
            flush=True,
        )
    print(f"Early training mixture: {early_spec}")
    if late_stream is not None:
        print(f"Late training mixture: {args.late_train_mixture}")

    validation_batches = build_validation_batches(tokenizer, args) if args.eval_interval > 0 else []
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    gradient_steps = {}
    checkpoint_scope = args.gradient_checkpoint_scope if args.gradient_checkpointing else False
    profile_totals = {"data": 0.0, "forward_backward": 0.0, "mud": 0.0, "sync_wait": 0.0}
    profile_steps = 0
    apply_step, state = create_apply_step(
        model,
        optimizer,
        grad_clip=args.grad_clip,
        compile_step=args.compile,
    )
    for step in range(trainer_state["step"] + 1, total_steps + 1):
        progress = trainer_state["tokens_processed"] / max(args.total_tokens, 1)
        loop_fraction = min(max(
            (progress - args.loop_curriculum_start_ratio)
            / max(args.loop_curriculum_ratio - args.loop_curriculum_start_ratio, 1e-8),
            0.0,
        ), 1.0)
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
        model.recurrent_quantized_matmul = (
            args.recurrent_quantized_matmul and args.sequence_length >= 128 and weight_mix >= 1.0
        )
        rf_fraction = min(progress / max(args.rfmoe_curriculum_ratio, 1e-8), 1.0)
        if args.rfmoe_curriculum_ratio <= 0:
            rf_fraction = 1.0
        rf_s = rf_fraction * args.rfmoe_zipf_s
        rf_alpha = 1.0 + rf_fraction * (args.rfmoe_uniform_alpha - 1.0)
        key = (active_loops, active_blocks, model.recurrent_quantized_matmul)
        if key not in gradient_steps:
            compile_gradient = _gradient_compile_safe(
                config,
                args.compile,
                args.sequence_length,
                active_blocks,
            )
            gradient_step = create_gradient_step(
                model,
                compile_step=compile_gradient,
                num_loops=active_loops,
                z_loss_coef=args.z_loss_coef,
                mtp_loss_coef=args.mtp_loss_coef,
                locality_coef=args.rfmoe_locality_coef,
                diversity_coef=args.rfmoe_diversity_coef,
                gradient_checkpointing=checkpoint_scope,
            )
            gradient_steps[key] = (gradient_step, compile_gradient)
        gradient_step, gradient_is_compiled = gradient_steps[key]
        active_stream = late_stream if late_stream is not None and progress >= args.mixture_switch_ratio else early_stream
        step_started = time.perf_counter()
        accumulated_gradients = None
        losses = []
        hard_densities = []
        for microbatch_index in range(args.grad_accumulation_steps):
            if args.profile_phases:
                phase_started = time.perf_counter()
            batch = convert_batch(next(active_stream))
            if args.profile_phases:
                profile_totals["data"] += time.perf_counter() - phase_started
            gradient_args = [
                *batch,
                mx.array(trainer_state["density_lambda"]),
                mx.array(rf_s),
                mx.array(rf_alpha),
            ]
            if config.mtp_depth > 0:
                index = mtp_head_index(step, microbatch_index, args.grad_accumulation_steps, config.mtp_depth)
                gradient_args.extend(prepare_mtp_batch(batch[1], batch[2], batch[3], index, config.mtp_depth))
            if args.profile_phases:
                phase_started = time.perf_counter()
            loss, gradients = gradient_step(*gradient_args)
            hard_density = model.rfmoe_aux_losses(rf_s, rf_alpha)[3]
            if args.profile_phases:
                sync_started = time.perf_counter()
            try:
                mx.eval(loss, gradients, hard_density, model.state)
            except RuntimeError as error:
                if not gradient_is_compiled or "exhausted the available argument buffers" not in str(error):
                    raise
                print(
                    f"MLX compiler limit at loops={active_loops}, blocks={active_blocks}; "
                    "retrying gradients eagerly.",
                    flush=True,
                )
                gradient_step = create_gradient_step(
                    model,
                    compile_step=False,
                    num_loops=active_loops,
                    z_loss_coef=args.z_loss_coef,
                    mtp_loss_coef=args.mtp_loss_coef,
                    locality_coef=args.rfmoe_locality_coef,
                    diversity_coef=args.rfmoe_diversity_coef,
                    gradient_checkpointing=checkpoint_scope,
                )
                gradient_steps[key] = (gradient_step, False)
                gradient_is_compiled = False
                loss, gradients = gradient_step(*gradient_args)
                hard_density = model.rfmoe_aux_losses(rf_s, rf_alpha)[3]
                mx.eval(loss, gradients, hard_density, model.state)
            if args.profile_phases:
                profile_totals["forward_backward"] += time.perf_counter() - phase_started
                profile_totals["sync_wait"] += time.perf_counter() - sync_started
            losses.append(float(loss.item()))
            hard_densities.append(float(hard_density.item()))
            accumulated_gradients = gradients if accumulated_gradients is None else tree_map(
                lambda total, current: total + current,
                accumulated_gradients,
                gradients,
            )
        if args.profile_phases:
            phase_started = time.perf_counter()
        accumulated_gradients = tree_map(
            lambda gradient: gradient / args.grad_accumulation_steps,
            accumulated_gradients,
        )
        multiplier = lr_multiplier(
            step - 1, total_steps, warmup_steps, cooldown_steps, args.min_lr_ratio, args.lr_schedule
        )
        grad_norm = apply_step(accumulated_gradients, mx.array(multiplier))
        if args.profile_phases:
            sync_started = time.perf_counter()
        mx.eval(model.state, optimizer.state, grad_norm)
        if args.profile_phases:
            profile_totals["mud"] += time.perf_counter() - phase_started
            profile_totals["sync_wait"] += time.perf_counter() - sync_started
            profile_steps += 1
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
            if args.profile_phases:
                metrics.update(
                    {
                        f"profile_{phase}_seconds": value / profile_steps
                        for phase, value in profile_totals.items()
                    }
                )
                profile_totals = dict.fromkeys(profile_totals, 0.0)
                profile_steps = 0
            if config.use_rfmoe:
                metrics["rfmoe_density"] = sum(hard_densities) / len(hard_densities)
                metrics["rfmoe_lambda"] = trainer_state["density_lambda"]
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(" | ".join(f"{key}={value}" for key, value in metrics.items()))
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            if args.profile_phases:
                validation_started = time.perf_counter()
            validation = evaluate(model, validation_batches)
            if args.profile_phases and validation:
                validation["profile_validation_seconds"] = time.perf_counter() - validation_started
            if validation:
                validation_row = {
                    **validation,
                    "step": step,
                    "tokens_processed": trainer_state["tokens_processed"],
                    "time": time.time(),
                    "active_loops": active_loops,
                    "active_blocks": active_blocks,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation_row, sort_keys=True) + "\n")
            print(" | ".join(f"{key}={value}" for key, value in validation.items()))
            if validation and validation["val_loss"] < trainer_state["best_val_loss"]:
                trainer_state["best_val_loss"] = validation["val_loss"]
                save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    config,
                    trainer_state,
                    "best",
                    vars(args),
                    current_stream_state(),
                )
        if args.save_interval > 0 and step % args.save_interval == 0:
            stream_state = current_stream_state()
            path = save_checkpoint(
                output_dir,
                model,
                optimizer,
                config,
                trainer_state,
                f"step_{step:07d}",
                vars(args),
                stream_state,
            )
            save_checkpoint(
                output_dir,
                model,
                optimizer,
                config,
                trainer_state,
                "last",
                vars(args),
                stream_state,
            )
            print(f"Saved checkpoint to {path}")

    final_path = save_checkpoint(
        output_dir,
        model,
        optimizer,
        config,
        trainer_state,
        "final",
        vars(args),
        current_stream_state(),
    )
    wall_time = time.perf_counter() - started
    print(f"Training complete in {wall_time:.1f}s. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
