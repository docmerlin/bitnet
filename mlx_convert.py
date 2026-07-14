"""Convert a PyTorch BitNet checkpoint into a resumable MLX checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import fields
import math
from pathlib import Path
import re

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import torch

from config import TernaryConfig
from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_optim import CMUD
from mlx_train import save_checkpoint
from training.schedules import lr_schedule_multiplier
from utils import load_checkpoint_payload


def mlx_config_from_pytorch(values: dict) -> MLXBitNetConfig:
    valid = {field.name for field in fields(TernaryConfig)}
    source = TernaryConfig(**{key: value for key, value in values.items() if key in valid})
    if source.head_dim != source.hidden_size // source.num_attention_heads:
        raise ValueError("PyTorch head_dim must equal hidden_size / num_attention_heads")
    return MLXBitNetConfig(
        vocab_size=source.vocab_size,
        hidden_size=source.hidden_size,
        num_attention_heads=source.num_attention_heads,
        intermediate_size=source.intermediate_size,
        num_prelude_layers=source.num_prelude_layers,
        num_recurrent_layers=source.num_recurrent_layers,
        num_coda_layers=source.num_coda_layers,
        num_loops=source.num_loops,
        block_size=source.block_size,
        path_window_size=source.path_window_size,
        infini_memory_dim=source.infini_memory_dim,
        use_4bit_activations=source.use_4bit_activations,
        use_hadamard=source.use_hadamard,
        rms_norm_eps=source.rms_norm_eps,
        use_engram=source.use_engram,
        engram_layer_ids=source.engram_layer_ids,
        engram_vocab_size=source.engram_vocab_size,
        engram_max_ngram_size=source.engram_max_ngram_size,
        engram_num_heads=source.engram_num_heads,
        engram_head_dim=source.engram_head_dim,
        engram_kernel_size=source.engram_kernel_size,
        engram_pad_id=source.engram_pad_id,
        engram_seed=source.engram_seed,
        use_rfmoe=source.use_rfmoe,
        rfmoe_num_experts=source.rfmoe_num_experts,
        rfmoe_expert_dim=source.rfmoe_expert_dim,
        rfmoe_rank=source.rfmoe_rank,
        rfmoe_theta=source.rfmoe_theta,
        mtp_depth=source.mtp_depth,
    )


def map_pytorch_key(key: str) -> tuple[str | None, bool]:
    top_level = {
        "embed_tokens.weight": "embedding.weight",
        "subln.weight": "subln.weight",
        "norm.weight": "norm.weight",
    }
    if key in top_level:
        return top_level[key], False
    if key == "lm_head.weight":
        return None, False
    if key.startswith("loop_hc."):
        return key, False

    mtp = re.fullmatch(r"mtp_transforms\.(\d+)\.(\d+)\.weight", key)
    if mtp:
        return f"mtp_transforms.{mtp.group(1)}.layers.{mtp.group(2)}.weight", False

    layer = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if not layer:
        raise ValueError(f"Unsupported PyTorch model key: {key}")
    prefix = f"blocks.{layer.group(1)}."
    suffix = layer.group(2)
    direct = {
        "gate": "attn_gate",
        "attn_norm.weight": "attn_norm.weight",
        "mlp_norm.weight": "mlp_norm.weight",
        "attn_res.scale": "attn_scale",
        "attn_res.norm.weight": "attn_post.weight",
        "mlp_res.scale": "mlp_scale",
        "mlp_res.norm.weight": "mlp_post.weight",
        "ffn_up.weight": "up.weight",
        "ffn_mid.weight": "mid.weight",
        "ffn_down.weight": "down.weight",
    }
    if suffix in direct:
        return prefix + direct[suffix], False
    if suffix.startswith("moe."):
        return prefix + suffix, False
    if suffix.startswith("engram."):
        engram_suffix = suffix.removeprefix("engram.")
        if engram_suffix == "short_conv.weight":
            return prefix + "engram.short_conv_weight", True
        return prefix + "engram." + engram_suffix, False
    if suffix.startswith("infini_attn."):
        attention_suffix = suffix.removeprefix("infini_attn.")
        attention_names = {
            "o_proj.weight": "out.weight",
            "path_w_down.weight": "path_down.weight",
            "path_w_up.weight": "path_up.weight",
            "gate": "memory_gate",
        }
        return prefix + "attn." + attention_names.get(attention_suffix, attention_suffix), False
    raise ValueError(f"Unsupported PyTorch model key: {key}")


def _to_mlx(tensor: torch.Tensor, *, squeeze: bool = False) -> mx.array:
    value = tensor.detach().cpu()
    if squeeze:
        value = value.squeeze(1)
    if value.is_floating_point():
        value = value.float()
    return mx.array(value.numpy())


def load_pytorch_weights(model: MLXBitNet, state: dict[str, torch.Tensor]) -> dict[str, str]:
    if "lm_head.weight" in state and not torch.equal(state["lm_head.weight"], state["embed_tokens.weight"]):
        raise ValueError("PyTorch checkpoint has untied lm_head and embedding weights")
    targets = dict(tree_flatten(model.parameters()))
    converted = []
    name_map = {}
    for source_name, tensor in state.items():
        target_name, squeeze = map_pytorch_key(source_name)
        if target_name is None:
            continue
        if target_name not in targets:
            raise ValueError(f"PyTorch key {source_name!r} maps to missing MLX key {target_name!r}")
        value = _to_mlx(tensor, squeeze=squeeze)
        if value.shape != targets[target_name].shape:
            raise ValueError(
                f"Shape mismatch for {source_name}: PyTorch {value.shape}, MLX {targets[target_name].shape}"
            )
        converted.append((target_name, value))
        name_map[source_name] = target_name

    required = {key for key, _ in tree_flatten(model.trainable_parameters())}
    missing = sorted(required - {key for key, _ in converted})
    if missing:
        raise ValueError(f"PyTorch checkpoint is missing MLX trainable parameters: {missing}")
    model.load_weights(converted, strict=False)
    mx.eval(model.parameters())
    return name_map


def _pytorch_parameter_groups(
    model_state: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    buffers = (".offsets", ".multipliers")
    embeddings = ("embed_tokens.weight", ".engram.embedding.weight", "loop_hc.loop_embed.weight")
    names = [
        name
        for name in model_state
        if name != "lm_head.weight" and not name.endswith(buffers)
    ]
    mud = [
        name
        for name in names
        if model_state[name].ndim == 2 and not any(name == item or name.endswith(item) for item in embeddings)
    ]
    fallback = [name for name in names if name not in mud]
    return mud, fallback


def _optimizer_from_pytorch(
    payload: dict,
    model: MLXBitNet,
    model_state: dict[str, torch.Tensor],
    name_map: dict[str, str],
    *,
    allow_legacy_optimizer_order: bool,
) -> CMUD:
    source = payload.get("optimizer")
    if not source:
        raise ValueError("PyTorch checkpoint has no optimizer state")
    source_groups = source["param_groups"]
    groups = {group.get("kind"): group for group in source_groups}
    if len(source_groups) != 2 or set(groups) != {"mud", "clion"}:
        raise ValueError("Only C-MUD/C-Lion PyTorch optimizer checkpoints can be warm-converted")
    mud_group, lion_group = groups["mud"], groups["clion"]
    if not mud_group.get("nesterov", True):
        raise ValueError("MLX conversion requires Nesterov C-MUD")
    if lion_group.get("weight_decay", 0.0) != 0.0:
        raise ValueError("MLX conversion requires zero C-Lion weight decay")
    optimizer = CMUD(
        mud_learning_rate=mud_group.get("initial_lr", mud_group["lr"]),
        fallback_learning_rate=lion_group.get("initial_lr", lion_group["lr"]),
        weight_decay=mud_group["weight_decay"],
        momentum=mud_group["momentum"],
        passes=mud_group["passes"],
        betas=tuple(lion_group["betas"]),
        eight_bit=lion_group["eight_bit"],
    )
    optimizer.init(model.trainable_parameters())
    flat_state = dict(tree_flatten(optimizer.state))
    mud_names, fallback_names = _pytorch_parameter_groups(model_state)
    if not all("param_names" in group for group in groups.values()) and not allow_legacy_optimizer_order:
        raise ValueError(
            "Legacy optimizer checkpoint has no parameter names; rerun with "
            "--allow-legacy-optimizer-order only for an unmodified checkpoint produced by this repository"
        )
    group_names = {
        "mud": mud_group.get("param_names", mud_names),
        "clion": lion_group.get("param_names", fallback_names),
    }

    for kind, group_index in (("mud", 0), ("clion", 1)):
        group = groups[kind]
        names = group_names[kind]
        expected_names = mud_names if kind == "mud" else fallback_names
        if len(group["params"]) != len(names) or len(names) != len(set(names)) or set(names) != set(expected_names):
            raise ValueError(
                f"PyTorch optimizer {kind} parameter names do not match model parameters"
            )
        for parameter_id, source_name in zip(group["params"], names):
            if source_name not in name_map:
                raise ValueError(f"Optimizer parameter {source_name!r} is absent from converted model")
            source_state = source["state"].get(parameter_id)
            if not source_state:
                continue
            target_name = name_map[source_name]
            prefix = f"states.{group_index}.{target_name}."
            squeeze = source_name.endswith("engram.short_conv.weight")
            if kind == "mud":
                flat_state[prefix + "momentum_buffer"] = _to_mlx(
                    source_state["momentum_buffer"], squeeze=squeeze
                )
                continue
            for state_name in ("exp_avg", "exp_avg_q", "exp_avg_scale"):
                flat_state.pop(prefix + state_name, None)
            if "exp_avg_q" in source_state:
                flat_state[prefix + "exp_avg_q"] = _to_mlx(source_state["exp_avg_q"])
                flat_state[prefix + "exp_avg_scale"] = _to_mlx(source_state["exp_avg_scale"])
            elif "exp_avg" in source_state:
                flat_state[prefix + "exp_avg"] = _to_mlx(source_state["exp_avg"], squeeze=squeeze)

    step = payload.get("trainer_state", {}).get("step", 0)
    flat_state["states.0.step"] = mx.array(step, dtype=mx.uint64)
    flat_state["states.1.step"] = mx.array(step, dtype=mx.uint64)
    optimizer.state = tree_unflatten(list(flat_state.items()))
    return optimizer


def _set_quantization_state(model: MLXBitNet, payload: dict) -> None:
    args = payload.get("args", {})
    trainer_state = payload.get("trainer_state", {})
    progress = trainer_state.get("tokens_processed", 0) / max(args.get("total_tokens", 1), 1)
    ratio = args.get("stage1_ratio", 0.12)
    fraction = 1.0 if ratio <= 0 else min(progress / ratio, 1.0)
    weight_start = args.get("stage1_weight_mix_start", 0.25)
    activation_start = args.get("stage1_activation_mix_start", 0.0)
    stage_bits = args.get("stage1_activation_bits", 8)
    final_bits = args.get("final_activation_bits", 4)
    weight_mix = weight_start + fraction * (1.0 - weight_start)
    activation_mix = activation_start + fraction * (1.0 - activation_start)
    if not payload["model_config"].get("use_4bit_activations", True):
        activation_mix = 0.0
    bits = round(stage_bits - fraction * (stage_bits - final_bits))
    model.set_quantization_state(weight_mix, activation_mix, bits)


def convert_pytorch_checkpoint(
    source_path: Path,
    output_dir: Path,
    *,
    name: str = "imported",
    allow_legacy_optimizer_order: bool = False,
) -> Path:
    payload = load_checkpoint_payload(source_path, map_location="cpu")
    required = {"model", "optimizer", "scheduler", "trainer_state", "model_config", "args"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Incomplete BitNet training checkpoint; missing {sorted(missing)}")
    step = payload["trainer_state"].get("step", 0)
    if payload["scheduler"].get("last_epoch") != step:
        raise ValueError("PyTorch scheduler position does not match trainer step")
    args = payload["args"]
    tokens_per_step = (
        args.get("micro_batch_size", 1)
        * args.get("sequence_length", 1024)
        * args.get("grad_accumulation_steps", 16)
    )
    total_steps = max(1, math.ceil(args.get("total_tokens", 50_000_000) / tokens_per_step))
    warmup_steps = args.get("warmup_steps", 0) or math.ceil(total_steps * args.get("warmup_ratio", 0.08))
    cooldown_steps = args.get("cooldown_steps", 0) or math.ceil(total_steps * args.get("cooldown_ratio", 0.05))
    multiplier = lr_schedule_multiplier(
        step,
        total_steps,
        warmup_steps,
        cooldown_steps,
        args.get("min_lr_ratio", 0.1),
    )
    for group in payload["optimizer"]["param_groups"]:
        base_lr = group.get("initial_lr")
        if base_lr is None or not math.isclose(group["lr"], base_lr * multiplier, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError("PyTorch optimizer LR does not match the supported cosine scheduler")
    config = mlx_config_from_pytorch(payload["model_config"])
    model = MLXBitNet(config)
    name_map = load_pytorch_weights(model, payload["model"])
    _set_quantization_state(model, payload)
    optimizer = _optimizer_from_pytorch(
        payload,
        model,
        payload["model"],
        name_map,
        allow_legacy_optimizer_order=allow_legacy_optimizer_order,
    )
    trainer_state = {
        "step": 0,
        "tokens_processed": 0,
        "best_val_loss": float("inf"),
        "density_lambda": 1e-3,
        **payload.get("trainer_state", {}),
    }
    training_args = dict(payload.get("args", {}))
    training_args.pop("optimizer", None)
    return save_checkpoint(output_dir, model, optimizer, config, trainer_state, name, training_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="imported")
    parser.add_argument("--allow-legacy-optimizer-order", action="store_true")
    args = parser.parse_args()
    path = convert_pytorch_checkpoint(
        args.checkpoint,
        args.output_dir,
        name=args.name,
        allow_legacy_optimizer_order=args.allow_legacy_optimizer_order,
    )
    print(f"Converted checkpoint: {path}")


if __name__ == "__main__":
    main()
