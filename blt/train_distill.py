"""Online distillation trainer for the separate BLT package."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F

from blt.config import TernaryBLTConfig
from blt.data import (
    BatchByteStream,
    ByteBatch,
    ByteVocabulary,
    PackedByteSequenceStream,
    PrefetchStream,
    iter_hf_dataset,
    iter_text_file,
    iter_texts,
)
from blt.losses import DistillationLossWeights, compute_blt_distillation_loss
from blt.model import TernaryBLTModel, TernaryBLTOutput
from blt.patching.student_entropy import StudentEntropyModel
from blt.patching.teacher_patcher import UniformPatcher, normalize_patch_lengths_to_targets, patch_start_mask_from_lengths
from optim import build_cmud
from utils import load_checkpoint_payload, seed_everything


class TeacherModel(Protocol):
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        patch_lengths: torch.Tensor | None = None,
    ) -> TernaryBLTOutput: ...


@dataclass(slots=True)
class BLTDistillationBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    patch_lengths: torch.Tensor | None = None

    def to(self, device: torch.device) -> "BLTDistillationBatch":
        return BLTDistillationBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=self.attention_mask.to(device),
            patch_lengths=None if self.patch_lengths is None else self.patch_lengths.to(device),
        )

    @classmethod
    def from_byte_batch(cls, batch: ByteBatch) -> "BLTDistillationBatch":
        return cls(
            input_ids=batch.input_ids,
            labels=batch.labels,
            attention_mask=batch.attention_mask,
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    total = (values * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return total / denom


class BLTDistillationTrainer:
    def __init__(
        self,
        student: TernaryBLTModel,
        *,
        optimizer: torch.optim.Optimizer,
        config: TernaryBLTConfig,
        teacher: TeacherModel | None = None,
        weights: DistillationLossWeights | None = None,
        device: torch.device | None = None,
        student_patcher: StudentEntropyModel | None = None,
        patcher_optimizer: torch.optim.Optimizer | None = None,
        patcher_mode: str = "off",
        patcher_loss_weight: float = 1.0,
        patcher_threshold: float = 0.0,
        patcher_warmup_steps: int = 0,
        patch_length_provider: Any | None = None,
        start_step: int = 0,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.student = student
        self.optimizer = optimizer
        self.config = config
        self.teacher = teacher
        self.weights = weights or DistillationLossWeights()
        self.device = device or next(student.parameters()).device
        self.student.to(self.device)
        self.amp_dtype = amp_dtype

        self.student_patcher = student_patcher
        self.patcher_optimizer = patcher_optimizer
        self.patcher_mode = patcher_mode
        self.patcher_loss_weight = patcher_loss_weight
        self.patcher_threshold = patcher_threshold
        self.patcher_warmup_steps = patcher_warmup_steps
        self.patch_length_provider = patch_length_provider
        self.global_step = start_step

        if self.student_patcher is not None:
            self.student_patcher.to(self.device)
        if self.student_patcher is None and self.patcher_mode != "off":
            raise ValueError("patcher_mode requires a student_patcher")
        if self.student_patcher is not None and self.patcher_optimizer is None:
            raise ValueError("student_patcher requires a patcher_optimizer")

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.amp_dtype is None or self.device.type not in {"cuda", "cpu"}:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _provided_patch_lengths(
        self,
        batch: BLTDistillationBatch,
    ) -> torch.Tensor | None:
        valid_lengths = batch.attention_mask.sum(dim=1)
        if batch.patch_lengths is not None:
            return normalize_patch_lengths_to_targets(batch.patch_lengths, valid_lengths)
        if self.patch_length_provider is not None:
            generated = self.patch_length_provider.patch(batch.input_ids)
            return normalize_patch_lengths_to_targets(generated, valid_lengths)
        return None

    def _teacher_patch_lengths(
        self,
        provided_patch_lengths: torch.Tensor | None,
        teacher_outputs: TernaryBLTOutput | None,
    ) -> torch.Tensor | None:
        if teacher_outputs is not None:
            return teacher_outputs.patch_lengths
        return provided_patch_lengths

    def _step_uses_student_patcher(self, step: int) -> bool:
        if self.student_patcher is None:
            return False
        if self.patcher_mode == "student":
            return True
        if self.patcher_mode == "teacher_then_student":
            return step > self.patcher_warmup_steps
        return False

    def _compute_patcher_loss(
        self,
        batch: BLTDistillationBatch,
        target_patch_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
        if self.student_patcher is None or target_patch_lengths is None:
            return None, None, {}

        boundary_logits = self.student_patcher(batch.input_ids)
        target_mask = patch_start_mask_from_lengths(target_patch_lengths, batch.input_ids.size(1)).to(dtype=boundary_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(boundary_logits, target_mask, reduction="none")
        patcher_loss = _masked_mean(bce, batch.attention_mask)
        predicted_patch_lengths = self.student_patcher.predict_patch_lengths_from_logits(
            boundary_logits,
            threshold=self.patcher_threshold,
        )
        metrics = {"student_patcher_bce": float(patcher_loss.detach().item())}
        return patcher_loss, predicted_patch_lengths, metrics

    def _select_patch_lengths(
        self,
        provided_patch_lengths: torch.Tensor | None,
        teacher_patch_lengths: torch.Tensor | None,
        student_patch_lengths: torch.Tensor | None,
        *,
        step: int,
    ) -> tuple[torch.Tensor | None, bool]:
        if self._step_uses_student_patcher(step) and student_patch_lengths is not None:
            return student_patch_lengths, True
        if teacher_patch_lengths is not None:
            return teacher_patch_lengths, False
        return provided_patch_lengths, False

    def _maybe_rerun_teacher_for_selected_patch_lengths(
        self,
        batch: BLTDistillationBatch,
        *,
        teacher_outputs: TernaryBLTOutput | None,
        teacher_patch_lengths: torch.Tensor | None,
        selected_patch_lengths: torch.Tensor | None,
        using_student_patcher: bool,
    ) -> TernaryBLTOutput | None:
        if self.teacher is None or teacher_outputs is None or not using_student_patcher or selected_patch_lengths is None:
            return teacher_outputs
        if teacher_patch_lengths is not None and torch.equal(selected_patch_lengths, teacher_patch_lengths):
            return teacher_outputs
        with torch.no_grad():
            return self.teacher.forward(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                patch_lengths=selected_patch_lengths,
            )

    def _run_forward(
        self, batch: BLTDistillationBatch, *, step: int
    ) -> tuple[TernaryBLTOutput, TernaryBLTOutput | None, torch.Tensor, dict[str, float]]:
        """Run the teacher/student forward shared by training and evaluation.

        Returns ``(student_outputs, teacher_outputs, total_loss, metrics)``. The
        teacher always runs under ``no_grad``; ``eval_step`` additionally wraps the
        whole call so the student forward is gradient-free.
        """
        batch = batch.to(self.device)
        provided_patch_lengths = self._provided_patch_lengths(batch)
        teacher_outputs = None

        with self._autocast():
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher.forward(
                        batch.input_ids,
                        attention_mask=batch.attention_mask,
                        patch_lengths=provided_patch_lengths,
                    )

            teacher_patch_lengths = self._teacher_patch_lengths(provided_patch_lengths, teacher_outputs)
            patcher_loss, predicted_patch_lengths, patcher_metrics = self._compute_patcher_loss(batch, teacher_patch_lengths)
            patch_lengths, using_student_patcher = self._select_patch_lengths(
                provided_patch_lengths,
                teacher_patch_lengths,
                predicted_patch_lengths,
                step=step,
            )
            teacher_outputs = self._maybe_rerun_teacher_for_selected_patch_lengths(
                batch,
                teacher_outputs=teacher_outputs,
                teacher_patch_lengths=teacher_patch_lengths,
                selected_patch_lengths=patch_lengths,
                using_student_patcher=using_student_patcher,
            )

            student_outputs = self.student(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                patch_lengths=patch_lengths,
            )

            model_loss, metrics = compute_blt_distillation_loss(
                student_outputs,
                teacher_outputs,
                labels=batch.labels,
                attention_mask=batch.attention_mask,
                weights=self.weights,
                temperature=self.config.distill_temperature,
            )

            total_loss = model_loss
            metrics["model_loss"] = float(model_loss.detach().item())
            metrics.update(patcher_metrics)
            if patcher_loss is not None:
                total_loss = total_loss + self.patcher_loss_weight * patcher_loss
            metrics["student_patcher_active"] = 1.0 if using_student_patcher else 0.0
            metrics["loss"] = float(total_loss.detach().item())

        return student_outputs, teacher_outputs, total_loss, metrics

    def train_step(self, batch: BLTDistillationBatch) -> tuple[TernaryBLTOutput, TernaryBLTOutput | None, dict[str, float]]:
        self.student.train()
        if self.student_patcher is not None:
            self.student_patcher.train()

        current_step = self.global_step + 1
        student_outputs, teacher_outputs, total_loss, metrics = self._run_forward(batch, step=current_step)

        self.optimizer.zero_grad(set_to_none=True)
        if self.patcher_optimizer is not None:
            self.patcher_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        if self.patcher_optimizer is not None:
            self.patcher_optimizer.step()
        self.global_step = current_step
        return student_outputs, teacher_outputs, metrics

    @torch.no_grad()
    def eval_step(self, batch: BLTDistillationBatch) -> tuple[TernaryBLTOutput, TernaryBLTOutput | None, dict[str, float]]:
        self.student.eval()
        if self.student_patcher is not None:
            self.student_patcher.eval()

        student_outputs, teacher_outputs, _, metrics = self._run_forward(batch, step=self.global_step)
        return student_outputs, teacher_outputs, metrics


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """Resolve the autocast dtype for student/teacher forwards (None = full precision).

    Only bf16 is offered for training: it needs no gradient scaler. autocast is
    enabled on CUDA/CPU; MPS keeps full precision.
    """
    if precision == "fp32" or device.type not in {"cuda", "cpu"}:
        return None
    if precision == "bf16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


RESUME_STICKY_ARG_NAMES = (
    "optimizer",
    "disable_teacher_patcher",
    "student_patcher_mode",
    "student_patcher_dim",
    "student_patcher_layers",
    "student_patcher_heads",
    "student_patcher_threshold",
    "student_patcher_warmup_steps",
    "student_patcher_loss_weight",
    "student_patcher_learning_rate",
    "student_patcher_weight_decay",
)


def apply_checkpoint_training_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    training_args = checkpoint.get("training_args") or {}
    for name in RESUME_STICKY_ARG_NAMES:
        if name in training_args:
            setattr(args, name, training_args[name])
    # Checkpoints written before C-MUD became the default store no "optimizer"
    # key; they were trained with AdamW, so resume with AdamW rather than loading
    # an AdamW state dict into a C-MUD optimizer.
    if "optimizer" not in training_args:
        args.optimizer = "adamw"


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_config_from_args(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any] | None = None,
) -> TernaryBLTConfig:
    if checkpoint_config is not None:
        return TernaryBLTConfig(**checkpoint_config)
    return TernaryBLTConfig(
        local_dim=args.local_dim,
        global_dim=args.global_dim,
        decoder_dim=args.decoder_dim,
        n_layers_local_encoder=args.n_layers_local_encoder,
        n_layers_global=args.n_layers_global,
        n_layers_local_decoder=args.n_layers_local_decoder,
        n_heads_local_encoder=args.n_heads_local_encoder,
        n_heads_global=args.n_heads_global,
        n_heads_local_decoder=args.n_heads_local_decoder,
        n_heads_cross=args.n_heads_cross,
        ffn_multiplier_local=args.ffn_multiplier_local,
        ffn_multiplier_global=args.ffn_multiplier_global,
        ffn_multiplier_decoder=args.ffn_multiplier_decoder,
        local_window=args.local_window,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        patch_size=args.patch_size,
        max_patch_length=args.max_patch_length,
        use_hadamard=not args.disable_hadamard,
        use_4bit_activations=not args.disable_4bit_activations,
        distill_temperature=args.distill_temperature,
    )


def _source_value(args: argparse.Namespace, name: str, *, eval_mode: bool) -> Any:
    if eval_mode:
        eval_name = f"eval_{name}"
        value = getattr(args, eval_name)
        if value is not None:
            return value
        if name == "shuffle_dataset":
            return False
        if name == "shuffle_buffer_size":
            return getattr(args, name)
        if name in {"hf_config", "hf_split", "hf_text_field"}:
            return getattr(args, name)
    return getattr(args, name)


def _has_explicit_eval_source(args: argparse.Namespace) -> bool:
    return (
        args.eval_text is not None
        or args.eval_text_file is not None
        or args.eval_hf_dataset is not None
    )


def build_text_stream(args: argparse.Namespace, *, eval_mode: bool = False):
    text = _source_value(args, "text", eval_mode=eval_mode)
    text_file = _source_value(args, "text_file", eval_mode=eval_mode)
    hf_dataset = _source_value(args, "hf_dataset", eval_mode=eval_mode)

    if text is not None:
        return iter_texts(text, restart_on_eof=True)
    if text_file is not None:
        return iter_text_file(text_file, restart_on_eof=True)
    return iter_hf_dataset(
        hf_dataset,
        config_name=_source_value(args, "hf_config", eval_mode=eval_mode),
        split=_source_value(args, "hf_split", eval_mode=eval_mode),
        text_field=_source_value(args, "hf_text_field", eval_mode=eval_mode),
        restart_on_eof=True,
        shuffle=_source_value(args, "shuffle_dataset", eval_mode=eval_mode),
        shuffle_buffer_size=_source_value(args, "shuffle_buffer_size", eval_mode=eval_mode),
        seed=args.seed + (10000 if eval_mode else 0),
    )


def build_batch_stream(
    args: argparse.Namespace,
    config: TernaryBLTConfig,
    *,
    eval_mode: bool = False,
) -> BatchByteStream:
    if eval_mode and not _has_explicit_eval_source(args):
        print(
            "No explicit eval source provided; reusing the training text/HF stream for evaluation.",
            flush=True,
        )
    vocabulary = ByteVocabulary(config)
    text_stream = build_text_stream(args, eval_mode=eval_mode)
    sequence_stream = PackedByteSequenceStream(
        text_stream,
        vocabulary,
        sequence_length=args.sequence_length,
        max_document_bytes=args.max_document_bytes,
    )
    batch_size = args.eval_batch_size if eval_mode else args.batch_size
    return BatchByteStream(sequence_stream, batch_size)


def build_teacher(
    args: argparse.Namespace,
    *,
    teacher_override: TeacherModel | None = None,
) -> TeacherModel | None:
    if teacher_override is not None:
        return teacher_override
    if args.no_teacher:
        return None

    from blt.teacher.facebook_blt import FacebookBLTTeacher

    return FacebookBLTTeacher.from_pretrained(
        model_id=args.teacher_model_id,
        entropy_model_id=None if args.disable_teacher_patcher else args.teacher_entropy_model_id,
        upstream_repo_path=args.teacher_upstream_repo,
        device=args.teacher_device,
    )


def build_patch_length_provider(args: argparse.Namespace, config: TernaryBLTConfig) -> UniformPatcher | None:
    if args.disable_teacher_patcher:
        return UniformPatcher(config.patch_size)
    return None


def build_student_patcher(
    args: argparse.Namespace,
    config: TernaryBLTConfig,
) -> tuple[StudentEntropyModel | None, torch.optim.Optimizer | None]:
    if args.student_patcher_mode == "off":
        return None, None

    patcher = StudentEntropyModel(
        config,
        dim=args.student_patcher_dim,
        num_layers=args.student_patcher_layers,
        num_heads=args.student_patcher_heads,
    )
    optimizer = torch.optim.AdamW(
        patcher.parameters(),
        lr=args.student_patcher_learning_rate if args.student_patcher_learning_rate is not None else args.learning_rate,
        weight_decay=args.student_patcher_weight_decay if args.student_patcher_weight_decay is not None else args.weight_decay,
    )
    return patcher, optimizer


def checkpoint_path_for_step(save_path: str | Path, step: int, *, final: bool) -> Path:
    path = Path(save_path).expanduser().resolve()
    if final:
        return path
    suffix = path.suffix
    if suffix:
        return path.with_name(f"{path.stem}-step{step}{suffix}")
    return path.with_name(f"{path.name}-step{step}")


def save_checkpoint(
    save_path: str | Path,
    *,
    student: TernaryBLTModel,
    optimizer: torch.optim.Optimizer,
    config: TernaryBLTConfig,
    step: int,
    metrics: dict[str, float],
    eval_metrics: dict[str, float] | None,
    args: argparse.Namespace,
    student_patcher: StudentEntropyModel | None = None,
    patcher_optimizer: torch.optim.Optimizer | None = None,
    final: bool = True,
) -> Path:
    checkpoint_path = checkpoint_path_for_step(save_path, step, final=final)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap torch.compile so checkpoint keys stay free of the _orig_mod prefix.
    model_to_save = getattr(student, "_orig_mod", student)
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "eval_metrics": eval_metrics,
            "training_args": vars(args),
            "student_patcher": None if student_patcher is None else student_patcher.state_dict(),
            "student_patcher_optimizer": None if patcher_optimizer is None else patcher_optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(checkpoint_path: str | Path, *, device: torch.device) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return load_checkpoint_payload(path, map_location=device)


def evaluate(
    trainer: BLTDistillationTrainer,
    eval_stream: BatchByteStream,
    *,
    eval_steps: int,
) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for _ in range(eval_steps):
        batch = BLTDistillationBatch.from_byte_batch(next(eval_stream))
        _, _, metrics = trainer.eval_step(batch)
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value)
    return {key: value / eval_steps for key, value in aggregate.items()}


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    summary = " | ".join(f"{prefix}{key}={value:.6g}" for key, value in sorted(metrics.items()))
    print(summary, flush=True)


def run_distillation(
    args: argparse.Namespace,
    *,
    teacher_override: TeacherModel | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    device = choose_device(args.device)
    teacher_device = choose_device(args.teacher_device)
    args.teacher_device = str(teacher_device)
    amp_dtype = resolve_amp_dtype(device, args.precision)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    checkpoint = None
    if args.resume_from is not None:
        checkpoint = load_checkpoint(args.resume_from, device=device)
        apply_checkpoint_training_args(args, checkpoint)

    config = build_config_from_args(args, checkpoint_config=None if checkpoint is None else checkpoint.get("config"))
    student = TernaryBLTModel(config)
    if args.optimizer == "cmud":
        optimizer = build_cmud(
            student,
            lr=args.mud_learning_rate,
            fallback_lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    student_patcher, patcher_optimizer = build_student_patcher(args, config)
    start_step = 0
    last_metrics: dict[str, float] = {} if checkpoint is None else dict(checkpoint.get("metrics") or {})
    last_eval_metrics: dict[str, float] | None = None if checkpoint is None else dict(checkpoint.get("eval_metrics") or {})

    if checkpoint is not None:
        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(
            "Resume restores optimizer and student-patcher state only. Data streams restart from "
            "the beginning and RNG state is not checkpointed.",
            flush=True,
        )

        saved_patcher_state = checkpoint.get("student_patcher")
        saved_patcher_optimizer_state = checkpoint.get("student_patcher_optimizer")
        if saved_patcher_state is not None:
            if student_patcher is None:
                raise ValueError("checkpoint includes a student patcher; resume with --student-patcher-mode enabled")
            student_patcher.load_state_dict(saved_patcher_state)
            if patcher_optimizer is not None and saved_patcher_optimizer_state is not None:
                patcher_optimizer.load_state_dict(saved_patcher_optimizer_state)

    weights = DistillationLossWeights(
        hard_ce=args.hard_ce_weight,
        logits_kl=args.logits_kl_weight,
        encoder_patch_mse=args.encoder_patch_mse_weight,
        global_patch_mse=args.global_patch_mse_weight,
        decoder_hidden_mse=args.decoder_hidden_mse_weight,
    )
    teacher = build_teacher(args, teacher_override=teacher_override)
    patch_length_provider = build_patch_length_provider(args, config)
    trainer = BLTDistillationTrainer(
        student,
        optimizer=optimizer,
        config=config,
        teacher=teacher,
        weights=weights,
        device=device,
        student_patcher=student_patcher,
        patcher_optimizer=patcher_optimizer,
        patcher_mode=args.student_patcher_mode,
        patcher_loss_weight=args.student_patcher_loss_weight,
        patcher_threshold=args.student_patcher_threshold,
        patcher_warmup_steps=args.student_patcher_warmup_steps,
        patch_length_provider=patch_length_provider,
        start_step=start_step,
        amp_dtype=amp_dtype,
    )

    move_optimizer_to_device(trainer.optimizer, trainer.device)
    if trainer.patcher_optimizer is not None:
        move_optimizer_to_device(trainer.patcher_optimizer, trainer.device)

    # Compile after the optimizer is built (params are shared) and after any resume
    # load (which targets the uncompiled module). CUDA only.
    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        trainer.student = torch.compile(trainer.student)

    batch_stream = build_batch_stream(args, config)
    if args.prefetch_batches > 0:
        batch_stream = PrefetchStream(batch_stream, buffer_size=args.prefetch_batches)
    eval_stream = build_batch_stream(args, config, eval_mode=True) if args.eval_every > 0 else None
    end_step = start_step + args.steps

    for step in range(start_step + 1, end_step + 1):
        batch = BLTDistillationBatch.from_byte_batch(next(batch_stream))
        _, _, metrics = trainer.train_step(batch)
        metrics = {**metrics, "step": float(step)}
        last_metrics = metrics
        if step == start_step + 1 or step % args.log_every == 0 or step == end_step:
            print_metrics("", metrics)

        if eval_stream is not None and (step % args.eval_every == 0 or step == end_step):
            last_eval_metrics = evaluate(trainer, eval_stream, eval_steps=args.eval_steps)
            eval_summary = {**last_eval_metrics, "step": float(step)}
            print_metrics("eval_", eval_summary)

        if args.save_path is not None and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                args.save_path,
                student=trainer.student,
                optimizer=trainer.optimizer,
                config=config,
                step=step,
                metrics=last_metrics,
                eval_metrics=last_eval_metrics,
                args=args,
                student_patcher=trainer.student_patcher,
                patcher_optimizer=trainer.patcher_optimizer,
                final=False,
            )

    checkpoint_path = None
    if args.save_path is not None:
        checkpoint_path = save_checkpoint(
            args.save_path,
            student=trainer.student,
            optimizer=trainer.optimizer,
            config=config,
            step=trainer.global_step,
            metrics=last_metrics,
            eval_metrics=last_eval_metrics,
            args=args,
            student_patcher=trainer.student_patcher,
            patcher_optimizer=trainer.patcher_optimizer,
            final=True,
        )

    result = {
        "device": str(device),
        "teacher_enabled": teacher is not None,
        "student_patcher_enabled": trainer.student_patcher is not None,
        "start_step": start_step,
        "steps_run": args.steps,
        "final_step": trainer.global_step,
        "metrics": last_metrics,
    }
    if last_eval_metrics is not None:
        result["eval_metrics"] = last_eval_metrics
    if checkpoint_path is not None:
        result["checkpoint_path"] = str(checkpoint_path)
    if args.resume_from is not None:
        result["resumed_from"] = str(Path(args.resume_from).expanduser().resolve())
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the separate ternary BLT student with optional teacher distillation")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", action="append", help="Inline text example. Repeat for multiple training strings.")
    source_group.add_argument("--text-file", help="Path to a UTF-8 text file. Non-empty lines are treated as documents.")
    source_group.add_argument("--hf-dataset", help="Hugging Face dataset path for streaming text documents.")

    eval_group = parser.add_mutually_exclusive_group(required=False)
    eval_group.add_argument("--eval-text", action="append", help="Optional inline eval text. Repeat for multiple strings.")
    eval_group.add_argument("--eval-text-file", help="Optional UTF-8 text file for evaluation.")
    eval_group.add_argument("--eval-hf-dataset", help="Optional Hugging Face dataset path for evaluation.")

    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--hf-split", default="train", help="Hugging Face dataset split.")
    parser.add_argument("--hf-text-field", default="text", help="Field name containing text in the dataset examples.")
    parser.add_argument("--shuffle-dataset", action="store_true", help="Shuffle the Hugging Face dataset stream.")
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000, help="Shuffle buffer size for HF streaming datasets.")

    parser.add_argument("--eval-hf-config", default=None, help="Optional eval Hugging Face dataset config name.")
    parser.add_argument("--eval-hf-split", default=None, help="Optional eval Hugging Face dataset split.")
    parser.add_argument("--eval-hf-text-field", default=None, help="Optional eval dataset text field name.")
    parser.add_argument("--eval-shuffle-dataset", action="store_true", help="Shuffle the eval Hugging Face dataset stream.")

    parser.add_argument("--steps", type=int, default=10, help="Number of optimization steps to run in this invocation.")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Eval micro-batch size.")
    parser.add_argument("--sequence-length", type=int, default=128, help="Packed byte sequence length for autoregressive training.")
    parser.add_argument("--max-document-bytes", type=int, default=2048, help="Maximum encoded byte length per document before packing.")
    parser.add_argument("--optimizer", choices=("cmud", "adamw"), default="cmud",
                        help="Student optimizer: C-MUD (default) or legacy AdamW.")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="LR for AdamW and for the C-Lion fallback group of C-MUD.")
    parser.add_argument("--mud-learning-rate", type=float, default=1e-3,
                        help="LR for the C-MUD matrix (MUD) group.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument("--log-every", type=int, default=1, help="Print metrics every N steps.")
    parser.add_argument("--eval-every", type=int, default=0, help="Run eval every N train steps. Set to 0 to disable.")
    parser.add_argument("--eval-steps", type=int, default=1, help="Number of eval batches per evaluation run.")
    parser.add_argument("--save-path", default=None, help="Optional checkpoint output path.")
    parser.add_argument("--save-every", type=int, default=0, help="Write periodic step checkpoints every N steps.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--device", default="auto", help="Student training device.")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto",
                        help="Mixed-precision mode for student/teacher forwards (bf16 autocast; default auto).")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile the student with torch.compile on CUDA (default: on; --no-compile to disable).")
    parser.add_argument("--prefetch-batches", type=int, default=2,
                        help="Background-thread batch prefetch depth to overlap tokenization with compute (0 disables).")
    parser.add_argument("--teacher-device", default="auto", help="Teacher execution device.")
    parser.add_argument("--no-teacher", action="store_true", help="Disable distillation and run hard-CE-only training.")
    parser.add_argument("--teacher-model-id", default="facebook/blt-1b", help="Teacher BLT model ID.")
    parser.add_argument("--teacher-entropy-model-id", default="facebook/blt-entropy", help="Teacher entropy model ID.")
    parser.add_argument("--teacher-upstream-repo", default=None, help="Path to a local facebookresearch/blt checkout.")
    parser.add_argument("--disable-teacher-patcher", action="store_true", help="Disable teacher entropy patching and fall back to static patch lengths unless the batch already provides them.")

    parser.add_argument(
        "--student-patcher-mode",
        choices=["off", "distill_only", "teacher_then_student", "student"],
        default="off",
        help=(
            "Student patcher rollout mode. "
            "off=disabled; "
            "distill_only=train the patcher on teacher boundaries but keep teacher patches for forward; "
            "teacher_then_student=use teacher patches until warmup, then student patches; "
            "student=always use student patches."
        ),
    )
    parser.add_argument("--student-patcher-dim", type=int, default=128)
    parser.add_argument("--student-patcher-layers", type=int, default=2)
    parser.add_argument("--student-patcher-heads", type=int, default=4)
    parser.add_argument("--student-patcher-threshold", type=float, default=0.0)
    parser.add_argument("--student-patcher-warmup-steps", type=int, default=0)
    parser.add_argument("--student-patcher-loss-weight", type=float, default=1.0)
    parser.add_argument("--student-patcher-learning-rate", type=float, default=None)
    parser.add_argument("--student-patcher-weight-decay", type=float, default=None)

    parser.add_argument("--local-dim", type=int, default=256)
    parser.add_argument("--global-dim", type=int, default=512)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--n-layers-local-encoder", type=int, default=4)
    parser.add_argument("--n-layers-global", type=int, default=8)
    parser.add_argument("--n-layers-local-decoder", type=int, default=4)
    parser.add_argument("--n-heads-local-encoder", type=int, default=4)
    parser.add_argument("--n-heads-global", type=int, default=8)
    parser.add_argument("--n-heads-local-decoder", type=int, default=4)
    parser.add_argument("--n-heads-cross", type=int, default=4)
    parser.add_argument("--ffn-multiplier-local", type=float, default=4.0)
    parser.add_argument("--ffn-multiplier-global", type=float, default=4.0)
    parser.add_argument("--ffn-multiplier-decoder", type=float, default=4.0)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--max-patch-length", type=int, default=32)
    parser.add_argument("--disable-hadamard", action="store_true")
    parser.add_argument("--disable-4bit-activations", action="store_true")
    parser.add_argument("--distill-temperature", type=float, default=1.0)

    parser.add_argument("--hard-ce-weight", type=float, default=1.0)
    parser.add_argument("--logits-kl-weight", type=float, default=1.0)
    parser.add_argument("--encoder-patch-mse-weight", type=float, default=1.0)
    parser.add_argument("--global-patch-mse-weight", type=float, default=1.0)
    parser.add_argument("--decoder-hidden-mse-weight", type=float, default=0.5)

    args = parser.parse_args(argv)
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.max_document_bytes <= 1:
        parser.error("--max-document-bytes must be greater than 1")
    if args.eval_every < 0:
        parser.error("--eval-every must be non-negative")
    if args.eval_steps <= 0:
        parser.error("--eval-steps must be positive")
    if args.save_every < 0:
        parser.error("--save-every must be non-negative")
    if args.prefetch_batches < 0:
        parser.error("--prefetch-batches must be non-negative")
    if args.student_patcher_warmup_steps < 0:
        parser.error("--student-patcher-warmup-steps must be non-negative")
    if args.teacher_upstream_repo is None and not args.no_teacher:
        parser.error("--teacher-upstream-repo is required unless --no-teacher is set")
    if args.resume_from is not None and not Path(args.resume_from).expanduser().exists():
        parser.error("--resume-from must point to an existing checkpoint")
    if args.student_patcher_mode != "off" and args.no_teacher:
        parser.error("--student-patcher-mode requires a teacher or externally supplied patch lengths")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_distillation(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
