"""MLX training for the ternary BLT student, from a raw corpus or a cached teacher.

The teacher never runs here. :mod:`blt.teacher_cache` holds its top-k logits and
patch boundaries, dumped once under torch, so this loop touches numpy and MLX
only -- which is the whole reason the cache exists.

Two consequences of training from a cache, both deliberate:

- Patch boundaries are the teacher's, replayed from disk. There is no
  student-patcher mode here; the torch trainer's ``teacher_then_student``
  schedule needs a live teacher to compare against.
- Only ``hard_ce`` and ``logits_kl`` are available. See :mod:`blt.mlx_losses` for
  why the hidden-state MSE terms are absent rather than silently zero.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from blt.config import TernaryBLTConfig
from blt.mlx_data import ByteCorpus
from blt.mlx_entropy_model import MLXByteEntropyModel, load_entropy_model
from blt.mlx_losses import MLXDistillationLossWeights, blt_distillation_loss
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_patching import build_uniform_patch_lengths, pad_patch_lengths_to_bucket
from blt.teacher_cache import TeacherCache
from mlx_optim import CMUD


# MUD whitens each 2D parameter in row blocks. Left unset, ``mud_decorrelate``
# takes the whole matrix as one block and does a [rows, rows] triangular solve
# per parameter, which at BLT's widths costs 8.4s per step against a 195ms
# forward+backward -- 96% of training spent in the optimizer. Blocking at 64
# rows measures 26x faster end to end and is what the BitNet trainer already
# defaults to (``mlx_train.py --mud-block-size``). 32 is marginally faster still
# but was never validated for quality, and this repo's own sweep found 64
# reached better perplexity than 256.
MUD_BLOCK_SIZE = 64


class BLTCMUD(CMUD):
    """CMUD with BLT's embedding routed to the fallback optimizer.

    ``CMUD._is_mud_parameter`` excludes parameters whose path ends in
    ``embedding.weight``. BLT's is ``byte_embeddings.weight`` -- plural -- so it
    would otherwise be handed to MUD, which whitens 2D parameters and is the
    wrong treatment for a lookup table.
    """

    def __init__(self, *, block_size: int | None = MUD_BLOCK_SIZE, **kwargs) -> None:
        super().__init__(block_size=block_size, **kwargs)

    @staticmethod
    def _is_mud_parameter(path: str, parameter: mx.array) -> bool:
        if path.endswith("byte_embeddings.weight"):
            return False
        return CMUD._is_mud_parameter(path, parameter)


@dataclass
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 2e-3
    fallback_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    temperature: float = 1.0
    hard_ce: float = 1.0
    logits_kl: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 10
    seed: int = 0
    mud_block_size: int = MUD_BLOCK_SIZE
    compile_step: bool = True
    # Patch-count bucket for compiled runs; see pad_patch_lengths_to_bucket.
    # Ignored when patches_per_sequence is set, and unsafe with the BitNet
    # global backbone -- see blt.mlx_global.
    patch_bucket: int = 32
    # Fixed patches per sequence. Gives stable shapes without any padding, which
    # is what the BitNet global backbone needs to run compiled.
    patches_per_sequence: int | None = None
    # Quantisation ramp. Full 4-bit activations from a cold start diverge to NaN
    # within two steps; the weights are fine at full ternary. Defaults match
    # mlx_train.py's --stage1-* schedule.
    quant_ramp_ratio: float = 0.25
    weight_mix_start: float = 0.25
    activation_mix_start: float = 0.0
    activation_bits_start: int = 16
    activation_bits_final: int = 8


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    """Linear warmup into a cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    span = max(config.steps - config.warmup_steps, 1)
    progress = min((step - config.warmup_steps) / span, 1.0)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_gradients(gradients, max_norm: float):
    """Global-norm clip. Returns ``(gradients, norm)``; norm is pre-clip."""
    flat = [g for _, g in tree_flatten(gradients)]
    total = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat))
    if max_norm <= 0.0:
        return gradients, total
    scale = mx.minimum(max_norm / mx.maximum(total, 1e-6), 1.0)
    return tree_unflatten([(name, g * scale) for name, g in tree_flatten(gradients)]), total


def shifted_labels(tokens: mx.array, mask: mx.array, pad_id: int) -> tuple[mx.array, mx.array]:
    """Next-byte targets, with the final position dropped from the loss.

    Position ``t`` predicts byte ``t+1``; the last position has no target, so its
    mask entry is cleared rather than left pointing at wrapped-around garbage.
    """
    labels = mx.concatenate([tokens[:, 1:], mx.full((tokens.shape[0], 1), pad_id, tokens.dtype)], axis=1)
    loss_mask = mx.concatenate([mask[:, 1:], mx.zeros((mask.shape[0], 1), dtype=mx.bool_)], axis=1)
    return labels.astype(mx.int32), loss_mask & mask


class MLXBLTTrainer:
    def __init__(
        self,
        model: MLXTernaryBLTModel,
        source: TeacherCache | ByteCorpus,
        config: TrainingConfig,
        *,
        optimizer=None,
        patcher: MLXByteEntropyModel | None = None,
    ) -> None:
        self.model = model
        self.source = source
        self.config = config
        self.patcher = patcher
        self.weights = MLXDistillationLossWeights(
            hard_ce=config.hard_ce, logits_kl=config.logits_kl
        )
        # A raw corpus carries no teacher, so the KL term has nothing to match.
        # Better to say so than to train silently on hard CE alone while the
        # caller believes distillation is running.
        self.has_teacher = isinstance(source, TeacherCache)
        if self.weights.logits_kl > 0.0 and not self.has_teacher:
            raise ValueError(
                "logits_kl > 0 needs a TeacherCache; a ByteCorpus has no teacher "
                "distribution. Pass --logits-kl 0 to train from scratch."
            )
        if self.weights.hard_ce <= 0.0 and not self.has_teacher:
            raise ValueError("training from a raw corpus needs hard_ce > 0")

        self.optimizer = optimizer or BLTCMUD(
            mud_learning_rate=config.learning_rate,
            fallback_learning_rate=config.fallback_learning_rate,
            weight_decay=config.weight_decay,
            block_size=config.mud_block_size,
        )
        self._rng = np.random.default_rng(config.seed)
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)
        # Config-time rather than per-batch: detecting padding means reading the
        # mask, which forces a sync and blocks compilation.
        if (
            not getattr(model.global_transformer, "accepts_padding", True)
            and config.patches_per_sequence is None
            and config.patch_bucket > 0
            and patcher is not None
        ):
            raise ValueError(
                "this global backbone cannot take padded patches, but entropy "
                "patching with patch_bucket > 0 produces them. Set "
                "patches_per_sequence for stable shapes without padding, or "
                "patch_bucket=0 to accept recompiles."
            )
        if config.compile_step:
            # The forward validates its attention mask by reading it, which
            # forces a sync and makes compilation impossible. Batches here are
            # built from numpy masks that are suffix-padded by construction, so
            # the check has nothing to catch and is turned off rather than
            # working around.
            self.model.validate_inputs = False
            if hasattr(self.model.global_transformer, "validate_inputs"):
                self.model.global_transformer.validate_inputs = False
            self._loss_and_grad = mx.compile(
                self._loss_and_grad, inputs=[self.model.state], outputs=[self.model.state]
            )

    def _terms(self, batch: dict[str, mx.array]):
        output = self.model(
            batch["tokens"],
            attention_mask=batch["mask"],
            patch_lengths=batch["patch_lengths"],
        )
        return blt_distillation_loss(
            output.logits,
            labels=batch["labels"],
            attention_mask=batch["loss_mask"],
            weights=self.weights,
            temperature=self.config.temperature,
            teacher_topk_indices=batch.get("topk_indices"),
            teacher_topk_logits=batch.get("topk_logits"),
        )

    def _loss(self, batch: dict[str, mx.array]) -> mx.array:
        """Scalar loss, and nothing else.

        Deliberately free of side effects: stashing the metric breakdown on
        ``self`` here would make the function impure and ``mx.compile`` rejects
        it. The breakdown is recovered by :meth:`loss_breakdown` on the steps
        that actually log one.
        """
        return self._terms(batch)[0]

    def loss_breakdown(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """Per-term metrics for one batch. Uncompiled, so only call it to log."""
        _, metrics = self._terms(batch)
        mx.eval(list(metrics.values()))
        return {name: float(value) for name, value in metrics.items()}

    def patch_lengths_for(self, tokens: mx.array) -> mx.array:
        """Boundaries for a raw batch: the entropy patcher, or uniform widths.

        Kept out of the gradient either way -- patching is a decision about the
        input, not something the student's loss should be able to move.
        """
        if self.patcher is None:
            return build_uniform_patch_lengths(
                tokens.shape[0], tokens.shape[1], self.model.config.patch_size
            )
        # A fixed count already yields one shape, so no padding is needed --
        # the only option the BitNet global backbone can use, since padding
        # perturbs it (see blt.mlx_global).
        if self.config.patches_per_sequence is not None:
            return mx.stop_gradient(
                self.patcher.predict_patch_lengths(
                    tokens, num_patches=self.config.patches_per_sequence
                )
            )
        lengths = mx.stop_gradient(self.patcher.predict_patch_lengths(tokens))
        # Otherwise entropy patching changes the patch count almost every batch,
        # which would make a compiled step recompile continuously. patch_bucket=0
        # opts out.
        if self.config.patch_bucket <= 0:
            return lengths
        return pad_patch_lengths_to_bucket(lengths, self.config.patch_bucket)

    def sample_batch(self) -> dict[str, mx.array]:
        indices = self._rng.integers(0, len(self.source), size=self.config.batch_size)
        raw = self.source.batch(indices)
        tokens = mx.array(raw["tokens"])
        mask = mx.array(raw["mask"])
        labels, loss_mask = shifted_labels(tokens, mask, self.model.config.pad_id)
        batch = {
            "tokens": tokens,
            "mask": mask,
            "patch_lengths": (
                mx.array(raw["patch_lengths"])
                if "patch_lengths" in raw
                else self.patch_lengths_for(tokens)
            ),
            "labels": labels,
            "loss_mask": loss_mask,
        }
        if self.has_teacher:
            batch["topk_indices"] = mx.array(raw["topk_indices"])
            batch["topk_logits"] = mx.array(raw["topk_logits"])
        return batch

    def quantization_at(self, step_index: int) -> tuple[float, float, int]:
        """Weight mix, activation mix and activation bits for this step."""
        config = self.config
        if config.quant_ramp_ratio <= 0:
            fraction = 1.0
        else:
            progress = step_index / max(config.steps, 1)
            fraction = min(progress / config.quant_ramp_ratio, 1.0)
        weight_mix = config.weight_mix_start + fraction * (1.0 - config.weight_mix_start)
        activation_mix = config.activation_mix_start + fraction * (1.0 - config.activation_mix_start)
        bits = round(
            config.activation_bits_start
            - fraction * (config.activation_bits_start - config.activation_bits_final)
        )
        return weight_mix, activation_mix, bits

    def step(
        self, batch: dict[str, mx.array], step_index: int, *, breakdown: bool = False
    ) -> dict[str, float]:
        rate = learning_rate_at(step_index, self.config)
        multiplier = rate / self.config.learning_rate if self.config.learning_rate else 0.0
        if hasattr(self.optimizer, "set_lr_multiplier"):
            self.optimizer.set_lr_multiplier(multiplier)
        else:
            self.optimizer.learning_rate = rate
        self.model.set_quantization_state(*self.quantization_at(step_index))

        loss, gradients = self._loss_and_grad(batch)
        gradients, grad_norm = clip_gradients(gradients, self.config.grad_clip)
        self.optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters(), self.optimizer.state, loss, grad_norm)

        metrics = {"loss": float(loss), "grad_norm": float(grad_norm), "learning_rate": rate}
        if breakdown:
            # Costs a second forward, so it is reserved for steps that log.
            metrics.update(self.loss_breakdown(batch))
        return metrics

    def train(self, *, log=print) -> list[dict[str, float]]:
        history = []
        started = time.perf_counter()
        for step_index in range(self.config.steps):
            logging = bool(self.config.log_every) and step_index % self.config.log_every == 0
            metrics = self.step(self.sample_batch(), step_index, breakdown=logging)
            metrics["step"] = step_index
            history.append(metrics)
            if self.config.log_every and step_index % self.config.log_every == 0:
                elapsed = time.perf_counter() - started
                parts = " ".join(
                    f"{name}={metrics[name]:.4f}"
                    for name in ("loss", "hard_ce", "logits_kl", "grad_norm")
                    if name in metrics
                )
                log(f"step {step_index:>6} {parts} lr={metrics['learning_rate']:.2e} {elapsed:.1f}s")
        return history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(path / "model.safetensors"), dict(tree_flatten(self.model.parameters()))
        )
        # TernaryBLTConfig is a slots dataclass, so it has no __dict__.
        (path / "config.json").write_text(json.dumps(asdict(self.model.config), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the ternary BLT student in MLX, from raw bytes or a cached teacher."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", nargs="+", help="Raw byte files to train on from scratch.")
    source.add_argument("--teacher-cache", help="Directory written by TeacherCacheWriter.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length when using --corpus.")
    parser.add_argument(
        "--entropy-model",
        help="safetensors of a trained MLXByteEntropyModel to patch with. "
        "Without it a --corpus run falls back to fixed-width patches.",
    )
    parser.add_argument("--output", default="checkpoints/blt-mlx")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--fallback-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard-ce", type=float, default=1.0)
    parser.add_argument("--logits-kl", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mud-block-size",
        type=int,
        default=MUD_BLOCK_SIZE,
        help="Row block for MUD whitening. Unset means the whole matrix, which is ~26x slower.",
    )
    parser.add_argument("--local-dim", type=int, default=256)
    parser.add_argument("--global-dim", type=int, default=512)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    config = TernaryBLTConfig(
        local_dim=args.local_dim,
        global_dim=args.global_dim,
        decoder_dim=args.decoder_dim,
        patch_size=args.patch_size,
    )
    mx.random.seed(args.seed)

    patcher = None
    if args.teacher_cache:
        source = TeacherCache(args.teacher_cache)
        coverage = source.coverage(np.arange(min(len(source), 64)))
        print(f"teacher cache: {len(source)} sequences, top-{source.meta.top_k}, coverage {coverage:.4f}")
        if coverage < 0.99:
            print(
                f"  warning: top-{source.meta.top_k} keeps only {coverage:.3f} of the teacher's mass; "
                "the KL target is visibly not the teacher's distribution. Re-dump with a larger top_k."
            )
        if config.vocab_size != source.meta.vocab_size:
            raise ValueError(
                f"student vocab {config.vocab_size} != cached teacher vocab {source.meta.vocab_size}"
            )
    else:
        source = ByteCorpus(args.corpus, seq_len=args.seq_len, offset=config.offset)
        print(
            f"corpus: {source.total_bytes / 1e6:.2f}MB over {len(args.corpus)} file(s), "
            f"{len(source)} sequences of {args.seq_len}"
        )
        if args.entropy_model:
            # Architecture and threshold ride in the sidecar, so no flags here
            # have to be kept in sync with how the patcher was trained.
            patcher = load_entropy_model(args.entropy_model, config)
            print(f"patcher: entropy model, threshold {patcher.default_threshold:.3f} nats")
        else:
            print(
                f"  warning: no --entropy-model, so patches are fixed at {config.patch_size} bytes. "
                "Entropy patching is what BLT's efficiency rests on; train one first."
            )

    model = MLXTernaryBLTModel(config)
    mx.eval(model.parameters())
    parameters = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"student: {parameters / 1e6:.2f}M parameters")

    training = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fallback_learning_rate=args.fallback_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        hard_ce=args.hard_ce,
        logits_kl=args.logits_kl,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        seed=args.seed,
        mud_block_size=args.mud_block_size,
    )
    trainer = MLXBLTTrainer(model, source, training, patcher=patcher)
    trainer.train()
    trainer.save(args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
