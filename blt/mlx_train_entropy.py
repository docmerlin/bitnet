"""Train the byte entropy model that drives BLT patching.

Runs before the student: the entropy model decides where patches begin, and the
student is trained against those boundaries. Objective is plain next-byte
cross-entropy on raw bytes -- no teacher, no labels beyond the corpus itself.

The last step is calibration. The model produces entropies in nats, but the knob
that governs cost is the average patch width, since that sets how often the
global model runs. :func:`blt.mlx_entropy_model.calibrate_threshold` solves for
the cutoff that yields the requested width and stores it on the model, so the
saved checkpoint carries a patcher that is ready to use rather than one that
still needs a magic number supplied alongside it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from blt.config import TernaryBLTConfig
from blt.mlx_data import ByteCorpus
from blt.mlx_entropy_model import MLXByteEntropyModel, calibrate_threshold
from blt.mlx_train import clip_gradients


def train_entropy_model(
    model: MLXByteEntropyModel,
    corpus: ByteCorpus,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 50,
    log=print,
) -> list[dict[str, float]]:
    """Next-byte cross-entropy training. Returns per-step metrics."""
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_and_grad = mx.value_and_grad(lambda m, tokens: m.loss(tokens))
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for step in range(steps):
        if step < warmup_steps:
            rate = learning_rate * (step + 1) / max(warmup_steps, 1)
        else:
            progress = min((step - warmup_steps) / max(steps - warmup_steps, 1), 1.0)
            rate = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.learning_rate = rate

        tokens = mx.array(corpus.batch(rng.integers(0, len(corpus), size=batch_size))["tokens"])
        loss, gradients = loss_and_grad(model, tokens)
        gradients, grad_norm = clip_gradients(gradients, grad_clip)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, grad_norm)

        metrics = {
            "step": step,
            "loss": float(loss),
            # Bits per byte is the readable form: 8.0 means the model has learnt
            # nothing about a byte stream, and good text models sit near 1.
            "bits_per_byte": float(loss) / math.log(2),
            "grad_norm": float(grad_norm),
            "learning_rate": rate,
        }
        history.append(metrics)
        if log_every and step % log_every == 0:
            log(
                f"step {step:>6} loss={metrics['loss']:.4f} "
                f"bpb={metrics['bits_per_byte']:.3f} grad_norm={metrics['grad_norm']:.3f} "
                f"lr={rate:.2e} {time.perf_counter() - started:.1f}s"
            )
    return history


def save_entropy_model(model: MLXByteEntropyModel, path: str | Path, *, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
    # The threshold is calibration state rather than a parameter, so it does not
    # ride in the safetensors; keep it beside the weights where a loader can find it.
    path.with_suffix(".json").write_text(
        json.dumps({**meta, "threshold": model.default_threshold}, indent=2)
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BLT byte entropy model on a raw corpus.")
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--output", default="checkpoints/blt-entropy.safetensors")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--target-patch-size",
        type=float,
        default=4.0,
        help="Average bytes per patch to calibrate the entropy threshold for.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TernaryBLTConfig()
    corpus = ByteCorpus(args.corpus, seq_len=args.seq_len, offset=config.offset)
    print(
        f"corpus: {corpus.total_bytes / 1e6:.2f}MB over {len(args.corpus)} file(s), "
        f"{len(corpus)} sequences of {args.seq_len}"
    )

    mx.random.seed(args.seed)
    model = MLXByteEntropyModel(
        config,
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        max_seq_len=max(args.seq_len, 512),
    )
    mx.eval(model.parameters())
    parameters = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"entropy model: {parameters / 1e6:.2f}M parameters")

    history = train_entropy_model(
        model,
        corpus,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
    )

    rng = np.random.default_rng(args.seed + 1)
    sample_size = min(len(corpus), 64)
    sample = mx.array(corpus.batch(rng.integers(0, len(corpus), size=sample_size))["tokens"])
    threshold = calibrate_threshold(model, sample, target_patch_size=args.target_patch_size)
    lengths = model.predict_patch_lengths(sample)
    width = float(mx.sum(lengths)) / float(mx.sum((lengths > 0).astype(mx.int32)))
    print(
        f"calibrated: threshold {threshold:.3f} nats -> mean patch width {width:.2f} "
        f"(target {args.target_patch_size})"
    )

    save_entropy_model(
        model,
        args.output,
        meta={
            "dim": args.dim,
            "layers": args.layers,
            "heads": args.heads,
            "max_seq_len": max(args.seq_len, 512),
            "target_patch_size": args.target_patch_size,
            "mean_patch_width": width,
            "final_bits_per_byte": history[-1]["bits_per_byte"] if history else None,
        },
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
