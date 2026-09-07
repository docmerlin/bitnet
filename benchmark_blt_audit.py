"""Raw-byte ingestion through CMUD: dense/eager vs windowed/compiled apply.

Uses the same initialized model and optimizer in alternating order. Training
continues across arms; this is a throughput benchmark, not a quality experiment.
"""

import argparse
import json
from pathlib import Path
from statistics import median
import tempfile
import time
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from blt.config import TernaryBLTConfig
from blt.mlx_data import ByteCorpus
from blt.mlx_model import MLXTernaryBLTModel
from blt.mlx_train import MLXBLTTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if len(args.sequence_lengths) > 1:
        # Compiled MLX graphs can retain the previous model's state. Separate
        # processes keep peak memory and timing comparable between lengths.
        for length in args.sequence_lengths:
            subprocess.run([
                sys.executable, __file__, "--pairs", str(args.pairs),
                "--warmup-rounds", str(args.warmup_rounds),
                "--batch-size", str(args.batch_size), "--sequence-lengths", str(length),
            ], check=True)
        return
    config = TernaryBLTConfig(
        local_dim=512, global_dim=1024, decoder_dim=512,
        n_layers_local_encoder=1, n_layers_global=16, n_layers_local_decoder=7,
        n_heads_local_encoder=8, n_heads_global=16, n_heads_local_decoder=8,
        n_heads_cross=8, local_window=128, patch_size=4,
    )
    with tempfile.TemporaryDirectory(prefix="blt-audit-") as directory:
        corpus_path = Path(directory) / "corpus.bin"
        corpus_path.write_bytes(np.random.default_rng(1337).integers(0, 256, 65536, dtype=np.uint8).tobytes())
        for length in args.sequence_lengths:
            mx.random.seed(1337)
            model = MLXTernaryBLTModel(config)
            model.set_dtype(mx.bfloat16)
            mx.eval(model.parameters())
            count = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
            print(json.dumps({"parameters": count, "length": length, "batch": args.batch_size,
                              "dtype": "bfloat16"}), flush=True)
            corpus = ByteCorpus(corpus_path, seq_len=length)
            trainer = MLXBLTTrainer(model, corpus, TrainingConfig(
                steps=100, batch_size=args.batch_size, compile_step=True,
                logits_kl=0, patches_per_sequence=length // 4, warmup_steps=0,
            ))
            mx.eval(trainer.optimizer.state)
            apply = trainer._apply_step

            def make_gradient(unpadded):
                def loss(batch):
                    trainer._unpadded = unpadded
                    return trainer._terms(batch)
                return mx.compile(nn.value_and_grad(model, loss), inputs=[model.state], outputs=[model.state])

            gradients = {False: make_gradient(False), True: make_gradient(True)}
            settings = {"baseline": (False, False), "windowed": (True, False), "compiled_apply": (True, True)}

            def run(mode):
                unpadded, compiled = settings[mode]
                trainer._unpadded = unpadded
                trainer._loss_and_grad = gradients[unpadded]
                trainer._apply_step = apply if compiled else None
                trainer._rng = np.random.default_rng(1337)
                mx.reset_peak_memory()
                start = time.perf_counter()
                metrics = trainer.step(trainer.sample_batch(), 0)
                elapsed = time.perf_counter() - start
                return {"seconds": elapsed, "bytes_per_second": args.batch_size * length / elapsed,
                        "peak_gib": mx.get_peak_memory() / 2**30, "loss": metrics["loss"]}

            print(json.dumps({"parameters": count, "length": length, "batch": args.batch_size,
                              "dtype": "bfloat16", "cold": {mode: run(mode) for mode in settings}}), flush=True)
            for _ in range(args.warmup_rounds):
                for mode in settings:
                    run(mode)
            samples = {mode: [] for mode in settings}
            for pair in range(args.pairs):
                order = list(settings) if pair % 2 == 0 else list(reversed(settings))
                for mode in order:
                    samples[mode].append(run(mode))
            print(json.dumps({"length": length, "samples": samples,
                              "median_seconds": {mode: median(row["seconds"] for row in rows)
                                                 for mode, rows in samples.items()}}), flush=True)
            # Release all compiled closures before allocating the next scale.
            del gradients, apply, trainer, model
            mx.clear_cache()


if __name__ == "__main__":
    main()
