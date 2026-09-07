"""629M BLT follow-up benchmark; synchronize every measured result.

Training: run legacy/native/compact in isolated processes (legacy needs a checkout
of 3257f70 via --baseline). All use identical seeds and batch order. BF16 storage
in legacy promoted projections to FP32; native fixes this, so this comparison is
not precision-neutral. Loss on synthetic bytes does not establish model quality.

Generation: --workload generation --baseline PATH loads the old generation loop
against the same FP32 model for warmed, interleaved comparisons at equal precision.
"""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mode", choices=["legacy", "native", "compact"], default="native")
    parser.add_argument("--workload", choices=["training", "generation"], default="training")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--new-bytes", type=int, default=16)
    parser.add_argument("--speculation-window", type=int, default=4)
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--replicate", type=int, default=1, help="Label an independent process run.")
    args = parser.parse_args()
    if args.mode == "legacy":
        if args.baseline is None:
            parser.error("legacy requires --baseline")
        sys.path.insert(0, str(args.baseline.resolve()))

    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from blt.config import TernaryBLTConfig
    from blt.mlx_data import ByteCorpus
    from blt.mlx_model import MLXTernaryBLTModel
    from blt.mlx_train import MLXBLTTrainer, TrainingConfig

    config = TernaryBLTConfig(
        local_dim=512, global_dim=1024, decoder_dim=512,
        n_layers_local_encoder=1, n_layers_global=16, n_layers_local_decoder=7,
        n_heads_local_encoder=8, n_heads_global=16, n_heads_local_decoder=8,
        n_heads_cross=8, local_window=128, patch_size=4,
    )
    mx.random.seed(1337)
    model = MLXTernaryBLTModel(config)
    model.set_dtype(mx.bfloat16 if args.workload == "training" else mx.float32)
    mx.eval(model.parameters())
    parameters = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    print(json.dumps({"parameters": parameters, "settings": vars(args)}, default=str), flush=True)

    if args.workload == "generation":
        if args.baseline is None:
            parser.error("generation requires --baseline")
        from blt.mlx_generate import generate
        from blt.mlx_entropy_model import MLXByteEntropyModel
        spec = importlib.util.spec_from_file_location("blt._followup_baseline", args.baseline / "blt/mlx_generate.py")
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        patcher = MLXByteEntropyModel(config, max_seq_len=args.length + args.new_bytes + 8) if args.entropy else None
        if patcher is not None:
            # Isolate length-cap patching, avoiding threshold ties in random weights.
            patcher.set_threshold(100.0)
        prompt = mx.array(np.random.default_rng(1337).integers(4, 260, (1, args.length)))
        modes = {"legacy": baseline.generate, "native": generate}
        expected = None

        def run(mode):
            nonlocal expected
            mx.reset_peak_memory()
            started = time.perf_counter()
            output, stats = modes[mode](model, prompt, max_new_bytes=args.new_bytes,
                                       speculation_window=args.speculation_window, patcher=patcher, eos_id=-1)
            mx.eval(output)
            elapsed = time.perf_counter() - started
            row = output.tolist()
            if expected is None:
                expected = row
            assert row == expected, "generation changed committed bytes"
            return {"seconds": elapsed, "bytes_per_second": args.new_bytes / elapsed,
                    "peak_gib": mx.get_peak_memory() / 2**30, "stats": asdict(stats)}

        print(json.dumps({"cold": {mode: run(mode) for mode in modes}}), flush=True)
        for _ in range(args.warmup):
            for mode in modes:
                run(mode)
        samples = {mode: [] for mode in modes}
        for pair in range(args.samples):
            for mode in (list(modes) if pair % 2 == 0 else list(reversed(modes))):
                samples[mode].append(run(mode))
        print(json.dumps({"samples": samples, "median_seconds": {
            mode: median(row["seconds"] for row in rows) for mode, rows in samples.items()}}), flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="blt-followup-") as directory:
        path = Path(directory) / "bytes.bin"
        path.write_bytes(np.random.default_rng(1337).integers(0, 256, 65536, dtype=np.uint8).tobytes())
        options = {"mud_eight_bit": True, "mud_master_dtype": "bfloat16"} if args.mode == "compact" else {}
        trainer = MLXBLTTrainer(model, ByteCorpus(path, seq_len=args.length), TrainingConfig(
            steps=100, batch_size=1, warmup_steps=0, logits_kl=0, **options))
        mx.eval(trainer.optimizer.state)
        state_bytes = sum(x.nbytes for _, x in tree_flatten(trainer.optimizer.state) if isinstance(x, mx.array))

        def step(index):
            mx.reset_peak_memory()
            started = time.perf_counter()
            metrics = trainer.step(trainer.sample_batch(), index)
            mx.eval(model.parameters(), trainer.optimizer.state)
            seconds = time.perf_counter() - started
            return {"seconds": seconds, "bytes_per_second": args.length / seconds,
                    "loss": metrics["loss"], "peak_gib": mx.get_peak_memory() / 2**30}

        print(json.dumps({"state_gib": state_bytes / 2**30, "cold": step(0)}), flush=True)
        for i in range(args.warmup):
            step(i + 1)
        samples = [step(i + args.warmup + 1) for i in range(args.samples)]
        print(json.dumps({"samples": samples, "median_seconds": median(x["seconds"] for x in samples)}), flush=True)


if __name__ == "__main__":
    main()
