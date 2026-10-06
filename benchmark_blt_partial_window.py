"""Synchronized, interleaved BLT partial-window A/B on 629M parameters.

The baseline restores the old window-divisibility dispatch gate. Both modes
use the same precision, weights, data, and optimizer settings. Generation
checks committed bytes. Training alternates on one evolving model; loss is
reported for finiteness, not as an equal-token quality comparison. Each length
and workload should run in a separate process to isolate allocator peaks.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
import tempfile
import time
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from blt.config import TernaryBLTConfig
from blt.mlx_layers import MLXTernarySelfAttention
from blt.mlx_model import MLXTernaryBLTModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', choices=['training', 'generation'], default='generation')
    parser.add_argument('--length', type=int, default=1025)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--new-bytes', type=int, default=16)
    parser.add_argument('--speculation-window', type=int, default=4)
    args = parser.parse_args()
    if args.length < 1 or args.pairs < 1 or args.warmup < 0 or args.new_bytes < 1 or args.speculation_window < 0:
        parser.error('length, pairs, and new-bytes must be positive; warmup and speculation-window must be nonnegative')
    config = TernaryBLTConfig(
        local_dim=512, global_dim=1024, decoder_dim=512,
        n_layers_local_encoder=1, n_layers_global=16, n_layers_local_decoder=7,
        n_heads_local_encoder=8, n_heads_global=16, n_heads_local_decoder=8,
        n_heads_cross=8, local_window=128, patch_size=4,
    )
    mx.random.seed(1337)
    model = MLXTernaryBLTModel(config)
    dtype = mx.bfloat16 if args.workload == 'training' else mx.float32
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    count = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    print(json.dumps({'parameters': count, 'dtype': str(dtype), 'device': mx.device_info(),
                      'settings': vars(args)}), flush=True)
    chunkable = MLXTernarySelfAttention._chunkable

    def old_gate(self, length, mask):
        return chunkable(self, length, mask) and length % self.local_window == 0

    def dispatch(mode):
        return patch.object(MLXTernarySelfAttention, '_chunkable', old_gate) if mode == 'baseline' else nullcontext()

    with tempfile.TemporaryDirectory(prefix='blt-partial-window-') as directory:
        if args.workload == 'training':
            import mlx.nn as nn
            from blt.mlx_data import ByteCorpus
            from blt.mlx_train import MLXBLTTrainer, TrainingConfig
            path = Path(directory) / 'corpus.bin'
            path.write_bytes(np.random.default_rng(1337).integers(0, 256, 65536, dtype=np.uint8).tobytes())
            trainer = MLXBLTTrainer(model, ByteCorpus(path, seq_len=args.length), TrainingConfig(
                steps=100, batch_size=1, warmup_steps=0, logits_kl=0))
            mx.eval(trainer.optimizer.state)
            gradients = {mode: mx.compile(nn.value_and_grad(model, trainer._loss),
                         inputs=[model.state], outputs=[model.state]) for mode in ['baseline', 'current']}
            def operation(mode):
                trainer._loss_and_grad = gradients[mode]
                # Repeated source bytes remove batch-complexity differences.
                trainer._rng = np.random.default_rng(1337)
                metrics = trainer.step(trainer.sample_batch(), 0)
                mx.eval(model.parameters(), trainer.optimizer.state)
                return {'loss': metrics['loss']}
            units = args.length
        else:
            from blt.mlx_generate import generate
            tokens = mx.array(np.random.default_rng(1337).integers(4, 260, (1, args.length)))
            mx.eval(tokens)
            expected = None
            def operation(mode):
                nonlocal expected
                output, stats = generate(model, tokens, max_new_bytes=args.new_bytes,
                                         speculation_window=args.speculation_window, eos_id=-1)
                mx.eval(output)
                row = output.tolist()
                if expected is None:
                    expected = row
                assert row == expected, 'committed generation bytes changed'
                return {'stats': asdict(stats)}
            units = args.new_bytes

        def run(mode):
            mx.reset_peak_memory()
            with dispatch(mode):
                start = time.perf_counter()
                result = operation(mode)
                seconds = time.perf_counter() - start
            return {**result, 'seconds': seconds, 'bytes_per_second': units / seconds,
                    'peak_gib': mx.get_peak_memory() / 2**30}

        modes = ['baseline', 'current']
        print(json.dumps({'cold': {mode: run(mode) for mode in modes}}), flush=True)
        for _ in range(args.warmup):
            for mode in modes:
                run(mode)
        samples = {mode: [] for mode in modes}
        for pair in range(args.pairs):
            for mode in (modes if pair % 2 == 0 else modes[::-1]):
                samples[mode].append(run(mode))
        seconds = {mode: median(row['seconds'] for row in rows) for mode, rows in samples.items()}
        print(json.dumps({'samples': samples, 'median_seconds': seconds,
                          'speedup': seconds['baseline'] / seconds['current']}), flush=True)


if __name__ == '__main__':
    main()
