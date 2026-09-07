"""Interleaved, synchronized A/B for September PaTH generation changes.

Defaults to >500M physical parameters. Synthetic weights establish runtime only;
they cannot establish trained speculative acceptance or generation quality.
"""

import argparse
from contextlib import ExitStack
import json
from statistics import median
import time
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_model import MLXBitNet, MLXBitNetConfig, MLXPaTHAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()
    mx.random.seed(1337)
    config = MLXBitNetConfig(
        vocab_size=1024, hidden_size=args.hidden_size, intermediate_size=2 * args.hidden_size,
        num_attention_heads=args.hidden_size // 64,
        num_prelude_layers=4, num_recurrent_layers=args.layers - 8, num_coda_layers=4,
        num_loops=1, path_window_size=args.window, infini_memory_dim=64,
        use_engram=False, use_path_kernel=True,
    )
    model = MLXBitNet(config)
    model.set_dtype(mx.bfloat16)
    model.eval()
    model.set_inference_block_width(args.window)
    mx.eval(model.parameters())
    count = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
    model.pin_inference_weights(prefer_packed=False)
    print(json.dumps({"parameters": count, "config": vars(args), "dtype": "bfloat16",
                      "weights": "pinned dense", "compiled": False}), flush=True)
    full_chunk = MLXPaTHAttention.path_chunk_with_state

    def serial_chunks(self, q, k, v, w, beta, forget, segments, count):
        width = q.shape[2] // count
        return mx.stack([
            self.path_chunk(q[:, :, i*width:(i+1)*width], k[:, :, i*width:(i+1)*width],
                            v[:, :, i*width:(i+1)*width], w[:, i*width:(i+1)*width],
                            beta[:, i*width:(i+1)*width], forget[:, i*width:(i+1)*width], None)
            for i in range(count)
        ], axis=1)

    def full_queries(self, q, k, v, w, beta, forget, segments, query_start=0):
        output, t, wk = full_chunk(self, q, k, v, w, beta, forget, segments)
        # Before the audit, incomplete extensions solved T a second time.
        if q.shape[2] < args.window:
            t = self.path_system_t_inverse(w, beta)
        return output[:, :, query_start:], t, wk

    def old_border(self, previous, w, beta):
        length = w.shape[1]
        if previous is None or length == 1:
            return self.path_system_t_inverse(w, beta)
        wf = w.transpose(0, 2, 1, 3).astype(mx.float32)
        b = beta[:, -1].astype(mx.float32)
        dots = mx.sum(wf[:, :, -1:] * wf[:, :, :-1], axis=-1)
        row = -mx.sum((b[:, :, None] * dots)[..., None] * previous, axis=2)
        top = mx.concatenate((previous, mx.zeros((*previous.shape[:2], length-1, 1))), axis=-1)
        bottom = mx.concatenate((row[:, :, None], b[:, :, None, None]), axis=-1)
        return mx.concatenate((top, bottom), axis=2)

    cases = [("prefill_short", args.window // 2, 0), ("prefill_long", args.window * 4, 0),
             ("extend", args.window - 17, 8), ("decode_wk", args.window - 17, 8),
             ("decode_border", args.window - 17, 8)]
    for case, prompt_len, new_len in cases:
        for block in model.blocks:
            block.attn.cache_path_products = case in {"decode_wk", "decode_border"}
        tokens = mx.random.randint(0, config.vocab_size, (1, prompt_len + new_len))
        mx.eval(tokens)
        seed = model.new_inference_cache()
        if new_len:
            mx.eval(model.prefill(tokens[:, :prompt_len], seed), seed.arrays())

        def run(baseline):
            cache = seed.clone() if new_len else model.new_inference_cache()
            with ExitStack() as stack:
                if baseline:
                    if case.startswith("prefill"):
                        stack.enter_context(patch.object(MLXPaTHAttention, "_batched_path_chunks", serial_chunks))
                    elif case == "extend":
                        stack.enter_context(patch.object(MLXPaTHAttention, "path_chunk_with_state", full_queries))
                    elif case == "decode_wk":
                        stack.enter_context(patch.object(MLXPaTHAttention, "path_wk_extend", lambda *args: None))
                    elif case == "decode_border":
                        stack.enter_context(patch.object(MLXPaTHAttention, "path_border_update_t", old_border))
                mx.reset_peak_memory()
                start = time.perf_counter()
                if not new_len:
                    output = model.prefill(tokens, cache)
                    mx.eval(output, cache.arrays())
                elif case == "extend":
                    output = model.inference_extend(tokens[:, prompt_len:], cache)
                    mx.eval(output, cache.arrays())
                else:
                    for index in range(prompt_len, prompt_len + new_len):
                        output = model.inference_step(tokens[:, index:index+1], cache)
                        mx.eval(output, cache.arrays())
                elapsed = time.perf_counter() - start
                peak = mx.get_peak_memory() / 2**30
                wk_bytes = sum(layer.attention.wk.nbytes for layer in cache.layers
                               if layer.attention.wk is not None)
            return elapsed, peak, wk_bytes

        cold = {"baseline": run(True), "current": run(False)}
        samples = {"baseline": [], "current": []}
        for pair in range(args.pairs):
            for baseline in ((True, False) if pair % 2 == 0 else (False, True)):
                samples["baseline" if baseline else "current"].append(run(baseline))
        summary = {mode: {"seconds": median(row[0] for row in rows),
                          "peak_gib": max(row[1] for row in rows),
                          "wk_mib": rows[-1][2] / 2**20} for mode, rows in samples.items()}
        print(json.dumps({"case": case, "prompt": prompt_len, "new": new_len,
                          "cold": cold, "samples": samples, "summary": summary,
                          "speedup": summary["baseline"]["seconds"] / summary["current"]["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
