Performance pass against `d6e9e19`, on Apple M1 Max with MLX
`0.31.2.dev20260426+72ec298`. Focus: the existing MLX BLT workloads; this is
not a CUDA or PyTorch performance claim.

Implemented partial-window local attention in `blt/mlx_layers.py`.
Previously, `_chunkable` required `sequence_length % local_window == 0`.
A 1,024-byte prefix could use blocked attention, then switch back to dense
attention after appending a byte. The new path pads Q/K/V to the next window
boundary and discards padded query outputs. Causal masking prevents real
queries from seeing padded keys. K/V caches retain their original lengths.
Explicit caller masks retain the existing fallback. Local attention work is
bounded by O(sequence_length × local_window), including partial final windows.
This does not remove quadratic global or encoder cross-attention work.

Measurements use 629,004,800 physical parameters, batch one, window 128,
uniform patches of four bytes, FP32 generation and BF16 training. Generation
produces 16 new bytes with speculation window four. Both arms share precision
and weights; the baseline restores only the old divisibility gate. Runs alternate
order for five warmed pairs after two warmups per arm. Cold results are recorded
separately. Outputs and training state are explicitly synchronized. Each
workload/length uses a separate process. Do not compare these absolute timings
with the September 7 runs: this is a shared desktop, and timing variation is
visible even between adjacent samples.

Training timings include corpus sampling, forward/backward, gradient clipping,
and CMUD application. Both arms update the same evolving model on repeated
source bytes; reported losses establish finiteness only. They are not paired
quality measurements. Validation and checkpoint pauses are outside these step
timings. Peak memory is MLX's allocator peak, including live model/state and
compiled graphs, not a process RSS measurement.

| Workload | Prefix/sequence bytes | Median seconds, old → new | Bytes/s, old → new | Throughput ratio | Peak GiB, old → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generation | 1,025 | 1.2082 → 1.1505 | 13.2 → 13.9 | 1.050× | 5.414 → 5.414 |
| Generation | 4,097 | 6.9975 → 5.7137 | 2.3 → 2.8 | 1.225× | 5.947 → 6.369 |
| Training | 1,025 | 1.0113 → 1.0335 | 1013.6 → 991.8 | 0.978× | 9.649 → 9.649 |
| Training | 4,097 | 7.3791 → 1.9220 | 555.2 → 2131.6 | 3.839× | 13.986 → 11.979 |

A fresh-process repeat of 4,097-byte training, with three warmups per arm
and five alternating pairs, measured **8.9921 → 2.3936 seconds/update
(3.76× throughput)** and 13.986 → 11.979 GiB peak.
See `2026-09-08-partial-window-training-4097-repeat.jsonl`.

The 1,025-byte training case is roughly flat/slightly slower. Generation at
4,097 bytes trades higher allocator peak for lower latency; this is not a
universal speed-and-memory win. Window-aligned inputs retain the original
attention computation.

Reproduce each workload in a separate process:

```sh
.venv/bin/python benchmark_blt_partial_window.py --workload generation --length 1025
.venv/bin/python benchmark_blt_partial_window.py --workload generation --length 4097
.venv/bin/python benchmark_blt_partial_window.py --workload training --length 1025
.venv/bin/python benchmark_blt_partial_window.py --workload training --length 4097
```

Raw samples, cold timings, model/device settings, and generation counters are
in `2026-09-08-partial-window-*.jsonl` beside this report. All generation runs
assert identical committed bytes across both arms. Synthetic weights do not
establish trained speculative acceptance rates or long-run model quality.

Validation: **771 tests passed** (`.venv/bin/python -m pytest -q`). Added
coverage includes partial windows at both ends of a window boundary,
non-power-of-two windows, FP32/FP16/BF16 outputs and input gradients,
whole-model parameter gradients, original K/V cache lengths, and explicit
mask fallback. Existing greedy/speculative generation regressions also pass.
`git diff --check` is clean.

Further opportunities, in priority order:

1. **Keep completed global patches cached across generation rounds.**
   `blt/mlx_generate.py::_run_global` reruns the encoder/global stack over the
   committed prefix; `_verify` also evaluates the full candidate. Local draft
   caches already exist, but global refresh discards that advantage. Cache
   completed-patch global K/V and latents, recomputing from the first changed
   patch. Preserve the open patch, entropy boundaries, speculative rollback,
   and the optional BitNet global backbone. This is the strongest next
   algorithmic candidate for long-prefix generation; no speedup measured here.
2. **Make encoder patch cross-attention operate on patch spans.**
   `blt/mlx_model.py::MLXLocalEncoder.__call__` constructs a `[batch, patches,
   bytes]` membership mask, then runs dense patch-to-byte attention even though
   each query sees only its own patch. Uniform patches admit a reshape; ragged
   patches suggest a span-aware Metal attention kernel with a matching backward.
   `blt/mlx_patching.py::patch_ids_from_lengths` also constructs a patches-by-bytes
   comparison before reducing it; upper-bound lookup on cumulative lengths
   could remove that intermediate. Profile the combined path before building
   the kernel. This is separate from the previously rejected decoder multi-slot
   gather-attention experiment.
3. **Packed BLT inference weights remain a memory candidate.** BLT locals still
   pin dense effective weights. Exploratory packed/native projection comparisons
   did not show a consistent latency win across decode and prefill shapes, so
   no packed default was added. Reuse of the BitNet packer must preserve BLT's
   dtype-sensitive ternarization thresholds and scales; low-precision matmul
   accumulation can differ even with identical reconstructed weights. Measure
   the complete generation path and memory before adopting it.

Optimizer experiments were rejected for this change. A custom Metal triangular
solve with fully unrolled thread-private intermediates matched the existing
output exactly but was slower in the sampled shapes. An exact small triangular
inverse followed by native matmul had mixed microbenchmarks and only about a 3%
full-training improvement in an exploratory 1,024-byte run, with changed
summation order. Neither result justifies changing the optimizer by default.
These were temporary probes, not a maintained benchmark or a quality study.
