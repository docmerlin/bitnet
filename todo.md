# TODO

Direction: MoE for **local, memory-bound** inference (single node, VRAM/RAM/disk),
not data-center load-balanced serving. Concentrate expert usage into small hot set,
stay VRAM-resident; offload cold tail. Self-gating experts → also extensible
(append experts to cold tier). Repo already got ternary weights + 4-bit acts
(`layers/h_bitlinear.py`) + logit z-loss.

## Status

**Training status: COLD / phase 1 smoke complete.** Only two diagnostic optimization
steps (128 tokens) have run. No usable trained checkpoint, baseline, convergence result,
or downstream evaluation exists yet. Items below are implemented and unit-tested, not
empirically validated by a real training run.

Training progress:
- [x] Model, loss, curriculum, checkpoint, and RFMoE append paths implemented.
- [x] Unit tests pass.
- [x] Phase 1: run a short end-to-end smoke train; verify finite loss, gradients,
  checkpoint save/resume, and memory/runtime behavior.
- [ ] Phase 2: choose data and model scale; train the first base model from scratch.
- [ ] Phase 3: evaluate base-model loss/quality, RFMoE density, expert usage, locality,
  loop health, and inference performance.
- [ ] Phase 4: only after a usable base checkpoint exists, append experts and run the
  new-domain retention/specialization experiment.

Phase 1 smoke result (2026-07-11, MPS, FineWeb-Edu, 0.03M-parameter RFMoE diagnostic):
- Step 1: 64 tokens, train loss 6.24388, grad norm 1.38943, val loss 6.20489.
- Resumed from saved checkpoint for step 2: 128 total tokens, train loss 6.23199,
  grad norm 1.38470, val loss 6.20075; final checkpoint saved under `runs/smoke-resume/`.
- RFMoE density was 1.0, expected during flat warmup; no sparsity conclusion possible.
- MPS requires sequence length divisible by Infini-Attention memory dimension 64;
  training CLI now rejects incompatible lengths before startup.
- This proves wiring only. Token count and model size are intentionally too small for
  quality, convergence, throughput, or RFMoE specialization claims.
- PaTH-FoX follow-up smoke (MPS): two checkpointed RFMoE steps with 16-token local
  windows completed and resumed successfully; loss stayed finite (6.28328 → 6.27290).

Implemented (see `layers/rfmoe.py`, `train.py`, `config.py`, `model.py`):
- RFMoE self-gating FFN, off by default behind `use_rfmoe`. θ inference knob.
- Adaptive-λ density control → global density target. `--rfmoe-density-target/-eta`.
- Staircase locality loss KL(π‖sorted p), EMA-ranked stop-grad. `--rfmoe-locality-coef/-zipf-s/-uniform-alpha`.
- Flat→skew curriculum (anneal s:0→s, α:1→α). `--rfmoe-curriculum-ratio`.
- Functional-diversity loss (decorrelate per-token firing). `--rfmoe-diversity-coef`.
- MTP (multi-token prediction) for AR data efficiency, k extra heads reuse tied unembedding. `--mtp-depth/-loss-coef`.
- **Looped / recurrent-depth BitNetDeep:** prelude 8 + recurrent 32 × R + coda 8 (default R=4).
  Flat `layers.i` ModuleList; forward schedules loops. CLI structure flags + `--num-loops`.
  Infini policy B: read every loop, write only last recurrent loop. Eval override: `num_loops=`.
- **Hyperloop loop HC** (`layers/loop_mhc.py`): 4 streams, diagonal `H_res` (no Sinkhorn),
  pre/post + loop embeds at each recurrent iteration. Hardcoded, not config knobs.
- RFMoE grouped/padded GEMM execution: scores batch across experts; active token/expert pairs
  pack into batched expert-body matmuls while checkpoint parameter keys stay stable.
- Ternary RFMoE experts: all score, gate, and body projections use `HBitLinear` weight/
  activation quantization while retaining grouped execution.
- Extensible RFMoE primitive: append cold experts dynamically, inherit quantization state,
  grow usage buffers, freeze old model weights, and train only appended experts with existing
  diversity loss as niche objective. Model config tracks new count for checkpoint reconstruction.
- PaTH-FoX replaces RoPE/YaRN in the BitNet path: low-rank data-dependent Householder
  transitions plus forget gates use paper-exact logits. Local UT attention is capped by
  `--path-window-size` (64 default), so attention storage/work stays linear in total context;
  fixed-size Infini memory carries compressed information beyond local windows. BLT keeps RoPE.

## Next actions

1. **First smoke train:** use a small configuration and short run to validate the complete
   data→forward→loss→backward→optimizer→checkpoint/resume path. This is not a quality run.
2. **First base-model run:** choose dataset, tokenizer/checkpoint strategy, model scale,
   token budget, batch/accumulation, and evaluation cadence; then train from scratch.
3. **Base-model evaluation:** establish loss/quality baseline and measure RFMoE density,
   expert usage/locality, loop health, PaTH long-context retrieval, throughput, and memory
   before adding experts. Compare 8K/16K/32K/64K contexts and local-window ablations.
4. **Extensible MoE experiment:** choose domain/task boundary, append experts flat (`b≈0`),
   train new-only with diversity, then raise bias into cold tier and measure retained old-domain loss.
5. **Serving:** tier experts by usage (hot→VRAM, warm→RAM, cold→SSD), offload + prefetch.
   Optional temporal-stickiness loss (penalize active-set change token-to-token → less page thrash).
6. **PaTH performance:** replace the current correct PyTorch PaTH-FoX UT reference path with
   a full optimized Triton kernel (FlashLinearAttention-style block scan, online softmax,
   efficient transformed-query/key preprocessing, and decode/cache support).
7. **Diffusion (thread 3):** large new direction; locality reg (built) is prerequisite (see below).
8. **Looped follow-ups (optional):** stochastic/Poisson R, input injection each loop,
   adaptive halt at eval, thinner middle rebalance.
9. **Implemented trainability (2026-07):** R curriculum (`--min-num-loops` → `--num-loops` over
   `--loop-curriculum-ratio`), loop health metrics in logger, checkpoint XOR compile
   (checkpoint default on), ×0.01 init scale on attn `o_proj` + FFN down for deep residual.

Note: linear R curriculum is the default train path; full always-max R via
`--loop-curriculum-ratio 0`.

### MLX inference performance

Full-depth synthetic 1.085B BF16 baseline on M1 Max, batch one and a 32-token
prompt: prefill took 1.02-1.18s, cached vanilla decode reached 1.10 tok/s, and
the four-head MTP proposal path reached 1.07 calls/s. Peak memory was 2.87 GiB
without MTP drafts and 3.37 GiB with them. Random weights cannot measure useful
speculative acceptance; repeat generated-token throughput on a trained 1B checkpoint.

Small-model diagnosis (2026-07-19, M1 Max): a trained 1.8M h96 probe
(`1+4×4+1` = 18 effective layers) only reached ~44–56 tok/s vanilla decode; a
10.5M h256 probe with the same looped depth stayed ~44 tok/s. Speed is nearly
flat in param count, so decode is architecture/launch-bound, not FLOP-bound.
Main costs: looped depth multiplies work; PaTH `incremental` re-runs full
`path_chunk` (triangular solve + local logits) on the open chunk every token;
~7 HBitLinear + 2 Linear per block (~126 ternary matmuls/token at 18 execs);
uncompiled eager graphs with per-token `mx.eval`; packed 2-bit matmul gated
off when `min(weight.shape) < 512`; production Infini/PaTH geometry
(`infini_memory_dim=64`, `path_window_size=64`) oversized for tiny probes.

- [x] **Persist inference ternary weights.** Each generation cache now retains effective or
  packed weights from first use; speculative branch clones share that immutable cache.
- [x] **Use the inference weight cache everywhere.** `prefill`, `inference_step`, and
  `inference_extend` now share one cache instead of re-scaling, thresholding, casting, and
  running fresh dense BF16 weights on every token and recurrent loop. Fully ternary checkpoints
  loaded by `mlx_generate.py` enable packed inference automatically.
- [x] **Fused ternary M=1 decode kernel.** `ternary_fused_linear_m1` (act quant + add/sub GEMV) and `ternary_fused_ffn_m1` (up/mid/down) in one Metal dispatch for token decode; wired into `MLXHBitLinear` / dense FFN when M=1 and fully quantized.
- [x] **Benchmark the 2-bit decode kernel at M=1.** An interleaved six-sample full-depth
  1.086B BF16 benchmark measured uncached dense at 1.23 tok/s, persistent dense at 3.71 tok/s,
  and persistent packed `mx.quantized_matmul` at 4.38 tok/s. The generic packed kernel wins;
  defer a custom Metal GEMV until profiling shows another kernel can recover meaningful time.
- [x] **True incremental PaTH decode.** Open-chunk T = S^{-1}D is **border-updated O(L²) per
  token** via `path_border_update_t` (not a full L×L re-solve). Last-query attention uses
  `path_chunk_last_with_t` without rebuilding the system. Prefill/extend seed T with one
  full solve for the residual open chunk. `--path-decode recompute` keeps the old full
  open-chunk baseline. Parity tests cover border T vs full solve and incremental vs recompute.
- [x] **Compile static incremental work.** Functionalized cache flatten/apply + `mx.compile` specialized per open-chunk length; enabled by default in `mlx_generate.py` (`--compile-step`). Falls back cleanly if compile fails. Functionalize cache arrays as explicit graph
  inputs/outputs and benchmark `mx.compile` around one physical block, recurrent loop, or
  full `inference_step`. A direct compiled closure fails today because mutable cache arrays
  are not explicit outputs. Biggest expected win for small hidden sizes where Metal launch
  and Python graph rebuild dominate FLOPs. Avoid one giant full-depth graph unless MLX
  compiler limits and compile latency prove acceptable.
- [x] **Eval-time loop override.** `mlx_generate.py --num-loops N` overrides checkpoint schedule; default remains curriculum/scheduled R. Expose `num_loops=1` (or lower R) for interactive
  generation; linear speedup vs full curriculum depth. Keep training/eval quality path at
  scheduled R.
- [ ] **Simpler generate-time FFN path.** Decode currently pays the full 3-mat dense FFN
  (`up` → square `mid` → `down`) plus PaTH extras every block. Consider a generate-only
  simplification or fused kernel if quality allows; otherwise fuse the three matmuls into
  fewer launches.
- [x] **Lower packed-kernel size gate / always-cache effective weights.** Removed `min(shape)>=512` gate; `pin_inference_weights()` materializes one dense/packed weight per HBitLinear for the generation lifetime (default on in generate). Packed ternary
  matmul only runs when `min(weight.shape) >= 512` and last dim % 32 == 0, so h96/h256
  probes never hit it and stay launch-bound dense GEMVs. Revisit the 512 gate, or ensure
  small-width decode always uses a single cached effective weight without per-call
  rematerialization overhead.
- [ ] **Scale Infini/PaTH geometry with model size.** Keep production `infini_memory_dim=64`
  and `path_window_size=64` for full models, but default smaller probes to matching dims /
  chunk widths so open-chunk solves and memory scores are not production-sized on toy nets.
- [x] **Instrument generate phase breakdown.** `mlx_generate.py --profile` reports prefill/decode/eval-sync and e2e tok/s. Add optional timing in `mlx_generate.py` /
  a small harness for prefill, `path_chunk`, linear/HBit, Infini, Engram, and `mx.eval` sync
  so future optimizations report end-to-end tok/s plus per-phase ms at 2M, 10M, and 1B.
- [x] **Batch speculative verification.** `inference_extend` already evaluates candidate
  spans together, converting several token-wise GEMVs toward GEMMs. Measure its net benefit
  and accepted tokens/target call once a trained 1B MTP checkpoint exists.

CMUD momentum, BF16 optimizer masters, backward fusion, and activation checkpointing are
training-only and do not apply to inference.

### MLX training performance

Priority order after current cache/MTP inference work:

- [x] **Reuse effective ternary weights across recurrent loops.** Small 32M-parameter tests
  were inconclusive, so the decision benchmark used the full 1.089B-parameter physical
  model at hidden 1024, `8 + 48×4 + 8` layers, sequence 64, and BF16. Two 20-step runs per
  mode measured median throughput of 69.35 tok/s normally and 73.38 tok/s with reuse
  (+5.8%). Loss and gradients match exactly; reuse is enabled by default.
- [x] **Benchmark a delayed loop-depth curriculum.** Full-1B phase measurements project
  141.31 effective tok/s for a 70–90% ramp versus 77.56 for the current 0–20% schedule
  (1.82x). Equal-token 48.33M sweeps measured 30–70% at 243.8s/PPL 20.51, 50–80% at
  218.6s/PPL 22.13, and 70–90% at 191.4s/PPL 23.14, versus the 0–20% baseline at
  285s/PPL 18.58. An equal-wall-clock 30–70% follow-up processed 770,048 tokens in
  265.8s and reached best PPL 17.73; 819,200 tokens in 303.2s reached PPL 16.64.
  Delayed start remains opt-in pending larger-scale/multi-seed confirmation.
- [x] **Quantize CMUD matrix momentum state to 8-bit.** At 1.089B scale, optimizer state
  dropped 7.68→5.03 GiB (-2.64 GiB), peak initialization memory dropped 9.98→7.28 GiB,
  and CMUD apply throughput improved 0.123→0.234 steps/s (1.91x). An equal-token 48.33M
  run measured 246.8s/PPL 18.91 versus FP32 momentum at 247.3s/PPL 18.67. Enabled by
  default for new MLX runs; full-1B sequence-256 CMUD training measured 5.11→18.24 tok/s
  (3.57x) with matching loss. `--no-cmud-momentum-8bit` restores FP32 momentum. Production
  still uses CMUD only. Follow-up batched independent 64-row whitening cut 48M CMUD time
  0.286→0.046s and raised end-to-end throughput 2,319→2,900 tok/s (+25%). At physical
  1.089B scale with one active loop, CMUD fell 2.405→0.893s and end-to-end throughput rose
  75.17→134.09 tok/s (+78%). The equal-token 48.33M run reached PPL 17.19 versus 18.91
  with 256-row blocks. New MLX runs therefore default to 64 rows; saved configs retain their
  original block size.
- [x] **Add selective MLX activation checkpointing.** `--gradient-checkpointing` now
  checkpoints only the repeated recurrent core by default; `--gradient-checkpoint-scope all`
  preserves full-stack behavior. At full 1.089B scale and sequence 256, no checkpointing,
  recurrent-only, and all-block scopes measured 13.257/12.434/12.490 GiB peak memory and
  19.56/26.80/24.34 tok/s. A repeated recurrent/no-checkpoint A/B under thermal slowdown
  remained faster at 20.70/17.76 tok/s. Recurrent-only therefore saves 0.82 GiB without
  paying needless prelude/coda recomputation; checkpointing remains opt-in.
- [x] **Build fused recurrent ternary Metal training kernel.** Packed 2-bit affine codes
  preserve ternary values; custom Metal packs weights and MLX `quantized_matmul` handles
  forward/input-gradient with STE weight gradients. Full-1B sequence-256 throughput improved
  44.19→55.38 tok/s (+25.3%). Equal-token 48.33M training improved 285→247.3s (1.15x)
  while validation PPL moved 18.58→18.67 (+0.5%). Enabled for new MLX runs at sequence ≥128;
  old checkpoints resume without it unless their saved arguments opted in.
- [x] **Profile phase costs before each optimization.** `mlx_train.py --profile-phases`
  reports per-step data, forward/backward, CMUD, and synchronization wait plus validation
  wall time; `mlx_benchmark.py --profile-phases` provides the same split without checkpoint
  writes. At steady-state 48.33M scale, data/forward-backward/CMUD were 0.011/2.823/0.289s
  and validation was 0.312s. Initial stream refills instead cost 2.6–3.1s. At full 1.089B
  scale with materialized gradients, forward/backward took 12.37s, CMUD 25.95s, validation
  5.66s, and peak memory reached 15.00 GiB. Synchronization wait overlaps compute phases
  because MLX executes lazy graphs during `mx.eval`; it was 3.03s at 48M and 37.32s at 1B.
  Future optimizations must report end-to-end throughput at both representative scales.
- [x] **Reject final-backward/CMUD fusion after measurement.** Exact accumulation, global
  clipping, and CMUD fusion preserved loss and optimizer state but reduced throughput from
  2,661→2,354 tok/s at 50.43M. A three-warmup, five-step physical-1.093B active-loop-1
  test measured 119.74 tok/s split versus 85.63 lazy-fused; fusion saved only 0.131 GiB peak.
  Monolithic compiled fusion was slower still, so production retains separate compiled graphs.
- [x] **Parallelise the Mamba-3 selective-scan backward.** Profiling the default config
  (52M, seq 1024) found the backward at 8.5x the forward, and disabling Mamba-3 removed 77%
  of it. Two causes, both in `_SCAN_BWD`: it reduced dB/dC on thread 0 alone — `2*N*P` serial
  float ops per timestep with `P-1` threads parked at a barrier, 4096 ops x 1024 steps per
  threadgroup — plus 8 barriers per timestep for three scalar reductions; and the per-thread
  `state`/`prev_bu` arrays were sized to the 128 worst case rather than the real `d_state`,
  so they spilled. Fixed by striping `n` over the P threads, collapsing the scalar reductions
  onto threads 0/1/2 behind one barrier pair, and templating `d_state` as a Metal
  compile-time constant (`NMAX`). Reduction order is unchanged, so gradients are
  bit-identical.

  Kernel microbenchmark at (1,1024,32,32,64), the clean measurement: scan forward
  14.66 -> 5.53 ms (2.65x), scan backward 219.47 -> 53.66 ms (**4.09x**). The state-history
  recompute is 5.0 ms and never was the bottleneck — the bwd kernel body was.

  End to end (3 interleaved A/B rounds, since this machine drifts): batch 1 fwd+bwd
  1,822.7 -> 732.5 ms (**2.49x**), batch 4 5,740 -> 3,023 ms (**1.90x**). Mamba-3 is 3 of 8
  layers, hence less than the kernel's own speedup. Batch 4 peaks at 20.3 GB on a 32 GB
  machine and the pre-fix version falls off that cliff under sustained load — 21,214 ms
  observed once — so worst-case improvement is far larger than 1.9x. Do not quote a
  single-run batch-4 number; it is not reproducible.
- [ ] **Deferred: use faster/distributed hardware for full 1B training.** Hardware changes
  are unavailable for now. Single M1 Max estimates are roughly 30–60 tok/s at 1B scale;
  revisit PyTorch/CUDA or distributed benchmarking when hardware access changes.

  Measured 2026-07-31, BLT byte front end + `MLXBitNetGlobalTransformer`, compiled step
  including the optimizer, at each size's best batch on a 32 GB M1 Max:

  | params | best batch | bytes/s | peak | 1 TB, one epoch |
  |---|---|---|---|---|
  | 51M | 16 | 11,252 | 15.2 GB | 2.8 years |
  | 227M | 2 | 2,096 | 8.1 GB | 15.1 years |
  | 522M | 8 | 1,241 | 23.6 GB | 25.6 years |
  | ~1B | — | ~700 (extrapolated) | — | ~45 years |

  With `use_mamba3_layers=False`, which swaps each Mamba-3 layer for a PaTH attention layer
  rather than removing it (51.0 -> 49.2M and 522.3 -> 484.9M params, so the layers are still
  there), same-session pairs: 51M 11,270 -> 16,965 B/s (**1.51x**), 522M 1,117 -> 1,587 B/s
  (**1.42x**). 1 TB drops to 1.9 and 20.0 years respectively. Smaller than the 1.73x the
  BitNet stack alone shows at batch 1, because BLT's byte-level encoder and decoder run no
  Mamba and dilute the share. No perplexity comparison exists either way — this is a pure
  speed measurement and Mamba-3 is on every third layer by design.

  Throughput at 522M by batch: 425 / 673 / 968 / 1,241 for 1 / 2 / 4 / 8 — still rising at
  batch 8 but only +28% for the last doubling, and 23.6 GB leaves no headroom, so ~1.4 KB/s
  is this machine's ceiling at that size.

  A 1 TB epoch is not the right target anyway. At ~4 bytes per patch, 1 TB is roughly 250B
  patches, i.e. Chinchilla-optimal (20 tokens/param) for a **~12B** model, not 1B. For 1B the
  matched budget is ~80 GB, which is still ~3.6 years here. What *is* feasible on this box:
  a 51M model at its Chinchilla budget (~4 GB) is a **4.2 day** run. 227M at ~18 GB is
  ~100 days. Everything above that needs different hardware — 1 TB at 1B in 30 days requires
  ~386 KB/s, about **550x** this machine.

#### BLT + BitNet throughput pass (2026-07-31)

Profiled the combined stack rather than the BitNet half. Forward at batch 16 x 1024 bytes
splits local encoder 124 ms / global BitNet 143 ms / local decoder 151 ms — **the byte-level
models are 67%**, because they run over 4x the positions the global model sees. Step phases:
forward 449 ms, backward 806 ms (1.8x, healthy), optimizer 120 ms, other overhead 12 ms.
No hidden pathology left; the remaining time is genuinely arithmetic.

- [x] **Fused straight-through activation quantiser** (`mlx_ternary_kernel.ste_activation_quantize`).
  `prepare_input` was ~9 elementwise passes over the activation tensor — absmax, divide,
  round, clip, rescale, and the `x + stop_gradient(q - x)` STE — costing **41% / 49% / 75%**
  of a 256->1024 / 1024->1024 / 1024->256 projection. Down-projections are worst because the
  pass count scales with the *input* width while the matmul scales with input x output; that
  layer measured **4.10x a plain matmul**. One Metal kernel, one threadgroup per row: 2-7x on
  the quantiser, and the down projection drops to 1.80x a plain matmul (5.37 -> 2.60 ms).
  Bit-identical for float32 and bfloat16 at every bit width, verified in
  `tests/test_mlx_fused_activation_quant.py`; float16 at 16 bits can differ by one
  quantisation step, a degenerate combination. Levels are a runtime input, not a template
  constant, so ramping the bit width does not recompile. Wired into both stacks.
- [x] **Chunked local sliding-window attention.** `local_window=256` at sequence 1024 built a
  dense [1024, 1024] bias and threw away 75% of the scores. Each query block now sees only the
  previous and current key blocks, folded onto the batch axis — one SDPA call, bit-identical
  output. Worth 1.4% end to end: real but small, since the SDPA is only 3.9 ms of a 30 ms block.
- [x] **Removed `use_ffn_mid` from both stacks; the square FFN mid is now unconditional.**
  It was briefly added to BLT to quantify the cost (below) and then taken out along with the
  BitNet stack's copy: the mid is load-bearing and the flag was one more architecture axis to
  keep two implementations honest about. The measurement stands as a record of what the mid
  costs, not as an option.

End to end, BLT + BitNet at 51M, batch 16, interleaved against master:

| configuration | bytes/s | vs master |
|---|---|---|
| master at session start | 7,790 | 1.00x |
| now, identical architecture | 11,907 | **1.53x** |
| (measured, then removed: no square FFN mid) | 13,621 | 1.75x |
| now, `use_mamba3_layers=False` (**now the default**) | 16,017 | 2.06x |
| now, both | 20,367 | **2.61x** |

The square FFN mid is unconditional -- the flag that produced the middle row was removed after
the measurement. A 200-step smoke on repeating text reached loss 2.279 with the mid and 2.457
without, so it earns its 16%. `use_mamba3_layers` now defaults to False in both stacks, so the
2.06x row is the shipped configuration; `--mamba3-layers` turns the SSM layers back on. No
quality comparison has been run in either direction -- that A/B is still open.

- [x] **Rejected: gathering the decoder cross-attention.** Implemented and measured at
  **0.85x** -- slower than the masked-dense path -- then reverted. The 3.7x headroom this file
  previously claimed was a bad comparison: it timed the full cross-attention layer (norms plus
  q/k/v/out projections) against a bare attention core with no projections. Isolated, the
  `take_along_axis` gathers for k and v cost 1.40 ms against 1.67 ms of dense scores, and the
  explicit sum/softmax over `[B, H, L, K, head_dim]` materialises 8.4M-element intermediates
  that fused SDPA never forms. The 0.391% mask density is real, it just does not convert:
  MLX's fused SDPA beats hand-written gather arithmetic at these sizes. Only worth revisiting
  as a single Metal kernel that gathers and attends in one pass.
- [ ] **`--recurrent-quantized-matmul` is likely wrong at small scale.** The packed 2-bit path
  loses to a dense matmul at every training token count measured: 1024->2048 bf16 runs
  0.96/0.81/0.87/0.93/0.55/0.38/0.61x dense at 1/16/64/256/1024/4096/16384 tokens, and
  1024->1024 never exceeds 0.85x. It only wins below ~16 tokens on narrow shapes, i.e. decode.
  On the 52M BitNet step, turning it off measured 599 -> 543 ms forward (**1.10x**) with
  identical peak memory. But this file records +25.3% for the same flag at full 1.089B and
  sequence 256, the opposite sign -- plausibly a memory effect, since packing shrinks the
  weights 16x and 1B is memory-bound on a 32 GB machine. Do not flip the default from the
  small-scale measurement alone; gate on token count and validate at 1B first.
- [x] **Hash n-gram embeddings added** (`blt/ngram_hash.py`, `blt/layers/ngram.py`,
  `MLXHashNgramEmbedding`). Meta's eq. 3 -- a rolling polynomial hash over the n-gram ending
  at each position, one table per n, summed into the byte embedding -- with Engram's narrow
  tables plus a projection instead of Meta's full-width tables, which at 500K x local_dim
  would be 128M parameters against a 51M model. Defaults n = 3..8, 16384 hashes per size,
  width `local_dim // 4`.

  Both stacks index the same tables, so `blt/ngram_hash.reference_ngram_hashes` is the
  executable spec and both are tested bit-for-bit against it. Every path -- training and both
  generation paths -- routes through a new `embed_bytes`, so the n-grams cannot be skipped at
  decode. Cost: +6.3M params (51.0 -> 57.3M) and 3.1% throughput. A 200-step smoke on
  repeating text reached 2.076 against 2.177 without, which is suggestive and nothing more.

  Two things this shook out. Eq. 3 has no normaliser; an earlier draft followed a summary that
  said "normalised by count of n-gram sizes plus one" and that measured as dividing the byte
  embedding by 7 (std 0.994 -> 0.204). And four generate tests were passing on a coin flip:
  they called `generate()` without `eos_id`, which falls back to `config.eos_id`, so whether
  an untrained model stopped early was luck that any architecture change re-tosses. They now
  pass `eos_id=-1` where they mean unbounded.
- [x] **Local decoder raised to 7 layers.** `n_layers_local_decoder` 4 -> 7, giving a 1/7
  split. Follows Meta's encoder-light/decoder-heavy budget (1/9 at 400M-1B, 3/7 at 2B-8B):
  the decoder turns a patch latent back into individual bytes and, unlike the encoder, has no
  hash n-gram embeddings to lean on. Costs +1.77M parameters and 193 vs 146 ms/step. Adopted
  on the published architecture, not on a local quality measurement -- held-out loss does not
  separate 1/4 from 1/7 at this scale (see the noise floor below).

- [x] **Local encoder cut to 1 layer.** `n_layers_local_encoder` 4 -> 1, matching Meta's
  400M/1B configuration now that hash n-gram embeddings are in. 3.4x on the encoder forward,
  1.19x end to end.

  A/B at 600 steps on the repo's own source as a byte corpus, 13-15M parameters, held-out
  loss on a 10% split -- runnable only after the NaN fix below, since 4 encoder layers used
  to die on the first update:

  | encoder / decoder | params | train | held-out | ms/step |
  |---|---|---|---|---|
  | 4 / 4 (old default) | 15.04M | 0.8811 | 3.0982 | 177 |
  | **1 / 4 (new default)** | 13.46M | 0.8688 | 3.0465 | **146** |
  | 1 / 7 (Meta's shape) | 15.23M | 0.8622 | 3.0786 | 193 |

  **These held-out numbers do not separate the configurations.** Re-running the 1/4 config at
  three seeds gave 3.0635 / 3.1909 / 3.0529 -- a spread of 0.138 against a 4-vs-1 gap of
  0.052. The speed column is solid; the quality column is noise. Ternary vs full precision at
  1 encoder layer is the same story: 3.0771 vs 3.0950, a 0.018 gap inside a 0.138 band.

  The cause is the corpus, not the seed: 1.07 MB against 600 x 8 x 512 = 2.4M bytes of
  training, so 2.3 epochs and heavy overfitting (train 0.87, held-out 3.05). A real
  comparison needs a corpus large enough to stay in one epoch, and several seeds. Until then
  the 1-layer encoder rests on Meta's architecture plus 1.19x measured throughput, and the
  honest statement is that no quality claim -- better *or* no worse -- is supported.

- [x] **FIXED: L2 norms with a NaN gradient at zero.** `mx.linalg.norm` differentiates to
  `x / ||x||`, which is 0/0 at the origin. The code guarded the *result* --
  `x / mx.maximum(mx.linalg.norm(x), eps)` -- which fixes the forward and does nothing for
  the backward: `maximum` multiplies the already-NaN cotangent by zero, and NaN survives
  that. Six sites in `mlx_model.py`, now `_safe_norm` / `_safe_normalize`, which put eps
  inside the square root.

  This was the divergence. Localised by dumping which parameter gradients went non-finite
  first: everything downstream of the global backbone was clean, `blocks.1` had NaN on
  exactly `path_down`, `path_up` and `path_conv_weight` while `qkv`, `out` and `path_beta`
  stayed finite -- and those three are precisely what forms the PaTH Householder vector in
  `_path_vectors`. A collapsed activation (encoder_patches std 2.6e-5) drove one of those
  vectors to exactly zero and the NaN propagated back through every parameter feeding it.
  Depth mattered only because more encoder layers made the collapse more likely.

  Also resolves `test_blt_mlx_optimization.py::test_training_survives_the_quantisation_ramp[True]`,
  which had been failing in full-suite order for the whole session with the same signature.

- [x] **FIXED: the identity FFN mid was not an identity in the forward pass.** Every site
  setting `mid.weight = eye(N)` -- torch and MLX `TernaryMLP`, `MLXHybridBlock.mid`, and both
  RFMoE experts -- was defeated by ternarisation. `effective_weight` scales each output
  channel by `mean(|row|)`, and an identity row is one 1 and N-1 zeros, so the scale is `1/N`
  and the quantised weight came out as `eye(N)/N`: a 1/1024 attenuator, not a pass-through.
  A signal through the mid arrived at std 0.000977 against 1.0 in.

  Fixed by initialising `eye(N) * N`, which makes `mean(|row|) = 1` so the quantised weight is
  exactly `eye(N)`, and pinning that layer's weight mix -- the straight-through blend only
  means anything when raw and quantised share a scale, and here they differ by N.

  Two follow-ons. The STE is now `stop_gradient(q) + (w - stop_gradient(w))` rather than
  `w + stop_gradient(q - w)`: same value, same identity gradient, but the old form computed
  `q - w` with `w ~ N` against `q ~ 1` and lost most of its significant digits. And three
  tests asserted the *raw* weight was `eye(N)`, which is what let this survive -- they now
  check the effective weight.

#### NanoGPT-speedrun audit

Full pass over modded-nanogpt's 86 records (KellerJordan/modded-nanogpt, read 2026-07-31).

**Already in this repo.** R2 rotary embeddings; R3/R4 Muon (as MUD, a triangular-whitening
variant, plus cautious masking); R5 zero-init output projections (`out`/`down` x 0.01) and
QK-norm; R10 bf16 activations; R13 attention-window warmup (`--initial-blocks` /
`--final-blocks`); R20 merged QKV weights and batched Muon (batched 64-row block whitening);
R53 multi-token prediction; R62 bigram hash embedding (Engram, a stronger hashed-n-gram
version); R73/R77 hyperconnections (Hyperloop); R11 U-net skip connections (approximately —
Kimi AttnRes does depth mixing over a residual stream).

**Adopted in this pass.** R8 untied embedding/head; R37 cross entropy at the logits' dtype
(3-4% on the step, measured isolated: 2403/2245 vs 2494/2308 ms; peak memory unchanged
because MLX frees the fp32 copy before the peak); R43/R50 Cautious Weight Decay; R19/R72
LR decays to a floor rather than to zero; R26 long cooldown fraction (their `cooldown_frac`
is 0.45-0.60, `--lr-schedule wsd` enforces >= 0.1 and documents ~0.4).

**Not applicable.** FP8 head and FP8 MLP up-projection (R19, R84) — no FP8 path on Metal and
the weights here are ternary. Triton kernels (R27, R59, R60, R79) — would be Metal kernels,
and the fused-FFN M=1 decode kernel already exists. Flash Attention 3 (R29). All the
distributed-communication records (R6, R22-24, R36, R71) — single device. PyTorch version
bumps.

**Worth doing, in value order:**

- [~] **NorMuon (R41, R42) — do not swap for CMUD; port its fix into MUD.** NorMuon
  (arXiv:2510.05491) and MUD (arXiv:2603.17970) change different stages: MUD replaces
  Muon's Newton-Schulz polar step with triangular whitening, NorMuon keeps Newton-Schulz and
  adds a per-neuron second-moment EMA rescale *after* it. They compose; they do not compete,
  and no head-to-head exists (the two papers' numbers are from different setups and are not
  comparable).

  Measured here at [2048, 512] on M1 Max: Newton-Schulz (5 steps) 2.93 ms, MUD block=64
  1.15 ms, MUD full-block 46.5 ms. So MUD is 2.5x cheaper than the step NorMuon builds on —
  moving to Muon to get NorMuon would pay 2.5x on the optimizer's core op — and MUD's entire
  speed advantage is the `block_size=64` default, not the algorithm.

  NorMuon's *premise* does hold here, and more strongly than for Muon. MUD makes exactly one
  axis uniform, and transposes tall matrices first, so for `[out, in]` weights with
  `out > in` the uniform axis is the input channels, not the neurons. Neuron-axis
  coefficient of variation, this model's shapes:

  | weight | shape | transposed | MUD | Muon |
  |---|---|---|---|---|
  | `qkv` | 1536x512 | yes | **0.0723** | 0.0291 |
  | `ffn up` | 2048x512 | yes | **0.0726** | 0.0311 |
  | `out` | 512x512 | no | 0.0000 | 0.0183 |
  | `ffn down` | 512x2048 | no | 0.0000 | 0.0138 |

  `--mud-neuron-norm` (default off) is the stateless form of the fix: re-normalise the
  neuron rows of the returned matrix. Drives the CV to 0 at no measurable cost and rotates
  the update by cos 0.9975. Whether it helps perplexity is unmeasured — it needs a training
  A/B, which is the open part of this item.
- [ ] **NorMuon proper on top of MUD ("C-NorMUD").** The above is per-step uniformity;
  NorMuon's actual mechanism is a second-moment EMA per neuron, so neurons with a
  historically large update get damped over time. Costs one vector per matrix. Only worth
  building if the stateless version above shows a signal.
- [ ] **Logit softcap (R9, R18, R54).** `23 * sigmoid((logits + 5) / 7.5)` in the current
  script. Previously dismissed here as redundant with `--z-loss-coef`, which was probably
  wrong: it entered at R9, was retuned at R18 and R54, and by R60/R79 they had written fused
  kernels for it — it stayed in the recipe for 70 records. z-loss penalises logit growth but
  does not bound it.
- [ ] **Tie early, untie at 2/3 of training (R51, R53).** The current head-tying default here
  (untied from step 0) matches R8-R50. Their later refinement re-ties (R51) and then splits at
  2/3 through the run, copying optimizer state from `lm_head` to `embed` at the split. Needs a
  mid-run parameter and optimizer-state transition, so it is real work, not a flag.
- [ ] **Per-head-pair orthogonalisation of Q/K (R80).** They stopped whitening the full
  6-head Q/K matrix and switched to pairs of heads. Here `qkv` is one fused
  `[3*hidden, hidden]` and MUD whitens it in 64-row blocks that straddle head boundaries
  arbitrarily. Aligning `--mud-block-size` to `head_dim` (32) or `2*head_dim` (64, which is
  the current default and happens to align at head_dim 32) is a no-code experiment.
- [ ] **ReLU^2 MLP (R5).** Two matmuls instead of SwiGLU's three (here four, with the square
  `mid`). A real FLOP cut, but the identity-initialised `mid` is a deliberate design here,
  so this is an architecture A/B rather than a swap.
- [ ] **Value embeddings (R14-R17, R55, R63, R65, R70).** Seven records' worth of tuning and
  one of their largest architectural wins: per-token learned V injected as an alternative
  value stream, adding capacity without FLOPs. Needs token ids, so for BLT it belongs in the
  byte-level encoder rather than the patch-level global model.
- [ ] **Batch-size schedule (R46) and max_seq_len schedule (R72).** Cheap curriculum knobs;
  this repo already has loop-count, block-count and quantisation curricula to hang them on.
- [ ] **Drop the first MLP layer (R30) and the first attention layer (R35).** Two separate
  records, each a straight win. Costs nothing to try given the prelude/recurrent/coda split.
- [ ] **Update the elementwise optimizer only every other step (R39).** Halves CLion work.
- [ ] **Smear token embeddings one position forward (R34).** Very cheap, but Engram already
  supplies a much stronger version of the same signal — low expected value here.

Superseded detail from the earlier pass:

- [x] **Untie the LM head from the embedding and give the embedding its own rate.**
  Default is now untied (`tie_word_embeddings=False`, `--no-tie-word-embeddings` on the MLX
  trainer); `MLXBitNet.logits_from` is the single place that knows which. CMUD gained a
  third group so the token-indexed tables — input embedding, loop embedding, Engram tables
  and the untied `lm_head` — run at `--embedding-learning-rate` while MUD keeps the body,
  matching the speedrun's split. The torch side (`split_parameters_for_cmud`) routes
  `lm_head.weight` the same way, and `mlx_convert` maps PyTorch's single C-Lion group across
  the two MLX groups. Costs 16.8M params at vocab 32768 / hidden 512 and ~nothing in step
  time: 2,405 vs 2,829 ms fwd+bwd at batch 4, peak 20.31 vs 20.27 GB. Checkpoint
  compatibility deliberately not preserved — see the README note.
- [ ] **Sweep `--embedding-learning-rate`.** Wired but still defaults to `--learning-rate`,
  which is the pre-split behaviour and leaves the untied head's main benefit on the table.
  modded-nanogpt runs embeddings well above the body rate; 10-30x is the range to try.
- [ ] **Decide the attention-window curriculum direction.** `_chunk_bounds` sets
  `block_width = length // num_blocks`, so the default `--initial-blocks 8 --final-blocks 16`
  *shrinks* the local window 128 -> 64 over training. The speedrun grows it (cheap early,
  long-context late). More chunks does mean more Infini memory writes, so this may be
  deliberate — but if it is not, swapping the two flags is free and both faster early and
  better late.
- [ ] **Momentum warmup in MUD (R9, 0.85 -> 0.95).** Small speedrun win. Needs momentum to
  become a traced `mx.array` like `learning_rate` so `apply_step` stays compiled, plus a
  `float()` in `checkpoint_config`. Skipped as speculative for a whitening optimizer.

### BLT as the main model

- [x] **Resolved: the NaN was a missing quantisation ramp, not CMUD.** Bisected to 4-bit
  *activation* quantisation from a cold start: at `activation_mix=1.0` from step 0 the
  model collapses to uniform output after one update (loss lands exactly on ln(vocab))
  and NaNs on the next, at every width, learning rate and optimizer variant tried.
  Ternary weights are innocent -- `weight_mix=1.0` with `activation_mix=0.0` trains fine.
  `mlx_train.py` had always ramped (`--stage1-activation-mix-start 0.0`); the BLT trainer
  did not, and BLT's own `MLXHBitLinear` had no `set_quantization_state` at all, so it
  could not be ramped. Added the knobs to match `layers/h_bitlinear.py`, plus
  `MLXTernaryBLTModel.set_quantization_state` and a ramp in `MLXBLTTrainer`. Both the
  plain and BitNet global backbones now train under CMUD.
- [x] **Resolved by 8-bit activations.** The residual instability was specific to 4-bit.
  Measured on the BitNet backbone at 8 patches over 64 bytes: 4-bit reaches NaN once the
  ramp completes, 8-bit trains (5.674 -> 5.561). Costs nothing: activation quantisation
  here is fake (`x + stop_gradient(q - x)`), the tensor stays float and the matmul is
  float x ternary regardless, so bit width only sets the rounding grid. Benchmarked at
  86M, batch 4, sequence 512: 4-bit 354.6ms/step, 8-bit 345.4ms, 16-bit 348.3ms, no
  quantisation 345.9ms -- and on the forward alone, skipping quantisation entirely is
  1.29x (110.3ms -> 85.5ms). `TernaryBLTConfig.activation_bits` now defaults to 8.
- [ ] **Decide the BitNet stack's activation width separately.** `TernaryConfig` still
  defaults to 4 and `mlx_train.py --final-activation-bits` to 4, unchanged here because
  that stack has trained checkpoints whose behaviour would shift. The same argument
  applies -- 4-bit buys no training speed -- so it is worth switching unless inference
  really runs 4-bit activation kernels (`ternary_fused_linear_m1`, the M=1 decode path,
  is the only place low-bit activations become real compute).
- [ ] **Padding is not inert for the BitNet backbone.** Zero-length patches perturb it:
  measured drift 0.0 / 1.4e-3 / 2.2e-1 at 1 / 2 / 8 layers, the last a relative error of
  1.0. It does not grow with the amount of padding, so a small perturbation is being
  amplified through depth by the 4-bit activation quantisation (a step function -- one
  flipped bucket cascades). Fixing the PaTH block width does not help, so the cause is
  elsewhere in the block, most likely the AttnRes stream mixing across layers. Worked
  around by patching to a fixed count (`patches_per_sequence`) so no padding exists;
  worth finding the real cause, since it also means the backbone's output depends on
  sequence length in a way it probably should not.
- [ ] **Hash n-gram embeddings in the local encoder.** Engram cannot follow BLT into the
  global model -- it hashes token n-grams and patches have no ids. Meta's BLT puts hash
  n-gram embeddings in the local *encoder*, over bytes, which is the level at which
  n-grams exist, and this repo has none. Closest existing code is `MLXEngram`.
- [ ] **Deduplicate HBitLinear.** Three implementations: `layers/h_bitlinear.py` (torch),
  `mlx_model.py` (BitNet MLX), `blt/mlx_layers.py` (BLT MLX). The two MLX ones use
  identical quantisation maths -- verified numerically -- but BitNet's adds weight
  pinning, packed ternary kernels and quantisation-state ramping that BLT's lacks. BLT
  should adopt it and inherit the fast paths; the blocker is that it takes an
  `MLXBitNetConfig` and reads `activation_bits`, which `TernaryBLTConfig` has no field
  for. The two MLX transformer blocks are *not* redundant: BLT's plain SwiGLU block is
  right for the local encoder/decoder, where PaTH and Infini would not be.

### BLT generation performance

- [ ] **Deferred: evaluate BLT-S self-speculation once a BLT student is trained.** BLT-S
  (Kallini et al., *Fast Byte Latent Transformer*, arXiv:2605.08044, §5.1) lets the local
  decoder draft past patch boundaries against the last available latent; one full forward
  re-patches the candidate and accepts bytes up to the first mismatch. Under greedy decoding
  the output is byte-identical to plain autoregressive decoding, so it costs no quality —
  the paper reports the same BLEU/pass@1 to two decimals at every window k, at both 1B and
  3B. It is inference-only: no architectural change, no retraining. `blt/generate.py`
  implements it behind `speculation_window`; `GenerationStats` reports the per-component
  forward counts needed to judge it. **Blocked on a trained `StudentEntropyModel`, not on
  code.** The whole saving is skipped global-model passes, and the baseline already runs the
  global model only once per patch, so the win collapses as patch length grows. With the
  current 9.38M encoder / 67.12M global / 9.64M decoder split the ceiling is
  `patch < 1.14·(k+1)`: k=8 can win only below patch 10, k=4 only below 5.7. At the
  `patch_size=4` default (lowered from 6 on 2026-07-30 to sit nearer Meta's ~4.5-byte
  teacher) best case at 100% acceptance is 31% for k=8 and 40% for k=16, with break-even
  acceptance at 66% / 58% against the 91% / 77% the paper measures. Finer still would help:
  at patch 2 the ceiling is 54% and break-even drops to 40%. So this is worth revisiting
  only if the trained student patches at roughly 2–4 bytes, BLT's actual design point. Note large patches and BLT-S are substitutes, not complements — both buy speed by
  running the global model less, and coarser patches pay for it in quality. Measure
  `GenerationStats.bytes_per_global_pass` and `acceptance_rate` on the trained model rather
  than assuming these projections hold.
- [ ] **Add a KV cache to the BLT decode path.** `blt/generate.py` re-runs the encoder and
  decoder over the whole prefix for every drafted byte, so it is O(L²) and the BLT-S
  projections above are ceilings it cannot currently reach. Also batch size 1 only:
  verification accepts a different count per row, so batching needs ragged bookkeeping.
  Both matter before any wall-clock claim — at ~86M parameters generation is launch-bound
  rather than memory-bandwidth-bound, so the paper's bandwidth metric may not translate.
- [ ] **Deferred: BLT-D / BLT-DV block diffusion.** Same paper, §3 and §5.2. Much faster than
  BLT-S (up to 86% bandwidth reduction at block 16) but requires retraining with a block
  diffusion objective and costs real quality: at 1B, best-setting D-8 loses 14% BLEU on
  FR→EN, 20% on DE→EN, and 38–39% pass@1 on HumanEval/MBPP; D-16 loses up to 52%. DV
  recovers part of it and is not even monotonic (DV-4 scores below D-4 on HumanEval). Table 1
  shows 3–8 points on the likelihood benchmarks before decoding starts. Not worth it unless
  generation speed becomes the binding constraint and the quality loss is acceptable.

---

## RFMoE design reference

Ref: RFMoE (arXiv 2604.00801), on AoE (2501.13074) + ReMoE.

Per expert body (3 mats, depth matches dense hybrid FFN):  
`E_i(x) = W_down(silu(W_mid([σ(x A_gate B_gate) ⊙ (x W_up)])))`.  
Expand gate is sigmoid-GLU (paper), not SwiGLU; mid/down are shared depth with dense up→mid→down.  
`A_gate` D×r dual-use (score + gate).
- Score `s_i = ‖x A_gate‖₂`; gate `G_i = ReLU(s_i − b_i)`; fire `1{G_i ≥ θ}`.
- `z = x A_gate` computed ONCE: norm decides, same z feeds B_gate. Skip path = FLOP saving
  (skip W_up/W_mid/W_down + B_gate when not firing).
- Combine `h = x + Σ G_i·E_i` — NO divide-by-count (score = mix weight; RMSNorm renormalizes).
- Sizing `r ≈ D/16`. Decision-dedicated params = scalar `b_i` + global `θ`.
- Expert tensors: `{A_gate, B_gate, W_up, W_mid, W_down, b_i}`.
- Train: pre-threshold `G_i` = differentiable proxy. Bias warmup `b_i≈1e-6` (all fire early),
  λ ramps sparsity. GLOBAL density target, not per-layer.
- Gains (paper, ≤0.8B): PPL −12–19%, θ gives 20× fewer acts / −31% FLOPs. UNPROVEN >0.8B. Research bet.

## Locality reg reference

Load-balance = data-center assumption. Local wants concentration: hot set VRAM-resident, tail offloads.
Target = staircase (uniform within memory tier, step down across): `π = (1−α)·Zipf(s) + α·Uniform(1/N)`.
- Head Zipf → hot-tier ordering; tail uniform → floor π_i ≥ α/N keeps cold experts alive.
- Knobs: `s` head skew (→ hot-set size), `α` tail floor. Set s so top-M (VRAM-fitting) mass ≥ ~0.95.
- Loss `λ_loc·KL(π ‖ sort_desc(p))`. p = usage EMA. KL(π‖p) direction → ∞ as p_i→0 (forbids dead expert).
- On RFMoE: bias `b_i` IS usage control; locality shapes b_i distribution; appended experts → high bias → cold.
- Curriculum: flat early (α≈1) so whole population trains, anneal to skew → cold experts competent but rare.

## Extensible MoE reference (thread 2)

Why self-gating enables it: standard router `G∈D×N` bakes in N — adding expert N+1 renormalizes ALL
routing (softmax over N), breaks load balance, needs router retrain. RFMoE has no central router: append
`{W_up, W_mid, W_down, A_gate, B_gate, b_i}`, existing fire decisions unchanged, residual-add preserves old behavior when frozen.

Procedure, after base training: append with low b_i so the new expert receives gradients → freeze
everything else → train it on a new domain → push into an unclaimed niche (diversity term) → raise
b_i into the cold tier → re-tune θ so density stays pinned (else cost grows with N).

Hard problems: niche-finding (bias too high=dead, too low=duplicates) is THE problem; density drift with N;
joint-optimality loss vs from-scratch (frozen olds can't co-adapt); new-expert under-training (train flat
first, then cold); needs task boundaries (online append unsolved).

## Diffusion + MoE (thread 3) — locality reg is LOAD-BEARING

Why: local inference (batch=1, idle parallel compute) → diffusion beats AR on latency (T≪N parallel
denoising steps vs N sequential). Mercury 2 ~1000 tok/s; LLaDA 2.0 = MoE+diffusion @100B; DiffusionGemma
26B-A4B. Production block size 128–256 (not ~32 ablation number).

The conflict — per-step expert union: MoE offload needs tiny active set. Diffusion runs whole block
in parallel → union touched ≈ `1−(1−k/E)^N`. AR 1 tok k=8 E=64 → 12.5% (offload works). 256-block →
≈100% (even E=256 → ≈99.97%). ⇒ ~all experts every step → sparse working set GONE → naive load-balanced
MoE+diffusion is WORST local combo (memory-bound on expert weights, offload thrashes).
- Survives: per-token FLOP saving (each token runs k experts). MoE HELPS diffusion (LLaDA 2.0 proof).
- Dies: memory/bandwidth/offload win — whole point of locality section.

The rescue: concentrated routing shrinks union — if ~90% mass on small hot set, parallel tokens
mostly pick SAME experts → union ≈ hot set → offload viable. So locality reg MORE valuable for
diffusion than AR. Bonus: across T denoising steps of ONE block it's same tokens → working set stable
→ load hot set once/block, amortize over T (pairs with temporal-stickiness loss).

Levers/risks: smaller block = smaller union but more sequential steps (measure curve); block-level routing
(all block tokens share expert set) kills per-token specialization; early denoising steps = noisy routing;
concentration NOT optional for local diffusion+MoE.

Full local stack (only composes if routing concentrated): diffusion (cut steps) + locality MoE (working
set resident despite parallelism) + ternary/low-bit (cut bandwidth).

## Prior art

- RFMoE 2604.00801 (github.com/liuyilun2000/RoutingFreeMoE); AoE 2501.13074; ReMoE.
- Aux-loss-free balancing: DeepSeek 2408.15664 (bias-nudge, retarget uniform→staircase).
- Offload: Mixtral-offloading, MoE-Infinity, Fiddler, Pre-gated MoE, EdgeMoE/AdapMoE. Objective: 2512.09277.
- Continual/extensible: Lifelong-MoE 2305.12281, MoE-Adapters 2403.11549 (CVPR24), R²MoE 2507.13107
  (redundancy/niche), CP-MoE 2605.20247, LLaVA-CMoE, TRGE 2508.07738, MoTE. Mostly router-patch, not
  pure self-gating — our angle under-explored.
- Diffusion: SEDD, MDLM/MD4, D3PM (absorbing >> uniform); BD3-LM (block: AR-over-blocks + diffusion-within,
  recovers KV-cache); LLaDA/LLaDA 2.0, DiffuLLaMA/DiffuGPT, Mercury 2; Fast-dLLM, dKV-Cache, d²Cache;
  distillation Di[M]O/T3D/CDLM. Masking schedule = inference algorithm (sets T = latency).
- Data efficiency (why diffusion/any-order learn more per token): Super Data Learners 2511.03276,
  What Makes DLMs SDL 2510.04071; any-order prior XLNet 1906.08237, u-PMLM 2004.11579; data-constrained
  scaling 2305.16264 / 2606.06888. MTP: 2404.19737.
