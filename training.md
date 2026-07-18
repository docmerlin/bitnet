# Training Curriculum

## Goal

Train a general language model with strong coding ability by expanding from clean,
short, common patterns into composition, technical knowledge, reasoning, and
repository-scale work. Difficulty should rise gradually, but foundational data must
remain in every phase to prevent forgetting and abrupt distribution shifts.

This is a competence-based curriculum, not a set of hard grade levels. Phase changes
should follow validation results rather than token count alone.

## Principles

- Start with clean and learnable data, not artificial toy language alone.
- Introduce difficult material gradually while replaying earlier material.
- Prefer examples with useful loss: neither already memorized nor overwhelmingly hard.
- Track difficulty separately by length, rarity, syntax, reasoning depth, context
  distance, and noise. One scalar cannot describe every kind of difficulty.
- Preserve document and repository boundaries when packing sequences.
- Keep base pretraining, instruction tuning, and evaluation data separate.
- Compare any curriculum against a shuffled-data baseline using the same data, token
  budget, optimizer schedule, and seeds.

Relevant foundations include curriculum learning (Bengio et al., 2009), self-paced
learning (Kumar et al., 2010), competence-based curricula (Platanios et al., 2019),
mastery learning, interleaving, spaced practice, and the zone of proximal development.

## Base-Training Schedule

Target about 35-40% code over the complete run. Code rises from short, well-documented
examples to complete files and repository-level relationships.

| Training progress | Natural text | Educational/reference | Math/reasoning | Standalone code | Repository code |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-15% | 45% | 25% | 10% | 20% | 0% |
| 15-50% | 35% | 20% | 15% | 25% | 5% |
| 50-85% | 25% | 15% | 15% | 30% | 15% |
| 85-100% | 25% | 15% | 15% | 25% | 20% |

Treat these ratios as starting points. Adjust them using held-out loss and skill probes,
not training loss alone. Earlier categories never fall to zero.

### Phase 1: Foundations

- Clean, short prose with common vocabulary and punctuation.
- Simple factual statements, questions, stories, and instructions.
- Basic arithmetic, comparison, classification, and symbolic transformations.
- READMEs, API documentation, standard-library examples, short functions, basic tests,
  JSON, YAML, SQL, shell, and configuration files.

### Phase 2: Composition

- Paragraphs, coreference, temporal and causal relations, and multi-step instructions.
- Introductory textbooks, science, mathematics, and programming material.
- Complete source files, implementation/test pairs, docstring-to-code, code explanation,
  algorithms, and data structures.

### Phase 3: Reasoning

- Multi-step mathematics, logic, debugging, verification, and error correction.
- Technical documents and longer code contexts.
- Related files, imports, tests, issues, patches, and commit messages.

### Phase 4: Long Context and Repositories

- Document-level retrieval with distant dependencies and distractors.
- File trees followed by selected documentation, source files, and tests.
- Cross-file symbol use, build failures, API migrations, refactoring, and security review.
- Long code and prose contexts that exercise PaTH-FoX and Infini memory.

## Public Data Sources

Public availability does not guarantee permission to train. Record source, version,
license, and filtering decisions. Recheck dataset cards before each download.

| Role | Dataset | Use |
| --- | --- | --- |
| General language | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Primary web source; begin with highest education scores. |
| General language | [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | High-quality filtered web baseline. |
| Reproducible mixture | [Dolma](https://huggingface.co/datasets/allenai/dolma) | Transparent source mixture and processing. |
| Small-model mixture | [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Convenient curated source for smaller runs. |
| Foundational prose | Wikipedia and Simple English Wikipedia | Clean factual prose; Simple English fits early phases. |
| Elementary narrative | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Early supplement only; synthetic style must not dominate. |
| Textbook prose | [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Synthetic educational supplement. |
| Open textbooks | OpenStax and Wikibooks | Structured educational progression; retain attribution. |
| Mathematics | [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Primary mathematical web source. |
| Mathematics | OpenWebMath | Broader math source; clean markup and duplicates carefully. |
| Formal reasoning | [Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) | Specialized late-phase material. |
| Code | [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Use license-filtered files and respect opt-outs. |
| Instruction tuning | [Tulu 3 SFT Mix](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | Post-training only. |
| Instruction tuning | [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Post-training for smaller assistants. |
| Dialogue | [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Human dialogue for post-training. |

For a minimal first base run, use FineWeb-Edu, Wikipedia, FineMath, and a permissively
licensed subset of The Stack v2. Add OpenStax and TinyStories only as early supplements.

## Code Curriculum

Default language allocation inside the code portion:

- 25% Python
- 20% JavaScript and TypeScript
- 15% C and C++
- 10% Java
- 10% Go and Rust
- 10% SQL and shell
- 10% HTML, CSS, configuration, and other languages

Shift this mix toward intended use. For a Python-first model, 40-50% Python is
reasonable.

Use fill-in-the-middle on 30-50% of code sequences beginning in phase 2, while keeping
ordinary left-to-right prediction. Fill-in-the-middle better matches completion,
editing, and patch generation.

Repository examples should retain meaningful structure:

```text
repository metadata
file tree
relevant documentation
source files
tests
```

Do not concatenate unrelated repositories into fake projects. Filter vendored
dependencies, generated files, minified JavaScript, lockfiles, build outputs, data
blobs, secrets, credentials, and near-duplicate forks.

## Post-Training

After base pretraining, use a coding-focused instruction mixture:

- 30% code generation
- 20% debugging and repair
- 15% test generation
- 10% repository navigation
- 10% explanation and code review
- 15% general instruction following

Favor executable solutions with tests over prose-only coding conversations.

## Advancement Criteria

Advance difficulty when:

- Validation loss for the current tier plateaus.
- Earlier-tier validation remains stable.
- The next tier produces finite, useful loss rather than persistent outliers.
- Skill probes show transfer beyond memorized examples.
- Long-context retrieval, code completion, and repository probes improve without a
  material regression in general language quality.

If earlier-tier performance regresses, increase replay before adding more advanced data.
If a sample has extreme loss, inspect it for corruption or excessive difficulty rather
than automatically emphasizing it.

## Evaluation and Contamination

Keep evaluation sets out of all training and deduplicate against prompts and solutions.
Reserve at least:

- General language: MMLU, ARC, HellaSwag
- Mathematics: GSM8K and selected held-out mathematical probes
- Code: HumanEval, MBPP, and LiveCodeBench
- Repository work: SWE-bench
- Long context: LongBench and custom PaTH-FoX retrieval probes

Track results by curriculum tier, domain, context length, and code language. Also track
forgetting after each mixture transition. A curriculum succeeds only if it beats the
shuffled baseline at equal tokens and compute, or reaches equivalent quality sooner.

## First Experiment

1. Build three pools: foundational, core, and advanced.
2. Start with phase-1 ratios and transition smoothly rather than swapping datasets.
3. Keep at least 15-20% foundational replay throughout training.
4. Run one shuffled-mixture baseline and one curriculum run with matched settings.
5. Evaluate every pool plus held-out language, math, code, and long-context probes.
6. Change ratios only after measurements identify a bottleneck.

## Pre-Run Result: 2026-07-12

A bounded MPS pilot verified that the proposed early mixture learns and established a
rough local throughput range. This was a pipeline and learning-slope measurement, not a
quality result or curriculum-versus-shuffle comparison.

Pilot configuration:

- Apple M1 Max, 32 GiB unified memory, PyTorch MPS, FP32
- 0.25M parameters, 2 layers, hidden size 64, sequence length 64
- PaTH-FoX window 16, Engram enabled, 2-expert RFMoE
- 2,048-token vocabulary and 8,192 training tokens
- 45% FineWeb-Edu, 25% DCLM, 20% CodeSearchNet, 10% FineMath
- Validation: 45% FineWeb-Edu, 30% CodeSearchNet, 25% FineMath
- Two 64-token validation batches every 1,024 training tokens

| Tokens | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 1,024 | 6.835 | 930.3 |
| 2,048 | 5.413 | 224.3 |
| 3,072 | 4.213 | 67.6 |
| 4,096 | 3.693 | 40.2 |
| 5,120 | 3.507 | 33.4 |
| 6,144 | 3.428 | 30.8 |
| 7,168 | 3.389 | 29.6 |
| 8,192 | 3.367 | 29.0 |

Loss fell rapidly through about 4,000 tokens, then gains diminished. The run completed
training in 89.8 seconds, including frequent validation, for 91 effective tokens/second;
steady training steps usually reported 300-500 tokens/second. The validation sample is
too small for model-selection decisions, but the monotonic curve confirms basic
trainability.

The repository's local profile was also measured at its real startup scale:

- 47.28M parameters, hidden size 512, sequence length 512
- Effective depth 8 at curriculum start and 20 at maximum loop count
- One-step measurements: approximately 123-153 tokens/second

At that rate, the configured 10M-token local run takes roughly 19-23 hours before
evaluation, checkpoint, and data stalls. It remains a shakeout run: 10M tokens are only
0.21 tokens per parameter. A rough compute-optimal floor of 20 tokens per parameter is
about 946M tokens, or 73-91 uninterrupted days at measured MPS speed. Meaningful
convergence at this scale therefore requires faster hardware, distributed training, or a
smaller model. Measure sustained throughput over at least 100 steps before scheduling a
long run; one-step MPS timings have substantial startup and synchronization variance.

The pilot artifacts are under ignored directories `runs/curriculum-prerun/`,
`runs/local-scale-benchmark/`, and `runs/local-max-depth-benchmark/`. Training completed
and checkpoints were written. The Python 3.14 shutdown hang found during the pilot was
traced to threaded PyArrow parquet reads entering Python-backed HTTPS files during
interpreter finalization. The trainer now uses synchronous parquet batches on Python 3.14;
other Python versions retain the threaded path. A full MPS training/checkpoint smoke test
now exits normally.

## MLX Training-Step Benchmark: 2026-07-13

`mlx_benchmark.py` ports the dominant dense block path to MLX: ternary weight and
activation STE, dense Hadamard preprocessing, exact chunked PaTH-FoX attention,
sandwich RMSNorm, SwiGLU FFN, and tied embeddings. It intentionally excludes Engram,
Infini state writes, RFMoE, and loop HC so the first measurement isolates the dense
training path. Both backends run the same dimensions and 100 optimizer steps over one
fixed synthetic batch; loss values are not quality or convergence measurements.

Small launch-bound shape (`2` layers, hidden `64`, sequence `64`, vocabulary `2,048`,
PaTH window `16`):

| Backend | Precision | Tokens/second | Relative to PyTorch |
| --- | --- | ---: | ---: |
| PyTorch MPS | FP32 | 1,877 | 1.00x |
| MLX compiled | BF16 | 4,208 | 2.24x |

Local-profile shape with two representative blocks (hidden `512`, sequence `512`,
vocabulary `32,768`, PaTH window `64`, 24.22M parameters):

| Backend | Precision | Tokens/second | Relative to PyTorch |
| --- | --- | ---: | ---: |
| PyTorch MPS | FP32 | 5,091 | 1.00x |
| MLX compiled, substitution fallback | BF16 | 2,552 | 0.50x |
| MLX compiled, custom Metal solve | BF16 | 13,532 | 2.66x |

MLX has no GPU `solve_triangular`, which PaTH-FoX uses once per attention window. The
initial port's exact compiled forward substitution dominated at window 64. The custom
kernel in `mlx_path_kernel.py` assigns one Metal thread to each right-hand-side column and
performs dependency-ordered substitution over rows. Its custom VJP solves against `A^T`
and applies the analytical matrix gradient, so it supports training rather than forward
timing only. Forward and gradient parity pass at `1e-4` tolerance.

The kernel path is about 5.3x faster than the MLX fallback and makes this representative
MLX path 2.66x faster than PyTorch MPS. The reusable model also uses MLX's native orthonormal
Hadamard transform instead of a dense matrix multiplication.

Reproduce the realistic measurements:

```bash
python3 mlx_benchmark.py --backend mlx --steps 100 --warmup-steps 5 \
  --sequence-length 512 --vocab-size 32768 --hidden-size 512 \
  --num-heads 16 --intermediate-size 1024 --num-layers 2 --path-window-size 64

python3 mlx_benchmark.py --backend torch --steps 100 --warmup-steps 5 \
  --sequence-length 512 --vocab-size 32768 --hidden-size 512 \
  --num-heads 16 --intermediate-size 1024 --num-layers 2 --path-window-size 64
```

### MLX Port Status

Implemented:

- `mlx_model.py`: ternary blocks, native Hadamard transform, custom Metal PaTH solve,
  packed-document masking, Engram, Infini memory, grouped sparse RFMoE dispatch through
  conditional Metal kernels, four-stream Hyperloop HC, prelude/recurrent/coda execution,
  and MTP heads.
- `mlx_rfmoe_kernel.py`: conditional grouped expert projections plus sparse custom
  input/weight VJPs. The default hybrid path uses one host compaction, compact
  `gather_mm` forwards, and compact route-wise backward kernels.
- `mlx_train.py`: streaming Hugging Face mixtures through the existing tokenizer and
  packer, compiled BF16 gradients and optimizer updates, activation checkpointing,
  gradient accumulation, CE/z/MTP/RF auxiliary losses, quantization/loop/block/RF/data/LR
  curricula, validation, JSONL metrics, and resumable safetensor model/optimizer
  checkpoints. Resume restores MLX RNG, mixture RNG, Hugging Face iterator positions,
  shuffle buffers, and partial packed sequences.
- `mlx_generate.py`: vanilla and MTP speculative greedy generation. MTP proposals use only
  the final hidden position, verification accepts only target-model argmax matches, and
  generated tokens are therefore identical to vanilla greedy decoding.
- `mlx_optim.py`: 64-row blockwise C-MUD for non-embedding matrices and blockwise-int8
  C-Lion for embeddings, norms, biases, and gates, including cautious masking, Metal
  triangular whitening, independent learning rates, optional int8 CMUD matrix momentum,
  and resumable optimizer state.
- MLX defaults to four 4-sequence microbatches per optimizer update and leaves activation
  checkpointing off. Sampled MTP defaults to depth 4; pass `--mtp-depth 0` to disable it.
  Use smaller microbatches with `--gradient-checkpointing` on memory-constrained machines.
  Override blockwise whitening with `--mud-block-size`; converted and legacy checkpoints
  preserve their original full-matrix C-MUD behavior.
- Five 4-sequence validation batches preserve the previous default sample count while
  avoiding the 4x validation expansion caused by larger microbatches. Validation batches
  are materialized once so repeated evaluations do not rescan the held-out data offset.
- Whole-gradient compilation remains limited to block counts that divide sequence length.
  Irregular layouts run eagerly because retaining each compiled curriculum graph eventually
  exhausts unified-memory headroom. Any remaining MLX argument-buffer failure automatically
  retries that layout eagerly.
- All-feature FineWeb-Edu smoke exercised Engram, Infini safety, RFMoE, Hyperloop, MTP,
  two-microbatch accumulation, validation, best/numbered/last/final saves, and resumed at
  the next optimizer step with finite loss.

Remaining differences are implementation and training-trajectory parity, not missing
model features:

- RFMoE preserves independent self-gating: every expert scores every token and every pair
  above `theta` executes. Default `--rfmoe-backend auto` selects `hybrid` when Metal is
  available and `host` otherwise. Hybrid was fastest in local RFMoE and one/two-block
  measurements, `metal` keeps static shapes and activation recomputation for lower-memory
  compiled runs, and `host` retains native `gather_mm` backward for parity checks and
  inference.
- Multi-host distributed training is not implemented.
- Converted PyTorch and older MLX checkpoints have no dataset stream state, so their first
  MLX resume warns and restarts data. Checkpoints subsequently written by `mlx_train.py`
  resume exactly.

At hidden size 512, 8 experts, expert width 256, 512 tokens, BF16, and 25% active
density, RFMoE layer measurements were:

| Backend | Forward | Forward + backward |
| --- | ---: | ---: |
| Host | 3.3 ms | 244.7 ms |
| Static Metal | 18.6 ms | 33.5 ms |
| Hybrid host-forward/Metal-backward | 3.9 ms | 9.1 ms |

Hybrid training beat static Metal by 2.99x at 12.5% density, 3.86x at 50%, and 4.05x
at 100%. Hybrid retains RFMoE intermediates instead of activation-checkpointing that
sublayer, trading memory for speed.

With 256-row C-MUD blocks and 16 accumulated 512-token microbatches, the full dense MLX
trainer measured 2,760 tokens/second at curriculum depth 8 and 1,304 tokens/second at
maximum depth 20. These synthetic-step figures include activation recomputation, gradient
clipping, and optimizer updates, but exclude data loading, validation, and checkpoints.

On the same M1 Max, fixed maximum-depth real-data steps measured 1,297 tokens/second with
the original one-sequence, 16-accumulation checkpointed defaults. Disabling activation
checkpointing reached 1,711; four-sequence microbatches with four accumulations reached
2,594 tokens/second with matching short-run losses and gradient norms. Eight-sequence
microbatches added only about 4% with less memory headroom, while 16 sequences exceeded
practical unified-memory capacity. The four-by-four configuration is the new default.
Compiling irregular curriculum layouts added 17-30% in short probes but regressed sustained
training through retained-graph memory pressure, so they remain eager. FP16 added only
about 2% and changed the loss trajectory immediately, so BF16 remains the default.
A 100-update run with the new defaults finished at training loss 2.7600 and averaged 2,999
tokens/second over maximum-depth checkpoints, versus 2.7644 and 1,306 tokens/second for
the previous defaults. Validation loss is not directly comparable because the new run
evaluated 20 sequences instead of the earlier five-sequence A/B sample.

Effective ternary-weight reuse must be measured at production parameter scale. A 32M-
parameter proxy was inconclusive, while the full 1.089B-parameter physical model at hidden
1024, `8 + 48×4 + 8` layers, sequence 64, and BF16 showed a repeatable gain. Two 20-step
runs per mode measured median throughput of 69.35 tokens/second normally and 73.38 with
forward-scoped recurrent weight reuse, a 5.8% speedup with identical loss. Reuse is enabled
by default and recomputed each forward so optimizer updates remain visible. Reproduce the
A/B with `mlx_benchmark.py` using `--num-prelude-layers 8 --num-layers 48
--num-coda-layers 8 --num-loops 4`; add `--reuse-recurrent-weights` for the cached run.

Recurrent dense projections also use a packed 2-bit Metal path once the ternary curriculum
reaches full weight quantization. `mlx_ternary_kernel.py` fuses row-scale reduction and
ternary packing; MLX's native `quantized_matmul` handles forward and input-gradient while a
custom VJP keeps the STE weight gradient. On the full 1.089B physical model at sequence 256,
two 10-step BF16 measurements reached 55.38 versus 44.19 tokens/second, a 25.3% speedup.
At sequence 64 packing overhead instead caused a 4.6% regression, so `mlx_train.py` enables
the path only for sequence lengths at least 128. An equal-token 48.33M sampled-MTP run
finished in 247.3 versus 285 seconds (1.15x) with validation perplexity 18.67 versus 18.58.
New MLX runs enable it by default; use `--no-recurrent-quantized-matmul` for strict old
arithmetic. Checkpoints written before the flag existed resume with the old path.

CMUD matrix momentum is stored blockwise-int8 by default; this does not introduce a
standalone MUD training path. On the full 1.089B model, optimizer state
fell from 7.68 to 5.03 GiB and initialization peak memory from 9.98 to 7.28 GiB. Three
compiled CMUD-only apply steps improved from 0.123 to 0.234 steps/second. A seeded
48.33M-parameter equal-token run showed no end-to-end speed gain and moved validation
perplexity from 18.67 to 18.91. Use `--no-cmud-momentum-8bit` to restore FP32 momentum.
Legacy optimizer configs omit `mud_eight_bit` and retain FP32 momentum when resumed.
At full 1.089B scale with sequence 256, three measured end-to-end CMUD steps improved from
5.11 to 18.24 tokens/second (3.57x) with matching loss; FP32 state pushed the 32 GiB M1 Max
into severe unified-memory pressure.

The dominant CMUD operation was blockwise triangular decorrelation: 22.46 ms of a 23.76 ms
`2048×1024` matrix update. Reducing independent whitening blocks from 256 to 64 rows and
batching all block solves into one Metal launch preserved exact blockwise arithmetic while
cutting common `1024×1024` and `2048×1024` decorrelation to 0.98 and 1.20 ms. At the 48M
benchmark shape, CMUD fell from 0.286 to 0.046 seconds and end-to-end throughput rose from
2,319 to 2,900 tokens/second (+25%). A physical-1.089B, active-loop-1 A/B kept optimizer
size identical while limiting thermal interference: CMUD fell 2.405→0.893 seconds and
end-to-end throughput rose 75.17→134.09 tokens/second (+78%). Full-depth sequential 1B
runs remained thermally unstable on the M1 Max.

Smaller blocks did not hurt the short quality check. The equal-token 48.33M run reached
validation perplexity 17.19 and training loss 3.644, versus 18.91 and 3.758 with 256-row
blocks. Its 274.6-second wall time is excluded from speed comparison because it ran after
multiple full-1B thermal stress tests. New MLX runs default to `--mud-block-size 64`; resumed
checkpoints reconstruct their saved optimizer block size.

MLX activation checkpointing can target only the repeated recurrent core. Enable it with
`--gradient-checkpointing`; use `--gradient-checkpoint-scope all` only when prelude and coda
activations also need recomputation. On the full 1.089B model at sequence 256, a three-step
CMUD sweep measured:

| Checkpoint scope | Peak memory | Throughput |
| --- | ---: | ---: |
| None | 13.257 GiB | 19.56 tok/s |
| Recurrent | 12.434 GiB | 26.80 tok/s |
| All blocks | 12.490 GiB | 24.34 tok/s |

A repeated recurrent/no-checkpoint A/B after thermal slowdown still favored recurrent-only
at 20.70 versus 17.76 tokens/second, while peak memory repeated exactly. Full-stack
checkpointing provided no additional measured saving and needlessly recomputed the one-pass
prelude and coda. Checkpointing remains opt-in because smaller models and batches may not
benefit. Checkpoints saved before the scope flag retain their prior all-block behavior.

Phase profiling is opt-in with `--profile-phases` in either `mlx_train.py` or
`mlx_benchmark.py`. Trainer logs average data, forward/backward, CMUD, and synchronization
wait per step since the previous log, plus validation wall time on evaluation lines.
Synchronization wait is a subset of forward/backward and CMUD time, not an additive phase:
MLX executes most lazy graph work while `mx.eval` waits for completion.

At the 48.33M shape, a three-step real-data run at maximum loop depth and full quantization
reached steady-state data/forward-backward/CMUD times of 0.011/2.823/0.289 seconds and
2,622 tokens/second. Validation took 0.312 seconds. The first two steps waited 2.6–3.1
seconds on stream refill, demonstrating that data stalls can dominate short measurements
even though buffered data was only 0.4% of the steady-state training step.

At the full 1.089B shape with sequence 256, batch one, packed recurrent matmuls, and int8
CMUD momentum, the materialized-gradient profile measured 12.37 seconds for
forward/backward, 25.95 seconds for CMUD, and 5.66 seconds for one validation forward.
End-to-end training reached 6.68 tokens/second with a 15.00 GiB peak. Synchronization wait
was 37.32 seconds of the 38.32-second step and overlaps both compute phases. This is slower
than the fused-step benchmark because profiling follows the trainer's separate gradient and
optimizer evaluations and retains the full gradient tree. At 1B scale, CMUD and gradient
materialization are the next software bottlenecks; at 48M scale, forward/backward dominates
once the input stream is warm.

The delayed loop-depth curriculum is available through
`--loop-curriculum-start-ratio`; the existing `--loop-curriculum-ratio` remains the point
where maximum depth is reached. On the full 1.089B physical model, BF16 sequence-64 phase
measurements at active loop depths 1 through 4 were 184.08, 122.93, 92.81, and 72.77
tokens/second. Weighting those costs by a 70–90% delayed ramp projects 141.31 effective
tokens/second versus 77.56 for the default 0–20% ramp, a 1.82x compute-throughput gain.

Equal-token quality did not support making any delayed schedule the default. Seeded
48.33M-parameter, 655,360-token sampled-MTP sweeps measured:

| Loop ramp | Time | Speedup | Final validation PPL | PPL change |
| --- | ---: | ---: | ---: | ---: |
| 0–20% (default) | 285.0s | 1.00x | 18.58 | baseline |
| 30–70% | 243.8s | 1.17x | 20.51 | +10.4% |
| 50–80% | 218.6s | 1.30x | 22.13 | +19.1% |
| 70–90% | 191.4s | 1.49x | 23.14 | +24.5% |

The 30–70% ramp is the least harmful measured speed/quality tradeoff, but remains opt-in:

```bash
python3 mlx_train.py \
  --loop-curriculum-start-ratio 0.3 \
  --loop-curriculum-ratio 0.7
```

Equal wall-clock comparison favored that 30–70% ramp. The 0–20% baseline processed
655,360 tokens in 285 seconds and reached validation perplexity 18.58. A 30–70% run
processed 770,048 tokens in 265.8 seconds and reached best measured perplexity 17.73;
a second bracketing run processed 819,200 tokens in 303.2 seconds and reached 16.64.
Thus the delayed ramp lost quality per token but improved quality per local training time
in this single-seed 48.33M experiment. It remains opt-in until this result survives larger
models and multiple seeds. Later sequential sweeps thermally throttled the M1 Max and were
excluded from wall-clock comparison.

An 80-update MTP sweep used identical data, seed, curriculum, and a fixed four-sequence
validation sample every 10 updates. Depths 0 through 4 reached perplexity 25 in 173, 189,
221, 247, and 264 training seconds. Depth 2 reached perplexity 23.5 ten updates earlier
than depth 0, but still took 221 versus 204 seconds. At 80 updates, validation perplexities
were 18.97, 19.23, 18.76, 19.06, and 18.70 while training times were 267, 294, 345, 381,
and 396 seconds. Exact all-head MTP improved some equal-token results but not
time-to-perplexity in this single-seed short run, so the sampled policy below replaces it.

The cheaper MTP trainer rotates one future head through each accumulation microbatch. At
depth 4 this reduces auxiliary vocabulary projections from 16 to 4 per optimizer update
without changing the expected depth-mean loss or checkpoint tensors. A maximum-depth probe
reached 2,293 tokens/second with a 13.2 GiB peak footprint, versus 1,718 tokens/second and
16.5 GiB for exact depth 4. An 80-update sampled run finished in 285 seconds at validation
perplexity 18.58, compared with 271 seconds and 18.97 without MTP and 401 seconds and 18.70
for exact depth 4. Generation fixes PaTH chunk width to the final training layout so logits
remain prefix-invariant during verification. On that sampled checkpoint, exact greedy
speculation generated the same 128 tokens as vanilla decoding at 22.27 versus 16.95
tokens/second, a 1.31x speedup, while reducing target-model calls from 128 to 96. This
full-prefix reference has no incremental PaTH/Infini cache yet.

A 100-update real-data A/B with identical seed, stream, curriculum, and five validation
batches every 20 updates compared 256-row blocks against full-matrix whitening. Blockwise
C-MUD finished with validation loss 2.7053 versus 2.7601, sustained 1,306 versus 1,118
tokens/second at maximum depth, and completed in 1,078 versus 1,181 seconds. This single
seed supports the faster default but is not a statistical convergence study.

Use `mlx_train.py` for native MLX experiments; do not expect identical step-by-step loss
to `train.py` because backend kernels and RFMoE execution order differ.

Convert a PyTorch C-MUD checkpoint, including optimizer momentum and exact Engram hash
multipliers, before resuming with MLX:

```bash
python3 mlx_convert.py runs/bitnet/checkpoints/last.pt \
  --output-dir runs/mlx_bitnet --name imported
python3 mlx_train.py \
  --output-dir runs/mlx_bitnet \
  --resume-from runs/mlx_bitnet/checkpoints/imported.safetensors
```

Checkpoints created before optimizer parameter names were saved require
`--allow-legacy-optimizer-order`. Use that flag only for an unmodified checkpoint
written by this repository.
