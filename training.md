# Training Curriculum

## Goal

Train general LM w/ strong coding: clean short patterns → composition, technical knowledge, reasoning, repo-scale work. Raise difficulty gradual; keep foundational data every phase — prevent forget + abrupt distribution shift.

Competence-based curriculum, not hard grade levels. Phase change follow validation, not token count alone.

## Principles

- Start clean learnable data, not toy language alone.
- Introduce hard material gradual; replay earlier material.
- Prefer useful loss: not already memorized, not overwhelmingly hard.
- Track difficulty separate: length, rarity, syntax, reasoning depth, context distance, noise. One scalar not enough.
- Preserve document + repo boundaries when packing.
- Keep base pretrain, instruction tune, eval data separate.
- Compare curriculum vs shuffled-data baseline: same data, token budget, optimizer schedule, seeds.

Foundations: curriculum learning (Bengio et al., 2009), self-paced learning (Kumar et al., 2010), competence-based curricula (Platanios et al., 2019), mastery learning, interleaving, spaced practice, zone of proximal development.

## Dense FFN middle layer (square mid)

Default dense FFN = three stages, not classic two-mat SwiGLU alone:

1. **up** — expand `H → 2I` (fused gate + value for SwiGLU)
2. **mid** — **square** projection `I → I` (same width in/out)
3. **down** — project `I → H`

Mid intentionally square: intermediate width fixed between up post-SwiGLU features and down. Disable w/ `--no-use-ffn-mid` (MLX) for traditional two-mat; RFMoE experts same square mid (`w_mid`).

### Cold start from scratch: identity on the square mid

**Start train from scratch** (no checkpoint): square mid **master weight = identity matrix**, not random Kaiming/uniform.

- **Why:** Random mid scramble expand features under ternary STE. Identity keep 3-mat near classic 2-mat: `silu(I @ h)` = mild pointwise nonlinearity until train move mid off `I`.
- **Where:** dense `ffn_mid` / MLX `mid`, RFMoE `w_mid`, BLT `mid_proj` (PyTorch + MLX). Helpers: `training/arch_upgrade.py` (`copy_square_identity_`, `init_all_ffn_mid_identity`).
- **Resume / soft upgrade:** missing mid tensors → fill identity (`init_missing_ffn_mid_identity`); 2-mat warm start not blow when third stage added.
- **Do not** re-randomize mid after model build on from-scratch run. Trained ckpts keep learned mid; only cold start + missing-key upgrades force `I`.

Short equal-token A/B: random mid ~48M lose bad vs 2-mat SwiGLU; identity cold-start ~100M bring 3-mat near equal-param 2-mat final PPL. Identity mid at step 0 = training contract, not optional tweak.

## Base-Training Schedule

Target ~35-40% code over full run. Code rise short well-documented examples → complete files + repo-level relations.

| Training progress | Natural text | Educational/reference | Math/reasoning | Standalone code | Repository code |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-15% | 45% | 25% | 10% | 20% | 0% |
| 15-50% | 35% | 20% | 15% | 25% | 5% |
| 50-85% | 25% | 15% | 15% | 30% | 15% |
| 85-100% | 25% | 15% | 15% | 25% | 20% |

Ratios = starting points. Adjust via held-out loss + skill probes, not train loss alone. Earlier categories never → zero.

### Phase 1: Foundations

- Clean short prose, common vocab + punctuation.
- Simple facts, questions, stories, instructions.
- Basic arithmetic, comparison, classification, symbolic transforms.
- READMEs, API docs, stdlib examples, short functions, basic tests, JSON, YAML, SQL, shell, config files.

### Phase 2: Composition

- Paragraphs, coreference, temporal/causal relations, multi-step instructions.
- Intro textbooks, science, math, programming.
- Complete source files, impl/test pairs, docstring-to-code, code explain, algorithms, data structures.

### Phase 3: Reasoning

- Multi-step math, logic, debug, verification, error correction.
- Technical docs + longer code contexts.
- Related files, imports, tests, issues, patches, commit messages.

### Phase 4: Long Context and Repositories

- Document-level retrieval w/ distant deps + distractors.
- File trees → selected docs, source, tests.
- Cross-file symbols, build fails, API migrations, refactor, security review.
- Long code/prose contexts exercise PaTH-FoX + Infini memory.

## Public Data Sources

Public availability ≠ train permission. Record source, version, license, filter decisions. Recheck dataset cards each download.

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

Minimal first base run: FineWeb-Edu, Wikipedia, FineMath, permissive Stack v2 subset. Add OpenStax + TinyStories only as early supplements.

## Code Curriculum

Default language alloc inside code portion:

- 25% Python
- 20% JavaScript and TypeScript
- 15% C and C++
- 10% Java
- 10% Go and Rust
- 10% SQL and shell
- 10% HTML, CSS, configuration, and other languages

Shift mix toward intended use. Python-first model → 40-50% Python OK.

Fill-in-the-middle on 30-50% code sequences from phase 2; keep ordinary LTR prediction. FIM better match completion, edit, patch gen.

Repo examples keep meaningful structure:

```text
repository metadata
file tree
relevant documentation
source files
tests
```

Do not concat unrelated repos into fake projects. Filter vendored deps, generated files, minified JS, lockfiles, build outputs, data blobs, secrets, credentials, near-duplicate forks.

## Post-Training

After base pretrain, coding-focused instruction mix:

- 30% code generation
- 20% debugging and repair
- 15% test generation
- 10% repository navigation
- 10% explanation and code review
- 15% general instruction following

Favor executable solutions w/ tests over prose-only coding chat.

## Advancement Criteria

Advance difficulty when:

- Validation loss for current tier plateau.
- Earlier-tier validation stay stable.
- Next tier produce finite useful loss, not persistent outliers.
- Skill probes show transfer beyond memorized examples.
- Long-context retrieval, code completion, repo probes improve w/o material general-language regression.

Earlier-tier regress → increase replay before more advanced data. Extreme sample loss → inspect corruption/excessive difficulty; do not auto-emphasize.

## Evaluation and Contamination

Keep eval sets out of all train; dedupe vs prompts + solutions. Reserve at least:

- General language: MMLU, ARC, HellaSwag
- Mathematics: GSM8K and selected held-out mathematical probes
- Code: HumanEval, MBPP, and LiveCodeBench
- Repository work: SWE-bench
- Long context: LongBench and custom PaTH-FoX retrieval probes

Track by curriculum tier, domain, context length, code language. Track forget after each mixture transition. Curriculum succeed only if beat shuffled baseline at equal tokens + compute, or reach equal quality sooner.

## First Experiment

1. Build three pools: foundational, core, advanced.
2. Start phase-1 ratios; transition smooth, not swap datasets.
3. Keep ≥15-20% foundational replay throughout.
4. Run one shuffled-mixture baseline + one curriculum run, matched settings.
5. Eval every pool + held-out language, math, code, long-context probes.
6. Change ratios only after measurements find bottleneck.

## Pre-Run Result: 2026-07-12

Bounded MPS pilot: early mixture learns + rough local throughput. Pipeline + learning-slope measure, not quality or curriculum-vs-shuffle.

Pilot config:

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

Loss fall fast through ~4,000 tokens, then gains diminish. Run 89.8s incl frequent val → 91 effective tok/s; steady steps often 300-500 tok/s. Val sample too small for model-select; monotonic curve confirm basic trainability.

Repo local profile at real startup scale:

- 47.28M parameters, hidden size 512, sequence length 512
- Effective depth 8 at curriculum start and 20 at maximum loop count
- One-step measurements: approximately 123-153 tokens/second

At that rate, 10M-token local run ~19-23h before eval/ckpt/data stalls. Shakeout only: 10M tokens = 0.21 tok/param. Rough compute-optimal floor 20 tok/param ≈ 946M tokens → 73-91 uninterrupted days at measured MPS speed. Meaningful convergence need faster HW, distributed train, or smaller model. Measure sustained throughput ≥100 steps before long run; one-step MPS timings have big startup + sync variance.

Pilot artifacts under ignored dirs `runs/curriculum-prerun/`, `runs/local-scale-benchmark/`, `runs/local-max-depth-benchmark/`. Training complete + ckpts written. Python 3.14 shutdown hang: threaded PyArrow parquet reads enter Python-backed HTTPS during interpreter finalization. Trainer now sync parquet batches on Python 3.14; other Pythons keep threaded path. Full MPS train/ckpt smoke now exit normal.

## MLX Training-Step Benchmark: 2026-07-13

`mlx_benchmark.py` ports dominant dense block path to MLX: ternary weight + activation STE, dense Hadamard preprocess, exact chunked PaTH-FoX attention, sandwich RMSNorm, SwiGLU FFN, tied embeddings. Intentionally exclude Engram, Infini state writes, RFMoE, loop HC — isolate dense train path. Same dims + 100 optimizer steps over one fixed synthetic batch; loss not quality/convergence measure.

Small launch-bound shape (`2` layers, hidden `64`, sequence `64`, vocabulary `2,048`, PaTH window `16`):

| Backend | Precision | Tokens/second | Relative to PyTorch |
| --- | --- | ---: | ---: |
| PyTorch MPS | FP32 | 1,877 | 1.00x |
| MLX compiled | BF16 | 4,208 | 2.24x |

Local-profile shape, two representative blocks (hidden `512`, sequence `512`, vocabulary `32,768`, PaTH window `64`, 24.22M parameters):

| Backend | Precision | Tokens/second | Relative to PyTorch |
| --- | --- | ---: | ---: |
| PyTorch MPS | FP32 | 5,091 | 1.00x |
| MLX compiled, substitution fallback | BF16 | 2,552 | 0.50x |
| MLX compiled, custom Metal solve | BF16 | 13,532 | 2.66x |

MLX no GPU `solve_triangular` (PaTH-FoX once per attention window). Initial port exact compiled forward substitution dominate at window 64. Custom kernel `mlx_path_kernel.py`: one Metal thread per RHS column, dependency-ordered row substitution. Custom VJP solve vs `A^T` + analytical matrix gradient → train, not forward-only timing. Forward + grad parity pass at `1e-4` tolerance.

Kernel path ~5.3x faster than MLX fallback; representative MLX path 2.66x faster than PyTorch MPS. Reusable model use MLX native orthonormal Hadamard, not dense matmul.

Reproduce realistic measurements:

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

- `mlx_model.py`: ternary blocks, native Hadamard, custom Metal PaTH solve, packed-document mask, Engram, Infini memory, grouped sparse RFMoE via conditional Metal kernels, four-stream Hyperloop HC, prelude/recurrent/coda, MTP heads, optional dense square mid (`use_ffn_mid`; cold-start mid master = identity), classic 2-mat SwiGLU when mid off.
- `mlx_rfmoe_kernel.py`: conditional grouped expert projections + sparse custom input/weight VJPs. Default hybrid: one host compaction, compact `gather_mm` forwards, compact route-wise backward kernels.
- `mlx_train.py`: stream HF mixtures via existing tokenizer/packer, compiled BF16 grads + optimizer updates, activation ckpt, grad accum, CE/z/MTP/RF aux losses, quantization/loop/block/RF/data/LR curricula, val, JSONL metrics, resumable safetensor model/optimizer ckpts. Resume restore MLX RNG, mixture RNG, HF iterator positions, shuffle buffers, partial packed sequences.
- `mlx_generate.py`: vanilla + MTP speculative greedy. MTP proposals use final hidden position only; verification accept only target-model argmax matches → generated tokens = vanilla greedy.
- `mlx_optim.py`: 64-row blockwise C-MUD for non-embedding mats + blockwise-int8 C-Lion for embeddings/norms/biases/gates; cautious mask, Metal triangular whitening, independent LRs, optional int8 CMUD matrix momentum, resumable optimizer state.
- MLX default: four 4-sequence microbatches per optimizer update; activation ckpt off. Sampled MTP default depth 4; `--mtp-depth 0` disable. Smaller microbatches + `--gradient-checkpointing` on memory-tight machines. Override whitening `--mud-block-size`; converted/legacy ckpts keep original full-matrix C-MUD.
- Five 4-sequence val batches keep previous default sample count; avoid 4x val expansion from larger microbatches. Val batches materialize once — repeated eval no rescan held-out offset.
- Whole-gradient compile only when block counts divide sequence length. Irregular layouts run eager: retained compiled curriculum graphs exhaust unified-memory headroom. Remaining MLX argument-buffer failure auto-retry that layout eager.
- All-feature FineWeb-Edu smoke: Engram, Infini safety, RFMoE, Hyperloop, MTP, two-microbatch accum, val, best/numbered/last/final saves, resume next optimizer step w/ finite loss.
- From-scratch: every square FFN mid (`mid` / `w_mid`) → identity matrix; see **Dense FFN middle layer (square mid)** above. `--use-ffn-mid` / `--no-use-ffn-mid` select 3-mat vs 2-mat dense FFN.

Remaining diffs = impl + train-trajectory parity, not missing features:

- RFMoE keep independent self-gating: every expert score every token; every pair above `theta` execute. Default `--rfmoe-backend auto` → `hybrid` when Metal available else `host`. Hybrid fastest local RFMoE + one/two-block measures; `metal` keep static shapes + act recompute for lower-memory compile; `host` keep native `gather_mm` backward for parity + inference.
- Multi-host distributed train not implemented.
- Converted PyTorch / older MLX ckpts lack dataset stream state → first MLX resume warn + restart data. Ckpts later written by `mlx_train.py` resume exact.

At hidden size 512, 8 experts, expert width 256, 512 tokens, BF16, 25% active density, RFMoE layer measures:

| Backend | Forward | Forward + backward |
| --- | ---: | ---: |
| Host | 3.3 ms | 244.7 ms |
| Static Metal | 18.6 ms | 33.5 ms |
| Hybrid host-forward/Metal-backward | 3.9 ms | 9.1 ms |

Hybrid train beat static Metal 2.99x at 12.5% density, 3.86x at 50%, 4.05x at 100%. Hybrid keep RFMoE intermediates (no act-ckpt that sublayer) — trade memory for speed.

256-row C-MUD blocks + 16 accumulated 512-token microbatches: full dense MLX trainer 2,760 tok/s at curriculum depth 8, 1,304 tok/s at max depth 20. Synthetic-step figures include act recompute, grad clip, optimizer updates; exclude data load, val, ckpts.

Same M1 Max, fixed max-depth real-data steps: 1,297 tok/s w/ original one-sequence, 16-accum checkpointed defaults. Disable act ckpt → 1,711; four-sequence microbatches × four accum → 2,594 tok/s, matching short-run loss + grad norms. Eight-sequence microbatches +~4% w/ less memory headroom; 16 sequences exceed practical unified-memory. Four-by-four = new default. Compile irregular curriculum layouts +17-30% short probes but regress sustained train via retained-graph memory pressure → stay eager. FP16 +~2% only, change loss trajectory immediately → BF16 stay default. 100-update run new defaults: train loss 2.7600, avg 2,999 tok/s over max-depth ckpts vs 2.7644 and 1,306 tok/s previous defaults. Val loss not directly comparable (new run 20 sequences vs earlier five-sequence A/B sample).

Effective ternary-weight reuse measure at production param scale. 32M-param proxy inconclusive; full 1.089B physical model hidden 1024, `8 + 48×4 + 8` layers, sequence 64, BF16 → repeatable gain. Two 20-step runs/mode: median 69.35 tok/s normal, 73.38 w/ forward-scoped recurrent weight reuse (+5.8%, identical loss). Reuse default on; recompute each forward so optimizer updates stay visible. Reproduce A/B w/ `mlx_benchmark.py` using `--num-prelude-layers 8 --num-layers 48 --num-coda-layers 8 --num-loops 4`; add `--reuse-recurrent-weights` for cached run.

Recurrent dense projections use packed 2-bit Metal path once ternary curriculum reach full weight quant. `mlx_ternary_kernel.py` fuse row-scale reduction + ternary packing; MLX native `quantized_matmul` forward + input-grad; custom VJP keep STE weight grad. Full 1.089B physical model sequence 256: two 10-step BF16 → 55.38 vs 44.19 tok/s (+25.3%). Sequence 64 packing overhead → 4.6% regression, so `mlx_train.py` enable path only seq ≥128. Equal-token 48.33M sampled-MTP: 247.3 vs 285s (1.15x), val PPL 18.67 vs 18.58. New MLX runs enable by default; `--no-recurrent-quantized-matmul` for strict old arithmetic. Ckpts before flag resume old path.

CMUD matrix momentum blockwise-int8 by default; not standalone MUD train path. Full 1.089B: optimizer state 7.68 → 5.03 GiB, init peak mem 9.98 → 7.28 GiB. Three compiled CMUD-only apply steps 0.123 → 0.234 steps/s. Seeded 48.33M equal-token: no end-to-end speed gain; val PPL 18.67 → 18.91. `--no-cmud-momentum-8bit` restore FP32 momentum. Legacy optimizer configs omit `mud_eight_bit` → FP32 momentum on resume. Full 1.089B sequence 256: three end-to-end CMUD steps 5.11 → 18.24 tok/s (3.57x), matching loss; FP32 state push 32 GiB M1 Max into severe unified-memory pressure.

Dominant CMUD op = blockwise triangular decorrelation: 22.46 ms of 23.76 ms `2048×1024` matrix update. Cut independent whitening blocks 256 → 64 rows + batch all block solves one Metal launch: preserve exact blockwise arithmetic; common `1024×1024` + `2048×1024` decorrelation → 0.98 + 1.20 ms. 48M bench shape: CMUD 0.286 → 0.046s, end-to-end throughput 2,319 → 2,900 tok/s (+25%). Physical-1.089B active-loop-1 A/B keep optimizer size identical, limit thermal interference: CMUD 2.405→0.893s, end-to-end 75.17→134.09 tok/s (+78%). Full-depth sequential 1B runs stay thermally unstable on M1 Max.

Smaller blocks not hurt short quality check. Equal-token 48.33M: val PPL 17.19, train loss 3.644 vs 18.91 / 3.758 w/ 256-row blocks. 274.6s wall excluded from speed compare (ran after multiple full-1B thermal stress tests). New MLX default `--mud-block-size 64`; resumed ckpts reconstruct saved optimizer block size.

New MLX runs: BF16 MUD master weights, C-Lion masters stay FP32; `--cmud-master-dtype float32` restore FP32 MUD masters. Seeded 400-update, 3,276,800-token 48.33M A/B end 20-batch val loss/PPL 1.87602/6.52745 BF16 masters vs 1.87144/6.49765 FP32 masters (0.46% PPL regression). BF16 cut physical-1.089B active-loop-1 peak 10.57 → 8.80 GiB, matching five-step loss. Sequential runtime thermally unstable → memory save not speed supports default. Saved optimizer configs w/o dtype field retain FP32 masters on resume.

MLX act ckpt can target only repeated recurrent core. Enable `--gradient-checkpointing`; `--gradient-checkpoint-scope all` only if prelude+coda also need recompute. Full 1.089B sequence 256, three-step CMUD sweep:

| Checkpoint scope | Peak memory | Throughput |
| --- | ---: | ---: |
| None | 13.257 GiB | 19.56 tok/s |
| Recurrent | 12.434 GiB | 26.80 tok/s |
| All blocks | 12.490 GiB | 24.34 tok/s |

Repeated recurrent/no-ckpt A/B after thermal slowdown still favor recurrent-only 20.70 vs 17.76 tok/s; peak mem repeat exact. Full-stack ckpt no extra measured saving; needless recompute one-pass prelude+coda. Ckpt remain opt-in — smaller models/batches may not benefit. Ckpts before scope flag retain prior all-block behavior.

Phase profiling opt-in `--profile-phases` in `mlx_train.py` or `mlx_benchmark.py`. Trainer log avg data, forward/backward, CMUD, sync wait per step since previous log + val wall on eval lines. Sync wait subset of forward/backward + CMUD time, not additive phase: MLX run most lazy graph work while `mx.eval` wait complete.

48.33M shape, three-step real-data max loop depth + full quant: steady-state data/forward-backward/CMUD 0.011/2.823/0.289s, 2,622 tok/s. Val 0.312s. First two steps wait 2.6–3.1s on stream refill — data stalls dominate short measures even when buffered data only 0.4% of steady-state train step.

Full 1.089B sequence 256, batch one, packed recurrent matmuls, int8 CMUD momentum: materialized-gradient profile 12.37s forward/backward, 25.95s CMUD, 5.66s one val forward. End-to-end train 6.68 tok/s, 15.00 GiB peak. Sync wait 37.32s of 38.32s step, overlap both compute phases. Slower than fused-step bench: profile follow trainer separate gradient + optimizer evals, retain full gradient tree. Exact final-backward, grad average, global clip, CMUD fusion tested not retained: 50.43M shape cut throughput 2,661 → 2,354 tok/s, peak mem 6.881 → 6.834 GiB. Stabilized physical-1.093B active-loop-1: three warmups + five timed steps — separate compiled graphs median 119.74 tok/s; lazy one-eval fusion 85.63 tok/s, save only 0.131 GiB peak. Monolithic compiled fusion slower still. Separate compiled backward + CMUD graphs = production path.

Delayed loop-depth curriculum via `--loop-curriculum-start-ratio`; `--loop-curriculum-ratio` = max depth reached. Full 1.089B physical model, BF16 sequence-64 phase measures active loop depths 1–4: 184.08, 122.93, 92.81, 72.77 tok/s. Weight costs by 70–90% delayed ramp → 141.31 effective tok/s vs 77.56 default 0–20% ramp (1.82x compute-throughput gain).

Equal-token quality not support delayed schedule as default. Seeded 48.33M-param, 655,360-token sampled-MTP sweeps:

| Loop ramp | Time | Speedup | Final validation PPL | PPL change |
| --- | ---: | ---: | ---: | ---: |
| 0–20% (default) | 285.0s | 1.00x | 18.58 | baseline |
| 30–70% | 243.8s | 1.17x | 20.51 | +10.4% |
| 50–80% | 218.6s | 1.30x | 22.13 | +19.1% |
| 70–90% | 191.4s | 1.49x | 23.14 | +24.5% |

30–70% ramp = least harmful measured speed/quality tradeoff; remain opt-in:

```bash
python3 mlx_train.py \
  --loop-curriculum-start-ratio 0.3 \
  --loop-curriculum-ratio 0.7
```

Equal wall-clock favor 30–70% ramp. 0–20% baseline: 655,360 tokens in 285s, val PPL 18.58. 30–70% run: 770,048 tokens in 265.8s, best measured PPL 17.73; second bracketing run 819,200 tokens in 303.2s, PPL 16.64. Delayed ramp lose quality per token, improve quality per local train time in single-seed 48.33M experiment. Opt-in until result survive larger models + multi-seed. Later sequential sweeps thermally throttle M1 Max — excluded from wall-clock compare.

80-update MTP sweep: identical data, seed, curriculum, fixed four-sequence val every 10 updates. Depths 0–4 reach PPL 25 in 173, 189, 221, 247, 264 train seconds. Depth 2 reach PPL 23.5 ten updates earlier than depth 0, still 221 vs 204s. At 80 updates, val PPLs 18.97, 19.23, 18.76, 19.06, 18.70; train times 267, 294, 345, 381, 396s. Exact all-head MTP improve some equal-token results not time-to-PPL in this single-seed short run → sampled policy below replace it.

Cheaper MTP trainer rotate one future head through each accum microbatch. Depth 4: aux vocab projections 16 → 4 per optimizer update; expected depth-mean loss + ckpt tensors unchanged. Max-depth probe: 2,293 tok/s, 13.2 GiB peak vs 1,718 tok/s + 16.5 GiB exact depth 4. 80-update sampled run: 285s, val PPL 18.58 vs 271s / 18.97 no MTP, 401s / 18.70 exact depth 4. Generation fix PaTH chunk width to final train layout → logits prefix-invariant during verification. Sampled ckpt: exact greedy speculation same 128 tokens as vanilla decode at 22.27 vs 16.95 tok/s (1.31x), target-model calls 128 → 96. Full-prefix reference no incremental PaTH/Infini cache yet.

Incremental MLX inference retain effective weights across prefill, token steps, batched verification; fully ternary ckpts use same packed 2-bit rep for cache lifetime. Interleaved six-sample synthetic 1.086B BF16, full-four-loop M1 Max: 1.23 tok/s rebuild dense effective weights each call, 3.71 persistent dense, 4.38 persistent packed. Random weights validate structural throughput only; speculative acceptance still need trained 1B MTP ckpt.

100-update real-data A/B identical seed, stream, curriculum, five val batches every 20 updates: 256-row blocks vs full-matrix whitening. Blockwise C-MUD: val loss 2.7053 vs 2.7601, sustained 1,306 vs 1,118 tok/s at max depth, complete 1,078 vs 1,181s. Single seed support faster default; not statistical convergence study.

Use `mlx_train.py` for native MLX experiments; do not expect identical step-by-step loss vs `train.py` (backend kernels + RFMoE execution order differ).

Convert PyTorch C-MUD ckpt (incl optimizer momentum + exact Engram hash multipliers) before MLX resume:

```bash
python3 mlx_convert.py runs/bitnet/checkpoints/last.pt \
  --output-dir runs/mlx_bitnet --name imported
python3 mlx_train.py \
  --output-dir runs/mlx_bitnet \
  --resume-from runs/mlx_bitnet/checkpoints/imported.safetensors
```

Ckpts created before optimizer param names saved need `--allow-legacy-optimizer-order`. Use that flag only for unmodified ckpt written by this repo.
