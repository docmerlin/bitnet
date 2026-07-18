# bitnet

Experimental PyTorch repo, two research tracks:

- deep ternary BitNet-style LM around hybrid transformer block
- separate ternary Byte Latent Transformer (`blt/`) stack for BLT distillation

Early-stage research code. Interfaces, defaults, training behavior evolving fast.

**Distillation teacher.** This repo distills from [Ornith 1.0](https://huggingface.co/deepreinforce-ai) (DeepReinforce AI) — an open-weights, MIT-licensed model family on HuggingFace. Open-weights teachers under permissive licenses only.

⚠️ **Not for Claude distillation.** This repo cannot and should not be used to distill Claude or any Anthropic model. See the Anthropic Terms of Service for policies on model distillation and derivative use.

## What Is In This Repo

Two distinct model/training paths.

### 1. BitNet-style Hybrid LM

Original path, around:

- `model.py`
- `train.py`
- `run_train.sh`
- `run_local_train.sh`

Key properties:

- ternary linear layers via `HBitLinear`
- compact DeepSeek Engram conditional memory at layers 1 and 15 by default:
  deterministic hashed bigram/trigram lookup, hidden-state gating, causal short convolution
  (`--no-engram` disables it; table capacity via `--engram-vocab-size`)
- unified hybrid block every layer (`layers/hybrid_block.py`)
- each block combines:
  - PaTH-FoX data-dependent positional attention in bounded local windows
  - Infini-Attention style fixed-size memory for context beyond each window
  - local attention work scales linearly with sequence length at fixed
    `--path-window-size` (default 64); no full-sequence attention matrix
  - **Sandwich RMSNorm** residual: `post(x + scale * sublayer(pre(x)))` on attn and FFN
    (pre-norm + residual-stream post-norm; stabilizes looped unrolls)
  - Attention Residuals (learned scale on each residual branch)
- **looped / recurrent-depth** layout (default): 8 prelude + 48 unique recurrent × `R` loops + 8 coda
  - same 64 unique params as a flat 64-layer stack; effective depth = `8 + 48R + 8` (default `R=4` → 208)
  - CLI: `--num-prelude-layers`, `--num-recurrent-layers`, `--num-coda-layers`, `--num-loops`
  - flat ablation: `--num-layers N` alone → `(0, N, 0)` with `R=1`
  - test-time depth: `model(..., num_loops=R)` without new parameters
  - Infini memory **policy B**: every loop can **read** compressive memory; only the
    **last** recurrent loop **writes** (no R-fold self-pollution of the bank)
  - **Hyperloop-style loop HC** (always on for the recurrent core): 4 residual streams,
    input-dependent `H_pre`/`H_post`, **diagonal** `H_res=diag(σ(·))` (no Sinkhorn),
    plus loop position embeddings — applied only at loop boundaries (Zeitoun et al.)
  - **Trainability:** R curriculum (`--min-num-loops`→`--num-loops` from
    `--loop-curriculum-start-ratio` to `--loop-curriculum-ratio`, default 1→4 over
    0–20% tokens). **Recommended for local wall-clock-limited training:** use the 30–70%
    schedule with `--loop-curriculum-start-ratio 0.3 --loop-curriculum-ratio 0.7`; it won
    the measured equal-wall-clock comparison by processing more tokens. Use 0–20% when
    optimizing quality per token;
    gradient checkpointing
    default on / compile off (XOR); `--checkpoint-granularity loop|layer` —
    **loop** (default) fewer recompute segments, **layer** max VRAM save;
    small-scale sublayer outs; logs `active_loops` + HC/AttnRes health
- hierarchical tokenizer path under `tokenizer/`
- streaming Hugging Face dataset training in `train.py`

### 2. Separate Ternary BLT Stack

Newer, isolated BLT path under `blt/`.

Key properties:

- raw-byte input instead of hierarchical tokenizer
- separate local encoder, global latent transformer, local decoder
- optional teacher-guided distillation from Meta BLT
- student patcher training and rollout
- separate CLI entrypoint via `python3 -m blt`
- separate runner scripts in `blt/`

BLT code isolated from `train.py` path so BLT experiments don't entangle original BitNet trainer.

## Repository Layout

### Top-level BitNet path

- `config.py`: BitNet model and trainer config (loop structure + effective depth)
- `model.py`: main `BitNetDeep` model (looped prelude/core/coda forward; checkpointing + MTP)
- `train.py`: CLI + training loop
- `data/`: dataset presets, mixture parsing, packing
- `training/`: losses, schedules, checkpoints, runtime helpers
- `mlx_model.py` / `mlx_train.py`: native MLX model and curriculum trainer
- `mlx_convert.py`: warm-convert PyTorch C-MUD checkpoints to MLX
- `mlx_path_kernel.py`: trainable custom Metal solve for PaTH-FoX
- `mlx_rfmoe_kernel.py`: differentiable conditional Metal expert projections
- `mlx_ternary_kernel.py`: packed 2-bit recurrent ternary Metal matmul with STE backward
- `mlx_optim.py`: native blockwise C-MUD with default int8 matrix momentum and int8 C-Lion
- `utils.py`: shared BLT RoPE, attention-mask, and checkpoint helpers
- `data/streams.py`: packing + PrefetchStream
- `training/runtime.py`: choose_device, AMP, logger, evaluate
- `layers/hybrid_block.py`: main hybrid transformer block
- `layers/engram.py`: DeepSeek Engram-style hashed N-gram conditional memory
- `layers/infini_attention.py`: local PaTH-FoX attention + Infini memory handling
- `layers/h_bitlinear.py`: ternary / Hadamard linear layer
- `layers/rfmoe.py`: routing-free MoE FFN
- `tokenizer/`: hierarchical tokenizer
- `run_train.sh`: full BitNet training launcher
- `run_local_train.sh`: smaller local BitNet launcher

### BLT path

- `blt/config.py`: `TernaryBLTConfig`
- `blt/model.py`: full ternary BLT student model
- `blt/data.py`: raw-byte dataset and batch stream utils
- `blt/losses.py`: BLT distillation losses
- `blt/train_distill.py`: BLT CLI, trainer, checkpointing, eval, resume flow
- `blt/teacher/facebook_blt.py`: optional adapter for Meta BLT teacher inference
- `blt/patching/teacher_patcher.py`: patch-length helpers and teacher-forced patch utils
- `blt/patching/student_entropy.py`: student boundary model for learned patching
- `blt/layers/`: BLT local encoder / global transformer / local decoder / cross-attention modules
- `blt/run_blt_distill.sh`: real BLT distillation runner
- `blt/run_blt_local.sh`: quick local BLT smoke runner
- `blt/README.md`: BLT-specific notes

### Tests

Repo uses pytest (`python3 -m pytest tests/`).

BitNet path examples:

- `tests/test_hybrid_block.py`
- `tests/test_infini_attention_memory.py`
- `tests/test_h_bitlinear.py`
- `tests/test_config_validation.py`

BLT path examples:

- `tests/test_blt_shapes.py`
- `tests/test_blt_distill_smoke.py`
- `tests/test_blt_patching.py`
- `tests/test_blt_masking.py`
- `tests/test_blt_teacher_adapter.py`
- `tests/test_blt_train_cli.py`
- `tests/test_blt_resume_eval_patcher.py`

## Dependencies

Install repo deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` currently includes:

- `torch`
- `datasets`
- `tiktoken`
- optional: `pytest`

Dependency notes:

- `train.py` needs `datasets` and `tiktoken`
- `python3 -m blt --hf-dataset ...` needs `datasets`
- real Meta BLT distillation also needs:
  - gated access to `facebook/blt-1b`
  - gated access to `facebook/blt-entropy`
  - local checkout of `facebookresearch/blt`
  - upstream BLT runtime deps that project uses

## Quick Start

### BitNet forward smoke test

```bash
python3 tests/test_train_smoke.py
```

Exercises:

- hybrid BitNet model stack
- hierarchical tokenizer path
- `HBitLinear`

### BitNet training launchers

Local smoke profile:

```bash
./run_local_train.sh
```

Fuller GPU profile:

```bash
./run_train.sh
```

These launchers drive `train.py`, not BLT stack.

### MLX generation

Generate from a native MLX checkpoint. Checkpoints trained with `--mtp-depth` use exact
greedy speculative decoding by default; pass `--no-speculative` for the vanilla baseline.

```bash
python3 mlx_generate.py runs/bitnet/checkpoints/final.safetensors \
  --prompt "The quick brown fox" --max-new-tokens 64
```

### BLT local smoke run

```bash
./blt/run_blt_local.sh
```

Easiest end-to-end BLT check. It:

- stays on CPU by default
- uses inline text
- runs without teacher
- exercises BLT trainer, eval loop, checkpoint save path, CLI entrypoint

### BLT teacher-guided distillation run

```bash
BLT_UPSTREAM_REPO=/path/to/facebookresearch/blt ./blt/run_blt_distill.sh
```

That launcher:

- runs `python3 -m blt`
- uses `facebook/blt-1b` as default teacher model
- uses `facebook/blt-entropy` as default entropy model
- defaults student patcher rollout mode to `teacher_then_student`

## BitNet Training Path

Original BitNet path uses `train.py`, supports:

- streaming Hugging Face datasets
- train/validation mixtures
- curriculum switching between early and late mixtures
- progressive block growth
- gradient checkpointing
- optional `torch.compile`

### Optimizer: C-MUD (+ 8-bit C-Lion fallback)

Default optimizer for BitNet path is **C-MUD**: cautious variant of **MUD**
(MomentUm Decorrelation, from *Beyond Muon: MUD (MomentUm Decorrelation) for
Faster Transformer Training*, Southworth & Thomas 2026). MUD decorrelates the
matrix-valued momentum update, like Muon, but replaces Muon's polar /
Newton-Schulz iteration with cheaper **triangular whitening** surrogate: per
pass it row-normalizes momentum, forms row Gram `G = Q Qᵀ`, takes lower
triangle `T = tril(G)` as Cholesky-like factor, applies forward triangular
solve `Q ← T⁻¹ Q`, re-normalizes. Single pass (MUD1, default) costs one `k×k`
triangular solve where `k = min(rows, cols)` — ~12× fewer FLOPs than Muon — and
map converges quadratically toward row-orthonormal `Q Qᵀ ≈ I_k` as passes
increase. Matrix update scaled by `0.2·√(max(rows, cols))`.

**C-** prefix is cautious-optimizer mod from *Cautious Optimizers: Improving
Training with One Line of Code*, which zeroes any per-coordinate update whose
sign disagrees with current gradient (and rescales survivors to preserve mean
step size). Same cautious mask applied to fallback below.

Usage:

- **C-MUD for 2D matrix weights** — bulk of model (`HBitLinear` projections,
  attention and FFN weights). Where triangular momentum decorrelation defined.
- **8-bit C-Lion as fallback** for params MUD doesn't target: embeddings,
  RMSNorm/SubLN gains, biases, scalar/vector gates and AttnRes scales. Cautious
  variant of Lion (not plain Lion), 8-bit optimizer state to keep memory low.

Implementation: `optim.py` provides `CMUD` (one `torch.optim.Optimizer`
dispatching per parameter group) plus `build_cmud` / `split_parameters_for_cmud`
for routing. `train.py` builds it by default; legacy full-precision `Lion`
still available via `--optimizer lion`.

Relevant flags:

- `--optimizer {cmud,lion}` (default `cmud`)
- `--learning-rate` — LR for C-Lion fallback group (and legacy Lion path)
- `--mud-learning-rate` — LR for MUD matrix group (MUD paper default `1e-3`)
- `--mud-momentum` — MUD heavy-ball momentum (Nesterov lookahead)
- `--mud-passes` — triangular-whitening passes `p` (default `1` = MUD1; `2` for
  harder landscapes)
- `--no-cautious` — drop cautious mask (plain MUD + Lion)
- `--no-optimizer-8bit` — keep C-Lion fallback momentum full precision

8-bit state applied to fallback params at or above 2048-element block size, so
main beneficiary is large embedding / tied output table; small tensors (norms,
biases, scalar gates) keep full-precision momentum.

### Current BitNet launcher defaults

`run_train.sh` uses larger research profile:

- hidden size `1024`
- `64` layers
- `32` heads
- sequence length `1024`
- broader early web mixture, later code/math-heavier curriculum

`run_local_train.sh` uses smaller local profile:

- hidden size `512`
- `12` layers
- `16` heads
- sequence length `512`

### Programming data presets

`train.py` includes built-in streaming programming presets based on CodeSearchNet shards:

- `code_search_net_all`
- `code_search_net_python`
- `code_search_net_go`
- `code_search_net_javascript`
- `code_search_net_java`
- `code_search_net_php`
- `code_search_net_ruby`

Recommended default for broader coding coverage:

- `code_search_net_all`

Example:

```bash
python3 train.py \
  --train-mixture fineweb_edu=0.55,dclm=0.25,code_search_net_all=0.20 \
  --val-mixture fineweb_edu=0.5,code_search_net_all=0.5
```

Custom Hugging Face dataset entries also supported, form:

```text
path|config|split|text_field=weight
```

Example:

```bash
python3 train.py \
  --train-mixture fineweb_edu=0.8,bigcode/starcoderdata|python|train|content=0.2
```

Requires appropriate authenticated Hugging Face access.

### Math data presets

Built-in math presets currently include:

- `finemath_3plus`
- `open_web_math`

Example:

```bash
python3 train.py \
  --train-mixture fineweb_edu=0.45,dclm=0.20,code_search_net_python=0.10,finemath_3plus=0.25 \
  --val-mixture fineweb_edu=0.30,code_search_net_python=0.20,finemath_3plus=0.50
```

### Early / late data curriculum

`train.py` supports switching mixtures during training with:

- `--early-train-mixture`
- `--late-train-mixture`
- `--mixture-switch-ratio`

Example:

```bash
python3 train.py \
  --early-train-mixture fineweb_edu=0.60,dclm=0.25,code_search_net_all=0.10,finemath_3plus=0.05 \
  --late-train-mixture fineweb_edu=0.35,dclm=0.15,code_search_net_all=0.20,finemath_3plus=0.30 \
  --mixture-switch-ratio 0.70 \
  --val-mixture fineweb_edu=0.35,dclm=0.15,code_search_net_all=0.20,finemath_3plus=0.30
```

## BLT Path

BLT stack is separate research implementation, distills Meta BLT behavior into ternary student.

### BLT architecture in this repo

BLT student path includes:

- local byte encoder
- latent/global transformer over patches
- local byte decoder
- teacher-forced patch utils
- optional student patcher training
- online teacher-guided distillation
- resume/eval/checkpoint support

Implementation under:

- `blt/model.py`
- `blt/layers/`
- `blt/train_distill.py`

### BLT CLI entrypoint

Everything runs through:

```bash
python3 -m blt --help
```

CLI supports:

- inline text: `--text`
- text files: `--text-file`
- streaming Hugging Face datasets: `--hf-dataset`
- eval sources via `--eval-text`, `--eval-text-file`, `--eval-hf-dataset` (if omitted, trainer reuses training stream, prints warning)
- checkpoint save / resume
- student patcher training and rollout (`distill_only`, `teacher_then_student`, `student`)
- teacher disable via `--no-teacher`
- teacher entropy patch disable via `--disable-teacher-patcher`

### BLT teacher adapter requirements

Real teacher adapter in `blt/teacher/facebook_blt.py` optional, not needed for local smoke runs.

Expects:

- local `facebookresearch/blt` checkout
- gated model access to:
  - `facebook/blt-1b`
  - `facebook/blt-entropy`
- upstream BLT runtime deps installed in active environment

### BLT current behavior / contracts

BLT path currently assumes:

- suffix-padded masks only
- patch lengths correspond to valid prefix, not full padded width
- padded rows zeroed in encoder and decoder outputs
- when student patcher takes over, teacher rerun on selected patch layout so patch-level distillation targets stay aligned

### BLT local example

Quick local smoke run:

```bash
./blt/run_blt_local.sh
```

Equivalent direct CLI shape:

```bash
python3 -m blt \
  --text "Byte latent transformers can be distilled into a ternary student." \
  --text "This local runner is only a quick smoke test for the BLT package." \
  --eval-text "Short held out text for a local BLT smoke evaluation." \
  --no-teacher \
  --device cpu \
  --steps 2 \
  --batch-size 2 \
  --sequence-length 32 \
  --save-path checkpoints/blt_local.pt
```

### BLT distillation example

Recommended entrypoint is runner script:

```bash
BLT_UPSTREAM_REPO=/path/to/facebookresearch/blt ./blt/run_blt_distill.sh
```

Equivalent direct CLI shape:

```bash
python3 -m blt \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --hf-split train \
  --hf-text-field text \
  --eval-text "A short held-out eval sample for BLT distillation." \
  --teacher-upstream-repo /path/to/facebookresearch/blt \
  --teacher-model-id facebook/blt-1b \
  --teacher-entropy-model-id facebook/blt-entropy \
  --device cuda \
  --teacher-device cuda \
  --steps 100 \
  --batch-size 2 \
  --sequence-length 256 \
  --save-path checkpoints/blt_distill.pt \
  --save-every 20 \
  --student-patcher-mode teacher_then_student \
  --student-patcher-warmup-steps 50
```

## Testing

Repo relies on script-style tests, run directly with `python3`.

Examples:

```bash
python3 tests/test_train_smoke.py
python3 tests/test_hybrid_block.py
python3 tests/test_infini_attention_memory.py
python3 tests/test_utils.py
python3 tests/test_tokenizer_roundtrip.py
python3 tests/test_train_smoke.py
python3 tests/test_blt_shapes.py
python3 tests/test_blt_masking.py
python3 tests/test_blt_patching.py
python3 tests/test_blt_distill_smoke.py
python3 tests/test_blt_teacher_adapter.py
python3 tests/test_blt_train_cli.py
python3 tests/test_blt_resume_eval_patcher.py

Or run the full script-style suite with pytest from the repo root:

```bash
python3 -m pytest
```
```

Syntax-only verification for edited files:

```bash
python3 -m py_compile path/to/file.py
```

## Current Caveats

- Research code, not production training framework.
- Resume restores optimizer/scheduler (and BLT student-patcher) state only. Training streams restart from beginning of each dataset; RNG state not checkpointed.
- Real Meta BLT teacher path designed and regression-tested through local stubs and interface checks here, but actual upstream Meta runtime still depends on external gated assets and upstream packages.
- BLT path explicitly supports suffix-padded masks; non-suffix masks rejected.
- BLT local smoke runner is functionality check, not quality benchmark.
- Cache and performance optimizations must be benchmarked with at least 500M physical
  parameters, preferably the full 1B configuration. Smaller shapes are smoke tests only.
- BitNet and BLT paths intentionally separate; features added to one not auto-mirrored in other.

## Notes

- Use `run_train.sh` / `run_local_train.sh` for older BitNet path.
- Use `blt/run_blt_distill.sh` / `blt/run_blt_local.sh` or `python3 -m blt` for BLT path.
- See `blt/README.md` for BLT-specific notes scoped to that directory.
