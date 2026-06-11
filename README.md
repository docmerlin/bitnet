# bitnet

Experimental PyTorch repo for two related research tracks:

- a deep ternary BitNet-style language model built around a hybrid transformer block
- a separate ternary Byte Latent Transformer (`blt/`) stack for BLT distillation work

This is still early-stage research code. Interfaces, defaults, and training behavior are evolving quickly.

## What Is In This Repo

There are now two distinct model/training paths in this repository.

### 1. BitNet-style Hybrid LM

This is the original repo path built around:

- `model.py`
- `train.py`
- `run_train.sh`
- `run_local_train.sh`

Key properties:

- ternary linear layers via `HBitLinear`
- a unified hybrid block in every layer (`layers/hybrid_block.py`)
- each block combines:
  - Infini-Attention style local + memory attention
  - Attention Residuals (AttnRes)
- hierarchical tokenizer path under `tokenizer/`
- streaming Hugging Face dataset training path in `train.py`

### 2. Separate Ternary BLT Stack

This is the newer, isolated BLT research path under `blt/`.

Key properties:

- raw-byte input path instead of the hierarchical tokenizer path
- separate local encoder, global latent transformer, and local decoder
- optional teacher-guided distillation from Meta BLT
- student patcher training and rollout support
- separate CLI entrypoint via `python3 -m blt`
- separate runner scripts in `blt/`

The BLT code is intentionally isolated from the older `train.py` path so BLT experiments do not entangle the original BitNet trainer.

## Repository Layout

### Top-level BitNet path

- `config.py`: BitNet model and trainer configuration
- `model.py`: main `BitNetDeep` model
- `train.py`: streaming training pipeline for the BitNet path
- `layers/hybrid_block.py`: main hybrid transformer block
- `layers/block_attnres.py`: standalone block-local attention module (legacy/alternate block; production model uses AttnRes wrappers inside `hybrid_block.py`)
- `layers/infini_attention.py`: Infini-Attention-style module with memory handling
- `layers/h_bitlinear.py`: ternary / Hadamard linear layer implementation
- `tokenizer/`: hierarchical tokenizer implementation
- `utils.py`: rotary embedding and ternary helper functions
- `test_forward.py`: quick forward-pass smoke test for the BitNet path
- `run_train.sh`: full BitNet training launcher
- `run_local_train.sh`: smaller local BitNet training launcher

### BLT path

- `blt/config.py`: `TernaryBLTConfig`
- `blt/model.py`: full ternary BLT student model
- `blt/data.py`: raw-byte dataset and batch stream utilities
- `blt/losses.py`: BLT distillation losses
- `blt/train_distill.py`: BLT CLI, trainer, checkpointing, eval, and resume flow
- `blt/teacher/facebook_blt.py`: optional adapter for Meta BLT teacher inference
- `blt/patching/teacher_patcher.py`: patch-length helpers and teacher-forced patch utilities
- `blt/patching/student_entropy.py`: student boundary model for learned patching
- `blt/layers/`: BLT local encoder / global transformer / local decoder / cross-attention modules
- `blt/run_blt_distill.sh`: real BLT distillation runner
- `blt/run_blt_local.sh`: quick local BLT smoke runner
- `blt/README.md`: BLT-specific notes

### Tests

The repo currently uses simple script-style regression tests rather than a full pytest suite.

BitNet path examples:

- `tests/test_block_attnres.py`
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

Install the repo dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` currently includes:

- `torch`
- `datasets`
- `tiktoken`
- optional logging integrations: `tensorboard`, `wandb`

Dependency notes:

- `test_forward.py` needs `tiktoken`
- `train.py` needs `datasets` and `tiktoken`
- `python3 -m blt --hf-dataset ...` needs `datasets`
- real Meta BLT distillation also needs:
  - gated access to `facebook/blt-1b`
  - gated access to `facebook/blt-entropy`
  - a local checkout of `facebookresearch/blt`
  - the upstream BLT runtime dependencies used by that project

## Quick Start

### BitNet forward smoke test

```bash
python3 test_forward.py
```

This exercises:

- the hybrid BitNet model stack
- the hierarchical tokenizer path
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

These launchers drive `train.py`, not the BLT stack.

### BLT local smoke run

```bash
./blt/run_blt_local.sh
```

This is the easiest end-to-end BLT check. It:

- stays on CPU by default
- uses inline text
- runs without a teacher
- exercises the BLT trainer, eval loop, checkpoint save path, and CLI entrypoint

### BLT teacher-guided distillation run

```bash
BLT_UPSTREAM_REPO=/path/to/facebookresearch/blt ./blt/run_blt_distill.sh
```

That launcher:

- runs `python3 -m blt`
- uses `facebook/blt-1b` as the default teacher model
- uses `facebook/blt-entropy` as the default entropy model
- defaults to a student patcher rollout mode of `teacher_then_student`

## BitNet Training Path

The original BitNet path uses `train.py` and supports:

- streaming Hugging Face datasets
- train/validation mixtures
- curriculum switching between early and late mixtures
- progressive block growth
- gradient checkpointing
- optional `torch.compile`

### Current BitNet launcher defaults

`run_train.sh` uses a larger research profile:

- hidden size `1024`
- `64` layers
- `32` heads
- sequence length `1024`
- broader early web mixture and later code/math-heavier curriculum

`run_local_train.sh` uses a smaller local profile:

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

Custom Hugging Face dataset entries are also supported in the form:

```text
path|config|split|text_field=weight
```

Example:

```bash
python3 train.py \
  --train-mixture fineweb_edu=0.8,bigcode/starcoderdata|python|train|content=0.2
```

That requires the appropriate authenticated Hugging Face access.

### Math data presets

Built-in math-oriented presets currently include:

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

The BLT stack is a separate research implementation aimed at distilling Meta BLT behavior into a ternary student.

### BLT architecture in this repo

The BLT student path includes:

- local byte encoder
- latent/global transformer over patches
- local byte decoder
- teacher-forced patch utilities
- optional student patcher training
- online teacher-guided distillation
- resume/eval/checkpoint support

The implementation is under:

- `blt/model.py`
- `blt/layers/`
- `blt/train_distill.py`

### BLT CLI entrypoint

Everything runs through:

```bash
python3 -m blt --help
```

The CLI supports:

- inline text: `--text`
- text files: `--text-file`
- streaming Hugging Face datasets: `--hf-dataset`
- eval sources via `--eval-text`, `--eval-text-file`, `--eval-hf-dataset` (if omitted, the trainer reuses the training stream and prints a warning)
- checkpoint save / resume
- student patcher training and rollout (`distill_only`, `teacher_then_student`, `student`)
- teacher disable path via `--no-teacher`
- teacher entropy patch disable path via `--disable-teacher-patcher`

### BLT teacher adapter requirements

The real teacher adapter in `blt/teacher/facebook_blt.py` is optional and not needed for local smoke runs.

It expects:

- a local `facebookresearch/blt` checkout
- gated model access to:
  - `facebook/blt-1b`
  - `facebook/blt-entropy`
- the upstream BLT runtime dependencies installed in the active environment

### BLT current behavior / contracts

The BLT path currently assumes:

- suffix-padded masks only
- patch lengths correspond to the valid prefix, not the full padded width
- padded rows are zeroed in encoder and decoder outputs
- when the student patcher takes over, the teacher is rerun on the selected patch layout so patch-level distillation targets stay aligned

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

The recommended entrypoint is the runner script:

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

This repo currently relies on script-style tests that can be run directly with `python3`.

Examples:

```bash
python3 test_forward.py
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

Syntax-only verification for edited files can also be done with:

```bash
python3 -m py_compile path/to/file.py
```

## Current Caveats

- This is research code, not a production training framework.
- Resume restores optimizer/scheduler (and BLT student-patcher) state only. Training streams restart from the beginning of each dataset and RNG state is not checkpointed.
- The real Meta BLT teacher path was designed and regression-tested through local stubs and interface checks in this repo, but the actual upstream Meta runtime still depends on external gated assets and upstream packages.
- The BLT path explicitly supports suffix-padded masks; non-suffix masks are rejected.
- The BLT local smoke runner is a functionality check, not a quality benchmark.
- The BitNet and BLT paths are intentionally separate; features added to one are not automatically mirrored in the other.

## Notes

- Use `run_train.sh` / `run_local_train.sh` for the older BitNet path.
- Use `blt/run_blt_distill.sh` / `blt/run_blt_local.sh` or `python3 -m blt` for the BLT path.
- See `blt/README.md` for BLT-specific notes scoped to that directory.
