# BLT

Separate Byte Latent Transformer stack for ternary BLT distillation. Isolated from old `train.py` BitNet path.

Main entrypoint:

- `python3 -m blt ...`

Runner scripts here:

- `./blt/run_blt_distill.sh`
  - real teacher-guided distillation runner
  - needs gated access to `facebook/blt-1b` and `facebook/blt-entropy`
  - needs local `facebookresearch/blt` checkout via `BLT_UPSTREAM_REPO`

- `./blt/run_blt_local.sh`
  - quick local smoke runner
  - uses inline text and `--no-teacher`
  - verifies BLT student/trainer/checkpoint path without Meta teacher stack

Example:

```bash
BLT_UPSTREAM_REPO=/path/to/facebookresearch/blt ./blt/run_blt_distill.sh
```

Quick smoke run:

```bash
./blt/run_blt_local.sh
```

Notes:

- Run from repo root, or let scripts relocate there automatically.
- BLT package lives under `blt/`, but runs as Python module from repo root via `python3 -m blt`.
- `run_train.sh` and `run_local_train.sh` stay for old BitNet stack. Not BLT entrypoints.
- Omit `--eval-text`, `--eval-text-file`, and `--eval-hf-dataset` → eval reuses training stream, prints warning.
- `--student-patcher-mode distill_only` trains student patcher on teacher boundaries but keeps teacher patches for forward pass.
- Resume restores optimizer and student-patcher state only. Data streams restart from start. RNG state not checkpointed.