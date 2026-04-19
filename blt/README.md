# BLT

This directory contains the separate Byte Latent Transformer stack used for ternary BLT distillation work. It is intentionally isolated from the older `train.py` BitNet path.

Main entrypoint:

- `python3 -m blt ...`

Runner scripts in this directory:

- `./blt/run_blt_distill.sh`
  - real teacher-guided distillation runner
  - expects gated access to `facebook/blt-1b` and `facebook/blt-entropy`
  - expects a local `facebookresearch/blt` checkout via `BLT_UPSTREAM_REPO`

- `./blt/run_blt_local.sh`
  - quick local smoke runner
  - uses inline text and `--no-teacher`
  - meant to verify the BLT student/trainer/checkpoint path without requiring the Meta teacher stack

Example:

```bash
BLT_UPSTREAM_REPO=/path/to/facebookresearch/blt ./blt/run_blt_distill.sh
```

Quick smoke run:

```bash
./blt/run_blt_local.sh
```

Notes:

- Run these from the repo root, or let the scripts relocate to the repo root automatically.
- The BLT package lives under `blt/`, but it is executed as a Python module from the repository root with `python3 -m blt`.
- The existing `run_train.sh` and `run_local_train.sh` scripts remain for the older BitNet training stack and are not BLT entrypoints.
