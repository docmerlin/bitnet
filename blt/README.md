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

- Activations stay full precision. Absmax fake-quant / `activation_bits` were removed;
  native fp8-e4m3 is MLX BitNet only (`mlx_model.MLXHBitLinear`), not this stack.
- Run from repo root, or let scripts relocate there automatically.
- BLT package lives under `blt/`, but runs as Python module from repo root via `python3 -m blt`.
- `run_train.sh` and `run_local_train.sh` stay for old BitNet stack. Not BLT entrypoints.
- Omit `--eval-text`, `--eval-text-file`, and `--eval-hf-dataset` → eval reuses training stream, prints warning.
- `--student-patcher-mode distill_only` trains student patcher on teacher boundaries but keeps teacher patches for forward pass.
- Resume restores optimizer and student-patcher state only. Data streams restart from start. RNG state not checkpointed.

## Patching and teacher semantics

- The Meta adapter runs upstream native `ByteLatentTransformer.forward`, capturing
  encoder outputs, global inputs/outputs, and pre-norm decoder features with temporary
  hooks. Upstream owns decoder patch alignment, cross-attention masks, causal masks,
  and EOS document masks; encoder patch IDs must not be reused as decoder IDs.
- Native BLT requires a singleton first real-byte patch. The adapter splits longer
  first patches while preserving other boundaries, and returns the resulting lengths
  for the student (including after student-patcher reruns). Rows are grouped by valid
  byte length and patch count so native forward sees no byte or patch-axis padding.
- Teacher execution can use a different device from the student. All teacher outputs,
  including patch lengths and IDs, transfer to the student device before losses.
- MLX `TrainingConfig.patches_per_sequence` uses **positional uniform** boundaries,
  not whole-sequence entropy ranking. For a fixed sequence length/count, changing
  future content cannot move earlier boundaries. This mode skips entropy inference
  and ignores `max_patch_length` to preserve the exact, unpadded patch count. Changing
  sequence length/count changes the positional schedule; this is not an adaptive
  entropy segmentation policy for variable-length generation.
- Without a fixed count, both entropy backends use causal next-byte thresholding
  plus `max_patch_length`. Generation uses those same rules, including cap-forced
  boundaries, regardless of speculation window.
- Any explicit eval source replaces the training source, even when source types
  differ. With no explicit eval source, the training source is reused. Unspecified
  HF config/split/text-field options still inherit their training values.
