#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Quick local BLT smoke runner.
#
# This path intentionally avoids the Meta teacher stack and runs the separate
# BLT student on CPU with tiny inline text samples so the end-to-end trainer,
# eval loop, and checkpoint path can be exercised quickly.
#
# Usage:
#   ./blt/run_blt_local.sh
#
# Optional overrides:
#   BLT_LOCAL_OUTPUT=checkpoints/blt_local.pt
#   BLT_LOCAL_STEPS=2
#   BLT_LOCAL_BATCH_SIZE=2
#   BLT_LOCAL_SEQUENCE_LENGTH=32

cd "${REPO_ROOT}"

python3 -m blt \
  --text "Byte latent transformers can be distilled into a ternary student." \
  --text "This local runner is only a quick smoke test for the BLT package." \
  --eval-text "Short held out text for a local BLT smoke evaluation." \
  --no-teacher \
  --device "${BLT_LOCAL_DEVICE:-cpu}" \
  --teacher-device "${BLT_LOCAL_TEACHER_DEVICE:-cpu}" \
  --steps "${BLT_LOCAL_STEPS:-2}" \
  --batch-size "${BLT_LOCAL_BATCH_SIZE:-2}" \
  --eval-batch-size "${BLT_LOCAL_EVAL_BATCH_SIZE:-1}" \
  --sequence-length "${BLT_LOCAL_SEQUENCE_LENGTH:-32}" \
  --max-document-bytes "${BLT_LOCAL_MAX_DOCUMENT_BYTES:-256}" \
  --learning-rate "${BLT_LOCAL_LEARNING_RATE:-3e-4}" \
  --weight-decay "${BLT_LOCAL_WEIGHT_DECAY:-0.01}" \
  --eval-every "${BLT_LOCAL_EVAL_EVERY:-1}" \
  --eval-steps "${BLT_LOCAL_EVAL_STEPS:-1}" \
  --save-path "${BLT_LOCAL_OUTPUT:-checkpoints/blt_local.pt}" \
  --save-every "${BLT_LOCAL_SAVE_EVERY:-1}" \
  --local-dim "${BLT_LOCAL_DIM:-64}" \
  --global-dim "${BLT_GLOBAL_DIM:-128}" \
  --decoder-dim "${BLT_DECODER_DIM:-64}" \
  --n-layers-local-encoder "${BLT_LOCAL_ENCODER_LAYERS:-2}" \
  --n-layers-global "${BLT_GLOBAL_LAYERS:-2}" \
  --n-layers-local-decoder "${BLT_LOCAL_DECODER_LAYERS:-2}" \
  --n-heads-local-encoder "${BLT_LOCAL_ENCODER_HEADS:-4}" \
  --n-heads-global "${BLT_GLOBAL_HEADS:-4}" \
  --n-heads-local-decoder "${BLT_LOCAL_DECODER_HEADS:-4}" \
  --n-heads-cross "${BLT_CROSS_HEADS:-4}" \
  --patch-size "${BLT_LOCAL_PATCH_SIZE:-4}" \
  --max-patch-length "${BLT_LOCAL_MAX_PATCH_LENGTH:-8}" \
  --disable-hadamard \
  --disable-4bit-activations
