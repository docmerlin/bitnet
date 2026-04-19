#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Example runner for the separate ternary BLT distillation stack.
#
# Requirements:
# - gated access to `facebook/blt-1b` and `facebook/blt-entropy`
# - a local checkout of `facebookresearch/blt`
# - upstream BLT runtime dependencies installed in this environment
#
# Usage:
#   BLT_UPSTREAM_REPO=/path/to/facebookresearch/blt ./blt/run_blt_distill.sh
#
# Optional overrides:
#   BLT_HF_DATASET=HuggingFaceFW/fineweb-edu
#   BLT_HF_SPLIT=train
#   BLT_HF_TEXT_FIELD=text
#   BLT_DEVICE=cuda
#   BLT_TEACHER_DEVICE=cuda
#   BLT_OUTPUT=checkpoints/blt_distill.pt
#   BLT_STEPS=100
#   BLT_BATCH_SIZE=2
#   BLT_SEQUENCE_LENGTH=256

: "${BLT_UPSTREAM_REPO:?Set BLT_UPSTREAM_REPO to your local facebookresearch/blt checkout}"

cd "${REPO_ROOT}"

python3 -m blt \
  --hf-dataset "${BLT_HF_DATASET:-HuggingFaceFW/fineweb-edu}" \
  --hf-split "${BLT_HF_SPLIT:-train}" \
  --hf-text-field "${BLT_HF_TEXT_FIELD:-text}" \
  --eval-text "${BLT_EVAL_TEXT:-A short held-out eval sample for BLT distillation.}" \
  --teacher-upstream-repo "${BLT_UPSTREAM_REPO}" \
  --teacher-model-id "${BLT_TEACHER_MODEL_ID:-facebook/blt-1b}" \
  --teacher-entropy-model-id "${BLT_TEACHER_ENTROPY_MODEL_ID:-facebook/blt-entropy}" \
  --device "${BLT_DEVICE:-cuda}" \
  --teacher-device "${BLT_TEACHER_DEVICE:-${BLT_DEVICE:-cuda}}" \
  --steps "${BLT_STEPS:-100}" \
  --batch-size "${BLT_BATCH_SIZE:-2}" \
  --eval-batch-size "${BLT_EVAL_BATCH_SIZE:-1}" \
  --sequence-length "${BLT_SEQUENCE_LENGTH:-256}" \
  --max-document-bytes "${BLT_MAX_DOCUMENT_BYTES:-2048}" \
  --learning-rate "${BLT_LEARNING_RATE:-3e-4}" \
  --weight-decay "${BLT_WEIGHT_DECAY:-0.01}" \
  --eval-every "${BLT_EVAL_EVERY:-20}" \
  --eval-steps "${BLT_EVAL_STEPS:-2}" \
  --save-path "${BLT_OUTPUT:-checkpoints/blt_distill.pt}" \
  --save-every "${BLT_SAVE_EVERY:-20}" \
  --student-patcher-mode "${BLT_STUDENT_PATCHER_MODE:-teacher_then_student}" \
  --student-patcher-warmup-steps "${BLT_STUDENT_PATCHER_WARMUP_STEPS:-50}" \
  --local-dim "${BLT_LOCAL_DIM:-256}" \
  --global-dim "${BLT_GLOBAL_DIM:-512}" \
  --decoder-dim "${BLT_DECODER_DIM:-256}" \
  --n-layers-local-encoder "${BLT_LOCAL_ENCODER_LAYERS:-4}" \
  --n-layers-global "${BLT_GLOBAL_LAYERS:-8}" \
  --n-layers-local-decoder "${BLT_LOCAL_DECODER_LAYERS:-4}"
