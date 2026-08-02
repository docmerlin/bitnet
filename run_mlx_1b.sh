#!/usr/bin/env bash
set -euo pipefail

# M1 Max 32 GiB preset: 1,024 tokens per optimizer update without a 1K-token
# activation footprint. Extra arguments override these defaults.
exec "${PYTHON:-./.venv/bin/python}" mlx_train.py \
  --output-dir runs/mlx_bitnet_1b \
  --total-tokens 30000000000 \
  --sequence-length 256 \
  --path-window-size 64 \
  --micro-batch-size 1 \
  --grad-accumulation-steps 4 \
  --vocab-size 32768 \
  --hidden-size 1024 \
  --num-heads 16 \
  --intermediate-size 2048 \
  --num-prelude-layers 8 \
  --num-recurrent-layers 48 \
  --num-coda-layers 8 \
  --num-loops 4 \
  --gradient-checkpointing \
  --gradient-checkpoint-scope recurrent \
  --precision bfloat16 \
  --mud-block-size 64 \
  --cmud-momentum-8bit \
  --cmud-master-dtype bfloat16 \
  --recurrent-quantized-matmul \
  "$@"
