#!/usr/bin/env bash
set -euo pipefail

# Full training profile for regular GPU hardware.
#
# This uses the intended research model shape:
# - hidden size 1024
# - 64 layers
# - 32 heads
# - 128k hierarchical vocabulary target
#
# The defaults below assume a reasonably capable CUDA machine. If you have less
# VRAM, reduce --sequence-length, keep --micro-batch-size at 1, and increase
# --grad-accumulation-steps.

python3 train.py \
  --output-dir runs/bitnet_full \
  --train-mixture fineweb_edu=0.7,dclm=0.3 \
  --val-mixture fineweb_edu=0.5,dclm=0.5 \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --grad-accumulation-steps 64 \
  --total-tokens 500000000 \
  --learning-rate 3e-4 \
  --weight-decay 0.05 \
  --warmup-ratio 0.08 \
  --cooldown-ratio 0.05 \
  --stage1-ratio 0.12 \
  --initial-blocks 8 \
  --final-blocks 32 \
  --hidden-size 1024 \
  --num-layers 64 \
  --num-heads 32 \
  --intermediate-size 2048 \
  --vocab-size 131072 \
  --precision bf16 \
  --gradient-checkpointing \
  --compile \
  --log-interval 10 \
  --eval-interval 250 \
  --save-interval 500
