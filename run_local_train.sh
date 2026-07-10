#!/usr/bin/env bash
set -euo pipefail

# Reasonable local training profile for weak hardware.
#
# Smaller looped shape than the full research target so the end-to-end training
# stack can run on a Mac Mini or similar machine:
# - 2 prelude + 4 recurrent × 4 loops + 2 coda (8 unique, effective depth 20)
# Remove the model-shape overrides below to train the full configuration from
# config.py (8/48/8 × R=4).

python3 train.py \
  --output-dir runs/bitnet_local \
  --train-mixture fineweb_edu=0.7,dclm=0.3 \
  --val-mixture fineweb_edu=0.5,dclm=0.5 \
  --sequence-length 512 \
  --micro-batch-size 1 \
  --grad-accumulation-steps 16 \
  --total-tokens 10000000 \
  --learning-rate 3e-4 \
  --weight-decay 0.05 \
  --warmup-ratio 0.08 \
  --cooldown-ratio 0.05 \
  --stage1-ratio 0.12 \
  --initial-blocks 8 \
  --final-blocks 16 \
  --hidden-size 512 \
  --num-prelude-layers 2 \
  --num-recurrent-layers 4 \
  --num-coda-layers 2 \
  --num-loops 4 \
  --min-num-loops 1 \
  --loop-curriculum-ratio 0.2 \
  --num-heads 16 \
  --intermediate-size 1024 \
  --vocab-size 32768 \
  --log-interval 10 \
  --eval-interval 100 \
  --save-interval 250
