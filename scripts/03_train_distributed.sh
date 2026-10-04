#!/usr/bin/env bash
# Phase 2B — train ALL four predictor variants, ONE PER GPU, streaming rewards.
# Snowflake box: 4x A10 23GB -> nproc_per_node=4. Checkpoints land in checkpoints/<variant>/
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=10          # 48 vCPU / 4 ranks + loader headroom
export TORCH_CUDNN_V8_API_ENABLED=1
torchrun --standalone --nproc_per_node="${NPROC:-4}" \
  -m snow9.training.ddp_launch \
  --data-cache data/cache \
  --variants configs/variants.yaml \
  --training configs/training.yaml \
  --out checkpoints "$@"
