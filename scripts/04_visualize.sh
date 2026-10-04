#!/usr/bin/env bash
# Phase 2C — random day (2020-2026) replayed through every trained model -> 30 fps video.
# Usage: scripts/04_visualize.sh [random|YYYY-MM-DD] [duration_sec]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
DATE="${1:-random}"
DUR="${2:-86400}"
NPZ="runs/rollout_${DATE}.npz"
python -m snow9.viz.rollout --cache data/cache --ckpt-root checkpoints \
  --date "$DATE" --duration-sec "$DUR" --out "$NPZ"
python -m snow9.viz.animate "$NPZ" --out "videos/predictions_${DATE}.mp4" \
  --fps 30 --seconds-per-frame "${SPF:-120}"
echo "Done -> videos/predictions_${DATE}.mp4"
