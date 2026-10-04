#!/usr/bin/env bash
# Phase 1 — full 2020 -> today second-by-second download (run on a machine WITH internet).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m snow9.data.download --config configs/data.yaml "$@"
# Result: data/raw/BTCUSDT-1s-YYYY-MM.parquet  (~40-80 MB per month, resumable)
