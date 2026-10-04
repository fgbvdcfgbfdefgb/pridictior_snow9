#!/usr/bin/env bash
# Phase 1.5 — one vectorised pass: parquet shards -> flat feature memmaps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python - <<'PY'
from snow9.data import ShardStore, build_feature_cache
from snow9.models import MarketAnalyzer
meta = build_feature_cache(ShardStore("data/raw"), "data/cache", MarketAnalyzer())
print(meta)
PY
