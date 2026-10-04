#!/usr/bin/env python
"""
scripts/smoke_test.py — end-to-end sanity check on the committed sample data.

Runs the WHOLE pipeline small: feature cache -> streaming reward training
(tiny model, CPU) -> checkpointing -> rollout -> 30 fps animation.

    PYTHONPATH=src python scripts/smoke_test.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snow9.data import ShardStore, build_feature_cache, ContinuousFeed   # noqa: E402
from snow9.models import MarketAnalyzer                                  # noqa: E402


def main() -> None:
    print("1/5  feature cache from sample shard")
    shutil.rmtree(ROOT / "data/cache", ignore_errors=True)
    meta = build_feature_cache(ShardStore(ROOT / "data/sample"),
                               ROOT / "data/cache", MarketAnalyzer())
    assert meta["n_features"] == MarketAnalyzer.n_features, meta
    assert meta["n_rows"] > 10_000, meta
    print("     ", meta)

    print("2/5  streaming reward training (tiny 'smoke' variant, CPU)")
    shutil.rmtree(ROOT / "checkpoints", ignore_errors=True)
    from snow9.training.ddp_launch import main as launch
    launch(["--data-cache", str(ROOT / "data/cache"),
            "--variants", str(ROOT / "configs/variants.yaml"),
            "--training", str(ROOT / "configs/training.yaml"),
            "--out", str(ROOT / "checkpoints"),
            "--variant-override", "smoke", "--max-sec", "3000", "--no-resume"])
    ckpt = ROOT / "checkpoints/smoke/latest.pt"
    assert ckpt.exists(), "no checkpoint written"
    metrics = (ROOT / "checkpoints/smoke/metrics.jsonl").read_text().strip().splitlines()
    assert metrics, "no metrics logged"
    print(f"      {len(metrics)} metric lines, checkpoint OK")

    print("3/5  rollout a slice through the trained variant")
    from snow9.viz.rollout import run_rollout
    import yaml
    # run the smoke variant only: build a mini variants file
    mini = {"horizon_sec": 150, "lookback_sec": 600, "quantiles": [0.1, 0.5, 0.9],
            "variants": [{"name": "smoke", "d_model": 64, "nhead": 4,
                          "num_layers": 2, "ff_dim": 128, "dropout": 0.0}]}
    (ROOT / "runs").mkdir(exist_ok=True)
    yaml.safe_dump(mini, open(ROOT / "runs/smoke_variants.yaml", "w"))
    # map smoke checkpoint to expected name
    npz = run_rollout(str(ROOT / "data/cache"), str(ROOT / "runs/smoke_variants.yaml"),
                      str(ROOT / "checkpoints"), str(ROOT / "runs/smoke_rollout.npz"),
                      date=_first_day_of_cache(ROOT), duration_sec=1200,
                      horizon=150, lookback=600, batch=32)
    print("      rollout npz OK:", npz)

    print("4/5  render 30 fps animation")
    from snow9.viz.animate import render_animation
    vid = render_animation(npz, out=str(ROOT / "videos/smoke_demo.mp4"),
                           fps=30, seconds_per_frame=8, accuracy_window=120, dpi=70)
    print("      video:", vid)

    print("5/5  PASSED — pipeline works end to end.")


def _first_day_of_cache(root: Path) -> str:
    import datetime as dt
    feed = ContinuousFeed(root / "data/cache")
    return dt.datetime.fromtimestamp(int(feed.ts[0]) + 800,
                                     tz=dt.timezone.utc).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
