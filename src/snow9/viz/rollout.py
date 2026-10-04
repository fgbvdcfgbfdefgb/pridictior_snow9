"""
snow9.viz.rollout — Phase 2C (step 1): replay a random market day.
====================================================================

Picks a random day between 2020 and today (or a given --date), replays it
second-by-second through EVERY trained Price Predictor variant, and records
for each second and each model:

  * the predicted price 25 min ahead (median, dotted line in the animation)
  * the q10 / q90 uncertainty band
  * the actual market price at that instant (solid line in the animation)

Output is a single `.npz` the animator renders — decoupled on purpose so a
full-day rollout is computed once and can be re-rendered at any speed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.dataset import ContinuousFeed
from ..models.predictor import PricePredictor

ARCH_KEYS = ("d_model", "nhead", "num_layers", "ff_dim", "dropout")


def _load_variant(variant: dict, n_features: int, ckpt_path: Path, device) -> PricePredictor:
    arch = {k: variant[k] for k in ARCH_KEYS if k in variant}
    model = PricePredictor(n_features=n_features, **arch).to(device).eval()
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=False)
        print(f"  loaded {ckpt_path.name} (step {ckpt.get('step')})")
    else:
        print(f"  WARNING no checkpoint for {variant['name']} — using random init")
    return model


def run_rollout(cache: str, variants_yaml: str, ckpt_root: str, out: str,
                date: str = "random", duration_sec: int = 86400,
                horizon: int = 1500, lookback: int = 43200,
                batch: int = 64, device: str | None = None, seed: int = 0) -> str:
    feed = ContinuousFeed(cache)
    vcfg = yaml.safe_load(open(variants_yaml))
    horizon = int(vcfg.get("horizon_sec", horizon))
    lookback = int(vcfg.get("lookback_sec", lookback)) if lookback == 43200 else lookback
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    first, last = int(feed.ts[0]), int(feed.ts[-1])
    if date == "random":
        rng = random.Random(seed)
        d0 = dt.datetime.fromtimestamp(first + lookback, tz=dt.timezone.utc)
        d1 = dt.datetime.fromtimestamp(last - duration_sec - horizon, tz=dt.timezone.utc)
        day = (d0 + dt.timedelta(seconds=rng.randrange(int((d1 - d0).total_seconds())))).date()
    else:
        day = dt.date.fromisoformat(date)
    start_ts = int(dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc).timestamp())
    end_ts = min(start_ts + duration_sec, last - 1)
    i_lo, i_hi = feed.index_of(start_ts), feed.index_of(end_ts)
    print(f"rollout: {day} UTC  rows [{i_lo},{i_hi})  ({i_hi - i_lo:,} s)")

    preds = {}
    for variant in vcfg["variants"]:
        name = variant["name"]
        ckpt = Path(ckpt_root) / name / "latest.pt"
        if not ckpt.exists():  # allow rank-suffixed dirs
            cands = sorted(Path(ckpt_root).glob(f"{name}*_r*/latest.pt"))
            ckpt = cands[0] if cands else ckpt
        model = _load_variant(variant, feed.n_features, ckpt, device)
        rows = []
        with torch.no_grad():
            for a in range(i_lo, i_hi, batch):
                b = min(a + batch, i_hi)
                x = torch.from_numpy(np.stack([feed.feature_window(i, lookback)
                                               for i in range(a, b)])).to(device)
                lc = torch.tensor([float(feed.close[i]) for i in range(a, b)], device=device)
                pred = model(x, lc)
                rows.append(pred["q_price"].float().cpu())
        q = torch.cat(rows).numpy()                    # (n, 3)
        preds[name] = {"lo": q[:, 0], "med": q[:, 1], "hi": q[:, 2]}
        print(f"  {name}: done")

    ts = feed.ts[i_lo:i_hi].astype("int64")
    result = {"ts": ts, "close": feed.closes(i_lo, i_hi), "horizon": horizon,
              "date": str(day)}
    for k, v in preds.items():
        result[f"{k}__lo"], result[f"{k}__med"], result[f"{k}__hi"] = v["lo"], v["med"], v["hi"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print(f"wrote {out}")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--variants", default="configs/variants.yaml")
    p.add_argument("--ckpt-root", default="checkpoints")
    p.add_argument("--out", default=None)
    p.add_argument("--date", default="random", help="YYYY-MM-DD or 'random' (2020→today)")
    p.add_argument("--duration-sec", type=int, default=86400)
    p.add_argument("--horizon", type=int, default=1500)
    p.add_argument("--lookback", type=int, default=43200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = args.out or f"runs/rollout_{args.date if args.date != 'random' else 'random_day'}.npz"
    run_rollout(args.cache, args.variants, args.ckpt_root, out,
                date=args.date, duration_sec=args.duration_sec,
                horizon=args.horizon, lookback=args.lookback,
                batch=args.batch, seed=args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
