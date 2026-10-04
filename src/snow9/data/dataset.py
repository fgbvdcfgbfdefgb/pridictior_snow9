"""
snow9.data.dataset — shard access + one-pass feature cache + continuous feed.
=============================================================================

`build_feature_cache` walks every monthly parquet shard once, computes the
Market-Analyser feature matrix (vectorised), and writes three flat memmaps:

    cache/features.f32   (N, F) float32  — model inputs
    cache/close.f64      (N,)   float64  — close price per second (reward source)
    cache/ts.i64         (N,)   int64    — epoch-second per row

Training and the simulator then read the market strictly in time order —
there are no epochs: every second is seen exactly once, exactly as in live
trading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class ShardStore:
    """Discovers and loads monthly parquet shards."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.shards = sorted(self.root.glob("*.parquet"))
        if not self.shards:
            raise FileNotFoundError(f"No parquet shards under {self.root} — "
                                    f"run `python -m snow9.data.download` first.")

    def load(self, path: Path) -> pd.DataFrame:
        return pq.read_table(path).to_pandas()

    def iter_frames(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None):
        for sh in self.shards:
            df = self.load(sh)
            if start is not None:
                df = df[df["ts"] >= int(start.timestamp())]
            if end is not None:
                df = df[df["ts"] < int(end.timestamp())]
            if not df.empty:
                yield df


def build_feature_cache(store: ShardStore, cache_dir: str | Path, analyzer) -> dict:
    """One vectorised pass over all shards -> memmaps ready for training."""
    from tqdm import tqdm

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    frames = [df for df in tqdm(list(store.iter_frames()), desc="loading shards")]
    if not frames:
        raise ValueError("No data found.")
    df = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    feats = analyzer.featurize_frame(df)                      # (N, F) float32
    n, f = feats.shape

    np.save(cache / "_feats.npy", feats)
    np.save(cache / "_close.npy", df["close"].to_numpy("float64"))
    np.save(cache / "_ts.npy", df["ts"].to_numpy("int64"))
    (cache / "_feats.npy").rename(cache / "features.npy")
    (cache / "_close.npy").rename(cache / "close.npy")
    (cache / "_ts.npy").rename(cache / "ts.npy")

    meta = {
        "n_rows": int(n),
        "n_features": int(f),
        "first_ts": int(df["ts"].iloc[0]),
        "last_ts": int(df["ts"].iloc[-1]),
        "feature_names": list(analyzer.feature_names),
        "coverage_days": round((df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86400, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    del frames, df, feats
    return meta


class ContinuousFeed:
    """
    Memmap-backed, strictly-sequential view of the whole 2020->today market.

    * `bars(i)`            -> dict with the i-th second's OHLCV bar (for the simulator)
    * `feature_window(i,k)`-> (k, F) float32 array of the k seconds ending at i
    * `close(i)`           -> close at second i (the reward's ground truth)
    """

    def __init__(self, cache_dir: str | Path, raw_root: str | Path | None = None):
        cache = Path(cache_dir)
        meta = json.loads((cache / "meta.json").read_text())
        self.meta = meta
        n, f = meta["n_rows"], meta["n_features"]
        self.features = np.load(cache / "features.npy", mmap_mode="r").reshape(n, f)
        self.close = np.load(cache / "close.npy", mmap_mode="r")
        self.ts = np.load(cache / "ts.npy", mmap_mode="r")
        self.n_rows = n
        self.n_features = f
        # volume is needed for realistic simulator ticks; recompute cheaply if absent
        self._volumes: np.ndarray | None = None
        self._raw_root = Path(raw_root) if raw_root else None

    def __len__(self) -> int:
        return self.n_rows

    def index_of(self, epoch_sec: int) -> int:
        i = int(np.searchsorted(self.ts, epoch_sec))
        if i >= self.n_rows:
            raise IndexError(f"{epoch_sec} beyond end of feed")
        return i

    def feature_window(self, i: int, lookback: int) -> np.ndarray:
        lo = max(0, i - lookback + 1)
        w = self.features[lo:i + 1]
        if w.shape[0] < lookback:  # left-pad with the first valid row (anchors scaling)
            w = np.concatenate([np.repeat(w[:1], lookback - w.shape[0], axis=0), w], axis=0)
        return np.array(w, dtype=np.float32, copy=True)  # writable copy (memmap is read-only)

    def bar(self, i: int) -> dict:
        i = min(max(int(i), 0), self.n_rows - 1)
        return {
            "ts": int(self.ts[i]),
            "open": float(self.close[i]),   # cache stores closes; O/H/L optional for model
            "high": float(self.close[i]),
            "low": float(self.close[i]),
            "close": float(self.close[i]),
        }

    def closes(self, lo: int, hi: int) -> np.ndarray:
        return np.asarray(self.close[lo:hi])
