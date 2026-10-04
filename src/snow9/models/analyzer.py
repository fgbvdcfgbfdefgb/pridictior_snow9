"""
snow9.models.analyzer — Phase 2B (CPU): Market Analyser.
===========================================================

Extracts a compact feature vector (F=23) from the simulated live feed every
second. Runs on CPU; its output is the ONLY market input the Price Predictor
sees — together with the last 12 h of these vectors (43,200 × F window).

Two interoperable paths:
  * `featurize_frame(df)` — vectorised; used to pre-build the training cache
  * `update(bar)`          — incremental; used in true live/simulated mode

All magnitudes are scaled to O(1) (multiplicative PCT=100 for returns/vol),
and robustly clipped so flash-wick prints cannot blow up the activations.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

PCT = 100.0  # express returns/vol in percent
CLIP = 10.0

FEATURE_NAMES = [
    "ret_1s", "ret_5s", "ret_15s", "ret_60s", "ret_300s", "ret_900s",
    "rv_60", "rv_300", "rv_900",
    "ema12_r", "ema60_r", "ema300_r",
    "rsi14",
    "vol_log", "vol_ratio_300", "buy_frac", "trades_log",
    "range_pct", "z_3600",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "gap_flag",
]
F_DIM = len(FEATURE_NAMES)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return (100 - 100 / (1 + rs)) / 100.0


class MarketAnalyzer:
    feature_names = FEATURE_NAMES
    n_features = F_DIM

    # ------------------------------------------------------------------ live
    def __init__(self, maxlen: int = 3700):
        self.maxlen = maxlen
        self.reset()

    def reset(self) -> None:
        m = self.maxlen
        self.ts, self.c, self.h, self.l = deque(maxlen=m), deque(maxlen=m), deque(maxlen=m), deque(maxlen=m)
        self.v, self.bv, self.tr = deque(maxlen=m), deque(maxlen=m), deque(maxlen=m)
        self._ema = {12: None, 60: None, 300: None}
        self._ema_vol300 = None
        self._rsi_up = None
        self._rsi_dn = None
        self._prev_c = None

    def update(self, bar: dict) -> np.ndarray:
        c = float(bar["close"]); t = int(bar["ts"])
        v = float(bar.get("volume", 0.0))
        bv = float(bar.get("taker_buy_volume", 0.0))
        tr = float(bar.get("trades", 0.0))
        gap = 1.0 if bar.get("was_gap", False) else 0.0
        self.ts.append(t); self.c.append(c)
        self.h.append(float(bar.get("high", c))); self.l.append(float(bar.get("low", c)))
        self.v.append(v); self.bv.append(bv); self.tr.append(tr)
        # incremental EMAs
        for k in self._ema:
            a = 2 / (k + 1)
            self._ema[k] = c if self._ema[k] is None else a * c + (1 - a) * self._ema[k]
        a = 2 / (300 + 1)
        self._ema_vol300 = v if self._ema_vol300 is None else a * v + (1 - a) * self._ema_vol300
        d = 0.0 if self._prev_c is None else c - self._prev_c
        au = max(d, 0.0); ad = max(-d, 0.0)
        al = 1 / 14
        self._rsi_up = au if self._rsi_up is None else al * au + (1 - al) * self._rsi_up
        self._rsi_dn = ad if self._rsi_dn is None else al * ad + (1 - al) * self._rsi_dn
        self._prev_c = c
        # window-based terms
        C = np.asarray(self.c); Lr = np.diff(np.log(C)) if len(C) > 1 else np.zeros(1)

        def ret(k):
            return (np.log(C[-1]) - np.log(C[-1 - k])) * PCT if len(C) > k else 0.0

        def rv(k):
            w = Lr[-k:] if len(Lr) >= 1 else np.zeros(1)
            return float(np.std(w)) * PCT

        mean3600 = C.mean(); std3600 = C.std() + 1e-12
        tt = np.datetime64(t, "s")
        tod = (t % 86400) / 86400 * 2 * np.pi
        dow = ((t // 86400 + 4) % 7) / 7 * 2 * np.pi  # epoch day0 = Thursday
        feats = np.array([
            ret(1), ret(5), ret(15), ret(60), ret(300), ret(900),
            rv(60), rv(300), rv(900),
            c / (self._ema[12] + 1e-12) - 1, c / (self._ema[60] + 1e-12) - 1,
            c / (self._ema[300] + 1e-12) - 1,
            (100 - 100 / (1 + self._rsi_up / (self._rsi_dn + 1e-12))) / 100,
            np.log1p(v), v / (self._ema_vol300 + 1e-12) - 1,
            (bv / v - 0.5) if v > 0 else 0.0, np.log1p(tr),
            (self.h[-1] - self.l[-1]) / (c + 1e-12) * PCT,
            (c - mean3600) / std3600,
            np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow),
            gap,
        ], dtype=np.float64)
        feats[[9, 10, 11]] *= PCT  # ema ratios -> percent
        return np.clip(feats, -CLIP, CLIP).astype(np.float32)

    # ------------------------------------------------------------ vectorised
    def featurize_frame(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].astype("float64")
        v = df["volume"].astype("float64") if "volume" in df else pd.Series(0.0, index=df.index)
        bv = df["taker_buy_volume"].astype("float64") if "taker_buy_volume" in df else pd.Series(0.0, index=df.index)
        tr = df["trades"].astype("float64") if "trades" in df else pd.Series(0.0, index=df.index)
        hi = df["high"].astype("float64") if "high" in df else c
        lo = df["low"].astype("float64") if "low" in df else c
        gap = df["was_gap"].astype("float64") if "was_gap" in df else pd.Series(0.0, index=df.index)

        lc = np.log(c)
        lret = lc.diff()
        out = {
            "rsi14": _rsi(c),
            "vol_log": np.log1p(v),
            "vol_ratio_300": (v / v.ewm(span=300).mean() - 1),
            "buy_frac": (bv / (v + 1e-12) - 0.5).fillna(0.0),
            "trades_log": np.log1p(tr),
            "range_pct": (hi - lo) / (c + 1e-12) * PCT,
            "z_3600": (c - c.rolling(3600, min_periods=2).mean())
                      / (c.rolling(3600, min_periods=2).std() + 1e-12),
            "gap_flag": gap,
        }
        for k in (1, 5, 15, 60, 300, 900):
            out[f"ret_{k}s"] = (lc - lc.shift(k)) * PCT
        for k in (60, 300, 900):
            out[f"rv_{k}"] = lret.rolling(k, min_periods=2).std() * PCT
        for k in (12, 60, 300):
            out[f"ema{k}_r"] = (c / c.ewm(span=k).mean() - 1) * PCT

        ts = df["ts"].astype("int64").to_numpy()
        tod = (ts % 86400) / 86400 * 2 * np.pi
        dow = ((ts // 86400 + 4) % 7) / 7 * 2 * np.pi
        out["tod_sin"], out["tod_cos"] = np.sin(tod), np.cos(tod)
        out["dow_sin"], out["dow_cos"] = np.sin(dow), np.cos(dow)

        M = np.column_stack([out[n] for n in FEATURE_NAMES]).astype("float64")
        M[~np.isfinite(M)] = 0.0
        return np.clip(M, -CLIP, CLIP).astype(np.float32)
