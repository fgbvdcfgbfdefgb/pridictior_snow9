"""
snow9.simulator.replay — Phase 2A: live-market simulator.
==========================================================

Replays the stored second-by-second history as if Binance were streaming it
to you right now. Consumers subscribe with a callback `on_bar(bar)` where

    bar = {"ts": epoch_sec, "open":…, "high":…, "low":…, "close":…,
           "volume":…, "trades":…, "taker_buy_volume":…}

* `speed`  : simulated seconds per wall-second. `speed=None | "max"` runs flat
  out (used by the trainer); `speed=1.0` is true real-time.
* Emits exactly one bar per second(grid-aligned, gap-filled upstream), so the
  predictors can update their 25-min forecast every single second.
* Deterministic: same (start, end) always yields the same stream.

Used by
  * training   — the stream drives the online reward loop (no epochs)
  * visualisation — replaying a random day for the animation
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd


class MarketFeedSimulator:
    def __init__(self, frames: pd.DataFrame | list[pd.DataFrame], speed: float | None = None):
        if isinstance(frames, pd.DataFrame):
            frames = [frames]
        df = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
        self.df = df
        self.speed = speed
        self._subscribers: list[Callable[[dict], None]] = []
        self._cols = {c: df[c].to_numpy() for c in
                      ("ts", "open", "high", "low", "close", "volume", "trades", "taker_buy_volume")
                      if c in df.columns}

    def subscribe(self, cb: Callable[[dict], None]) -> None:
        self._subscribers.append(cb)

    def __len__(self) -> int:
        return len(self.df)

    def run(self, start_i: int = 0, stop_i: int | None = None,
            on_tick: Callable[[int, dict], None] | None = None, max_ticks: int | None = None) -> int:
        n = len(self.df)
        stop_i = n if stop_i is None else min(stop_i, n)
        stop_i = min(stop_i, start_i + max_ticks) if max_ticks else stop_i
        t0_wall = time.monotonic()
        t0_sim = float(self._cols["ts"][start_i]) if start_i < n else 0.0
        emitted = 0
        for i in range(start_i, stop_i):
            bar = {k: (float(v[i]) if k != "ts" else int(v[i])) for k, v in self._cols.items()}
            for cb in self._subscribers:
                cb(bar)
            if on_tick is not None:
                on_tick(i, bar)
            emitted += 1
            if self.speed:  # pace the replay (skipped entirely at "max")
                delay = (float(bar["ts"]) - t0_sim) / self.speed - (time.monotonic() - t0_wall)
                if delay > 0:
                    time.sleep(delay)
        return emitted

    def window(self, center_i: int, seconds: int) -> pd.DataFrame:
        return self.df.iloc[max(0, center_i - seconds + 1):center_i + 1]

    def closes(self, lo: int, hi: int) -> np.ndarray:
        return self.df["close"].to_numpy()[lo:hi]
