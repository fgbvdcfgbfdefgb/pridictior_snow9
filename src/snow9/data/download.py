"""
snow9.data.download — Phase 1: fetch complete BTCUSDT market data.
===================================================================

Pulls second-by-second klines (OHLCV + taker-buy volume + trade count) from
Binance's public historical archive (https://data.binance.vision), one monthly
parquet shard per month, resumably:

* monthly zip first, then daily zips for the current (incomplete) month
* automatic fallback to 1-second bars aggregated from aggTrades when klines
  are not published for a month (e.g. very old ranges)
* optional checksum verification against the published .CHECKSUM files
* idempotent: existing shards are skipped, so re-running resumes downloads
* works fully OFFLINE afterwards — shards are plain parquet files, which is
  exactly what you upload to the Snowflake stage (no internet inside Snowflake)

Usage
-----
    python -m snow9.data.download --config configs/data.yaml
    python -m snow9.data.download --start 2020-01-01 --end 2026-10-04 \
        --symbol BTCUSDT --interval 1s --out data/raw
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml

BASE = "https://data.binance.vision/data/spot"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1s"
    start: str = "2020-01-01"
    end: str | None = None
    out_dir: str = "data/raw"
    fallback_to_aggtrades: bool = True
    verify_checksums: bool = True
    gap_fill: bool = True


def _load_config(path: str | None, overrides: dict) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if cfg.end is None:
        cfg.end = dt.date.today().isoformat()
    return cfg


def _get(url: str, retries: int = 5, timeout: int = 120) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 400):
                return None
            time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def _month_iter(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _day_iter(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _klines_url(freq: str, cfg: Config, stamp: str) -> str:
    # freq: "monthly" | "daily"; stamp: "2020-01" | "2020-01-15"
    return (f"{BASE}/{freq}/klines/{cfg.symbol}/{cfg.interval}/"
            f"{cfg.symbol}-{cfg.interval}-{stamp}.zip")


def _trades_url(freq: str, cfg: Config, stamp: str) -> str:
    return (f"{BASE}/{freq}/aggTrades/{cfg.symbol}/{cfg.symbol}-aggTrades-{stamp}.zip")


def _verify(url: str, blob: bytes) -> bool:
    r = _get(url + ".CHECKSUM")
    if r is None:
        return True  # archive doesn't publish one -> accept
    expected = r.text.split()[0].strip()
    return hashlib.sha256(blob).hexdigest() == expected


def _read_kline_zip(blob: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = zf.namelist()[0]
    raw = zf.read(name)
    head = raw[:64].decode(errors="ignore").lower()
    header = 0 if head.startswith("open_time") else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=KLINE_COLS)
    # Binance switched open_time units (us in newer dumps) — normalise to seconds
    ts = df["open_time"].astype("int64")
    unit = np.where(ts > 10**14, 10**6, np.where(ts > 10**11, 10**3, 1))
    df["ts"] = (ts // unit).astype("int64")
    return df


def _read_aggtrades_zip(blob: bytes) -> pd.DataFrame:
    """Build 1-second OHLCV bars from aggregated trades."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    raw = zf.read(zf.namelist()[0])
    head = raw[:64].decode(errors="ignore").lower()
    header = 0 if head.startswith("agg_trade_id") else None
    cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker"]
    df = pd.read_csv(io.BytesIO(raw), header=header, names=cols,
                     usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    tt = df["transact_time"].astype("int64")
    unit = np.where(tt > 10**14, 10**6, np.where(tt > 10**11, 10**3, 1))
    df["ts"] = (tt // unit).astype("int64")
    maker_sell = ~df["is_buyer_maker"].astype(bool)  # taker BUY volume
    g = df.groupby("ts")
    bars = pd.DataFrame({
        "ts": g["price"].first().index.values,
        "open": g["price"].first().values,
        "high": g["price"].max().values,
        "low": g["price"].min().values,
        "close": g["price"].last().values,
        "volume": g["quantity"].sum().values,
        "count": g.size().values,
        "taker_buy_volume": g.apply(
            lambda x: x.loc[maker_sell.reindex(x.index, fill_value=False), "quantity"].sum(),
            include_groups=False).reindex(g.size().index).fillna(0.0).values,
    })
    return bars


def _bars_to_frame(df: pd.DataFrame, from_klines: bool) -> pd.DataFrame:
    if from_klines:
        out = pd.DataFrame({
            "ts": df["ts"].astype("int64"),
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["volume"].astype("float64"),
            "trades": df["count"].astype("float64"),
            "taker_buy_volume": df["taker_buy_volume"].astype("float64"),
        })
    else:
        out = df.rename(columns={"count": "trades"})[
            ["ts", "open", "high", "low", "close", "volume", "trades", "taker_buy_volume"]
        ]
    out = out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return out


def gap_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a strict 1-second grid; ffill prices, zero volumes, add gap flag."""
    if df.empty:
        return df
    full = pd.DataFrame({"ts": np.arange(df["ts"].iloc[0], df["ts"].iloc[-1] + 1, dtype="int64")})
    m = full.merge(df, on="ts", how="left")
    m["was_gap"] = m["close"].isna().to_numpy()
    for c in ("open", "high", "low", "close"):
        m[c] = m[c].ffill()
    first_valid = m["close"].first_valid_index()
    m = m.loc[first_valid:] if first_valid is not None else m.iloc[0:0]
    m[["volume", "trades", "taker_buy_volume"]] = m[["volume", "trades", "taker_buy_volume"]].fillna(0.0)
    return m.reset_index(drop=True)


def write_shard(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def _download_month(cfg: Config, year: int, month: int, today: dt.date) -> pd.DataFrame | None:
    stamp = f"{year:04d}-{month:02d}"
    url = _klines_url("monthly", cfg, stamp)
    r = _get(url)
    if r is not None and (not cfg.verify_checksums or _verify(url, r.content)):
        print(f"  [klines:monthly] {stamp}")
        return _bars_to_frame(_read_kline_zip(r.content), True)

    # current/partial month -> stitch daily files
    frames = []
    d0 = dt.date(year, month, 1)
    d1 = min(today, dt.date.fromisoformat(cfg.end))
    if d0 <= d1 and (year, month) == (today.year, today.month):
        for d in _day_iter(d0, d1):
            url = _klines_url("daily", cfg, d.isoformat())
            r = _get(url)
            if r is not None:
                frames.append(_read_kline_zip(r.content))
        if frames:
            print(f"  [klines:daily x{len(frames)}] {stamp}")
            return _bars_to_frame(pd.concat(frames, ignore_index=True), True)

    if cfg.fallback_to_aggtrades:
        url = _trades_url("monthly", cfg, stamp)
        r = _get(url)
        if r is not None:
            print(f"  [aggTrades->1s bars] {stamp}")
            return _bars_to_frame(_read_aggtrades_zip(r.content), False)
    print(f"  [MISSING] {stamp}")
    return None


def download_range(cfg: Config) -> list[Path]:
    out = Path(cfg.out_dir)
    start = dt.date.fromisoformat(cfg.start)
    end = dt.date.fromisoformat(cfg.end)
    today = dt.date.today()
    written: list[Path] = []
    for year, month in _month_iter(start, end):
        shard = out / f"{cfg.symbol}-{cfg.interval}-{year:04d}-{month:02d}.parquet"
        if shard.exists() and not (year, month) == (today.year, today.month):
            print(f"  [skip, exists] {shard.name}")
            written.append(shard)
            continue
        df = _download_month(cfg, year, month, today)
        if df is None:
            continue
        if cfg.gap_fill:
            df = gap_fill(df)
        write_shard(df, shard)
        written.append(shard)
        print(f"  [wrote] {shard}  ({len(df):,} rows)")
    print(f"Done. {len(written)} monthly shards in '{out}'.")
    return written


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Download BTCUSDT 1s market data (Binance archive).")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--symbol", default=None)
    p.add_argument("--interval", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", dest="out_dir", default=None)
    args = p.parse_args(argv)
    cfg = _load_config(args.config, {k: v for k, v in vars(args).items() if k != "config"})
    download_range(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
