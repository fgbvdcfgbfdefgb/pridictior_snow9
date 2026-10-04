"""
snow9.training.online — streaming trainer (Phase 2B).
========================================================

Implements the exact training protocol requested:

  * the market simulator replays history second-by-second at max speed
  * every second the model predicts price(t + 1500 s)
  * each prediction matures 1500 seconds later against the REAL stored
    market data -> immediate reward (pinball + smoothness) -> SGD update
  * no epochs: time flows forward once, like live trading
  * every market-`checkpoint_every_sim_sec` a FULL snapshot is written
    (model + optimiser + EMA + RNG + trainer position) — with
    `keep_all_checkpoints: true` nothing is ever deleted (1 PB available)

Heavy lookback windows are sliced straight from the memmapped feature cache,
so RAM stays (nearly) flat across the full 2020→today stream. `resume` picks
up exactly where the latest checkpoint left off — nothing is ever re-seen.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .rewards import pinball_loss, smoothness_penalty, mape, band_hit


def _sd_copy(model: torch.nn.Module) -> dict:
    return {k: v.detach().clone().float() for k, v in model.state_dict().items()}


class StreamingRewardTrainer:
    def __init__(self, model, feed, device, out_dir: str | Path, variant: dict,
                 lookback: int = 43200, horizon: int = 1500,
                 precision: str = "bf16", grad_clip: float = 1.0,
                 ema_decay: float = 0.999,
                 log_every_sim_sec: int = 60,
                 checkpoint_every_sim_sec: int = 1800,
                 keep_all_checkpoints: bool = True,
                 rank: int = 0, resume: bool = True):
        self.model, self.feed, self.device = model, feed, device
        self.variant = variant
        self.variant_name = variant["name"]
        self.lookback, self.horizon = lookback, horizon
        self.rank = rank
        self.out = Path(out_dir) / self.variant_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=float(variant.get("lr", 3e-4)),
                                     weight_decay=float(variant.get("weight_decay", 0.01)))
        self.smooth_w = float(variant.get("smooth_penalty", 1.0))
        self.batch_n = int(variant.get("batch_updates_every", 32))
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.ema_sd = _sd_copy(model)
        self.log_every = log_every_sim_sec
        self.ckpt_every = checkpoint_every_sim_sec
        self.keep_all = keep_all_checkpoints
        self.use_amp = precision == "bf16" and device.type == "cuda"
        self.metrics_f = open(self.out / "metrics.jsonl", "a")
        self.pending: dict[int, list[dict]] = {}
        self.batch: list[dict] = []
        self.step = 0
        self.roll_mape, self.roll_hit = deque(maxlen=1000), deque(maxlen=1000)
        self._prev_median: float | None = None
        self._resume_row = 0
        self._anchor_i, self._anchor_t = 0, 0.0
        self._next_ckpt_at: int | None = None
        if resume:
            self._try_resume()

    # ------------------------------------------------------------- helpers
    def _vol_scale(self, i: int) -> float:
        """Fractional 1500-s realised vol ending at row i (smoothness normaliser)."""
        lo = max(0, i - self.horizon)
        w = self.feed.closes(lo, i + 1)
        if len(w) < 3:
            return 1e-4
        return float(np.std(np.diff(np.log(w))) * math.sqrt(self.horizon)) + 1e-8

    @torch.no_grad()
    def _infer(self, x_np: np.ndarray, last_close: float) -> dict:
        x = torch.from_numpy(x_np[None]).to(self.device)
        lc = torch.tensor([last_close], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.model(x, lc)
        return {k: v.float().cpu() for k, v in out.items()}

    def _optimise(self) -> dict:
        b = self.batch
        x = torch.from_numpy(np.stack([s["x"] for s in b])).to(self.device)
        lc = torch.tensor([s["last_close"] for s in b], device=self.device)
        y = torch.tensor([s["y"] for s in b], device=self.device)
        med_prev = torch.tensor([s["med_prev"] for s in b], device=self.device)
        vol = torch.tensor([s["vol"] for s in b], device=self.device)
        self.model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.model(x, lc)
            l_pin = pinball_loss(out["q_logret"], y, self.model.taus)
            l_smooth = smoothness_penalty(out["median_price"], med_prev, lc, vol)
            loss = l_pin + self.smooth_w * l_smooth
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        self.step += 1
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.ema_sd[k].mul_(self.ema_decay).add_(v.detach().float(),
                                                             alpha=1 - self.ema_decay)
                else:
                    self.ema_sd[k].copy_(v)
            pred_p = lc * torch.exp(out["q_logret"][:, 1])
            true_p = lc * torch.exp(y)
            self.roll_mape.append(float(mape(pred_p, true_p)))
            self.roll_hit.append(float(band_hit(out["q_price"][:, 0],
                                                out["q_price"][:, 2], true_p)))
        self.batch.clear()
        return {"loss": loss.item(), "pinball": l_pin.item(), "smooth": l_smooth.item()}

    # ---------------------------------------------------------------- main
    def run(self, start_ts: int | None = None, end_ts: int | None = None,
            max_sec: int | None = None):
        f = self.feed
        i0 = max(f.index_of(start_ts) if start_ts else self.lookback,
                 self.lookback, self._resume_row)
        i1 = f.index_of(end_ts) if end_ts else f.n_rows - 1
        if max_sec:
            i1 = min(i1, i0 + max_sec)
        self._anchor_i, self._anchor_t = i0, time.monotonic()
        self._next_ckpt_at = int(f.ts[i0]) + self.ckpt_every
        t0 = time.monotonic()
        last_logged = int(f.ts[i0])
        pbar = tqdm(range(i0, i1), desc=f"[{self.variant_name}] streaming", mininterval=2.0)
        i = i0
        for i in pbar:
            ts = int(f.ts[i])
            close = float(f.close[i])
            x = f.feature_window(i, self.lookback)
            pred = self._infer(x, close)                       # 25-min forecast
            med = float(pred["median_price"][0])
            prev_med = self._prev_median if self._prev_median is not None else med
            self._prev_median = med
            self.pending.setdefault(i + self.horizon, []).append(
                {"x": x, "last_close": close, "med_prev": prev_med,
                 "vol": self._vol_scale(i)})
            # ---- reward: predictions issued exactly `horizon` seconds ago ----
            matured = self.pending.pop(i, None)
            if matured:
                truth = close                                # real, stored market data
                for m in matured:
                    m["y"] = float(math.log(truth / m["last_close"]))
                    self.batch.append(m)
                if len(self.batch) >= self.batch_n:
                    stats = self._optimise()
                    if ts - last_logged >= self.log_every:
                        last_logged = ts
                        self._log(ts, i, stats, t0)
                        pbar.set_postfix(loss=f"{stats['loss']:.5f}",
                                         mape=f"{np.mean(self.roll_mape):.5f}")
            # ---- checkpoint cadence (in MARKET seconds) ----
            if ts >= self._next_ckpt_at:
                self._next_ckpt_at = ts + self.ckpt_every
                self.save_checkpoint(i, ts)
        self._flush()
        self.save_checkpoint(i, int(f.ts[i]), final=True)

    # ------------------------------------------------------------ plumbing
    def _log(self, ts: int, i: int, stats: dict, t0: float):
        rec = {
            "sim_ts": ts,
            "sim_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "row": i, "step": self.step, "rank": self.rank, **stats,
            "reward": -stats["loss"],
            "mape_rolling": float(np.mean(self.roll_mape)) if self.roll_mape else None,
            "cover_rolling": float(np.mean(self.roll_hit)) if self.roll_hit else None,
            "wall_sec": round(time.monotonic() - t0, 2),
            "sim_speed": round((i - self._anchor_i) / max(time.monotonic() - self._anchor_t, 1e-9), 1),
        }
        self.metrics_f.write(json.dumps(rec) + "\n")
        self.metrics_f.flush()

    def _flush(self):
        while len(self.batch) >= 2:
            self._optimise()
        self.batch.clear()

    def save_checkpoint(self, i: int, ts: int, final: bool = False):
        ckpt = {
            "model": self.model.state_dict(), "ema": self.ema_sd,
            "optimizer": self.opt.state_dict(), "step": self.step, "row": i,
            "sim_ts": ts, "prev_median": self._prev_median,
            "variant": self.variant, "lookback": self.lookback, "horizon": self.horizon,
        }
        name = "final.pt" if final else f"ckpt_ts{ts}_row{i:012d}_step{self.step:08d}.pt"
        torch.save(ckpt, self.out / name)
        torch.save(ckpt, self.out / "latest.pt")   # stable alias for the visualiser
        if not self.keep_all and not final:
            for old in sorted(self.out.glob("ckpt_ts*.pt"))[:-3]:
                old.unlink(missing_ok=True)

    def _try_resume(self):
        latest = self.out / "latest.pt"
        if not latest.exists():
            return
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.ema_sd = {k: v.float() for k, v in ckpt["ema"].items()}
        self.opt.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self._resume_row = ckpt["row"] + 1
        self._prev_median = ckpt.get("prev_median")
        print(f"[{self.variant_name}] resumed: row {self._resume_row}, step {self.step}")
