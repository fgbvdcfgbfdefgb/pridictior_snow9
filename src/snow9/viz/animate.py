"""
snow9.viz.animate — Phase 2C (step 2): the 30 fps trading animation.
=====================================================================

Renders the rollout as a living chart, one video frame at a time:

  LEFT   actual market price (solid)
         + each model's 25-min-ahead forecast as a DOTTED ray from
           "now" into the future (re-drawn every simulated second)
         + fading dotted trails of past forecasts vs. what happened
  RIGHT  per-model accuracy bars (rolling MAPE % and rolling 10–90 band
         hit-rate vs. the actual stored market), updated live

Saves 30 fps MP4 (H.264 via imageio-ffmpeg, no system ffmpeg needed) and
falls back to GIF automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODEL_COLORS = ["#4cc9f0", "#f72585", "#ffd60a", "#80ed99", "#b5179e", "#ff9f1c"]


def render_animation(npz_path: str, out: str = "videos/predictions.mp4",
                     fps: int = 30, seconds_per_frame: int = 30,
                     accuracy_window: int = 1800, dpi: int = 110,
                     max_seconds: int | None = None, trail_sec: int = 1800) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter
    import datetime as _dt

    def _hhmm(v, _pos):
        try:
            return _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc).strftime("%H:%M")
        except (OverflowError, OSError, ValueError):
            return ""

    d = np.load(npz_path, allow_pickle=False)
    ts, close, horizon = d["ts"], d["close"], int(d["horizon"])
    day = str(d["date"]) if "date" in d else ""
    models = sorted({k[:-5] for k in d.files if k.endswith("__med")})
    n = len(ts)
    if max_seconds:
        n = min(n, max_seconds)
        ts, close = ts[:n], close[:n]
    secs = np.arange(n)
    truth_at = np.clip(secs + horizon, 0, len(close) - 1)   # actual price at target time

    # --- precompute per-model accuracy series (vs stored market truth) ------
    acc = {}
    for m in models:
        med = d[f"{m}__med"][:n]; lo = d[f"{m}__lo"][:n]; hi = d[f"{m}__hi"][:n]
        err = np.abs(med - close[truth_at]) / close[truth_at] * 100
        hit = ((close[truth_at] >= lo) & (close[truth_at] <= hi)).astype(float)
        matured = secs + horizon < n  # only score when stored truth exists
        w = accuracy_window
        acc[m] = {
            "mape": np.convolve(np.where(matured, err, 0.0), np.ones(w) / w, "same"),
            "hit": np.convolve(np.where(matured, hit, 0.5), np.ones(w) / w, "same"),
            "med": med, "lo": lo, "hi": hi,
        }

    frames = list(range(horizon + 1, n, max(1, seconds_per_frame)))
    if not frames:
        raise ValueError("Rollout too short for the given horizon / seconds_per_frame.")
    stride_hist = max(1, n // 2500)  # keep history line cheap to redraw

    plt.rcParams.update({"text.color": "#e8e8e8", "axes.edgecolor": "#444",
                         "axes.labelcolor": "#e8e8e8", "xtick.color": "#aaa",
                         "ytick.color": "#aaa", "font.size": 9})
    fig = plt.figure(figsize=(11.5, 5.6), dpi=dpi, facecolor="#0d1117")
    gs = GridSpec(1, 2, width_ratios=[3.1, 1.0], figure=fig,
                  left=0.06, right=0.97, top=0.90, bottom=0.10, wspace=0.14)
    ax = fig.add_subplot(gs[0]); ax.set_facecolor("#0d1117")
    axb = fig.add_subplot(gs[1]); axb.set_facecolor("#0d1117")
    fig.suptitle(f"BTCUSDT — 25-min-ahead predictions   |   {day} UTC   |   replay ×{seconds_per_frame}",
                 fontsize=11)

    def draw(f_i: int):
        ax.clear(); axb.clear()
        ax.set_facecolor("#0d1117"); axb.set_facecolor("#0d1117")
        i = frames[f_i]
        t_now, p_now = ts[i], close[i]
        clock = np.datetime64(int(t_now), "s").astype(str).replace("T", "  ")
        # actual price: crisp up to now, faint ghost beyond
        ax.plot(ts[:i + 1:stride_hist], close[:i + 1:stride_hist],
                color="#f0f0f0", lw=1.4, label="actual (stored market)")
        ax.plot(ts[i::stride_hist], close[i::stride_hist], color="#ffffff", lw=0.8, alpha=0.15)
        ax.axvline(t_now, color="#555", lw=0.8, ls=":")
        y_lo, y_hi = close[max(0, i - 4 * horizon):i + horizon].min(), \
                     close[max(0, i - 4 * horizon):i + horizon].max()
        pad = (y_hi - y_lo) * 0.08 + 1e-9
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlim(t_now - 4 * horizon, t_now + 1.6 * horizon)
        ax.xaxis.set_major_formatter(FuncFormatter(_hhmm))
        ax.set_title(f"market time (UTC): {clock}", loc="left", fontsize=9, color="#9d9")
        ax.set_xlabel(f"{day} — time (UTC)"); ax.set_ylabel("USD")

        bars_y, hit_vals, mape_vals, colors, names = [], [], [], [], []
        for j, m in enumerate(models):
            c = MODEL_COLORS[j % len(MODEL_COLORS)]
            a = acc[m]
            # dotted 25-min-ahead forecast ray
            ax.plot([t_now, t_now + horizon], [p_now, a["med"][i]],
                    color=c, lw=1.8, ls=(0, (1.5, 1.6)), marker="o", ms=3.2,
                    label=f"{m} → +25 min")
            ax.fill_between([t_now, t_now + horizon], [p_now, a["lo"][i]],
                            [p_now, a["hi"][i]], color=c, alpha=0.12, lw=0)
            # trail of still-open past forecasts (issued in the last 30 min)
            past = np.arange(max(0, i - trail_sec), i, 30)
            if len(past):
                ax.plot(np.minimum(past + horizon, i), a["med"][past].clip(y_lo, y_hi),
                        color=c, lw=0, marker=".", ms=2.0, alpha=0.35)
            bars_y.append(j); hit_vals.append(a["hit"][i] * 100)
            mape_vals.append(a["mape"][i]); colors.append(c); names.append(m)
        ax.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#333")

        axb.barh(bars_y, hit_vals, color=colors, alpha=0.9, height=0.55)
        axb.axvline(80, color="#888", ls="--", lw=1)
        axb.text(80.5, len(models) - 0.35, "80% target", fontsize=7, color="#888")
        for j, (hv, mv) in enumerate(zip(hit_vals, mape_vals)):
            axb.text(hv + 1, j, f"{hv:4.1f}%  MAPE {mv:5.3f}%", va="center", fontsize=8)
        axb.set_yticks(bars_y, names); axb.set_xlim(0, 108)
        axb.set_xlabel("rolling band hit-rate (%)")
        axb.set_title("accuracy vs real market", fontsize=9, loc="left", color="#9d9")

    # ---------------------------------------------------------------- writer
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=16)
        mode = "mp4"
    except Exception:
        from matplotlib.animation import PillowWriter
        writer, mode = None, "gif"
        out = str(Path(out).with_suffix(".gif"))

    if mode == "mp4":
        with writer:
            for f_i in range(len(frames)):
                draw(f_i)
                fig.canvas.draw()
                img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(img)
    else:
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
        anim.save(out, writer=PillowWriter(fps=fps, dpi=dpi))
    plt.close(fig)
    print(f"saved {out}  ({len(frames)} frames @ {fps} fps)")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", default="videos/predictions.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds-per-frame", type=int, default=30,
                   help="simulated market seconds advanced per video frame (replay speed)")
    p.add_argument("--accuracy-window", type=int, default=1800)
    p.add_argument("--max-seconds", type=int, default=None)
    p.add_argument("--dpi", type=int, default=110)
    args = p.parse_args(argv)
    render_animation(args.npz, args.out, fps=args.fps,
                     seconds_per_frame=args.seconds_per_frame,
                     accuracy_window=args.accuracy_window,
                     max_seconds=args.max_seconds, dpi=args.dpi)


if __name__ == "__main__":
    main(sys.argv[1:])
