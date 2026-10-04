# pridictior_snow9 — Bitcoin 25-min-ahead Price Predictor

Second-by-second BTC predictor trained with a **streaming real-time reward** on
your Snowflake GPU box (**4 × A10 23 GB, 48 vCPU, 100 GB RAM**), plus a
live-market simulator and a 30 fps prediction animation.

> **Not financial advice.** This is a research/training project. Nothing here is a
> recommendation to trade. Predicting 25 min ahead in crypto is genuinely hard —
> treat the accuracy bars in the video as the honest scoreboard.

---

## Architecture

```
Binance archive ──(Phase 1)──► data/raw/*.parquet  (1-sec klines, 2020→today)
                                    │
                              build_feature_cache (vectorised, CPU)
                                    ▼
                    data/cache/{features,close,ts} memmaps
                                    │
   ┌────────────────────────────────┼───────────────────────────────┐
   │  MarketFeedSimulator           │ replays the market 1 s at a time
   │  MarketAnalyser  (CPU)  ──────►│ 23 features / second, ffill-gapped
   │                                ▼
   │  PricePredictor ×4 (one per A10 GPU)
   │    causal-conv 43200s→720 tok  +  Transformer ×N  →  q10/q50/q90 of P(t+1500s)
   │    reward: pinball  +  λ·smoothness(vis trading-stable forecasts)
   │    NO EPOCHS — every second matures a reward 1500 s later, step immediately
   └────────────────────────────────┬───────────────────────────────┘
                                    ▼
                    checkpoints/<variant>/  (full snapshot per 30 market-min, kept forever)
                                    │
                              viz: rollout random day (2020–2026) → 30 fps MP4
```

| Component | Runs on | Purpose |
|---|---|---|
| `MarketAnalyzer` | CPU | 23 per-second features from the simulated feed |
| `PricePredictor` | GPU — **one model per GPU** (`v0_base`, `v1_wide`, `v2_deep`, `v3_smooth`) | price 25 min ahead, refreshed every second |

Model size per variant: **≈28 M params** (`v0_base`/`v3_smooth`), **≈45 M** (`v1_wide`,
768-wide), **≈20 M×10 layers** (`v2_deep`). Lookback is the full 12 h = 43,200 one-second
rows, as specified — the conv front-end downsamples it 60× so a 43.2 k-second window is
one forward pass.

---

## Quick start (machine WITH internet — your laptop, not Snowflake)

```bash
git clone <your-fork>/pridictior_snow9.git && cd pridictior_snow9
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt

# Phase 1 — full dataset (BTCUSDT, 1-sec, 2020→today; resumable; ~4-8 GB total)
scripts/01_download_data.sh              # -> data/raw/BTCUSDT-1s-YYYY-MM.parquet

# sanity-check the whole pipeline on the committed 1-day sample (CPU, ~2 min)
PYTHONPATH=src python scripts/smoke_test.py
```

The downloader hits `data.binance.vision` monthly zips (with `.CHECKSUM`
verification), falls back to daily files for the current month, and can rebuild
1-sec bars from raw `aggTrades` if klines are missing for an old month.
Every shard stores open/high/low/close/volume/trades/taker-buy-volume; missing
seconds are forward-filled and flagged (`was_gap`) so the model knows.

> **GitHub size note:** the full multi-GB dataset does **not** belong in git
> (GitHub caps files at 100 MB / LFS at 1 GB). The repo carries the downloader +
> one committed sample day (`data/sample/`); everyone regenerates the full data
> with `scripts/01_download_data.sh`. Checked into the repo is everything else.

---

## Snowflake (OFFLINE — no internet inside)

**One-time, from a machine that has internet:**

```bash
scripts/01_download_data.sh                    # get data/raw/*.parquet
zip -r pridictior_snow9.zip . -x "data/raw/*" "checkpoints/*"
snowsql -f snowflake/upload_data_to_stage.sql  # creates SNOW9.PUBLIC.BTC_SNOW9_STAGE
# then in snowsql (paths/macro shown in the .sql file):
#   PUT file:///.../data/raw/*.parquet  @SNOW9.PUBLIC.BTC_SNOW9_STAGE/data/ ...
#   PUT file:///.../pridictior_snow9.zip @SNOW9.PUBLIC.BTC_SNOW9_STAGE/code/ ...
```

**Inside Snowflake:** import `snowflake/BTC_Price_Predictor_Snowflake.ipynb`
and run top-to-bottom. It installs packages from PyPI (allowed), reads
data only from the stage (offline-safe), builds features, launches
`torchrun --nproc_per_node=4` (one variant per A10), then replays a random day
and shows the MP4 inline.

Equivalent CLI on the Snowflake compute pool:

```bash
scripts/02_build_features.sh                   # -> data/cache/ memmaps
scripts/03_train_distributed.sh                # 4 GPUs × 4 variants, streaming reward
scripts/04_visualize.sh random 86400           # random day -> videos/predictions_random.mp4
```

---

## Training protocol (as specified)

- **Real-time reward, zero epochs.** The simulator streams 2020→today
  second-by-second at max speed. Each second `t` the model predicts
  `price(t+1500)`. At `t+1500` the *stored* market truth matures the prediction
  into `pinball(q10,q50,q90) + λ·smoothness` and an immediate AdamW step
  (every 32 matured rewards → batch update).
- **Stability.** `smoothness_penalty` punishes second-to-second jumps of the
  median forecast normalised by running 25-min realised vol — forecasts only
  swing when the market itself is swinging. `v3_smooth` cranks λ×3.
- **One variant per GPU, slightly different hyper-parameters** (width / depth /
  lr / smoothness), seeded separately, streaming the same market feed. The
  animation draws all of them side-by-side so you can watch the ensemble agree
  and disagree.
- **Checkpointing = keep everything.** Full snapshot (model, optimiser, EMA
  weights, trainer position) every 30 **market**-minutes per variant, plus a
  stable `latest.pt` alias. With 1 PB you never delete a checkpoint; set
  `keep_all_checkpoints: false` in `configs/training.yaml` to keep only the
  last 3.
- **Resume-safe.** Re-running `03_train_distributed.sh` picks up at
  `latest.pt` — the stream continues where it stopped, never re-sees data.

## Files

```
configs/            data / variants / training YAML
data/sample/        one committed day of real BTCUSDT 1-sec bars (for smoke tests)
src/snow9/
  data/             downloader (+ aggTrades fallback), shard store, feature cache
  simulator/        MarketFeedSimulator — replays history like a live feed
  models/           analyzer.py (CPU features), predictor.py (conv+transformer)
  training/         rewards.py, online.py (streaming trainer), ddp_launch.py
  viz/              rollout.py (random-day replay), animate.py (30 fps MP4)
scripts/            01→04 pipeline scripts + smoke_test.py
snowflake/          offline notebook + stage-upload SQL
```

## Repo hygiene / security

- **Never commit credentials.** `.gitignore` blocks `.env`, `*.key`, tokens.
  If a token is ever pasted anywhere (chats included), rotate it at
  <https://github.com/settings/tokens> — GitHub also auto-revokes tokens it
  spots in public pushes.
- Data / checkpoints / videos are regenerated artifacts — excluded from git by
  design.
