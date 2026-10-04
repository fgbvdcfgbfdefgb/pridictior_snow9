"""
snow9.training.ddp_launch — multi-GPU entry (Phase 2B).
=========================================================

Launch one independent Price Predictor per GPU, each with slightly different
training parameters (configs/variants.yaml), exactly as requested:

    torchrun --standalone --nproc_per_node=4 \
        -m snow9.training.ddp_launch --data-cache data/cache --out checkpoints/

On your Snowflake box (4 × A10 23 GB, 48 vCPU, 100 GB RAM) that runs the four
variants v0_base / v1_wide / v2_deep / v3_smooth simultaneously, one per GPU.

This is deliberately NOT gradient-averaged DDP: the market stream is cheap to
replicate on each rank, so every rank owns its full model + optimiser and its
own checkpoint directory. The four models form the ensemble that the
visualiser later draws side-by-side.

Single-process run (CPU / 1 GPU) for smoke tests:

    python -m snow9.training.ddp_launch --data-cache data/cache \
        --variant-override smoke --max-sec 2000
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

from ..data.dataset import ContinuousFeed
from ..models.predictor import PricePredictor, param_count
from .online import StreamingRewardTrainer

ARCH_KEYS = ("d_model", "nhead", "num_layers", "ff_dim", "dropout")


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        if world > 1:
            torch.distributed.init_process_group("nccl")
    else:
        device = torch.device("cpu")
        if world > 1:
            torch.distributed.init_process_group("gloo")
    return rank, world, local, device


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-cache", default="data/cache")
    p.add_argument("--variants", default="configs/variants.yaml")
    p.add_argument("--training", default="configs/training.yaml")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--start-ts", type=int, default=None)
    p.add_argument("--end-ts", type=int, default=None)
    p.add_argument("--max-sec", type=int, default=None,
                   help="cap the stream length (useful for smoke tests)")
    p.add_argument("--variant-override", default=None,
                   help="force a small test config, ignoring per-rank variants")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    rank, world, local, device = _setup_distributed()
    vcfg = yaml.safe_load(open(args.variants))
    tcfg = yaml.safe_load(open(args.training)) or {}
    variants = list(vcfg["variants"])
    variant = dict(variants[rank % len(variants)])
    if args.variant_override == "smoke":
        variant = {"name": "smoke", "d_model": 64, "nhead": 4, "num_layers": 2,
                   "ff_dim": 128, "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0,
                   "smooth_penalty": 1.0, "batch_updates_every": 8, "seed": 7}
    variant["name"] = f"{variant['name']}_r{rank}" if world > 1 else variant["name"]

    torch.manual_seed(int(variant.get("seed", 0)) + rank)
    random.seed(int(variant.get("seed", 0)) + rank)
    np.random.seed((int(variant.get("seed", 0)) + rank) % (2**31))

    feed = ContinuousFeed(args.data_cache)
    lookback = int(vcfg.get("lookback_sec", 43200))
    horizon = int(vcfg.get("horizon_sec", 1500))
    if args.variant_override == "smoke":
        lookback, horizon = 600, 150

    arch = {k: variant[k] for k in ARCH_KEYS if k in variant}
    model = PricePredictor(n_features=feed.n_features,
                           quantiles=vcfg.get("quantiles", [0.1, 0.5, 0.9]),
                           **arch).to(device)
    if rank == 0:
        print(f"[{variant['name']}] device={device} params={param_count(model)/1e6:.1f}M "
              f"lookback={lookback}s horizon={horizon}s world={world}")

    trainer = StreamingRewardTrainer(
        model, feed, device, args.out, variant,
        lookback=lookback, horizon=horizon,
        precision=tcfg.get("precision", "bf16"),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        ema_decay=float(tcfg.get("ema_decay", 0.999)),
        log_every_sim_sec=int(tcfg.get("log_every_sim_sec", 60)),
        checkpoint_every_sim_sec=int(tcfg.get("checkpoint_every_sim_sec", 1800)),
        keep_all_checkpoints=bool(tcfg.get("keep_all_checkpoints", True)),
        rank=rank, resume=not args.no_resume)
    trainer.run(start_ts=args.start_ts, end_ts=args.end_ts, max_sec=args.max_sec)
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
