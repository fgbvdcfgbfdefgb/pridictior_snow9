"""
snow9.models.predictor — Phase 2B (GPU): Price Predictor.
============================================================

Predicts BTC price 25 minutes (1500 s) ahead, refreshed every second.

Input   : (B, 43200, F)  — the Market Analyser's last-12-hour feature window
Output  : dict with
  * `q_logret`     (B, Q)  — predicted quantiles of the 1500-s log-return
  * `q_price`      (B, Q)  — the same, mapped to price via the last close
  * `median_price` (B,)    — tradable point forecast

Architecture (≈28M params for the v0_base variant, larger for v1_wide):
  1. Causal conv front-end downsamples 43,200 s -> 720 minute-tokens
     (strides 4×3×5 = 60). Purely causal: no leakage from the future.
  2. Pre-norm Transformer encoder over the 720 tokens (learnable positions).
  3. Quantile head on the last (most recent) token. Quantiles are sorted so
     q10 ≤ q50 ≤ q90 by construction.

The 12-h window is accepted as-is; any shorter window (warm-up, tests) also
works because positions are taken from the tail of the table.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, s: int):
        super().__init__()
        self.pad = nn.ConstantPad1d((k - 1, 0), 0.0)
        self.conv = nn.Conv1d(cin, cout, k, stride=s)
        self.norm = nn.GroupNorm(8, cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))


class CausalConvFront(nn.Module):
    def __init__(self, in_ch: int, channels=(64, 128, 256), strides=(4, 3, 5), kernel: int = 5):
        super().__init__()
        blocks = []
        c = in_ch
        for ch, s in zip(channels, strides):
            blocks.append(_CausalConvBlock(c, ch, kernel, s))
            c = ch
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = c

    def forward(self, x):           # (B, F, T)
        return self.blocks(x)       # (B, C, ceil(T/60))


class PricePredictor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 8, ff_dim: int = 2048, dropout: float = 0.1,
                 quantiles=(0.1, 0.5, 0.9), max_tokens: int = 768, **_ignored):
        super().__init__()
        self.front = CausalConvFront(n_features)
        self.in_proj = nn.Linear(self.front.out_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, len(quantiles)))
        self.register_buffer("taus", torch.tensor(list(quantiles), dtype=torch.float32))

    def _pos(self, tokens: int) -> torch.Tensor:
        if tokens <= self.pos.shape[1]:
            return self.pos[:, -tokens:]
        p = self.pos.transpose(1, 2)
        return F.interpolate(p, size=tokens, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x: torch.Tensor, last_close: torch.Tensor) -> dict:
        # x: (B, T, F) ; last_close: (B,)
        z = self.front(x.transpose(1, 2))          # (B, C, tok)
        h = self.in_proj(z.transpose(1, 2))        # (B, tok, d)
        h = h + self._pos(h.shape[1])
        h = self.enc(h)
        q_logret = self.head(self.ln(h[:, -1]))    # (B, Q)
        q_logret, _ = torch.sort(q_logret, dim=-1)  # monotone quantiles
        q_price = last_close.unsqueeze(-1) * torch.exp(q_logret)
        return {"q_logret": q_logret, "q_price": q_price,
                "median_price": q_price[..., q_price.shape[-1] // 2]}


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
