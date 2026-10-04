"""
snow9.training.rewards — the real-time reward, exactly as specified.
======================================================================

There are no epochs. Every market-second the predictor issues a 25-min-ahead
forecast; 1500 seconds later the *stored, non-simulated* market data reveals
what actually happened, and the forecast matures into a reward → the model is
updated immediately (delayed-supervision online learning):

    reward_t  =  − loss(forecast_{t−1500}, truth_t) / vol_scale

Loss = pinball(quantile regression)  +  λ · smoothness_penalty

* pinball — trains the q10/q50/q90 band (calibrated uncertainty)
* smoothness — penalises erratic second-to-second jumps of the median
  forecast relative to current realised volatility, keeping the signal
  stable enough to trade on
"""
from __future__ import annotations

import torch


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """q_pred: (B, Q) predicted log-return quantiles ; y: (B,) true log-return."""
    d = y.unsqueeze(-1) - q_pred
    return torch.maximum(taus * d, (taus - 1.0) * d).mean()


def smoothness_penalty(med_price: torch.Tensor, prev_med_price: torch.Tensor,
                       ref_price: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    """
    Penalise consecutive-second forecast jumps, normalised by how much the
    market was actually moving (1500-s realised vol), so the constraint
    tightens in quiet markets and relaxes in storms.
    med/prev/ref: (B,) prices ; vol: (B,) fractional 1500-s stdev.
    """
    jump = (med_price - prev_med_price) / (ref_price + 1e-12)
    denom = vol.clamp_min(1e-5)
    return ((jump / denom) ** 2).mean()


def mape(pred_price: torch.Tensor, true_price: torch.Tensor) -> torch.Tensor:
    return ((pred_price - true_price).abs() / (true_price + 1e-12)).mean()


def band_hit(lo: torch.Tensor, hi: torch.Tensor, true_price: torch.Tensor) -> torch.Tensor:
    return ((true_price >= lo) & (true_price <= hi)).float().mean()


def reward_from_loss(loss: torch.Tensor, baseline: float = 0.01) -> torch.Tensor:
    """Scalar reward for logging/potential RL fine-tuning: negative, vol-free loss."""
    return -(loss / (baseline + 1e-12))
