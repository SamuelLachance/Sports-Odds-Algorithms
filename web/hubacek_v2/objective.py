"""Decorrelated learning objective (Hubáček et al. 2019, §4.4).

Loss per example (binary, raw score s, p = sigmoid(s), outcome y, bookmaker's
de-vigged probability q):

    L = -[y·log p + (1-y)·log(1-p)] - c·(p - q)²

The second term REWARDS distance from the bookmaker's estimate, trading
accuracy for decorrelation. The paper found c in [0.4, 0.8] optimal on NBA
(their base loss was MSE; we keep log-loss as the base to stay consistent
with the rest of the pipeline — the decorrelation term is identical).

Rows without a known book probability contribute the plain log-loss only.

CatBoost integration: used as a custom objective on CatBoostRegressor over
raw scores (predict -> sigmoid -> probability). CatBoost maximizes, so the
derivatives returned are those of -L w.r.t. s:

    d(-L)/ds   = (y - p) + 2c·(p - q)·p(1-p)
    d²(-L)/ds² = -[p(1-p) - 2c·p(1-p)·(p(1-p) + (p - q)(1 - 2p))]

The second derivative is clamped to stay negative for training stability
(same treatment CatBoost applies to near-flat regions of built-in losses).
"""

from __future__ import annotations

import numpy as np

_HESS_FLOOR = 1e-6


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def decorrelated_loss(
    scores: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    c: float,
) -> float:
    """Mean decorrelated loss (for numeric tests / torch parity)."""
    p = np.clip(sigmoid(np.asarray(scores, dtype=float)), 1e-12, 1 - 1e-12)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    dec = np.where(np.isfinite(q), (p - q) ** 2, 0.0)
    return float(np.mean(ll - c * dec))


def decorrelated_ders(
    scores: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(d(-L)/ds, d²(-L)/ds²) with the stability clamp applied."""
    s = np.asarray(scores, dtype=float)
    p = sigmoid(s)
    pq = np.where(np.isfinite(q), p - q, 0.0)
    has_q = np.isfinite(q).astype(float)
    w = p * (1.0 - p)
    der1 = (y - p) + 2.0 * c * pq * w * has_q
    der2 = -(w - 2.0 * c * w * (w + pq * (1.0 - 2.0 * p)) * has_q)
    der2 = np.minimum(der2, -_HESS_FLOOR)
    return der1, der2


class DecorrelatedLogloss:
    """CatBoost custom objective carrying the per-row book probabilities.

    ``q`` must be aligned with the TRAINING pool's row order (NaN where no
    market exists). CatBoost calls calc_ders_range with global document
    indices preserved, so positional alignment holds.
    """

    def __init__(self, q: np.ndarray, c: float):
        self.q = np.asarray(q, dtype=float)
        self.c = float(c)

    def calc_ders_range(self, approxes, targets, weights):
        s = np.asarray(approxes, dtype=float)
        y = np.asarray(targets, dtype=float)
        n = len(s)
        q = self.q[:n] if len(self.q) >= n else np.full(n, np.nan)
        der1, der2 = decorrelated_ders(s, y, q, self.c)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            der1 = der1 * w
            der2 = der2 * w
        return list(zip(der1.tolist(), der2.tolist()))
