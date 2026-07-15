"""Markowitz max-Sharpe bet allocation per round (Hubáček et al. 2019, §5.1).

One round = one slate of betting opportunities. The bettor spends a unit
budget across the +EV opportunities (p̂·o > 1), never both sides of the same
game, allocating to maximize the Sharpe ratio Ê[P]/σ̂[P] where

    Ê[P]   = Σ (p̂ᵢ·oᵢ - 1)·bᵢ
    V̂ar[P] = Σ (1-p̂ᵢ)·p̂ᵢ·bᵢ²·oᵢ²   (pairwise independence within a round)

solved by SLSQP on the simplex (paper: sequential quadratic programming).
If no opportunity has positive expected value, no bets are placed.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def expected_profit(probs: np.ndarray, odds: np.ndarray, bets: np.ndarray) -> float:
    return float(np.sum((probs * odds - 1.0) * bets))


def profit_variance(probs: np.ndarray, odds: np.ndarray, bets: np.ndarray) -> float:
    return float(np.sum((1.0 - probs) * probs * (bets ** 2) * (odds ** 2)))


def max_sharpe_bets(
    probs: np.ndarray,
    odds: np.ndarray,
    *,
    game_ids: np.ndarray | None = None,
    max_bet: float | None = None,
) -> np.ndarray:
    """Unit-budget allocation maximizing the estimated Sharpe ratio.

    ``probs``/``odds`` describe candidate outcomes (already φ-filtered by the
    caller). Only outcomes with p̂·o > 1 receive positive bets; when two
    candidates share a ``game_id`` only the higher-EV side stays eligible
    (the paper's variance formula assumes one bet per game).
    Returns the bet vector (fractions of the round budget, sum 1 over the
    eligible set, all-zero when nothing is +EV).
    """
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    n = len(probs)
    bets = np.zeros(n)
    ev = probs * odds - 1.0
    eligible = ev > 0
    if game_ids is not None:
        gid = np.asarray(game_ids)
        for g in np.unique(gid):
            idx = np.where((gid == g) & eligible)[0]
            if len(idx) > 1:
                keep = idx[np.argmax(ev[idx])]
                drop = [i for i in idx if i != keep]
                eligible[drop] = False
    idx = np.where(eligible)[0]
    if len(idx) == 0:
        return bets
    if len(idx) == 1:
        bets[idx[0]] = 1.0
        return bets

    p, o = probs[idx], odds[idx]
    k = len(idx)

    def neg_sharpe(b: np.ndarray) -> float:
        e = np.sum((p * o - 1.0) * b)
        var = np.sum((1.0 - p) * p * (b ** 2) * (o ** 2))
        return -e / max(np.sqrt(var), _EPS)

    from scipy.optimize import minimize

    upper = 1.0 if max_bet is None else float(max_bet)
    x0 = np.full(k, 1.0 / k)
    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=[(0.0, upper)] * k,
        constraints=[{"type": "eq", "fun": lambda b: np.sum(b) - 1.0}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    sol = res.x if res.success else x0
    sol = np.clip(sol, 0.0, None)
    total = sol.sum()
    if total <= 0:
        sol = np.full(k, 1.0 / k)
        total = 1.0
    bets[idx] = sol / total
    return bets


def uniform_bets(
    probs: np.ndarray,
    odds: np.ndarray,
    *,
    game_ids: np.ndarray | None = None,
) -> np.ndarray:
    """The paper's ``unif`` baseline: equal split across +EV opportunities."""
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    bets = np.zeros(len(probs))
    ev = probs * odds - 1.0
    eligible = ev > 0
    if game_ids is not None:
        gid = np.asarray(game_ids)
        for g in np.unique(gid):
            idx = np.where((gid == g) & eligible)[0]
            if len(idx) > 1:
                keep = idx[np.argmax(ev[idx])]
                drop = [i for i in idx if i != keep]
                eligible[drop] = False
    idx = np.where(eligible)[0]
    if len(idx):
        bets[idx] = 1.0 / len(idx)
    return bets
