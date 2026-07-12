"""Closing-line value (CLV) helpers for pick tracking and backtests."""

from __future__ import annotations

import math
from typing import Any


def american_to_implied_prob(odds: int | None) -> float | None:
    if odds is None:
        return None
    # ESPN/EVEN sometimes arrives as 0; treat as +100 (even money → 50%).
    if odds == 0:
        odds = 100
    # American odds require |x| >= 100; reject garbage like +50 / -50.
    if abs(odds) < 100:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way(home_odds: int, away_odds: int) -> tuple[float, float] | None:
    """Remove vig from a two-way market. Returns None when either price is invalid."""
    home_raw = american_to_implied_prob(home_odds)
    away_raw = american_to_implied_prob(away_odds)
    if home_raw is None or away_raw is None:
        return None
    total = home_raw + away_raw
    if total <= 0:
        return None
    return home_raw / total, away_raw / total


def clv_vs_market_pct(pick_odds: int, market_odds: int) -> float | None:
    """Implied-probability CLV (industry standard).

    Positive = pick price better than reference market (lower implied prob /
    higher payout than the closing/reference price).
    """
    pick_prob = american_to_implied_prob(pick_odds)
    market_prob = american_to_implied_prob(market_odds)
    if pick_prob is None or market_prob is None or market_prob <= 0:
        return None
    return round((market_prob - pick_prob) / market_prob * 100.0, 2)


def clv_vs_market_pct_threeway(
    pick_odds: int,
    *,
    side: str,
    market_home: int | None,
    market_draw: int | None,
    market_away: int | None,
) -> float | None:
    """Soccer 1X2 CLV: recorded pick price vs closing price on the same outcome."""
    market_by_side = {
        "home": market_home,
        "draw": market_draw,
        "away": market_away,
    }
    market_odds = market_by_side.get(side)
    if market_odds is None:
        return None
    try:
        return clv_vs_market_pct(int(pick_odds), int(market_odds))
    except (TypeError, ValueError):
        return None


def model_edge_vs_devigged_market(
  model_home_prob: float,
  home_odds: int | None,
  away_odds: int | None,
) -> dict[str, Any]:
    empty = {
        "market_home_prob": None,
        "model_minus_market_pp": None,
    }
    if home_odds is None or away_odds is None:
        return empty
    probs = devig_two_way(home_odds, away_odds)
    if probs is None:
        return empty
    market_home, _market_away = probs
    model_p = model_home_prob / 100.0
    return {
        "market_home_prob": round(market_home * 100.0, 2),
        "model_minus_market_pp": round((model_p - market_home) * 100.0, 2),
    }


def log_loss_binary(model_prob_pct: float, outcome_home_win: bool) -> float:
    prob = min(max(model_prob_pct / 100.0, 1e-6), 1.0 - 1e-6)
    return -math.log(prob if outcome_home_win else 1.0 - prob)
