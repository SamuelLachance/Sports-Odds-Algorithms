"""Closing-line value (CLV) helpers for pick tracking and backtests."""

from __future__ import annotations

import math
from typing import Any


def american_to_implied_prob(odds: int | None) -> float | None:
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way(home_odds: int, away_odds: int) -> tuple[float, float]:
    home_raw = american_to_implied_prob(home_odds) or 0.5
    away_raw = american_to_implied_prob(away_odds) or 0.5
    total = home_raw + away_raw
    if total <= 0:
        return 0.5, 0.5
    return home_raw / total, away_raw / total


def clv_vs_market_pct(pick_odds: int, market_odds: int) -> float | None:
    """Positive CLV = pick price better than reference market (higher payout)."""
    pick_prob = american_to_implied_prob(pick_odds)
    market_prob = american_to_implied_prob(market_odds)
    if pick_prob is None or market_prob is None or market_prob <= 0:
        return None
    return round((market_prob - pick_prob) / market_prob * 100.0, 2)


def model_edge_vs_devigged_market(
  model_home_prob: float,
  home_odds: int | None,
  away_odds: int | None,
) -> dict[str, Any]:
    if home_odds is None or away_odds is None:
        return {"clv_home_pct": None, "market_home_prob": None}
    market_home, _market_away = devig_two_way(home_odds, away_odds)
    model_p = model_home_prob / 100.0
    return {
        "market_home_prob": round(market_home * 100.0, 2),
        "model_minus_market_pp": round((model_p - market_home) * 100.0, 2),
        "clv_home_pct": clv_vs_market_pct(home_odds, home_odds),
    }


def log_loss_binary(model_prob_pct: float, outcome_home_win: bool) -> float:
    prob = min(max(model_prob_pct / 100.0, 1e-6), 1.0 - 1e-6)
    return -math.log(prob if outcome_home_win else 1.0 - prob)
