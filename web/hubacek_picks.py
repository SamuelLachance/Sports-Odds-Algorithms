"""Hubáček-only official pick policy — decorrelation gap, not flat EV%."""

from __future__ import annotations

from typing import Any

# Minimum decorrelated model − market gap (percentage points) to bet officially.
HUBACEK_MIN_MARKET_GAP_PP = 8.0

# Spread: minimum cover-probability gap vs book juice-implied cover.
HUBACEK_MIN_SPREAD_COVER_GAP_PP = 6.0

# Paper-style confidence: outcome prob must be at least this far from a coin flip.
HUBACEK_MIN_WIN_CONFIDENCE_PP = 5.0

_SPORT_PRED_KEYS = ("hockey_pred", "basketball_pred", "baseball_pred", "soccer_pred")


def _blend_is_decorrelated(blended: dict[str, Any]) -> bool:
    if blended.get("market_decorrelated"):
        return True
    if blended.get("blend_mode") == "ensemble_ml" and blended.get("ensemble_ml"):
        return True
    for key in _SPORT_PRED_KEYS:
        pred = blended.get(key)
        if isinstance(pred, dict) and pred.get("market_decorrelated"):
            return True
    return False


def official_hubacek_thresholds() -> dict[str, Any]:
    """Live official-pick gates (Hubáček theory only)."""
    return {
        "pick_system": "hubacek",
        "min_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP,
        "min_spread_cover_gap_pp": HUBACEK_MIN_SPREAD_COVER_GAP_PP,
        "min_win_confidence_pp": HUBACEK_MIN_WIN_CONFIDENCE_PP,
        "min_ev_pct": 0.0,
        "min_edge": 0.0,
    }


def passes_hubacek_moneyline_gate(
    *,
    model_prob_pct: float,
    market_implied_pct: float | None,
    ev_pct: float,
    min_market_gap_pp: float = HUBACEK_MIN_MARKET_GAP_PP,
    min_win_confidence_pp: float = HUBACEK_MIN_WIN_CONFIDENCE_PP,
) -> bool:
    """Bet only when decorrelated model meaningfully disagrees with the market."""
    if market_implied_pct is None:
        return False
    gap = model_prob_pct - market_implied_pct
    if gap < min_market_gap_pp:
        return False
    if abs(model_prob_pct - 50.0) < min_win_confidence_pp:
        return False
    if ev_pct <= 0:
        return False
    return True


def passes_hubacek_spread_gate(
    *,
    blended: dict[str, Any] | None,
    side: str,
    point_edge: float,
    side_cover_prob: float,
    spread_odds: int,
    ev_pct: float,
    consensus_spread: float,
    min_cover_gap_pp: float = HUBACEK_MIN_SPREAD_COVER_GAP_PP,
    min_win_gap_pp: float = HUBACEK_MIN_MARKET_GAP_PP,
) -> bool:
    """Spread official pick: decorrelated margin vs line + cover gap vs juice."""
    if blended is None or not _blend_is_decorrelated(blended):
        return False
    if point_edge <= 0:
        return False

    if spread_odds >= 0:
        market_cover = 100.0 / (spread_odds + 100.0) * 100.0
    else:
        market_cover = abs(spread_odds) / (abs(spread_odds) + 100.0) * 100.0
    cover_gap = side_cover_prob - market_cover
    if cover_gap < min_cover_gap_pp:
        return False

    decor_home = blended.get("blended_home_win_probability") if blended else None
    if decor_home is not None:
        from web.cbb_calibrate import spread_to_home_prob

        spread_implied_home = spread_to_home_prob(float(consensus_spread))
        decor_side = float(decor_home) if side == "home" else 100.0 - float(decor_home)
        market_side = (
            spread_implied_home if side == "home" else 100.0 - spread_implied_home
        )
        if decor_side - market_side < min_win_gap_pp:
            return False

    if ev_pct <= 0:
        return False
    return True


def passes_hubacek_tracked_pick(pick: dict[str, Any]) -> bool:
    """Whether a slate/tracking pick qualifies under Hubáček official rules."""
    if pick.get("strategy") != "hubacek":
        return False
    gap = pick.get("model_market_gap_pp")
    if gap is None or float(gap) < HUBACEK_MIN_MARKET_GAP_PP:
        return False
    if (pick.get("ev_pct") or 0) <= 0:
        return False
    return True
