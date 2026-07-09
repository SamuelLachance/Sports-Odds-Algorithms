"""Hubáček-inspired official pick policy with real threshold floors.

Hubáček et al. (2019) bet on +EV decorrelated spots with a confidence filter.
Live gates add non-zero floors so vig noise and rounding jitter cannot qualify:
a minimum decorrelation gap vs the de-vigged market, a minimum honest EV%, and
a per-bet-type confidence bar.
"""

from __future__ import annotations

from typing import Any

# Minimum decorrelated model edge vs the de-vigged market price (pp).
HUBACEK_MIN_MARKET_GAP_PP = 2.0

# Spread cover: minimum cover-probability edge vs the juice-implied break-even.
HUBACEK_MIN_SPREAD_COVER_GAP_PP = 2.0

# Minimum honest EV% (computed from calibrated pre-decorrelation probability).
HUBACEK_MIN_EV_PCT = 2.0

# Section 5.2 confidence threshold φ ≤ 0.2 → |p̂ − 0.5| > 20 pp (default sports).
HUBACEK_MIN_WIN_CONFIDENCE_PP = 20.0

# Baseball moneylines cluster tighter — use a lower φ bar for all baseball leagues.
HUBACEK_BASEBALL_MIN_WIN_CONFIDENCE_PP = 10.0

# Soccer 1X2 outcome probabilities rarely leave the 25–60% band — lower φ bar.
HUBACEK_SOCCER_MIN_WIN_CONFIDENCE_PP = 10.0

# Spread cover probabilities live near 50% by construction (the book sets the
# line); requiring |cover − 50| ≥ 20 pp would need a ~7+ point disagreement
# under the empirical margin model. Use a 5 pp bar (≥55% cover) instead.
HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP = 5.0

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
    """Live official-pick gates (Hubáček theory with real floors)."""
    return {
        "pick_system": "hubacek",
        "min_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP,
        "min_spread_cover_gap_pp": HUBACEK_MIN_SPREAD_COVER_GAP_PP,
        "min_win_confidence_pp": HUBACEK_MIN_WIN_CONFIDENCE_PP,
        "min_spread_confidence_pp": HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP,
        "min_ev_pct": HUBACEK_MIN_EV_PCT,
        "min_edge": 0.0,
    }


def passes_hubacek_confidence(model_prob_pct: float, *, min_pp: float = HUBACEK_MIN_WIN_CONFIDENCE_PP) -> bool:
    """|p̂ − 50| ≥ φ (paper Section 5.2, φ = 0.2)."""
    return abs(model_prob_pct - 50.0) >= min_pp


def passes_hubacek_moneyline_gate(
    *,
    model_prob_pct: float,
    market_implied_pct: float | None,
    ev_pct: float,
    min_market_gap_pp: float = HUBACEK_MIN_MARKET_GAP_PP,
    min_win_confidence_pp: float = HUBACEK_MIN_WIN_CONFIDENCE_PP,
    min_ev_pct: float = HUBACEK_MIN_EV_PCT,
) -> bool:
    """Bet when decorrelated p̂ beats the book with real gap/EV/confidence floors."""
    if market_implied_pct is None:
        return False
    gap = model_prob_pct - market_implied_pct
    if gap < min_market_gap_pp:
        return False
    if not passes_hubacek_confidence(model_prob_pct, min_pp=min_win_confidence_pp):
        return False
    if ev_pct < min_ev_pct:
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
    min_win_confidence_pp: float = HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP,
    min_ev_pct: float = HUBACEK_MIN_EV_PCT,
) -> bool:
    """Spread official pick: decorrelated model, +EV cover with real floors."""
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

    if not passes_hubacek_confidence(side_cover_prob, min_pp=min_win_confidence_pp):
        return False

    if ev_pct < min_ev_pct:
        return False
    return True


def hubacek_min_win_confidence_pp(league: str | None = None) -> float:
    """Per-league Hubáček φ threshold (10 pp baseball/soccer, 20 pp elsewhere)."""
    if league:
        from web.baseball_pred_model import is_baseball_league
        from web.league_profiles import is_soccer_league

        if is_baseball_league(league.lower()):
            return HUBACEK_BASEBALL_MIN_WIN_CONFIDENCE_PP
        if is_soccer_league(league.lower()):
            return HUBACEK_SOCCER_MIN_WIN_CONFIDENCE_PP
    return HUBACEK_MIN_WIN_CONFIDENCE_PP


def passes_hubacek_tracked_pick(pick: dict[str, Any]) -> bool:
    """Whether a slate/tracking pick qualifies under Hubáček official rules."""
    if pick.get("strategy") != "hubacek":
        return False
    if (pick.get("ev_pct") or 0) < HUBACEK_MIN_EV_PCT:
        return False
    win_prob = pick.get("win_probability")
    if (pick.get("bet_type") or "moneyline") == "spread":
        min_pp = HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
    else:
        min_pp = hubacek_min_win_confidence_pp(pick.get("league"))
    if win_prob is not None and not passes_hubacek_confidence(float(win_prob), min_pp=min_pp):
        return False
    gap = pick.get("model_market_gap_pp")
    if gap is not None and float(gap) < HUBACEK_MIN_MARKET_GAP_PP:
        return False
    return True
