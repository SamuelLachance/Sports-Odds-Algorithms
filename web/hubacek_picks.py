"""Hubáček-inspired official pick policy with real threshold floors.

Hubáček et al. (2019) bet on +EV decorrelated spots with a confidence filter.
Live gates add non-zero floors so vig noise and rounding jitter cannot qualify:
a minimum decorrelation gap vs the de-vigged market, a minimum honest EV%, and
a per-bet-type confidence bar (defaults: 20 pp moneyline, 10 pp baseball/soccer,
5 pp spread cover).

Default live thresholds live in this module; per-league overrides are read from
``data/pick_strategy.json`` (same keys as that file's policy note):
``min_market_gap_pp``, ``min_win_confidence_pp``, ``min_ev_pct``, ``ml_lo``,
``ml_hi``, ``allowed_sides``, ``min_spread_cover_gap_pp``,
``min_spread_confidence_pp``, ``min_spread_point_edge``. MLB uses a 6.7 pp
decorrelated gap (≈6 pp raw), no confidence bar, and a [-200, +200] price
window; NHL uses a 7.8 pp decorrelated gap (≈7 pp raw), no confidence bar, and
a [-250, +250] window — both from walk-forward bet backtests against
opening/closing lines. Spread leagues (NBA/WNBA/…) override cover-gap /
confidence / point-edge here via the same JSON.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.pick_strategy_schema import validate_pick_strategy_payload

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_PATH = _PROJECT_ROOT / "data" / "pick_strategy.json"

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


@lru_cache(maxsize=1)
def _strategy_config() -> dict[str, Any]:
    if not _STRATEGY_PATH.is_file():
        return {}
    try:
        payload = json.loads(_STRATEGY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return validate_pick_strategy_payload(payload)


def _league_category(league: str) -> str | None:
    try:
        from web.league_profiles import get_league_profile

        return get_league_profile(league.lower())["category"]
    except Exception:  # noqa: BLE001
        return None


def league_pick_overrides(league: str | None) -> dict[str, Any]:
    """Backtest-tuned per-league gate overrides from data/pick_strategy.json."""
    if not league:
        return {}
    config = _strategy_config()
    league = league.lower()
    entry = config.get(league)
    if not isinstance(entry, dict):
        category = _league_category(league)
        entry = config.get(category) if category else None
    return entry if isinstance(entry, dict) else {}


def clear_strategy_cache() -> None:
    _strategy_config.cache_clear()


def hubacek_min_market_gap_pp(league: str | None = None) -> float:
    override = league_pick_overrides(league).get("min_market_gap_pp")
    return float(override) if override is not None else HUBACEK_MIN_MARKET_GAP_PP


def hubacek_min_ev_pct(league: str | None = None) -> float:
    override = league_pick_overrides(league).get("min_ev_pct")
    return float(override) if override is not None else HUBACEK_MIN_EV_PCT


def hubacek_ml_range(league: str | None = None) -> tuple[float, float] | None:
    overrides = league_pick_overrides(league)
    lo = overrides.get("ml_lo")
    hi = overrides.get("ml_hi")
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def within_hubacek_ml_range(league: str | None, american_odds: float | None) -> bool:
    if american_odds is None:
        return True
    ml_range = hubacek_ml_range(league)
    if ml_range is None:
        return True
    return ml_range[0] <= float(american_odds) <= ml_range[1]


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
    """Per-league Hubáček φ threshold (override > baseball/soccer default > 20 pp)."""
    override = league_pick_overrides(league).get("min_win_confidence_pp")
    if override is not None:
        return float(override)
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
    league = pick.get("league")
    if (pick.get("ev_pct") or 0) < hubacek_min_ev_pct(league):
        return False
    win_prob = pick.get("win_probability")
    if (pick.get("bet_type") or "moneyline") == "spread":
        min_pp = HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
    else:
        min_pp = hubacek_min_win_confidence_pp(league)
    if win_prob is not None and not passes_hubacek_confidence(float(win_prob), min_pp=min_pp):
        return False
    gap = pick.get("model_market_gap_pp")
    if gap is not None and float(gap) < hubacek_min_market_gap_pp(league):
        return False
    if (pick.get("bet_type") or "moneyline") == "moneyline" and not within_hubacek_ml_range(
        league, pick.get("market_odds")
    ):
        return False
    return True
