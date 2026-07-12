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

_SPORT_PRED_KEYS = (
    "hockey_pred",
    "basketball_pred",
    "baseball_pred",
    "soccer_pred",
    "football_pred",
)


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


def hubacek_min_spread_cover_gap_pp(league: str | None = None) -> float:
    override = league_pick_overrides(league).get("min_spread_cover_gap_pp")
    return float(override) if override is not None else HUBACEK_MIN_SPREAD_COVER_GAP_PP


def hubacek_min_spread_confidence_pp(league: str | None = None) -> float:
    override = league_pick_overrides(league).get("min_spread_confidence_pp")
    return (
        float(override)
        if override is not None
        else HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
    )


def hubacek_min_ev_pct(league: str | None = None) -> float:
    override = league_pick_overrides(league).get("min_ev_pct")
    return float(override) if override is not None else HUBACEK_MIN_EV_PCT


def hubacek_allowed_sides(league: str | None = None) -> set[str] | None:
    sides = league_pick_overrides(league).get("allowed_sides")
    if not sides:
        return None
    return {str(side).lower() for side in sides}


def hubacek_ml_range(league: str | None = None) -> tuple[float, float] | None:
    overrides = league_pick_overrides(league)
    lo = overrides.get("ml_lo")
    hi = overrides.get("ml_hi")
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def within_hubacek_ml_range(league: str | None, american_odds: float | None) -> bool:
    # Always require a valid American price — missing/garbage juice fail closed
    # even when the league has no ml_lo/ml_hi window (soccer, etc.).
    from web.bet_advisor import normalize_american_odds

    if american_odds is None:
        return False
    try:
        odds_int = int(float(american_odds))
    except (TypeError, ValueError):
        return False
    normalized = normalize_american_odds(odds_int)
    if normalized is None:
        return False
    ml_range = hubacek_ml_range(league)
    if ml_range is None:
        return True
    return ml_range[0] <= float(normalized) <= ml_range[1]


def _blend_is_decorrelated(blended: dict[str, Any]) -> bool:
    """True only when Hubáček market decorrelation actually ran.

    Ensemble ML sets ``market_decorrelated`` only for the moneyline path
    (see ``apply_ensemble_ml``). Spread-only / invalid-juice ensemble runs must
    not short-circuit official gates via ``blend_mode`` alone.
    """
    if blended.get("market_decorrelated"):
        return True
    for key in _SPORT_PRED_KEYS:
        pred = blended.get(key)
        if isinstance(pred, dict) and pred.get("market_decorrelated"):
            return True
    return False


def official_hubacek_thresholds() -> dict[str, Any]:
    """Global baseline official-pick floors (leagues may raise via overrides)."""
    return {
        "pick_system": "hubacek",
        "thresholds_scope": "global_baseline",
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

    from web.bet_advisor import american_implied_prob, normalize_american_odds

    # Fail closed on garbage juice — never raise into the slate pipeline.
    # ESPN EVEN (0) must map like +100 → 50% cover, not 100/(0+100)=100%.
    juice = normalize_american_odds(spread_odds)
    if juice is None:
        return False
    market_cover = american_implied_prob(juice) * 100.0
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
    try:
        ev_pct = float(pick.get("ev_pct") if pick.get("ev_pct") is not None else 0)
    except (TypeError, ValueError):
        return False
    if ev_pct < hubacek_min_ev_pct(league):
        return False
    # Fail closed: gap and win probability are required official-book fields.
    win_prob = pick.get("win_probability")
    gap = pick.get("model_market_gap_pp")
    if win_prob is None or gap is None:
        return False
    try:
        win_prob_f = float(win_prob)
        gap_f = float(gap)
    except (TypeError, ValueError):
        return False
    bet_type = str(pick.get("bet_type") or "moneyline").lower()
    allowed = hubacek_allowed_sides(league)
    if allowed is not None and str(pick.get("side") or "").lower() not in allowed:
        return False
    if bet_type == "spread":
        min_pp = hubacek_min_spread_confidence_pp(league)
        min_gap = hubacek_min_spread_cover_gap_pp(league)
    else:
        min_pp = hubacek_min_win_confidence_pp(league)
        min_gap = hubacek_min_market_gap_pp(league)
    if not passes_hubacek_confidence(win_prob_f, min_pp=min_pp):
        return False
    if gap_f < min_gap:
        return False
    if bet_type == "spread":
        from web.bet_advisor import normalize_american_odds

        # Spread juice is required — never track ATS picks without a posted price.
        if normalize_american_odds(pick.get("spread_odds")) is None:
            return False
    elif bet_type in ("moneyline", "soccer_1x2") and not within_hubacek_ml_range(
        league, pick.get("market_odds")
    ):
        return False
    return True
