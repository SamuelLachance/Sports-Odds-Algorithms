"""Value-bet recommendation logic adapted from backtester strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from web.league_profiles import (
    DEFAULT_SPREAD_JUICE,
    MIN_CROSS_SIGN_EV_PCT,
    MIN_EXPECTED_VALUE_PCT,
    MIN_RECOMMENDED_EDGE,
    SOCCER_DRAW_BASE,
)

# Legacy display scale (margin → pseudo-units); spread edge uses cover probability.
SPREAD_POINT_TO_EDGE = 20.0
MIN_SPREAD_POINT_EDGE = MIN_RECOMMENDED_EDGE / SPREAD_POINT_TO_EDGE

# Std dev of (actual home margin − closing-spread-implied margin), per league.
# NBA fitted on 13k closing lines (data/supplemental/closing-odds/nba.csv);
# WNBA fitted on v2 walk-forward margin residuals (scripts/train_wnba_model.py,
# 2010-2026 OOS); others use published market-residual values.
SPREAD_MARGIN_SIGMA: dict[str, float] = {
    "nba": 12.98,
    "wnba": 12.58,
    "cbb": 12.0,
    "nfl": 13.5,
    "cfb": 15.5,
}
DEFAULT_SPREAD_MARGIN_SIGMA = 13.0

# Block home/away soccer picks when projected goals favor the other side by at least this margin.
SOCCER_PROJECTED_SCORE_CONFLICT_MARGIN = 1.5

# Fractional Kelly cap for pick ranking and backtest gates (25% of full Kelly).
MAX_KELLY_FRACTION = 0.25

# Win-probability → projected home margin (points), calibrated per league.
LEAGUE_MARGIN_SCALE: dict[str, float] = {
    "nba": 0.14,
    "wnba": 0.12,
    "cbb": 0.16,
    "nfl": 0.22,
    "cfb": 0.18,
}


@dataclass
class BetPick:
    side: str
    team_name: str
    team_slug: str
    strategy: str
    confidence: str
    edge: float
    model_projection: int
    market_odds: int
    win_probability: float
    reason: str
    bet_type: str = "moneyline"
    spread_line: float | None = None
    spread_odds: int | None = None
    consensus_spread: float | None = None
    model_margin: float | None = None
    ev_pct: float = 0.0
    profit_score: float = 0.0
    market_implied_prob: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _probability_to_american(probability_pct: float) -> int:
    probability = min(max(probability_pct, 0.1), 99.9)
    if probability > 50.0:
        return -round((probability / (100.0 - probability)) * 100)
    return round(((100.0 - probability) / probability) * 100)


def projections_from_win_probs(
    home_prob: float, away_prob: float
) -> tuple[int, int]:
    """Derive away/home American odds from win probabilities."""
    return (
        _probability_to_american(away_prob),
        _probability_to_american(home_prob),
    )


def model_moneylines(total_score: float) -> tuple[int, int]:
    """Return projected American odds for away and home teams."""
    away_prob, home_prob = _side_win_probs(total_score)
    return projections_from_win_probs(home_prob, away_prob)


def soccer_threeway_probs(total_score: float, league: str) -> tuple[float, float, float]:
    """Return home win, draw, and away win probabilities (0-100 scale)."""
    win_prob = abs(total_score)
    # total_score == 0 is pick'em — not "home favorite at 0%".
    if win_prob < 1e-9:
        home_binary = 50.0
    else:
        home_is_favorite = total_score < 0
        home_binary = win_prob if home_is_favorite else 100.0 - win_prob
    away_binary = 100.0 - home_binary

    base_draw = SOCCER_DRAW_BASE.get(league.lower(), SOCCER_DRAW_BASE["default"])
    closeness = 1.0 - abs(win_prob - 50.0) / 50.0
    draw_prob = min(35.0, max(18.0, base_draw + closeness * 8.0))

    scale = (100.0 - draw_prob) / 100.0
    return home_binary * scale, draw_prob, away_binary * scale


def soccer_model_moneylines(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
) -> tuple[int, int, int]:
    """Return projected American odds for away, draw, and home outcomes."""
    return (
        _probability_to_american(away_prob),
        _probability_to_american(draw_prob),
        _probability_to_american(home_prob),
    )


def model_home_margin(total_score: float, league: str) -> float:
    """Projected home margin in spread convention (negative = home favored)."""
    if abs(total_score) < 1e-9:
        return 0.0
    win_prob = abs(total_score)
    scale = LEAGUE_MARGIN_SCALE.get(league.lower(), 0.14)
    margin = (win_prob - 50.0) * scale
    return -margin if total_score < 0 else margin


def spread_line_for_side(home_spread: float, side: str) -> float:
    """Spread line for the given side (home_spread is the book's home line)."""
    return home_spread if str(side).lower() == "home" else -home_spread


def spread_point_edge(model_margin_home: float, home_spread: float, side: str) -> float:
    """Point cushion vs the consensus spread for the bet side.

    ``model_margin_home`` uses spread convention (negative = home favored).
    """
    if str(side).lower() == "home":
        return -model_margin_home + home_spread
    return model_margin_home - home_spread


def spread_margin_sigma(league: str | None = None) -> float:
    """Empirical std dev of final margin around the closing spread."""
    if league:
        return SPREAD_MARGIN_SIGMA.get(league.lower(), DEFAULT_SPREAD_MARGIN_SIGMA)
    return DEFAULT_SPREAD_MARGIN_SIGMA


def spread_cover_probability(point_edge: float, league: str | None = None) -> float:
    """ATS cover probability from point cushion, via the empirical margin model.

    P(cover) = Φ(point_edge / σ) where σ is the league's margin-vs-closing-spread
    residual std dev (e.g. ~13.3 points for NBA). Negative cushions are < 50%;
    callers that require a positive edge must gate separately.
    """
    sigma = spread_margin_sigma(league)
    prob = 0.5 * (1.0 + math.erf(point_edge / (sigma * math.sqrt(2.0))))
    return min(max(prob * 100.0, 5.0), 95.0)


def spread_odds_edge(point_edge: float, spread_odds: int, league: str | None = None) -> float:
    """American-odds edge for a spread bet from model cover probability vs book price."""
    if point_edge < MIN_SPREAD_POINT_EDGE:
        return 0.0
    cover_prob = spread_cover_probability(point_edge, league)
    fair_odds = _probability_to_american(cover_prob)
    return _odds_edge(fair_odds, spread_odds, cover_prob)


def spread_edge_from_points(
    point_edge: float,
    spread_odds: int | None = None,
    league: str | None = None,
) -> float:
    """Backward-compatible alias for spread_odds_edge."""
    juice = spread_odds if spread_odds is not None else DEFAULT_SPREAD_JUICE
    return spread_odds_edge(point_edge, juice, league)


def _breakeven_american(probability_pct: float, *, as_underdog: bool) -> float:
    """American odds with zero EV for the given win probability on one side."""
    probability = min(max(probability_pct, 0.1), 99.9) / 100.0
    if as_underdog:
        return ((1.0 - probability) / probability) * 100.0
    return -((probability / (1.0 - probability)) * 100.0)


def _odds_edge(model_projection: int, market_odds: int, model_prob_pct: float) -> float:
    """American-odds edge when the book price beats the model fair line.

    Same-sign lines (+/+ or -/-) compare directly. When favorite/underdog signs
    differ, compare the market quote to the model's breakeven line on that side
    instead of subtracting across zero (e.g. +109 vs -121 is not +230).
    """
    # ESPN/EVEN sometimes arrives as 0; treat as +100 (match EV helpers).
    # Reject |odds| < 100 garbage — same fail-closed as normalize_american_odds.
    normalized = normalize_american_odds(market_odds)
    if normalized is None:
        return 0.0
    market_odds = normalized
    fair_odds = _probability_to_american(model_prob_pct)
    same_sign = (fair_odds >= 0 and market_odds >= 0) or (
        fair_odds <= 0 and market_odds <= 0
    )
    if same_sign and market_odds <= fair_odds:
        return 0.0

    if same_sign:
        return float(market_odds - fair_odds)

    if market_odds > 0:
        breakeven = _breakeven_american(model_prob_pct, as_underdog=True)
        if market_odds > breakeven:
            return float(market_odds - breakeven)
        return 0.0

    breakeven = _breakeven_american(model_prob_pct, as_underdog=False)
    if market_odds > breakeven:
        return float(market_odds - breakeven)
    return 0.0


def _strategy_label(code: str) -> str:
    labels = {
        "hubacek": "Hubáček spot",
        "strong_value": "Strong value",
        "value": "Value bet",
        "model_favorite": "Model favorite",
        "lean": "Lean",
    }
    return labels.get(code, code)


def _format_spread(value: float) -> str:
    return f"{value:+.1f}".replace(".0", "")


def normalize_american_odds(american_odds: int | float | str | None) -> int | None:
    """Map ESPN EVEN (0) / ``EVEN`` / ``PK`` → +100; reject |odds| < 100 garbage.

    Accepts JSON floats and numeric strings (``-110.0``) so callers do not have
    to pre-coerce; bool is rejected (``True`` is a subclass of ``int``).
    Text labels match ``espn_client._parse_american_odds`` so raw book text
    does not silently fail closed on EV / Kelly / grading paths.
    Non-finite floats (±inf / NaN) fail closed as ``None`` (never raise).
    """
    import math

    if american_odds is None or isinstance(american_odds, bool):
        return None
    try:
        if isinstance(american_odds, str):
            text = american_odds.strip()
            if not text:
                return None
            upper = text.upper()
            if upper in {"EVEN", "PK"}:
                return 100
            if upper in {"OFF", "N/A", "NA"}:
                return None
            parsed = float(text)
            if not math.isfinite(parsed):
                return None
            odds = int(round(parsed))
        else:
            if isinstance(american_odds, float) and not math.isfinite(american_odds):
                return None
            odds = int(round(american_odds))
    except (TypeError, ValueError, OverflowError):
        return None
    if odds == 0:
        return 100
    if abs(odds) < 100:
        return None
    return odds


def american_to_decimal(american_odds: int) -> float:
    # ESPN/EVEN sometimes arrives as 0; treat as +100 (even money).
    odds = normalize_american_odds(american_odds)
    if odds is None:
        raise ValueError(f"invalid American odds: {american_odds}")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_implied_prob(american_odds: int) -> float:
    # ESPN/EVEN sometimes arrives as 0; treat as +100 (even money → 50%).
    odds = normalize_american_odds(american_odds)
    if odds is None:
        raise ValueError(f"invalid American odds: {american_odds}")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way_probs(
    away_odds: int | None,
    home_odds: int | None,
) -> tuple[float | None, float | None]:
    """Remove book vig from a two-way moneyline (0–100 scale)."""
    away_n = normalize_american_odds(away_odds)
    home_n = normalize_american_odds(home_odds)
    if away_n is None or home_n is None:
        return None, None
    away_raw = american_implied_prob(away_n)
    home_raw = american_implied_prob(home_n)
    total = away_raw + home_raw
    if total <= 0:
        return None, None
    return away_raw / total * 100.0, home_raw / total * 100.0


def expected_value_pct(model_prob_pct: float, american_odds: int) -> float:
    """Expected profit per $1 staked, as a percentage (positive = +EV)."""
    odds = normalize_american_odds(american_odds)
    if odds is None:
        raise ValueError(f"invalid American odds: {american_odds}")
    # bool / NaN / ±inf must not clamp into 0.1–99.9% (inf→max EV, True→1%).
    if isinstance(model_prob_pct, bool) or not math.isfinite(model_prob_pct):
        raise ValueError(f"invalid model probability: {model_prob_pct}")
    probability = min(max(float(model_prob_pct), 0.1), 99.9) / 100.0
    payout = american_to_decimal(odds)
    return (probability * payout - 1.0) * 100.0


def kelly_fraction(
    model_prob_pct: float,
    american_odds: int,
    *,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Kelly criterion stake fraction for a +EV bet (capped)."""
    # Fail closed: bool / non-finite probs must not size a max Kelly stake.
    if isinstance(model_prob_pct, bool) or not math.isfinite(model_prob_pct):
        return 0.0
    probability = min(max(float(model_prob_pct), 0.1), 99.9) / 100.0
    decimal_odds = american_to_decimal(american_odds)
    edge = probability * decimal_odds - 1.0
    if edge <= 0:
        return 0.0
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    fraction = edge / b
    return max(0.0, min(fraction, max_fraction))


def expected_units_per_bet(model_prob_pct: float, american_odds: int) -> float:
    """Expected units won/lost per 1u flat bet."""
    return expected_value_pct(model_prob_pct, american_odds) / 100.0


def pick_profit_score(
    *,
    model_prob_pct: float,
    american_odds: int,
    edge: float = 0.0,
) -> float:
    """Composite score for ranking picks: EV + Kelly growth heuristic."""
    ev_pct = expected_value_pct(model_prob_pct, american_odds)
    if ev_pct <= 0 and edge <= 0:
        return 0.0
    kelly = kelly_fraction(model_prob_pct, american_odds)
    # Weight EV heavily; Kelly rewards asymmetric +EV spots (longshots/favorites).
    return ev_pct * (1.0 + 2.5 * kelly) + edge * 0.15


def _pick_extra_with_league(
    *,
    base_ev_prob: float,
    outcome_prob: float,
    league: str | None = None,
    games_played_proxy: int | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if base_ev_prob != outcome_prob:
        extra["base_win_probability"] = round(base_ev_prob, 2)
    if league:
        extra["league"] = league
    if games_played_proxy is not None:
        extra["games_played_proxy"] = games_played_proxy
    return extra


def enrich_pick_profit_metrics(pick: BetPick) -> BetPick:
    """Attach EV%, Kelly, and profit_score used for official pick ranking."""
    if pick.bet_type == "spread":
        if pick.spread_odds is None:
            # Fail closed — do not invent -110 for unpriced spreads.
            return pick
        odds = normalize_american_odds(pick.spread_odds)
    else:
        odds = normalize_american_odds(pick.market_odds)
    if odds is None:
        return pick
    # EV/Kelly use the calibrated (pre-decorrelation) probability when available;
    # the decorrelated win_probability only drives the gap/confidence gates.
    base_prob = pick.extra.get("base_win_probability")
    raw_prob = base_prob if base_prob is not None else pick.win_probability
    if isinstance(raw_prob, bool):
        return pick
    try:
        prob = float(raw_prob)
    except (TypeError, ValueError):
        return pick
    if not math.isfinite(prob):
        return pick
    try:
        uncapped_ev = expected_value_pct(prob, odds)
    except ValueError:
        return pick
    ev_pct = uncapped_ev
    league = pick.extra.get("league")
    if league:
        from web.context_signals import sparse_sample_ev_cap

        games_proxy = pick.extra.get("games_played_proxy")
        try:
            games_i = int(games_proxy) if games_proxy is not None else None
        except (TypeError, ValueError):
            games_i = None
        ev_pct = sparse_sample_ev_cap(str(league), games_i, uncapped_ev)
    pick.ev_pct = round(ev_pct, 2)
    # When sparse/thin caps haircut EV, size Kelly / profit_score / units off an
    # effective probability that reproduces the capped EV at these odds — not
    # the raw uncapped edge.
    sizing_prob = prob
    if uncapped_ev > 0 and ev_pct < uncapped_ev:
        decimal = american_to_decimal(odds)
        if decimal > 0:
            sizing_prob = min(
                max((ev_pct / 100.0 + 1.0) / decimal * 100.0, 0.1),
                99.9,
            )
    kelly = kelly_fraction(sizing_prob, odds)
    pick.profit_score = round(
        pick_profit_score(
            model_prob_pct=sizing_prob, american_odds=odds, edge=pick.edge
        ),
        4,
    )
    pick.extra["kelly_pct"] = round(kelly * 100.0, 2)
    pick.extra["expected_units"] = round(ev_pct / 100.0, 4)
    if pick.market_implied_prob is not None:
        pick.extra["model_market_gap_pp"] = round(
            pick.win_probability - pick.market_implied_prob, 2
        )
    return pick


def model_vs_market_prob_edge(model_prob_pct: float, market_implied_pct: float) -> float:
    """Model probability minus de-vigged market implied probability."""
    return model_prob_pct - market_implied_pct


def passes_moneyline_pick_gate(
    *,
    edge: float,
    ev_pct: float,
    strategy: str,
    min_edge: float = MIN_RECOMMENDED_EDGE,
    min_ev_pct: float = MIN_EXPECTED_VALUE_PCT,
) -> bool:
    """Moneyline gate: EV% when min_edge is 0; otherwise edge with optional EV bar."""
    if min_edge <= 0:
        return min_ev_pct <= 0 or ev_pct >= min_ev_pct
    if edge >= min_edge:
        return min_ev_pct <= 0 or ev_pct >= min_ev_pct
    if strategy == "model_favorite" and min_ev_pct > 0:
        return ev_pct >= MIN_CROSS_SIGN_EV_PCT
    return False


def passes_hubacek_official_pick_gate(
    *,
    model_prob_pct: float,
    market_implied_pct: float | None,
    ev_pct: float,
    min_ev_pct: float | None = None,
    min_market_gap_pp: float | None = None,
    min_win_confidence_pp: float | None = None,
) -> bool:
    """Official Hubáček moneyline gate (decorrelation gap with real floors)."""
    from web.hubacek_picks import (
        HUBACEK_MIN_EV_PCT,
        HUBACEK_MIN_MARKET_GAP_PP,
        HUBACEK_MIN_WIN_CONFIDENCE_PP,
        passes_hubacek_moneyline_gate,
    )

    gap_floor = (
        HUBACEK_MIN_MARKET_GAP_PP if min_market_gap_pp is None else min_market_gap_pp
    )
    confidence_floor = (
        HUBACEK_MIN_WIN_CONFIDENCE_PP
        if min_win_confidence_pp is None
        else min_win_confidence_pp
    )
    ev_floor = HUBACEK_MIN_EV_PCT if min_ev_pct is None else min_ev_pct
    return passes_hubacek_moneyline_gate(
        model_prob_pct=model_prob_pct,
        market_implied_pct=market_implied_pct,
        ev_pct=ev_pct,
        min_market_gap_pp=gap_floor,
        min_win_confidence_pp=confidence_floor,
        min_ev_pct=ev_floor,
    )


def _hubacek_pick_reason(
    *,
    label: str,
    model_prob_pct: float,
    market_implied_pct: float,
    ev_pct: float,
    bet_type: str = "moneyline",
) -> str:
    gap = model_prob_pct - market_implied_pct
    if bet_type == "spread":
        return (
            f"Hubáček spot on {label}: decorrelated cover {model_prob_pct:.1f}% "
            f"vs market {market_implied_pct:.1f}% (+{gap:.1f} pp, +{ev_pct:.1f}% EV)."
        )
    return (
        f"Hubáček spot on {label}: decorrelated model {model_prob_pct:.1f}% "
        f"vs market {market_implied_pct:.1f}% (+{gap:.1f} pp, +{ev_pct:.1f}% EV)."
    )


def passes_hubacek_spread_pick_gate(
    *,
    blended: dict[str, Any] | None,
    side: str,
    point_edge: float,
    side_cover_prob: float,
    spread_odds: int,
    ev_pct: float,
    min_ev_pct: float | None = None,
    consensus_spread: float,
    min_cover_gap_pp: float | None = None,
    min_win_confidence_pp: float | None = None,
) -> bool:
    """Spread official pick: decorrelated margin vs line + cover gap vs juice."""
    from web.hubacek_picks import (
        HUBACEK_MIN_EV_PCT,
        HUBACEK_MIN_SPREAD_COVER_GAP_PP,
        HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP,
        passes_hubacek_spread_gate,
    )

    cover_floor = (
        HUBACEK_MIN_SPREAD_COVER_GAP_PP
        if min_cover_gap_pp is None
        else min_cover_gap_pp
    )
    confidence_floor = (
        HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
        if min_win_confidence_pp is None
        else min_win_confidence_pp
    )
    ev_floor = HUBACEK_MIN_EV_PCT if min_ev_pct is None else min_ev_pct
    return passes_hubacek_spread_gate(
        blended=blended,
        side=side,
        point_edge=point_edge,
        side_cover_prob=side_cover_prob,
        spread_odds=spread_odds,
        ev_pct=ev_pct,
        consensus_spread=consensus_spread,
        min_cover_gap_pp=cover_floor,
        min_win_confidence_pp=confidence_floor,
        min_ev_pct=ev_floor,
    )


def resolve_binary_win_probs(
    blended: dict[str, Any] | None,
    total_score: float,
) -> tuple[float, float]:
    """Prefer calibrated blend probabilities over legacy total_score conversion."""
    if blended:
        if blended.get("threeway"):
            home_prob = float(blended.get("home_win_probability", 50.0))
            away_prob = float(blended.get("away_win_probability", 50.0))
            # 1X2 mass excludes draw — renormalize so callers get a true 2-way pair.
            mass = home_prob + away_prob
            if mass <= 0:
                return 50.0, 50.0
            return away_prob / mass * 100.0, home_prob / mass * 100.0
        if blended.get("blended_home_win_probability") is not None:
            home_prob = float(blended["blended_home_win_probability"])
            return 100.0 - home_prob, home_prob
    return _side_win_probs(total_score)


_SPORT_PRED_KEYS = (
    "hockey_pred",
    "basketball_pred",
    "baseball_pred",
    "soccer_pred",
    "football_pred",
)


def _sport_pred_payload(blended: dict[str, Any]) -> dict[str, Any] | None:
    for key in _SPORT_PRED_KEYS:
        payload = blended.get(key)
        if isinstance(payload, dict):
            return payload
    return None


def blend_outputs_are_market_decorrelated(blended: dict[str, Any]) -> bool:
    """True when unified or sport-layer probabilities already use Hubáček decorrelation.

    Ensemble ML is decorrelated only when ``apply_ensemble_ml`` set
    ``market_decorrelated`` (moneyline path). Do not treat ``blend_mode`` alone
    as proof — spread-only ensemble must still run ``ensure_hubacek_in_blend``.
    """
    if blended.get("market_decorrelated"):
        return True
    pred = _sport_pred_payload(blended)
    return bool(pred and pred.get("market_decorrelated"))


def market_home_prob_pct(
    *,
    away_market: int | None = None,
    home_market: int | None = None,
    consensus_spread: float | None = None,
) -> float | None:
    """De-vigged or spread-implied home win probability on 0–100 scale."""
    _away, home = devig_two_way_probs(away_market, home_market)
    if home is not None:
        return home
    # Reject bools: False is not None and float(False)==0 invents a 50% market.
    if consensus_spread is not None and not isinstance(consensus_spread, bool):
        from web.cbb_calibrate import spread_to_home_prob

        try:
            spread = float(consensus_spread)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(spread) or abs(spread) >= 100.0:
            return None
        return spread_to_home_prob(spread)
    return None


def official_pick_binary_prob_sets(
    blended: dict[str, Any],
    total_score: float,
    *,
    league: str,
    away_market: int | None = None,
    home_market: int | None = None,
    consensus_spread: float | None = None,
) -> tuple[float, float, float, float]:
    """Decorrelated and calibrated (pre-decorrelation) home/away win %.

    Returns (away_decor, home_decor, away_base, home_base). The decorrelated
    pair drives the Hubáček gap/confidence gates; the calibrated pair is the
    honest probability used for EV and Kelly.
    """
    from web.market_decorrelation import decorrelate_binary

    away_prob, home_prob = resolve_binary_win_probs(blended, total_score)
    pred = _sport_pred_payload(blended)

    def _base_home(default: float) -> float:
        if blended.get("pre_decorrelation_home_win_probability") is not None:
            return float(blended["pre_decorrelation_home_win_probability"])
        if pred and pred.get("pre_decorrelation_home_win_probability") is not None:
            return float(pred["pre_decorrelation_home_win_probability"])
        return default

    if blended.get("market_decorrelated"):
        base_home = _base_home(home_prob)
        return away_prob, home_prob, 100.0 - base_home, base_home

    if pred and pred.get("market_decorrelated") and pred.get("home_win_probability") is not None:
        source = pred.get("market_decorrelation_source", "moneyline")
        if source != "spread":
            home_decor = float(pred["home_win_probability"])
            base_home = _base_home(home_decor)
            return 100.0 - home_decor, home_decor, 100.0 - base_home, base_home

    market_home = market_home_prob_pct(
        away_market=away_market,
        home_market=home_market,
        consensus_spread=None if away_market is not None and home_market is not None else consensus_spread,
    )
    if market_home is None:
        return away_prob, home_prob, away_prob, home_prob

    base_home = _base_home(home_prob)
    home_decor = decorrelate_binary(base_home, market_home)
    return 100.0 - home_decor, home_decor, 100.0 - base_home, base_home


def official_pick_binary_probs(
    blended: dict[str, Any],
    total_score: float,
    *,
    league: str,
    away_market: int | None = None,
    home_market: int | None = None,
    consensus_spread: float | None = None,
) -> tuple[float, float]:
    """Hubáček-adjusted home/away win % (decorrelated pair only)."""
    away_decor, home_decor, _away_base, _home_base = official_pick_binary_prob_sets(
        blended,
        total_score,
        league=league,
        away_market=away_market,
        home_market=home_market,
        consensus_spread=consensus_spread,
    )
    return away_decor, home_decor


def ensure_hubacek_in_blend(
    blended: dict[str, Any],
    *,
    league: str,
    away_market: int | None = None,
    home_market: int | None = None,
    consensus_spread: float | None = None,
) -> dict[str, Any]:
    """Apply Hubáček decorrelation to unified blend output when not yet applied."""
    from web.blend_service import home_win_prob_to_total_score
    from web.market_decorrelation import decorrelate_binary

    if blend_outputs_are_market_decorrelated(blended):
        return blended
    # Binary Hubáček must not rewrite three-way (1X2) blends — it ignores draw
    # and would set a false market_decorrelated flag over home/draw/away probs.
    if blended.get("threeway") or blended.get("draw_probability") is not None:
        return blended

    home_prob = blended.get("blended_home_win_probability")
    if home_prob is None and blended.get("total_score") is not None:
        home_prob = resolve_binary_win_probs(blended, float(blended["total_score"]))[1]
    if home_prob is None:
        return blended

    market_home = market_home_prob_pct(
        away_market=away_market,
        home_market=home_market,
        consensus_spread=consensus_spread,
    )
    if market_home is None:
        return blended

    dec_home = decorrelate_binary(float(home_prob), market_home)
    total, win_prob = home_win_prob_to_total_score(dec_home)
    updated = dict(blended)
    updated["pre_decorrelation_home_win_probability"] = round(float(home_prob), 2)
    updated["blended_home_win_probability"] = round(dec_home, 2)
    updated["total_score"] = round(total, 2)
    updated["win_probability"] = round(win_prob, 2)
    updated["favorite_side"] = "home" if total <= 0 else "away"
    updated["market_decorrelated"] = True
    return updated


def best_pick_only(picks: list[BetPick]) -> list[BetPick]:
    """Return at most one pick per event (highest profit score, then EV, then edge)."""
    if not picks:
        return []
    enriched = [enrich_pick_profit_metrics(pick) for pick in picks]
    enriched.sort(
        key=lambda item: (item.profit_score, item.ev_pct, item.edge),
        reverse=True,
    )
    return [enriched[0]]


def _side_win_probs(total_score: float) -> tuple[float, float]:
    if abs(total_score) < 1e-9:
        return 50.0, 50.0
    win_prob = abs(total_score)
    home_is_favorite = total_score <= 0
    home_prob = win_prob if home_is_favorite else 100.0 - win_prob
    away_prob = 100.0 - home_prob
    return away_prob, home_prob


def _moneyline_reason(
    *,
    name: str,
    projection: int,
    market: int,
    edge: float,
    ev_pct: float,
    outcome_prob: float,
    market_implied_prob: float | None,
    is_model_favorite: bool,
    is_market_underdog: bool,
    outcome_label: str | None = None,
) -> tuple[str, str, str]:
    label = outcome_label or name
    strategy = "value"
    confidence = "medium"
    fair_underdog = round(_breakeven_american(outcome_prob, as_underdog=True))

    if is_model_favorite and is_market_underdog:
        strategy = "model_favorite"
        confidence = "high"
        reason = (
            f"Model makes {label} a {projection:+d} favorite ({outcome_prob:.1f}% win) "
            f"but the book offers underdog odds {market:+d} "
            f"(fair underdog +{fair_underdog}, +{ev_pct:.1f}% EV, +{edge:.0f} American edge)."
        )
    elif is_model_favorite and not is_market_underdog and edge >= 15:
        strategy = "strong_value"
        confidence = "high"
        reason = (
            f"Model favors {label} at {projection:+d}; "
            f"book line {market:+d} is softer (+{ev_pct:.1f}% EV, +{edge:.0f} American edge)."
        )
    elif not is_model_favorite and not is_market_underdog and edge >= 15:
        strategy = "strong_value"
        confidence = "high"
        reason = (
            f"Model has {label} as {projection:+d} underdog; "
            f"book favorite price {market:+d} is too short "
            f"(+{ev_pct:.1f}% EV, +{edge:.0f} American edge)."
        )
    else:
        reason = (
            f"Sportsbook offers {market:+d} vs model fair {projection:+d} "
            f"on {label} (+{edge:.0f} American edge, +{ev_pct:.1f}% EV)."
        )

    if edge >= 8 and strategy == "value":
        strategy = "value"
        confidence = "medium"
    elif edge < 8:
        strategy = "lean"
        confidence = "low"

    return strategy, confidence, reason


def evaluate_picks(
    *,
    away_name: str,
    home_name: str,
    away_slug: str,
    home_slug: str,
    total_score: float,
    win_probability: float,
    away_market: int | None,
    home_market: int | None,
    away_prob: float | None = None,
    home_prob: float | None = None,
    base_away_prob: float | None = None,
    base_home_prob: float | None = None,
    min_edge: float = MIN_RECOMMENDED_EDGE,
    min_ev_pct: float = MIN_EXPECTED_VALUE_PCT,
    hubacek_only: bool = False,
    min_market_gap_pp: float | None = None,
    min_win_confidence_pp: float | None = None,
    ml_lo: float | None = None,
    ml_hi: float | None = None,
    league: str | None = None,
    games_played_proxy: int | None = None,
) -> list[BetPick]:
    if away_prob is None or home_prob is None:
        away_prob, home_prob = _side_win_probs(total_score)
    away_proj, home_proj = projections_from_win_probs(home_prob, away_prob)
    devig_away, devig_home = devig_two_way_probs(away_market, home_market)
    picks: list[BetPick] = []

    # Honest EV uses calibrated (pre-decorrelation) probabilities when supplied.
    ev_away_prob = base_away_prob if base_away_prob is not None else away_prob
    ev_home_prob = base_home_prob if base_home_prob is not None else home_prob

    candidates: list[tuple[str, str, str, float, float, int, int | None, float | None]] = [
        ("away", away_name, away_slug, away_prob, ev_away_prob, away_proj, away_market, devig_away),
        ("home", home_name, home_slug, home_prob, ev_home_prob, home_proj, home_market, devig_home),
    ]

    for side, name, slug, outcome_prob, ev_prob, projection, market, market_implied in candidates:
        market = normalize_american_odds(market)
        if market is None:
            continue

        edge = _odds_edge(projection, market, outcome_prob)
        ev_pct = expected_value_pct(ev_prob, market)
        if league:
            from web.context_signals import sparse_sample_ev_cap

            ev_pct = sparse_sample_ev_cap(league, games_played_proxy, ev_pct)
        is_model_favorite = projection < 0
        is_market_underdog = market > 0

        if hubacek_only:
            if market_implied is None:
                continue
            if ml_lo is not None and float(market) < ml_lo:
                continue
            if ml_hi is not None and float(market) > ml_hi:
                continue
            if not passes_hubacek_official_pick_gate(
                model_prob_pct=outcome_prob,
                market_implied_pct=market_implied,
                ev_pct=ev_pct,
                min_ev_pct=min_ev_pct,
                min_market_gap_pp=min_market_gap_pp,
                min_win_confidence_pp=min_win_confidence_pp,
            ):
                continue
            strategy = "hubacek"
            confidence = "high" if outcome_prob - market_implied >= 5.0 else "medium"
            reason = _hubacek_pick_reason(
                label=name,
                model_prob_pct=outcome_prob,
                market_implied_pct=market_implied,
                ev_pct=ev_pct,
            )
        else:
            strategy, confidence, reason = _moneyline_reason(
                name=name,
                projection=projection,
                market=market,
                edge=edge,
                ev_pct=ev_pct,
                outcome_prob=outcome_prob,
                market_implied_prob=market_implied,
                is_model_favorite=is_model_favorite,
                is_market_underdog=is_market_underdog,
            )

            if not passes_moneyline_pick_gate(
                edge=edge,
                ev_pct=ev_pct,
                strategy=strategy,
                min_edge=min_edge,
                min_ev_pct=min_ev_pct,
            ):
                continue

            if edge >= 8 and strategy == "lean":
                strategy = "value"
                confidence = "medium"

        extra: dict[str, Any] = {}
        if ev_prob != outcome_prob:
            extra["base_win_probability"] = round(ev_prob, 2)
        if league:
            extra["league"] = league
        if games_played_proxy is not None:
            extra["games_played_proxy"] = games_played_proxy

        picks.append(
            BetPick(
                side=side,
                team_name=name,
                team_slug=slug,
                strategy=strategy,
                confidence=confidence,
                edge=edge,
                ev_pct=round(ev_pct, 2),
                model_projection=projection,
                market_odds=market,
                win_probability=outcome_prob,
                market_implied_prob=round(market_implied, 2) if market_implied is not None else None,
                reason=reason,
                extra=extra,
            )
        )

    picks = best_pick_only(picks)
    for pick in picks:
        if pick.market_implied_prob is not None:
            pick.extra.setdefault(
                "model_market_gap_pp",
                round(pick.win_probability - pick.market_implied_prob, 2),
            )
    return picks


def soccer_team_pick_blocked_by_projected_score(
    side: str,
    *,
    expected_home_goals: float | None,
    expected_away_goals: float | None,
    margin: float = SOCCER_PROJECTED_SCORE_CONFLICT_MARGIN,
) -> bool:
    """True when a home/away pick conflicts with the projected goal margin."""
    if side == "draw":
        return False
    if expected_home_goals is None or expected_away_goals is None:
        return False

    goal_margin = float(expected_home_goals) - float(expected_away_goals)
    if side == "home" and goal_margin <= -margin:
        return True
    if side == "away" and goal_margin >= margin:
        return True
    return False


def evaluate_soccer_picks(
    *,
    away_name: str,
    home_name: str,
    away_slug: str,
    home_slug: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    away_proj: int,
    draw_proj: int,
    home_proj: int,
    away_market: int | None,
    draw_market: int | None,
    home_market: int | None,
    expected_home_goals: float | None = None,
    expected_away_goals: float | None = None,
    base_home_prob: float | None = None,
    base_draw_prob: float | None = None,
    base_away_prob: float | None = None,
    min_edge: float = MIN_RECOMMENDED_EDGE,
    min_ev_pct: float = MIN_EXPECTED_VALUE_PCT,
    hubacek_only: bool = False,
    min_market_gap_pp: float | None = None,
    min_win_confidence_pp: float | None = None,
    league: str | None = None,
    games_played_proxy: int | None = None,
) -> list[BetPick]:
    """Evaluate 3-way soccer moneyline outcomes vs the book."""
    picks: list[BetPick] = []

    # Hubáček official soccer requires a full 1X2 book so de-vig stays on the
    # same 0–100 scale as model probs. Incomplete / invalid boards fail closed.
    devig_probs: dict[str, float | None] = {}
    if away_market is not None and draw_market is not None and home_market is not None:
        from web.soccer_decorrelation import devig_threeway_from_odds

        devigged = devig_threeway_from_odds(home_market, draw_market, away_market)
        if devigged is None:
            if hubacek_only:
                return []
        else:
            mkt_h, mkt_d, mkt_a = devigged
            devig_probs = {"home": mkt_h, "draw": mkt_d, "away": mkt_a}
    elif hubacek_only:
        return []

    base_probs: dict[str, float | None] = {
        "home": base_home_prob,
        "draw": base_draw_prob,
        "away": base_away_prob,
    }

    candidates: list[tuple[str, str, str, float, int, int | None]] = [
        ("away", away_name, away_slug, away_prob, away_proj, away_market),
        ("draw", "Draw", "draw", draw_prob, draw_proj, draw_market),
        ("home", home_name, home_slug, home_prob, home_proj, home_market),
    ]

    for side, name, slug, outcome_prob, projection, market in candidates:
        market = normalize_american_odds(market)
        if market is None:
            continue

        base_prob = base_probs.get(side)
        ev_prob = float(base_prob) if base_prob is not None else outcome_prob
        edge = _odds_edge(projection, market, outcome_prob)
        ev_pct = expected_value_pct(ev_prob, market)
        if league:
            from web.context_signals import sparse_sample_ev_cap

            ev_pct = sparse_sample_ev_cap(league, games_played_proxy, ev_pct)
        is_model_favorite = projection < 0
        is_market_underdog = market > 0
        # Always store market implied on the 0–100 probability scale.
        market_implied = (
            devig_probs.get(side)
            if devig_probs
            else american_implied_prob(market) * 100.0
        )

        if soccer_team_pick_blocked_by_projected_score(
            side,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
        ):
            continue

        outcome_label = "Draw" if side == "draw" else name
        if hubacek_only:
            if market_implied is None:
                continue
            if not passes_hubacek_official_pick_gate(
                model_prob_pct=outcome_prob,
                market_implied_pct=market_implied,
                ev_pct=ev_pct,
                min_market_gap_pp=min_market_gap_pp,
                min_win_confidence_pp=min_win_confidence_pp,
                min_ev_pct=min_ev_pct,
            ):
                continue
            strategy = "hubacek"
            confidence = "high" if outcome_prob - market_implied >= 5.0 else "medium"
            reason = _hubacek_pick_reason(
                label=outcome_label,
                model_prob_pct=outcome_prob,
                market_implied_pct=market_implied,
                ev_pct=ev_pct,
            )
        else:
            strategy, confidence, reason = _moneyline_reason(
                name=name,
                projection=projection,
                market=market,
                edge=edge,
                ev_pct=ev_pct,
                outcome_prob=outcome_prob,
                market_implied_prob=market_implied,
                is_model_favorite=is_model_favorite,
                is_market_underdog=is_market_underdog,
                outcome_label=outcome_label,
            )

            if not passes_moneyline_pick_gate(
                edge=edge,
                ev_pct=ev_pct,
                strategy=strategy,
                min_edge=min_edge,
                min_ev_pct=min_ev_pct,
            ):
                continue

            if edge >= 8 and strategy == "lean":
                strategy = "value"
                confidence = "medium"

        picks.append(
            BetPick(
                side=side,
                team_name=name,
                team_slug=slug,
                strategy=strategy,
                confidence=confidence,
                edge=edge,
                ev_pct=round(ev_pct, 2),
                model_projection=projection,
                market_odds=market,
                win_probability=outcome_prob,
                market_implied_prob=round(market_implied, 2) if market_implied is not None else None,
                reason=reason,
                extra=_pick_extra_with_league(
                    base_ev_prob=ev_prob,
                    outcome_prob=outcome_prob,
                    league=league,
                    games_played_proxy=games_played_proxy,
                ),
            )
        )

    picks = best_pick_only(picks)
    for pick in picks:
        if pick.market_implied_prob is not None:
            pick.extra.setdefault(
                "model_market_gap_pp",
                round(pick.win_probability - pick.market_implied_prob, 2),
            )
    return picks


def evaluate_spread_picks(
    *,
    league: str,
    away_name: str,
    home_name: str,
    away_slug: str,
    home_slug: str,
    total_score: float,
    win_probability: float,
    consensus_spread: float | None,
    away_spread_odds: int | None = None,
    home_spread_odds: int | None = None,
    model_margin_home: float | None = None,
    min_edge: float = MIN_RECOMMENDED_EDGE,
    min_point_edge: float | None = None,
    min_ev_pct: float = MIN_EXPECTED_VALUE_PCT,
    hubacek_only: bool = False,
    blended: dict[str, Any] | None = None,
    min_cover_gap_pp: float | None = None,
    min_win_confidence_pp: float | None = None,
    games_played_proxy: int | None = None,
) -> list[BetPick]:
    """Recommend spread bets when decorrelated margin disagrees with the book line."""
    if consensus_spread is None or isinstance(consensus_spread, bool):
        return []
    try:
        consensus_spread = float(consensus_spread)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(consensus_spread) or abs(consensus_spread) >= 100.0:
        return []
    if hubacek_only and (blended is None or not blend_outputs_are_market_decorrelated(blended)):
        return []

    model_margin = (
        model_margin_home
        if model_margin_home is not None
        else model_home_margin(total_score, league)
    )
    picks: list[BetPick] = []

    candidates: list[tuple[str, str, str, int | None]] = [
        ("away", away_name, away_slug, away_spread_odds),
        ("home", home_name, home_slug, home_spread_odds),
    ]

    for side, name, slug, spread_odds in candidates:
        point_edge = spread_point_edge(model_margin, consensus_spread, side)
        if point_edge <= 0:
            continue
        # Official Hubáček must not invent -110 when ESPN juice is missing.
        # Non-Hubáček paths also fail closed — invented juice skews EV/ROI.
        if spread_odds is None:
            continue
        normalized_juice = normalize_american_odds(spread_odds)
        if normalized_juice is None:
            continue
        juice = normalized_juice
        edge = spread_odds_edge(point_edge, juice, league)
        if min_point_edge is not None and point_edge < min_point_edge:
            continue

        line = spread_line_for_side(consensus_spread, side)
        side_cover_prob = spread_cover_probability(point_edge, league)
        ev_pct = expected_value_pct(side_cover_prob, juice)
        from web.context_signals import sparse_sample_ev_cap

        ev_pct = sparse_sample_ev_cap(league, games_played_proxy, ev_pct)
        market_cover = american_implied_prob(juice) * 100.0

        if hubacek_only:
            if not passes_hubacek_spread_pick_gate(
                blended=blended,
                side=side,
                point_edge=point_edge,
                side_cover_prob=side_cover_prob,
                spread_odds=juice,
                ev_pct=ev_pct,
                min_ev_pct=min_ev_pct,
                consensus_spread=consensus_spread,
                min_cover_gap_pp=min_cover_gap_pp,
                min_win_confidence_pp=min_win_confidence_pp,
            ):
                continue
            strategy = "hubacek"
            confidence = "high" if point_edge >= 2.5 else "medium"
            reason = _hubacek_pick_reason(
                label=f"{name} {_format_spread(line)}",
                model_prob_pct=side_cover_prob,
                market_implied_pct=market_cover,
                ev_pct=ev_pct,
                bet_type="spread",
            )
        else:
            if min_ev_pct > 0 and ev_pct < min_ev_pct:
                continue
            if min_edge > 0 and edge < min_edge:
                continue
            fair_spread_odds = _probability_to_american(side_cover_prob)
            strategy = "value"
            confidence = "medium"
            if point_edge >= 4:
                strategy = "strong_value"
                confidence = "high"
            elif point_edge >= 2.5:
                strategy = "value"
                confidence = "medium"
            else:
                strategy = "lean"
                confidence = "low"
            reason = (
                f"Model home margin {_format_spread(model_margin)}; "
                f"{name} {_format_spread(line)} has {_format_spread(point_edge)} pt cushion "
                f"({side_cover_prob:.1f}% cover vs fair {fair_spread_odds:+d}, "
                f"+{edge:.0f} edge)."
            )

        fair_spread_odds = _probability_to_american(side_cover_prob)
        extra: dict[str, Any] = {
            "model_market_gap_pp": round(side_cover_prob - market_cover, 2),
            "league": league,
            # Cover % is the EV probability; expose for Honest EV UI.
            "base_win_probability": round(side_cover_prob, 2),
        }
        if games_played_proxy is not None:
            extra["games_played_proxy"] = games_played_proxy
        picks.append(
            BetPick(
                side=side,
                team_name=name,
                team_slug=slug,
                strategy=strategy,
                confidence=confidence,
                edge=edge,
                ev_pct=round(ev_pct, 2),
                model_projection=fair_spread_odds,
                market_odds=juice,
                win_probability=side_cover_prob,
                market_implied_prob=round(market_cover, 2),
                reason=reason,
                bet_type="spread",
                spread_line=line,
                spread_odds=juice,
                consensus_spread=consensus_spread,
                model_margin=round(model_margin, 2),
                extra=extra,
            )
        )

    picks.sort(key=lambda item: item.edge, reverse=True)
    return best_pick_only(picks)


def spread_pick_blocked_by_model_margin(
    side: str,
    model_margin_home: float,
    consensus_spread: float,
) -> bool:
    """Block spread picks that contradict the model's projected margin vs the line."""
    return spread_point_edge(model_margin_home, consensus_spread, side) <= 0


def pick_to_dict(pick: BetPick) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "side": pick.side,
        "team_name": pick.team_name,
        "team_slug": pick.team_slug,
        "strategy": pick.strategy,
        "strategy_label": _strategy_label(pick.strategy),
        "confidence": pick.confidence,
        "edge": round(pick.edge, 1),
        "ev_pct": round(pick.ev_pct, 2),
        "profit_score": round(pick.profit_score, 2),
        "kelly_pct": pick.extra.get("kelly_pct"),
        "expected_units": pick.extra.get("expected_units"),
        "model_projection": pick.model_projection,
        "market_odds": pick.market_odds,
        "win_probability": round(pick.win_probability, 2),
        "market_implied_prob": pick.market_implied_prob,
        "model_market_gap_pp": pick.extra.get("model_market_gap_pp"),
        "reason": pick.reason,
        "bet_type": pick.bet_type,
    }
    # Honest-EV / three-track UI reads these pick-scoped fields (not home-only).
    if pick.extra.get("base_win_probability") is not None:
        payload["base_win_probability"] = pick.extra["base_win_probability"]
    if pick.extra.get("games_played_proxy") is not None:
        payload["games_played_proxy"] = pick.extra["games_played_proxy"]
    if pick.bet_type == "spread":
        payload.update(
            {
                "spread_line": pick.spread_line,
                "spread_odds": pick.spread_odds,
                "consensus_spread": pick.consensus_spread,
                "model_margin": pick.model_margin,
                "consensus_odds": pick.spread_odds,
                "consensus_label": (
                    f"{_format_spread(pick.spread_line or 0)} "
                    f"({pick.spread_odds:+d})"
                    if pick.spread_line is not None and pick.spread_odds is not None
                    else None
                ),
            }
        )
    return payload
