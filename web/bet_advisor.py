"""Value-bet recommendation logic adapted from backtester strategies."""

from __future__ import annotations

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

# Cover-probability boost per point of cushion vs the consensus spread.
SPREAD_COVER_PROB_PER_POINT = 5.0

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
    home_is_favorite = total_score <= 0
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
    return home_spread if side == "home" else -home_spread


def spread_point_edge(model_margin_home: float, home_spread: float, side: str) -> float:
    """Point cushion vs the consensus spread for the bet side.

    ``model_margin_home`` uses spread convention (negative = home favored).
    """
    if side == "home":
        return -model_margin_home + home_spread
    return model_margin_home - home_spread


def spread_cover_probability(point_edge: float) -> float:
    """Heuristic ATS cover probability from point cushion vs the spread."""
    if point_edge <= 0:
        return 50.0
    return min(
        max(50.0 + point_edge * SPREAD_COVER_PROB_PER_POINT, 5.0),
        95.0,
    )


def spread_odds_edge(point_edge: float, spread_odds: int) -> float:
    """American-odds edge for a spread bet from model cover probability vs book price."""
    if point_edge < MIN_SPREAD_POINT_EDGE:
        return 0.0
    cover_prob = spread_cover_probability(point_edge)
    fair_odds = _probability_to_american(cover_prob)
    return _odds_edge(fair_odds, spread_odds, cover_prob)


def spread_edge_from_points(
    point_edge: float,
    spread_odds: int | None = None,
) -> float:
    """Backward-compatible alias for spread_odds_edge."""
    juice = spread_odds if spread_odds is not None else DEFAULT_SPREAD_JUICE
    return spread_odds_edge(point_edge, juice)


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
        "strong_value": "Strong value",
        "value": "Value bet",
        "model_favorite": "Model favorite",
        "lean": "Lean",
    }
    return labels.get(code, code)


def _format_spread(value: float) -> str:
    return f"{value:+.1f}".replace(".0", "")


def american_to_decimal(american_odds: int) -> float:
    if american_odds >= 0:
        return 1.0 + american_odds / 100.0
    return 1.0 + 100.0 / abs(american_odds)


def american_implied_prob(american_odds: int) -> float:
    if american_odds >= 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def devig_two_way_probs(
    away_odds: int | None,
    home_odds: int | None,
) -> tuple[float | None, float | None]:
    """Remove book vig from a two-way moneyline (0–100 scale)."""
    if away_odds is None or home_odds is None:
        return None, None
    away_raw = american_implied_prob(away_odds)
    home_raw = american_implied_prob(home_odds)
    total = away_raw + home_raw
    if total <= 0:
        return None, None
    return away_raw / total * 100.0, home_raw / total * 100.0


def expected_value_pct(model_prob_pct: float, american_odds: int) -> float:
    """Expected profit per $1 staked, as a percentage (positive = +EV)."""
    probability = min(max(model_prob_pct, 0.1), 99.9) / 100.0
    payout = american_to_decimal(american_odds)
    return (probability * payout - 1.0) * 100.0


def kelly_fraction(
    model_prob_pct: float,
    american_odds: int,
    *,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Kelly criterion stake fraction for a +EV bet (capped)."""
    probability = min(max(model_prob_pct, 0.1), 99.9) / 100.0
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


def enrich_pick_profit_metrics(pick: BetPick) -> BetPick:
    """Attach EV%, Kelly, and profit_score used for official pick ranking."""
    if pick.bet_type == "spread":
        odds = int(pick.spread_odds if pick.spread_odds is not None else DEFAULT_SPREAD_JUICE)
        prob = float(pick.win_probability)
    else:
        odds = int(pick.market_odds)
        prob = float(pick.win_probability)
    pick.ev_pct = round(expected_value_pct(prob, odds), 2)
    kelly = kelly_fraction(prob, odds)
    pick.profit_score = round(
        pick_profit_score(model_prob_pct=prob, american_odds=odds, edge=pick.edge),
        4,
    )
    pick.extra["kelly_pct"] = round(kelly * 100.0, 2)
    pick.extra["expected_units"] = round(expected_units_per_bet(prob, odds), 4)
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
    """Official moneyline gate: +EV required; American edge OR model-favorite cross-sign."""
    if strategy == "model_favorite":
        return ev_pct >= MIN_CROSS_SIGN_EV_PCT
    if ev_pct < min_ev_pct:
        return False
    return edge >= min_edge


def resolve_binary_win_probs(
    blended: dict[str, Any] | None,
    total_score: float,
) -> tuple[float, float]:
    """Prefer calibrated blend probabilities over legacy total_score conversion."""
    if blended:
        if blended.get("threeway"):
            home_prob = float(blended.get("home_win_probability", 50.0))
            away_prob = float(blended.get("away_win_probability", 50.0))
            return away_prob, home_prob
        if blended.get("blended_home_win_probability") is not None:
            home_prob = float(blended["blended_home_win_probability"])
            return 100.0 - home_prob, home_prob
    return _side_win_probs(total_score)


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
    min_edge: float = MIN_RECOMMENDED_EDGE,
    min_ev_pct: float = MIN_EXPECTED_VALUE_PCT,
) -> list[BetPick]:
    if away_prob is None or home_prob is None:
        away_prob, home_prob = _side_win_probs(total_score)
    away_proj, home_proj = projections_from_win_probs(home_prob, away_prob)
    devig_away, devig_home = devig_two_way_probs(away_market, home_market)
    picks: list[BetPick] = []

    candidates: list[tuple[str, str, str, float, int, int | None, float | None]] = [
        ("away", away_name, away_slug, away_prob, away_proj, away_market, devig_away),
        ("home", home_name, home_slug, home_prob, home_proj, home_market, devig_home),
    ]

    for side, name, slug, outcome_prob, projection, market, market_implied in candidates:
        if market is None:
            continue

        edge = _odds_edge(projection, market, outcome_prob)
        ev_pct = expected_value_pct(outcome_prob, market)
        is_model_favorite = projection < 0
        is_market_underdog = market > 0

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
            )
        )

    return best_pick_only(picks)


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
    min_edge: float = MIN_RECOMMENDED_EDGE,
    min_ev_pct: float = MIN_EXPECTED_VALUE_PCT,
) -> list[BetPick]:
    """Evaluate 3-way soccer moneyline outcomes vs the book."""
    picks: list[BetPick] = []

    candidates: list[tuple[str, str, str, float, int, int | None]] = [
        ("away", away_name, away_slug, away_prob, away_proj, away_market),
        ("draw", "Draw", "draw", draw_prob, draw_proj, draw_market),
        ("home", home_name, home_slug, home_prob, home_proj, home_market),
    ]

    for side, name, slug, outcome_prob, projection, market in candidates:
        if market is None:
            continue

        edge = _odds_edge(projection, market, outcome_prob)
        ev_pct = expected_value_pct(outcome_prob, market)
        is_model_favorite = projection < 0
        is_market_underdog = market > 0

        if soccer_team_pick_blocked_by_projected_score(
            side,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
        ):
            continue

        outcome_label = "Draw" if side == "draw" else name
        strategy, confidence, reason = _moneyline_reason(
            name=name,
            projection=projection,
            market=market,
            edge=edge,
            ev_pct=ev_pct,
            outcome_prob=outcome_prob,
            market_implied_prob=american_implied_prob(market),
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
                market_implied_prob=round(american_implied_prob(market), 2),
                reason=reason,
            )
        )

    return best_pick_only(picks)


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
) -> list[BetPick]:
    """Recommend spread bets when model margin beats the consensus book line."""
    if consensus_spread is None:
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
        juice = spread_odds if spread_odds is not None else DEFAULT_SPREAD_JUICE
        edge = spread_odds_edge(point_edge, juice)
        if edge < min_edge:
            continue
        if min_point_edge is not None and point_edge < min_point_edge:
            continue

        line = spread_line_for_side(consensus_spread, side)
        side_cover_prob = spread_cover_probability(point_edge)
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

        picks.append(
            BetPick(
                side=side,
                team_name=name,
                team_slug=slug,
                strategy=strategy,
                confidence=confidence,
                edge=edge,
                ev_pct=round(expected_value_pct(side_cover_prob, juice), 2),
                model_projection=fair_spread_odds,
                market_odds=juice,
                win_probability=side_cover_prob,
                reason=reason,
                bet_type="spread",
                spread_line=line,
                spread_odds=juice,
                consensus_spread=consensus_spread,
                model_margin=round(model_margin, 2),
            )
        )

    picks.sort(key=lambda item: item.edge, reverse=True)
    return best_pick_only(picks)


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
