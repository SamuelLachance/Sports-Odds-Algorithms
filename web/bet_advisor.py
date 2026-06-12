"""Value-bet recommendation logic adapted from backtester strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from web.league_profiles import DEFAULT_SPREAD_JUICE, MIN_RECOMMENDED_EDGE, SOCCER_DRAW_BASE

# Approximate points of spread cushion per 1.0 American-odds edge unit.
SPREAD_POINT_TO_EDGE = 20.0

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
    extra: dict[str, Any] = field(default_factory=dict)


def _probability_to_american(probability_pct: float) -> int:
    probability = min(max(probability_pct, 0.1), 99.9)
    if probability >= 50.0:
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
    win_prob = abs(total_score)
    home_is_favorite = total_score <= 0
    home_prob = win_prob if home_is_favorite else 100.0 - win_prob
    away_prob = 100.0 - home_prob
    return projections_from_win_probs(home_prob, away_prob)


def soccer_threeway_probs(total_score: float, league: str) -> tuple[float, float, float]:
    """Return home win, draw, and away win probabilities (0–100 scale)."""
    win_prob = abs(total_score)
    home_is_favorite = total_score <= 0
    home_binary = win_prob if home_is_favorite else 100.0 - win_prob
    away_binary = 100.0 - home_binary

    base_draw = SOCCER_DRAW_BASE.get(league.lower(), SOCCER_DRAW_BASE["default"])
    closeness = 1.0 - abs(win_prob - 50.0) / 50.0
    draw_prob = min(35.0, max(18.0, base_draw + closeness * 8.0))

    scale = (100.0 - draw_prob) / 100.0
    home_prob = home_binary * scale
    away_prob = away_binary * scale
    return home_prob, draw_prob, away_prob


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
    """Projected home scoring margin (positive = home favored)."""
    win_prob = abs(total_score)
    scale = LEAGUE_MARGIN_SCALE.get(league.lower(), 0.14)
    margin = (win_prob - 50.0) * scale
    return margin if total_score < 0 else -margin


def spread_line_for_side(home_spread: float, side: str) -> float:
    """Spread line for the given side (home_spread is the book's home line)."""
    return home_spread if side == "home" else -home_spread


def spread_point_edge(model_margin_home: float, home_spread: float, side: str) -> float:
    """Point cushion vs the consensus spread for the bet side."""
    if side == "home":
        return model_margin_home + home_spread
    # Away covers when model away margin exceeds the number laid (home_spread when home is dog).
    return -model_margin_home - home_spread


def spread_edge_from_points(point_edge: float) -> float:
    return max(0.0, point_edge * SPREAD_POINT_TO_EDGE)


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
    if market_odds <= fair_odds:
        return 0.0

    if (fair_odds >= 0 and market_odds >= 0) or (fair_odds <= 0 and market_odds <= 0):
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


def best_pick_only(picks: list[BetPick]) -> list[BetPick]:
    """Return at most one pick per event (highest edge)."""
    if not picks:
        return []
    return [picks[0]]


def _side_win_probs(total_score: float) -> tuple[float, float]:
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
    outcome_prob: float,
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
            f"(fair underdog equivalent +{fair_underdog}, +{edge:.0f} edge)."
        )
    elif is_model_favorite and not is_market_underdog and edge >= 15:
        strategy = "strong_value"
        confidence = "high"
        reason = (
            f"Model favors {label} at {projection:+d}; "
            f"book line {market:+d} is softer (+{edge:.0f} edge)."
        )
    elif not is_model_favorite and not is_market_underdog and edge >= 15:
        strategy = "strong_value"
        confidence = "high"
        reason = (
            f"Model has {label} as {projection:+d} underdog; "
            f"book favorite price {market:+d} is too short (+{edge:.0f} edge)."
        )
    else:
        reason = (
            f"Sportsbook offers {market:+d} vs model fair {projection:+d} "
            f"on {label} (+{edge:.0f} edge on American odds)."
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
) -> list[BetPick]:
    away_proj, home_proj = model_moneylines(total_score)
    away_prob, home_prob = _side_win_probs(total_score)
    picks: list[BetPick] = []

    candidates: list[tuple[str, str, str, float, int, int | None]] = [
        ("away", away_name, away_slug, away_prob, away_proj, away_market),
        ("home", home_name, home_slug, home_prob, home_proj, home_market),
    ]

    for side, name, slug, outcome_prob, projection, market in candidates:
        if market is None:
            continue

        edge = _odds_edge(projection, market, outcome_prob)
        is_model_favorite = projection < 0
        is_market_underdog = market > 0

        if edge < MIN_RECOMMENDED_EDGE:
            continue

        strategy, confidence, reason = _moneyline_reason(
            name=name,
            projection=projection,
            market=market,
            edge=edge,
            outcome_prob=outcome_prob,
            is_model_favorite=is_model_favorite,
            is_market_underdog=is_market_underdog,
        )

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
                model_projection=projection,
                market_odds=market,
                win_probability=outcome_prob,
                reason=reason,
            )
        )

    picks.sort(key=lambda item: item.edge, reverse=True)
    return best_pick_only(picks)


def evaluate_soccer_picks(
    *,
    away_name: str,
    home_name: str,
    away_slug: str,
    home_slug: str,
    total_score: float,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    away_proj: int,
    draw_proj: int,
    home_proj: int,
    away_market: int | None,
    draw_market: int | None,
    home_market: int | None,
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
        is_model_favorite = projection < 0
        is_market_underdog = market > 0

        if edge < MIN_RECOMMENDED_EDGE:
            continue

        outcome_label = "Draw" if side == "draw" else name
        strategy, confidence, reason = _moneyline_reason(
            name=name,
            projection=projection,
            market=market,
            edge=edge,
            outcome_prob=outcome_prob,
            is_model_favorite=is_model_favorite,
            is_market_underdog=is_market_underdog,
            outcome_label=outcome_label,
        )

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
                model_projection=projection,
                market_odds=market,
                win_probability=outcome_prob,
                reason=reason,
            )
        )

    picks.sort(key=lambda item: item.edge, reverse=True)
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
) -> list[BetPick]:
    """Recommend spread bets when model margin beats the consensus book line."""
    if consensus_spread is None:
        return []

    model_margin = model_home_margin(total_score, league)
    picks: list[BetPick] = []

    candidates: list[tuple[str, str, str, int | None]] = [
        ("away", away_name, away_slug, away_spread_odds),
        ("home", home_name, home_slug, home_spread_odds),
    ]

    for side, name, slug, spread_odds in candidates:
        point_edge = spread_point_edge(model_margin, consensus_spread, side)
        edge = spread_edge_from_points(point_edge)
        if edge < MIN_RECOMMENDED_EDGE:
            continue

        line = spread_line_for_side(consensus_spread, side)
        juice = spread_odds if spread_odds is not None else DEFAULT_SPREAD_JUICE
        side_margin = model_margin if side == "home" else -model_margin

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
            f"Model projects {name} by {_format_spread(side_margin)} vs "
            f"consensus {_format_spread(line)} ({_format_spread(point_edge)} pt cushion, "
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
                model_projection=round(side_margin * SPREAD_POINT_TO_EDGE),
                market_odds=juice,
                win_probability=win_probability,
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
        "model_projection": pick.model_projection,
        "market_odds": pick.market_odds,
        "win_probability": round(pick.win_probability, 2),
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
