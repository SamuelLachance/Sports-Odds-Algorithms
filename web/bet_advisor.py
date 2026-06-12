"""Value-bet recommendation logic adapted from backtester strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from web.league_profiles import DEFAULT_SPREAD_JUICE, MIN_RECOMMENDED_EDGE

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


def _american_from_probability(win_probability: float) -> tuple[int, int]:
    probability = min(max(abs(win_probability), 0.1), 99.9)
    line = round((100 / (100 - probability) - 1) * 100)
    return line, line


def model_moneylines(total_score: float) -> tuple[int, int]:
    """Return projected American odds for away and home teams."""
    odds = abs(total_score)
    favorite_line, underdog_line = _american_from_probability(odds)

    if total_score < 0:
        away_proj = underdog_line
        home_proj = -favorite_line
    else:
        away_proj = favorite_line
        home_proj = -underdog_line

    return away_proj, home_proj


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
    return home_spread - model_margin_home


def spread_edge_from_points(point_edge: float) -> float:
    return max(0.0, point_edge * SPREAD_POINT_TO_EDGE)


def _odds_edge(model_projection: int, market_odds: int) -> float:
    if market_odds > model_projection:
        return float(market_odds - model_projection)
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
    picks: list[BetPick] = []

    candidates: list[tuple[str, str, str, int, int | None]] = [
        ("away", away_name, away_slug, away_proj, away_market),
        ("home", home_name, home_slug, home_proj, home_market),
    ]

    for side, name, slug, projection, market in candidates:
        if market is None:
            continue

        edge = _odds_edge(projection, market)
        is_model_favorite = projection < 0
        is_market_underdog = market > 0
        diff = abs(projection - market)
        if diff > 200:
            diff = abs(projection - market) - 200

        if edge < MIN_RECOMMENDED_EDGE:
            continue

        strategy = "value"
        confidence = "medium"
        reason = (
            f"Sportsbook offers {market:+d} vs model {projection:+d} "
            f"(+{edge:.0f} edge on American odds)."
        )

        if is_model_favorite and edge >= 15:
            strategy = "strong_value"
            confidence = "high"
            reason = (
                f"Model favors {name} and the book price ({market:+d}) "
                f"beats the model line ({projection:+d})."
            )
        elif is_model_favorite and is_market_underdog and diff >= 25:
            strategy = "model_favorite"
            confidence = "high"
            reason = (
                f"Model favorite priced as underdog at {market:+d}; "
                f"model implies {projection:+d}."
            )
        elif edge >= 8:
            strategy = "value"
            confidence = "medium"
        else:
            strategy = "lean"
            confidence = "low"

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
                win_probability=win_probability,
                reason=reason,
            )
        )

    picks.sort(key=lambda item: item.edge, reverse=True)
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
    return picks


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
