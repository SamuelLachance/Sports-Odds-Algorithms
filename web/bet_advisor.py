"""Value-bet recommendation logic adapted from backtester strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web.league_profiles import MIN_RECOMMENDED_EDGE


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


def pick_to_dict(pick: BetPick) -> dict[str, Any]:
    return {
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
    }
