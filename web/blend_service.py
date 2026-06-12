"""Blend legacy Algo_V2 with Sports-pred power ratings into a unified signal."""

from __future__ import annotations

from typing import Any

from web.power_model import predict_matchup
from web.season_games import get_league_power_context

LEGACY_BLEND_WEIGHT = 0.5
POWER_BLEND_WEIGHT = 0.5


def total_score_to_home_win_prob(total_score: float) -> float:
    """Convert Algo_V2 total_score to home win probability (0–100)."""
    if total_score <= 0:
        return abs(total_score)
    return 100.0 - abs(total_score)


def home_win_prob_to_total_score(home_win_prob: float) -> tuple[float, float]:
    """Convert home win probability to (total_score, win_probability)."""
    if home_win_prob >= 50.0:
        return -home_win_prob, home_win_prob
    away_prob = 100.0 - home_win_prob
    return away_prob, away_prob


def run_power_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Run power ratings for a matchup; None if insufficient data."""
    context = get_league_power_context(league, cutoff_date)
    if not context:
        return None

    teams, _games, param = context
    home_key = home_abbr.lower()
    away_key = away_abbr.lower()

    prediction = predict_matchup(teams, param, home_key, away_key)
    if not prediction:
        return None

    return {
        "algorithm": "PowerRatings",
        "home_power": prediction["home_power"],
        "away_power": prediction["away_power"],
        "power_diff": prediction["power_diff"],
        "home_win_probability": prediction["home_win_probability"],
        "param": prediction["param"],
        "home_games": prediction["home_games"],
        "away_games": prediction["away_games"],
    }


def blend_predictions(
    *,
    legacy_total_score: float,
    legacy_win_probability: float,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    legacy_weight: float = LEGACY_BLEND_WEIGHT,
    power_weight: float = POWER_BLEND_WEIGHT,
) -> dict[str, Any]:
    """
    Blend Algo_V2 and power model into unified total_score / win_probability.

    Falls back to legacy-only when power model cannot run.
    """
    legacy_payload = {
        "algorithm": "Algo_V2",
        "total_score": round(legacy_total_score, 2),
        "win_probability": round(legacy_win_probability, 2),
        "favorite_side": "home" if legacy_total_score < 0 else "away",
    }

    power_payload = run_power_model(league, cutoff_date, home_abbr, away_abbr)

    if not power_payload:
        total = legacy_total_score
        win_prob = legacy_win_probability
        return {
            "algorithm": "Unified",
            "blend_mode": "legacy_only",
            "blend_note": "Power model unavailable — using Algo V2 only.",
            "legacy": legacy_payload,
            "power": None,
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": legacy_payload["favorite_side"],
        }

    legacy_home = total_score_to_home_win_prob(legacy_total_score)
    power_home = float(power_payload["home_win_probability"])
    weight_sum = legacy_weight + power_weight
    blended_home = (
        legacy_weight * legacy_home + power_weight * power_home
    ) / weight_sum
    total, win_prob = home_win_prob_to_total_score(blended_home)

    return {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "blend_weights": {
            "legacy": legacy_weight,
            "power": power_weight,
        },
        "legacy": legacy_payload,
        "power": power_payload,
        "blended_home_win_probability": round(blended_home, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total < 0 else "away",
    }
