"""Blend legacy Algo_V2 with Sports-pred power ratings into a unified signal."""

from __future__ import annotations

from typing import Any

from web.baseball_pred_model import (
    baseball_unavailable_reason,
    is_baseball_league,
    run_baseball_pred_model,
)
from web.basketball_pred_model import (
    is_basketball_league,
    run_basketball_pred_model,
)
from web.bet_advisor import soccer_threeway_probs
from web.league_profiles import is_soccer_league
from web.live_data import resolve_team
from web.power_model import PowerTeam, predict_matchup
from web.season_games import get_league_power_context, power_unavailable_reason
from web.soccer_pred_model import (
    run_soccer_pred_model,
    soccer_unavailable_reason,
)

LEGACY_BLEND_WEIGHT = 0.5
POWER_BLEND_WEIGHT = 0.5
THREE_LAYER_WEIGHT = 1.0 / 3.0


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


def threeway_probs_to_total_score(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
) -> tuple[float, float, str]:
    """Convert blended 1X2 probabilities back to binary total_score fields."""
    if draw_prob > home_prob and draw_prob > away_prob:
        return 0.0, draw_prob, "home"
    non_draw = home_prob + away_prob
    if non_draw <= 0:
        return 0.0, 50.0, "home"
    home_binary = home_prob / non_draw * 100.0
    total, win_prob = home_win_prob_to_total_score(home_binary)
    favorite = "home" if home_prob >= away_prob else "away"
    return total, win_prob, favorite


def _normalize_threeway_blend(
    home: float, draw: float, away: float
) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return 33.33, 33.33, 33.34
    scale = 100.0 / total
    return round(home * scale, 2), round(draw * scale, 2), round(away * scale, 2)


def _blend_threeway_layers(
    layers: list[tuple[float, float, float]],
    weights: list[float],
) -> tuple[float, float, float]:
    weight_sum = sum(weights)
    home = sum(w * layer[0] for w, layer in zip(weights, layers)) / weight_sum
    draw = sum(w * layer[1] for w, layer in zip(weights, layers)) / weight_sum
    away = sum(w * layer[2] for w, layer in zip(weights, layers)) / weight_sum
    return _normalize_threeway_blend(home, draw, away)


def _power_threeway_probs(
    power_home: float, league: str
) -> tuple[float, float, float]:
    power_total, _ = home_win_prob_to_total_score(power_home)
    return soccer_threeway_probs(power_total, league)


def _find_team_key(
    teams: dict[str, PowerTeam],
    league: str,
    abbr: str,
    display_name: str | None = None,
) -> str | None:
    """Resolve registry/ESPN abbreviations to power-rating team keys."""
    candidates: list[str] = []
    resolved = resolve_team(league, abbr, display_name)
    if resolved:
        candidates.append(resolved[0].lower())
    candidates.append(abbr.lower())

    for key in candidates:
        if key in teams:
            return key

    if display_name:
        target = display_name.lower()
        for key, team in teams.items():
            if team.name.lower() == target:
                return key

    return None


def _sport_pred_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    if is_basketball_league(league):
        from web.basketball_pred_model import MIN_LEAGUE_GAMES, MIN_TEAM_GAMES
        from web.season_games import load_league_completed_games

        games = load_league_completed_games(league, cutoff_date)
        if len(games) < MIN_LEAGUE_GAMES:
            return (
                f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
                "— likely off-season or sparse schedule."
            )
        from web.basketball_pred_model import build_basketball_model

        model = build_basketball_model(games, league)
        if not model:
            return "Could not build basketball matrix model on available games."
        counts = model["team_game_counts"]
        home = home_abbr.lower()
        away = away_abbr.lower()
        if home not in counts or away not in counts:
            missing = [k for k in (home, away) if k not in counts]
            return f"Teams not found in basketball model: {', '.join(missing)}."
        if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
            return "Teams have insufficient games in the basketball model sample."
        return "Basketball matrix model unavailable."
    if is_baseball_league(league):
        return baseball_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    if is_soccer_league(league):
        return soccer_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    return "Sport-specific model unavailable."


def _run_sport_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_name: str | None = None,
    away_name: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return (payload_key, payload) for sport-specific third layer."""
    if is_basketball_league(league):
        payload = run_basketball_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("basketball_pred", payload) if payload else (None, None)
    if is_baseball_league(league):
        payload = run_baseball_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("baseball_pred", payload) if payload else (None, None)
    if is_soccer_league(league):
        payload = run_soccer_pred_model(
            league,
            cutoff_date,
            home_abbr,
            away_abbr,
            home_name=home_name,
            away_name=away_name,
        )
        return ("soccer_pred", payload) if payload else (None, None)
    return None, None


def _uses_three_layer_blend(league: str) -> bool:
    return (
        is_basketball_league(league)
        or is_baseball_league(league)
        or is_soccer_league(league)
    )


def _blend_soccer_predictions(
    *,
    legacy_total_score: float,
    legacy_win_probability: float,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str | None = None,
    away_name: str | None = None,
    legacy_payload: dict[str, Any],
    power_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy_threeway = soccer_threeway_probs(legacy_total_score, league)
    legacy_threeway_payload = {
        "algorithm": "Algo_V2",
        "home_win_probability": round(legacy_threeway[0], 2),
        "draw_probability": round(legacy_threeway[1], 2),
        "away_win_probability": round(legacy_threeway[2], 2),
        "total_score": legacy_payload["total_score"],
        "favorite_side": legacy_payload["favorite_side"],
    }

    if not power_payload:
        total = legacy_total_score
        win_prob = legacy_win_probability
        reason = power_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        home_p, draw_p, away_p = legacy_threeway
        return {
            "algorithm": "Unified",
            "blend_mode": "legacy_only",
            "blend_note": f"Power model unavailable — {reason} Using Algo V2 only.",
            "legacy": legacy_payload,
            "legacy_threeway": legacy_threeway_payload,
            "power": None,
            "threeway": True,
            "home_win_probability": round(home_p, 2),
            "draw_probability": round(draw_p, 2),
            "away_win_probability": round(away_p, 2),
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": legacy_payload["favorite_side"],
        }

    power_threeway = _power_threeway_probs(float(power_payload["home_win_probability"]), league)
    power_threeway_payload = {
        "algorithm": "PowerRatings",
        "home_win_probability": round(power_threeway[0], 2),
        "draw_probability": round(power_threeway[1], 2),
        "away_win_probability": round(power_threeway[2], 2),
        "home_power": power_payload["home_power"],
        "away_power": power_payload["away_power"],
    }

    sport_key, sport_payload = _run_sport_pred_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
    )

    if sport_payload and sport_key:
        soccer_threeway = (
            float(sport_payload["home_win_probability"]),
            float(sport_payload["draw_probability"]),
            float(sport_payload["away_win_probability"]),
        )
        home_p, draw_p, away_p = _blend_threeway_layers(
            [legacy_threeway, power_threeway, soccer_threeway],
            [THREE_LAYER_WEIGHT, THREE_LAYER_WEIGHT, THREE_LAYER_WEIGHT],
        )
        total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
        return {
            "algorithm": "Unified",
            "blend_mode": "blended",
            "blend_layers": 3,
            "blend_weights": {
                "legacy": THREE_LAYER_WEIGHT,
                "power": THREE_LAYER_WEIGHT,
                sport_key: THREE_LAYER_WEIGHT,
            },
            "legacy": legacy_payload,
            "legacy_threeway": legacy_threeway_payload,
            "power": power_payload,
            "power_threeway": power_threeway_payload,
            sport_key: sport_payload,
            "threeway": True,
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
            "blended_threeway": {
                "home_win_probability": home_p,
                "draw_probability": draw_p,
                "away_win_probability": away_p,
            },
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": favorite,
        }

    home_p, draw_p, away_p = _blend_threeway_layers(
        [legacy_threeway, power_threeway],
        [LEGACY_BLEND_WEIGHT, POWER_BLEND_WEIGHT],
    )
    total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
    reason = _sport_pred_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    return {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "blend_layers": 2,
        "blend_weights": {
            "legacy": LEGACY_BLEND_WEIGHT,
            "power": POWER_BLEND_WEIGHT,
        },
        "legacy": legacy_payload,
        "legacy_threeway": legacy_threeway_payload,
        "power": power_payload,
        "power_threeway": power_threeway_payload,
        "threeway": True,
        "home_win_probability": home_p,
        "draw_probability": draw_p,
        "away_win_probability": away_p,
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": favorite,
        "blend_note": (
            f"Football-predictor layer unavailable — {reason} Using 2-layer blend."
        ),
    }


def run_power_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_name: str | None = None,
    away_name: str | None = None,
) -> dict[str, Any] | None:
    """Run power ratings for a matchup; None if insufficient data."""
    context = get_league_power_context(league, cutoff_date)
    if not context:
        return None

    teams, _games, param = context
    home_key = _find_team_key(teams, league, home_abbr, home_name)
    away_key = _find_team_key(teams, league, away_abbr, away_name)
    if not home_key or not away_key:
        return None

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
    home_name: str | None = None,
    away_name: str | None = None,
    legacy_weight: float = LEGACY_BLEND_WEIGHT,
    power_weight: float = POWER_BLEND_WEIGHT,
) -> dict[str, Any]:
    """
    Blend Algo_V2 and power model into unified total_score / win_probability.

    Basketball and baseball use a third layer with equal 1/3 weights on home win
    probability. Soccer uses equal 1/3 weights on home, draw, and away separately.
    """
    legacy_payload = {
        "algorithm": "Algo_V2",
        "total_score": round(legacy_total_score, 2),
        "win_probability": round(legacy_win_probability, 2),
        "favorite_side": "home" if legacy_total_score < 0 else "away",
    }

    power_payload = run_power_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
    )

    if is_soccer_league(league):
        return _blend_soccer_predictions(
            legacy_total_score=legacy_total_score,
            legacy_win_probability=legacy_win_probability,
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_name=home_name,
            away_name=away_name,
            legacy_payload=legacy_payload,
            power_payload=power_payload,
        )

    if not power_payload:
        total = legacy_total_score
        win_prob = legacy_win_probability
        reason = power_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        return {
            "algorithm": "Unified",
            "blend_mode": "legacy_only",
            "blend_note": f"Power model unavailable — {reason} Using Algo V2 only.",
            "legacy": legacy_payload,
            "power": None,
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": legacy_payload["favorite_side"],
        }

    legacy_home = total_score_to_home_win_prob(legacy_total_score)
    power_home = float(power_payload["home_win_probability"])
    sport_key, sport_payload = _run_sport_pred_model(
        league, cutoff_date, home_abbr, away_abbr
    )

    if _uses_three_layer_blend(league) and sport_payload and sport_key:
        third_home = float(sport_payload["home_win_probability"])
        blended_home = (
            THREE_LAYER_WEIGHT * legacy_home
            + THREE_LAYER_WEIGHT * power_home
            + THREE_LAYER_WEIGHT * third_home
        )
        total, win_prob = home_win_prob_to_total_score(blended_home)
        result: dict[str, Any] = {
            "algorithm": "Unified",
            "blend_mode": "blended",
            "blend_layers": 3,
            "blend_weights": {
                "legacy": THREE_LAYER_WEIGHT,
                "power": THREE_LAYER_WEIGHT,
                sport_key: THREE_LAYER_WEIGHT,
            },
            "legacy": legacy_payload,
            "power": power_payload,
            sport_key: sport_payload,
            "blended_home_win_probability": round(blended_home, 2),
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": "home" if total < 0 else "away",
        }
        return result

    weight_sum = legacy_weight + power_weight
    blended_home = (
        legacy_weight * legacy_home + power_weight * power_home
    ) / weight_sum
    total, win_prob = home_win_prob_to_total_score(blended_home)

    result = {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "blend_layers": 2,
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

    if _uses_three_layer_blend(league) and not sport_payload:
        if is_basketball_league(league):
            layer_name = "Basketball matrix"
        elif is_baseball_league(league):
            layer_name = "MLB-Model"
        else:
            layer_name = "Football-predictor"
        reason = _sport_pred_unavailable_reason(
            league, cutoff_date, home_abbr, away_abbr
        )
        result["blend_note"] = (
            f"{layer_name} layer unavailable — {reason} Using 2-layer blend."
        )

    return result
