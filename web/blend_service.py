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
from web.football_pred_model import (
    football_unavailable_reason,
    is_football_league,
    run_football_pred_model,
)
from web.hockey_pred_model import (
    hockey_unavailable_reason,
    is_hockey_league,
    run_hockey_pred_model,
)
from web.bet_advisor import (
    _odds_edge,
    model_home_margin,
    model_moneylines,
    spread_edge_from_points,
    spread_point_edge,
)
from web.league_profiles import DEFAULT_SPREAD_JUICE, MIN_RECOMMENDED_EDGE, uses_spread_bets
from web.live_data import resolve_team
from web.power_model import PowerTeam, predict_matchup
from web.season_games import get_league_power_context, power_unavailable_reason

LEGACY_BLEND_WEIGHT = 0.5
POWER_BLEND_WEIGHT = 0.5
THREE_LAYER_WEIGHT = 1.0 / 3.0


def total_score_to_home_win_prob(total_score: float) -> float:
    """Convert Algo_V2 total_score to home win probability (0–100)."""
    if abs(total_score) < 1e-9:
        return 50.0
    if total_score < 0:
        return abs(total_score)
    return 100.0 - abs(total_score)


def home_win_prob_to_total_score(home_win_prob: float) -> tuple[float, float]:
    """Convert home win probability to (total_score, win_probability)."""
    if abs(home_win_prob - 50.0) < 1e-9:
        return 0.0, 50.0
    if home_win_prob > 50.0:
        return -home_win_prob, home_win_prob
    away_prob = 100.0 - home_win_prob
    return away_prob, away_prob


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
    if is_hockey_league(league):
        return hockey_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    if is_football_league(league):
        return football_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
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
    if is_hockey_league(league):
        payload = run_hockey_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("hockey_pred", payload) if payload else (None, None)
    if is_football_league(league):
        payload = run_football_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("football_pred", payload) if payload else (None, None)
    return None, None


def _uses_three_layer_blend(league: str) -> bool:
    return (
        is_basketball_league(league)
        or is_baseball_league(league)
        or is_hockey_league(league)
        or is_football_league(league)
    )


def requires_three_layer_agreement(league: str) -> bool:
    """Leagues that require unanimous agreement across all three model layers."""
    return _uses_three_layer_blend(league)


def _layer_binary_total_score(layer: dict[str, Any]) -> float | None:
    if layer.get("total_score") is not None:
        return float(layer["total_score"])
    if layer.get("home_win_probability") is not None:
        total, _ = home_win_prob_to_total_score(float(layer["home_win_probability"]))
        return total
    return None


def _layer_side_win_probs(total_score: float) -> tuple[float, float]:
    if abs(total_score) < 1e-9:
        return 50.0, 50.0
    win_prob = abs(total_score)
    home_is_favorite = total_score <= 0
    home_prob = win_prob if home_is_favorite else 100.0 - win_prob
    away_prob = 100.0 - home_prob
    return away_prob, home_prob


def _layer_home_margin(layer: dict[str, Any], league: str) -> float | None:
    """Home scoring margin for spread checks; uses sport-specific projections when present."""
    if layer.get("predicted_margin") is not None:
        return float(layer["predicted_margin"])
    if layer.get("projected_spread") is not None:
        # nfelo projected_spread uses book sign (negative = home favored).
        return -float(layer["projected_spread"])
    total = _layer_binary_total_score(layer)
    if total is not None:
        return model_home_margin(total, league)
    return None


def blended_home_spread_margin(blended: dict[str, Any], league: str) -> float:
    """Average sport-aware home margins across blend layers for spread picks."""
    layer_keys = ("legacy", "power", "basketball_pred", "baseball_pred", "hockey_pred", "football_pred")
    margins: list[float] = []
    for key in layer_keys:
        layer = blended.get(key)
        if not layer:
            continue
        margin = _layer_home_margin(layer, league)
        if margin is not None:
            margins.append(margin)
    if margins:
        return sum(margins) / len(margins)
    total = blended.get("total_score")
    if total is not None:
        return model_home_margin(float(total), league)
    return 0.0


def _best_value_side_binary(
    total_score: float,
    away_market: int | None,
    home_market: int | None,
) -> str | None:
    away_prob, home_prob = _layer_side_win_probs(total_score)
    away_proj, home_proj = model_moneylines(total_score)
    edges: list[tuple[str, float]] = []
    if away_market is not None:
        edge = _odds_edge(away_proj, away_market, away_prob)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append(("away", edge))
    if home_market is not None:
        edge = _odds_edge(home_proj, home_market, home_prob)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append(("home", edge))
    if not edges:
        return None
    return max(edges, key=lambda item: item[1])[0]


def _layer_has_value_on_side_binary(
    total_score: float,
    side: str,
    away_market: int | None,
    home_market: int | None,
) -> bool:
    away_prob, home_prob = _layer_side_win_probs(total_score)
    away_proj, home_proj = model_moneylines(total_score)
    if side == "away":
        return (
            away_market is not None
            and _odds_edge(away_proj, away_market, away_prob) >= MIN_RECOMMENDED_EDGE
        )
    return (
        home_market is not None
        and _odds_edge(home_proj, home_market, home_prob) >= MIN_RECOMMENDED_EDGE
    )


def _layer_has_spread_value_on_side(
    layer: dict[str, Any],
    league: str,
    side: str,
    consensus_spread: float,
    *,
    spread_odds: int | None = None,
) -> bool:
    margin = _layer_home_margin(layer, league)
    if margin is None:
        return False
    point_edge = spread_point_edge(margin, consensus_spread, side)
    juice = spread_odds if spread_odds is not None else DEFAULT_SPREAD_JUICE
    return spread_edge_from_points(point_edge, juice) >= MIN_RECOMMENDED_EDGE


def _best_value_spread_side(
    layer: dict[str, Any],
    league: str,
    consensus_spread: float,
    *,
    away_spread_odds: int | None = None,
    home_spread_odds: int | None = None,
) -> str | None:
    margin = _layer_home_margin(layer, league)
    if margin is None:
        return None
    edges: list[tuple[str, float]] = []
    for side, juice in (
        ("away", away_spread_odds),
        ("home", home_spread_odds),
    ):
        point_edge = spread_point_edge(margin, consensus_spread, side)
        edge = spread_edge_from_points(point_edge, juice)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append((side, edge))
    if not edges:
        return None
    return max(edges, key=lambda item: item[1])[0]


def _third_layer_key(blended: dict[str, Any]) -> str | None:
    for key in ("basketball_pred", "baseball_pred", "hockey_pred", "football_pred"):
        if blended.get(key):
            return key
    return None


def compute_model_agreement(
    blended: dict[str, Any],
    league: str,
    *,
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    For 3-layer leagues, all layers must independently find value on the same side.

    Spread leagues: each layer needs spread edge >= MIN_RECOMMENDED_EDGE on one shared side.
    Other 3-layer sports: each layer needs moneyline edge >= MIN_RECOMMENDED_EDGE on one shared side.
    """
    if not requires_three_layer_agreement(league):
        return {"required": 0, "agreed": True, "agreement_mode": "value"}

    legacy = blended.get("legacy")
    power = blended.get("power")
    third_key = _third_layer_key(blended)
    third_payload = blended.get(third_key) if third_key else None
    market = market or {}

    away_market = market.get("away_moneyline")
    home_market = market.get("home_moneyline")
    consensus_spread = market.get("spread")
    away_spread_odds = market.get("away_spread_odds")
    home_spread_odds = market.get("home_spread_odds")
    use_spread = uses_spread_bets(league) and consensus_spread is not None

    def _spread_kwargs() -> dict[str, Any]:
        return {
            "away_spread_odds": away_spread_odds,
            "home_spread_odds": home_spread_odds,
        }

    def _incomplete_payload() -> dict[str, Any]:
        if use_spread and consensus_spread is not None:
            spread_line = float(consensus_spread)
            sk = _spread_kwargs()
            legacy_side = (
                _best_value_spread_side(legacy, league, spread_line, **sk)
                if legacy
                else None
            )
            power_side = (
                _best_value_spread_side(power, league, spread_line, **sk)
                if power
                else None
            )
            third_side = (
                _best_value_spread_side(third_payload, league, spread_line, **sk)
                if third_payload
                else None
            )
        else:
            legacy_total = _layer_binary_total_score(legacy) if legacy else None
            power_total = _layer_binary_total_score(power) if power else None
            third_total = (
                _layer_binary_total_score(third_payload) if third_payload else None
            )
            legacy_side = (
                _best_value_side_binary(legacy_total, away_market, home_market)
                if legacy_total is not None
                else None
            )
            power_side = (
                _best_value_side_binary(power_total, away_market, home_market)
                if power_total is not None
                else None
            )
            third_side = (
                _best_value_side_binary(third_total, away_market, home_market)
                if third_total is not None
                else None
            )
        return {
            "required": 3,
            "agreed": False,
            "legacy_side": legacy_side,
            "power_side": power_side,
            "third_side": third_side,
            "third_source": third_key,
            "agreement_mode": "value",
            "value_sides": [],
            "value_outcomes": [],
        }

    if not legacy or not power or not third_payload or blended.get("blend_layers", 0) < 3:
        return _incomplete_payload()

    legacy_total = _layer_binary_total_score(legacy)
    power_total = _layer_binary_total_score(power)
    third_total = _layer_binary_total_score(third_payload)
    if legacy_total is None or power_total is None or third_total is None:
        return _incomplete_payload()

    if use_spread:
        spread_line = float(consensus_spread)
        sk = _spread_kwargs()
        legacy_side = _best_value_spread_side(legacy, league, spread_line, **sk)
        power_side = _best_value_spread_side(power, league, spread_line, **sk)
        third_side = _best_value_spread_side(third_payload, league, spread_line, **sk)
        value_sides = [
            side
            for side in ("away", "home")
            if _layer_has_spread_value_on_side(
                legacy,
                league,
                side,
                spread_line,
                spread_odds=away_spread_odds if side == "away" else home_spread_odds,
            )
            and _layer_has_spread_value_on_side(
                power,
                league,
                side,
                spread_line,
                spread_odds=away_spread_odds if side == "away" else home_spread_odds,
            )
            and _layer_has_spread_value_on_side(
                third_payload,
                league,
                side,
                spread_line,
                spread_odds=away_spread_odds if side == "away" else home_spread_odds,
            )
        ]
    else:
        legacy_side = _best_value_side_binary(legacy_total, away_market, home_market)
        power_side = _best_value_side_binary(power_total, away_market, home_market)
        third_side = _best_value_side_binary(third_total, away_market, home_market)
        value_sides = [
            side
            for side in ("away", "home")
            if _layer_has_value_on_side_binary(
                legacy_total, side, away_market, home_market
            )
            and _layer_has_value_on_side_binary(
                power_total, side, away_market, home_market
            )
            and _layer_has_value_on_side_binary(
                third_total, side, away_market, home_market
            )
        ]

    return {
        "required": 3,
        "agreed": bool(value_sides),
        "legacy_side": legacy_side,
        "power_side": power_side,
        "third_side": third_side,
        "third_source": third_key,
        "agreement_mode": "value",
        "value_sides": value_sides,
        "value_outcomes": value_sides,
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
        "predicted_margin": prediction["power_diff"],
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

    Basketball, baseball, hockey, and football use a third layer with equal 1/3 weights
    on home win probability.
    """
    legacy_payload = {
        "algorithm": "Algo_V2",
        "total_score": round(legacy_total_score, 2),
        "win_probability": round(legacy_win_probability, 2),
        "favorite_side": "home" if legacy_total_score <= 0 else "away",
    }

    power_payload = run_power_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
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
        blended_home = min(max(blended_home, 0.0), 100.0)
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
            "favorite_side": "home" if total <= 0 else "away",
        }
        return result

    weight_sum = legacy_weight + power_weight
    blended_home = (
        legacy_weight * legacy_home + power_weight * power_home
    ) / weight_sum
    blended_home = min(max(blended_home, 0.0), 100.0)
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
        "favorite_side": "home" if total <= 0 else "away",
    }

    if _uses_three_layer_blend(league) and not sport_payload:
        if is_basketball_league(league):
            layer_name = "Basketball matrix"
        elif is_baseball_league(league):
            layer_name = "MLB-Model"
        elif is_hockey_league(league):
            layer_name = "Hockey-predictions"
        else:
            layer_name = "nfelo"
        reason = _sport_pred_unavailable_reason(
            league, cutoff_date, home_abbr, away_abbr
        )
        result["blend_note"] = (
            f"{layer_name} layer unavailable — {reason} Using 2-layer blend."
        )

    return result
