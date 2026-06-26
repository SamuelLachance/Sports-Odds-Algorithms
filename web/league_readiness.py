"""Gate baseball, hockey, and soccer leagues until the full three-layer model can run."""

from __future__ import annotations

from typing import Any

from web.baseball_pred_model import MIN_LEAGUE_GAMES as BASEBALL_MIN_LEAGUE_GAMES
from web.baseball_pred_model import get_baseball_pred_context, is_baseball_league
from web.hockey_pred_model import MIN_LEAGUE_GAMES as HOCKEY_MIN_LEAGUE_GAMES
from web.hockey_pred_model import get_hockey_pred_context, is_hockey_league
from web.league_profiles import LEAGUE_PROFILES, MIN_GAMES_FOR_POWER
from web.season_games import get_league_power_context, load_league_completed_games
from web.soccer_pred_model import MIN_LEAGUE_GAMES as SOCCER_MIN_LEAGUE_GAMES
from web.soccer_pred_model import get_soccer_pred_context

THREE_LAYER_MIN_TEAMS = 4


def uses_three_layer_readiness_gate(league: str) -> bool:
    profile = LEAGUE_PROFILES.get(league.lower())
    return profile is not None and profile["category"] in (
        "baseball",
        "hockey",
        "soccer",
    )


def assess_three_layer_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Return league-wide readiness for power + sport-specific third layers."""
    league = league.lower()
    result: dict[str, Any] = {
        "league": league,
        "ready": False,
        "power": False,
        "third_layer": False,
        "game_count": 0,
        "team_count": 0,
        "reason": "",
    }

    if not uses_three_layer_readiness_gate(league):
        result["ready"] = True
        result["reason"] = "Not a gated three-layer league"
        return result

    games = load_league_completed_games(league, cutoff_date)
    result["game_count"] = len(games)
    if len(games) < MIN_GAMES_FOR_POWER:
        result["reason"] = (
            f"Need {MIN_GAMES_FOR_POWER}+ completed games (have {len(games)})"
        )
        return result

    power_ctx = get_league_power_context(league, cutoff_date)
    if not power_ctx:
        result["reason"] = "Power ratings unavailable"
        return result

    teams, _, param = power_ctx
    result["team_count"] = len(teams)
    if param is None or len(teams) < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "Power ratings need 4+ teams with fitted curve"
        return result
    result["power"] = True

    if is_baseball_league(league):
        min_games = BASEBALL_MIN_LEAGUE_GAMES
    elif is_hockey_league(league):
        min_games = HOCKEY_MIN_LEAGUE_GAMES
    else:
        min_games = SOCCER_MIN_LEAGUE_GAMES

    if len(games) < min_games:
        result["reason"] = (
            f"Need {min_games}+ games for sport model (have {len(games)})"
        )
        return result

    if is_baseball_league(league):
        model = get_baseball_pred_context(league, cutoff_date)
    elif is_hockey_league(league):
        model = get_hockey_pred_context(league, cutoff_date)
    else:
        model = get_soccer_pred_context(league, cutoff_date)

    if not model:
        result["reason"] = "Sport-specific third layer unavailable"
        return result

    team_counts = model.get("team_game_counts") or {}
    if len(team_counts) < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "Sport model needs 4+ teams with game history"
        return result

    result["third_layer"] = True
    result["ready"] = True
    result["reason"] = "Full three-layer model ready"
    return result


def is_league_ready_for_daily_slate(league: str, cutoff_date: str) -> bool:
    return assess_three_layer_readiness(league, cutoff_date)["ready"]

