"""Gate baseball, hockey, and soccer leagues until the full three-layer model can run."""

from __future__ import annotations

from typing import Any

from web.baseball_pred_model import MIN_LEAGUE_GAMES as BASEBALL_MIN_LEAGUE_GAMES
from web.baseball_pred_model import get_baseball_pred_context
from web.basketball_pred_model import (
    MIN_LEAGUE_GAMES as BASKETBALL_MIN_LEAGUE_GAMES,
    get_basketball_pred_context,
    is_basketball_league,
)
from web.hockey_pred_model import is_hockey_league
from web.mlb_pred_model import MIN_LEAGUE_GAMES as MLB_MIN_LEAGUE_GAMES
from web.mlb_pred_model import get_mlb_pred_context, is_mlb_league
from web.league_profiles import LEAGUE_PROFILES, MIN_GAMES_FOR_POWER, is_soccer_league
from web.season_games import get_league_power_context, load_league_completed_games
from web.soccer_pred_model import MIN_LEAGUE_GAMES as SOCCER_MIN_LEAGUE_GAMES
from web.soccer_pred_model import get_soccer_pred_context

THREE_LAYER_MIN_TEAMS = 4


def uses_three_layer_readiness_gate(league: str) -> bool:
    league = league.lower()
    if is_mlb_league(league) or is_soccer_league(league):
        return False
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] in (
        "baseball",
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

    # Only non-MLB baseball reaches this gate (hockey/soccer/MLB use assess_* helpers).
    # Do not branch on hockey/soccer here — those names were never imported and are unreachable.
    if len(games) < BASEBALL_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {BASEBALL_MIN_LEAGUE_GAMES}+ games for sport model (have {len(games)})"
        )
        return result

    model = get_baseball_pred_context(league, cutoff_date)
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


ALGO_V1_MIN_LEAGUE_GAMES = 10


def assess_hockey_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Hockey slate readiness — Algo V1 only (legacy weighted-factor model)."""
    league = league.lower()
    result: dict[str, Any] = {
        "league": league,
        "ready": False,
        "game_count": 0,
        "team_count": 0,
        "reason": "",
    }
    if not is_hockey_league(league):
        result["reason"] = "Not a hockey league"
        return result

    games = load_league_completed_games(league, cutoff_date)
    result["game_count"] = len(games)
    if len(games) < ALGO_V1_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {ALGO_V1_MIN_LEAGUE_GAMES}+ completed games (have {len(games)})"
        )
        return result

    team_keys = {home for home, away, *_ in games} | {away for home, away, *_ in games}
    result["team_count"] = len(team_keys)
    if len(team_keys) < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "Algo V1 needs 4+ teams with game history"
        return result

    result["ready"] = True
    result["reason"] = "Algo V1 ready"
    return result


def assess_basketball_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Basketball slate readiness — BasketballMatrix only (soft-impute OR × pace)."""
    league = league.lower()
    result: dict[str, Any] = {
        "league": league,
        "ready": False,
        "game_count": 0,
        "team_count": 0,
        "reason": "",
    }
    if not is_basketball_league(league):
        result["reason"] = "Not a basketball league"
        return result

    games = load_league_completed_games(league, cutoff_date)
    result["game_count"] = len(games)
    if len(games) < BASKETBALL_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {BASKETBALL_MIN_LEAGUE_GAMES}+ completed games (have {len(games)})"
        )
        return result

    model = get_basketball_pred_context(league, cutoff_date)
    if not model:
        result["reason"] = "BasketballMatrix model unavailable"
        return result

    team_counts = model.get("team_game_counts") or {}
    result["team_count"] = len(team_counts)
    if len(team_counts) < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "BasketballMatrix needs 4+ teams with game history"
        return result

    result["ready"] = True
    result["reason"] = "BasketballMatrix ready"
    return result


def assess_mlb_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """MLB slate readiness — MLBRunCast only (no power / legacy stack)."""
    league = league.lower()
    result: dict[str, Any] = {
        "league": league,
        "ready": False,
        "game_count": 0,
        "team_count": 0,
        "reason": "",
    }
    if not is_mlb_league(league):
        result["reason"] = "Not an MLB league"
        return result

    games = load_league_completed_games(league, cutoff_date)
    result["game_count"] = len(games)
    if len(games) < MLB_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {MLB_MIN_LEAGUE_GAMES}+ completed games (have {len(games)})"
        )
        return result

    model = get_mlb_pred_context(league, cutoff_date)
    if not model:
        result["reason"] = "MLBRunCast model unavailable"
        return result

    team_counts = model.get("team_game_counts") or {}
    result["team_count"] = len(team_counts)
    if len(team_counts) < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "MLBRunCast needs 4+ teams with game history"
        return result

    result["ready"] = True
    result["reason"] = "MLBRunCast ready"
    return result


def assess_soccer_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Soccer slate readiness — SoccerPathA only (no power / legacy stack)."""
    league = league.lower()
    result: dict[str, Any] = {
        "league": league,
        "ready": False,
        "game_count": 0,
        "team_count": 0,
        "reason": "",
    }
    if not is_soccer_league(league):
        result["reason"] = "Not a soccer league"
        return result

    games = load_league_completed_games(league, cutoff_date)
    result["game_count"] = len(games)
    if len(games) < SOCCER_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {SOCCER_MIN_LEAGUE_GAMES}+ completed games (have {len(games)})"
        )
        return result

    model = get_soccer_pred_context(league, cutoff_date)
    if not model:
        result["reason"] = "SoccerPathA model unavailable"
        return result

    team_counts = model.get("team_game_counts") or {}
    result["team_count"] = len(team_counts)
    if len(team_counts) < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "SoccerPathA needs 4+ teams with game history"
        return result

    result["ready"] = True
    result["reason"] = "SoccerPathA ready"
    return result


def assess_league_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Unified readiness assessment with a human-readable ``reason``."""
    if is_basketball_league(league):
        return assess_basketball_readiness(league, cutoff_date)
    if is_hockey_league(league):
        return assess_hockey_readiness(league, cutoff_date)
    if is_mlb_league(league):
        return assess_mlb_readiness(league, cutoff_date)
    if is_soccer_league(league):
        return assess_soccer_readiness(league, cutoff_date)
    return assess_three_layer_readiness(league, cutoff_date)


def is_league_ready_for_daily_slate(league: str, cutoff_date: str) -> bool:
    return bool(assess_league_readiness(league, cutoff_date).get("ready"))

