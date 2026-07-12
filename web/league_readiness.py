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


def _completed_game_stats(league: str, cutoff_date: str) -> tuple[int, int]:
    games = load_league_completed_games(league, cutoff_date)
    team_keys = {home for home, away, *_ in games} | {away for home, away, *_ in games}
    return len(games), len(team_keys)


def _v2_artifacts_ready(import_path: str) -> bool:
    """Best-effort check that a live v2 stack has on-disk artifacts."""
    try:
        module = __import__(import_path, fromlist=["artifacts_available"])
        return bool(module.artifacts_available())
    except Exception:
        return False


def assess_hockey_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Hockey slate readiness — prefer NHLGradientBoost v2, else Algo V1."""
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

    game_count, team_count = _completed_game_stats(league, cutoff_date)
    result["game_count"] = game_count
    result["team_count"] = team_count

    if league == "nhl" and _v2_artifacts_ready("web.nhl_v2.live"):
        result["ready"] = True
        result["reason"] = "NHLGradientBoost v2 ready"
        return result

    if game_count < ALGO_V1_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {ALGO_V1_MIN_LEAGUE_GAMES}+ completed games (have {game_count})"
        )
        return result

    if team_count < THREE_LAYER_MIN_TEAMS:
        result["reason"] = "Algo V1 needs 4+ teams with game history"
        return result

    result["ready"] = True
    result["reason"] = "Algo V1 ready"
    return result


def assess_basketball_readiness(league: str, cutoff_date: str) -> dict[str, Any]:
    """Basketball slate readiness — prefer GB v2 stacks, else BasketballMatrix."""
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

    game_count, team_count = _completed_game_stats(league, cutoff_date)
    result["game_count"] = game_count
    result["team_count"] = team_count

    v2_paths = {
        "nba": ("web.nba_v2.live", "NBAGradientBoost v2 ready"),
        "wnba": ("web.wnba_v2.live", "WNBAGradientBoost v2 ready"),
        "cbb": ("web.cbb_v2.live", "CBBGradientBoost v2 ready"),
        "ncaab": ("web.cbb_v2.live", "CBBGradientBoost v2 ready"),
    }
    v2 = v2_paths.get(league)
    if v2 and _v2_artifacts_ready(v2[0]):
        result["ready"] = True
        result["reason"] = v2[1]
        return result

    if game_count < BASKETBALL_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {BASKETBALL_MIN_LEAGUE_GAMES}+ completed games (have {game_count})"
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
    """MLB slate readiness — prefer MLBGradientBoost v2, else MLBRunCast."""
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

    game_count, team_count = _completed_game_stats(league, cutoff_date)
    result["game_count"] = game_count
    result["team_count"] = team_count

    if _v2_artifacts_ready("web.mlb_v2.live"):
        result["ready"] = True
        result["reason"] = "MLBGradientBoost v2 ready"
        return result

    if game_count < MLB_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {MLB_MIN_LEAGUE_GAMES}+ completed games (have {game_count})"
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
    """Soccer slate readiness — prefer SoccerGradientBoost v2, else SoccerPathA."""
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

    game_count, team_count = _completed_game_stats(league, cutoff_date)
    result["game_count"] = game_count
    result["team_count"] = team_count

    if _v2_artifacts_ready("web.soccer_v2.live"):
        result["ready"] = True
        result["reason"] = "SoccerGradientBoost v2 ready"
        return result

    if game_count < SOCCER_MIN_LEAGUE_GAMES:
        result["reason"] = (
            f"Need {SOCCER_MIN_LEAGUE_GAMES}+ completed games (have {game_count})"
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

