"""Blend internal model ratings into database team snapshots."""

from __future__ import annotations

from typing import Any

from web.baseball_pred_model import is_baseball_league
from web.basketball_pred_model import is_basketball_league
from web.football_pred_model import is_football_league
from web.hockey_pred_model import is_hockey_league
from web.league_profiles import LEAGUE_PROFILES
from web.season_games import get_league_power_context


def league_ratings_snapshot(league: str, cutoff_date: str) -> dict[str, Any]:
    """Return per-team model ratings when power/sport models are available."""
    league = league.lower()
    snapshot: dict[str, Any] = {
        "league": league,
        "cutoff_date": cutoff_date,
        "power": {},
        "sport_model": {},
        "source": [],
    }

    try:
        context = get_league_power_context(league, cutoff_date)
    except Exception:
        context = None

    if context:
        teams, _games, param = context
        snapshot["source"].append("power_ratings")
        snapshot["param"] = param
        for key, team in teams.items():
            snapshot["power"][key] = {
                "name": team.name,
                "power": round(team.power, 2),
                "schedule": round(team.schedule, 2),
                "apd": round(team.apd, 2),
                "wins": team.wins,
                "losses": team.losses,
                "ties": team.ties,
            }

    category = LEAGUE_PROFILES.get(league, {}).get("category")
    try:
        if is_basketball_league(league):
            from web.basketball_pred_model import get_basketball_pred_context

            model = get_basketball_pred_context(league, cutoff_date)
            if model:
                snapshot["source"].append("basketball_matrix")
                snapshot["sport_model"]["algorithm"] = "BasketballMatrix"
        elif is_baseball_league(league):
            from web.baseball_pred_model import get_baseball_pred_context

            model = get_baseball_pred_context(league, cutoff_date)
            if model and model.get("elos"):
                snapshot["source"].append("baseball_elo")
                snapshot["sport_model"]["algorithm"] = "BaseballElo"
                for key, elo in model["elos"].items():
                    snapshot["sport_model"].setdefault("teams", {})[key] = {
                        "elo_fast": round(elo.rating_fast, 1),
                        "elo_slow": round(elo.rating_slow, 1),
                    }
        elif is_hockey_league(league):
            from web.hockey_pred_model import get_hockey_pred_context

            model = get_hockey_pred_context(league, cutoff_date)
            if model:
                snapshot["source"].append("hockey_poisson")
                snapshot["sport_model"]["algorithm"] = "HockeyPoisson"
        elif is_football_league(league):
            from web.football_pred_model import get_football_pred_context

            model = get_football_pred_context(league, cutoff_date)
            if model and model.get("ratings"):
                snapshot["source"].append("nfelo")
                snapshot["sport_model"]["algorithm"] = "NfeloElo"
                for key, team in model["ratings"].items():
                    snapshot["sport_model"].setdefault("teams", {})[key] = {
                        "elo": round(team.rating, 1),
                        "games": team.games,
                    }
    except Exception:
        pass

    if category == "soccer":
        snapshot["source"].append("soccer_elo")
        snapshot["sport_model"]["algorithm"] = "SoccerElo"

    return snapshot


def team_rating_slice(
    ratings_snapshot: dict[str, Any],
    team_key: str,
) -> dict[str, Any]:
    key = team_key.lower()
    power = (ratings_snapshot.get("power") or {}).get(key)
    sport = (ratings_snapshot.get("sport_model") or {}).get("teams", {}).get(key)
    return {
        "power": power,
        "sport_model": sport,
        "sources": ratings_snapshot.get("source") or [],
    }
