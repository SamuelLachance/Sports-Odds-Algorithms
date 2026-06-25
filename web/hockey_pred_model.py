"""NHL win probability via Poisson xG model (ported from gmalbert/hockey-predictions).

Source: https://github.com/gmalbert/hockey-predictions
- expected_goals.calculate_expected_goals
- win_probability.calculate_win_probability
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

from web.league_profiles import LEAGUE_PROFILES
from web.season_games import load_league_completed_games

HOCKEY_LEAGUES: tuple[str, ...] = ("nhl",)

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3
HOME_ADVANTAGE = 0.15
HOME_OT_ADVANTAGE = 0.52
MAX_GOALS = 10


class GameProbabilities(NamedTuple):
    home_win: float
    away_win: float
    home_regulation: float
    away_regulation: float
    overtime: float


@dataclass
class TeamMetrics:
    team: str
    goals_for_pg: float
    goals_against_pg: float
    games_played: int = 0


def is_hockey_league(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] == "hockey"


def _poisson_prob(expected: float, actual: int) -> float:
    if actual < 0:
        return 0.0
    return (math.exp(-expected) * (expected**actual)) / math.factorial(actual)


def calculate_win_probability(
    home_xg: float,
    away_xg: float,
    *,
    home_ot_advantage: float = HOME_OT_ADVANTAGE,
    max_goals: int = MAX_GOALS,
) -> GameProbabilities:
    """Poisson regulation + OT split (hockey-predictions win_probability.py)."""
    home_reg_win = 0.0
    away_reg_win = 0.0
    tie_prob = 0.0

    for home_goals in range(max_goals + 1):
        home_prob = _poisson_prob(home_xg, home_goals)
        for away_goals in range(max_goals + 1):
            away_prob = _poisson_prob(away_xg, away_goals)
            combined = home_prob * away_prob
            if home_goals > away_goals:
                home_reg_win += combined
            elif away_goals > home_goals:
                away_reg_win += combined
            else:
                tie_prob += combined

    ot_home_win = tie_prob * home_ot_advantage
    ot_away_win = tie_prob * (1 - home_ot_advantage)

    return GameProbabilities(
        home_win=round(home_reg_win + ot_home_win, 4),
        away_win=round(away_reg_win + ot_away_win, 4),
        home_regulation=round(home_reg_win, 4),
        away_regulation=round(away_reg_win, 4),
        overtime=round(tie_prob, 4),
    )


def calculate_expected_goals(
    home_team: TeamMetrics,
    away_team: TeamMetrics,
    *,
    home_advantage: float = HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Blend offense vs opponent defense with home ice (expected_goals.py)."""
    home_xg = (home_team.goals_for_pg + away_team.goals_against_pg) / 2
    home_xg *= 1 + home_advantage

    away_xg = (away_team.goals_for_pg + home_team.goals_against_pg) / 2
    away_xg *= 1 - home_advantage / 2

    return round(max(0.5, home_xg), 2), round(max(0.5, away_xg), 2)


def _build_team_metrics(games: list[tuple]) -> dict[str, TeamMetrics]:
    stats: dict[str, dict[str, float]] = {}

    for home_abbr, away_abbr, _home_name, _away_name, home_score, away_score in games:
        for abbr, scored, allowed in (
            (home_abbr.lower(), home_score, away_score),
            (away_abbr.lower(), away_score, home_score),
        ):
            bucket = stats.setdefault(abbr, {"gf": 0.0, "ga": 0.0, "gp": 0.0})
            bucket["gf"] += scored
            bucket["ga"] += allowed
            bucket["gp"] += 1

    metrics: dict[str, TeamMetrics] = {}
    for abbr, bucket in stats.items():
        gp = int(bucket["gp"])
        if gp <= 0:
            continue
        metrics[abbr] = TeamMetrics(
            team=abbr,
            goals_for_pg=round(bucket["gf"] / gp, 3),
            goals_against_pg=round(bucket["ga"] / gp, 3),
            games_played=gp,
        )
    return metrics


def build_hockey_model(
    games: list[tuple],
    league: str,
) -> dict[str, Any] | None:
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    team_metrics = _build_team_metrics(games)
    if not team_metrics:
        return None

    team_game_counts = {abbr: m.games_played for abbr, m in team_metrics.items()}
    return {
        "league": league.lower(),
        "team_metrics": team_metrics,
        "team_game_counts": team_game_counts,
        "games_sampled": len(games),
    }


def predict_matchup_from_model(
    model: dict[str, Any],
    home_abbr: str,
    away_abbr: str,
) -> dict[str, float | str] | None:
    home = home_abbr.lower()
    away = away_abbr.lower()
    metrics: dict[str, TeamMetrics] = model["team_metrics"]
    if home not in metrics or away not in metrics:
        return None

    home_xg, away_xg = calculate_expected_goals(metrics[home], metrics[away])
    probs = calculate_win_probability(home_xg, away_xg)
    home_win_prob = probs.home_win * 100.0

    return {
        "home_key": home,
        "away_key": away,
        "expected_home_goals": home_xg,
        "expected_away_goals": away_xg,
        "expected_total_goals": round(home_xg + away_xg, 2),
        "home_win_probability": round(home_win_prob, 2),
        "away_win_probability": round(probs.away_win * 100.0, 2),
        "overtime_probability": round(probs.overtime * 100.0, 2),
    }


def hockey_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    league = league.lower()
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season or sparse schedule."
        )

    model = build_hockey_model(games, league)
    if not model:
        return "Could not build hockey Poisson model on available games."

    counts = model["team_game_counts"]
    home = home_abbr.lower()
    away = away_abbr.lower()
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in hockey model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the hockey model sample."
    return "Hockey-predictions model unavailable."


@lru_cache(maxsize=32)
def get_hockey_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_hockey_league(league):
        return None
    games = load_league_completed_games(league, cutoff_date)
    return build_hockey_model(games, league)


def run_hockey_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Run Poisson xG hockey model for an NHL matchup."""
    context = get_hockey_pred_context(league, cutoff_date)
    if not context:
        return None

    prediction = predict_matchup_from_model(context, home_abbr, away_abbr)
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    return {
        "algorithm": "HockeyPoisson",
        "source": "hockey-predictions",
        "home_win_probability": prediction["home_win_probability"],
        "away_win_probability": prediction["away_win_probability"],
        "expected_home_goals": prediction["expected_home_goals"],
        "expected_away_goals": prediction["expected_away_goals"],
        "expected_total_goals": prediction["expected_total_goals"],
        "overtime_probability": prediction["overtime_probability"],
        "home_games": home_games,
        "away_games": away_games,
    }
