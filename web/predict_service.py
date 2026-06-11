"""Non-interactive prediction service for the web UI."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from web.league_profiles import (  # noqa: E402
    SUPPORTED_LEAGUES,
    get_algo_league,
    list_leagues_metadata,
)
from web.team_registry import fetch_espn_teams  # noqa: E402
ALGO_VERSIONS = ("Algo_V1", "Algo_V2")

FACTOR_LABELS = {
    "record_points": "Seasonal record",
    "home_away_points": "Home / away split",
    "home_away_10_games_points": "Home / away (last 10)",
    "last_10_games_points": "Last 10 games",
    "avg_points": "Average scoring margin",
    "avg_points_10_games": "Scoring margin (last 10)",
    "win_streak_home_away": "Home/away win streak",
}


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def get_leagues() -> list[dict[str, str]]:
    return list_leagues_metadata()


def get_teams(league: str) -> list[dict[str, str]]:
    league = league.lower()
    if league not in SUPPORTED_LEAGUES:
        raise ValueError(f"Unsupported league: {league}")

    teams = fetch_espn_teams(league)
    if teams:
        return teams

    _ensure_project_root()
    from universal_functions import Universal_Functions

    universal = Universal_Functions(league)
    return [
        {"abbr": team[0], "slug": team[1], "label": team[1].replace("-", " ").title()}
        for team in universal.load_league_teams()
    ]


def get_seasons(league: str) -> list[str]:
    from datetime import date

    league = league.lower()
    data_dir = PROJECT_ROOT / league / "team_data"
    if data_dir.is_dir():
        seasons = sorted(
            folder.name
            for folder in data_dir.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        )
        if seasons:
            return seasons

    year = date.today().year
    return [str(year), str(year - 1)]


def _find_team(league: str, team_slug: str) -> list[str]:
    for team in get_teams(league):
        if team["slug"] == team_slug:
            return [team["abbr"], team["slug"]]
    raise ValueError(f"Unknown team: {team_slug}")


def _american_odds(win_probability: float) -> tuple[float, float]:
    probability = abs(win_probability)
    if probability >= 100:
        probability = 99.9
    if probability <= 0:
        probability = 50.0

    favorite = (100 / (100 - probability) - 1) * 100
    underdog = favorite
    return round(favorite, 2), round(underdog, 2)


def predict_match(
    league: str,
    away_slug: str,
    home_slug: str,
    date: str,
    season_year: str,
    algo_version: str = "Algo_V2",
) -> dict[str, Any]:
    """Run the original algorithm against bundled historical CSV data."""
    _ensure_project_root()

    from algo import Algo
    from odds_calculator import Odds_Calculator
    from universal_functions import Universal_Functions

    league = league.lower()
    algo_version = algo_version or "Algo_V2"

    if league not in SUPPORTED_LEAGUES:
        raise ValueError(f"Unsupported league: {league}")
    if algo_version not in ALGO_VERSIONS:
        raise ValueError(f"Unsupported algorithm version: {algo_version}")
    if away_slug == home_slug:
        raise ValueError("Away and home teams must be different.")

    away = _find_team(league, away_slug)
    home = _find_team(league, home_slug)

    algo_league = get_algo_league(league)
    universal = Universal_Functions(league)
    odds_calculator = Odds_Calculator(algo_league)
    algo = Algo(algo_league)

    data_away = universal.load_data(away, date, season_year)
    data_home = universal.load_data(home, date, season_year)

    if not data_away or not data_home:
        raise ValueError(
            "No game data found for that date/season. "
            "Use bundled seasons (NBA/NHL through 2017, MLB through 2016)."
        )

    with redirect_stdout(io.StringIO()):
        returned_away = odds_calculator.analyze2(away, home, data_away, "away")
        returned_home = odds_calculator.analyze2(home, away, data_home, "home")

        if algo_version == "Algo_V1":
            algo_data = algo.calculate(date, returned_away, returned_home)
        else:
            algo_data = algo.calculate_V2(date, returned_away, returned_home)

    total = float(algo_data["total"])
    if algo_version == "Algo_V1":
        winning_odds = odds_calculator.get_odds(total)
    else:
        winning_odds = abs(total)

    favorite_side = "home" if total < 0 else "away"
    favorite_team = home if favorite_side == "home" else away
    underdog_team = away if favorite_side == "home" else home
    favorite_odds, underdog_odds = _american_odds(winning_odds)

    factors = []
    for key, label in FACTOR_LABELS.items():
        if key not in algo_data:
            continue
        value = float(algo_data[key])
        factors.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "favors": "away" if value > 0 else "home" if value < 0 else "neutral",
            }
        )

    return {
        "league": league,
        "algorithm": algo_version,
        "date": date,
        "season_year": season_year,
        "matchup": {
            "away": {"abbr": away[0], "slug": away[1]},
            "home": {"abbr": home[0], "slug": home[1]},
        },
        "prediction": {
            "favorite_side": favorite_side,
            "favorite_team": favorite_team[1],
            "underdog_team": underdog_team[1],
            "win_probability": round(winning_odds, 2),
            "total_score": total,
            "american_odds": {
                "favorite": f"-{favorite_odds}",
                "underdog": f"+{underdog_odds}",
            },
        },
        "factors": factors,
        "notes": [
            "Negative totals favor the home team; positive totals favor the away team.",
            "Predictions use bundled historical CSV data — live ESPN scraping is optional via the CLI.",
        ],
    }
