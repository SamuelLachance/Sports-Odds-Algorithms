"""Season year resolution for live model data."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.espn_client import (  # noqa: E402
    current_season_year,
    guess_season_years,
    prior_season_year,
)
from web.league_profiles import INTERNATIONAL_SOCCER_LEAGUES, MIN_GAMES_FOR_MODEL  # noqa: E402
from web.live_data import load_live_team_data, resolve_team  # noqa: E402


def test_calendar_year_leagues_use_cutoff_year() -> None:
    cutoff = date(2026, 6, 11)
    assert current_season_year("mlb", cutoff) == 2026
    assert prior_season_year("mlb", cutoff) == 2025
    assert guess_season_years("mlb", cutoff) == [2026, 2025]


def test_split_year_leagues_before_october() -> None:
    cutoff = date(2026, 6, 11)
    assert current_season_year("nba", cutoff) == 2026
    assert current_season_year("nhl", cutoff) == 2026
    assert current_season_year("wnba", cutoff) == 2026
    assert current_season_year("nfl", cutoff) == 2026


def test_split_year_leagues_from_october() -> None:
    cutoff = date(2026, 11, 1)
    assert current_season_year("nba", cutoff) == 2027
    assert prior_season_year("nba", cutoff) == 2026


def test_college_season_rollover() -> None:
    assert current_season_year("cbb", date(2026, 7, 1)) == 2026
    assert current_season_year("cbb", date(2026, 9, 1)) == 2027


def test_min_games_threshold() -> None:
    assert MIN_GAMES_FOR_MODEL == 10


def test_international_soccer_uses_calendar_year() -> None:
    cutoff = date(2026, 6, 11)
    for league in INTERNATIONAL_SOCCER_LEAGUES:
        assert current_season_year(league, cutoff) == 2026
        assert prior_season_year(league, cutoff) == 2025
        assert guess_season_years(league, cutoff) == [2026, 2025]


def test_worldcup_supplements_friendlies_history() -> None:
    team = resolve_team("worldcup", "CAN", "Canada")
    assert team is not None
    data = load_live_team_data("worldcup", team, "206", "6-12-2026")
    assert data
    assert len(data[0]["dates"]) >= MIN_GAMES_FOR_MODEL
    assert any(
        season.startswith("fifa_friendlies:")
        for season in data[0]["seasons_used"]
    )


if __name__ == "__main__":
    test_calendar_year_leagues_use_cutoff_year()
    test_split_year_leagues_before_october()
    test_split_year_leagues_from_october()
    test_college_season_rollover()
    test_min_games_threshold()
    test_international_soccer_uses_calendar_year()
    test_worldcup_supplements_friendlies_history()
    print("test_season_selection.py: all tests passed")
