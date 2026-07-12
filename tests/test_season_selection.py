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
from web.league_profiles import MIN_GAMES_FOR_MODEL  # noqa: E402


def test_calendar_year_leagues_use_cutoff_year() -> None:
    cutoff = date(2026, 6, 11)
    assert current_season_year("mlb", cutoff) == 2026
    assert prior_season_year("mlb", cutoff) == 2025
    assert guess_season_years("mlb", cutoff) == [2026, 2025]


def test_split_year_leagues_before_october() -> None:
    cutoff = date(2026, 6, 11)
    assert current_season_year("nba", cutoff) == 2026
    assert current_season_year("nhl", cutoff) == 2026
    # WNBA is calendar-year (May–Oct); NFL uses fall start year.
    assert current_season_year("wnba", cutoff) == 2026
    assert current_season_year("nfl", cutoff) == 2026


def test_split_year_leagues_from_october() -> None:
    cutoff = date(2026, 11, 1)
    assert current_season_year("nba", cutoff) == 2027
    assert prior_season_year("nba", cutoff) == 2026
    # NFL November is still the 2026 season (not NBA's ending-year +1).
    assert current_season_year("nfl", cutoff) == 2026
    # WNBA Finals window stays on the calendar year.
    assert current_season_year("wnba", date(2025, 10, 12)) == 2025


def test_nfl_playoffs_keep_fall_season_year() -> None:
    assert current_season_year("nfl", date(2026, 2, 8)) == 2025
    assert current_season_year("nfl", date(2026, 3, 1)) == 2026


def test_cfb_bowls_keep_fall_season_year() -> None:
    # CFB must not share CBB's Aug→ending-year rollover.
    assert current_season_year("cfb", date(2025, 1, 15)) == 2024
    assert current_season_year("cfb", date(2025, 9, 1)) == 2025
    assert current_season_year("cbb", date(2025, 9, 1)) == 2026


def test_college_season_rollover() -> None:
    assert current_season_year("cbb", date(2026, 7, 1)) == 2026
    assert current_season_year("cbb", date(2026, 9, 1)) == 2027


def test_college_hockey_uses_split_year() -> None:
    assert current_season_year("ncaah", date(2026, 6, 11)) == 2026
    assert current_season_year("ncaah", date(2026, 11, 1)) == 2027


def test_ncaa_baseball_uses_calendar_year() -> None:
    assert current_season_year("ncaabb", date(2026, 3, 15)) == 2026
    assert prior_season_year("ncaabb", date(2026, 3, 15)) == 2025


def test_european_club_soccer_uses_season_start_year() -> None:
    """ESPN ?season= for EPL/etc is Aug start year — not calendar year after Jan 1."""
    midwinter = date(2026, 1, 15)
    for league in ("epl", "laliga", "bundesliga", "seriea", "ligue1", "ucl"):
        assert current_season_year(league, midwinter) == 2025
        assert prior_season_year(league, midwinter) == 2024
        assert guess_season_years(league, midwinter) == [2025, 2024]
    # After July kickoff, start year matches calendar year.
    assert current_season_year("epl", date(2025, 8, 15)) == 2025
    # MLS / World Cup stay calendar-year.
    assert current_season_year("mls", midwinter) == 2026
    assert current_season_year("worldcup", midwinter) == 2026


def test_min_games_threshold() -> None:
    assert MIN_GAMES_FOR_MODEL == 10


if __name__ == "__main__":
    test_calendar_year_leagues_use_cutoff_year()
    test_split_year_leagues_before_october()
    test_split_year_leagues_from_october()
    test_nfl_playoffs_keep_fall_season_year()
    test_cfb_bowls_keep_fall_season_year()
    test_college_season_rollover()
    test_min_games_threshold()
    print("test_season_selection.py: all tests passed")
