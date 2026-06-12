"""Season game collection and power context tests."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import MIN_GAMES_FOR_POWER  # noqa: E402
from web.season_games import (  # noqa: E402
    _collect_from_scoreboards,
    _parse_event_to_game,
    get_league_power_context,
    load_league_completed_games,
)


def _completed_event(
    event_id: str,
    *,
    home_abbr: str,
    away_abbr: str,
    home_score: int,
    away_score: int,
) -> dict:
    return {
        "id": event_id,
        "date": "2026-06-10T20:00Z",
        "competitions": [
            {
                "status": {"type": {"completed": True}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": {"value": home_score},
                        "team": {
                            "abbreviation": home_abbr,
                            "displayName": f"Team {home_abbr}",
                        },
                    },
                    {
                        "homeAway": "away",
                        "score": {"value": away_score},
                        "team": {
                            "abbreviation": away_abbr,
                            "displayName": f"Team {away_abbr}",
                        },
                    },
                ],
            }
        ],
    }


def test_parse_event_to_game_normalizes_abbreviations() -> None:
    cutoff = datetime(2026, 6, 12, tzinfo=timezone.utc)
    parsed = _parse_event_to_game(
        "nba",
        _completed_event(
            "1",
            home_abbr="BOS",
            away_abbr="NY",
            home_score=110,
            away_score=99,
        ),
        cutoff,
    )
    assert parsed is not None
    _event_id, game = parsed
    assert game[0] == "bos"
    assert game[1] == "ny"
    assert game[4] == 110
    assert game[5] == 99


def test_get_league_power_context_requires_min_games() -> None:
    get_league_power_context.cache_clear()
    with patch("web.season_games.load_league_completed_games", return_value=[]):
        assert get_league_power_context("nba", "6-12-2026") is None


def test_get_league_power_context_builds_ratings_from_sample_games() -> None:
    get_league_power_context.cache_clear()
    sample = [
        ("a", "b", "Team A", "Team B", 80, 70),
        ("a", "c", "Team A", "Team C", 75, 72),
        ("b", "c", "Team B", "Team C", 68, 70),
        ("c", "a", "Team C", "Team A", 65, 78),
        ("b", "a", "Team B", "Team A", 60, 82),
    ]
    with patch("web.season_games.load_league_completed_games", return_value=sample):
        context = get_league_power_context("nba", "6-12-2026")
    assert context is not None
    teams, games, param = context
    assert len(games) == len(sample)
    assert len(teams) == 3
    assert param is not None


def test_worldcup_supplements_friendlies_for_power_context() -> None:
    get_league_power_context.cache_clear()
    cutoff = "6-12-2026"
    tournament_only = [
        ("usa", "par", "United States", "Paraguay", 2, 1),
        ("can", "bih", "Canada", "Bosnia-Herzegovina", 1, 0),
    ]
    friendly_games = [
        ("usa", "mex", "United States", "Mexico", 2, 0),
        ("usa", "crc", "United States", "Costa Rica", 3, 1),
        ("par", "chi", "Paraguay", "Chile", 1, 1),
        ("par", "uru", "Paraguay", "Uruguay", 2, 0),
        ("can", "jam", "Canada", "Jamaica", 2, 1),
    ]
    merged = tournament_only + friendly_games

    with patch("web.season_games._collect_from_team_schedules", return_value={}):
        with patch("web.season_games._load_friendlies_supplement", return_value=friendly_games):
            with patch(
                "web.season_games._collect_from_scoreboards",
                side_effect=lambda league, cutoff, lookback: {},
            ):
                games = load_league_completed_games("worldcup", cutoff)
    assert len(games) >= MIN_GAMES_FOR_POWER

    get_league_power_context.cache_clear()
    with patch("web.season_games.load_league_completed_games", return_value=merged):
        context = get_league_power_context("worldcup", cutoff)
    assert context is not None


def test_collect_from_scoreboards_deduplicates_events() -> None:
    cutoff = datetime(2026, 6, 12, tzinfo=timezone.utc)
    event = _completed_event(
        "evt-1",
        home_abbr="DUKE",
        away_abbr="UNC",
        home_score=70,
        away_score=68,
    )

    with patch(
        "web.season_games._fetch_scoreboard_payload",
        side_effect=[{"events": [event]}, {"events": [event]}, {"events": []}],
    ):
        games = _collect_from_scoreboards("cbb", cutoff, lookback_days=3)

    assert len(games) == 1


if __name__ == "__main__":
    test_parse_event_to_game_normalizes_abbreviations()
    test_get_league_power_context_requires_min_games()
    test_get_league_power_context_builds_ratings_from_sample_games()
    test_worldcup_supplements_friendlies_for_power_context()
    test_collect_from_scoreboards_deduplicates_events()
    print("test_season_games.py: all tests passed")
