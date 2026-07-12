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
    _event_date_iso,
    _load_league_game_map,
    _parse_event_to_game,
    get_league_power_context,
    load_league_completed_games,
    load_league_dated_games_for_backtest,
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
    _event_id, _event_date, game = parsed
    assert game[0] == "bos"
    assert game[1] == "ny"
    assert game[4] == 110
    assert game[5] == 99


def test_parse_event_to_game_skips_missing_scores_instead_of_inventing_zero() -> None:
    """Completed events with null scores must not enter power history as 0–0."""
    from web.live_data import _score_value

    cutoff = datetime(2026, 6, 12, tzinfo=timezone.utc)
    event = _completed_event(
        "2",
        home_abbr="BOS",
        away_abbr="NY",
        home_score=110,
        away_score=99,
    )
    event["competitions"][0]["competitors"][0]["score"] = {"value": None}
    event["competitions"][0]["competitors"][1]["score"] = None

    assert _score_value(event["competitions"][0]["competitors"][0]) is None
    assert _score_value(event["competitions"][0]["competitors"][1]) is None
    assert _parse_event_to_game("nba", event, cutoff) is None

    # Genuine 0–0 (or 0–N) still parses.
    zero = _completed_event(
        "3",
        home_abbr="BOS",
        away_abbr="NY",
        home_score=0,
        away_score=0,
    )
    parsed = _parse_event_to_game("nba", zero, cutoff)
    assert parsed is not None
    assert parsed[2][4] == 0
    assert parsed[2][5] == 0

    # ESPN sometimes sends numeric strings / floats.
    floaty = _completed_event(
        "4",
        home_abbr="BOS",
        away_abbr="NY",
        home_score=110,
        away_score=99,
    )
    floaty["competitions"][0]["competitors"][0]["score"] = {"value": "110.0"}
    floaty["competitions"][0]["competitors"][1]["score"] = 99.0
    parsed_f = _parse_event_to_game("nba", floaty, cutoff)
    assert parsed_f is not None
    assert parsed_f[2][4] == 110
    assert parsed_f[2][5] == 99


def test_load_league_dated_games_for_backtest_preserves_iso_dates() -> None:
    sample_map = {
        "1": ("2024-11-01T19:00Z", ("bos", "ny", "Boston", "New York", 110, 100)),
        "2": ("2024-11-03T19:00Z", ("lal", "gs", "Lakers", "Warriors", 105, 108)),
    }
    with patch("web.season_games._load_league_game_map", return_value=sample_map):
        dated = load_league_dated_games_for_backtest("nba", "11-10-2024")
    assert dated == [
        ("2024-11-01", sample_map["1"][1]),
        ("2024-11-03", sample_map["2"][1]),
    ]


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


def test_event_date_iso_uses_toronto_not_utc_for_odds_keys() -> None:
    """Late-ET tips must key closing odds on the Toronto calendar day."""
    from web.closing_odds_db import _lookup_us_odds_row
    from web.espn_client import iso_to_project_date

    # Jul 1 10pm ET → Toronto 2025-07-01, UTC calendar date 2025-07-02.
    iso = "2025-07-02T02:00:00Z"
    assert iso_to_project_date(iso) == "7-1-2025"
    assert _event_date_iso(iso) == "2025-07-01"
    assert _event_date_iso("2025-07-01") == "2025-07-01"

    index = {
        ("2025-07-01", "bos", "nyy"): {"home_close_ml": -130, "tag": "g1"},
        ("2025-07-02", "bos", "nyy"): {"home_close_ml": -150, "tag": "g2"},
    }
    hit = _lookup_us_odds_row(index, _event_date_iso(iso), "bos", "nyy")
    assert hit is not None
    assert hit["tag"] == "g1"


if __name__ == "__main__":
    test_parse_event_to_game_normalizes_abbreviations()
    test_get_league_power_context_requires_min_games()
    test_get_league_power_context_builds_ratings_from_sample_games()
    test_collect_from_scoreboards_deduplicates_events()
    test_event_date_iso_uses_toronto_not_utc_for_odds_keys()
    print("test_season_games.py: all tests passed")
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
    test_collect_from_scoreboards_deduplicates_events()
    print("test_season_games.py: all tests passed")
