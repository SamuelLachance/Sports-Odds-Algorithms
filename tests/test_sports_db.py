"""Sports database normalization tests."""

from web.sports_db.betting_context import game_betting_sheet, league_betting_context
from web.sports_db.normalize import (
    build_player_snapshot,
    parse_news,
    parse_player_game_log,
    parse_standings,
    parse_team_statistics,
)


def test_parse_standings_groups() -> None:
    payload = {
        "children": [
            {
                "name": "East",
                "standings": {
                    "entries": [
                        {
                            "team": {"abbreviation": "BOS", "displayName": "Boston Celtics"},
                            "stats": [
                                {"name": "wins", "value": 50.0, "displayValue": "50"},
                                {"name": "losses", "value": 20.0, "displayValue": "20"},
                                {"name": "winPercent", "value": 0.714, "displayValue": ".714"},
                                {"name": "rank", "value": 2.0, "displayValue": "2"},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    parsed = parse_standings(payload)
    assert len(parsed["groups"]) == 1
    assert parsed["teams"][0]["abbr"] == "bos"
    assert parsed["teams"][0]["wins"] == 50.0


def test_parse_news_headlines() -> None:
    payload = {
        "articles": [
            {
                "id": "1",
                "headline": "Test headline",
                "description": "Body",
                "published": "2026-06-27T12:00:00Z",
                "links": {"web": {"href": "https://example.com"}},
            }
        ]
    }
    rows = parse_news(payload)
    assert len(rows) == 1
    assert rows[0]["headline"] == "Test headline"


def test_parse_team_statistics_categories() -> None:
    payload = {
        "results": {
            "stats": {
                "categories": [
                    {
                        "name": "general",
                        "displayName": "General",
                        "stats": [
                            {"name": "avgPoints", "displayName": "Points", "value": 110.0, "displayValue": "110.0"}
                        ],
                    }
                ]
            }
        }
    }
    parsed = parse_team_statistics(payload)
    assert parsed["categories"][0]["name"] == "General"
    assert parsed["categories"][0]["stats"][0]["display"] == "110.0"


def test_parse_player_game_log_sorts_newest_first() -> None:
    overview = {
        "gameLog": {
            "events": {
                "1": {"gameDate": "2026-01-01", "atVs": "@ BOS", "score": "L 98-102"},
                "2": {"gameDate": "2026-06-20", "atVs": "vs NYK", "score": "W 110-99"},
            }
        }
    }
    rows = parse_player_game_log(overview, limit=5)
    assert len(rows) == 2
    assert rows[0]["date"] == "2026-06-20"
    assert rows[0]["opponent"] == "vs NYK"


def test_build_player_snapshot_merges_roster_and_stats() -> None:
    roster = {
        "id": "123",
        "name": "Test Player",
        "position": "G",
        "jersey": "7",
        "status": "Active",
    }
    stats_payload = {
        "categories": [
            {
                "displayName": "Per Game",
                "stats": [
                    {"displayName": "PTS", "value": 20.0, "displayValue": "20.0"},
                ],
            }
        ]
    }
    snapshot = build_player_snapshot(
        league="nba",
        player_id="123",
        roster_row=roster,
        overview={"gameLog": {"events": {}}},
        stats_payload=stats_payload,
        team_abbr="bos",
    )
    assert snapshot["player"]["name"] == "Test Player"
    assert snapshot["team_abbr"] == "bos"
    assert snapshot["season_stats"][0]["stats"][0]["display"] == "20.0"


def test_parse_stat_categories_unwraps_espn_results() -> None:
    from web.sports_db.normalize import _parse_stat_categories

    nested = {
        "results": {
            "stats": {
                "categories": [
                    {
                        "displayName": "Per Game",
                        "stats": [{"displayName": "REB", "displayValue": "8.1", "value": 8.1}],
                    }
                ]
            }
        }
    }
    rows = _parse_stat_categories(nested)
    assert rows[0]["name"] == "Per Game"
    assert rows[0]["stats"][0]["display"] == "8.1"


def test_league_betting_context_counts_games() -> None:
    slate = {
        "games": [
            {
                "league": "nba",
                "event_id": "1",
                "matchup": {"home": {"abbr": "bos"}, "away": {"abbr": "nyk"}},
                "model": {"win_probability": 62, "model_agreement": {"required": True, "agreed": True}},
                "market": {"home_moneyline": -150},
                "top_pick": {"team_name": "Boston", "edge": 45},
            },
            {
                "league": "nhl",
                "event_id": "2",
                "matchup": {"home": {"abbr": "mtl"}, "away": {"abbr": "tor"}},
            },
        ]
    }
    ctx = league_betting_context(slate, "nba")
    assert ctx["game_count"] == 1
    assert ctx["official_pick_count"] == 1
    sheet = game_betting_sheet(slate["games"][0])
    assert sheet["model"]["win_probability"] == 62
    assert sheet["top_pick"]["edge"] == 45
