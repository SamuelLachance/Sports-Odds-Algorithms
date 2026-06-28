"""Sports database normalization tests."""

import pytest

from web.sports_db.betting_context import game_betting_sheet, league_betting_context
from web.sports_db.build import (
    FAST_DEEP_OTHER,
    _recent_games_for_team,
    _recent_games_from_live,
    deep_player_targets,
    enrich_roster_headshots,
    should_fetch_recent_games,
    stats_only_player_targets,
    team_build_targets,
)
from web.sports_db.normalize import (
    build_player_roster_snapshot,
    build_player_snapshot,
    extract_headshot_url,
    headshot_cdn_url,
    parse_news,
    parse_player_game_log,
    parse_roster,
    parse_standings,
    parse_team_statistics,
    resolve_player_headshot,
    roster_index_row,
)


def test_parse_roster_flattens_grouped_nfl_buckets() -> None:
    payload = {
        "athletes": [
            {
                "position": "offense",
                "items": [
                    {
                        "id": "1",
                        "displayName": "Tight End",
                        "position": {"abbreviation": "TE", "name": "Tight End"},
                        "jersey": "87",
                    }
                ],
            },
            {
                "position": "defense",
                "items": [
                    {
                        "id": "2",
                        "displayName": "Linebacker",
                        "position": "LB",
                        "jersey": "54",
                    }
                ],
            },
        ]
    }
    roster = parse_roster(payload)
    assert len(roster) == 2
    assert roster[0]["position"] == "TE"
    assert roster[1]["position"] == "LB"


def test_extract_headshot_url_dict_and_string() -> None:
    assert extract_headshot_url({"href": "https://cdn.example/a.png"}) == "https://cdn.example/a.png"
    assert extract_headshot_url("https://cdn.example/b.jpg") == "https://cdn.example/b.jpg"
    assert extract_headshot_url({"href": ""}) is None
    assert extract_headshot_url(None) is None


def test_parse_roster_extracts_headshot_dict_and_string() -> None:
    payload = {
        "athletes": [
            {
                "id": "1",
                "displayName": "Dict Headshot",
                "headshot": {"href": "https://example.com/dict.png", "alt": "Dict Headshot"},
            },
            {
                "id": "2",
                "displayName": "String Headshot",
                "headshot": "https://example.com/string.png",
            },
        ]
    }
    roster = parse_roster(payload)
    assert roster[0]["headshot"] == "https://example.com/dict.png"
    assert roster[1]["headshot"] == "https://example.com/string.png"


def test_resolve_player_headshot_prefers_roster_then_overview_then_cdn() -> None:
    overview = {"athlete": {"headshot": {"href": "https://example.com/overview.png"}}}
    assert (
        resolve_player_headshot(
            "nba",
            "99",
            roster_headshot="https://example.com/roster.png",
            overview=overview,
        )
        == "https://example.com/roster.png"
    )
    assert (
        resolve_player_headshot("nba", "99", overview=overview)
        == "https://example.com/overview.png"
    )
    assert resolve_player_headshot("nba", "4065648") == headshot_cdn_url("nba", "4065648")


def test_enrich_roster_headshots_fast_uses_cdn_only() -> None:
    roster = [
        {"id": "4065648", "name": "Player A"},
        {"id": "999", "name": "Player B", "headshot": "https://example.com/existing.png"},
    ]
    enrich_roster_headshots("nba", roster, fast=True)
    assert roster[0]["headshot"] == headshot_cdn_url("nba", "4065648")
    assert roster[1]["headshot"] == "https://example.com/existing.png"


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


def test_parse_team_statistics_mlb_pitching_category() -> None:
    payload = {
        "results": {
            "stats": {
                "categories": [
                    {
                        "name": "pitching",
                        "displayName": "Pitching",
                        "stats": [
                            {
                                "name": "earnedRunAvg",
                                "displayName": "Earned Run Average",
                                "abbreviation": "ERA",
                                "value": 3.42,
                                "displayValue": "3.42",
                            },
                            {
                                "name": "opponentAvg",
                                "displayName": "Opponent Batting Average",
                                "abbreviation": "OBA",
                                "value": 0.228851,
                                "displayValue": ".229",
                            },
                        ],
                    }
                ]
            }
        }
    }
    parsed = parse_team_statistics(payload)
    assert len(parsed["categories"]) == 1
    assert parsed["categories"][0]["name"] == "Pitching"
    assert len(parsed["categories"][0]["stats"]) == 2
    assert parsed["categories"][0]["stats"][1]["name"] == "Opponent Batting Average"
    assert parsed["categories"][0]["stats"][1]["display"] == ".229"


def test_parse_team_statistics_accepts_statistics_key() -> None:
    payload = {
        "results": {
            "stats": {
                "categories": [
                    {
                        "name": "batting",
                        "displayName": "Batting",
                        "statistics": [
                            {
                                "name": "hits",
                                "displayName": "Hits",
                                "value": 653.0,
                                "displayValue": "653",
                            }
                        ],
                    }
                ]
            }
        }
    }
    parsed = parse_team_statistics(payload)
    assert parsed["categories"][0]["name"] == "Batting"
    assert parsed["categories"][0]["stats"][0]["display"] == "653"


def test_parse_team_statistics_omits_empty_categories() -> None:
    payload = {
        "results": {
            "stats": {
                "categories": [
                    {"name": "pitching", "displayName": "Pitching", "stats": []},
                    {
                        "name": "batting",
                        "displayName": "Batting",
                        "stats": [
                            {"displayName": "Hits", "displayValue": "100", "value": 100.0},
                        ],
                    },
                ]
            }
        }
    }
    parsed = parse_team_statistics(payload)
    assert len(parsed["categories"]) == 1
    assert parsed["categories"][0]["name"] == "Batting"


def test_parse_team_statistics_mlb_espn_payload_sample() -> None:
    """Representative MLB team statistics payload (Yankees-style ESPN shape)."""
    payload = {
        "results": {
            "stats": {
                "id": "0",
                "name": "All stats",
                "abbreviation": "All",
                "categories": [
                    {
                        "name": "batting",
                        "displayName": "Batting",
                        "abbreviation": "batting",
                        "stats": [
                            {
                                "name": "teamGamesPlayed",
                                "displayName": "Team Games Played",
                                "abbreviation": "G",
                                "value": 162.0,
                                "displayValue": "162",
                            },
                            {
                                "name": "hits",
                                "displayName": "Hits",
                                "abbreviation": "H",
                                "value": 1305.0,
                                "displayValue": "1305",
                            },
                        ],
                    },
                    {
                        "name": "pitching",
                        "displayName": "Pitching",
                        "abbreviation": "pitching",
                        "stats": [
                            {
                                "name": "earnedRunAvg",
                                "displayName": "Earned Run Average",
                                "abbreviation": "ERA",
                                "value": 3.42,
                                "displayValue": "3.42",
                            },
                            {
                                "name": "opponentAvg",
                                "displayName": "Opponent Batting Average",
                                "abbreviation": "OBA",
                                "value": 0.228851,
                                "displayValue": ".229",
                            },
                        ],
                    },
                    {
                        "name": "fielding",
                        "displayName": "Fielding",
                        "abbreviation": "fielding",
                        "stats": [
                            {
                                "name": "totalChances",
                                "displayName": "Total Chances",
                                "abbreviation": "TC",
                                "value": 5832.0,
                                "displayValue": "5832",
                            }
                        ],
                    },
                ],
            }
        }
    }
    parsed = parse_team_statistics(payload)
    assert [cat["name"] for cat in parsed["categories"]] == ["Batting", "Pitching", "Fielding"]
    pitching = parsed["categories"][1]
    assert pitching["stats"][0]["display"] == "3.42"
    assert pitching["stats"][1]["name"] == "Opponent Batting Average"


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
    assert rows[0]["opponent"] == "NYK"
    assert rows[0]["opponent_abbr"] == "nyk"
    assert rows[0]["location"] == "home"
    assert rows[0]["result"] == "W"
    assert rows[0]["score"] == "110-99"
    assert rows[1]["opponent"] == "BOS"
    assert rows[1]["location"] == "away"


def test_parse_player_game_log_wnba_opponent_object() -> None:
    overview = {
        "gameLog": {
            "events": {
                "1": {
                    "gameDate": "2026-06-24T23:30:00.000+00:00",
                    "atVs": "@",
                    "score": "78-76",
                    "gameResult": "W",
                    "opponent": {
                        "displayName": "Washington Mystics",
                        "abbreviation": "WSH",
                    },
                }
            }
        }
    }
    rows = parse_player_game_log(overview, limit=5)
    assert len(rows) == 1
    assert rows[0]["opponent"] == "Washington Mystics"
    assert rows[0]["opponent_abbr"] == "wsh"
    assert rows[0]["location"] == "away"
    assert rows[0]["result"] == "W"
    assert rows[0]["score"] == "78-76"


def test_parse_wnba_athlete_stats_categories() -> None:
    from web.sports_db.normalize import _parse_stat_categories, parse_player_overview_stats

    stats_payload = {
        "categories": [
            {
                "displayName": "Regular Season Averages",
                "labels": ["GP", "MIN", "PTS"],
                "displayNames": ["Games Played", "Minutes Per Game", "Points Per Game"],
                "statistics": [
                    {
                        "season": {"year": 2025},
                        "stats": ["10", "20.0", "12.0"],
                    },
                    {
                        "season": {"year": 2026},
                        "stats": ["18", "29.4", "17.5"],
                    },
                ],
            }
        ]
    }
    rows = _parse_stat_categories(stats_payload)
    assert rows[0]["name"] == "Regular Season Averages"
    assert rows[0]["stats"][0]["display"] == "18"
    assert rows[0]["stats"][2]["name"] == "Points Per Game"

    overview = {
        "statistics": {
            "labels": ["GP", "PTS"],
            "displayNames": ["Games Played", "Points Per Game"],
            "splits": [{"displayName": "Regular Season", "stats": ["18", "17.5"]}],
        }
    }
    overview_rows = parse_player_overview_stats(overview)
    assert overview_rows[0]["stats"][1]["name"] == "Points Per Game"
    assert overview_rows[0]["stats"][1]["display"] == "17.5"


def test_stats_only_player_targets_fast_mode() -> None:
    roster = [{"id": str(i)} for i in range(10)]
    deep = deep_player_targets(roster, "bos", "nba", {"bos"}, fast=True)
    stats = stats_only_player_targets(roster, deep, fast=True)
    assert deep | stats == {str(i) for i in range(10)}
    assert not (deep & stats)
    assert stats_only_player_targets(roster, deep, fast=False) == set()


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


@pytest.mark.parametrize(
    ("league", "fast", "expected_all"),
    [
        ("nba", True, True),
        ("cbb", True, False),
        ("nba", False, True),
    ],
)
def test_team_build_targets(league, fast, expected_all) -> None:
    team_ids = {"bos": "1", "nyk": "2", "lal": "3"}
    slate_keys = {"bos", "nyk"}
    standings = {"teams": [{"abbr": "lal"}, {"abbr": "bos"}]}
    targets = team_build_targets(league, team_ids, slate_keys, standings, fast=fast)
    if expected_all:
        assert targets == set(team_ids.keys())
    else:
        assert "bos" in targets
        assert "lal" in targets


def test_deep_player_targets_fast_slate_vs_other() -> None:
    roster = [{"id": str(i)} for i in range(20)]
    slate = deep_player_targets(roster, "bos", "nba", {"bos"}, fast=True)
    other = deep_player_targets(roster, "nyk", "nba", {"bos"}, fast=True)
    assert len(slate) == 20
    assert len(other) == FAST_DEEP_OTHER


def test_deep_player_targets_skips_idle_league() -> None:
    roster = [{"id": str(i)} for i in range(20)]
    assert deep_player_targets(roster, "nyk", "nba", set(), fast=True, games_today=0) == set()
    assert len(deep_player_targets(roster, "bos", "nba", {"bos"}, fast=True, games_today=0)) == 20


def test_should_fetch_recent_games_fast_mode() -> None:
    assert should_fetch_recent_games(fast=True, on_slate=False) is False
    assert should_fetch_recent_games(fast=True, on_slate=True) is True
    assert should_fetch_recent_games(fast=False, on_slate=False) is True


def test_recent_games_for_team_reuses_cache_when_skipping_fetch() -> None:
    cached = {"recent_games": [{"date": "6-1-2026", "opponent": "NYK", "result": "W"}]}
    games = _recent_games_for_team(
        "nba",
        "bos",
        "1",
        "6-27-2026",
        {},
        fast=True,
        on_slate=False,
        cached_team=cached,
    )
    assert games == cached["recent_games"]


def test_build_player_roster_snapshot_lightweight() -> None:
    roster = {
        "id": "99",
        "name": "Roster Only",
        "position": "F",
        "jersey": "11",
        "headshot": "https://example.com/p.jpg",
    }
    snap = build_player_roster_snapshot(
        league="nba",
        player_id="99",
        roster_row=roster,
        team_abbr="bos",
    )
    assert snap["profile_depth"] == "roster"
    assert snap["player"]["name"] == "Roster Only"
    assert snap["season_stats"] == []


def test_roster_index_row_includes_headshot() -> None:
    row = roster_index_row(
        {"id": "1", "name": "A", "position": "G", "headshot": "https://x/y.jpg"}
    )
    assert row["headshot"] == "https://x/y.jpg"


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


def test_recent_games_from_live_extracts_opponents(monkeypatch) -> None:
    sample_entry = [
        {
            "dates": ["6-20-2026", "6-25-2026"],
            "other_team": ["nyy", "tor"],
            "home_away": ["home", "away"],
            "game_scores": [[5, 3], [2, 4]],
        }
    ]

    monkeypatch.setattr(
        "web.sports_db.build.load_live_team_data",
        lambda *args, **kwargs: sample_entry,
    )
    monkeypatch.setattr(
        "web.sports_db.build.resolve_team",
        lambda *args, **kwargs: ["bos", "boston-red-sox"],
    )

    games = _recent_games_from_live(
        "mlb",
        "bos",
        "2",
        "6-27-2026",
        {"nyy": "New York Yankees", "tor": "Toronto Blue Jays"},
    )
    assert len(games) == 2
    assert games[0]["opponent_abbr"] == "tor"
    assert games[0]["opponent"] == "Toronto Blue Jays"
    assert games[0]["result"] == "L"
    assert games[0]["location"] == "away"
    assert games[0]["score"] == "2-4"
    assert games[1]["opponent_abbr"] == "nyy"
    assert games[1]["opponent"] == "New York Yankees"
    assert games[1]["result"] == "W"
