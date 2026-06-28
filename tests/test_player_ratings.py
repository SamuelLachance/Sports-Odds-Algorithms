"""Player algo rating computation tests."""

from web.sports_db.normalize import build_player_roster_snapshot, build_player_snapshot
from web.sports_db.player_ratings import (
    enrich_team_roster_ratings,
    player_algo_rating,
    rating_tier,
    team_average_player_rating,
)


def test_nba_star_rates_above_average() -> None:
    stats = [
        {
            "name": "Per Game",
            "stats": [
                {"name": "PTS", "value": 28.0},
                {"name": "REB", "value": 8.0},
                {"name": "AST", "value": 7.0},
                {"name": "STL", "value": 1.2},
                {"name": "BLK", "value": 0.8},
            ],
        }
    ]
    rating = player_algo_rating("nba", stats, [], "G", {"experience": 8})
    assert rating >= 75
    assert rating_tier(rating) in {"elite", "good"}


def test_nba_bench_rates_near_baseline() -> None:
    stats = [
        {
            "name": "Per Game",
            "stats": [
                {"name": "PTS", "value": 6.0},
                {"name": "REB", "value": 2.0},
                {"name": "AST", "value": 1.0},
            ],
        }
    ]
    rating = player_algo_rating("nba", stats, [], "F", {"experience": 2})
    assert 35 <= rating <= 65


def test_mlb_hitter_ops_drives_rating() -> None:
    elite = player_algo_rating(
        "mlb",
        [
            {
                "name": "Hitting",
                "stats": [
                    {"name": "AVG", "value": 0.310},
                    {"name": "OPS", "value": 0.950},
                    {"name": "HR", "value": 35.0},
                    {"name": "RBI", "value": 95.0},
                ],
            }
        ],
        [],
        "OF",
        {},
    )
    bench = player_algo_rating(
        "mlb",
        [
            {
                "name": "Hitting",
                "stats": [
                    {"name": "AVG", "value": 0.220},
                    {"name": "OPS", "value": 0.620},
                    {"name": "HR", "value": 4.0},
                    {"name": "RBI", "value": 22.0},
                ],
            }
        ],
        [],
        "OF",
        {},
    )
    assert elite > bench
    assert elite >= 75


def test_mlb_pitcher_era_inverts() -> None:
    ace = player_algo_rating(
        "mlb",
        [
            {
                "name": "Pitching",
                "stats": [
                    {"name": "ERA", "value": 2.50},
                    {"name": "WHIP", "value": 0.95},
                    {"name": "SO", "value": 210.0},
                    {"name": "W", "value": 15.0},
                ],
            }
        ],
        [],
        "SP",
        {},
    )
    weak = player_algo_rating(
        "mlb",
        [
            {
                "name": "Pitching",
                "stats": [
                    {"name": "ERA", "value": 5.50},
                    {"name": "WHIP", "value": 1.55},
                    {"name": "SO", "value": 80.0},
                    {"name": "W", "value": 4.0},
                ],
            }
        ],
        [],
        "SP",
        {},
    )
    assert ace > weak


def test_roster_only_snapshot_has_rating() -> None:
    snap = build_player_roster_snapshot(
        league="nba",
        player_id="1",
        roster_row={"name": "Rookie", "position": "G", "experience": 0},
        team_abbr="bos",
    )
    assert snap["algo_rating"] is not None
    assert 38 <= snap["algo_rating"] <= 62
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_full_snapshot_includes_rating() -> None:
    snap = build_player_snapshot(
        league="nba",
        player_id="2",
        roster_row={"name": "Star", "position": "G"},
        overview={},
        stats_payload={
            "categories": [
                {
                    "displayName": "Per Game",
                    "stats": [{"displayName": "PTS", "value": 25.0}],
                }
            ]
        },
        team_abbr="bos",
    )
    assert snap["algo_rating"] is not None
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_team_average_player_rating() -> None:
    assert team_average_player_rating([80.0, 70.0, 60.0]) == 70.0
    assert team_average_player_rating([]) is None


def test_enrich_team_roster_ratings() -> None:
    roster = [
        {"id": "1", "name": "A", "position": "G", "experience": 5},
        {"id": "2", "name": "B", "position": "F", "experience": 1},
    ]
    enriched, avg = enrich_team_roster_ratings("nba", roster, {"1": 82.0})
    assert enriched[0]["algo_rating"] == 82.0
    assert enriched[1]["algo_rating"] is not None
    assert avg is not None
