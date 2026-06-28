"""Player algo rating computation tests."""

from web.sports_db.normalize import build_player_roster_snapshot, build_player_snapshot
from web.sports_db.player_ratings import (
    enrich_team_roster_ratings,
    player_algo_rating,
    rating_tier,
    team_average_player_rating,
)


def _wnba_espn_like_stats(
    *,
    ppg: float,
    rpg: float,
    apg: float,
    spg: float = 0.5,
    bpg: float = 0.3,
    fg_pct: float = 44.0,
    season_pts: float | None = None,
) -> list[dict]:
    """Mimic ESPN v3 averages + totals blocks merged into one category list."""
    season_pts = season_pts if season_pts is not None else ppg * 18
    return [
        {
            "name": "Regular Season Averages",
            "stats": [
                {"name": "Points Per Game", "short_name": "PPG", "value": ppg},
                {"name": "Rebounds Per Game", "short_name": "RPG", "value": rpg},
                {"name": "Assists Per Game", "short_name": "APG", "value": apg},
                {"name": "Steals Per Game", "short_name": "SPG", "value": spg},
                {"name": "Blocks Per Game", "short_name": "BPG", "value": bpg},
                {"name": "Field Goal Percentage", "short_name": "FG%", "value": fg_pct},
            ],
        },
        {
            "name": "Regular Season Totals",
            "stats": [
                {"name": "Points", "value": season_pts},
                {"name": "Rebounds", "value": rpg * 18},
                {"name": "Assists", "value": apg * 18},
            ],
        },
    ]


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


def test_wnba_mixed_per_game_and_totals_not_all_capped() -> None:
    """ESPN sends averages + totals; totals must not drive every player to 99."""
    star = player_algo_rating(
        "wnba",
        _wnba_espn_like_stats(ppg=18.6, rpg=11.8, apg=2.6, spg=1.1, bpg=0.6, fg_pct=48.0),
        [],
        "F",
        {"experience": 4},
    )
    bench = player_algo_rating(
        "wnba",
        _wnba_espn_like_stats(ppg=1.6, rpg=1.1, apg=0.2, spg=0.1, bpg=0.0, fg_pct=35.0),
        [],
        "G",
        {"experience": 1},
    )
    assert star != bench
    assert star > bench + 15
    assert star < 90
    assert bench < 55
    assert star != 99.0
    assert bench != 99.0


def test_wnba_roster_only_stays_in_prior_band() -> None:
    snap = build_player_roster_snapshot(
        league="wnba",
        player_id="99",
        roster_row={"name": "Rookie", "position": "G", "experience": 0},
        team_abbr="atl",
    )
    assert snap["algo_rating"] is not None
    assert 38 <= snap["algo_rating"] <= 62
