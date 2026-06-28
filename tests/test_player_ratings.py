"""Player rating integration tests (external game DB, not ESPN stats)."""

from web.sports_db.external_ratings import clear_rating_cache
from web.sports_db.normalize import build_player_roster_snapshot, build_player_snapshot
from web.sports_db.player_ratings import (
    enrich_team_roster_ratings,
    player_algo_rating,
    rating_tier,
    team_average_player_rating,
)


def test_nba_star_from_2k_not_stats() -> None:
    clear_rating_cache()
    rating = player_algo_rating(
        "nba",
        [{"name": "Per Game", "stats": [{"name": "PTS", "value": 99.0}]}],
        [],
        "G",
        {"name": "Jayson Tatum", "experience": 0},
        team_abbr="bos",
    )
    assert rating == 96.0
    assert rating_tier(rating) == "elite"


def test_stats_do_not_inflate_unknown_player() -> None:
    clear_rating_cache()
    rating = player_algo_rating(
        "nba",
        [{"name": "Per Game", "stats": [{"name": "PTS", "value": 35.0}]}],
        [],
        "G",
        {"name": "Nobody On Roster", "experience": 0},
        team_abbr="bos",
    )
    assert 42 <= rating <= 58


def test_wnba_dream_player_snapshot() -> None:
    clear_rating_cache()
    snap = build_player_roster_snapshot(
        league="wnba",
        player_id="1",
        roster_row={"name": "Rhyne Howard", "position": "G"},
        team_abbr="atl",
    )
    assert snap["algo_rating"] == 88.0
    assert snap["rating_source"] == "2k"
    assert snap["rating_year"] == 2026
    assert snap["player"]["algo_rating"] == 88.0


def test_roster_only_unknown_gets_prior() -> None:
    clear_rating_cache()
    snap = build_player_roster_snapshot(
        league="nba",
        player_id="1",
        roster_row={"name": "Rookie Unknown", "position": "G", "experience": 0},
        team_abbr="bos",
    )
    assert snap["algo_rating"] is not None
    assert 42 <= snap["algo_rating"] <= 58
    assert snap["rating_source"] == "prior"


def test_full_snapshot_uses_external_not_stats() -> None:
    clear_rating_cache()
    snap = build_player_snapshot(
        league="nba",
        player_id="2",
        roster_row={"name": "Jaylen Brown", "position": "G"},
        overview={},
        stats_payload={
            "categories": [
                {
                    "displayName": "Per Game",
                    "stats": [{"displayName": "PTS", "value": 5.0}],
                }
            ]
        },
        team_abbr="bos",
    )
    assert snap["algo_rating"] == 89.0
    assert snap["rating_source"] == "2k"
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_team_average_player_rating() -> None:
    assert team_average_player_rating([80.0, 70.0, 60.0]) == 70.0
    assert team_average_player_rating([]) is None


def test_enrich_team_roster_ratings() -> None:
    clear_rating_cache()
    roster = [
        {"id": "1", "name": "Jayson Tatum", "position": "G", "experience": 5},
        {"id": "2", "name": "Unknown", "position": "F", "experience": 1},
    ]
    enriched, avg = enrich_team_roster_ratings(
        "nba",
        roster,
        {"1": {"algo_rating": 96.0, "rating_source": "2k", "rating_year": 2026}},
        team_abbr="bos",
    )
    assert enriched[0]["algo_rating"] == 96.0
    assert enriched[0]["rating_source"] == "2k"
    assert enriched[1]["rating_source"] == "prior"
    assert avg is not None
