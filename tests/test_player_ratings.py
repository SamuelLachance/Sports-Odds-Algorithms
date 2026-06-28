"""Player algo rating computation tests (external publisher OVR only)."""

from web.sports_db.external_ratings import RATING_PRIOR_SOURCE, clear_rating_cache
from web.sports_db.normalize import build_player_roster_snapshot, build_player_snapshot
from web.sports_db.player_ratings import (
    enrich_team_roster_ratings,
    player_algo_rating,
    rating_source_label,
    rating_tier,
    resolve_player_rating,
    team_average_player_rating,
)


def test_external_match_used_for_known_nba_player() -> None:
    clear_rating_cache()
    snap = resolve_player_rating(
        "nba",
        player_name="Jayson Tatum",
        team_abbr="bos",
        cutoff_date="4-16-2017",
    )
    assert snap["algo_rating"] == 96.0
    assert snap["rating_source"] == "2k"
    assert snap["rating_year"] == 2026
    assert snap.get("rating_layer") is None


def test_unknown_player_has_no_algo_rating() -> None:
    clear_rating_cache()
    snap = resolve_player_rating(
        "nba",
        player_name="Unknown Rookie",
        team_abbr="bos",
        position="G",
        roster_meta={"experience": 12},
        cutoff_date="4-16-2017",
    )
    assert snap["algo_rating"] is None
    assert snap["rating_source"] == RATING_PRIOR_SOURCE


def test_roster_only_snapshot_has_external_rating() -> None:
    clear_rating_cache()
    snap = build_player_roster_snapshot(
        league="nba",
        player_id="1",
        roster_row={"name": "Jayson Tatum", "position": "SF", "experience": 8},
        team_abbr="bos",
        cutoff_date="4-16-2017",
    )
    assert snap["algo_rating"] == 96.0
    assert snap["rating_source"] == "2k"
    assert snap["rating_year"] == 2026
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_full_snapshot_includes_external_rating() -> None:
    clear_rating_cache()
    snap = build_player_snapshot(
        league="nba",
        player_id="2",
        roster_row={"name": "Jaylen Brown", "position": "SG", "experience": 8},
        overview={},
        stats_payload={"categories": []},
        team_abbr="bos",
        cutoff_date="4-16-2017",
    )
    assert snap["algo_rating"] == 89.0
    assert snap["rating_source"] == "2k"
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_rating_source_label_external() -> None:
    assert rating_source_label("2k", year=2026) == "2K '26"
    assert rating_source_label("madden", year=2026) == "Madden '26"


def test_team_average_player_rating() -> None:
    assert team_average_player_rating([80.0, 70.0, 60.0]) == 70.0
    assert team_average_player_rating([]) is None


def test_enrich_team_roster_ratings_external_and_cache() -> None:
    clear_rating_cache()
    roster = [
        {"id": "1", "name": "Jayson Tatum", "position": "SF", "experience": 5},
        {"id": "2", "name": "Unknown Bench Guy", "position": "F", "experience": 1},
    ]
    enriched, avg = enrich_team_roster_ratings(
        "nba",
        roster,
        {
            "1": {
                "algo_rating": 82.0,
                "rating_source": "2k",
                "rating_year": 2026,
            }
        },
        team_abbr="bos",
        cutoff_date="4-16-2017",
    )
    assert enriched[0]["algo_rating"] == 82.0
    assert enriched[1]["algo_rating"] is None
    assert enriched[1]["rating_source"] == RATING_PRIOR_SOURCE
    assert avg == 82.0


def test_player_algo_rating_none_when_missing() -> None:
    clear_rating_cache()
    assert (
        player_algo_rating(
            "nba",
            [],
            [],
            "G",
            {"name": "Nobody"},
            team_abbr="bos",
            cutoff_date="4-16-2017",
        )
        is None
    )
