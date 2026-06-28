"""Player algo rating computation tests (external game DB source)."""

from web.sports_db.external_ratings import clear_rating_cache
from web.sports_db.normalize import build_player_roster_snapshot, build_player_snapshot
from web.sports_db.player_ratings import (
    PlayerRatingModel,
    enrich_team_roster_ratings,
    player_algo_rating,
    rating_source_label,
    rating_tier,
    team_average_player_rating,
)


def test_nba_2k_star_rates_above_unknown_bench() -> None:
    clear_rating_cache()
    star = player_algo_rating(
        "nba",
        [],
        [],
        "G",
        {"name": "Jayson Tatum", "experience": 8},
        team_abbr="bos",
    )
    bench = player_algo_rating(
        "nba",
        [],
        [],
        "F",
        {"name": "Unknown Bench Player", "experience": 1},
        team_abbr="bos",
    )
    assert star == 96.0
    assert bench < star
    assert rating_tier(star) == "elite"


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
    assert snap["matched"] is True
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


def test_unknown_player_derived_from_team_csv() -> None:
    clear_rating_cache()
    model = PlayerRatingModel()
    payload = model.rate_player(
        "nba",
        {"name": "Two-Way Callup", "position": "G", "experience": 0},
        team_abbr="bos",
    )
    assert payload["matched"] is False
    assert payload["rating_source"] == "derived"
    assert 70 <= payload["algo_rating"] <= 90


def test_team_average_player_rating() -> None:
    assert team_average_player_rating([80.0, 70.0, 60.0]) == 70.0
    assert team_average_player_rating([]) is None


def test_enrich_team_roster_ratings() -> None:
    clear_rating_cache()
    roster = [
        {"id": "1", "name": "Jayson Tatum", "position": "SF", "experience": 8},
        {"id": "2", "name": "Unknown Bench", "position": "F", "experience": 1},
    ]
    enriched, avg = enrich_team_roster_ratings(
        "nba",
        roster,
        team_abbr="bos",
    )
    assert enriched[0]["algo_rating"] == 96.0
    assert enriched[0]["rating_source"] == "2k"
    assert enriched[1]["rating_source"] == "derived"
    assert avg is not None


def test_rating_source_label_external() -> None:
    assert rating_source_label("2k", year=2026) == "2K '26"
    assert rating_source_label("derived", year=2026) == "Estimated '26"
