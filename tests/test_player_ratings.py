"""Player algo rating computation tests (predictive model source)."""

from unittest.mock import patch

from web.sports_db.normalize import build_player_roster_snapshot, build_player_snapshot
from web.sports_db.player_ratings import (
    enrich_team_roster_ratings,
    player_algo_rating,
    rating_tier,
    team_average_player_rating,
)

_MODEL_PATCH_TARGET = "web.sports_db.model_player_ratings.league_model_rating_context"
_MODEL_PATCH_RETURN = ({"bos": 70.0, "lal": 65.0}, "basketball_matrix")


def test_nba_star_rates_above_bench() -> None:
    with patch(_MODEL_PATCH_TARGET, return_value=_MODEL_PATCH_RETURN):
        star = player_algo_rating(
            "nba",
            [],
            [],
            "G",
            {"name": "Star", "experience": 12},
            team_abbr="bos",
            cutoff_date="4-16-2017",
        )
        bench = player_algo_rating(
            "nba",
            [],
            [],
            "F",
            {"name": "Bench", "experience": 1},
            team_abbr="bos",
            cutoff_date="4-16-2017",
        )
    assert star > bench
    assert rating_tier(star) in {"elite", "good", "average"}


def test_roster_only_snapshot_has_model_rating() -> None:
    with patch(_MODEL_PATCH_TARGET, return_value=_MODEL_PATCH_RETURN):
        snap = build_player_roster_snapshot(
            league="nba",
            player_id="1",
            roster_row={"name": "Rookie", "position": "G", "experience": 0},
            team_abbr="bos",
            cutoff_date="4-16-2017",
        )
    assert snap["algo_rating"] is not None
    assert snap["rating_source"] == "model"
    assert snap["rating_layer"] == "basketball_matrix"
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_full_snapshot_includes_model_rating() -> None:
    with patch(_MODEL_PATCH_TARGET, return_value=_MODEL_PATCH_RETURN):
        snap = build_player_snapshot(
            league="nba",
            player_id="2",
            roster_row={"name": "Star", "position": "G", "experience": 8},
            overview={},
            stats_payload={"categories": []},
            team_abbr="bos",
            cutoff_date="4-16-2017",
        )
    assert snap["algo_rating"] is not None
    assert snap["rating_source"] == "model"
    assert snap["player"]["algo_rating"] == snap["algo_rating"]


def test_team_average_player_rating() -> None:
    assert team_average_player_rating([80.0, 70.0, 60.0]) == 70.0
    assert team_average_player_rating([]) is None


def test_enrich_team_roster_ratings() -> None:
    roster = [
        {"id": "1", "name": "A", "position": "G", "experience": 5},
        {"id": "2", "name": "B", "position": "F", "experience": 1},
    ]
    with patch(_MODEL_PATCH_TARGET, return_value=_MODEL_PATCH_RETURN):
        enriched, avg = enrich_team_roster_ratings(
            "nba",
            roster,
            {"1": 82.0},
            team_abbr="bos",
            cutoff_date="4-16-2017",
        )
    assert enriched[0]["algo_rating"] == 82.0
    assert enriched[0]["rating_source"] == "model"
    assert enriched[1]["algo_rating"] is not None
    assert avg is not None
