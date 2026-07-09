"""Model-derived player rating tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from web.basketball_pred_model import (  # noqa: E402
    build_basketball_model,
    extract_team_matrix_strengths,
)
from web.sports_db.model_player_ratings import (  # noqa: E402
    enrich_team_roster_model_ratings,
    league_model_rating_context,
    player_model_rating,
)
from test_basketball_pred_model import _sample_games  # noqa: E402


def test_extract_team_matrix_strengths_ranks_stronger_offense() -> None:
    model = build_basketball_model(_sample_games(), "nba")
    assert model is not None
    strengths = extract_team_matrix_strengths(model)
    assert strengths["a"] > strengths["c"]


def test_basketball_star_above_bench_same_team() -> None:
    team_ratings = {"bos": 74.0}
    with patch(
        "web.sports_db.model_player_ratings.league_model_rating_context",
        return_value=(team_ratings, "basketball_matrix"),
    ):
        star = player_model_rating(
            "nba",
            "bos",
            {"name": "Star Guard", "position": "G", "experience": 12},
            "4-16-2017",
        )
        bench = player_model_rating(
            "nba",
            "bos",
            {"name": "Bench Wing", "position": "F", "experience": 1},
            "4-16-2017",
        )
    assert star["algo_rating"] > bench["algo_rating"]
    assert star["rating_source"] == "model"
    assert star["rating_layer"] == "basketball_matrix"


def test_strong_team_player_above_weak_team_player() -> None:
    team_ratings = {"gsw": 82.0, "det": 41.0}
    with patch(
        "web.sports_db.model_player_ratings.league_model_rating_context",
        return_value=(team_ratings, "basketball_matrix"),
    ):
        strong = player_model_rating(
            "nba",
            "gsw",
            {"name": "Curry", "position": "G", "experience": 10},
            "4-16-2017",
        )
        weak = player_model_rating(
            "nba",
            "det",
            {"name": "Rookie", "position": "G", "experience": 1},
            "4-16-2017",
        )
    assert strong["algo_rating"] > weak["algo_rating"]


def test_enrich_team_roster_model_ratings() -> None:
    roster = [
        {"id": "1", "name": "Veteran", "position": "QB", "experience": 10},
        {"id": "2", "name": "Rookie", "position": "K", "experience": 0},
    ]
    with patch(
        "web.sports_db.model_player_ratings.league_model_rating_context",
        return_value=({"ne": 68.0}, "nfelo"),
    ):
        enriched, avg, layer = enrich_team_roster_model_ratings(
            "nfl", "ne", roster, "9-1-2025"
        )
    assert layer == "nfelo"
    assert enriched[0]["rating_source"] == "model"
    assert enriched[0]["algo_rating"] > enriched[1]["algo_rating"]
    assert avg is not None


def test_league_model_rating_context_nba_layer(monkeypatch) -> None:
    league_model_rating_context.cache_clear()
    fake_model = build_basketball_model(_sample_games(), "nba")
    monkeypatch.setattr(
        "web.sports_db.model_player_ratings.get_basketball_pred_context",
        lambda *_a, **_k: fake_model,
    )
    ratings, layer = league_model_rating_context("nba", "4-16-2017")
    assert layer == "basketball_matrix"
    assert "a" in ratings
    assert 25.0 <= ratings["a"] <= 99.0
    league_model_rating_context.cache_clear()
