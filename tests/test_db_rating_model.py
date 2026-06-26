"""Database player-rating layer tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.db_rating_model import (  # noqa: E402
    _ovr_rating_prob,
    _redistribute_weights,
    _soccer_rating_prob,
    apply_db_rating_blend,
    db_rating_blend_weight,
    sparse_schedule_factor,
)


def test_redistribute_weights_adds_db_layer() -> None:
    weights = _redistribute_weights({"legacy": 0.5, "power": 0.5}, 0.2)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert abs(weights["db_rating"] - 0.2) < 1e-6
    assert abs(weights["legacy"] - 0.4) < 1e-6


def test_sparse_schedule_factor_zero_when_enough_games(monkeypatch) -> None:
    monkeypatch.setattr(
        "web.db_rating_model._team_game_count",
        lambda *_a, **_k: 20,
    )
    assert sparse_schedule_factor("mlb", "6-26-2026", "bos", "nyy") == 0.0


def test_sparse_schedule_factor_high_when_few_games(monkeypatch) -> None:
    monkeypatch.setattr(
        "web.db_rating_model._team_game_count",
        lambda *_a, **_k: 2,
    )
    factor = sparse_schedule_factor("worldcup", "6-26-2026", "esp", "uru")
    assert factor > 0.8


def test_db_rating_blend_weight_increases_with_sparse_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        "web.db_rating_model.sparse_schedule_factor",
        lambda *_a, **_k: 0.0,
    )
    low = db_rating_blend_weight("mlb", "6-26-2026", "bos", "nyy", has_db_layer=True)
    monkeypatch.setattr(
        "web.db_rating_model.sparse_schedule_factor",
        lambda *_a, **_k: 1.0,
    )
    high = db_rating_blend_weight("worldcup", "6-26-2026", "esp", "uru", has_db_layer=True)
    assert high > low


def test_apply_db_rating_blend_binary() -> None:
    blended = {
        "blend_layers": 2,
        "blend_weights": {"legacy": 0.5, "power": 0.5},
        "legacy": {"home_win_probability": 40.0},
        "power": {"home_win_probability": 60.0},
        "blended_home_win_probability": 50.0,
        "total_score": 0.0,
        "win_probability": 50.0,
        "favorite_side": "home",
    }
    db_payload = {
        "source_home": "maddenratings.com",
        "source_away": "maddenratings.com",
        "home_rating": 87.0,
        "away_rating": 81.0,
        "home_win_probability": 62.0,
        "away_win_probability": 38.0,
        "sparse_schedule_factor": 0.5,
        "blend_weight_hint": 0.25,
    }
    with patch("web.db_rating_model.run_db_rating_model", return_value=db_payload):
        updated = apply_db_rating_blend(
            blended,
            league="nfl",
            cutoff_date="6-26-2026",
            home_abbr="bal",
            away_abbr="dal",
            home_slug="baltimore-ravens",
            away_slug="dallas-cowboys",
        )
    assert updated["db_rating"]["home_rating"] == 87.0
    assert updated["blend_layers"] == 3
    assert updated["win_probability"] != 50.0


def test_rating_prob_helpers() -> None:
    assert _ovr_rating_prob(90, 80) > 50
    assert _soccer_rating_prob(2000, 1800) > 50


if __name__ == "__main__":
    test_redistribute_weights_adds_db_layer()
    test_sparse_schedule_factor_zero_when_enough_games()
    test_sparse_schedule_factor_high_when_few_games()
    test_db_rating_blend_weight_increases_with_sparse_schedule()
    test_apply_db_rating_blend_binary()
    test_rating_prob_helpers()
    print("test_db_rating_model.py: all tests passed")
