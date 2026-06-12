"""Blend service unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.blend_service import (  # noqa: E402
    blend_predictions,
    home_win_prob_to_total_score,
    total_score_to_home_win_prob,
)
from web.season_games import get_league_power_context  # noqa: E402


def test_total_score_to_home_win_prob() -> None:
    assert total_score_to_home_win_prob(-62.0) == 62.0
    assert total_score_to_home_win_prob(55.0) == 45.0


def test_home_win_prob_to_total_score() -> None:
    total, win_prob = home_win_prob_to_total_score(62.0)
    assert total == -62.0
    assert win_prob == 62.0
    total, win_prob = home_win_prob_to_total_score(40.0)
    assert total == 60.0
    assert win_prob == 60.0


def test_blend_legacy_only_when_power_unavailable() -> None:
    import web.blend_service as blend_module

    original = blend_module.run_power_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="6-11-2026",
            home_abbr="bos",
            away_abbr="ny",
        )
        assert result["blend_mode"] == "legacy_only"
        assert result["algorithm"] == "Unified"
        assert result["total_score"] == -60.0
        assert result["legacy"]["algorithm"] == "Algo_V2"
        assert result["power"] is None
        assert "Power model unavailable" in result["blend_note"]
    finally:
        blend_module.run_power_model = original


def test_blend_averages_home_win_probs() -> None:
    import web.blend_service as blend_module

    original = blend_module.run_power_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 5.0,
            "away_power": 1.0,
            "home_win_probability": 70.0,
            "param": 10.0,
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="6-11-2026",
            home_abbr="bos",
            away_abbr="ny",
        )
        assert result["blend_mode"] == "blended"
        assert result["blended_home_win_probability"] == 65.0
        assert result["total_score"] == -65.0
        assert result["win_probability"] == 65.0
    finally:
        blend_module.run_power_model = original


def test_blend_with_power_integration_when_games_exist() -> None:
    sample = [
        ("bos", "ny", "Boston Celtics", "New York Knicks", 110, 99),
        ("bos", "mia", "Boston Celtics", "Miami Heat", 105, 100),
        ("ny", "mia", "New York Knicks", "Miami Heat", 98, 102),
        ("bos", "ny", "Boston Celtics", "New York Knicks", 112, 108),
        ("mia", "bos", "Miami Heat", "Boston Celtics", 95, 101),
    ]
    get_league_power_context.cache_clear()
    try:
        with patch("web.season_games.load_league_completed_games", return_value=sample):
            result = blend_predictions(
                legacy_total_score=-60.0,
                legacy_win_probability=60.0,
                league="nba",
                cutoff_date="6-12-2026",
                home_abbr="bos",
                away_abbr="ny",
                home_name="Boston Celtics",
                away_name="New York Knicks",
            )
        assert result["blend_mode"] == "blended"
        assert result["power"] is not None
        assert result["algorithm"] == "Unified"
    finally:
        get_league_power_context.cache_clear()


if __name__ == "__main__":
    test_total_score_to_home_win_prob()
    test_home_win_prob_to_total_score()
    test_blend_legacy_only_when_power_unavailable()
    test_blend_averages_home_win_probs()
    test_blend_with_power_integration_when_games_exist()
    print("test_blend_service.py: all tests passed")
