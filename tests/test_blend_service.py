"""Blend service unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.blend_service import (  # noqa: E402
    blend_predictions,
    compute_model_agreement,
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

    power_original = blend_module.run_power_model
    basketball_original = blend_module.run_basketball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 5.0,
            "away_power": 1.0,
            "home_win_probability": 70.0,
            "param": 10.0,
        }
        blend_module.run_basketball_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nhl",
            cutoff_date="6-11-2026",
            home_abbr="bos",
            away_abbr="ny",
        )
        assert result["blend_mode"] == "blended"
        assert result["blend_layers"] == 2
        assert result["blended_home_win_probability"] == 65.0
        assert result["total_score"] == -65.0
        assert result["win_probability"] == 65.0
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


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


def test_blend_mlb_three_way_when_layers_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    baseball_original = blend_module.run_baseball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 0.5,
            "away_power": -0.5,
            "home_win_probability": 58.0,
            "param": 10.0,
        }
        blend_module.run_baseball_pred_model = lambda *_a, **_k: {
            "algorithm": "BaseballElo",
            "source": "MLB-Model",
            "home_win_probability": 64.0,
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="mlb",
            cutoff_date="6-12-2026",
            home_abbr="nyy",
            away_abbr="bos",
        )
        assert result["blend_mode"] == "blended"
        assert result["blend_layers"] == 3
        assert result["baseball_pred"] is not None
        assert result["baseball_pred"]["source"] == "MLB-Model"
        assert result["blended_home_win_probability"] == round((60.0 + 58.0 + 64.0) / 3, 2)
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_baseball_pred_model = baseball_original


def test_blend_mlb_two_way_fallback_when_baseball_unavailable() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    baseball_original = blend_module.run_baseball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 0.5,
            "away_power": -0.5,
            "home_win_probability": 58.0,
            "param": 10.0,
        }
        blend_module.run_baseball_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="mlb",
            cutoff_date="6-12-2026",
            home_abbr="nyy",
            away_abbr="bos",
        )
        assert result["blend_mode"] == "blended"
        assert result["blend_layers"] == 2
        assert "baseball_pred" not in result
        assert result["blended_home_win_probability"] == 59.0
        assert "MLB-Model layer unavailable" in result.get("blend_note", "")
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_baseball_pred_model = baseball_original


def test_blend_nba_three_way_when_basketball_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    basketball_original = blend_module.run_basketball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 5.0,
            "away_power": 1.0,
            "home_win_probability": 70.0,
            "param": 10.0,
        }
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "source": "NBA-prediction",
            "home_win_probability": 62.0,
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="6-12-2026",
            home_abbr="bos",
            away_abbr="ny",
        )
        assert result["blend_layers"] == 3
        assert result["basketball_pred"] is not None
        assert result["blended_home_win_probability"] == round((60.0 + 70.0 + 62.0) / 3, 2)
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_blend_soccer_three_way_when_layers_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    soccer_original = blend_module.run_soccer_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 0.8,
            "away_power": -0.4,
            "home_win_probability": 58.0,
            "param": 10.0,
        }
        blend_module.run_soccer_pred_model = lambda *_a, **_k: {
            "algorithm": "SoccerRatings",
            "source": "football-predictor",
            "home_win_probability": 45.0,
            "draw_probability": 28.0,
            "away_win_probability": 27.0,
        }
        result = blend_predictions(
            legacy_total_score=-55.0,
            legacy_win_probability=55.0,
            league="epl",
            cutoff_date="6-12-2026",
            home_abbr="che",
            away_abbr="ars",
        )
        assert result["blend_mode"] == "blended"
        assert result["blend_layers"] == 3
        assert result["threeway"] is True
        assert result["soccer_pred"] is not None
        assert result["soccer_pred"]["source"] == "football-predictor"
        legacy_h, legacy_d, legacy_a = blend_module.soccer_threeway_probs(-55.0, "epl")
        power_total, _ = blend_module.home_win_prob_to_total_score(58.0)
        power_h, power_d, power_a = blend_module.soccer_threeway_probs(power_total, "epl")
        expected_h = round((legacy_h + power_h + 45.0) / 3, 2)
        expected_d = round((legacy_d + power_d + 28.0) / 3, 2)
        expected_a = round((legacy_a + power_a + 27.0) / 3, 2)
        scale = 100.0 / (expected_h + expected_d + expected_a)
        assert result["home_win_probability"] == round(expected_h * scale, 2)
        assert result["draw_probability"] == round(expected_d * scale, 2)
        assert result["away_win_probability"] == round(expected_a * scale, 2)
        assert abs(
            result["home_win_probability"]
            + result["draw_probability"]
            + result["away_win_probability"]
            - 100.0
        ) < 0.05
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_soccer_pred_model = soccer_original


def test_blend_soccer_two_way_fallback_when_soccer_unavailable() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    soccer_original = blend_module.run_soccer_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 0.8,
            "away_power": -0.4,
            "home_win_probability": 58.0,
            "param": 10.0,
        }
        blend_module.run_soccer_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-55.0,
            legacy_win_probability=55.0,
            league="epl",
            cutoff_date="6-12-2026",
            home_abbr="che",
            away_abbr="ars",
        )
        assert result["blend_layers"] == 2
        assert result["threeway"] is True
        assert "soccer_pred" not in result
        assert "Football-predictor layer unavailable" in result.get("blend_note", "")
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_soccer_pred_model = soccer_original


def test_model_agreement_nba_three_layers_agree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    basketball_original = blend_module.run_basketball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 1.0,
            "away_power": -0.5,
            "home_win_probability": 62.0,
            "param": 10.0,
        }
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "home_win_probability": 64.0,
            "source": "matrix",
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(result, "nba")
        assert agreement["required"] == 3
        assert agreement["agreed"] is True
        assert agreement["legacy_side"] == "home"
        assert agreement["power_side"] == "home"
        assert agreement["third_side"] == "home"
        assert agreement["third_source"] == "basketball_pred"
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_model_agreement_nba_three_layers_disagree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    basketball_original = blend_module.run_basketball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": -0.5,
            "away_power": 1.0,
            "home_win_probability": 42.0,
            "param": 10.0,
        }
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "home_win_probability": 64.0,
            "source": "matrix",
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(result, "nba")
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["legacy_side"] == "home"
        assert agreement["power_side"] == "away"
        assert agreement["third_side"] == "home"
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_model_agreement_two_layer_fallback_not_agreed() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    basketball_original = blend_module.run_basketball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 1.0,
            "away_power": -0.5,
            "home_win_probability": 62.0,
            "param": 10.0,
        }
        blend_module.run_basketball_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(result, "nba")
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert result["blend_layers"] == 2
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_model_agreement_soccer_threeway_agree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    soccer_original = blend_module.run_soccer_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 0.8,
            "away_power": -0.4,
            "home_win_probability": 58.0,
            "param": 10.0,
        }
        blend_module.run_soccer_pred_model = lambda *_a, **_k: {
            "algorithm": "FootballPredictor",
            "home_win_probability": 52.0,
            "draw_probability": 24.0,
            "away_win_probability": 24.0,
            "source": "football-predictor",
        }
        result = blend_predictions(
            legacy_total_score=-55.0,
            legacy_win_probability=55.0,
            league="epl",
            cutoff_date="4-15-2025",
            home_abbr="che",
            away_abbr="ars",
        )
        agreement = compute_model_agreement(result, "epl")
        assert agreement["required"] == 3
        assert agreement["agreed"] is True
        assert agreement["legacy_side"] == "home"
        assert agreement["power_side"] == "home"
        assert agreement["third_side"] == "home"
        assert agreement["third_source"] == "soccer_pred"
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_soccer_pred_model = soccer_original


def test_model_agreement_soccer_threeway_disagree_on_draw() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    soccer_original = blend_module.run_soccer_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 0.0,
            "away_power": 0.0,
            "home_win_probability": 50.0,
            "param": 10.0,
        }
        blend_module.run_soccer_pred_model = lambda *_a, **_k: {
            "algorithm": "FootballPredictor",
            "home_win_probability": 20.0,
            "draw_probability": 55.0,
            "away_win_probability": 25.0,
            "source": "football-predictor",
        }
        result = blend_predictions(
            legacy_total_score=-55.0,
            legacy_win_probability=55.0,
            league="epl",
            cutoff_date="4-15-2025",
            home_abbr="che",
            away_abbr="ars",
        )
        agreement = compute_model_agreement(result, "epl")
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["legacy_side"] == "home"
        assert agreement["third_side"] == "draw"
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_soccer_pred_model = soccer_original


def test_model_agreement_nhl_not_required() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "nhl")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


if __name__ == "__main__":
    test_total_score_to_home_win_prob()
    test_home_win_prob_to_total_score()
    test_blend_legacy_only_when_power_unavailable()
    test_blend_averages_home_win_probs()
    test_blend_with_power_integration_when_games_exist()
    test_blend_mlb_three_way_when_layers_available()
    test_blend_mlb_two_way_fallback_when_baseball_unavailable()
    test_blend_nba_three_way_when_basketball_available()
    test_blend_soccer_three_way_when_layers_available()
    test_blend_soccer_two_way_fallback_when_soccer_unavailable()
    test_model_agreement_nba_three_layers_agree()
    test_model_agreement_nba_three_layers_disagree()
    test_model_agreement_two_layer_fallback_not_agreed()
    test_model_agreement_soccer_threeway_agree()
    test_model_agreement_soccer_threeway_disagree_on_draw()
    test_model_agreement_nhl_not_required()
    print("test_blend_service.py: all tests passed")
