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


def test_blend_nhl_three_way_when_hockey_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    hockey_original = blend_module.run_hockey_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 2.0,
            "away_power": -1.0,
            "home_win_probability": 58.0,
            "param": 10.0,
        }
        blend_module.run_hockey_pred_model = lambda *_a, **_k: {
            "algorithm": "HockeyPoisson",
            "source": "hockey-predictions",
            "home_win_probability": 55.0,
            "expected_home_goals": 3.1,
            "expected_away_goals": 2.7,
        }
        result = blend_predictions(
            legacy_total_score=-55.0,
            legacy_win_probability=55.0,
            league="nhl",
            cutoff_date="4-12-2017",
            home_abbr="bos",
            away_abbr="mtl",
        )
        assert result["blend_mode"] == "blended"
        assert result["blend_layers"] == 3
        assert result["hockey_pred"] is not None
        assert result["hockey_pred"]["source"] == "hockey-predictions"
        assert result["blended_home_win_probability"] == round((55.0 + 58.0 + 55.0) / 3, 2)
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_hockey_pred_model = hockey_original


def test_blend_nfl_three_way_when_football_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 3.0,
            "away_power": -2.0,
            "home_win_probability": 62.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "source": "nfelo",
            "home_win_probability": 58.0,
            "projected_spread": -3.5,
        }
        result = blend_predictions(
            legacy_total_score=-58.0,
            legacy_win_probability=58.0,
            league="nfl",
            cutoff_date="1-15-2025",
            home_abbr="kc",
            away_abbr="den",
        )
        assert result["blend_layers"] == 3
        assert result["football_pred"] is not None
        assert result["football_pred"]["source"] == "nfelo"
        assert result["blended_home_win_probability"] == round((58.0 + 62.0 + 58.0) / 3, 2)
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_football_pred_model = football_original


def test_blend_cfb_three_way_when_football_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_win_probability": 55.0,
            "home_power": 1.0,
            "away_power": 0.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "source": "nfelo",
            "home_win_probability": 57.0,
        }
        result = blend_predictions(
            legacy_total_score=-57.0,
            legacy_win_probability=57.0,
            league="cfb",
            cutoff_date="12-15-2024",
            home_abbr="ala",
            away_abbr="ga",
        )
        assert result["blend_layers"] == 3
        assert result["football_pred"] is not None
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_football_pred_model = football_original


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
        agreement = compute_model_agreement(
            result,
            "nba",
            market={"away_moneyline": 110, "home_moneyline": -130},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is True
        assert agreement["agreement_mode"] == "value"
        assert "home" in agreement["value_sides"]
        assert agreement["legacy_side"] == "home"
        assert agreement["power_side"] == "home"
        assert agreement["third_side"] == "home"
        assert agreement["third_source"] == "basketball_pred"
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_model_agreement_nba_value_on_underdog_despite_favorite_disagreement() -> None:
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
        agreement = compute_model_agreement(
            result,
            "nba",
            market={"away_moneyline": 220, "home_moneyline": -260},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is True
        assert "away" in agreement["value_sides"]
        assert agreement["legacy_side"] == "away"
        assert agreement["power_side"] == "away"
        assert agreement["third_side"] == "away"
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
        agreement = compute_model_agreement(
            result,
            "nba",
            market={"away_moneyline": 70, "home_moneyline": -110},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["value_sides"] == []
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_model_agreement_nba_one_layer_lacks_shared_value() -> None:
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
        agreement = compute_model_agreement(
            result,
            "nba",
            market={"away_moneyline": 160, "home_moneyline": -190},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["legacy_side"] == "away"
        assert agreement["power_side"] is None
        assert agreement["third_side"] is None
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


def test_model_agreement_nhl_requires_three_layers() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "nhl")
    assert agreement["required"] == 3
    assert agreement["agreed"] is False


if __name__ == "__main__":
    test_total_score_to_home_win_prob()
    test_home_win_prob_to_total_score()
    test_blend_legacy_only_when_power_unavailable()
    test_blend_averages_home_win_probs()
    test_blend_with_power_integration_when_games_exist()
    test_blend_mlb_three_way_when_layers_available()
    test_blend_mlb_two_way_fallback_when_baseball_unavailable()
    test_blend_nba_three_way_when_basketball_available()
    test_model_agreement_nba_three_layers_agree()
    test_model_agreement_nba_value_on_underdog_despite_favorite_disagreement()
    test_model_agreement_nba_three_layers_disagree()
    test_model_agreement_nba_one_layer_lacks_shared_value()
    test_model_agreement_two_layer_fallback_not_agreed()
    test_model_agreement_nhl_requires_three_layers()
    print("test_blend_service.py: all tests passed")
