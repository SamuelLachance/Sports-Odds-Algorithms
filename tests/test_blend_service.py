"""Blend service unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.blend_service import (  # noqa: E402
    blended_home_spread_margin,
    blend_predictions,
    compute_model_agreement,
    home_win_prob_to_total_score,
    layer_home_win_probability,
    total_score_to_home_win_prob,
)
from web.season_games import get_league_power_context  # noqa: E402


@pytest.fixture(autouse=True)
def _neutral_sports_meta_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep blend unit tests deterministic (equal layers, no temperature shift)."""
    from web import sports_meta_model

    def _equal_config(league: str) -> dict:
        return {
            "blend_weights": {
                "legacy": 1.0 / 3.0,
                "power": 1.0 / 3.0,
                "sport_pred": 1.0 / 3.0,
            },
            "temperature": 1.0,
            "two_layer": False,
        }

    monkeypatch.setattr(sports_meta_model, "get_sports_meta_config", _equal_config)


def test_total_score_to_home_win_prob() -> None:
    assert total_score_to_home_win_prob(-62.0) == 62.0
    assert total_score_to_home_win_prob(55.0) == 45.0
    assert total_score_to_home_win_prob(0.0) == 50.0


def test_home_win_prob_to_total_score() -> None:
    total, win_prob = home_win_prob_to_total_score(62.0)
    assert total == -62.0
    assert win_prob == 62.0
    total, win_prob = home_win_prob_to_total_score(40.0)
    assert total == 60.0
    assert win_prob == 60.0
    total, win_prob = home_win_prob_to_total_score(50.0)
    assert total == 0.0
    assert win_prob == 50.0


def test_layer_home_win_probability_from_legacy_payload() -> None:
    assert layer_home_win_probability(
        {
            "total_score": -52.41,
            "win_probability": 52.41,
            "favorite_side": "home",
        }
    ) == 52.41
    assert layer_home_win_probability(
        {
            "total_score": 64.31,
            "win_probability": 64.31,
            "favorite_side": "away",
        }
    ) == pytest.approx(35.69, abs=0.01)


def test_blend_with_espn_ids_and_slugs_calls_db_rating_without_espn_kwargs() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    captured: dict[str, object] = {}

    def _capture_db_blend(blended: dict, **kwargs: object) -> dict:
        captured.update(kwargs)
        return blended

    try:
        blend_module.run_power_model = lambda *_a, **_k: None
        with patch("web.blend_service.apply_db_rating_blend", side_effect=_capture_db_blend):
            result = blend_predictions(
                legacy_total_score=-60.0,
                legacy_win_probability=60.0,
                league="nba",
                cutoff_date="6-12-2026",
                home_abbr="bos",
                away_abbr="mia",
                home_slug="boston-celtics",
                away_slug="miami-heat",
                home_espn_id="123",
                away_espn_id="456",
            )
        assert result["blend_mode"] == "legacy_only"
        assert "home_espn_id" not in captured
        assert "away_espn_id" not in captured
        assert captured.get("home_slug") == "boston-celtics"
    finally:
        blend_module.run_power_model = power_original


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
        assert result["legacy"]["home_win_probability"] == 60.0
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
            league="nba",
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


def test_blend_mlb_runcast_only_when_available() -> None:
    import web.blend_service as blend_module

    mlb_original = blend_module.run_mlb_pred_model
    try:
        blend_module.run_mlb_pred_model = lambda *_a, **_k: {
            "algorithm": "MLBRunCast",
            "source": "run-sim-xgb-calibrated",
            "home_win_probability": 56.0,
            "predicted_margin": 0.8,
            "mlb_pick_signals": {"disagreement_signal": True},
        }
        result = blend_predictions(
            legacy_total_score=-56.0,
            legacy_win_probability=56.0,
            league="mlb",
            cutoff_date="06-15-2024",
            home_abbr="nyy",
            away_abbr="bos",
        )
        assert result["blend_mode"] == "mlb_runcast"
        assert result["blend_layers"] == 1
        assert result["algorithm"] == "MLBRunCast"
        assert result["baseball_pred"] is not None
        assert result["baseball_pred"]["source"] == "run-sim-xgb-calibrated"
        assert result["blended_home_win_probability"] == 56.0
        assert result.get("legacy") is None
        assert result.get("power") is None
        assert result.get("home_spread_margin") == -0.8
        assert result.get("mlb_pick_signals", {}).get("disagreement_signal") is True
    finally:
        blend_module.run_mlb_pred_model = mlb_original


def test_model_agreement_mlb_single_model() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "mlb")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


def test_blend_mlb_runcast_unavailable_fallback() -> None:
    import web.blend_service as blend_module

    mlb_original = blend_module.run_mlb_pred_model
    try:
        blend_module.run_mlb_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-52.0,
            legacy_win_probability=52.0,
            league="mlb",
            cutoff_date="06-15-2024",
            home_abbr="nyy",
            away_abbr="bos",
        )
        assert result["blend_mode"] == "mlb_runcast_unavailable"
        assert result["algorithm"] == "MLBRunCast"
        assert result.get("baseball_pred") is None
    finally:
        blend_module.run_mlb_pred_model = mlb_original


def test_blend_cbb_torvik_only_when_available() -> None:
    import web.blend_service as blend_module

    cbb_original = blend_module.run_cbb_pred_model
    try:
        blend_module.run_cbb_pred_model = lambda *_a, **_k: {
            "algorithm": "CBBTorvik",
            "source": "torvik-efficiency-calibrated",
            "home_win_probability": 58.0,
            "predicted_margin": 4.5,
            "cbb_pick_signals": {"disagreement_signal": True},
        }
        result = blend_predictions(
            legacy_total_score=-58.0,
            legacy_win_probability=58.0,
            league="cbb",
            cutoff_date="11-15-2024",
            home_abbr="duke",
            away_abbr="unc",
        )
        assert result["blend_mode"] == "cbb_torvik"
        assert result["blend_layers"] == 1
        assert result["algorithm"] == "CBBTorvik"
        assert result["basketball_pred"] is not None
        assert result["basketball_pred"]["source"] == "torvik-efficiency-calibrated"
        assert result["blended_home_win_probability"] == 58.0
        assert result.get("legacy") is None
        assert result.get("power") is None
    finally:
        blend_module.run_cbb_pred_model = cbb_original


def test_blend_wnba_elo_xgb_only_when_available() -> None:
    import web.blend_service as blend_module

    wnba_original = blend_module.run_wnba_pred_model
    try:
        blend_module.run_wnba_pred_model = lambda *_a, **_k: {
            "algorithm": "WNBAEloXGB",
            "source": "elo-efficiency-xgb-calibrated",
            "home_win_probability": 57.0,
            "predicted_margin": 3.5,
            "wnba_pick_signals": {"disagreement_signal": True},
        }
        result = blend_predictions(
            legacy_total_score=-57.0,
            legacy_win_probability=57.0,
            league="wnba",
            cutoff_date="06-15-2025",
            home_abbr="sea",
            away_abbr="lv",
        )
        assert result["blend_mode"] == "wnba_elo_xgb"
        assert result["blend_layers"] == 1
        assert result["algorithm"] == "WNBAEloXGB"
        assert result["basketball_pred"] is not None
        assert result["basketball_pred"]["source"] == "elo-efficiency-xgb-calibrated"
        assert result["blended_home_win_probability"] == 57.0
        assert result.get("legacy") is None
        assert result.get("power") is None
        assert result.get("home_spread_margin") == -3.5
        assert result.get("wnba_pick_signals", {}).get("disagreement_signal") is True
    finally:
        blend_module.run_wnba_pred_model = wnba_original


def test_model_agreement_wnba_single_model() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "wnba")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


def test_model_agreement_cbb_single_model() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "cbb")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


def test_blend_nhl_puckcast_only_when_hockey_available() -> None:
    import web.blend_service as blend_module

    hockey_original = blend_module.run_hockey_pred_model
    try:
        blend_module.run_hockey_pred_model = lambda *_a, **_k: {
            "algorithm": "HockeyPuckCast",
            "source": "puckcast-xg-goalie",
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
        assert result["blend_mode"] == "hockey_puckcast"
        assert result["blend_layers"] == 1
        assert result["algorithm"] == "HockeyPuckCast"
        assert result["hockey_pred"] is not None
        assert result["hockey_pred"]["source"] == "puckcast-xg-goalie"
        assert result["blended_home_win_probability"] == 55.0
        assert result.get("legacy") is None
        assert result.get("power") is None
    finally:
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
            market={"away_moneyline": 110, "home_moneyline": -110},
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
            market={"away_moneyline": 200, "home_moneyline": -230},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["legacy_side"] == "away"
        # 25+ edge threshold: power finds away value; basketball still lacks ML edge.
        assert agreement["power_side"] == "away"
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


def test_model_agreement_nba_spread_three_layers_agree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    basketball_original = blend_module.run_basketball_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 5.0,
            "away_power": -2.0,
            "home_win_probability": 97.0,
            "param": 10.0,
        }
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "home_win_probability": 96.0,
            "predicted_margin": 8.0,
            "source": "matrix",
        }
        result = blend_predictions(
            legacy_total_score=-99.0,
            legacy_win_probability=99.0,
            league="nba",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(
            result,
            "nba",
            market={
                "spread": -1.5,
                "away_spread_odds": -110,
                "home_spread_odds": -110,
            },
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is True
        assert "home" in agreement["value_sides"]
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_basketball_pred_model = basketball_original


def test_model_agreement_nhl_single_model() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "nhl")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


def test_blended_home_spread_margin_matches_unified_total_score() -> None:
    """Spread picks use unified total_score margin, not a layer average."""
    blended = {
        "total_score": 61.43,
        "win_probability": 61.43,
        "favorite_side": "away",
        "legacy": {"total_score": -55.0, "predicted_margin": -4.0},
        "power": {"predicted_margin": 10.0},
        "basketball_pred": {"predicted_margin": 12.0, "home_win_probability": 70.0},
    }
    margin = blended_home_spread_margin(blended, "wnba")
    from web.bet_advisor import model_home_margin

    expected = model_home_margin(61.43, "wnba")
    assert margin == expected
    assert margin > 0


def test_attach_home_spread_margin_exposes_pick_margin() -> None:
    from web.blend_service import _attach_home_spread_margin

    blended = {
        "total_score": -76.74,
        "win_probability": 76.74,
        "favorite_side": "home",
        "basketball_pred": {"predicted_margin": 8.8, "home_win_probability": 71.94},
    }
    attached = _attach_home_spread_margin(dict(blended), "wnba")
    assert attached["home_spread_margin"] == round(
        blended_home_spread_margin(blended, "wnba"), 2
    )
    assert attached["home_spread_margin"] == -3.21
    assert blended_home_spread_margin(attached, "wnba") == attached["home_spread_margin"]


def test_attach_home_spread_margin_skips_moneyline_leagues() -> None:
    from web.blend_service import _attach_home_spread_margin

    blended = {"total_score": -60.0, "win_probability": 60.0}
    attached = _attach_home_spread_margin(dict(blended), "nhl")
    assert "home_spread_margin" not in attached


if __name__ == "__main__":
    test_total_score_to_home_win_prob()
    test_home_win_prob_to_total_score()
    test_blend_legacy_only_when_power_unavailable()
    test_blend_averages_home_win_probs()
    test_blend_with_power_integration_when_games_exist()
    test_blend_mlb_runcast_only_when_available()
    test_blend_mlb_runcast_unavailable_fallback()
    test_blend_nba_three_way_when_basketball_available()
    test_model_agreement_nba_three_layers_agree()
    test_model_agreement_nba_value_on_underdog_despite_favorite_disagreement()
    test_model_agreement_nba_three_layers_disagree()
    test_model_agreement_nba_one_layer_lacks_shared_value()
    test_model_agreement_two_layer_fallback_not_agreed()
    test_model_agreement_nba_spread_three_layers_agree()
    test_model_agreement_nhl_single_model()
    print("test_blend_service.py: all tests passed")
