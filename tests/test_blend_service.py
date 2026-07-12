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
                league="nfl",
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
            league="nfl",
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
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 5.0,
            "away_power": 1.0,
            "home_win_probability": 70.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nfl",
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
        blend_module.run_football_pred_model = football_original


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
                league="nfl",
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


def test_blend_mlb_v2_sets_mlb_v2_blend_mode() -> None:
    import web.blend_service as blend_module
    from web.daily_service import SINGLE_MODEL_BLEND_MODES

    mlb_original = blend_module.run_mlb_pred_model
    try:
        blend_module.run_mlb_pred_model = lambda *_a, **_k: {
            "algorithm": "MLBGradientBoost",
            "model_version": "v2",
            "home_win_probability": 58.0,
            "predicted_margin": 1.1,
        }
        result = blend_predictions(
            legacy_total_score=-58.0,
            legacy_win_probability=58.0,
            league="mlb",
            cutoff_date="06-15-2024",
            home_abbr="nyy",
            away_abbr="bos",
        )
        assert result["blend_mode"] == "mlb_v2"
        assert result["model_version"] == "v2"
        assert "mlb_v2" in SINGLE_MODEL_BLEND_MODES
    finally:
        blend_module.run_mlb_pred_model = mlb_original


def test_blend_mlb_hoists_hubacek_decorrelation_fields() -> None:
    """MLB single-layer blend must expose decorrelation flags like CBB/NHL."""
    import web.blend_service as blend_module

    mlb_original = blend_module.run_mlb_pred_model
    try:
        blend_module.run_mlb_pred_model = lambda *_a, **_k: {
            "algorithm": "MLBGradientBoost",
            "model_version": "v2",
            "home_win_probability": 60.0,
            "pre_decorrelation_home_win_probability": 55.0,
            "market_decorrelated": True,
            "predicted_margin": 0.8,
        }
        result = blend_predictions(
            legacy_total_score=-56.0,
            legacy_win_probability=56.0,
            league="mlb",
            cutoff_date="06-15-2024",
            home_abbr="nyy",
            away_abbr="bos",
        )
        assert result["market_decorrelated"] is True
        assert result["pre_decorrelation_home_win_probability"] == 55.0
        assert result["blended_home_win_probability"] == 60.0
    finally:
        blend_module.run_mlb_pred_model = mlb_original


def test_model_agreement_mlb_single_model() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "mlb")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


def test_run_sport_pred_model_forwards_mlb_dh_keys() -> None:
    """Ensemble / non-slate MLB path must forward kickoff + game_number for DH."""
    from web import blend_service as blend_module

    captured: dict[str, object] = {}

    def _fake_mlb(*_a, **kwargs):
        captured.update(kwargs)
        return {
            "algorithm": "MLBRunCast",
            "home_win_probability": 55.0,
            "predicted_margin": 0.4,
        }

    original = blend_module.run_mlb_pred_model
    try:
        blend_module.run_mlb_pred_model = _fake_mlb
        key, payload = blend_module._run_sport_pred_model(
            "mlb",
            "06-15-2025",
            "nyy",
            "bos",
            kickoff_iso="2025-06-15T20:10:00Z",
            game_number=2,
        )
    finally:
        blend_module.run_mlb_pred_model = original

    assert key == "baseball_pred"
    assert payload is not None
    assert captured.get("kickoff_iso") == "2025-06-15T20:10:00Z"
    assert captured.get("game_number") == 2


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


def test_blend_basketball_matrix_only_when_available() -> None:
    import web.blend_service as blend_module

    basketball_original = blend_module.run_basketball_pred_model
    cbb_original = blend_module.run_cbb_pred_model
    power_original = blend_module.run_power_model
    nba_v2_original = blend_module._run_nba_v2
    wnba_v2_original = blend_module._run_wnba_v2
    cbb_v2_original = blend_module._run_cbb_v2
    try:
        blend_module.run_power_model = lambda *_a, **_k: None
        blend_module._run_nba_v2 = lambda *_a, **_k: None
        blend_module._run_wnba_v2 = lambda *_a, **_k: None
        blend_module._run_cbb_v2 = lambda *_a, **_k: None
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "source": "soft-impute-svd",
            "home_win_probability": 58.0,
            "predicted_margin": 4.5,
            "predicted_home_score": 112.3,
            "predicted_away_score": 107.8,
            "predicted_home_offensive_rating": 115.2,
            "predicted_away_offensive_rating": 110.1,
            "predicted_pace": 98.5,
        }
        # Force CBB Torvik unavailable so matrix fallback is exercised.
        blend_module.run_cbb_pred_model = lambda *_a, **_k: None
        for league in ("nba", "wnba", "cbb"):
            result = blend_predictions(
                legacy_total_score=-58.0,
                legacy_win_probability=58.0,
                league=league,
                cutoff_date="11-15-2024",
                home_abbr="duke",
                away_abbr="unc",
            )
            assert result["blend_mode"] == "basketball_matrix"
            assert result["blend_layers"] >= 1
            assert result["algorithm"] == "BasketballMatrix"
            assert result["basketball_pred"] is not None
            assert result["basketball_pred"]["source"] == "soft-impute-svd"
            assert result["blended_home_win_probability"] == 58.0
            assert result.get("legacy") is not None
            assert result["legacy"]["home_win_probability"] is not None
            assert result.get("home_spread_margin") == -4.5
    finally:
        blend_module.run_basketball_pred_model = basketball_original
        blend_module.run_cbb_pred_model = cbb_original
        blend_module.run_power_model = power_original
        blend_module._run_nba_v2 = nba_v2_original
        blend_module._run_wnba_v2 = wnba_v2_original
        blend_module._run_cbb_v2 = cbb_v2_original


def test_blend_cbb_prefers_torvik_when_available() -> None:
    import web.blend_service as blend_module

    basketball_original = blend_module.run_basketball_pred_model
    cbb_original = blend_module.run_cbb_pred_model
    power_original = blend_module.run_power_model
    cbb_v2_original = blend_module._run_cbb_v2
    try:
        blend_module.run_power_model = lambda *_a, **_k: None
        blend_module._run_cbb_v2 = lambda *_a, **_k: None
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "source": "soft-impute-svd",
            "home_win_probability": 55.0,
            "predicted_margin": 3.0,
        }
        blend_module.run_cbb_pred_model = lambda *_a, **_k: {
            "algorithm": "CBBTorvik",
            "source": "torvik-efficiency-calibrated",
            "market_decorrelated": True,
            "home_win_probability": 62.0,
            "pre_decorrelation_home_win_probability": 64.5,
            "predicted_margin": 7.5,
            "cbb_pick_signals": {"steam": False},
        }
        result = blend_predictions(
            legacy_total_score=-58.0,
            legacy_win_probability=58.0,
            league="cbb",
            cutoff_date="11-15-2024",
            home_abbr="duke",
            away_abbr="unc",
            consensus_spread=-6.5,
        )
        assert result["blend_mode"] == "cbb_torvik"
        assert result["algorithm"] == "CBBTorvik"
        assert result["market_decorrelated"] is True
        assert result["pre_decorrelation_home_win_probability"] == 64.5
        assert result["blended_home_win_probability"] == 62.0
        assert result["home_spread_margin"] == -7.5
        assert result["cbb_pick_signals"] == {"steam": False}
        assert result["basketball_pred"]["source"] == "torvik-efficiency-calibrated"
    finally:
        blend_module.run_basketball_pred_model = basketball_original
        blend_module.run_cbb_pred_model = cbb_original
        blend_module.run_power_model = power_original
        blend_module._run_cbb_v2 = cbb_v2_original


def test_blend_cbb_falls_back_to_matrix_when_torvik_fails() -> None:
    import web.blend_service as blend_module

    basketball_original = blend_module.run_basketball_pred_model
    cbb_original = blend_module.run_cbb_pred_model
    power_original = blend_module.run_power_model
    cbb_v2_original = blend_module._run_cbb_v2
    try:
        def _boom(*_a, **_k):
            raise RuntimeError("torvik down")

        blend_module.run_power_model = lambda *_a, **_k: None
        blend_module._run_cbb_v2 = lambda *_a, **_k: None
        blend_module.run_cbb_pred_model = _boom
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "source": "soft-impute-svd",
            "home_win_probability": 57.0,
            "predicted_margin": 4.0,
        }
        result = blend_predictions(
            legacy_total_score=-57.0,
            legacy_win_probability=57.0,
            league="cbb",
            cutoff_date="11-15-2024",
            home_abbr="duke",
            away_abbr="unc",
        )
        assert result["blend_mode"] == "basketball_matrix"
        assert result["algorithm"] == "BasketballMatrix"
        assert result["blended_home_win_probability"] == 57.0
    finally:
        blend_module.run_basketball_pred_model = basketball_original
        blend_module.run_cbb_pred_model = cbb_original
        blend_module.run_power_model = power_original
        blend_module._run_cbb_v2 = cbb_v2_original


def test_model_agreement_basketball_single_model() -> None:
    for league in ("nba", "wnba", "cbb"):
        agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, league)
        assert agreement["required"] == 0
        assert agreement["agreed"] is True


def test_blend_soccer_path_a_only_when_available() -> None:
    import web.blend_service as blend_module

    soccer_original = blend_module.run_soccer_pred_model
    finalize_original = blend_module.finalize_soccer_path_a
    try:
        blend_module.run_soccer_pred_model = lambda *_a, **_k: {
            "algorithm": "SoccerPathA",
            "source": "path-a-xgb-decorrelated",
            "home_win_probability": 48.0,
            "draw_probability": 27.0,
            "away_win_probability": 25.0,
            "expected_home_goals": 1.6,
            "expected_away_goals": 1.1,
            "soccer_pick_signals": {"disagreement_signal": True},
        }
        blend_module.finalize_soccer_path_a = lambda result, **_kwargs: result
        result = blend_predictions(
            legacy_total_score=-48.0,
            legacy_win_probability=48.0,
            league="epl",
            cutoff_date="11-15-2024",
            home_abbr="ars",
            away_abbr="liv",
            home_moneyline=-120,
            draw_moneyline=260,
            away_moneyline=320,
        )
        assert result["blend_mode"] == "soccer_path_a"
        assert result["blend_layers"] == 1
        assert result["algorithm"] == "SoccerPathA"
        assert result["threeway"] is True
        assert result["soccer_pred"] is not None
        assert result["home_win_probability"] == 48.0
        assert result["draw_probability"] == 27.0
        assert result["away_win_probability"] == 25.0
        assert result.get("legacy") is None
        assert result.get("power") is None
        assert result.get("soccer_pick_signals", {}).get("disagreement_signal") is True
    finally:
        blend_module.run_soccer_pred_model = soccer_original
        blend_module.finalize_soccer_path_a = finalize_original


def test_blend_soccer_path_a_unavailable_fallback() -> None:
    import web.blend_service as blend_module

    soccer_original = blend_module.run_soccer_pred_model
    try:
        blend_module.run_soccer_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-52.0,
            legacy_win_probability=52.0,
            league="epl",
            cutoff_date="11-15-2024",
            home_abbr="ars",
            away_abbr="liv",
        )
        assert result["blend_mode"] == "soccer_path_a_unavailable"
        assert result["algorithm"] == "SoccerPathA"
        assert result.get("soccer_pred") is None
        assert result["threeway"] is True
        # Complete 1X2 so predict_live_game never KeyErrors; lock context.
        assert result["home_win_probability"] == pytest.approx(33.33)
        assert result["draw_probability"] == pytest.approx(33.33)
        assert result["away_win_probability"] == pytest.approx(33.34)
        assert result["context_adjustment_pp"] == 0.0
        float(result["home_win_probability"])
        float(result["draw_probability"])
        float(result["away_win_probability"])
    finally:
        blend_module.run_soccer_pred_model = soccer_original


def test_blend_soccer_v2_payload_skips_path_a_enrichment() -> None:
    """v2 keeps calibrated probs; ESPN context attaches for display only."""
    import web.blend_service as blend_module

    soccer_original = blend_module.run_soccer_pred_model
    finalize_original = blend_module.finalize_soccer_path_a
    display_original = blend_module.attach_soccer_context_display

    def _fail_finalize(result, **_kwargs):
        raise AssertionError("finalize_soccer_path_a must not run for v2 payloads")

    def _fake_display(result, **_kwargs):
        updated = dict(result)
        updated["soccer_context"] = {
            "algorithm": "SoccerContext",
            "factors": [{"label": "Home injury", "detail": "key midfielder out"}],
            "home_win_probability": 48.0,
            "draw_probability": 28.0,
            "away_win_probability": 24.0,
        }
        updated["context_adjustment_pp"] = 0.0
        return updated

    try:
        blend_module.run_soccer_pred_model = lambda *_a, **_k: {
            "algorithm": "SoccerGradientBoost v2",
            "model_version": "soccer_v2.2026.07",
            "source": "soccer-v2-xgb-ensemble",
            "home_win_probability": 51.2,
            "draw_probability": 26.4,
            "away_win_probability": 22.4,
            "pick_home_win_probability": 52.0,
            "pick_draw_probability": 26.1,
            "pick_away_win_probability": 21.9,
            "expected_home_goals": 1.9,
            "expected_away_goals": 1.1,
            "elo_home": 1760.0,
            "elo_away": 1655.0,
            "market_decorrelated": True,
            "market_calibrated": True,
            "soccer_pick_signals": {"disagreement_signal": False},
        }
        blend_module.finalize_soccer_path_a = _fail_finalize
        blend_module.attach_soccer_context_display = _fake_display
        result = blend_predictions(
            legacy_total_score=-48.0,
            legacy_win_probability=48.0,
            league="epl",
            cutoff_date="11-15-2025",
            home_abbr="ars",
            away_abbr="liv",
            home_moneyline=-120,
            draw_moneyline=260,
            away_moneyline=320,
        )
        assert result["algorithm"] == "SoccerGradientBoost v2"
        assert result["blend_mode"] == "soccer_v2"
        assert result["blend_layers"] == 1
        assert result["threeway"] is True
        assert result["market_decorrelated"] is True
        assert result["model_version"] == "soccer_v2.2026.07"
        # calibrated blend probs unchanged despite context payload
        assert result["home_win_probability"] == 51.2
        assert result["draw_probability"] == 26.4
        assert result["away_win_probability"] == 22.4
        assert result["soccer_pred"]["pick_home_win_probability"] == 52.0
        assert result.get("context_adjusted") is not True
        assert result["soccer_context"]["factors"][0]["label"] == "Home injury"
        assert result.get("context_adjustment_pp") == 0.0
    finally:
        blend_module.run_soccer_pred_model = soccer_original
        blend_module.finalize_soccer_path_a = finalize_original
        blend_module.attach_soccer_context_display = display_original


def test_blend_soccer_forwards_match_date_to_pred_model() -> None:
    import web.blend_service as blend_module

    soccer_original = blend_module.run_soccer_pred_model
    finalize_original = blend_module.finalize_soccer_path_a
    captured: dict = {}

    def _fake_soccer(*_a, **kwargs):
        captured.update(kwargs)
        return {
            "algorithm": "SoccerPathA",
            "home_win_probability": 40.0,
            "draw_probability": 30.0,
            "away_win_probability": 30.0,
        }

    try:
        blend_module.run_soccer_pred_model = _fake_soccer
        blend_module.finalize_soccer_path_a = lambda result, **_k: result
        blend_predictions(
            legacy_total_score=-40.0,
            legacy_win_probability=40.0,
            league="epl",
            cutoff_date="7-10-2026",
            home_abbr="ars",
            away_abbr="liv",
            match_date="2026-07-10",
        )
        assert captured.get("match_date") == "2026-07-10"
    finally:
        blend_module.run_soccer_pred_model = soccer_original
        blend_module.finalize_soccer_path_a = finalize_original


def test_model_agreement_soccer_single_model() -> None:
    agreement = compute_model_agreement({"legacy": {"favorite_side": "home"}}, "epl")
    assert agreement["required"] == 0
    assert agreement["agreed"] is True


def test_blend_nhl_algo_v1_fallback_when_v2_unavailable() -> None:
    """Historic cutoffs predate the v2 snapshots -> Algo V1 fallback."""
    result = blend_predictions(
        legacy_total_score=-4.5,
        legacy_win_probability=58.0,
        league="nhl",
        cutoff_date="4-12-2017",
        home_abbr="bos",
        away_abbr="mtl",
    )
    assert result["blend_mode"] == "algo_v1"
    assert result["blend_layers"] == 1
    assert result["algorithm"] == "Algo_V1"
    assert result["legacy"] is not None
    assert result["legacy"]["algorithm"] == "Algo_V1"
    assert result["blended_home_win_probability"] == 58.0
    assert result.get("hockey_pred") is None
    assert result.get("power") is None


def test_blend_nhl_v2_single_layer_when_available() -> None:
    import web.blend_service as blend_module

    original = blend_module.run_hockey_pred_model
    try:
        blend_module.run_hockey_pred_model = lambda *_a, **_k: {
            "algorithm": "NHLGradientBoost",
            "model_version": "v2",
            "market_decorrelated": True,
            "market_decorrelation_source": "moneyline",
            "pre_decorrelation_home_win_probability": 60.0,
            "home_win_probability": 63.5,
            "away_win_probability": 36.5,
            "expected_home_goals": 3.4,
            "expected_away_goals": 2.6,
            "predicted_margin": 0.8,
            "home_goalie": "Starter A",
            "away_goalie": "Starter B",
        }
        result = blend_predictions(
            legacy_total_score=-4.5,
            legacy_win_probability=58.0,
            league="nhl",
            cutoff_date="1-15-2026",
            home_abbr="tor",
            away_abbr="bos",
            home_moneyline=-140,
            away_moneyline=120,
        )
    finally:
        blend_module.run_hockey_pred_model = original

    assert result["algorithm"] == "NHLGradientBoost"
    assert result["blend_mode"] == "nhl_v2"
    assert result["blend_layers"] == 1
    assert result["blended_home_win_probability"] == 63.5
    assert result["market_decorrelated"] is True
    assert result["pre_decorrelation_home_win_probability"] == 60.0
    assert result["hockey_pred"]["home_goalie"] == "Starter A"
    assert result["home_spread_margin"] == -0.8
    assert result.get("legacy") is None


def test_blend_nhl_puckcast_payload_falls_back_to_algo_v1() -> None:
    """Non-v2 hockey payloads (PuckCast) keep NHL on Algo V1 for the blend."""
    import web.blend_service as blend_module

    original = blend_module.run_hockey_pred_model
    try:
        blend_module.run_hockey_pred_model = lambda *_a, **_k: {
            "algorithm": "HockeyPuckCast",
            "home_win_probability": 61.0,
            "away_win_probability": 39.0,
        }
        result = blend_predictions(
            legacy_total_score=-4.5,
            legacy_win_probability=58.0,
            league="nhl",
            cutoff_date="1-15-2026",
            home_abbr="tor",
            away_abbr="bos",
        )
    finally:
        blend_module.run_hockey_pred_model = original

    assert result["blend_mode"] == "algo_v1"
    assert result["algorithm"] == "Algo_V1"


def test_blend_hockey_algo_v1_all_leagues() -> None:
    for league in ("nhl", "ncaah", "ncaawh"):
        result = blend_predictions(
            legacy_total_score=3.0,
            legacy_win_probability=55.0,
            league=league,
            cutoff_date="1-15-2025",
            home_abbr="bos",
            away_abbr="tor",
        )
        assert result["blend_mode"] == "algo_v1"
        assert result["algorithm"] == "Algo_V1"
        assert result["blended_home_win_probability"] == 45.0


def test_blend_nfl_three_way_when_football_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    nfl_v2_original = blend_module._run_nfl_v2
    try:
        blend_module._run_nfl_v2 = lambda *_a, **_k: None
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
        blend_module._run_nfl_v2 = nfl_v2_original


def test_blend_cfb_three_way_when_football_available() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    cfb_v2_original = blend_module._run_cfb_v2
    try:
        blend_module._run_cfb_v2 = lambda *_a, **_k: None
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
        blend_module._run_cfb_v2 = cfb_v2_original


def test_blend_nba_matrix_only_keeps_legacy_layers_for_ensemble() -> None:
    import web.blend_service as blend_module

    basketball_original = blend_module.run_basketball_pred_model
    nba_v2_original = blend_module._run_nba_v2
    try:
        # Force matrix fallback path (v2 artifacts may exist in the repo).
        blend_module._run_nba_v2 = lambda *_a, **_k: None
        blend_module.run_basketball_pred_model = lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "source": "soft-impute-svd",
            "home_win_probability": 62.0,
            "predicted_margin": 3.0,
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nba",
            cutoff_date="6-12-2026",
            home_abbr="bos",
            away_abbr="ny",
        )
        assert result["blend_mode"] == "basketball_matrix"
        assert result["blend_layers"] >= 1
        assert result["basketball_pred"] is not None
        assert result["blended_home_win_probability"] == 62.0
        assert result.get("legacy") is not None
        assert result["legacy"]["home_win_probability"] == 60.0
    finally:
        blend_module.run_basketball_pred_model = basketball_original
        blend_module._run_nba_v2 = nba_v2_original


def test_nba_v2_availability_shift_moves_win_pct_keeps_margin() -> None:
    """ESPN availability nudge applies after nba_v2 payload; margin head stays put."""
    import web.blend_service as blend_module
    from web.availability_signals import AvailabilitySnapshot

    nba_v2_original = blend_module._run_nba_v2
    try:
        blend_module._run_nba_v2 = lambda *_a, **_k: {
            "algorithm": "NBAGradientBoost v2",
            "model_version": "v2",
            "model_variant": "market",
            "home_win_probability": 58.0,
            "predicted_margin": 4.5,
        }
        with (
            patch(
                "web.availability_signals.fetch_availability_snapshot",
                return_value=AvailabilitySnapshot(
                    home_injuries=0,
                    away_injuries=3,
                    home_out=0,
                    away_out=2,
                    sources=["espn_injuries_away"],
                ),
            ),
            patch(
                "web.availability_signals.availability_home_prob_shift",
                return_value=0.7,
            ),
        ):
            result = blend_predictions(
                legacy_total_score=-55.0,
                legacy_win_probability=55.0,
                league="nba",
                cutoff_date="6-12-2026",
                home_abbr="bos",
                away_abbr="ny",
                home_espn_id="2",
                away_espn_id="18",
            )
        assert result["blend_mode"] == "nba_v2"
        assert result["availability"]["shift_pp"] == 0.7
        assert result["blended_home_win_probability"] == 58.7
        assert result["win_probability"] == 58.7
        # Nested sport pred must track the board nudge (not stay at raw 58.0).
        assert result["basketball_pred"]["home_win_probability"] == 58.7
        assert result["home_spread_margin"] == -4.5
        assert blended_home_spread_margin(result, "nba") == -4.5
    finally:
        blend_module._run_nba_v2 = nba_v2_original


def test_availability_layer_syncs_away_win_probability() -> None:
    """Availability nudge must keep board + nested away% as 100 − home%."""
    from unittest.mock import patch

    from web.availability_signals import AvailabilitySnapshot
    from web.blend_service import _apply_availability_layer

    result = {
        "home_win_probability": 60.0,
        "away_win_probability": 40.0,
        "hockey_pred": {
            "home_win_probability": 60.0,
            "away_win_probability": 40.0,
        },
    }
    with (
        patch(
            "web.availability_signals.fetch_availability_snapshot",
            return_value=AvailabilitySnapshot(
                home_injuries=0,
                away_injuries=2,
                home_out=0,
                away_out=1,
                sources=["espn_injuries_away"],
            ),
        ),
        patch(
            "web.availability_signals.availability_home_prob_shift",
            return_value=2.0,
        ),
    ):
        out = _apply_availability_layer(
            result,
            league="nhl",
            home_espn_id="1",
            away_espn_id="2",
        )
    assert out["home_win_probability"] == pytest.approx(62.0)
    assert out["away_win_probability"] == pytest.approx(38.0)
    assert out["hockey_pred"]["home_win_probability"] == pytest.approx(62.0)
    assert out["hockey_pred"]["away_win_probability"] == pytest.approx(38.0)


def test_model_agreement_nfl_three_layers_agree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 1.0,
            "away_power": -0.5,
            "home_win_probability": 62.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "home_win_probability": 64.0,
            "source": "nfelo",
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nfl",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(
            result,
            "nfl",
            market={"away_moneyline": 110, "home_moneyline": -110},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is True
        assert agreement["agreement_mode"] == "value"
        assert "home" in agreement["value_sides"]
        assert agreement["legacy_side"] == "home"
        assert agreement["power_side"] == "home"
        assert agreement["third_side"] == "home"
        assert agreement["third_source"] == "football_pred"
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_football_pred_model = football_original


def test_model_agreement_nfl_value_on_underdog_despite_favorite_disagreement() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": -0.5,
            "away_power": 1.0,
            "home_win_probability": 42.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "home_win_probability": 64.0,
            "source": "nfelo",
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nfl",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(
            result,
            "nfl",
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
        blend_module.run_football_pred_model = football_original


def test_model_agreement_nfl_three_layers_disagree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": -0.5,
            "away_power": 1.0,
            "home_win_probability": 42.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "home_win_probability": 64.0,
            "source": "nfelo",
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nfl",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(
            result,
            "nfl",
            market={"away_moneyline": 70, "home_moneyline": -110},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["value_sides"] == []
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_football_pred_model = football_original


def test_model_agreement_nfl_one_layer_lacks_shared_value() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 1.0,
            "away_power": -0.5,
            "home_win_probability": 62.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "home_win_probability": 64.0,
            "source": "nfelo",
        }
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nfl",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(
            result,
            "nfl",
            market={"away_moneyline": 200, "home_moneyline": -230},
        )
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert agreement["legacy_side"] == "away"
        assert agreement["power_side"] == "away"
        assert agreement["third_side"] is None
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_football_pred_model = football_original


def test_model_agreement_two_layer_fallback_not_agreed() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 1.0,
            "away_power": -0.5,
            "home_win_probability": 62.0,
            "param": 10.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: None
        result = blend_predictions(
            legacy_total_score=-60.0,
            legacy_win_probability=60.0,
            league="nfl",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(result, "nfl")
        assert agreement["required"] == 3
        assert agreement["agreed"] is False
        assert result["blend_layers"] == 2
    finally:
        blend_module.run_power_model = power_original
        blend_module.run_football_pred_model = football_original


def test_model_agreement_nfl_spread_three_layers_agree() -> None:
    import web.blend_service as blend_module

    power_original = blend_module.run_power_model
    football_original = blend_module.run_football_pred_model
    try:
        blend_module.run_power_model = lambda *_a, **_k: {
            "algorithm": "PowerRatings",
            "home_power": 5.0,
            "away_power": -2.0,
            "home_win_probability": 97.0,
            "param": 10.0,
            "predicted_margin": 8.0,
        }
        blend_module.run_football_pred_model = lambda *_a, **_k: {
            "algorithm": "NfeloElo",
            "home_win_probability": 96.0,
            "projected_spread": -8.0,
            "source": "nfelo",
        }
        result = blend_predictions(
            legacy_total_score=-99.0,
            legacy_win_probability=99.0,
            league="nfl",
            cutoff_date="4-16-2017",
            home_abbr="bos",
            away_abbr="mia",
        )
        agreement = compute_model_agreement(
            result,
            "nfl",
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
        blend_module.run_football_pred_model = football_original


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


def test_cached_home_spread_margin_not_flipped_when_favorite_crosses_50() -> None:
    """Hubáček/context may flip favorite_side; dedicated margin head must stay put."""
    blended = {
        "blend_mode": "basketball_matrix",
        "home_spread_margin": -4.0,
        "favorite_side": "away",  # win% crossed 50% after context nudge
        "total_score": 51.0,
        "win_probability": 51.0,
    }
    assert blended_home_spread_margin(blended, "nba") == -4.0
    mlb = {
        "blend_mode": "mlb_v2",
        "home_spread_margin": -1.2,
        "favorite_side": "away",
        "total_score": 50.5,
        "win_probability": 50.5,
    }
    assert blended_home_spread_margin(mlb, "mlb") == -1.2


def test_blended_home_spread_margin_fail_closed_without_score() -> None:
    """Empty blend must not invent pick'em 0.0 ATS edge."""
    assert blended_home_spread_margin({}, "nba") is None
    assert blended_home_spread_margin({"favorite_side": "neutral"}, "wnba") is None


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


def test_spread_value_helpers_skip_missing_juice() -> None:
    """Agreement must not invent −110 or TypeError when juice is absent."""
    from web.blend_service import _best_value_spread_side, _layer_has_spread_value_on_side

    # predicted_margin: positive = home favored (sport-model convention).
    layer = {"predicted_margin": 12.0, "home_win_probability": 72.0}
    assert (
        _layer_has_spread_value_on_side(layer, "nba", "home", -3.5, spread_odds=None)
        is False
    )
    assert (
        _best_value_spread_side(
            layer,
            "nba",
            -3.5,
            away_spread_odds=None,
            home_spread_odds=None,
        )
        is None
    )
    assert (
        _best_value_spread_side(
            layer,
            "nba",
            -3.5,
            away_spread_odds=None,
            home_spread_odds=-110,
        )
        == "home"
    )


def test_model_agreement_normalizes_float_string_juice() -> None:
    """Agreement market juice from JSON float-strings must match int juice."""
    blended = {
        "blend_layers": 3,
        "legacy": {"total_score": -70.0, "home_win_probability": 70.0},
        "power": {"home_win_probability": 68.0},
        "football_pred": {"home_win_probability": 72.0},
    }
    as_int = compute_model_agreement(
        blended,
        "nfl",
        market={
            "spread": -1.5,
            "away_spread_odds": -110,
            "home_spread_odds": -110,
        },
    )
    as_str = compute_model_agreement(
        blended,
        "nfl",
        market={
            "spread": -1.5,
            "away_spread_odds": "-110.0",
            "home_spread_odds": "-110.0",
        },
    )
    assert as_str["required"] == 3
    assert as_str["agreed"] == as_int["agreed"]
    assert as_str["value_sides"] == as_int["value_sides"]
    # Invalid garbage juice must not invent agreement.
    garbage = compute_model_agreement(
        blended,
        "nfl",
        market={"spread": -1.5, "away_spread_odds": 50, "home_spread_odds": 50},
    )
    assert garbage["agreed"] is False
    assert garbage["value_sides"] == []


def test_power_layer_margin_uses_win_prob_not_power_diff() -> None:
    """PowerRatings power_diff is rating units — must not become predicted_margin points."""
    from web.bet_advisor import model_home_margin
    from web.blend_service import _layer_home_margin

    power = {
        "algorithm": "PowerRatings",
        "power_diff": 10.0,
        "home_win_probability": 62.0,
    }
    # Mimic run_power_model output: no predicted_margin from power_diff.
    assert "predicted_margin" not in power
    got = _layer_home_margin(power, "nba")
    total, _ = home_win_prob_to_total_score(62.0)
    want = model_home_margin(total, "nba")
    assert got == pytest.approx(want)
    assert abs(got - (-10.0)) > 5  # must not treat power_diff as points


if __name__ == "__main__":
    test_total_score_to_home_win_prob()
    test_home_win_prob_to_total_score()
    test_blend_legacy_only_when_power_unavailable()
    test_blend_averages_home_win_probs()
    test_blend_with_power_integration_when_games_exist()
    test_blend_mlb_runcast_only_when_available()
    test_blend_mlb_runcast_unavailable_fallback()
    test_blend_basketball_matrix_only_when_available()
    test_blend_nba_matrix_only_keeps_legacy_layers_for_ensemble()
    test_nba_v2_availability_shift_moves_win_pct_keeps_margin()
    test_model_agreement_nfl_three_layers_agree()
    test_model_agreement_nfl_value_on_underdog_despite_favorite_disagreement()
    test_model_agreement_nfl_three_layers_disagree()
    test_model_agreement_nfl_one_layer_lacks_shared_value()
    test_model_agreement_two_layer_fallback_not_agreed()
    test_model_agreement_nfl_spread_three_layers_agree()
    test_model_agreement_nhl_single_model()
    print("test_blend_service.py: all tests passed")
