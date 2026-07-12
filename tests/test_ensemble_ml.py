"""Tests for per-sport ensemble ML mega-models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from web.ensemble_ml.config import STACKING_FEATURES, ensemble_model_available
from web.ensemble_ml.model import (
    BinaryEnsembleModel,
    predict_binary,
    save_ensemble,
    train_binary_ensemble,
)
from web.ensemble_ml.apply import apply_ensemble_ml
from web.ensemble_ml.predict import clear_ensemble_caches


def _synthetic_binary_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    legacy = rng.uniform(35, 65, n)
    power = legacy + rng.normal(0, 4, n)
    sport = power + rng.normal(0, 3, n)
    market = power + rng.normal(0, 2, n)
    margin = (power - 50) * 0.2 + rng.normal(0, 3, n)
    home_win = (power > 50).astype(int)
    spread = -margin + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "game_date": [f"2024-01-{i % 28 + 1:02d}" for i in range(n)],
            "legacy_home_prob": legacy,
            "power_home_prob": power,
            "sport_home_prob": sport,
            "meta_stacked_home_prob": (legacy + power + sport) / 3.0,
            "legacy_margin": -margin,
            "power_margin": -margin,
            "sport_margin": -margin,
            "market_devig_home_prob": market,
            "market_spread": spread,
            "market_home_ml": -110,
            "market_away_ml": -110,
            "home_margin": margin,
            "home_win": home_win,
            "home_cover": (margin + spread > 0).astype(int),
        }
    )


def test_train_and_predict_binary_ensemble(tmp_path, monkeypatch) -> None:
    import web.ensemble_ml.config as config
    import web.ensemble_ml.model as model_module

    monkeypatch.setattr(config, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(config, "model_dir", lambda league: tmp_path / league.lower())
    monkeypatch.setattr(
        config,
        "model_artifact_path",
        lambda league: tmp_path / league.lower() / "ensemble.joblib",
    )
    monkeypatch.setattr(
        config,
        "metadata_path",
        lambda league: tmp_path / league.lower() / "metadata.json",
    )
    monkeypatch.setattr(model_module, "model_dir", config.model_dir)
    monkeypatch.setattr(model_module, "model_artifact_path", config.model_artifact_path)
    monkeypatch.setattr(model_module, "metadata_path", config.metadata_path)
    # Repo ships NBA v2 artifacts, which supersede EnsembleML for NBA in
    # ensemble_model_available(). Force the gate off to test the ensemble itself.
    monkeypatch.setattr("web.nba_v2.live.artifacts_available", lambda: False)

    frame = _synthetic_binary_frame()
    trained = train_binary_ensemble("nba", frame)
    assert trained is not None
    save_ensemble(
        "nba",
        trained,
        {
            "league": "nba",
            "train_rows": len(frame),
            "holdout_logloss": 0.55,
            "market_holdout_logloss": 0.60,
        },
    )
    clear_ensemble_caches()

    assert ensemble_model_available("nba")

    features = {col: 55.0 for col in STACKING_FEATURES}
    features["market_spread"] = -5.5
    pred = predict_binary(trained, features)
    assert 0 < pred["home_win_probability"] < 100
    assert pred["predicted_home_margin"] is not None


def test_apply_ensemble_ml_margin_uses_spread_convention(tmp_path, monkeypatch) -> None:
    import web.ensemble_ml.config as config
    import web.ensemble_ml.model as model_module

    monkeypatch.setattr("web.nba_v2.live.artifacts_available", lambda: False)
    monkeypatch.setattr(config, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(config, "model_dir", lambda league: tmp_path / league.lower())
    monkeypatch.setattr(
        config,
        "model_artifact_path",
        lambda league: tmp_path / league.lower() / "ensemble.joblib",
    )
    monkeypatch.setattr(
        config,
        "metadata_path",
        lambda league: tmp_path / league.lower() / "metadata.json",
    )
    monkeypatch.setattr(model_module, "model_dir", config.model_dir)
    monkeypatch.setattr(model_module, "model_artifact_path", config.model_artifact_path)
    monkeypatch.setattr(model_module, "metadata_path", config.metadata_path)

    frame = _synthetic_binary_frame()
    trained = train_binary_ensemble("nba", frame)
    assert trained
    save_ensemble(
        "nba",
        trained,
        {
            "league": "nba",
            "holdout_logloss": 0.55,
            "market_holdout_logloss": 0.60,
        },
    )
    clear_ensemble_caches()

    blended = {
        "blend_mode": "blended",
        "legacy": {"home_win_probability": 58.0, "total_score": -58.0},
        "power": {"home_win_probability": 60.0},
        "basketball_pred": {"home_win_probability": 62.0, "predicted_margin": 4.0},
        "total_score": -59.0,
        "win_probability": 59.0,
    }
    updated = apply_ensemble_ml(
        blended,
        "nba",
        consensus_spread=-5.5,
        home_moneyline=-220,
        away_moneyline=180,
    )
    assert updated["blend_mode"] == "ensemble_ml"
    assert updated.get("ensemble_ml")
    margin = updated.get("home_spread_margin")
    assert margin is not None
    pred_margin = updated["ensemble_ml"]["predicted_home_margin"]
    assert pred_margin is not None
    assert margin == round(-float(pred_margin), 2)
    assert updated["favorite_side"] == "home"
    assert margin < 0


def test_apply_ensemble_ml_updates_blend(tmp_path, monkeypatch) -> None:
    import web.ensemble_ml.config as config
    import web.ensemble_ml.model as model_module

    monkeypatch.setattr("web.nba_v2.live.artifacts_available", lambda: False)
    monkeypatch.setattr(config, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(config, "model_dir", lambda league: tmp_path / league.lower())
    monkeypatch.setattr(
        config,
        "model_artifact_path",
        lambda league: tmp_path / league.lower() / "ensemble.joblib",
    )
    monkeypatch.setattr(
        config,
        "metadata_path",
        lambda league: tmp_path / league.lower() / "metadata.json",
    )
    monkeypatch.setattr(model_module, "model_dir", config.model_dir)
    monkeypatch.setattr(model_module, "model_artifact_path", config.model_artifact_path)
    monkeypatch.setattr(model_module, "metadata_path", config.metadata_path)

    frame = _synthetic_binary_frame()
    trained = train_binary_ensemble("nba", frame)
    assert trained
    save_ensemble(
        "nba",
        trained,
        {
            "league": "nba",
            "holdout_logloss": 0.55,
            "market_holdout_logloss": 0.60,
        },
    )
    clear_ensemble_caches()

    blended = {
        "blend_mode": "blended",
        "legacy": {"home_win_probability": 58.0, "total_score": -58.0},
        "power": {"home_win_probability": 60.0},
        "basketball_pred": {"home_win_probability": 62.0, "predicted_margin": 4.0},
        "total_score": -59.0,
        "win_probability": 59.0,
    }
    updated = apply_ensemble_ml(
        blended,
        "nba",
        consensus_spread=-5.5,
        home_moneyline=-220,
        away_moneyline=180,
    )
    assert updated["blend_mode"] == "ensemble_ml"
    assert updated.get("ensemble_ml")
    assert updated.get("home_spread_margin") is not None


def test_basketball_blend_populates_ensemble_feature_layers(monkeypatch) -> None:
    import web.blend_service as blend_module
    from web.blend_service import blend_predictions
    from web.ensemble_ml.features import extract_binary_features

    # NBA v2 artifacts (shipped in the repo) route blend_predictions to a
    # v2-only payload without a legacy layer. Force the BasketballMatrix
    # fallback path with a deterministic sport payload so the ensemble
    # feature layers this test asserts on are populated.
    monkeypatch.setattr(blend_module, "_run_nba_v2", lambda *_a, **_k: None)
    monkeypatch.setattr(blend_module, "run_power_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        blend_module,
        "run_basketball_pred_model",
        lambda *_a, **_k: {
            "algorithm": "BasketballMatrix",
            "source": "soft-impute-svd",
            "home_win_probability": 62.0,
            "predicted_margin": 4.0,
        },
    )

    blended = blend_predictions(
        legacy_total_score=-58.0,
        legacy_win_probability=58.0,
        league="nba",
        cutoff_date="11-15-2024",
        home_abbr="BOS",
        away_abbr="LAL",
        home_moneyline=-150,
        away_moneyline=130,
    )
    features = extract_binary_features(
        blended,
        "nba",
        consensus_spread=-4.5,
        home_moneyline=-150,
        away_moneyline=130,
    )
    assert features["legacy_home_prob"] is not None
    assert features["sport_home_prob"] is not None
    assert all(
        features.get(col) is not None and not np.isnan(features[col])
        for col in ("legacy_home_prob", "sport_home_prob", "market_devig_home_prob")
        if features.get(col) is not None
    )


def test_optional_prob_does_not_coerce_missing_to_zero() -> None:
    """``payload.get(key) or 0`` used to poison soccer stacking with fake 0%."""
    from web.ensemble_ml.dataset import optional_prob

    payload = {"draw_probability": 28.0, "away_win_probability": 30.0}
    assert optional_prob(payload, "home_win_probability") is None
    assert optional_prob(payload, "draw_probability") == 28.0
    # Explicit 0.0 is allowed; only missing/invalid stay None.
    assert optional_prob({"home_win_probability": 0.0}, "home_win_probability") == 0.0
    assert float(payload.get("home_win_probability") or 0) == 0.0  # old bug


def test_apply_ensemble_ml_spread_only_does_not_flag_decorrelated(monkeypatch) -> None:
    """Spread alone does not decorrelate binary ensemble win probs."""
    monkeypatch.setattr(
        "web.ensemble_ml.apply.ensemble_model_available", lambda _league: True
    )
    monkeypatch.setattr(
        "web.ensemble_ml.apply.predict_ensemble_for_blend",
        lambda *_a, **_k: {
            "home_win_probability": 58.0,
            "pre_decorrelation_home_win_probability": 58.0,
            "predicted_home_margin": 3.0,
        },
    )
    updated = apply_ensemble_ml(
        {"total_score": -55.0, "win_probability": 55.0},
        "nfl",
        consensus_spread=-3.5,
        home_moneyline=None,
        away_moneyline=None,
    )
    assert updated.get("market_decorrelated") is not True


def test_apply_ensemble_ml_invalid_ml_does_not_flag_decorrelated(monkeypatch) -> None:
    """Garbage juice must not set market_decorrelated (Hubáček short-circuit)."""
    monkeypatch.setattr(
        "web.ensemble_ml.apply.ensemble_model_available", lambda _league: True
    )
    monkeypatch.setattr(
        "web.ensemble_ml.apply.predict_ensemble_for_blend",
        lambda *_a, **_k: {
            "home_win_probability": 58.0,
            "pre_decorrelation_home_win_probability": 58.0,
            "predicted_home_margin": 3.0,
        },
    )
    updated = apply_ensemble_ml(
        {"total_score": -55.0, "win_probability": 55.0},
        "nfl",
        consensus_spread=-3.5,
        home_moneyline=-50,  # |odds| < 100
        away_moneyline=130,
    )
    assert updated.get("market_decorrelated") is not True


def test_apply_ensemble_ml_valid_ml_flags_decorrelated(monkeypatch) -> None:
    monkeypatch.setattr(
        "web.ensemble_ml.apply.ensemble_model_available", lambda _league: True
    )
    monkeypatch.setattr(
        "web.ensemble_ml.apply.predict_ensemble_for_blend",
        lambda *_a, **_k: {
            "home_win_probability": 59.0,
            "pre_decorrelation_home_win_probability": 58.0,
            "predicted_home_margin": 3.0,
        },
    )
    updated = apply_ensemble_ml(
        {"total_score": -55.0, "win_probability": 55.0},
        "nfl",
        home_moneyline=-150,
        away_moneyline=130,
    )
    assert updated.get("market_decorrelated") is True


def test_market_devig_soft_fails_invalid_and_even_text() -> None:
    """Ensemble feature de-vig must not raise on garbage / text EVEN juice."""
    from web.ensemble_ml.features import market_devig_home, market_devig_threeway

    assert market_devig_home(50, 130) is None
    assert market_devig_home("EVEN", 200) is not None
    assert market_devig_threeway(50, 250, 200) == (None, None, None)
    # Text EVEN must normalize like numeric 0, not raise via int("EVEN").
    text_even = market_devig_threeway("EVEN", 250, 200)
    assert text_even[0] is not None
    assert abs(sum(text_even) - 100.0) < 0.05
    even = market_devig_threeway(0, 250, 200)
    assert even[0] is not None
    assert abs(sum(even) - 100.0) < 0.05


def test_extract_binary_features_normalizes_even_zero_ml() -> None:
    """Raw market_*_ml must map ESPN EVEN (0) → +100 like training closes."""
    from web.ensemble_ml.features import extract_binary_features

    feats = extract_binary_features(
        {
            "legacy": {"home_win_probability": 55.0},
            "power": {"home_win_probability": 54.0},
        },
        "nba",
        home_moneyline=0,
        away_moneyline=130,
    )
    assert feats["market_home_ml"] == 100.0
    assert feats["market_away_ml"] == 130.0
    # De-vig path already normalized; keep it consistent with +100.
    assert feats["market_devig_home_prob"] is not None
    assert feats["market_devig_home_prob"] == pytest.approx(
        extract_binary_features(
            {
                "legacy": {"home_win_probability": 55.0},
                "power": {"home_win_probability": 54.0},
            },
            "nba",
            home_moneyline=100,
            away_moneyline=130,
        )["market_devig_home_prob"]
    )


def test_extract_soccer_features_ignores_incomplete_1x2_layers() -> None:
    """Partial legacy/sport triples must not reach blend_weighted_threeway."""
    from web.ensemble_ml.features import extract_soccer_features

    feats = extract_soccer_features(
        {
            "legacy_threeway": {
                "home_win_probability": 45.0,
                # missing draw / away — previously crashed meta blend
            },
            "soccer_pred": {
                "home_win_probability": 40.0,
                "draw_probability": 28.0,
                "away_win_probability": 32.0,
            },
        },
        "epl",
    )
    assert feats["legacy_home_prob"] == 45.0
    assert feats["legacy_draw_prob"] is None
    assert feats["meta_home_prob"] is not None
    assert feats["meta_draw_prob"] is not None
    assert abs(feats["meta_home_prob"] + feats["meta_draw_prob"] + feats["meta_away_prob"] - 100.0) < 0.05


def test_ensure_hubacek_still_runs_on_spread_only_ensemble() -> None:
    """Without market_decorrelated, ensemble blend_mode must not skip Hubáček."""
    from web.bet_advisor import (
        blend_outputs_are_market_decorrelated,
        ensure_hubacek_in_blend,
    )

    blended = {
        "blend_mode": "ensemble_ml",
        "ensemble_ml": {"home_win_probability": 60.0},
        "blended_home_win_probability": 60.0,
        "total_score": -60.0,
        "win_probability": 60.0,
    }
    assert not blend_outputs_are_market_decorrelated(blended)
    updated = ensure_hubacek_in_blend(
        blended,
        league="nfl",
        away_market=130,
        home_market=-150,
    )
    assert updated.get("market_decorrelated") is True
    assert updated["blended_home_win_probability"] != 60.0


def test_soccer_legacy_proxy_uses_total_score_not_inverted_linear_map() -> None:
    """Ensemble soccer rows must not map power_home via ``p*2-50`` (inverts favorites)."""
    from web.bet_advisor import soccer_threeway_probs
    from web.blend_service import home_win_prob_to_total_score
    from web.soccer_blend import power_threeway_probs

    power_home = 60.0
    power_tw = power_threeway_probs(power_home, "epl")
    legacy_total, _ = home_win_prob_to_total_score(power_home)
    legacy_tw = soccer_threeway_probs(legacy_total, "epl")
    inverted = soccer_threeway_probs(power_home * 2 - 50, "epl")

    assert legacy_total < 0  # Algo: ≤0 means home favorite
    assert legacy_tw[0] == pytest.approx(power_tw[0])
    assert inverted[0] < legacy_tw[0]  # old formula treated home as underdog
    assert inverted[0] < 50.0
    assert power_tw[0] > inverted[0]

def test_soccer_ensemble_pickem_power_proxy_not_away_blowout() -> None:
    """50% power home must not become home=0% via total_score=0 pick'em bug."""
    from web.soccer_blend import power_threeway_probs

    home, draw, away = power_threeway_probs(50.0, "epl")
    assert abs(home - away) < 1e-9
    assert home > 20.0
    assert home + draw + away == pytest.approx(100.0)

