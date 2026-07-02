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

    frame = _synthetic_binary_frame()
    trained = train_binary_ensemble("nba", frame)
    assert trained is not None
    save_ensemble("nba", trained, {"league": "nba", "train_rows": len(frame)})
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
    save_ensemble("nba", trained, {"league": "nba"})
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
    save_ensemble("nba", trained, {"league": "nba"})
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
