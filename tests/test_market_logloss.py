"""Tests for closing-line sportsbook log-loss evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from web.ensemble_ml.config import SPORTSBOOK_LOGLOSS_BENCHMARKS, sportsbook_logloss_benchmark
from web.ensemble_ml.model import (
    BinaryEnsembleModel,
    _closing_market_logloss,
    evaluate_holdout_logloss,
)
from web.nba_ml.calibrate import log_loss


def test_sportsbook_benchmarks_include_cbb_nhl_mlb() -> None:
    assert sportsbook_logloss_benchmark("cbb") == 0.53720
    assert sportsbook_logloss_benchmark("wnba") == 0.58730
    assert sportsbook_logloss_benchmark("nhl") == 0.66040
    assert sportsbook_logloss_benchmark("mlb") == 0.67374
    assert sportsbook_logloss_benchmark("nba") == 0.60166
    assert set(SPORTSBOOK_LOGLOSS_BENCHMARKS) >= {"nba", "cbb", "wnba", "nhl", "mlb"}


def test_closing_market_logloss_uses_devigged_lines_only() -> None:
    frame = pd.DataFrame(
        {
            "game_date": [f"2024-01-{i:02d}" for i in range(1, 41)],
            "market_devig_home_prob": [60.0, 40.0] * 20,
            "home_win": [1, 0] * 20,
        }
    )
    ll, n = _closing_market_logloss(frame)
    probs = np.array([0.6, 0.4] * 20, dtype=float)
    expected = log_loss(probs, np.array([1, 0] * 20, dtype=float))
    assert n == 40
    assert ll == expected


def test_evaluate_holdout_does_not_emit_in_sample_model_market_score(
    tmp_path, monkeypatch
) -> None:
    import web.ensemble_ml.config as config
    import web.ensemble_ml.model as model_module
    from xgboost import XGBClassifier

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

    rng = np.random.default_rng(3)
    n = 120
    market = rng.uniform(35, 65, n)
    frame = pd.DataFrame(
        {
            "game_date": [f"2024-01-{i % 28 + 1:02d}" for i in range(n)],
            "sport_home_prob": market + rng.normal(0, 2, n),
            "meta_stacked_home_prob": market + rng.normal(0, 2, n),
            "market_devig_home_prob": market,
            "sport_minus_market": rng.normal(0, 1, n),
            "expected_home_goals": rng.uniform(2, 4, n),
            "expected_away_goals": rng.uniform(2, 4, n),
            "sport_xg_diff": rng.normal(0, 0.5, n),
            "overtime_probability": rng.uniform(0.05, 0.15, n),
            "power_home_prob": market + rng.normal(0, 2, n),
            "market_home_ml": -110,
            "market_away_ml": -110,
            "home_win": (market > 50).astype(int),
        }
    )

    win_model = XGBClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.2,
        objective="binary:logistic",
        n_jobs=1,
        random_state=1,
    )
    win_model.fit(frame.iloc[:90][list(config.NHL_STACKING_FEATURES)], frame.iloc[:90]["home_win"])

    model = BinaryEnsembleModel(
        win_model=win_model,
        margin_model=None,
        isotonic=None,
        margin_sigma=2.5,
        feature_columns=config.NHL_STACKING_FEATURES,
        spread_league=False,
        market_blend_weight=1.0,
        temperature=1.0,
    )
    metrics = evaluate_holdout_logloss(model, frame, league="nhl")
    assert "all_market_logloss" not in metrics
    assert metrics.get("closing_market_logloss") is not None
    assert metrics.get("beats_market") in (0.0, 1.0)
    assert metrics.get("sportsbook_benchmark_logloss") == 0.66040
