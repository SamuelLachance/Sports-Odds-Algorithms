"""Train and persist per-league XGBoost ensemble mega-models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier, XGBRegressor

from web.ensemble_ml.config import (
    CALIBRATION_FRACTION,
    DEFAULT_MARGIN_SIGMA,
    MIN_TRAIN_ROWS,
    SOCCER_STACKING_FEATURES,
    STACKING_FEATURES,
    is_spread_league,
    metadata_path,
    model_artifact_path,
    model_dir,
)
from web.nba_ml.model import cover_probability

WIN_PARAMS = dict(
    n_estimators=280,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=1.2,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=0,
    random_state=17,
)

MARGIN_PARAMS = dict(
    n_estimators=280,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=1.2,
    objective="reg:squarederror",
    n_jobs=0,
    random_state=17,
)


@dataclass
class BinaryEnsembleModel:
    win_model: XGBClassifier
    margin_model: XGBRegressor | None
    isotonic: IsotonicRegression | None
    margin_sigma: float
    feature_columns: tuple[str, ...]
    spread_league: bool


@dataclass
class SoccerEnsembleModel:
    home_model: XGBClassifier
    draw_model: XGBClassifier
    away_model: XGBClassifier
    feature_columns: tuple[str, ...]


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame[list(columns)].astype(float)


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "game_date" in frame.columns:
        ordered = frame.sort_values("game_date")
    else:
        ordered = frame
    split_at = max(MIN_TRAIN_ROWS, int(len(ordered) * (1.0 - CALIBRATION_FRACTION)))
    train = ordered.iloc[:split_at].copy()
    holdout = ordered.iloc[split_at:].copy()
    if holdout.empty:
        holdout = train.tail(max(20, len(train) // 5)).copy()
        train = ordered.iloc[: len(ordered) - len(holdout)].copy()
    return train, holdout


def train_binary_ensemble(league: str, frame: pd.DataFrame) -> BinaryEnsembleModel | None:
    if len(frame) < MIN_TRAIN_ROWS:
        return None

    spread = is_spread_league(league)
    train, holdout = _time_split(frame)
    x_train = _matrix(train, STACKING_FEATURES)
    y_win = train["home_win"].astype(int)

    win_model = XGBClassifier(**WIN_PARAMS)
    win_model.fit(x_train, y_win)

    margin_model = None
    margin_sigma = DEFAULT_MARGIN_SIGMA.get(league.lower(), 12.0)
    if spread and "home_margin" in train.columns:
        margin_model = XGBRegressor(**MARGIN_PARAMS)
        margin_model.fit(x_train, train["home_margin"].astype(float))
        resid = train["home_margin"].astype(float).to_numpy() - margin_model.predict(x_train)
        sigma = float(np.std(resid))
        if np.isfinite(sigma) and sigma > 0.5:
            margin_sigma = sigma

    raw_probs = win_model.predict_proba(_matrix(holdout, STACKING_FEATURES))[:, 1]
    iso = None
    if len(holdout) >= 30:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_probs, holdout["home_win"].astype(int))

    return BinaryEnsembleModel(
        win_model=win_model,
        margin_model=margin_model,
        isotonic=iso,
        margin_sigma=margin_sigma,
        feature_columns=STACKING_FEATURES,
        spread_league=spread,
    )


def train_soccer_ensemble(league: str, frame: pd.DataFrame) -> SoccerEnsembleModel | None:
    if len(frame) < MIN_TRAIN_ROWS:
        return None

    train, _holdout = _time_split(frame)
    x_train = _matrix(train, SOCCER_STACKING_FEATURES)

    home_model = XGBClassifier(**WIN_PARAMS)
    draw_model = XGBClassifier(**WIN_PARAMS)
    away_model = XGBClassifier(**WIN_PARAMS)
    home_model.fit(x_train, train["home_win"].astype(int))
    draw_model.fit(x_train, train["draw"].astype(int))
    away_model.fit(x_train, train["away_win"].astype(int))

    return SoccerEnsembleModel(
        home_model=home_model,
        draw_model=draw_model,
        away_model=away_model,
        feature_columns=SOCCER_STACKING_FEATURES,
    )


def _normalize_soccer_probs(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = max(home + draw + away, 1e-6)
    return home / total * 100.0, draw / total * 100.0, away / total * 100.0


def predict_binary(
    model: BinaryEnsembleModel,
    features: dict[str, float | None],
) -> dict[str, float]:
    frame = pd.DataFrame([{col: features.get(col) for col in model.feature_columns}]).astype(float)
    raw = float(model.win_model.predict_proba(frame)[0, 1])
    if model.isotonic is not None:
        raw = float(model.isotonic.predict([raw])[0])
    home_prob = min(max(raw * 100.0, 0.5), 99.5)

    margin = None
    cover_prob = None
    if model.margin_model is not None:
        margin = float(model.margin_model.predict(frame)[0])
        spread = features.get("market_spread")
        if spread is not None and np.isfinite(float(spread)):
            cover_prob = float(cover_probability(margin, float(spread), model.margin_sigma) * 100.0)

    return {
        "home_win_probability": round(home_prob, 2),
        "predicted_home_margin": round(margin, 2) if margin is not None else None,
        "home_cover_probability": round(cover_prob, 2) if cover_prob is not None else None,
        "margin_sigma": round(model.margin_sigma, 2),
    }


def predict_soccer(
    model: SoccerEnsembleModel,
    features: dict[str, float | None],
) -> dict[str, float]:
    frame = pd.DataFrame([{col: features.get(col) for col in model.feature_columns}]).astype(float)
    home = float(model.home_model.predict_proba(frame)[0, 1])
    draw = float(model.draw_model.predict_proba(frame)[0, 1])
    away = float(model.away_model.predict_proba(frame)[0, 1])
    home_p, draw_p, away_p = _normalize_soccer_probs(home, draw, away)
    return {
        "home_win_probability": round(home_p, 2),
        "draw_probability": round(draw_p, 2),
        "away_win_probability": round(away_p, 2),
    }


def save_ensemble(league: str, payload: Any, metadata: dict[str, Any]) -> Path:
    league = league.lower()
    out_dir = model_dir(league)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_artifact_path(league)
    joblib.dump(payload, artifact)
    metadata_path(league).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return artifact


def load_binary_ensemble(league: str) -> BinaryEnsembleModel | None:
    path = model_artifact_path(league.lower())
    if not path.is_file():
        return None
    payload = joblib.load(path)
    if isinstance(payload, BinaryEnsembleModel):
        return payload
    return None


def load_soccer_ensemble(league: str) -> SoccerEnsembleModel | None:
    path = model_artifact_path(league.lower())
    if not path.is_file():
        return None
    payload = joblib.load(path)
    if isinstance(payload, SoccerEnsembleModel):
        return payload
    return None
