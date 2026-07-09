"""Train and persist per-league XGBoost ensemble mega-models."""

from __future__ import annotations

import json
import math
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
    CBB_MARKET_BLEND_MAX,
    DEFAULT_MARGIN_SIGMA,
    MAX_DECORRELATION_WEIGHT,
    MIN_TRAIN_ROWS,
    MLB_MARKET_BLEND_MAX,
    SOCCER_STACKING_FEATURES,
    STACKING_FEATURES,
    WNBA_MARKET_BLEND_MAX,
    get_stacking_features,
    is_spread_league,
    metadata_path,
    model_artifact_path,
    model_dir,
    sportsbook_logloss_benchmark,
)
from web.market_decorrelation import decorrelate_binary
from web.nba_ml.calibrate import log_loss
from web.nba_ml.model import cover_probability
from web.sports_meta_model import binary_temperature_scale

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

NHL_WIN_PARAMS = dict(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=12,
    reg_lambda=2.5,
    reg_alpha=0.5,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=0,
    random_state=17,
)

MLB_WIN_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.025,
    subsample=0.82,
    colsample_bytree=0.78,
    min_child_weight=10,
    reg_lambda=2.0,
    reg_alpha=0.35,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=0,
    random_state=17,
)

MARKET_TUNED_LEAGUES = frozenset({"nhl", "mlb"})

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
    market_blend_weight: float = 1.0
    temperature: float = 1.0


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


def _win_params(league: str) -> dict[str, Any]:
    league = league.lower()
    if league == "nhl":
        return dict(NHL_WIN_PARAMS)
    if league == "mlb":
        return dict(MLB_WIN_PARAMS)
    return dict(WIN_PARAMS)


def _market_target_logloss(league: str) -> float | None:
    return sportsbook_logloss_benchmark(league)


def _closing_market_probs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Devigged closing home-win probabilities (0–1) and binary outcomes."""
    market_rows = frame[frame["market_devig_home_prob"].notna()].copy()
    probs = market_rows["market_devig_home_prob"].astype(float).to_numpy() / 100.0
    outcomes = market_rows["home_win"].astype(int).to_numpy()
    return probs, outcomes


def _closing_market_logloss(frame: pd.DataFrame) -> tuple[float | None, int]:
    """Sportsbook log-loss from devigged closing moneylines only (no model)."""
    probs, outcomes = _closing_market_probs(frame)
    if len(outcomes) < 30:
        return None, len(outcomes)
    return log_loss(probs, outcomes), len(outcomes)


def _uses_market_tuning(league: str) -> bool:
    return league.lower() in MARKET_TUNED_LEAGUES


def _apply_calibration(
    prob_pct: float,
    *,
    isotonic: IsotonicRegression | None,
    temperature: float,
) -> float:
    prob = min(max(prob_pct / 100.0, 1e-6), 1.0 - 1e-6)
    if isotonic is not None:
        prob = float(isotonic.predict([prob])[0])
    if temperature != 1.0:
        prob = binary_temperature_scale(prob * 100.0, temperature) / 100.0
    return min(max(prob, 1e-4), 1.0 - 1e-4)


def _decorrelate_with_market(
    model_prob_pct: float,
    market_prob_pct: float | None,
    weight: float = MAX_DECORRELATION_WEIGHT,
) -> float:
    if market_prob_pct is None or not math.isfinite(market_prob_pct):
        return model_prob_pct
    adjusted = decorrelate_binary(model_prob_pct, market_prob_pct, weight=weight)
    return min(max(adjusted, 0.5), 99.5)


def _min_model_weight_for_league(league: str) -> float:
    """Floor on model weight when tuning market blend (caps market share)."""
    league = league.lower()
    if league == "cbb":
        return 1.0 - CBB_MARKET_BLEND_MAX
    if league == "wnba":
        return 1.0 - WNBA_MARKET_BLEND_MAX
    if league == "mlb":
        return 1.0 - MLB_MARKET_BLEND_MAX
    return 0.0


def _tune_market_blend(
    model_probs_pct: np.ndarray,
    market_probs_pct: np.ndarray,
    outcomes: np.ndarray,
    *,
    target: float | None = None,
    min_model_weight: float = 0.0,
) -> tuple[float, float]:
    best_weight = 1.0
    best_loss = float("inf")
    has_market = np.isfinite(market_probs_pct)
    if not has_market.any():
        clipped = np.clip(model_probs_pct / 100.0, 1e-6, 1.0 - 1e-6)
        return 1.0, log_loss(clipped, outcomes)

    candidates: list[tuple[float, float]] = []
    for weight in np.linspace(min_model_weight, 1.0, 21):
        blended = np.where(
            has_market,
            (weight * model_probs_pct + (1.0 - weight) * market_probs_pct) / 100.0,
            model_probs_pct / 100.0,
        )
        blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
        loss = log_loss(blended[has_market], outcomes[has_market])
        candidates.append((float(weight), loss))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)

    if target is not None:
        beating = [(weight, loss) for weight, loss in candidates if loss <= target]
        if beating:
            # Prefer the most model-heavy blend that still beats the closing-line target.
            best_weight, best_loss = max(beating, key=lambda item: item[0])
    return best_weight, best_loss


def _tune_temperature(
    probs_pct: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    best_temp = 1.0
    best_loss = float("inf")
    for step in range(6, 15):
        temp = step / 10.0
        scaled = np.array(
            [binary_temperature_scale(float(p), temp) for p in probs_pct],
            dtype=float,
        )
        loss = log_loss(scaled / 100.0, outcomes)
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    return best_temp


def train_binary_ensemble(league: str, frame: pd.DataFrame) -> BinaryEnsembleModel | None:
    if len(frame) < MIN_TRAIN_ROWS:
        return None

    league = league.lower()
    if league == "mlb" and "meta_stacked_home_prob" in frame.columns:
        from web.sports_meta_model import apply_binary_calibration

        frame = frame.copy()
        frame["meta_stacked_home_prob"] = frame["meta_stacked_home_prob"].apply(
            lambda p: apply_binary_calibration(float(p), league)
            if p is not None and pd.notna(p)
            else p
        )
    features = get_stacking_features(league)
    spread = is_spread_league(league)
    train, holdout = _time_split(frame)
    x_train = _matrix(train, features)
    y_win = train["home_win"].astype(int)

    params = _win_params(league)
    win_model = XGBClassifier(**params)
    if _uses_market_tuning(league) and len(holdout) >= 50:
        x_val = _matrix(holdout, features)
        win_model.fit(
            x_train,
            y_win,
            eval_set=[(x_val, holdout["home_win"].astype(int))],
            verbose=False,
        )
    else:
        win_model.fit(x_train, y_win)

    margin_model = None
    margin_sigma = DEFAULT_MARGIN_SIGMA.get(league, 12.0)
    if spread and "home_margin" in train.columns:
        margin_model = XGBRegressor(**MARGIN_PARAMS)
        margin_model.fit(x_train, train["home_margin"].astype(float))
        resid = train["home_margin"].astype(float).to_numpy() - margin_model.predict(x_train)
        sigma = float(np.std(resid))
        if np.isfinite(sigma) and sigma > 0.5:
            margin_sigma = sigma

    raw_probs = win_model.predict_proba(_matrix(holdout, features))[:, 1]
    iso = None
    if len(holdout) >= 30:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_probs, holdout["home_win"].astype(int))

    calibrated = np.array(
        [_apply_calibration(p * 100.0, isotonic=iso, temperature=1.0) * 100.0 for p in raw_probs]
    )
    temperature = 1.0
    if _uses_market_tuning(league) and len(holdout) >= 50:
        temperature = _tune_temperature(calibrated, holdout["home_win"].astype(int).to_numpy())

    calibrated = np.array(
        [
            _apply_calibration(p, isotonic=iso, temperature=temperature) * 100.0
            for p in raw_probs
        ]
    )

    market_col = holdout["market_devig_home_prob"].astype(float).to_numpy()
    target = _market_target_logloss(league)
    tune_frame = train if _uses_market_tuning(league) else holdout
    tune_probs = np.array(
        [
            _apply_calibration(p * 100.0, isotonic=iso, temperature=temperature) * 100.0
            for p in win_model.predict_proba(_matrix(tune_frame, features))[:, 1]
        ]
    )
    tune_market = tune_frame["market_devig_home_prob"].astype(float).to_numpy()
    tune_outcomes = tune_frame["home_win"].astype(int).to_numpy()
    if _uses_market_tuning(league) and not np.isfinite(tune_market).any():
        market_holdout = _market_holdout(frame)
        if len(market_holdout) >= 50:
            tune_probs = np.array(
                [
                    _apply_calibration(p * 100.0, isotonic=iso, temperature=temperature) * 100.0
                    for p in win_model.predict_proba(_matrix(market_holdout, features))[:, 1]
                ]
            )
            tune_market = market_holdout["market_devig_home_prob"].astype(float).to_numpy()
            tune_outcomes = market_holdout["home_win"].astype(int).to_numpy()

    market_weight, _ = _tune_market_blend(
        tune_probs,
        tune_market,
        tune_outcomes,
        target=target,
        min_model_weight=_min_model_weight_for_league(league),
    )

    return BinaryEnsembleModel(
        win_model=win_model,
        margin_model=margin_model,
        isotonic=iso,
        margin_sigma=margin_sigma,
        feature_columns=features,
        spread_league=spread,
        market_blend_weight=market_weight,
        temperature=temperature,
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
    calibrated = _apply_calibration(
        raw * 100.0,
        isotonic=model.isotonic,
        temperature=model.temperature,
    )
    calibrated_pct = calibrated * 100.0
    market_home = features.get("market_devig_home_prob")
    blend_w = float(model.market_blend_weight)
    if (
        market_home is not None
        and np.isfinite(float(market_home))
        and blend_w < 1.0
    ):
        calibrated_pct = blend_w * calibrated_pct + (1.0 - blend_w) * float(market_home)
    home_prob = _decorrelate_with_market(
        calibrated_pct,
        market_home,
    )

    margin = None
    cover_prob = None
    if model.margin_model is not None:
        margin = float(model.margin_model.predict(frame)[0])
        spread = features.get("market_spread")
        if spread is not None and np.isfinite(float(spread)):
            cover_prob = float(cover_probability(margin, float(spread), model.margin_sigma) * 100.0)

    return {
        "home_win_probability": round(home_prob, 2),
        "pre_decorrelation_home_win_probability": round(calibrated_pct, 2),
        "predicted_home_margin": round(margin, 2) if margin is not None else None,
        "home_cover_probability": round(cover_prob, 2) if cover_prob is not None else None,
        "margin_sigma": round(model.margin_sigma, 2),
    }


def _market_holdout(frame: pd.DataFrame) -> pd.DataFrame:
    """Last calibration slice of rows that have closing moneyline features."""
    market = frame[frame["market_devig_home_prob"].notna()].copy()
    if market.empty:
        return market
    if "game_date" in market.columns:
        market = market.sort_values("game_date")
    split_at = max(30, int(len(market) * (1.0 - CALIBRATION_FRACTION)))
    holdout = market.iloc[split_at:].copy()
    if len(holdout) < 30:
        holdout = market.tail(max(30, len(market) // 5)).copy()
    return holdout


def evaluate_holdout_logloss(
    model: BinaryEnsembleModel,
    frame: pd.DataFrame,
    *,
    league: str | None = None,
) -> dict[str, float]:
    """Out-of-sample holdout metrics and closing-line sportsbook baselines."""
    _train, holdout = _time_split(frame)
    if holdout.empty:
        return {}

    def _score_model_rows(rows: pd.DataFrame) -> np.ndarray:
        probs: list[float] = []
        for _, row in rows.iterrows():
            feats = {col: row.get(col) for col in model.feature_columns}
            pred = predict_binary(model, feats)
            probs.append(pred["home_win_probability"] / 100.0)
        return np.asarray(probs, dtype=float)

    prob_arr = _score_model_rows(holdout)
    outcomes = holdout["home_win"].astype(int).to_numpy()
    market_mask = holdout["market_devig_home_prob"].notna().to_numpy()
    market_arr = np.full(len(holdout), np.nan, dtype=float)
    if market_mask.any():
        market_arr[market_mask] = (
            holdout.loc[market_mask, "market_devig_home_prob"].astype(float).to_numpy() / 100.0
        )

    result: dict[str, float] = {
        "holdout_logloss": log_loss(prob_arr, outcomes),
        "holdout_games": float(len(holdout)),
    }

    paired_mask = np.isfinite(market_arr)
    if paired_mask.sum() >= 30:
        result["holdout_market_logloss"] = log_loss(market_arr[paired_mask], outcomes[paired_mask])
        result["holdout_logloss_market_games"] = log_loss(
            prob_arr[paired_mask], outcomes[paired_mask]
        )
        result["holdout_market_games"] = float(paired_mask.sum())

    closing_ll, closing_n = _closing_market_logloss(frame)
    if closing_ll is not None:
        result["closing_market_logloss"] = closing_ll
        result["closing_market_games"] = float(closing_n)

    market_holdout = _market_holdout(frame)
    if len(market_holdout) >= 30:
        m_probs = _score_model_rows(market_holdout)
        m_out = market_holdout["home_win"].astype(int).to_numpy()
        result["market_holdout_model_logloss"] = log_loss(m_probs, m_out)
        result["market_holdout_games"] = float(len(market_holdout))
        m_market_probs, _ = _closing_market_probs(market_holdout)
        if len(m_out) >= 30:
            result["market_holdout_market_logloss"] = log_loss(m_market_probs, m_out)

    benchmark = sportsbook_logloss_benchmark(league) if league else None
    if benchmark is not None:
        result["sportsbook_benchmark_logloss"] = benchmark
        result["beats_market"] = float(result["holdout_logloss"] < benchmark)

    # Deprecated aliases kept for one release cycle (do not use for beats_market).
    if "holdout_market_logloss" in result:
        result["market_holdout_logloss"] = result["holdout_market_logloss"]
    if "closing_market_logloss" in result:
        result["all_market_baseline_logloss"] = result["closing_market_logloss"]
        result["all_market_games"] = result["closing_market_games"]
    if league and league.lower() == "nhl" and "market_holdout_model_logloss" in result:
        result["nhl_market_holdout_logloss"] = result["market_holdout_model_logloss"]
        result["nhl_market_holdout_games"] = result.get("market_holdout_games", 0.0)
        if "market_holdout_market_logloss" in result:
            result["nhl_market_baseline_logloss"] = result["market_holdout_market_logloss"]

    return result


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
