"""XGBoost NBA models: home-margin regression and home-win probability.

Both models consume ``FEATURE_COLUMNS`` (market line included) so they learn
small, well-calibrated deviations from the market rather than pretending to
ignore the sharpest public signal. XGBoost handles missing values natively, so
occasional NaN market features are fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from web.nba_ml.config import FEATURE_COLUMNS

MARGIN_PARAMS = dict(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=6,
    reg_lambda=1.5,
    reg_alpha=0.0,
    objective="reg:squarederror",
    n_jobs=0,
    random_state=13,
)

WINPROB_PARAMS = dict(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=6,
    reg_lambda=1.5,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=0,
    random_state=13,
)

# Normal-approximation SD of NBA game margin around the model's point estimate;
# refined from training residuals and stored in metadata.
DEFAULT_MARGIN_SIGMA = 13.0


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(FEATURE_COLUMNS)].astype(float)


@dataclass
class NBAModels:
    margin: XGBRegressor
    winprob: XGBClassifier
    margin_sigma: float = DEFAULT_MARGIN_SIGMA

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray:
        return self.margin.predict(_matrix(frame))

    def predict_winprob(self, frame: pd.DataFrame) -> np.ndarray:
        return self.winprob.predict_proba(_matrix(frame))[:, 1]


def train_margin_model(frame: pd.DataFrame) -> tuple[XGBRegressor, float]:
    model = XGBRegressor(**MARGIN_PARAMS)
    x = _matrix(frame)
    y = frame["home_margin"].astype(float)
    model.fit(x, y)
    resid = y.to_numpy() - model.predict(x)
    sigma = float(np.std(resid))
    if not np.isfinite(sigma) or sigma <= 1.0:
        sigma = DEFAULT_MARGIN_SIGMA
    return model, sigma


def train_winprob_model(frame: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(**WINPROB_PARAMS)
    model.fit(_matrix(frame), frame["home_win"].astype(int))
    return model


def train_models(frame: pd.DataFrame) -> NBAModels:
    margin, sigma = train_margin_model(frame)
    winprob = train_winprob_model(frame)
    return NBAModels(margin=margin, winprob=winprob, margin_sigma=sigma)


def _normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(_erf)(z / np.sqrt(2.0)))


def _erf(x: float) -> float:
    # Abramowitz-Stegun 7.1.26 approximation (no scipy dependency at inference).
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return sign * y


def cover_probability(pred_margin, home_spread, sigma: float = DEFAULT_MARGIN_SIGMA):
    """P(home covers) = P(home_margin + home_spread > 0) under a normal model."""
    pred_margin = np.asarray(pred_margin, dtype=float)
    home_spread = np.asarray(home_spread, dtype=float)
    z = (pred_margin + home_spread) / max(sigma, 1e-6)
    return _normal_cdf(z)


def ensemble_home_prob(model_prob, market_prob, weight: float = 0.5):
    """Market-aware blend for a stable displayed probability."""
    model_prob = np.asarray(model_prob, dtype=float)
    market_prob = np.asarray(market_prob, dtype=float)
    blended = np.where(
        np.isnan(market_prob),
        model_prob,
        weight * model_prob + (1.0 - weight) * market_prob,
    )
    return blended
