"""CBB probability calibration and market decorrelation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibratorKind = Literal["isotonic", "platt", "none"]


@dataclass
class CalibratorState:
    kind: CalibratorKind = "none"
    isotonic: IsotonicRegression | None = None
    platt: LogisticRegression | None = None


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def decorrelate_from_market(
    model_p: float,
    market_p: float,
    *,
    weight: float = 0.12,
) -> float:
    """Hubáček-style push away from market-correlated component."""
    model_p = min(max(model_p, 0.01), 0.99)
    market_p = min(max(market_p, 0.01), 0.99)
    adjusted = model_p - weight * (model_p - market_p)
    return min(max(adjusted, 0.01), 0.99)


def _fit_isotonic(probs: np.ndarray, outcomes: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs, outcomes)
    return iso


def _fit_platt(probs: np.ndarray, outcomes: np.ndarray) -> LogisticRegression | None:
    if len(probs) < 10:
        return None
    lr = LogisticRegression(max_iter=500)
    try:
        lr.fit(probs.reshape(-1, 1), outcomes)
        return lr
    except ValueError:
        return None


def fit_best_calibrator(
    raw_probs: list[float],
    outcomes: list[float],
) -> CalibratorState:
    """Walsh-style: pick isotonic or Platt with lowest walk-forward log-loss."""
    if len(raw_probs) < 15:
        return CalibratorState(kind="none")

    probs = np.clip(np.asarray(raw_probs, dtype=float) / 100.0, 0.01, 0.99)
    y = np.asarray(outcomes, dtype=float)
    candidates: list[tuple[float, CalibratorState]] = []

    iso = _fit_isotonic(probs, y)
    iso_preds = np.clip(iso.predict(probs), 1e-4, 1.0 - 1e-4)
    candidates.append((_log_loss(iso_preds, y), CalibratorState(kind="isotonic", isotonic=iso)))

    platt = _fit_platt(probs, y)
    if platt is not None:
        platt_preds = np.clip(platt.predict_proba(probs.reshape(-1, 1))[:, 1], 1e-4, 1.0 - 1e-4)
        candidates.append((_log_loss(platt_preds, y), CalibratorState(kind="platt", platt=platt)))

    candidates.append((_log_loss(probs, y), CalibratorState(kind="none")))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def apply_cbb_calibration(raw_prob: float, calibrator_state: CalibratorState | None) -> float:
    """Apply fitted calibrator; raw_prob is percentage 0-100."""
    if calibrator_state is None or calibrator_state.kind == "none":
        return round(raw_prob, 2)

    p = min(max(raw_prob / 100.0, 0.01), 0.99)
    if calibrator_state.kind == "isotonic" and calibrator_state.isotonic is not None:
        calibrated = float(calibrator_state.isotonic.predict([p])[0])
    elif calibrator_state.kind == "platt" and calibrator_state.platt is not None:
        calibrated = float(calibrator_state.platt.predict_proba([[p]])[0, 1])
    else:
        calibrated = p
    return round(min(max(calibrated * 100.0, 1.0), 99.0), 2)


def spread_to_home_prob(spread: float, sigma: float = 12.0) -> float:
    """Convert book spread (negative = home favored) to home win probability (%)."""
    # Spread sign: negative home favored -> positive margin for home.
    margin = -float(spread)
    return margin_to_home_win_prob_pct(margin, sigma)


def margin_to_home_win_prob_pct(margin: float, sigma: float = 12.0) -> float:
    if sigma <= 0:
        sigma = 12.0
    prob = 1.0 / (1.0 + math.exp(-margin / sigma))
    return round(prob * 100.0, 2)
