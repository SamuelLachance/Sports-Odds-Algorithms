"""Probability calibration (isotonic) and calibration diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(probs: np.ndarray, outcomes: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=float))
    return iso


def apply_isotonic(iso: IsotonicRegression | None, probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if iso is None:
        return probs
    return np.clip(iso.predict(probs), 1e-4, 1.0 - 1e-4)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> list[dict]:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "pred_mean": round(float(probs[mask].mean()), 4),
                "actual_rate": round(float(outcomes[mask].mean()), 4),
            }
        )
    return rows
