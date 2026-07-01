"""Tests for soccer log-loss meta calibration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_meta_model import (  # noqa: E402
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    apply_threeway_calibration,
    blend_weighted_threeway,
    fit_stat_weights_grid,
    fit_temperature_grid,
    multiclass_log_loss,
    normalize_threeway,
    stack_soccer_blend_layers,
    temperature_scale,
)


def test_normalize_threeway_sums_to_100() -> None:
    home, draw, away = normalize_threeway(40.0, 30.0, 30.0)
    assert abs(home + draw + away - 100.0) < 0.01


def test_temperature_scale_softens_confident_predictions() -> None:
    hot = temperature_scale(70.0, 20.0, 10.0, 1.0)
    soft = temperature_scale(70.0, 20.0, 10.0, 1.3)
    assert soft[0] < hot[0]
    assert soft[1] > hot[1]


def test_fit_stat_weights_prefers_informative_layer() -> None:
    samples = []
    for _ in range(30):
        samples.append(((80.0, 15.0, 5.0), (40.0, 30.0, 30.0), (10.0, 20.0, 70.0), OUTCOME_HOME))
    for _ in range(10):
        samples.append(((10.0, 20.0, 70.0), (40.0, 30.0, 30.0), (80.0, 15.0, 5.0), OUTCOME_AWAY))
    weights = fit_stat_weights_grid(samples)
    assert weights["elo"] > weights["dc"]


def test_fit_temperature_reduces_overconfidence_loss() -> None:
    samples = []
    for _ in range(40):
        samples.append((82.0, 12.0, 6.0, OUTCOME_HOME))
    for _ in range(20):
        samples.append((82.0, 12.0, 6.0, OUTCOME_DRAW))
    baseline = sum(multiclass_log_loss(*row[:3], row[3]) for row in samples) / len(samples)
    temperature = fit_temperature_grid(samples)
    tuned = sum(
        multiclass_log_loss(*temperature_scale(*row[:3], temperature), row[3])
        for row in samples
    ) / len(samples)
    assert tuned <= baseline


def test_stack_soccer_blend_layers_favors_soccer_pred_weight() -> None:
    legacy = (55.0, 25.0, 20.0)
    power = (50.0, 28.0, 22.0)
    soccer = (70.0, 18.0, 12.0)
    blended = stack_soccer_blend_layers(
        legacy=legacy,
        power=power,
        soccer_pred=soccer,
        league="epl",
    )
    assert blended is not None
    assert blended[0] > 60.0


def test_apply_threeway_calibration_returns_rounded_probs() -> None:
    home, draw, away = apply_threeway_calibration(45.0, 28.0, 27.0, "epl")
    assert abs(home + draw + away - 100.0) < 0.05
