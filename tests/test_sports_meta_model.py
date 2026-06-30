"""Tests for binary sports meta calibration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.sports_meta_model import (  # noqa: E402
    apply_binary_calibration,
    binary_log_loss,
    binary_temperature_scale,
    fit_binary_blend_weights_grid,
    fit_binary_temperature_grid,
    stack_binary_blend_layers,
)


def test_binary_log_loss_bounds() -> None:
    assert binary_log_loss(80.0, True) < binary_log_loss(55.0, True)
    assert binary_log_loss(20.0, False) < binary_log_loss(45.0, False)


def test_temperature_softens_extreme_home_prob() -> None:
    sharp = binary_temperature_scale(85.0, 1.0)
    soft = binary_temperature_scale(85.0, 1.2)
    assert soft < sharp


def test_fit_blend_weights_favor_sport_layer() -> None:
    samples = []
    for _ in range(40):
        samples.append((52.0, 54.0, 72.0, True))
    for _ in range(20):
        samples.append((48.0, 46.0, 28.0, False))
    weights = fit_binary_blend_weights_grid(samples, two_layer=False)
    assert weights.get("sport_pred", 0.0) >= weights.get("power", 0.0)


def test_stack_binary_blend_layers() -> None:
    blended = stack_binary_blend_layers(
        legacy_home=55.0,
        power_home=52.0,
        sport_home=68.0,
        league="nba",
    )
    assert blended is not None
    assert 50.0 < blended < 75.0


def test_apply_binary_calibration_rounds() -> None:
    value = apply_binary_calibration(63.456, "nba")
    assert 0.0 < value < 100.0


def test_fit_temperature_reduces_overconfidence() -> None:
    samples = [(85.0, True)] * 30 + [(85.0, False)] * 15
    baseline = sum(binary_log_loss(h, w) for h, w in samples) / len(samples)
    temp = fit_binary_temperature_grid(samples)
    tuned = sum(
        binary_log_loss(binary_temperature_scale(h, temp), w) for h, w in samples
    ) / len(samples)
    assert tuned <= baseline
