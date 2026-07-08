"""Tests for Soccer Path A market calibration blend."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_market_blend import blend_model_with_market  # noqa: E402
from web.soccer_meta_model import fit_market_blend_weight_grid, multiclass_log_loss  # noqa: E402


def test_blend_model_with_market_returns_normalized_probs() -> None:
    model = (40.0, 30.0, 30.0)
    market = (50.0, 25.0, 25.0)
    blended = blend_model_with_market(model, market, model_weight=0.5)
    assert abs(sum(blended) - 100.0) < 0.05
    assert blended[0] == 45.0


def test_market_blend_tuning_beats_pure_model_on_synthetic_sample() -> None:
    samples = [
        ((55.0, 25.0, 20.0), (48.0, 27.0, 25.0), 0),
        ((40.0, 28.0, 32.0), (42.0, 29.0, 29.0), 1),
        ((30.0, 30.0, 40.0), (33.0, 28.0, 39.0), 2),
    ] * 40
    model_only = sum(
        multiclass_log_loss(*model, outcome) for model, _market, outcome in samples
    ) / len(samples)
    weight = fit_market_blend_weight_grid(samples)
    blended_loss = 0.0
    for model, market, outcome in samples:
        blended = blend_model_with_market(model, market, model_weight=weight)
        blended_loss += multiclass_log_loss(*blended, outcome)
    blended_loss /= len(samples)
    assert blended_loss <= model_only + 0.01
