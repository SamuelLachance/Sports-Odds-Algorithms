"""Unit tests for shared Hubáček market decorrelation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.market_decorrelation import (  # noqa: E402
    decorrelate_binary,
    decorrelate_from_spread,
    decorrelate_three_way,
)


def test_decorrelate_binary_fraction_scale() -> None:
    adjusted = decorrelate_binary(0.70, 0.55, weight=0.12)
    assert adjusted > 0.70
    assert adjusted > 0.55


def test_decorrelate_binary_percent_scale() -> None:
    adjusted = decorrelate_binary(70.0, 55.0, weight=0.12)
    assert adjusted > 70.0
    assert adjusted > 55.0


def test_decorrelate_binary_treats_one_point_zero_as_one_percent() -> None:
    """market_p=1.0 on the 0–100 scale is 1%, not a 100% fraction."""
    from web.market_decorrelation import _as_fraction

    frac, was_pct = _as_fraction(1.0)
    assert was_pct is True
    assert frac == pytest.approx(0.01)
    # Model 50% vs true 1% market → push UP (away from market).
    up = decorrelate_binary(50.0, 1.0, weight=0.12)
    assert up > 50.0
    # Nearby percent 1.5% should move in the same direction.
    near = decorrelate_binary(50.0, 1.5, weight=0.12)
    assert near > 50.0
    assert abs(up - near) < 2.0


def test_decorrelate_three_way_renormalizes() -> None:
    model = (55.0, 25.0, 20.0)
    market = (45.0, 30.0, 25.0)
    adjusted = decorrelate_three_way(model, market)
    assert adjusted[0] > model[0]
    assert abs(sum(adjusted) - 100.0) < 0.05


def test_decorrelate_from_spread() -> None:
    adjusted = decorrelate_from_spread(65.0, -5.0, weight=0.12)
    assert 1.0 <= adjusted <= 99.0
