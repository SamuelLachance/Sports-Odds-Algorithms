"""Unit tests for shared Hubáček market decorrelation."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_decorrelate_three_way_renormalizes() -> None:
    model = (55.0, 25.0, 20.0)
    market = (45.0, 30.0, 25.0)
    adjusted = decorrelate_three_way(model, market)
    assert adjusted[0] > model[0]
    assert abs(sum(adjusted) - 100.0) < 0.05


def test_decorrelate_from_spread() -> None:
    adjusted = decorrelate_from_spread(65.0, -5.0, weight=0.12)
    assert 1.0 <= adjusted <= 99.0
