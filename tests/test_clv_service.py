"""Tests for CLV helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.clv_service import (  # noqa: E402
    american_to_implied_prob,
    clv_vs_market_pct,
    devig_two_way,
)


def test_american_to_implied_prob() -> None:
    assert american_to_implied_prob(-110) is not None
    assert 0.5 < american_to_implied_prob(-110) < 0.53


def test_devig_two_way_sums_to_one() -> None:
    home, away = devig_two_way(-110, -110)
    assert abs(home + away - 1.0) < 0.01


def test_clv_positive_when_better_price() -> None:
    clv = clv_vs_market_pct(150, 130)
    assert clv is not None
    assert clv > 0
