"""Tests for CLV helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.clv_service import (  # noqa: E402
    american_to_implied_prob,
    clv_vs_market_pct,
    clv_vs_market_pct_threeway,
    devig_two_way,
)

import pytest  # noqa: E402


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


def test_clv_implied_prob_formula() -> None:
    # +141 vs close +120: pick implied lower than close → positive CLV.
    pick_p = american_to_implied_prob(141)
    close_p = american_to_implied_prob(120)
    assert pick_p is not None and close_p is not None
    expected = round((close_p - pick_p) / close_p * 100.0, 2)
    assert clv_vs_market_pct(141, 120) == expected


def test_clv_threeway_soccer_draw() -> None:
    clv = clv_vs_market_pct_threeway(
        250,
        side="draw",
        market_home=-120,
        market_draw=220,
        market_away=300,
    )
    assert clv == clv_vs_market_pct(250, 220)
    assert clv is not None and clv > 0


def test_clv_threeway_missing_side() -> None:
    assert (
        clv_vs_market_pct_threeway(
            150,
            side="draw",
            market_home=-110,
            market_draw=None,
            market_away=140,
        )
        is None
    )
def test_american_odds_zero_is_even_for_clv() -> None:
    """ESPN EVEN (0) must participate in CLV as +100, not drop as None."""
    assert american_to_implied_prob(0) == pytest.approx(0.5)
    assert clv_vs_market_pct(0, -110) is not None
    home, away = devig_two_way(0, -110)
    assert abs(home + away - 1.0) < 0.01
