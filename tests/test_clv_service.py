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
    probs = devig_two_way(-110, -110)
    assert probs is not None
    home, away = probs
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


def test_clv_threeway_side_is_case_insensitive() -> None:
    """Title-case sides must resolve to the same closing price as lowercase."""
    lower = clv_vs_market_pct_threeway(
        250,
        side="draw",
        market_home=-120,
        market_draw=220,
        market_away=300,
    )
    titled = clv_vs_market_pct_threeway(
        250,
        side="Draw",
        market_home=-120,
        market_draw=220,
        market_away=300,
    )
    assert lower is not None
    assert titled == lower


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


def test_clv_threeway_rejects_bool_odds() -> None:
    """int(False)==0 must not become EVEN CLV through the three-way path."""
    assert (
        clv_vs_market_pct_threeway(
            False,  # type: ignore[arg-type]
            side="draw",
            market_home=-120,
            market_draw=220,
            market_away=300,
        )
        is None
    )
    assert (
        clv_vs_market_pct_threeway(
            250,
            side="draw",
            market_home=-120,
            market_draw=False,  # type: ignore[arg-type]
            market_away=300,
        )
        is None
    )
def test_american_odds_zero_is_even_for_clv() -> None:
    """ESPN EVEN (0) must participate in CLV as +100, not drop as None."""
    assert american_to_implied_prob(0) == pytest.approx(0.5)
    assert clv_vs_market_pct(0, -110) is not None
    probs = devig_two_way(0, -110)
    assert probs is not None
    home, away = probs
    assert abs(home + away - 1.0) < 0.01


def test_american_to_implied_prob_rejects_bool() -> None:
    """bool is a subclass of int; False must not map to EVEN (+100 → 50%)."""
    assert american_to_implied_prob(False) is None
    assert american_to_implied_prob(True) is None
    assert clv_vs_market_pct(False, -110) is None  # type: ignore[arg-type]
    assert clv_vs_market_pct(150, False) is None  # type: ignore[arg-type]


def test_american_odds_rejects_invalid_magnitude() -> None:
    """|odds| < 100 (except ESPN 0→EVEN) must not price CLV."""
    assert american_to_implied_prob(50) is None
    assert american_to_implied_prob(-50) is None
    assert clv_vs_market_pct(50, -110) is None


def test_devig_two_way_fails_closed_on_invalid_odds() -> None:
    """Bad American prices must not invent a fake 50/50 market."""
    from web.clv_service import model_edge_vs_devigged_market

    assert devig_two_way(-50, 130) is None
    assert devig_two_way(50, -110) is None
    edge = model_edge_vs_devigged_market(60.0, -50, 130)
    assert edge["market_home_prob"] is None
    assert edge["model_minus_market_pp"] is None
