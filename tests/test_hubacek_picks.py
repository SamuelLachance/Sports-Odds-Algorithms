"""Hubáček official pick policy unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hubacek_picks import (  # noqa: E402
    HUBACEK_MIN_MARKET_GAP_PP,
    official_hubacek_thresholds,
    passes_hubacek_moneyline_gate,
    passes_hubacek_spread_gate,
    passes_hubacek_tracked_pick,
)


def test_official_thresholds_use_gap_not_ev() -> None:
    thresholds = official_hubacek_thresholds()
    assert thresholds["pick_system"] == "hubacek"
    assert thresholds["min_ev_pct"] == 0.0
    assert thresholds["min_market_gap_pp"] == HUBACEK_MIN_MARKET_GAP_PP


def test_moneyline_gate_requires_big_gap_and_positive_ev() -> None:
    assert passes_hubacek_moneyline_gate(
        model_prob_pct=63.0,
        market_implied_pct=55.0,
        ev_pct=4.0,
    )
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=60.0,
        market_implied_pct=55.0,
        ev_pct=20.0,
    )
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=63.0,
        market_implied_pct=55.0,
        ev_pct=0.0,
    )


def test_spread_gate_requires_decorrelated_blend() -> None:
    blended = {"market_decorrelated": True, "blended_home_win_probability": 62.0}
    assert passes_hubacek_spread_gate(
        blended=blended,
        side="home",
        point_edge=2.0,
        side_cover_prob=60.0,
        spread_odds=-110,
        ev_pct=3.0,
        consensus_spread=-1.0,
    )
    assert not passes_hubacek_spread_gate(
        blended={"blended_home_win_probability": 62.0},
        side="home",
        point_edge=2.0,
        side_cover_prob=58.0,
        spread_odds=-110,
        ev_pct=3.0,
        consensus_spread=-3.5,
    )


def test_tracked_pick_requires_hubacek_strategy_and_gap() -> None:
    assert passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP,
            "ev_pct": 1.0,
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "value",
            "model_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP,
            "ev_pct": 30.0,
        }
    )
