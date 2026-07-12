"""Hubáček official pick policy unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hubacek_picks import (  # noqa: E402
    HUBACEK_MIN_EV_PCT,
    HUBACEK_MIN_MARKET_GAP_PP,
    HUBACEK_MIN_WIN_CONFIDENCE_PP,
    HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP,
    official_hubacek_thresholds,
    passes_hubacek_moneyline_gate,
    passes_hubacek_spread_gate,
    passes_hubacek_tracked_pick,
)


def test_official_thresholds_have_real_floors() -> None:
    thresholds = official_hubacek_thresholds()
    assert thresholds["pick_system"] == "hubacek"
    assert thresholds["min_ev_pct"] == HUBACEK_MIN_EV_PCT == 2.0
    assert thresholds["min_market_gap_pp"] == HUBACEK_MIN_MARKET_GAP_PP == 2.0
    assert thresholds["min_win_confidence_pp"] == HUBACEK_MIN_WIN_CONFIDENCE_PP == 20.0
    assert (
        thresholds["min_spread_confidence_pp"]
        == HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
        == 5.0
    )


def test_moneyline_gate_requires_gap_ev_and_phi_confidence() -> None:
    assert passes_hubacek_moneyline_gate(
        model_prob_pct=72.0,
        market_implied_pct=55.0,
        ev_pct=4.0,
    )
    # Confidence φ fails (|63 − 50| < 20 pp).
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=63.0,
        market_implied_pct=55.0,
        ev_pct=20.0,
    )
    # EV below the honest floor.
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=72.0,
        market_implied_pct=55.0,
        ev_pct=1.5,
    )
    # Gap below the 2 pp floor.
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=71.0,
        market_implied_pct=70.0,
        ev_pct=10.0,
    )
    # Confidence fails near coin-flip.
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=55.0,
        market_implied_pct=50.0,
        ev_pct=10.0,
    )


def test_spread_gate_requires_decorrelated_blend_and_cover_gap() -> None:
    blended = {"market_decorrelated": True, "blended_home_win_probability": 72.0}
    assert passes_hubacek_spread_gate(
        blended=blended,
        side="home",
        point_edge=2.0,
        side_cover_prob=72.0,
        spread_odds=-110,
        ev_pct=3.0,
        consensus_spread=-1.0,
    )
    # No decorrelation flag on the blend.
    assert not passes_hubacek_spread_gate(
        blended={"blended_home_win_probability": 72.0},
        side="home",
        point_edge=2.0,
        side_cover_prob=72.0,
        spread_odds=-110,
        ev_pct=3.0,
        consensus_spread=-3.5,
    )
    # Cover gap below the 2 pp floor (market cover at -110 is 52.4%).
    assert not passes_hubacek_spread_gate(
        blended=blended,
        side="home",
        point_edge=2.0,
        side_cover_prob=53.0,
        spread_odds=-110,
        ev_pct=3.0,
        consensus_spread=-1.0,
    )
    # EV below the honest floor.
    assert not passes_hubacek_spread_gate(
        blended=blended,
        side="home",
        point_edge=2.0,
        side_cover_prob=58.0,
        spread_odds=-110,
        ev_pct=1.0,
        consensus_spread=-1.0,
    )


def test_baseball_moneyline_gate_uses_lower_confidence_bar() -> None:
    from web.hubacek_picks import HUBACEK_BASEBALL_MIN_WIN_CONFIDENCE_PP

    assert HUBACEK_BASEBALL_MIN_WIN_CONFIDENCE_PP == 10.0
    assert passes_hubacek_moneyline_gate(
        model_prob_pct=62.0,
        market_implied_pct=55.0,
        ev_pct=4.0,
        min_win_confidence_pp=HUBACEK_BASEBALL_MIN_WIN_CONFIDENCE_PP,
    )
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=62.0,
        market_implied_pct=55.0,
        ev_pct=4.0,
        min_win_confidence_pp=20.0,
    )


def test_soccer_confidence_uses_lower_bar() -> None:
    from web.hubacek_picks import (
        HUBACEK_SOCCER_MIN_WIN_CONFIDENCE_PP,
        hubacek_min_win_confidence_pp,
    )

    assert HUBACEK_SOCCER_MIN_WIN_CONFIDENCE_PP == 10.0
    # Top-5 club leagues run the backtested soccer v2 policy: no confidence
    # bar (home-only + gap/EV floors carry the selection).
    assert hubacek_min_win_confidence_pp("epl") == 0.0
    # Internationals keep the Path A soccer default.
    assert hubacek_min_win_confidence_pp("worldcup") == 10.0
    assert hubacek_min_win_confidence_pp("nba") == 20.0


def test_tracked_pick_requires_hubacek_strategy_ev_and_confidence() -> None:
    assert passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 3.0,
            "win_probability": 72,
        }
    )
    # EV below the honest floor.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 1.0,
            "win_probability": 72,
        }
    )
    # Gap below the 2 pp floor.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 1.0,
            "ev_pct": 5.0,
            "win_probability": 72,
        }
    )
    # MLB uses the backtested 6.7 pp gap floor (no confidence bar).
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 3.0,
            "win_probability": 62,
            "league": "mlb",
        }
    )
    assert passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 7.0,
            "ev_pct": 3.0,
            "win_probability": 55,
            "league": "mlb",
        }
    )
    # MLB moneyline picks outside the [-200, +200] price window are rejected.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 7.0,
            "ev_pct": 3.0,
            "win_probability": 55,
            "market_odds": 240,
            "league": "mlb",
        }
    )
    # NBA spread uses the backtested 10 pp cover-gap floor (not the 2 pp ML floor).
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 3.0,
            "win_probability": 56,
            "bet_type": "spread",
            "league": "nba",
        }
    )
    assert passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 10.0,
            "ev_pct": 3.0,
            "win_probability": 56,
            "bet_type": "spread",
            "league": "nba",
        }
    )
    # Missing gap / win probability must fail closed (not skip those gates).
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "ev_pct": 5.0,
            "win_probability": 72,
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 5.0,
            "ev_pct": 5.0,
        }
    )
    # Soccer home-only leagues must reject away/draw on revalidation.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 5.0,
            "ev_pct": 6.0,
            "win_probability": 48.0,
            "side": "away",
            "bet_type": "soccer_1x2",
            "league": "epl",
            "market_odds": 150,
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "value",
            "model_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP + 5,
            "ev_pct": 30.0,
            "win_probability": 72,
        }
    )


def test_blend_is_decorrelated_reads_football_pred() -> None:
    from web.hubacek_picks import _blend_is_decorrelated

    assert _blend_is_decorrelated(
        {"football_pred": {"market_decorrelated": True, "home_win_probability": 58.0}}
    )
    assert not _blend_is_decorrelated(
        {"football_pred": {"market_decorrelated": False, "home_win_probability": 58.0}}
    )


if __name__ == "__main__":
    test_official_thresholds_have_real_floors()
    test_moneyline_gate_requires_gap_ev_and_phi_confidence()
    test_spread_gate_requires_decorrelated_blend_and_cover_gap()
    test_baseball_moneyline_gate_uses_lower_confidence_bar()
    test_soccer_confidence_uses_lower_bar()
    test_tracked_pick_requires_hubacek_strategy_ev_and_confidence()
    print("test_hubacek_picks.py: all tests passed")
