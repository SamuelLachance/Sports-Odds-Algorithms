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
    assert thresholds.get("thresholds_scope") == "global_baseline"
    assert thresholds["min_ev_pct"] == HUBACEK_MIN_EV_PCT == 2.0
    assert thresholds["min_market_gap_pp"] == HUBACEK_MIN_MARKET_GAP_PP == 2.0
    assert thresholds["min_win_confidence_pp"] == HUBACEK_MIN_WIN_CONFIDENCE_PP == 20.0
    assert (
        thresholds["min_spread_confidence_pp"]
        == HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
        == 5.0
    )


def test_within_hubacek_ml_range_fails_closed_on_missing_odds() -> None:
    from web.hubacek_picks import within_hubacek_ml_range

    assert within_hubacek_ml_range("mlb", None) is False
    assert within_hubacek_ml_range("mlb", -110) is True
    assert within_hubacek_ml_range("mlb", 240) is False
    # Invalid American magnitudes must fail closed (not raw interval membership).
    assert within_hubacek_ml_range("mlb", 50) is False
    assert within_hubacek_ml_range("mlb", -75) is False
    # EVEN (0) normalizes to +100 and stays in-window for MLB.
    assert within_hubacek_ml_range("mlb", 0) is True
    # Missing/garbage juice fails closed even when no ml_lo/ml_hi window.
    assert within_hubacek_ml_range("nba", None) is False
    assert within_hubacek_ml_range(None, None) is False
    assert within_hubacek_ml_range("nba", -110) is True
    assert within_hubacek_ml_range("epl", 150) is True
    assert within_hubacek_ml_range("epl", 50) is False


def test_within_hubacek_ml_range_rejects_bool_false_as_even() -> None:
    """False must not coerce via float→0→EVEN +100 and pass the ML window."""
    from web.hubacek_picks import within_hubacek_ml_range

    assert within_hubacek_ml_range("nba", False) is False
    assert within_hubacek_ml_range("mlb", False) is False
    assert within_hubacek_ml_range("epl", False) is False
    assert within_hubacek_ml_range("nba", True) is False
    # Numeric ESPN EVEN still allowed.
    assert within_hubacek_ml_range("nba", 0) is True
    assert within_hubacek_ml_range("nba", "EVEN") is True


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


def test_spread_gate_treats_even_odds_zero_like_plus_100() -> None:
    """ESPN EVEN (0) must use 50% market cover, not 100%."""
    blended = {"market_decorrelated": True, "blended_home_win_probability": 58.0}
    kwargs: dict = {
        "blended": blended,
        "side": "home",
        "point_edge": 2.0,
        "side_cover_prob": 58.0,
        "ev_pct": 5.0,
        "consensus_spread": -3.0,
    }
    assert passes_hubacek_spread_gate(**kwargs, spread_odds=0) == passes_hubacek_spread_gate(
        **kwargs, spread_odds=100
    )
    assert passes_hubacek_spread_gate(**kwargs, spread_odds=0)


def test_spread_gate_fails_closed_on_invalid_american_odds() -> None:
    """Garbage juice must return False, not raise into the slate pipeline."""
    blended = {"market_decorrelated": True, "blended_home_win_probability": 72.0}
    kwargs = dict(
        blended=blended,
        side="home",
        point_edge=2.0,
        side_cover_prob=72.0,
        ev_pct=5.0,
        consensus_spread=-3.0,
    )
    assert passes_hubacek_spread_gate(**kwargs, spread_odds=50) is False
    assert passes_hubacek_spread_gate(**kwargs, spread_odds=-75) is False
    assert passes_hubacek_spread_gate(**kwargs, spread_odds=-110) is True


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
            "market_odds": -150,
        }
    )
    # Missing / invalid American juice fails closed (even with no league ML window).
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 3.0,
            "win_probability": 72,
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 5.0,
            "ev_pct": 6.0,
            "win_probability": 48.0,
            "side": "home",
            "bet_type": "soccer_1x2",
            "league": "epl",
            "market_odds": 50,
        }
    )
    # EV below the honest floor.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 1.0,
            "win_probability": 72,
            "market_odds": -150,
        }
    )
    # Gap below the 2 pp floor.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 1.0,
            "ev_pct": 5.0,
            "win_probability": 72,
            "market_odds": -150,
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
            "market_odds": -150,
        }
    )
    assert passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 7.0,
            "ev_pct": 3.0,
            "win_probability": 55,
            "market_odds": -150,
            "league": "mlb",
        }
    )
    # MLB ML window is a hard gate — missing odds fail closed.
    assert not passes_hubacek_tracked_pick(
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
    # NBA spread uses the backtested 12 pp cover-gap floor (not the 2 pp ML floor).
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 3.0,
            "ev_pct": 3.0,
            "win_probability": 56,
            "bet_type": "spread",
            "league": "nba",
            "spread_odds": -110,
        }
    )
    assert passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 12.0,
            "ev_pct": 3.0,
            "win_probability": 56,
            "bet_type": "spread",
            "league": "nba",
            "spread_odds": -110,
        }
    )
    # Spread without valid juice fails closed.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 10.0,
            "ev_pct": 3.0,
            "win_probability": 56,
            "bet_type": "spread",
            "league": "nba",
            "spread_odds": 50,
        }
    )
    # Missing gap / win probability must fail closed (not skip those gates).
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "ev_pct": 5.0,
            "win_probability": 72,
            "market_odds": -150,
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 5.0,
            "ev_pct": 5.0,
            "market_odds": -150,
        }
    )
    # Soccer home-only internationals must reject away/draw on revalidation.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 9.0,
            "ev_pct": 9.0,
            "win_probability": 48.0,
            "side": "away",
            "bet_type": "soccer_1x2",
            "league": "worldcup",
            "market_odds": 150,
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "value",
            "model_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP + 5,
            "ev_pct": 30.0,
            "win_probability": 72,
            "market_odds": -150,
        }
    )
    # Non-numeric EV / gap from corrupt JSON must fail closed, not TypeError.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": "nope",
            "ev_pct": 5.0,
            "win_probability": 72,
            "side": "home",
            "league": "mlb",
            "market_odds": 140,
        }
    )
    # NaN comparisons are always false — must fail closed, not approve.
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": float("nan"),
            "ev_pct": 5.0,
            "win_probability": 72.0,
            "market_odds": -150,
            "league": "nba",
        }
    )
    assert not passes_hubacek_tracked_pick(
        {
            "strategy": "hubacek",
            "model_market_gap_pp": 5.0,
            "ev_pct": float("nan"),
            "win_probability": 72.0,
            "market_odds": -150,
            "league": "nba",
        }
    )
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=72.0,
        market_implied_pct=55.0,
        ev_pct=float("nan"),
    )
    assert not passes_hubacek_moneyline_gate(
        model_prob_pct=float("nan"),
        market_implied_pct=55.0,
        ev_pct=5.0,
    )


def test_blend_is_decorrelated_reads_football_pred() -> None:
    from web.hubacek_picks import _blend_is_decorrelated

    assert _blend_is_decorrelated(
        {"football_pred": {"market_decorrelated": True, "home_win_probability": 58.0}}
    )
    assert not _blend_is_decorrelated(
        {"football_pred": {"market_decorrelated": False, "home_win_probability": 58.0}}
    )


def test_blend_is_decorrelated_ignores_ensemble_mode_without_flag() -> None:
    """Spread-only ensemble must not fake Hubáček decorrelation."""
    from web.hubacek_picks import _blend_is_decorrelated, passes_hubacek_spread_gate

    spread_only = {
        "blend_mode": "ensemble_ml",
        "ensemble_ml": {"home_win_probability": 58.0, "predicted_home_margin": 3.0},
    }
    assert not _blend_is_decorrelated(spread_only)
    assert not passes_hubacek_spread_gate(
        blended=spread_only,
        side="home",
        point_edge=2.0,
        side_cover_prob=58.0,
        spread_odds=-110,
        ev_pct=5.0,
        consensus_spread=-3.5,
    )
    flagged = {**spread_only, "market_decorrelated": True}
    assert _blend_is_decorrelated(flagged)


def test_nan_threshold_overrides_do_not_fail_open() -> None:
    """Corrupt NaN floors must fall back to defaults — not disable gap/EV gates."""
    from unittest.mock import patch

    from web.hubacek_picks import (
        hubacek_min_ev_pct,
        hubacek_min_market_gap_pp,
        hubacek_min_win_confidence_pp,
    )

    with patch(
        "web.hubacek_picks.league_pick_overrides",
        return_value={
            "min_ev_pct": float("nan"),
            "min_market_gap_pp": float("nan"),
            "min_win_confidence_pp": float("inf"),
        },
    ):
        assert hubacek_min_ev_pct("nba") == HUBACEK_MIN_EV_PCT
        assert hubacek_min_market_gap_pp("nba") == HUBACEK_MIN_MARKET_GAP_PP
        assert hubacek_min_win_confidence_pp("nba") == HUBACEK_MIN_WIN_CONFIDENCE_PP
        # Below-floor pick must still be rejected (was True when NaN floors failed open).
        assert not passes_hubacek_tracked_pick(
            {
                "strategy": "hubacek",
                "model_market_gap_pp": 0.1,
                "ev_pct": 0.1,
                "win_probability": 50.0,
                "market_odds": -150,
                "league": "nba",
            }
        )


def test_negative_threshold_overrides_do_not_fail_open() -> None:
    """Negative gap/EV/φ floors make every comparison pass — must fall back."""
    from unittest.mock import patch

    from web.hubacek_picks import (
        hubacek_min_ev_pct,
        hubacek_min_market_gap_pp,
        hubacek_min_win_confidence_pp,
        passes_hubacek_confidence,
    )

    with patch(
        "web.hubacek_picks.league_pick_overrides",
        return_value={
            "min_ev_pct": -5.0,
            "min_market_gap_pp": -10.0,
            "min_win_confidence_pp": -1.0,
        },
    ):
        assert hubacek_min_ev_pct("nba") == HUBACEK_MIN_EV_PCT
        assert hubacek_min_market_gap_pp("nba") == HUBACEK_MIN_MARKET_GAP_PP
        assert hubacek_min_win_confidence_pp("nba") == HUBACEK_MIN_WIN_CONFIDENCE_PP
        assert not passes_hubacek_tracked_pick(
            {
                "strategy": "hubacek",
                "model_market_gap_pp": 0.1,
                "ev_pct": 0.1,
                "win_probability": 50.0,
                "market_odds": -150,
                "league": "nba",
            }
        )
    assert not passes_hubacek_confidence(50.0, min_pp=-5.0)
    assert not passes_hubacek_confidence(72.0, min_pp=-1.0)


def test_swapped_ml_range_bounds_still_admit_prices_inside_window() -> None:
    """ml_lo > ml_hi must normalize, not reject every American price."""
    from unittest.mock import patch

    from web.hubacek_picks import hubacek_ml_range, within_hubacek_ml_range

    with patch(
        "web.hubacek_picks.league_pick_overrides",
        return_value={"ml_lo": 200.0, "ml_hi": -200.0},
    ):
        assert hubacek_ml_range("mlb") == (-200.0, 200.0)
        assert within_hubacek_ml_range("mlb", -110)


if __name__ == "__main__":
    test_official_thresholds_have_real_floors()
    test_moneyline_gate_requires_gap_ev_and_phi_confidence()
    test_spread_gate_requires_decorrelated_blend_and_cover_gap()
    test_baseball_moneyline_gate_uses_lower_confidence_bar()
    test_soccer_confidence_uses_lower_bar()
    test_tracked_pick_requires_hubacek_strategy_ev_and_confidence()
    print("test_hubacek_picks.py: all tests passed")
