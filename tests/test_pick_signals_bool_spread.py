"""Regression: bool False must not invent pick'em disagreement."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cbb_pick_signals import build_cbb_pick_signals  # noqa: E402
from web.mlb_pick_signals import build_mlb_pick_signals  # noqa: E402
from web.wnba_pick_signals import build_wnba_pick_signals  # noqa: E402


def test_mlb_false_market_spread_matches_missing() -> None:
    """False is not None and float(False)==0.0 — must not fire disagreement."""
    poisoned = build_mlb_pick_signals(model_margin=1.2, market_spread=False)
    missing = build_mlb_pick_signals(model_margin=1.2, market_spread=None)
    assert poisoned["disagreement_signal"] is False
    assert poisoned["disagreement_runs"] is None
    assert poisoned["disagreement_signal"] == missing["disagreement_signal"]
    assert poisoned["disagreement_runs"] == missing["disagreement_runs"]


def test_cbb_false_market_spread_matches_missing() -> None:
    poisoned = build_cbb_pick_signals(model_margin=5.0, market_spread=False)
    missing = build_cbb_pick_signals(model_margin=5.0, market_spread=None)
    assert poisoned["disagreement_signal"] is False
    assert poisoned["disagreement_pts"] is None
    assert poisoned["big_favorite_side"] is None
    assert poisoned["disagreement_signal"] == missing["disagreement_signal"]
    assert poisoned["disagreement_pts"] == missing["disagreement_pts"]


def test_wnba_false_market_spread_matches_missing() -> None:
    poisoned = build_wnba_pick_signals(model_margin=5.0, market_spread=False)
    missing = build_wnba_pick_signals(model_margin=5.0, market_spread=None)
    assert poisoned["disagreement_signal"] is False
    assert poisoned["disagreement_pts"] is None
    assert poisoned["high_confidence_disagreement"] is False
    assert poisoned["disagreement_signal"] == missing["disagreement_signal"]
    assert poisoned["disagreement_pts"] == missing["disagreement_pts"]


def test_real_zero_spread_still_computes_disagreement() -> None:
    """Genuine pick'em (0.0) must still be compared to the model margin."""
    mlb = build_mlb_pick_signals(model_margin=1.2, market_spread=0.0)
    assert mlb["disagreement_runs"] == 1.2
    assert mlb["disagreement_signal"] is True
    cbb = build_cbb_pick_signals(model_margin=5.0, market_spread=0.0)
    assert cbb["disagreement_pts"] == 5.0
    assert cbb["disagreement_signal"] is True
