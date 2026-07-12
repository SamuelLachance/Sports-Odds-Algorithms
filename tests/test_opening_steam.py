"""Regression tests for opening-line steam margin nudges."""

from __future__ import annotations

import pytest

from web.cbb_opening import STEAM_MARGIN_THRESHOLD, opening_steam_adjustment


def test_opening_steam_nudges_toward_market_not_away() -> None:
    """Model home+7 vs market home−3 must shrink home margin toward the book."""
    model_margin = 7.0
    market_spread = -3.0  # book: home favored by 3
    # model_spread = -7 → diff = -4 (≥ threshold) → model too home-bullish.
    assert abs((-model_margin) - market_spread) >= STEAM_MARGIN_THRESHOLD

    adj, meta = opening_steam_adjustment(model_margin, market_spread)
    assert meta["steam_signal"] is True
    assert meta["steam_direction"] == "away"
    assert adj < model_margin
    assert adj == pytest.approx(model_margin - min(4.0 * 0.15, 1.5), abs=1e-9)


def test_opening_steam_nudges_toward_market_when_model_too_away() -> None:
    """Model home−4 vs market home−8 must increase home margin toward the book."""
    model_margin = -4.0  # away favored by 4 in margin space
    market_spread = -8.0  # book: home favored by 8
    # model_spread = +4 → diff = +12 → model too away-bullish.
    adj, meta = opening_steam_adjustment(model_margin, market_spread)
    assert meta["steam_signal"] is True
    assert meta["steam_direction"] == "home"
    assert adj > model_margin


def test_soccer_implied_shift_soft_fails_invalid_odds() -> None:
    """|odds| < 100 must return None, not raise into slate steam meta."""
    from web.soccer_opening import _implied_shift_pp, soccer_opening_steam_meta

    assert _implied_shift_pp(50, -110) is None
    assert _implied_shift_pp(-110, 75) is None
    assert _implied_shift_pp(0, -110) is not None  # ESPN EVEN → +100

    meta = soccer_opening_steam_meta(
        home_ml=-120,
        draw_ml=250,
        away_ml=320,
        open_home_ml=50,  # invalid open — shift skipped, no crash
        open_draw_ml=250,
        open_away_ml=320,
        model_home=48.0,
        model_draw=28.0,
        model_away=24.0,
    )
    assert meta["steam_signal"] is False
    assert meta["implied_shifts_pp"]["home"] is None
