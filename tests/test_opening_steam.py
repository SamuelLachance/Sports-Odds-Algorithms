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
