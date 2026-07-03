"""WNBA opening-line and steam signals."""

from __future__ import annotations

from web.cbb_opening import (
    STEAM_MARGIN_THRESHOLD,
    fetch_opening_spread_proxy,
    opening_steam_adjustment,
)

__all__ = [
    "STEAM_MARGIN_THRESHOLD",
    "fetch_opening_spread_proxy",
    "opening_steam_adjustment",
]
