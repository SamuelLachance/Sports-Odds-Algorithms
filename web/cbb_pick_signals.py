"""CBB pick signal flags (metadata only, not auto-picks)."""

from __future__ import annotations

from typing import Any

BIG_FAVORITE_ML = -800
BIG_FAVORITE_SPREAD = 20.0
DISAGREEMENT_SPREAD_THRESHOLD = 3.0


def build_cbb_pick_signals(
    *,
    model_margin: float,
    market_spread: float | None = None,
    home_ml: int | None = None,
    away_ml: int | None = None,
    steam_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Paul/Weinbach big-favorite dog and BetCommand-style disagreement flags.

    These are informational signals attached to the prediction payload — they do
    not auto-generate picks.
    """
    signals: dict[str, Any] = {
        "big_favorite_dog": False,
        "big_favorite_side": None,
        "disagreement_signal": False,
        "disagreement_pts": None,
        "steam_signal": bool((steam_meta or {}).get("steam_signal")),
    }

    favorite_side: str | None = None
    if home_ml is not None and away_ml is not None:
        if home_ml <= BIG_FAVORITE_ML:
            favorite_side = "home"
        elif away_ml <= BIG_FAVORITE_ML:
            favorite_side = "away"

    if market_spread is not None and abs(float(market_spread)) >= BIG_FAVORITE_SPREAD:
        favorite_side = "home" if float(market_spread) < 0 else "away"

    if favorite_side:
        signals["big_favorite_side"] = favorite_side
        signals["big_favorite_dog"] = True
        signals["underdog_signal"] = True

    if market_spread is not None:
        model_spread = -model_margin
        disagreement = abs(model_spread - float(market_spread))
        signals["disagreement_pts"] = round(disagreement, 2)
        if disagreement >= DISAGREEMENT_SPREAD_THRESHOLD:
            signals["disagreement_signal"] = True

    if steam_meta:
        signals["opening_steam"] = steam_meta

    return signals
