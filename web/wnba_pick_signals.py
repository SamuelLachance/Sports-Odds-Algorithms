"""WNBA pick signal flags (metadata only — subset strategies, not global LL claims)."""

from __future__ import annotations

from typing import Any

EARLY_SEASON_GAMES = 12
HIGH_CONFIDENCE_DISAGREEMENT = 4.0
DISAGREEMENT_SPREAD_THRESHOLD = 3.0
BIG_FAVORITE_ML = -650


def build_wnba_pick_signals(
    *,
    model_margin: float,
    market_spread: float | None = None,
    home_ml: int | None = None,
    away_ml: int | None = None,
    home_games: int = 0,
    away_games: int = 0,
    steam_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signals: dict[str, Any] = {
        "disagreement_signal": False,
        "disagreement_pts": None,
        "high_confidence_disagreement": False,
        "early_season_signal": False,
        "big_favorite_dog": False,
        "steam_signal": bool((steam_meta or {}).get("steam_signal")),
        "props_futures_note": "Use edge-wnba-style pipeline separately for props/futures.",
    }

    min_games = min(home_games, away_games)
    if min_games < EARLY_SEASON_GAMES:
        signals["early_season_signal"] = True

    if home_ml is not None and away_ml is not None:
        if home_ml <= BIG_FAVORITE_ML or away_ml <= BIG_FAVORITE_ML:
            signals["big_favorite_dog"] = True
            signals["underdog_side"] = "away" if home_ml <= BIG_FAVORITE_ML else "home"

    # bool is a subclass of int: False→0.0 would invent pick'em disagreement.
    if market_spread is not None and not isinstance(market_spread, bool):
        model_spread = -model_margin
        disagreement = abs(model_spread - float(market_spread))
        signals["disagreement_pts"] = round(disagreement, 2)
        if disagreement >= DISAGREEMENT_SPREAD_THRESHOLD:
            signals["disagreement_signal"] = True
        if disagreement >= HIGH_CONFIDENCE_DISAGREEMENT:
            signals["high_confidence_disagreement"] = True

    if steam_meta:
        signals["opening_steam"] = steam_meta

    return signals
