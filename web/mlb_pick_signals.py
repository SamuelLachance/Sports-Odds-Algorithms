"""MLB pick signal flags (subset strategies — not global LL beat claims)."""

from __future__ import annotations

from typing import Any

EARLY_SEASON_GAMES = 25
HIGH_CONFIDENCE_DISAGREEMENT = 1.25
DISAGREEMENT_RUNLINE_THRESHOLD = 0.75
BIG_FAVORITE_ML = -280


def build_mlb_pick_signals(
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
        "disagreement_runs": None,
        "high_confidence_disagreement": False,
        "early_season_signal": False,
        "big_favorite_dog": False,
        "steam_signal": bool((steam_meta or {}).get("steam_signal")),
        "props_futures_note": "Model F5/totals/props separately from daily ML.",
    }

    min_games = min(home_games, away_games)
    if min_games < EARLY_SEASON_GAMES:
        signals["early_season_signal"] = True

    if home_ml is not None and away_ml is not None:
        if home_ml <= BIG_FAVORITE_ML or away_ml <= BIG_FAVORITE_ML:
            signals["big_favorite_dog"] = True
            signals["underdog_side"] = "away" if home_ml <= BIG_FAVORITE_ML else "home"

    if market_spread is not None:
        model_spread = -model_margin
        disagreement = abs(model_spread - float(market_spread))
        signals["disagreement_runs"] = round(disagreement, 2)
        if disagreement >= DISAGREEMENT_RUNLINE_THRESHOLD:
            signals["disagreement_signal"] = True
        if disagreement >= HIGH_CONFIDENCE_DISAGREEMENT:
            signals["high_confidence_disagreement"] = True

    if steam_meta:
        signals["opening_steam"] = steam_meta

    return signals
