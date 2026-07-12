"""CBB opening-line and closing-spread steam signals."""

from __future__ import annotations

from typing import Any

from web.closing_odds_db import closing_odds_lookup

STEAM_MARGIN_THRESHOLD = 2.5


def _opening_spread_from_db(
    league: str,
    game_date: str,
    home_key: str,
    away_key: str,
) -> float | None:
    """Return true opening home spread when cached; never substitute close.

    Using close as an "opening" proxy made ``steam_move`` ~0 whenever live ≈
    close and hid real open→close line movement.
    """
    row = closing_odds_lookup(league, game_date, home_key, away_key)
    if not row:
        return None
    spread = row.get("home_open_spread")
    if spread is None:
        spread = row.get("away_open_spread")
        if spread is not None:
            spread = -float(spread)
    return float(spread) if spread is not None else None


def opening_steam_adjustment(
    model_margin: float,
    market_spread: float | None,
    *,
    opening_spread: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Detect steam when live market spread diverges from model margin.

    Returns adjusted margin and metadata dict. Spread convention: negative = home favored.
    Model margin: positive = home favored (points).
    """
    meta: dict[str, Any] = {
        "steam_signal": False,
        "steam_direction": None,
        "model_margin": round(model_margin, 2),
        "market_spread": market_spread,
        "opening_spread": opening_spread,
    }
    if market_spread is None:
        return model_margin, meta

    # Convert model margin to spread convention for comparison.
    model_spread = -model_margin
    diff = model_spread - float(market_spread)
    meta["spread_disagreement"] = round(abs(diff), 2)

    ref = opening_spread if opening_spread is not None else market_spread
    steam_move = float(market_spread) - float(ref)
    meta["steam_move"] = round(steam_move, 2)

    if abs(diff) >= STEAM_MARGIN_THRESHOLD:
        meta["steam_signal"] = True
        # diff < 0 ⇒ model more home-favored than the book ⇒ pull toward market
        # (away). steam_direction is the side we nudge toward for context floors.
        meta["steam_direction"] = "away" if diff < 0 else "home"
        # Nudge margin slightly toward market (conservative).
        nudge = min(abs(diff) * 0.15, 1.5)
        if diff < 0:
            model_margin -= nudge
        else:
            model_margin += nudge

    meta["adjusted_margin"] = round(model_margin, 2)
    return model_margin, meta


def fetch_opening_spread_proxy(
    league: str,
    game_date: str,
    home_key: str,
    away_key: str,
) -> float | None:
    """Opening home spread from the closing-odds DB when open columns exist."""
    return _opening_spread_from_db(league, game_date, home_key, away_key)
