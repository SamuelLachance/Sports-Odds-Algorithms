"""Shared open/close market + steam features (CBB/MLB pattern).

Missing opens never invent open=close. Callers always get finite floats with
``has_*`` flags. Safe for FEATURE_COLUMNS hard-gates (zero NaNs).
"""

from __future__ import annotations

import math
from typing import Any

STEAM_FEATURE_COLUMNS: tuple[str, ...] = (
    "has_market",
    "mkt_home_prob",
    "has_spread",
    "mkt_home_spread",
    "has_open_line",
    "spread_move",
    "ml_steam_pp",
    "has_steam",
    "n_books",
    "mkt_total",
    "has_total",
    "total_move",
    "steam_to_fav",
    "reverse_line_move",
    "chalk_steam",
)


def amer_implied(ml: float) -> float:
    if not math.isfinite(ml) or abs(ml) < 100:
        return 0.5
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def steam_features_from_game(game: dict[str, Any]) -> dict[str, float]:
    """Build steam/market features from a game dict (or odds row aliases)."""
    open_h = _opt_float(
        game.get("home_open_ml")
        if game.get("home_open_ml") is not None
        else game.get("home_ml_open")
    )
    open_a = _opt_float(
        game.get("away_open_ml")
        if game.get("away_open_ml") is not None
        else game.get("away_ml_open")
    )
    close_h = _opt_float(
        game.get("home_close_ml")
        if game.get("home_close_ml") is not None
        else game.get("home_ml")
    )
    close_a = _opt_float(
        game.get("away_close_ml")
        if game.get("away_close_ml") is not None
        else game.get("away_ml")
    )
    open_spread = _opt_float(
        game.get("home_open_spread")
        if game.get("home_open_spread") is not None
        else game.get("open_spread")
    )
    close_spread = _opt_float(
        game.get("home_close_spread")
        if game.get("home_close_spread") is not None
        else game.get("home_spread")
    )
    open_total = _opt_float(game.get("open_total"))
    close_total = _opt_float(
        game.get("close_total")
        if game.get("close_total") is not None
        else game.get("total_line")
    )
    n_books = _opt_float(game.get("n_books") if game.get("n_books") is not None else game.get("books"))
    n_books_f = 0.0 if n_books is None else float(n_books)

    has_open_line = 0.0
    has_steam = 0.0
    ml_steam_pp = 0.0
    spread_move = 0.0
    total_move = 0.0
    has_total = 0.0
    mkt_home_prob = 0.5
    has_market = 0.0
    mkt_home_spread = 0.0
    has_spread = 0.0
    steam_to_fav = 0.0
    reverse_line_move = 0.0
    chalk_steam = 0.0

    has_open_ml = open_h is not None and open_a is not None
    if has_open_ml or open_spread is not None or open_total is not None:
        has_open_line = 1.0

    if has_open_ml and close_h is not None and close_a is not None:
        open_ph = amer_implied(open_h)
        open_pa = amer_implied(open_a)
        close_ph = amer_implied(close_h)
        close_pa = amer_implied(close_a)
        open_tot = open_ph + open_pa
        close_tot = close_ph + close_pa
        open_p = open_ph / open_tot if open_tot > 0 else 0.5
        close_p = close_ph / close_tot if close_tot > 0 else 0.5
        ml_steam_pp = (close_p - open_p) * 100.0
        has_steam = 1.0

    if open_spread is not None and close_spread is not None:
        spread_move = close_spread - open_spread
        has_steam = 1.0
        has_open_line = 1.0

    if open_total is not None and close_total is not None:
        total_move = close_total - open_total
        has_open_line = 1.0

    if close_total is not None:
        has_total = 1.0

    if close_h is not None and close_a is not None:
        ch = amer_implied(close_h)
        ca = amer_implied(close_a)
        tot = ch + ca
        mkt_home_prob = ch / tot if tot > 0 else 0.5
        has_market = 1.0

    if close_spread is not None:
        mkt_home_spread = close_spread
        has_spread = 1.0

    if has_steam >= 1.0 and has_market >= 1.0:
        fav_is_home = mkt_home_prob >= 0.5
        if fav_is_home and ml_steam_pp > 0:
            steam_to_fav = 1.0
        elif (not fav_is_home) and ml_steam_pp < 0:
            steam_to_fav = 1.0
        if fav_is_home and ml_steam_pp < -0.5:
            reverse_line_move = 1.0
        elif (not fav_is_home) and ml_steam_pp > 0.5:
            reverse_line_move = 1.0
        chalk_steam = 1.0 if abs(ml_steam_pp) >= 2.0 and steam_to_fav >= 1.0 else 0.0

    return {
        "has_market": has_market,
        "mkt_home_prob": mkt_home_prob,
        "has_spread": has_spread,
        "mkt_home_spread": mkt_home_spread,
        "has_open_line": has_open_line,
        "spread_move": spread_move,
        "ml_steam_pp": ml_steam_pp,
        "has_steam": has_steam,
        "n_books": n_books_f,
        "mkt_total": 0.0 if close_total is None else close_total,
        "has_total": has_total,
        "total_move": total_move,
        "steam_to_fav": steam_to_fav,
        "reverse_line_move": reverse_line_move,
        "chalk_steam": chalk_steam,
    }


def soccer_steam_from_row(row: dict[str, Any]) -> dict[str, float]:
    """1X2 steam: open vs close home win% (decimal odds)."""
    from web.soccer_v2.data import devig_decimal

    out = {c: 0.0 for c in STEAM_FEATURE_COLUMNS}
    out["mkt_home_prob"] = 1.0 / 3.0
    oh = row.get("open_home")
    od = row.get("open_draw")
    oa = row.get("open_away")
    ch = row.get("close_home") if row.get("close_home") is not None else row.get("PSH")
    cd = row.get("close_draw") if row.get("close_draw") is not None else row.get("PSD")
    ca = row.get("close_away") if row.get("close_away") is not None else row.get("PSA")
    # football-data common aliases
    if ch is None:
        ch = row.get("B365H") or row.get("AvgH")
    if cd is None:
        cd = row.get("B365D") or row.get("AvgD")
    if ca is None:
        ca = row.get("B365A") or row.get("AvgA")

    open_trip = None
    close_trip = None
    try:
        if oh is not None and od is not None and oa is not None:
            open_trip = devig_decimal(float(oh), float(od), float(oa))
    except (TypeError, ValueError):
        open_trip = None
    try:
        if ch is not None and cd is not None and ca is not None:
            close_trip = devig_decimal(float(ch), float(cd), float(ca))
    except (TypeError, ValueError):
        close_trip = None

    if close_trip is not None:
        out["has_market"] = 1.0
        out["mkt_home_prob"] = float(close_trip[0])
    if open_trip is not None:
        out["has_open_line"] = 1.0
    if open_trip is not None and close_trip is not None:
        out["ml_steam_pp"] = (float(close_trip[0]) - float(open_trip[0])) * 100.0
        out["has_steam"] = 1.0
        fav_is_home = out["mkt_home_prob"] >= 0.5
        if fav_is_home and out["ml_steam_pp"] > 0:
            out["steam_to_fav"] = 1.0
        elif (not fav_is_home) and out["ml_steam_pp"] < 0:
            out["steam_to_fav"] = 1.0
        if fav_is_home and out["ml_steam_pp"] < -0.5:
            out["reverse_line_move"] = 1.0
        elif (not fav_is_home) and out["ml_steam_pp"] > 0.5:
            out["reverse_line_move"] = 1.0
        out["chalk_steam"] = (
            1.0 if abs(out["ml_steam_pp"]) >= 2.0 and out["steam_to_fav"] >= 1.0 else 0.0
        )
    return out
