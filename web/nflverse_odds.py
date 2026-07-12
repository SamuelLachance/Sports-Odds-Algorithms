"""Import nflverse closing odds into the local closing-odds cache."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from web.closing_odds_db import ODDS_DIR, normalize_team_key

NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def _parse_int(value: Any) -> int | None:
    """Parse American odds; EVEN ``0`` → ``+100``, reject ``|x| < 100``.

    Parity with ``closing_odds_db._parse_int``: string EVEN/PK → +100; bool rejected.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    if text.upper() in {"EVEN", "PK"}:
        return 100
    try:
        parsed = float(text)
        if not math.isfinite(parsed):
            return None
        number = int(parsed)
    except (TypeError, ValueError, OverflowError):
        return None
    if number == 0:
        return 100
    if abs(number) < 100:
        return None
    return number


def rows_from_nflverse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            # nflverse uses WC/DIV/CON/SB (not a single POST) for playoffs.
            if record.get("game_type") not in (
                None,
                "",
                "REG",
                "POST",
                "WC",
                "DIV",
                "CON",
                "SB",
            ):
                continue
            home = normalize_team_key("nfl", record.get("home_team", ""))
            away = normalize_team_key("nfl", record.get("away_team", ""))
            game_date = str(record.get("gameday") or "")[:10]
            if not home or not away or not game_date:
                continue
            spread = record.get("spread_line")
            # nflverse spread_line = expected home margin (positive = home favored).
            # Closing-odds cache uses betting convention (negative = home favored).
            try:
                spread_num = float(str(spread).strip()) if spread not in (None, "") else None
            except (TypeError, ValueError):
                spread_num = None
            if spread_num is not None and not math.isfinite(spread_num):
                spread_num = None
            # Juice/ML dumps (|x| ≥ 100) are never real NFL handicaps.
            if spread_num is not None and abs(spread_num) >= 100.0:
                spread_num = None
            home_close_spread = -spread_num if spread_num is not None else ""
            away_close_spread = spread_num if spread_num is not None else ""
            rows.append(
                {
                    "date": game_date,
                    "home_key": home,
                    "away_key": away,
                    "home_close_ml": _parse_int(record.get("home_moneyline")),
                    "away_close_ml": _parse_int(record.get("away_moneyline")),
                    "home_close_spread": home_close_spread,
                    "away_close_spread": away_close_spread,
                    "home_spread_odds": _parse_int(record.get("home_spread_odds")),
                    "away_spread_odds": _parse_int(record.get("away_spread_odds")),
                    "source": "nflverse",
                }
            )
    return rows


def write_nfl_cache(path: Path | None = None) -> int:
    source = ODDS_DIR / "nflverse_games.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing nflverse source file: {source}")
    rows = rows_from_nflverse_csv(source)
    out = path or (ODDS_DIR / "nfl.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "home_key",
        "away_key",
        "home_close_ml",
        "away_close_ml",
        "home_close_spread",
        "away_close_spread",
        "home_spread_odds",
        "away_spread_odds",
        "source",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
