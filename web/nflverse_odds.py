"""Import nflverse closing odds into the local closing-odds cache."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from web.closing_odds_db import ODDS_DIR, normalize_team_key

NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def rows_from_nflverse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            if record.get("game_type") not in (None, "", "REG", "POST"):
                continue
            home = normalize_team_key("nfl", record.get("home_team", ""))
            away = normalize_team_key("nfl", record.get("away_team", ""))
            game_date = str(record.get("gameday") or "")[:10]
            if not home or not away or not game_date:
                continue
            spread = record.get("spread_line")
            rows.append(
                {
                    "date": game_date,
                    "home_key": home,
                    "away_key": away,
                    "home_close_ml": _parse_int(record.get("home_moneyline")),
                    "away_close_ml": _parse_int(record.get("away_moneyline")),
                    "home_close_spread": spread,
                    "away_close_spread": f"{-float(spread)}" if spread not in (None, "") else "",
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
