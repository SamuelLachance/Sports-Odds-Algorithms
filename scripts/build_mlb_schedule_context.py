"""Build MLB game umpires + schedule weather from the public Stats API.

Hydrates each season schedule with officials + weather (one request per
gameType). Writes:
  data/supplemental/mlb-umpires/game_umpires.csv
  data/supplemental/mlb-weather/mlb_api_weather.csv

Weather from the schedule is game-day temp/wind reported by MLB — used to
backfill Open-Meteo gaps (has_weather=1 when temp is present). Humidity/precip
are left for Open-Meteo; when only MLB weather is available those fields use
neutral defaults with has_humidity=0 in the merged weather table.

Usage:
  python scripts/build_mlb_schedule_context.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_v2.statsapi_data import BASE_URL, get_json  # noqa: E402

UMP_OUT = PROJECT_ROOT / "data" / "supplemental" / "mlb-umpires" / "game_umpires.csv"
WX_OUT = PROJECT_ROOT / "data" / "supplemental" / "mlb-weather" / "mlb_api_weather.csv"

# Retractable / dome / fixed-roof venues.
ROOF_IDS: set[int] = {12, 14, 15, 32, 2395, 4169, 5325, 5150, 2397}

_GAME_TYPES = ("R", "F", "D", "L", "W")
_WIND_RE = re.compile(
    r"(?P<mph>\d+)\s*mph(?:,\s*(?P<dir>.+))?",
    re.IGNORECASE,
)


def _f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def _parse_wind(raw: str | None) -> tuple[float, str]:
    text = str(raw or "").strip()
    m = _WIND_RE.search(text)
    if not m:
        return 0.0, ""
    mph = float(m.group("mph"))
    return mph * 1.60934, (m.group("dir") or "").strip()


def _hp_ump_id(officials: list[dict[str, Any]] | None) -> int | None:
    for row in officials or []:
        if str(row.get("officialType") or "") == "Home Plate":
            oid = (row.get("official") or {}).get("id")
            if oid is not None:
                return int(oid)
    return None


def fetch_season(season: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ump_rows: list[dict[str, Any]] = []
    wx_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for game_type in _GAME_TYPES:
        url = (
            f"{BASE_URL}/schedule?sportId=1&season={season}&gameType={game_type}"
            "&hydrate=officials,weather"
        )
        try:
            payload = get_json(url)
        except OSError:
            if game_type == "R":
                raise
            continue
        for day in payload.get("dates") or []:
            for game in day.get("games") or []:
                pk = game.get("gamePk")
                if pk is None or int(pk) in seen:
                    continue
                seen.add(int(pk))
                day_iso = game.get("officialDate") or str(day.get("date") or "")
                venue_id = (game.get("venue") or {}).get("id")
                hp = _hp_ump_id(game.get("officials"))
                ump_rows.append(
                    {
                        "season": season,
                        "date": day_iso,
                        "game_pk": int(pk),
                        "venue_id": int(venue_id) if venue_id is not None else None,
                        "hp_ump_id": hp if hp is not None else 0,
                        "has_ump": 1.0 if hp is not None else 0.0,
                    }
                )
                weather = game.get("weather") or {}
                temp_raw = weather.get("temp")
                try:
                    temp_f = float(temp_raw) if temp_raw not in (None, "") else None
                except (TypeError, ValueError):
                    temp_f = None
                wind_kph, wind_dir = _parse_wind(weather.get("wind"))
                condition = str(weather.get("condition") or "")
                has_wx = 1.0 if temp_f is not None else 0.0
                is_roof = 1.0
                if venue_id is not None and int(venue_id) in ROOF_IDS:
                    is_roof = 1.0
                elif "dome" in condition.lower() or "roof" in condition.lower():
                    is_roof = 1.0
                else:
                    is_roof = 0.0
                wx_rows.append(
                    {
                        "season": season,
                        "date": day_iso,
                        "game_pk": int(pk),
                        "venue_id": int(venue_id) if venue_id is not None else None,
                        "temp_c": _f_to_c(temp_f) if temp_f is not None else 22.0,
                        "humidity": 50.0,
                        "precip_mm": 0.0,
                        "wind_kph": wind_kph,
                        "wind_dir_text": wind_dir,
                        "condition": condition,
                        "has_weather": has_wx,
                        "has_humidity": 0.0,
                        "is_roof": is_roof,
                        "source": "mlb_statsapi",
                    }
                )
    return ump_rows, wx_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()

    ump_all: list[dict[str, Any]] = []
    wx_all: list[dict[str, Any]] = []
    for season in range(args.start_season, args.end_season + 1):
        umps, wxs = fetch_season(season)
        ump_all.extend(umps)
        wx_all.extend(wxs)
        print(
            f"season {season}: umps={len(umps)} weather={sum(1 for r in wxs if r['has_weather']>0.5)}",
            flush=True,
        )

    ump_frame = pd.DataFrame(ump_all)
    wx_frame = pd.DataFrame(wx_all)
    UMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    WX_OUT.parent.mkdir(parents=True, exist_ok=True)
    ump_frame.to_csv(UMP_OUT, index=False)
    wx_frame.to_csv(WX_OUT, index=False)
    print(
        f"wrote {UMP_OUT} rows={len(ump_frame)} has_ump={ump_frame.has_ump.mean():.3f}",
        flush=True,
    )
    print(
        f"wrote {WX_OUT} rows={len(wx_frame)} has_weather={wx_frame.has_weather.mean():.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
