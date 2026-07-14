"""Build PIT-safe MLB weather features from Open-Meteo historical archive.

Public source: https://open-meteo.com (no API key). One row per (date, venue_id)
with afternoon (game-time proxy) temperature, humidity, precip, wind.

Usage:
  python scripts/build_mlb_weather.py
  python scripts/build_mlb_weather.py --start-season 2018 --end-season 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_v2.venues import VENUE_COORDS  # noqa: E402

CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "mlb-history"
OUT = PROJECT_ROOT / "data" / "supplemental" / "mlb-weather" / "game_weather.csv"
UA = {"User-Agent": "Sports-Odds-Algorithms/mlb-weather"}


# Retractable / dome / fixed-roof venues — outdoor weather less actionable.
ROOF_VENUE_IDS: set[int] = {
    12,  # Tropicana
    14,  # Rogers Centre
    15,  # Chase Field
    32,  # American Family Field
    2395,  # Oracle (partial)
    4169,  # loanDepot
    5325,  # Globe Life
    5150,  # Gocheok
    2397,  # Tokyo Dome
}


def _fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _afternoon_stats(hourly: dict[str, list]) -> dict[str, float]:
    """Use 16:00–20:00 local hours as a first-pitch weather proxy."""
    times = hourly.get("time") or []
    idxs = [i for i, t in enumerate(times) if 16 <= int(str(t)[11:13]) <= 20]
    if not idxs:
        idxs = list(range(len(times)))

    def _mean(key: str, default: float = 0.0) -> float:
        vals = hourly.get(key) or []
        picked = [float(vals[i]) for i in idxs if i < len(vals) and vals[i] is not None]
        return float(sum(picked) / len(picked)) if picked else default

    return {
        "temp_c": _mean("temperature_2m", 22.0),
        "humidity": _mean("relative_humidity_2m", 50.0),
        "precip_mm": _mean("precipitation", 0.0),
        "wind_kph": _mean("windspeed_10m", 10.0),
        "wind_dir": _mean("winddirection_10m", 0.0),
    }


def _venue_days(start: int, end: int) -> dict[int, list[str]]:
    by_venue: dict[int, set[str]] = {}
    for season in range(start, end + 1):
        path = CACHE_ROOT / str(season) / "games.json"
        if not path.is_file():
            continue
        games = json.loads(path.read_text(encoding="utf-8"))
        for game in games:
            if str(game.get("status") or "") not in {"F", "Final", "Completed"}:
                # Keep scheduled finals only; status F is used in our cache.
                if str(game.get("status") or "") != "F":
                    continue
            venue = game.get("venue_id")
            day = str(game.get("date") or "")[:10]
            if not venue or not day:
                continue
            by_venue.setdefault(int(venue), set()).add(day)
    return {vid: sorted(days) for vid, days in by_venue.items()}


def build(start: int, end: int) -> pd.DataFrame:
    venue_days = _venue_days(start, end)
    rows: list[dict[str, Any]] = []
    for venue_id, days in sorted(venue_days.items()):
        coords = VENUE_COORDS.get(venue_id)
        if not coords:
            for day in days:
                rows.append(
                    {
                        "date": day,
                        "venue_id": venue_id,
                        "temp_c": 22.0,
                        "humidity": 50.0,
                        "precip_mm": 0.0,
                        "wind_kph": 0.0,
                        "wind_dir": 0.0,
                        "has_weather": 0.0,
                        "is_roof": 1.0 if venue_id in ROOF_VENUE_IDS else 0.0,
                    }
                )
            continue
        lat, lon, _elev = coords
        # Fetch whole span for this venue (Open-Meteo allows multi-day ranges).
        day_min, day_max = days[0], days[-1]
        params = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "start_date": day_min,
                "end_date": day_max,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,winddirection_10m",
                "timezone": "auto",
            }
        )
        url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
        try:
            payload = _fetch_json(url)
            hourly = payload.get("hourly") or {}
            times = hourly.get("time") or []
            by_day: dict[str, dict[str, list]] = {}
            for i, ts in enumerate(times):
                day = str(ts)[:10]
                bucket = by_day.setdefault(
                    day,
                    {
                        "time": [],
                        "temperature_2m": [],
                        "relative_humidity_2m": [],
                        "precipitation": [],
                        "windspeed_10m": [],
                        "winddirection_10m": [],
                    },
                )
                bucket["time"].append(ts)
                for key in (
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "windspeed_10m",
                    "winddirection_10m",
                ):
                    vals = hourly.get(key) or []
                    bucket[key].append(vals[i] if i < len(vals) else None)
            wanted = set(days)
            for day, bucket in by_day.items():
                if day not in wanted:
                    continue
                stats = _afternoon_stats(bucket)
                rows.append(
                    {
                        "date": day,
                        "venue_id": venue_id,
                        **stats,
                        "has_weather": 1.0,
                        "is_roof": 1.0 if venue_id in ROOF_VENUE_IDS else 0.0,
                    }
                )
            # Days missing from API response → explicit has_weather=0 (no invented fills).
            got = {r["date"] for r in rows if r["venue_id"] == venue_id and r["has_weather"] > 0.5}
            for day in days:
                if day in got:
                    continue
                rows.append(
                    {
                        "date": day,
                        "venue_id": venue_id,
                        "temp_c": 22.0,
                        "humidity": 50.0,
                        "precip_mm": 0.0,
                        "wind_kph": 0.0,
                        "wind_dir": 0.0,
                        "has_weather": 0.0,
                        "is_roof": 1.0 if venue_id in ROOF_VENUE_IDS else 0.0,
                    }
                )
            print(f"venue {venue_id}: {len(days)} days ok", flush=True)
            time.sleep(0.8)
        except Exception as exc:  # noqa: BLE001
            print(f"venue {venue_id} FAIL {exc}; retrying once after 5s", flush=True)
            time.sleep(5.0)
            try:
                payload = _fetch_json(url)
                hourly = payload.get("hourly") or {}
                times = hourly.get("time") or []
                by_day: dict[str, dict[str, list]] = {}
                for i, ts in enumerate(times):
                    day = str(ts)[:10]
                    bucket = by_day.setdefault(
                        day,
                        {
                            "time": [],
                            "temperature_2m": [],
                            "relative_humidity_2m": [],
                            "precipitation": [],
                            "windspeed_10m": [],
                            "winddirection_10m": [],
                        },
                    )
                    bucket["time"].append(ts)
                    for key in (
                        "temperature_2m",
                        "relative_humidity_2m",
                        "precipitation",
                        "windspeed_10m",
                        "winddirection_10m",
                    ):
                        vals = hourly.get(key) or []
                        bucket[key].append(vals[i] if i < len(vals) else None)
                wanted = set(days)
                for day, bucket in by_day.items():
                    if day not in wanted:
                        continue
                    stats = _afternoon_stats(bucket)
                    rows.append(
                        {
                            "date": day,
                            "venue_id": venue_id,
                            **stats,
                            "has_weather": 1.0,
                            "is_roof": 1.0 if venue_id in ROOF_VENUE_IDS else 0.0,
                        }
                    )
                got = {
                    r["date"]
                    for r in rows
                    if r["venue_id"] == venue_id and r["has_weather"] > 0.5
                }
                for day in days:
                    if day in got:
                        continue
                    rows.append(
                        {
                            "date": day,
                            "venue_id": venue_id,
                            "temp_c": 22.0,
                            "humidity": 50.0,
                            "precip_mm": 0.0,
                            "wind_kph": 0.0,
                            "wind_dir": 0.0,
                            "has_weather": 0.0,
                            "is_roof": 1.0 if venue_id in ROOF_VENUE_IDS else 0.0,
                        }
                    )
                print(f"venue {venue_id}: retry ok", flush=True)
                time.sleep(1.0)
            except Exception as exc2:  # noqa: BLE001
                print(f"venue {venue_id} FAIL2 {exc2}", flush=True)
                for day in days:
                    rows.append(
                        {
                            "date": day,
                            "venue_id": venue_id,
                            "temp_c": 22.0,
                            "humidity": 50.0,
                            "precip_mm": 0.0,
                            "wind_kph": 0.0,
                            "wind_dir": 0.0,
                            "has_weather": 0.0,
                            "is_roof": 1.0 if venue_id in ROOF_VENUE_IDS else 0.0,
                        }
                    )
    frame = pd.DataFrame(rows).drop_duplicates(["date", "venue_id"], keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(frame)} has_weather={frame.has_weather.mean():.3f}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()
    build(args.start_season, args.end_season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
