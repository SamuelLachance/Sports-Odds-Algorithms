"""Build a CBB closing/opening odds table from ESPN's core odds API.

Walks daily mens-college-basketball scoreboards for each requested season,
pulls per-event odds, takes a median consensus across books, and writes
data/supplemental/closing-odds/cbb.csv. Completed games whose odds lookup
fails are still written (empty odds, n_books=0) so result history stays complete.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cbb_v2.data import canon_abbr  # noqa: E402
from web.nba_odds_espn import OUTPUT_FIELDS, _consensus, _get_json  # noqa: E402

CACHE_DIR = PROJECT_ROOT / ".build-cache" / "cbb-odds"
OUTPUT_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "cbb.csv"

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?dates={date}&groups=50&limit=300"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
    "mens-college-basketball/events/{event}/competitions/{comp}/odds"
)

SEASON_START = (11, 1)
SEASON_END = (4, 10)
THROTTLE_SECONDS = 0.18
ODDS_WORKERS = 4

_throttle_lock = threading.Lock()
_next_request_at = 0.0


def _throttled_get(url: str) -> dict[str, Any]:
    global _next_request_at
    with _throttle_lock:
        wait = _next_request_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next_request_at = time.monotonic() + THROTTLE_SECONDS
    return _get_json(url)


def _season_days(season: int) -> Iterator[date]:
    day = date(season - 1, *SEASON_START)
    end = date(season, *SEASON_END)
    while day <= end:
        yield day
        day += timedelta(days=1)


def _iter_completed_events(day: date) -> Iterator[dict[str, Any]]:
    datestr = day.strftime("%Y%m%d")
    cache_path = CACHE_DIR / f"{datestr}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        import json

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None
    else:
        payload = None
    if payload is None:
        try:
            payload = _throttled_get(SCOREBOARD_URL.format(date=datestr))
            cache_path.write_text(
                __import__("json").dumps(payload), encoding="utf-8"
            )
        except OSError:
            return
    for event in payload.get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        status = (competition.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_abbr = canon_abbr((home.get("team") or {}).get("abbreviation", ""))
        away_abbr = canon_abbr((away.get("team") or {}).get("abbreviation", ""))
        if not home_abbr or not away_abbr:
            continue
        try:
            home_score = int(home.get("score"))
            away_score = int(away.get("score"))
        except (TypeError, ValueError):
            continue
        yield {
            "date": day.isoformat(),
            "event": str(event.get("id") or ""),
            "comp": str(competition.get("id") or event.get("id") or ""),
            "home_key": home_abbr,
            "away_key": away_abbr,
            "home_final": home_score,
            "away_final": away_score,
        }


def _odds_row(event: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in OUTPUT_FIELDS}
    row.update(
        {
            "date": event["date"],
            "home_key": event["home_key"],
            "away_key": event["away_key"],
            "home_final": event["home_final"],
            "away_final": event["away_final"],
            "n_books": 0,
        }
    )
    try:
        payload = _throttled_get(
            ODDS_URL.format(event=event["event"], comp=event["comp"])
        )
    except OSError:
        return row
    items = payload.get("items") if isinstance(payload, dict) else None
    if not items:
        return row
    consensus = _consensus(items)
    if not consensus:
        return row
    row.update(consensus)
    row["n_books"] = len(items)
    return row


def fetch_season(season: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for day in _season_days(season):
        day_events = list(_iter_completed_events(day))
        if not day_events:
            continue
        with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
            rows = list(pool.map(_odds_row, day_events))
        events.extend(rows)
        print(f"  {day.isoformat()}: games={len(day_events)}", flush=True)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CBB ESPN closing odds")
    parser.add_argument("--start-season", type=int, default=2024)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fetch a single mid-season week only (sanity check)",
    )
    args = parser.parse_args()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    if args.smoke:
        # One week in Jan 2025 as a connectivity check
        for day in (date(2025, 1, 11) + timedelta(days=i) for i in range(7)):
            day_events = list(_iter_completed_events(day))
            with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
                all_rows.extend(pool.map(_odds_row, day_events))
            print(f"smoke {day}: {len(day_events)} games", flush=True)
    else:
        for season in range(args.start_season, args.end_season + 1):
            print(f"season {season}", flush=True)
            all_rows.extend(fetch_season(season))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in OUTPUT_FIELDS})

    with_odds = sum(1 for r in all_rows if int(r.get("n_books") or 0) > 0)
    print(
        f"wrote {len(all_rows)} rows ({with_odds} with odds) -> {OUTPUT_CSV}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
