"""Build a CBB closing/opening odds table from ESPN's core odds API.

Walks daily mens-college-basketball scoreboards for each requested season (or
date window), pulls per-event odds, takes a median consensus across books, and
merges into data/supplemental/closing-odds/cbb.csv. Completed games whose odds
lookup fails are still written (empty odds, n_books=0) so result history stays
complete. Per-day scoreboard JSON is cached under .build-cache/cbb-odds/ so
re-runs are resumable.

One-command multi-season backfill (merges with any existing rows):

    python scripts/fetch_cbb_odds.py --start-season 2024 --end-season 2026

Partial window (e.g. conference season only):

    python scripts/fetch_cbb_odds.py --start-date 2025-01-01 --end-date 2025-03-15

Smoke connectivity check (one mid-season week; merges, does not wipe CSV):

    python scripts/fetch_cbb_odds.py --smoke

Full seasons are large (~5k+ games each) and throttle at ~0.18s/request — expect
tens of minutes per season. Prefer --start-date/--end-date chunks if ESPN rate-
limits; re-run safely thanks to merge + day cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cbb_v2.data import canon_abbr  # noqa: E402
from web.nba_odds_espn import OUTPUT_FIELDS, _consensus, _get_json  # noqa: E402

# CBB blowouts can exceed NBA's 40-pt cap (archive max ~54.5).
MAX_CBB_SPREAD = 60.0

try:
    from web.closing_odds_db import clear_closing_odds_cache
except ImportError:  # pragma: no cover - optional in lean checkouts
    def clear_closing_odds_cache() -> None:
        return None

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
THROTTLE_SECONDS = 0.15
ODDS_WORKERS = 6

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


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _season_days(season: int) -> Iterator[date]:
    day = date(season - 1, *SEASON_START)
    end = date(season, *SEASON_END)
    while day <= end:
        yield day
        day += timedelta(days=1)


def _date_range(start: date, end: date) -> Iterator[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _iter_completed_events(day: date) -> Iterator[dict[str, Any]]:
    datestr = day.strftime("%Y%m%d")
    cache_path = CACHE_DIR / f"{datestr}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] | None = None
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None
    if payload is None:
        try:
            payload = _throttled_get(SCOREBOARD_URL.format(date=datestr))
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            return
    if not isinstance(payload, dict):
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
            "source": "espn-core",
        }
    )
    try:
        payload = _throttled_get(
            ODDS_URL.format(event=event["event"], comp=event["comp"])
        )
    except OSError:
        return row
    raw_items = payload.get("items") if isinstance(payload, dict) else None
    if not raw_items:
        return row
    # Exclude live/in-game books from "closing" consensus (same as CFB).
    items = [
        item
        for item in raw_items
        if "live" not in ((item.get("provider") or {}).get("name", "").lower())
    ]
    if not items:
        return row
    consensus = _consensus(items, max_handicap_abs=MAX_CBB_SPREAD)
    if not consensus:
        return row
    row.update(consensus)
    row["n_books"] = len(items)
    row["source"] = "espn-core"
    return row


def fetch_days(days: list[date]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for day in days:
        day_events = list(_iter_completed_events(day))
        if not day_events:
            print(f"  {day.isoformat()}: games=0", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
            rows = list(pool.map(_odds_row, day_events))
        events.extend(rows)
        with_odds = sum(1 for r in rows if int(r.get("n_books") or 0) > 0)
        print(
            f"  {day.isoformat()}: games={len(day_events)} odds={with_odds}",
            flush=True,
        )
    return events


def fetch_season(season: int) -> list[dict[str, Any]]:
    return fetch_days(list(_season_days(season)))


def _load_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("date") or ""),
        str(row.get("home_key") or ""),
        str(row.get("away_key") or ""),
    )


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in OUTPUT_FIELDS:
        value = row.get(field)
        if value is None:
            out[field] = ""
        else:
            out[field] = str(value)
    if not out.get("source"):
        out["source"] = "espn-core"
    return out


def merge_rows(
    existing: list[dict[str, Any]], fresh: list[dict[str, Any]]
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in existing:
        key = _row_key(row)
        if key[0]:
            merged[key] = _normalize_row(row)
    for row in fresh:
        key = _row_key(row)
        if key[0]:
            merged[key] = _normalize_row(row)
    return sorted(merged.values(), key=lambda item: (item["date"], item["home_key"]))


def write_csv(rows: list[dict[str, str]], path: Path = OUTPUT_CSV) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch CBB ESPN closing odds (merges into cbb.csv by default)."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2024,
        help="First season end-year inclusive (default 2024 = 2023-24)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2026,
        help="Last season end-year inclusive (default 2026 = 2025-26)",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help="Optional YYYY-MM-DD start (overrides season range when set with --end-date)",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="Optional YYYY-MM-DD end (overrides season range when set with --start-date)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fetch a single mid-season week only (sanity check; still merges)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite CSV with this run only (default merges with existing rows)",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []

    if args.smoke:
        days = [date(2025, 1, 11) + timedelta(days=i) for i in range(7)]
        print("smoke week 2025-01-11 .. 2025-01-17", flush=True)
        all_rows.extend(fetch_days(days))
    elif args.start_date is not None or args.end_date is not None:
        if args.start_date is None or args.end_date is None:
            parser.error("--start-date and --end-date must be used together")
        if args.end_date < args.start_date:
            parser.error("--end-date must be on or after --start-date")
        print(f"window {args.start_date} -> {args.end_date}", flush=True)
        all_rows.extend(fetch_days(list(_date_range(args.start_date, args.end_date))))
    else:
        for season in range(args.start_season, args.end_season + 1):
            print(f"season {season}", flush=True)
            all_rows.extend(fetch_season(season))

    existing: list[dict[str, Any]] = [] if args.replace else _load_existing(OUTPUT_CSV)
    merged = merge_rows(existing, all_rows)
    count = write_csv(merged)
    clear_closing_odds_cache()

    with_odds = sum(1 for r in merged if int(r.get("n_books") or 0) > 0)
    fresh_odds = sum(1 for r in all_rows if int(r.get("n_books") or 0) > 0)
    print(
        f"wrote {count} rows ({with_odds} with odds; +{len(all_rows)} fetched, "
        f"{fresh_odds} with odds this run) -> {OUTPUT_CSV}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
