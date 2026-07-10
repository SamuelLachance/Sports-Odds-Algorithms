"""Fetch WNBA v2 history into .build-cache/wnba-history/.

Per season:
  {season}/results.json  data.wnba.com final scores (1997+, authoritative)
  {season}/events.json   ESPN scoreboard events (ids for boxes/odds joins)
  {season}/boxes.json    ESPN team box stats per event (2003+)
  {season}/odds.json     ESPN per-book odds per event (2018+)

Box/odds fetches are incremental: already-cached events are skipped, so the
script can be re-run safely (and cheaply for the current season).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.wnba_v2.data import (  # noqa: E402
    BOX_FIRST_SEASON,
    CACHE_ROOT,
    FIRST_SEASON,
    ODDS_FIRST_SEASON,
    fetch_box_score,
    fetch_event_odds,
    fetch_league_results,
    fetch_season_events,
)

WORKERS = 8


def _load(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def fetch_season(season: int, refresh_events: bool) -> None:
    season_dir = CACHE_ROOT / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    events_path = season_dir / "events.json"
    results_path = season_dir / "results.json"

    if refresh_events or not results_path.is_file():
        try:
            results = fetch_league_results(season)
        except OSError:
            results = []  # some seasons 403 on data.wnba.com; ESPN events cover them
        _save(results_path, {"results": results})
    else:
        results = _load(results_path).get("results", [])

    if refresh_events or not events_path.is_file():
        events = fetch_season_events(season)
        _save(events_path, {"events": events})
    else:
        events = _load(events_path).get("events", [])
    completed = [e for e in events if e.get("completed")]
    print(
        f"season {season}: {len(results)} results, {len(events)} espn events "
        f"({len(completed)} final)",
        flush=True,
    )

    if season >= BOX_FIRST_SEASON:
        boxes_path = season_dir / "boxes.json"
        boxes = _load(boxes_path)
        todo = [e for e in completed if e["event_id"] not in boxes]
        if todo:
            t0 = time.time()

            def one_box(event: dict) -> tuple[str, dict | None]:
                try:
                    return event["event_id"], fetch_box_score(event["event_id"])
                except OSError:
                    return event["event_id"], None

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for i, (event_id, box) in enumerate(pool.map(one_box, todo)):
                    boxes[event_id] = box
                    if (i + 1) % 100 == 0:
                        _save(boxes_path, boxes)
            _save(boxes_path, boxes)
            ok = sum(1 for v in boxes.values() if v)
            print(
                f"  boxes: +{len(todo)} fetched, {ok}/{len(boxes)} usable "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )

    if season >= ODDS_FIRST_SEASON:
        odds_path = season_dir / "odds.json"
        odds = _load(odds_path)
        todo = [e for e in completed if e["event_id"] not in odds]
        if todo:
            t0 = time.time()

            def one_odds(event: dict) -> tuple[str, list]:
                return event["event_id"], fetch_event_odds(
                    event["event_id"], event["home_abbr"], event["away_abbr"]
                )

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for i, (event_id, rows) in enumerate(pool.map(one_odds, todo)):
                    odds[event_id] = rows
                    if (i + 1) % 100 == 0:
                        _save(odds_path, odds)
            _save(odds_path, odds)
            with_books = sum(1 for v in odds.values() if v)
            print(
                f"  odds: +{len(todo)} fetched, {with_books}/{len(odds)} with books "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch WNBA v2 history")
    parser.add_argument("--start-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--refresh-events", action="store_true")
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for season in range(args.start_season, args.end_season + 1):
        fetch_season(season, args.refresh_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
