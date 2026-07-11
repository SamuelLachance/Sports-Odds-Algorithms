"""Fetch NBA v2 history into .build-cache/nba-history/.

Per season (ending year):
  {season}/events.json   ESPN scoreboard events (scores + ids)
  {season}/boxes.json    ESPN team box stats per event (2003+)

Odds are joined later from data/supplemental/closing-odds/nba.csv (no per-event
odds crawl needed for the training table). Box fetches are incremental.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_v2.data import (  # noqa: E402
    BOX_FIRST_SEASON,
    CACHE_ROOT,
    FIRST_SEASON,
    fetch_box_score,
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


def fetch_season(season: int, refresh_events: bool, retry_missing: bool = False) -> None:
    season_dir = CACHE_ROOT / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    events_path = season_dir / "events.json"

    if refresh_events or not events_path.is_file():
        events = fetch_season_events(season)
        _save(events_path, {"events": events})
    else:
        events = _load(events_path).get("events", [])
    completed = [e for e in events if e.get("completed")]
    print(
        f"season {season}: {len(events)} espn events ({len(completed)} final)",
        flush=True,
    )

    if season >= BOX_FIRST_SEASON:
        boxes_path = season_dir / "boxes.json"
        boxes = _load(boxes_path)
        if retry_missing:
            todo = [e for e in completed if not boxes.get(e["event_id"])]
        else:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch NBA v2 history")
    parser.add_argument("--start", type=int, default=FIRST_SEASON)
    parser.add_argument(
        "--end",
        type=int,
        default=datetime.now(timezone.utc).year + (1 if datetime.now(timezone.utc).month >= 10 else 0),
    )
    parser.add_argument("--refresh-events", action="store_true")
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="re-fetch events whose cached box entry is null (transient failures)",
    )
    args = parser.parse_args()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for season in range(args.start, args.end + 1):
        fetch_season(season, args.refresh_events, args.retry_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
