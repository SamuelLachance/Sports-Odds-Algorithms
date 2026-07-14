"""Build point-in-time MLB injured-list burden from Stats API transactions.

Public source: https://statsapi.mlb.com/api/v1/transactions

Parses status-change descriptions for placed/activated IL events and emits a
daily team IL headcount (pitchers + position players + 60-day). Missing days
carry forward the last known count (no invented injuries).

Output: data/supplemental/mlb-injuries/team_il_daily.csv

Usage:
  python scripts/build_mlb_il_burden.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_v2.statsapi_data import BASE_URL, fetch_team_ids, get_json  # noqa: E402

OUT = PROJECT_ROOT / "data" / "supplemental" / "mlb-injuries" / "team_il_daily.csv"

_PLACE = re.compile(
    r"placed\s+\w+\s+.+?\s+on\s+the\s+(\d+)-day\s+(?:injured|disabled)\s+list",
    re.IGNORECASE,
)
_ACTIVATE = re.compile(
    r"activated\s+\w+\s+.+?\s+from\s+the\s+(\d+)-day\s+(?:injured|disabled)\s+list",
    re.IGNORECASE,
)
_TRANSFER = re.compile(
    r"transferred\s+\w+\s+.+?\s+from\s+the\s+(\d+)-day\s+(?:injured|disabled)\s+list\s+to\s+the\s+(\d+)-day",
    re.IGNORECASE,
)


def _parse_events(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for tx in transactions:
        desc = str(tx.get("description") or "")
        day = str(tx.get("date") or "")[:10]
        if not day:
            continue
        team = ((tx.get("toTeam") or tx.get("fromTeam") or {}) or {}).get("id")
        if team is None:
            # description usually starts with team name; fall back later via team loop
            pass
        person = (tx.get("person") or {}).get("id")
        place = _PLACE.search(desc)
        act = _ACTIVATE.search(desc)
        xfer = _TRANSFER.search(desc)
        is_pitcher = 1.0 if re.search(r"\b[LR]HP\b", desc) else 0.0
        if place:
            events.append(
                {
                    "date": day,
                    "team_id": int(team) if team is not None else None,
                    "player_id": int(person) if person is not None else None,
                    "days": int(place.group(1)),
                    "action": "place",
                    "is_pitcher": is_pitcher,
                }
            )
        elif act:
            events.append(
                {
                    "date": day,
                    "team_id": int(team) if team is not None else None,
                    "player_id": int(person) if person is not None else None,
                    "days": int(act.group(1)),
                    "action": "activate",
                    "is_pitcher": is_pitcher,
                }
            )
        elif xfer:
            # Still on IL; bump to 60-day bucket via days field.
            events.append(
                {
                    "date": day,
                    "team_id": int(team) if team is not None else None,
                    "player_id": int(person) if person is not None else None,
                    "days": int(xfer.group(2)),
                    "action": "place",
                    "is_pitcher": is_pitcher,
                }
            )
    return events


def fetch_team_season(team_id: int, season: int) -> list[dict[str, Any]]:
    start = f"{season}-02-01"
    end = f"{season}-11-15"
    url = f"{BASE_URL}/transactions?teamId={team_id}&startDate={start}&endDate={end}"
    payload = get_json(url)
    events = _parse_events(payload.get("transactions") or [])
    for ev in events:
        if ev.get("team_id") is None:
            ev["team_id"] = team_id
    return events


def build_daily(events: list[dict[str, Any]], season: int, team_ids: list[int]) -> pd.DataFrame:
    # Track set of player_ids on IL per team.
    on_il: dict[int, set[int]] = {tid: set() for tid in team_ids}
    pitcher_il: dict[int, set[int]] = {tid: set() for tid in team_ids}
    long_il: dict[int, set[int]] = {tid: set() for tid in team_ids}

    by_date: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_date.setdefault(ev["date"], []).append(ev)

    start = date(season, 3, 1)
    end = date(season, 11, 1)
    rows: list[dict[str, Any]] = []
    day = start
    while day <= end:
        iso = day.isoformat()
        for ev in by_date.get(iso, []):
            tid = int(ev["team_id"])
            pid = ev.get("player_id")
            if pid is None:
                # synthetic id from description hash when person missing
                pid = -(abs(hash((tid, ev["action"], iso, ev.get("days")))) % 10_000_000)
            pid = int(pid)
            if ev["action"] == "place":
                on_il.setdefault(tid, set()).add(pid)
                if ev.get("is_pitcher", 0) >= 0.5:
                    pitcher_il.setdefault(tid, set()).add(pid)
                if int(ev.get("days") or 0) >= 60:
                    long_il.setdefault(tid, set()).add(pid)
            else:
                on_il.setdefault(tid, set()).discard(pid)
                pitcher_il.setdefault(tid, set()).discard(pid)
                long_il.setdefault(tid, set()).discard(pid)
        for tid in team_ids:
            n = len(on_il.get(tid, set()))
            rows.append(
                {
                    "date": iso,
                    "season": season,
                    "team_id": tid,
                    "il_count": float(n),
                    "il_pitchers": float(len(pitcher_il.get(tid, set()))),
                    "il_60day": float(len(long_il.get(tid, set()))),
                    "has_il": 1.0,
                }
            )
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for season in range(args.start_season, args.end_season + 1):
        try:
            team_ids = fetch_team_ids(season)
        except OSError as exc:
            print(f"season {season} team_ids FAIL {exc}", flush=True)
            continue
        events: list[dict[str, Any]] = []
        for tid in team_ids:
            try:
                events.extend(fetch_team_season(tid, season))
            except OSError as exc:
                print(f"team {tid} season {season} FAIL {exc}", flush=True)
        frame = build_daily(events, season, team_ids)
        frames.append(frame)
        print(
            f"season {season}: events={len(events)} daily_rows={len(frame)} "
            f"mean_il={frame.il_count.mean():.2f}",
            flush=True,
        )

    if not frames:
        print("no IL rows built", flush=True)
        return 1
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
