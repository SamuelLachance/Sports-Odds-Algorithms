"""Incremental NHL refresh: append new final scores to the spine and capture the
upcoming slate — the light daily step (a handful of API calls, not the 500-request
full pull).

  data/nhl_games.csv      += completed games since the last row (deduped)
  data/nhl_upcoming.json   = scheduled games in the next 10 days (for serving
                             predictions in-season; empty file in the offseason)

Standard library only, so it can run in the minimal CI refresh job.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import os
import time
import urllib.request

API = "https://api-web.nhle.com/v1/schedule/{}"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
SPINE = "data/nhl_games.csv"
UPCOMING = "data/nhl_upcoming.json"


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=45).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def main() -> int:
    rows = list(csv.DictReader(open(SPINE, encoding="utf-8")))
    seen = {int(r["game_id"]) for r in rows}
    last_date = max(r["date"] for r in rows)
    today = dt.date.today()
    start = min(dt.date.fromisoformat(last_date) - dt.timedelta(days=3), today)
    end = today + dt.timedelta(days=10)

    new_rows, upcoming = [], []
    cursor = start.isoformat()
    n_req = 0
    while cursor <= end.isoformat() and n_req < 12:
        try:
            data = get(API.format(cursor))
        except Exception as ex:  # noqa: BLE001
            print(f"[nhl_update] fetch failed at {cursor}: {ex}")
            break
        n_req += 1
        for wk in data.get("gameWeek", []):
            gdate = wk.get("date")
            for g in wk.get("games", []):
                gid = g.get("id")
                gtype = g.get("gameType")
                if gtype not in (2, 3) or gid is None:
                    continue
                h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
                state = g.get("gameState")
                if state in ("OFF", "FINAL") and gid not in seen:
                    hs, as_ = h.get("score"), a.get("score")
                    if hs is None or as_ is None:
                        continue
                    seen.add(gid)
                    new_rows.append({
                        "game_id": gid, "date": gdate, "season": g.get("season"),
                        "type": gtype, "away": (a.get("abbrev") or "").strip(),
                        "home": (h.get("abbrev") or "").strip(),
                        "away_goals": as_, "home_goals": hs,
                        "home_win": 1 if hs > as_ else 0,
                        "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
                        "neutral": 1 if g.get("neutralSite") else 0,
                        "win_goalie": (g.get("winningGoalie") or {}).get("playerId", "")})
                elif state in ("FUT", "PRE") and gdate and gdate >= today.isoformat():
                    upcoming.append({
                        "id": gid, "d": gdate, "season": g.get("season"),
                        "playoff": 1 if gtype == 3 else 0,
                        "home": (h.get("abbrev") or "").strip(),
                        "away": (a.get("abbrev") or "").strip(),
                        "t": (g.get("startTimeUTC") or "")})
        nxt = data.get("nextStartDate")
        cursor = nxt if nxt and nxt > cursor else (
            dt.date.fromisoformat(cursor) + dt.timedelta(days=7)).isoformat()
        time.sleep(0.2)

    if new_rows:
        allr = rows + [{k: str(v) for k, v in r.items()} for r in new_rows]
        allr.sort(key=lambda r: (r["date"], int(r["game_id"])))
        # temp + os.replace: an interrupted rewrite must never truncate the spine
        with open(SPINE + ".tmp", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allr[0].keys()))
            w.writeheader()
            w.writerows(allr)
        os.replace(SPINE + ".tmp", SPINE)
    # dedupe upcoming by id, keep earliest listing
    ded = {}
    for u in upcoming:
        ded.setdefault(u["id"], u)
    json.dump(sorted(ded.values(), key=lambda u: (u["d"], u["id"])),
              open(UPCOMING, "w"), indent=0)
    print(f"[nhl_update] +{len(new_rows)} finals, {len(ded)} upcoming "
          f"({n_req} requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
