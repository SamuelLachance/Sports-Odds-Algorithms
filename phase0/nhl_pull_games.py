"""Pull the NHL game spine from the public NHL API (api-web.nhle.com).

Walks the schedule week-by-week via `nextStartDate` from 2010-10 to today,
capturing every regular-season and playoff game: date, teams, final scores,
outcome type (REG/OT/SO), neutral-site flag, and the winning goalie id.

Output data/nhl_games.csv, one row per completed game, chronological. This is
the market-blind spine — no odds, only who played, where, and the result.

The NHL API is public and unauthenticated; we throttle politely.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import time
import urllib.request

API = "https://api-web.nhle.com/v1/schedule/{}"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
START = "2010-10-01"


def get(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=45).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def team(side: dict) -> str:
    return (side.get("abbrev") or "").strip()


def main() -> None:
    today = dt.date.today().isoformat()
    cursor = START
    rows = []
    seen = set()
    t0 = time.time()
    n_req = 0
    while cursor <= today:
        for attempt in range(4):
            try:
                data = get(API.format(cursor))
                break
            except Exception as ex:  # noqa: BLE001
                if attempt == 3:
                    print(f"  giving up at {cursor}: {ex}", flush=True)
                    data = None
                else:
                    time.sleep(3 * (attempt + 1))
        n_req += 1
        if data is None:
            cursor = (dt.date.fromisoformat(cursor) + dt.timedelta(days=7)).isoformat()
            continue
        for wk in data.get("gameWeek", []):
            gdate = wk.get("date")
            for g in wk.get("games", []):
                gid = g.get("id")
                if gid in seen:
                    continue
                gtype = g.get("gameType")           # 2 = regular, 3 = playoff
                if gtype not in (2, 3):
                    continue
                state = g.get("gameState")
                if state not in ("OFF", "FINAL"):    # only completed
                    continue
                h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
                hs, as_ = h.get("score"), a.get("score")
                if hs is None or as_ is None:
                    continue
                last = (g.get("gameOutcome") or {}).get("lastPeriodType", "REG")
                seen.add(gid)
                rows.append({
                    "game_id": gid,
                    "date": gdate,
                    "season": g.get("season"),
                    "type": gtype,
                    "away": team(a), "home": team(h),
                    "away_goals": as_, "home_goals": hs,
                    "home_win": 1 if hs > as_ else 0,
                    "last_period": last,             # REG / OT / SO
                    "neutral": 1 if g.get("neutralSite") else 0,
                    "win_goalie": (g.get("winningGoalie") or {}).get("playerId", ""),
                })
        nxt = data.get("nextStartDate")
        if not nxt or nxt <= cursor:
            cursor = (dt.date.fromisoformat(cursor) + dt.timedelta(days=7)).isoformat()
        else:
            cursor = nxt
        time.sleep(0.25)
        if n_req % 40 == 0:
            print(f"  {cursor}: {len(rows):,} games, {n_req} reqs "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    rows.sort(key=lambda r: (r["date"] or "", r["game_id"]))
    with open("data/nhl_games.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"\n{len(rows):,} games -> data/nhl_games.csv")
    print("by season:", dict(sorted(Counter(r["season"] for r in rows).items())))
    print("by type:", dict(Counter(r["type"] for r in rows)))
    print("outcomes:", dict(Counter(r["last_period"] for r in rows)))
    hw = sum(r["home_win"] for r in rows) / len(rows)
    print(f"home win rate: {hw:.4f}")


if __name__ == "__main__":
    main()
