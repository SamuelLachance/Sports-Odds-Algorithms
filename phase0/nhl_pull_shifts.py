"""Pull NHL shift charts (who is on the ice, when) for every spine game.

The NHL shiftcharts API returns one row per player-shift with period + start/end
(MM:SS within the period). This is the hockey analog of NFL snap participation:
it lets us reconstruct the on-ice unit at the moment of every shot, so each shot
can be scored as a match between the players on the ice.

Output data/nhl_shifts.csv: game_id, player_id, team, period, start_s, end_s
(start_s/end_s are seconds within the period). Resumable — skips games already
present. Throttled; the NHL API is public and unauthenticated.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import time
import urllib.request

API = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
OUT = "data/nhl_shifts.csv"


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=45).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def to_sec(mmss):
    try:
        m, s = mmss.split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def main():
    spine = [r for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8"))]
    # regular season + playoffs, chronological
    gids = [int(r["game_id"]) for r in spine]

    done = set()
    if os.path.exists(OUT):
        for r in csv.reader(open(OUT, encoding="utf-8")):
            if r and r[0].isdigit():
                done.add(int(r[0]))
    todo = [g for g in gids if g not in done]
    print(f"{len(gids):,} spine games, {len(done):,} already pulled, "
          f"{len(todo):,} to go", flush=True)

    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not done:
        w.writerow(["game_id", "player_id", "team", "period", "start_s", "end_s"])
    t0 = time.time()
    fails = []
    for k, gid in enumerate(todo):
        rows = None
        for attempt in range(4):
            try:
                d = get(API.format(gid))
                rows = d.get("data", [])
                break
            except Exception as ex:  # noqa: BLE001
                if attempt == 3:
                    fails.append(gid)
                else:
                    time.sleep(3 * (attempt + 1))
        if rows is None:
            continue
        for s in rows:
            st, en = to_sec(s.get("startTime")), to_sec(s.get("endTime"))
            if st is None or en is None or s.get("playerId") is None:
                continue
            w.writerow([gid, s["playerId"], s.get("teamAbbrev", ""),
                        s.get("period", 1), st, en])
        if (k + 1) % 200 == 0:
            fh.flush()
            print(f"  {k+1:,}/{len(todo):,}  {fails and len(fails) or 0} fails "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        time.sleep(0.12)
    fh.close()
    print(f"done: {len(todo)-len(fails):,} pulled, {len(fails)} failed "
          f"[{time.time()-t0:.0f}s]", flush=True)
    if fails:
        json.dump(fails, open("data/nhl_shift_fails.json", "w"))


if __name__ == "__main__":
    main()
