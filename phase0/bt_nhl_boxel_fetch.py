"""Breakthrough pick nhl_boxel -- boxscore fetch for DEV regular-season games.

Pulls api-web.nhle.com/v1/gamecenter/{gid}/boxscore for every DEV regular-season
game (type 2, season <= 20172018, game_id < 2018000000) and writes ONE ROW PER
DRESSED PLAYER to data/bt_nhl_boxel_box.csv:

    gid, side, team, pid, pos, is_goalie, starter, G, A, SOG, BLK, PIM, toi_s

  side      'home' / 'away' (the screen joins on side, never on the abbrev)
  pos       boxscore position code: C / L / R / D / G
  starter   goalies only: the boxscore `starter` flag (0/1); skaters 0
  toi_s     all-situations time on ice, seconds

PROTOCOL
  * DEV ONLY. The gid list is filtered to season <= 20172018 and every gid is
    asserted < 2018000000 BEFORE any request is made. No TEST-era game is ever
    requested. (A TEST look would need 2018-26 boxscores; that fetch belongs to
    the lead engineer, not to this script.)
  * Market-blind: the payload's box stats only; no odds endpoint is touched.
  * Resumable: gids already present in the output (or recorded as permanently
    empty in data/bt_nhl_boxel_box_fail.csv) are skipped.
  * Rate-limited to <= 4 requests/second (MIN_INTERVAL between request starts),
    retries with exponential backoff (honours Retry-After on HTTP 429).

This is a network pull of ~9.3k public JSON documents (~9,371 gids). Per the
screen's pre-registration it must be APPROVED before it is run. Run it in the
background:

    python phase0/bt_nhl_boxel_fetch.py              # full pull (resumable)
    python phase0/bt_nhl_boxel_fetch.py --dry-run    # count only, no network
    python phase0/bt_nhl_boxel_fetch.py --selftest   # offline parser test only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api-web.nhle.com/v1/gamecenter/{}/boxscore"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
OUT = "data/bt_nhl_boxel_box.csv"
FAIL = "data/bt_nhl_boxel_box_fail.csv"
COLS = ["gid", "side", "team", "pid", "pos", "is_goalie", "starter",
        "G", "A", "SOG", "BLK", "PIM", "toi_s"]
MAX_GID = 2018000000
DEV_END = 20172018
MIN_INTERVAL = 0.30          # seconds between request starts  (<= 3.33 req/s)
MAX_TRIES = 5


def dev_gids():
    gids = []
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        if r["type"] != "2" or int(r["season"]) > DEV_END:
            continue
        gids.append((r["date"], int(r["game_id"])))
    gids.sort()
    out = [g for _, g in gids]
    assert out and max(out) < MAX_GID, "TEST-era gid in the fetch list"
    return out


def toi_sec(s):
    if s is None:
        return 0
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except ValueError:
        return 0


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def parse(gid, d):
    """Boxscore JSON -> list of output rows (lists in COLS order).

    Returns None when the payload carries no player stats (permanent empty)."""
    pbg = d.get("playerByGameStats") or d.get("boxscore", {}).get("playerByGameStats")
    if not pbg:
        return None
    rows = []
    for side, tkey in (("home", "homeTeam"), ("away", "awayTeam")):
        team = (d.get(tkey) or {}).get("abbrev", "")
        blk = pbg.get(tkey) or {}
        for grp in ("forwards", "defense"):
            for p in blk.get(grp, []) or []:
                pid = p.get("playerId")
                if pid is None:
                    continue
                pos = p.get("position") or ("D" if grp == "defense" else "F")
                sog = p.get("sog", p.get("shots", 0))
                bs = p.get("blockedShots", p.get("blocks", 0))
                rows.append([gid, side, team, int(pid), pos, 0, 0,
                             _i(p.get("goals")), _i(p.get("assists")), _i(sog),
                             _i(bs), _i(p.get("pim")), toi_sec(p.get("toi"))])
        for p in blk.get("goalies", []) or []:
            pid = p.get("playerId")
            if pid is None:
                continue
            st = p.get("starter")
            rows.append([gid, side, team, int(pid), "G", 1, int(bool(st)),
                         _i(p.get("goals")), _i(p.get("assists")), 0, 0,
                         _i(p.get("pim")), toi_sec(p.get("toi"))])
    return rows if rows else None


class Throttle:
    def __init__(self, interval):
        self.interval = interval
        self.last = 0.0

    def wait(self):
        dt = time.monotonic() - self.last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self.last = time.monotonic()


def get(url, thr):
    """GET with retries + exponential backoff. Returns (json | None, reason)."""
    delay = 2.0
    for attempt in range(MAX_TRIES):
        thr.wait()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            raw = urllib.request.urlopen(req, timeout=45).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return json.loads(raw), "ok"
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return None, "http404"
            wait = delay
            if ex.code == 429:
                ra = ex.headers.get("Retry-After") if ex.headers else None
                try:
                    wait = max(delay, float(ra))
                except (TypeError, ValueError):
                    wait = max(delay, 10.0)
            if attempt == MAX_TRIES - 1:
                return None, f"http{ex.code}"
            time.sleep(wait)
        except Exception as ex:  # noqa: BLE001  (timeouts, resets, bad json)
            if attempt == MAX_TRIES - 1:
                return None, f"error:{type(ex).__name__}"
            time.sleep(delay)
        delay = min(delay * 2, 60.0)
    return None, "exhausted"


def selftest():
    """Offline: the parser on a synthetic payload in the api-web schema."""
    d = {"id": 2011020001, "homeTeam": {"abbrev": "NYR"}, "awayTeam": {"abbrev": "LAK"},
         "playerByGameStats": {
             "homeTeam": {
                 "forwards": [{"playerId": 1, "position": "C", "goals": 1, "assists": 2,
                               "sog": 3, "blockedShots": 1, "pim": 2, "hits": 4,
                               "toi": "17:05"}],
                 "defense": [{"playerId": 2, "position": "D", "goals": 0, "assists": 0,
                              "shots": 1, "blocks": 3, "pim": 0, "toi": "22:30"}],
                 "goalies": [{"playerId": 3, "position": "G", "starter": True,
                              "toi": "60:00", "pim": 0}]},
             "awayTeam": {"forwards": [], "defense": [],
                          "goalies": [{"playerId": 4, "starter": False, "toi": "0:00"}]}}}
    rows = parse(2011020001, d)
    assert rows[0] == [2011020001, "home", "NYR", 1, "C", 0, 0, 1, 2, 3, 1, 2, 1025], rows[0]
    assert rows[1] == [2011020001, "home", "NYR", 2, "D", 0, 0, 0, 0, 1, 3, 0, 1350], rows[1]
    assert rows[2][4:7] == ["G", 1, 1] and rows[3][1] == "away" and rows[3][6] == 0
    assert parse(1, {"id": 1}) is None
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    gids = dev_gids()
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        # resume hygiene: a kill mid-game can leave a partial game at the tail.
        # Games with < 30 skater rows are dropped from the file and re-fetched.
        import pandas as pd
        df = pd.read_csv(OUT)
        sk = df[df.is_goalie == 0].groupby("gid").size()
        bad = set(int(g) for g in sk[sk < 30].index) | (set(df.gid.unique()) - set(sk.index))
        if bad and not args.dry_run:
            tmp = OUT + ".tmp"
            df[~df.gid.isin(bad)].to_csv(tmp, index=False)
            os.replace(tmp, OUT)
            print(f"resume: dropped {len(bad)} incomplete game(s) for re-fetch", flush=True)
        done = set(int(g) for g in df.gid.unique()) - bad
    perm_empty = set()
    if os.path.exists(FAIL):
        for r in csv.reader(open(FAIL, encoding="utf-8")):
            if r and r[0].isdigit() and r[1] in ("empty", "http404"):
                perm_empty.add(int(r[0]))
    todo = [g for g in gids if g not in done and g not in perm_empty]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(gids):,} DEV gids, {len(done):,} done, {len(perm_empty):,} permanently "
          f"empty, {len(todo):,} to fetch (<= {1/MIN_INTERVAL:.2f} req/s, "
          f"~{len(todo)*MIN_INTERVAL/60:.0f}+ min)", flush=True)
    if args.dry_run or not todo:
        return
    assert all(g < MAX_GID for g in todo)
    new_file = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new_file:
        w.writerow(COLS)
    ff = open(FAIL, "a", newline="", encoding="utf-8")
    fw = csv.writer(ff)
    thr = Throttle(MIN_INTERVAL)
    t0 = time.time()
    nrow = nfail = 0
    for k, gid in enumerate(todo):
        d, why = get(API.format(gid), thr)
        rows = parse(gid, d) if d is not None else None
        if rows is None:
            fw.writerow([gid, "empty" if d is not None else why])
            ff.flush()
            nfail += 1
        else:
            w.writerows(rows)          # one game's rows written together
            fh.flush()                 # flushed per game: a kill loses <= 1 game
            nrow += len(rows)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {k+1:,}/{len(todo):,} games  {nrow:,} rows  {nfail} fails  "
                  f"{el/60:.1f} min  eta {el/(k+1)*(len(todo)-k-1)/60:.0f} min",
                  flush=True)
    fh.close()
    ff.close()
    print(f"done: {nrow:,} rows, {nfail} fails, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
