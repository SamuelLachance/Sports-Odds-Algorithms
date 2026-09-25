"""Pull NHL shift charts (who is on the ice, when) for every spine game.

The NHL shiftcharts API returns one row per player-shift with period + start/end
(MM:SS within the period). This is the hockey analog of NFL snap participation:
it lets us reconstruct the on-ice unit at the moment of every shot, so each shot
can be scored as a match between the players on the ice.

Output data/nhl_shifts.csv: game_id, player_id, team, period, start_s, end_s
(start_s/end_s are seconds within the period). Resumable — skips games already
present. Throttled; the NHL API is public and unauthenticated.

HTML FALLBACK. The API returns NO rows for some games (582 spine games in 2024-25 /
2025-26). For a game of season >= --html-min-season (default 2026-27: future
games; the historical gaps go through the separate backfill + merge path in
phase0/nhl_shifts_html.py) whose API response is empty, the rows are rebuilt from
the official HTML time-on-ice reports (TH/TV{gg}.HTM) by phase0/nhl_shifts_html.py
— validated row-for-row against the API on games that have both sources
(data/nhl_shifts_html_validation.json). Fallback requests are spaced >= 0.5 s
(<= 2 req/s). A report not yet posted (404) or not yet final writes nothing, so
the game is simply retried on the next run. Every HTML-filled game is logged to
data/nhl_shifts_html_filled.csv (provenance); the games still waiting and why go
to data/nhl_shifts_html_pending.json. --no-html-fallback restores API-only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhl_shifts_html as html_fb  # noqa: E402

API = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
OUT = "data/nhl_shifts.csv"
FILLED = "data/nhl_shifts_html_filled.csv"
PENDING = "data/nhl_shifts_html_pending.json"


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


def html_fallback(gid, w, fh, throttle, filled_log, pending):
    """Try the HTML time-on-ice reports for one API-empty game. Writes the game's
    rows in one block (then flushes) only when both reports are final and every
    player maps to an NHL id; otherwise records why and writes nothing."""
    rows, status, qa = html_fb.fallback_rows(gid, throttle)
    if status != "ok":
        pending[str(gid)] = status if qa is None else f"{status}: {'; '.join(qa['reasons'])[:160]}"
        return False
    w.writerows(rows)
    fh.flush()
    new = not os.path.exists(filled_log)
    with open(filled_log, "a", newline="", encoding="utf-8") as lf:
        lw = csv.writer(lf)
        if new:
            lw.writerow(["gid", "rows", "source", "filled_utc"])
        lw.writerow([gid, len(rows), "html_toi_report", html_fb.utcnow()])
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="NHL shift-chart pull (API + HTML fallback)")
    ap.add_argument("--no-html-fallback", action="store_true",
                    help="API only (the pre-fallback behaviour)")
    ap.add_argument("--html-min-season", type=int, default=html_fb.FALLBACK_MIN_SEASON,
                    help="HTML fallback only for games of this season or later "
                         f"(default {html_fb.FALLBACK_MIN_SEASON})")
    args = ap.parse_args(argv)

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
    throttle = html_fb.Throttle()
    html_filled, pending = 0, {}
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
        if (not rows and not args.no_html_fallback
                and html_fb.season_of(gid) >= args.html_min_season):
            html_filled += html_fallback(gid, w, fh, throttle, FILLED, pending)
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
    print(f"done: {len(todo)-len(fails):,} pulled, {len(fails)} failed, "
          f"{html_filled} filled from HTML reports, {len(pending)} API-empty games "
          f"awaiting reports [{time.time()-t0:.0f}s]", flush=True)
    if fails:
        json.dump(fails, open("data/nhl_shift_fails.json", "w"))
    if not args.no_html_fallback:
        with open(PENDING, "w", encoding="utf-8") as pf:
            json.dump({"written": html_fb.utcnow(), "pending": pending}, pf, indent=1)


if __name__ == "__main__":
    main()
