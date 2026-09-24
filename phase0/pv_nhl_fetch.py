"""Player-value program, NHL -- raw play-by-play archive (the Retrosheet analog).

Pulls api-web.nhle.com/v1/gamecenter/{gid}/play-by-play for EVERY game in
data/nhl_games.csv (regular season + playoffs, 2010-11 onward) and stores each
response verbatim, gzip-compressed, under

    data/pv_nhl_pbp/{season}/{gid}.json.gz

The reduction to one-row-per-event lives in phase0/pv_nhl_events.py; this script
only moves bytes.

PROTOCOL
  * Fetching TEST-era games (gid >= 2018000000) is allowed -- serving needs them --
    but this script computes NO statistic from any payload: it checks only that a
    payload parses, carries the requested id and a non-empty `plays` list.
  * DEV games are fetched first (chronological), then TEST games.
  * Market-blind: only the gamecenter play-by-play endpoint is touched.
  * Resumable: a gid whose .json.gz already exists is skipped. Files are written
    to a .tmp and renamed, so a kill never leaves a truncated archive member.
  * Rate limit: a single global throttle spaces request STARTS >= MIN_INTERVAL
    apart across all worker threads (<= 3.7 req/s). HTTP 429 honours Retry-After
    and pauses the whole pool; 5xx / timeouts back off exponentially per request.
  * Failures are appended to data/pv_nhl_pbp_fail.csv (gid, reason, utc). A re-run
    retries every failure except http404 (use --retry-404 to force).

    python phase0/pv_nhl_fetch.py                 # full pull (resumable)
    python phase0/pv_nhl_fetch.py --dry-run       # counts only, no network
    python phase0/pv_nhl_fetch.py --limit 20      # first 20 missing gids
    python phase0/pv_nhl_fetch.py --dev-only      # DEV games only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
ROOT = "data/pv_nhl_pbp"
FAIL = "data/pv_nhl_pbp_fail.csv"
GAMES = "data/nhl_games.csv"
DEV_MAX_GID = 2018000000
MIN_INTERVAL = 0.27          # seconds between request starts, whole pool (<= 3.7 req/s)
WORKERS = 4
MAX_TRIES = 6


def path_for(gid: int, season: str) -> str:
    return f"{ROOT}/{season}/{gid}.json.gz"


def spine():
    """[(gid, season)] DEV first then TEST, each chronological."""
    rows = []
    for r in csv.DictReader(open(GAMES, encoding="utf-8")):
        g = int(r["game_id"])
        rows.append((0 if g < DEV_MAX_GID else 1, r["date"], g, r["season"]))
    rows.sort()
    return [(g, s) for _, _, g, s in rows]


class Throttle:
    """Global request-start spacing shared by all threads, plus a pool-wide pause."""

    def __init__(self, interval):
        self.interval = interval
        self.next_ok = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_ok)
            self.next_ok = t + self.interval
        dt = t - time.monotonic()
        if dt > 0:
            time.sleep(dt)

    def pause(self, secs):
        with self.lock:
            self.next_ok = max(self.next_ok, time.monotonic() + secs)


def fetch(gid, thr):
    """GET with retries. Returns (raw_json_bytes | None, reason)."""
    url = API.format(gid)
    delay = 2.0
    why = "exhausted"
    for attempt in range(MAX_TRIES):
        thr.wait()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            raw = urllib.request.urlopen(req, timeout=60).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            d = json.loads(raw)
            if int(d.get("id", -1)) != gid:
                why = "id-mismatch"
            elif not d.get("plays"):
                why = "empty"
            else:
                return raw, "ok"
            if attempt >= 1:            # a second identical bad payload is final
                return None, why
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return None, "http404"
            why = f"http{ex.code}"
            if ex.code == 429:
                ra = ex.headers.get("Retry-After") if ex.headers else None
                try:
                    w = max(delay, float(ra))
                except (TypeError, ValueError):
                    w = max(delay, 15.0)
                thr.pause(w)
        except Exception as ex:  # noqa: BLE001 (timeouts, resets, bad json)
            why = f"error:{type(ex).__name__}"
        time.sleep(delay)
        delay = min(delay * 2, 90.0)
    return None, why


def save(gid, season, raw):
    p = path_for(gid, season)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(raw)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--retry-404", action="store_true")
    args = ap.parse_args()

    games = spine()
    if args.dev_only:
        games = [(g, s) for g, s in games if g < DEV_MAX_GID]
    have = {g for g, s in games if os.path.exists(path_for(g, s))}
    perm = set()
    if os.path.exists(FAIL) and not args.retry_404:
        for r in csv.reader(open(FAIL, encoding="utf-8")):
            if r and r[0].isdigit() and r[1] == "http404":
                perm.add(int(r[0]))
    todo = [(g, s) for g, s in games if g not in have and g not in perm]
    if args.limit:
        todo = todo[:args.limit]
    n_dev = sum(g < DEV_MAX_GID for g, _ in todo)
    print(f"{len(games):,} spine games | {len(have):,} archived | {len(perm):,} http404 "
          f"| {len(todo):,} to fetch ({n_dev:,} DEV first, {len(todo)-n_dev:,} TEST) "
          f"| <= {1/MIN_INTERVAL:.2f} req/s, ~{len(todo)*MIN_INTERVAL/60:.0f}+ min",
          flush=True)
    if args.dry_run or not todo:
        return 0

    thr = Throttle(MIN_INTERVAL)
    flock = threading.Lock()
    new_fail = not os.path.exists(FAIL)
    ff = open(FAIL, "a", newline="", encoding="utf-8")
    fw = csv.writer(ff)
    if new_fail:
        fw.writerow(["gid", "reason", "utc"])
    t0 = time.time()
    ok = bad = 0

    def job(gs):
        g, s = gs
        raw, why = fetch(g, thr)
        if raw is not None:
            save(g, s, raw)
        return g, why

    # submit in chunks so the in-flight queue (and memory) stays small
    CH = 400
    done_n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i in range(0, len(todo), CH):
            futs = [ex.submit(job, gs) for gs in todo[i:i + CH]]
            for f in as_completed(futs):
                g, why = f.result()
                done_n += 1
                if why == "ok":
                    ok += 1
                else:
                    bad += 1
                    with flock:
                        fw.writerow([g, why, time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                           time.gmtime())])
                        ff.flush()
                if done_n % 200 == 0:
                    el = time.time() - t0
                    print(f"  {done_n:,}/{len(todo):,}  ok {ok:,}  fail {bad}  "
                          f"{el/60:.1f} min  {done_n/el:.2f}/s  "
                          f"eta {el/done_n*(len(todo)-done_n)/60:.0f} min", flush=True)
    ff.close()
    print(f"done: ok {ok:,}, fail {bad}, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
