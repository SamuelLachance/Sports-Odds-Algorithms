"""Game-time-aware watcher for official NHL starting goalies (prototype).

Serving-feasibility task of the breakthrough program. The official starter is
marked in the nhl.com Playing Roster report only ~5-19 minutes before the
SCHEDULED start (data/bt_nhl_starter_wayback.json, data/bt_nhl_starter_timing.json),
so the 4-hourly refresh (cron 0 */4 * * *, i.e. 16:00 and 20:00 ET) can never see
it for a 19:00 ET game. This watcher is the piece that can: it polls each game
on a short, start-anchored schedule and records what the official feed said and
when.

    python -X utf8 phase0/bt_nhl_starter_watch.py --date 2026-09-24 [--preseason]
    python -X utf8 phase0/bt_nhl_starter_watch.py --gids 2026010040 2026010042

Per game it calls bt_nhl_starter_feed.fetch_confirmed (2 requests) at
POLL_OFFSETS_MIN before the scheduled start; once both sides are confirmed it
thins the remaining polls to one every POST_CONFIRM_RECHECK_MIN (plus the last)
so a late warm-up swap is still caught and re-served, and it never polls at or
after the scheduled start (a read there is not pre-game). Every changed observation is appended to ARCHIVE
(data/bt_nhl_goalie_confirm.jsonl, append-only, deduped on content), with the
fetch time and minutes to the scheduled start — the "as known pre-game" record
that no historical source provides. With --truth it later reads the boxscore
`starter` flags of finished games and appends one truth record per game, so the
confirmed-vs-actual (late swap) rate is measured instead of assumed.

Nothing is served from here: `on_confirmed` is the hook where a goalie-aware
model would re-serve that game and write its ledger entry with tier CONFIRMED
before the scheduled start. Standard library only; market-blind.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt_nhl_starter_feed as feed  # noqa: E402

ARCHIVE = "data/bt_nhl_goalie_confirm.jsonl"
SCHED = "https://api-web.nhle.com/v1/schedule/{}"
BOX = "https://api-web.nhle.com/v1/gamecenter/{}/boxscore"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}

# minutes before the scheduled start; dense where the Wayback evidence puts the
# first bold goalie (T-19 .. T-0), sparse before (dressed roster only)
POLL_OFFSETS_MIN = (60, 40, 25, 20, 16, 13, 10, 8, 6, 4, 2, 1)
POST_CONFIRM_RECHECK_MIN = 4


def _get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=20).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def slate(date_iso, include_preseason=False):
    """[(gid, start_utc)] for one schedule date (US-Eastern date of the games)."""
    d = _get_json(SCHED.format(date_iso))
    types = (1, 2, 3) if include_preseason else (2, 3)
    out = []
    for wk in d.get("gameWeek", []):
        if wk.get("date") != date_iso:
            continue
        for g in wk.get("games", []):
            if g.get("gameType") in types:
                out.append((int(g["id"]), dt.datetime.fromisoformat(
                    g["startTimeUTC"].replace("Z", "+00:00"))))
    return out


def poll_plan(start_utc, now_utc):
    """Remaining poll times for a game, all strictly before the scheduled start."""
    return [start_utc - dt.timedelta(minutes=m) for m in POLL_OFFSETS_MIN
            if start_utc - dt.timedelta(minutes=m) > now_utc]


def fingerprint(rec):
    """Content key: what the feed said, not when we asked."""
    keep = {k: rec.get(k) for k in ("gid", "state", "ro_status", "pre_game")}
    for side in ("away", "home"):
        s = rec.get(side) or {}
        keep[side] = [s.get("dressed_goalies"), s.get("starter"), s.get("starter_id"),
                      s.get("confirmed")]
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)


def both_confirmed(rec):
    return all((rec.get(s) or {}).get("confirmed") for s in ("away", "home"))


def load_seen(path=ARCHIVE):
    seen = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                seen.add(fingerprint(json.loads(line)))
            except ValueError:
                continue
    return seen


def append(rec, seen, path=ARCHIVE):
    fp = fingerprint(rec)
    if fp in seen:
        return False
    seen.add(fp)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def on_confirmed(rec):
    """Serving hook (prototype): a goalie-aware model would re-serve this game
    here and write its ledger entry {hp, ct, t, tier: CONFIRMED, goalies} before
    the scheduled start."""
    print(f"[watch] CONFIRMED {rec['gid']} T-{rec['min_to_start']}m "
          f"away={rec['away']['starter']} home={rec['home']['starter']}", flush=True)


def watch(games, path=ARCHIVE, clock=None, sleeper=time.sleep, fetch=None):
    clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
    fetch = fetch or feed.fetch_confirmed
    seen = load_seen(path)
    plans = {gid: poll_plan(start, clock()) for gid, start in games}
    confirmed_at = {}
    while any(plans.values()):
        gid, t = min(((g, p[0]) for g, p in plans.items() if p), key=lambda x: x[1])
        wait = (t - clock()).total_seconds()
        if wait > 0:
            sleeper(min(wait, 300.0))
            continue
        plans[gid].pop(0)
        try:
            rec = fetch(gid)
        except Exception as ex:  # noqa: BLE001
            print(f"[watch] {gid} fetch failed: {ex}", flush=True)
            continue
        new = append(rec, seen, path)
        print(f"[watch] {gid} T-{rec.get('min_to_start')}m state={rec.get('state')} "
              f"ro={rec.get('ro_status')} away={(rec.get('away') or {}).get('starter')} "
              f"home={(rec.get('home') or {}).get('starter')} {'NEW' if new else 'same'}",
              flush=True)
        if both_confirmed(rec):
            key = (rec["away"]["starter"], rec["home"]["starter"])
            prev = confirmed_at.get(gid)
            if prev is None or (prev["away"]["starter"], prev["home"]["starter"]) != key:
                # first confirmation, or a swap after one: (re-)serve
                confirmed_at[gid] = rec
                on_confirmed(rec)
            # keep re-checking until the start (late warm-up swaps), but thinly:
            # polls at least POST_CONFIRM_RECHECK_MIN apart, the last one kept
            thin, last = [], clock()
            for x in plans[gid]:
                if x - last >= dt.timedelta(minutes=POST_CONFIRM_RECHECK_MIN) or x == plans[gid][-1]:
                    thin.append(x)
                    last = x
            plans[gid] = thin
    return confirmed_at


def truth(gids, path=ARCHIVE):
    """Append the official post-game starters (boxscore starter flag) per game."""
    seen = load_seen(path)
    for gid in gids:
        b = _get_json(BOX.format(gid))
        if b.get("gameState") not in ("OFF", "FINAL"):
            continue
        pg = b.get("playerByGameStats") or {}
        rec = {"gid": gid, "t_fetch": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "state": b.get("gameState"), "ro_status": None, "pre_game": False, "truth": True}
        for side, key in (("away", "awayTeam"), ("home", "homeTeam")):
            gl = (pg.get(key) or {}).get("goalies") or []
            st = [g["playerId"] for g in gl if g.get("starter") is True]
            rec[side] = {"dressed_goalies": [g["playerId"] for g in gl], "starter": None,
                         "starter_id": st[0] if len(st) == 1 else None, "confirmed": False,
                         "has_flag": any("starter" in g for g in gl)}
        append(rec, seen, path)
        time.sleep(0.3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--gids", nargs="*", type=int)
    ap.add_argument("--preseason", action="store_true")
    ap.add_argument("--truth", nargs="*", type=int)
    ap.add_argument("--archive", default=ARCHIVE)
    a = ap.parse_args(argv)
    if a.truth:
        truth(a.truth, a.archive)
        return 0
    games = []
    if a.date:
        games = slate(a.date, a.preseason)
    if a.gids:
        want = set(a.gids)
        if not games:
            d = feed._get(feed.PBP_URL.format(gid=a.gids[0]))  # noqa: SLF001
            day = json.loads(d[1])["gameDate"]
            games = slate(day, include_preseason=True)
        games = [(g, s) for g, s in games if g in want]
    if not games:
        print("[watch] no games")
        return 0
    print(f"[watch] {len(games)} games: " + ", ".join(f"{g}@{s:%H:%M}Z" for g, s in games),
          flush=True)
    done = watch(games, a.archive)
    print(f"[watch] done: {len(done)}/{len(games)} games confirmed before the scheduled start",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
