"""Incremental NHL refresh: append new final scores to the spine and capture the
upcoming slate — the light daily step (a handful of API calls, not the 500-request
full pull).

  data/nhl_games.csv      += completed games since the last row (deduped)
  data/nhl_upcoming.json   = scheduled games in the next 10 days (for serving
                             predictions in-season; empty file in the offseason)
  data/nhl_lineup_archive.jsonl += game-day pre-game roster/scratch snapshots
                             from the gamecenter landing endpoint (see
                             archive_lineups)

then runs phase0/nhl_site_fetch.py (rosters, production stats, standings
identity, the full season schedule - display data for the site, each step
cached and failure-tolerant).

"Today" is the US-Eastern date, not the runner's UTC date: a 00:00-04:00 UTC
run is still the previous evening in North America, and a UTC date dropped that
night's games from the upcoming slate. Games in progress (LIVE/CRIT) are kept
in the slate with their `state`, so a serve during a game does not drop its
row, pick and badge until the final.

Standard library only, so it can run in the minimal CI refresh job.

Bounded in wall-clock time: refresh.py runs this under a 900 s subprocess
timeout, and a run that overshoots aborts the whole refresh (MLB deploy
included). REQ_TIMEOUT per request; the schedule walk stops after
SCHEDULE_BUDGET_S, the lineup archive at ARCHIVE_BUDGET_S (or after
MAX_CONSEC_FAIL failed games in a row), and the site fetch gets what is left of
UPDATE_BUDGET_S, capped at its own BUDGET_S. Worst case ~UPDATE_BUDGET_S plus
one request, far under 900 s.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API = "https://api-web.nhle.com/v1/schedule/{}"
LANDING = "https://api-web.nhle.com/v1/gamecenter/{}/landing"
RIGHT_RAIL = "https://api-web.nhle.com/v1/gamecenter/{}/right-rail"
PLAY_BY_PLAY = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
SPINE = "data/nhl_games.csv"
UPCOMING = "data/nhl_upcoming.json"
LINEUP_ARCHIVE = "data/nhl_lineup_archive.jsonl"

# gameState values: FUT (scheduled), PRE (pre-game), LIVE/CRIT (in progress),
# OFF/FINAL (done). Only the last two are results.
UNPLAYED_STATES = ("FUT", "PRE", "LIVE", "CRIT")

REQ_TIMEOUT = 20            # seconds per request (was 45)
UPDATE_BUDGET_S = 600.0     # the whole run, site fetch included
SCHEDULE_BUDGET_S = 180.0   # the /schedule walk (spine finals + slate)
ARCHIVE_BUDGET_S = 330.0    # schedule walk + lineup archive, from the start
MAX_CONSEC_FAIL = 3         # lineup archive: failed games in a row before stopping


def et_today() -> dt.date:
    """The US-Eastern calendar date (the NHL's schedule date)."""
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001  (no tz database: EDT/EST by month)
        now = dt.datetime.now(dt.timezone.utc)
        off = 4 if 3 <= now.month <= 10 else 5
        return (now - dt.timedelta(hours=off)).date()


def get(url, timeout=REQ_TIMEOUT):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _lineup_subset(payload, rail=None, pbp=None):
    """Minimal raw-ish roster subset of a gamecenter landing payload.

    Observed structure (2026-07 probe): a far-future FUT game exposes only
    matchup.goalieComparison (season-stat goalie lists per side); rosterSpots
    and scratches are absent pre-game, and a FINAL game's scratches live in the
    right-rail endpoint instead. What landing shows for a PRE/game-day game
    once lineups post is exactly what this archive exists to discover, so all
    candidate keys are handled defensively.
    """
    out = {}
    rs = payload.get("rosterSpots")
    if rs:
        out["rosterSpots"] = [
            {k: p.get(k) for k in ("teamId", "playerId", "positionCode",
                                   "sweaterNumber")}
            for p in rs if isinstance(p, dict)]
    if payload.get("scratches"):
        out["scratches"] = payload["scratches"]
    gi = (payload.get("summary") or {}).get("gameInfo") or {}
    for side in ("homeTeam", "awayTeam"):
        sc = (gi.get(side) or {}).get("scratches")
        if sc:
            out.setdefault("gameInfoScratches", {})[side] = [
                p.get("id") for p in sc if isinstance(p, dict)]
    gc = (payload.get("matchup") or {}).get("goalieComparison") or {}
    for side in ("homeTeam", "awayTeam"):
        goalies = (gc.get(side) or {}).get("leaders") or []
        ids = [g.get("playerId") for g in goalies
               if isinstance(g, dict) and g.get("playerId")]
        if ids:
            out.setdefault("goalieComparison", {})[side] = ids

    # --- the two endpoints that actually carry the payload we came for ---
    rgi = ((rail or {}).get("gameInfo") or {})
    for side in ("homeTeam", "awayTeam"):
        sc = (rgi.get(side) or {}).get("scratches")
        if sc:
            out.setdefault("railScratches", {})[side] = [
                p.get("id") for p in sc if isinstance(p, dict) and p.get("id")]
        coach = ((rgi.get(side) or {}).get("headCoach") or {}).get("default")
        if coach:
            out.setdefault("headCoach", {})[side] = coach
    prs = (pbp or {}).get("rosterSpots")
    if prs:
        out["dressed"] = [
            {k: q.get(k) for k in ("teamId", "playerId", "positionCode",
                                   "sweaterNumber")}
            for q in prs if isinstance(q, dict)]
    return out


def _canon(obj):
    """Order-insensitive canonical form, for comparing two lineup snapshots.

    The API is free to return rosterSpots in any order, so a raw dump would
    treat a re-ordering as a change and archive a duplicate every cycle. Sorting
    every list by its own canonical text makes the fingerprint depend on content
    and nothing else.
    """
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return sorted((_canon(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    return obj


def lineup_fingerprint(sub) -> str:
    """Dedup key for an archived snapshot: its full canonical content.

    NOT the set of player ids, which is what this used to be. `_pid_set`
    flattens every id in the subset into one set, so a player moving from
    `dressed` to `railScratches` leaves that set UNCHANGED — the snapshot keys
    identically to the previous one and is dropped as a duplicate.

    A late scratch is precisely the event this archive exists to capture (ledger
    row 10), and it was the one change the key could not see. Keying on content
    means any difference — a scratch, a goalie swap, a coach change — is
    archived, and an identical re-fetch still is not.
    """
    return json.dumps(_canon(sub), sort_keys=True, separators=(",", ":"))


def _pid_set(obj):
    """All player ids in a lineup subset (kept for reading archived records)."""
    ids = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("playerId", "id") and isinstance(v, int):
                ids.add(v)
            else:
                ids |= _pid_set(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, int):
                ids.add(v)
            else:
                ids |= _pid_set(v)
    return ids


def archive_lineups(upcoming, today_iso, deadline=None):
    """Snapshot game-day pre-game rosters to data/nhl_lineup_archive.jsonl.

    Why this exists: ledger row 10 (scratch-absence) was the most promising
    null — its deployable version needs ANNOUNCED lineups valued with RAPM
    weights, and nobody archives announced lineups historically. This logger
    captures them ourselves from October on so the row-10 reopen condition can
    eventually be tested. One JSON line per (game, fetch) with whatever roster
    info the landing payload exposes; a (gid, player-id-set) already archived
    is skipped, so the 6x/day CI cadence stores only genuine changes.

    Stops (what was archived stays archived) once `deadline` - a
    time.monotonic() value - passes, or after MAX_CONSEC_FAIL games in a row
    whose required `landing` request failed.
    """
    todays = [u for u in upcoming if u.get("d") == today_iso]
    if not todays:
        print("[nhl_update] 0 lineups archived")
        return 0
    seen = set()  # (gid, frozenset(player ids)) pairs already on disk
    if os.path.exists(LINEUP_ARCHIVE):
        for line in open(LINEUP_ARCHIVE, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            seen.add((rec.get("gid"), lineup_fingerprint(rec.get("lineup"))))
    n = fails = 0
    with open(LINEUP_ARCHIVE, "a", encoding="utf-8") as fh:
        for u in todays:
            if deadline is not None and time.monotonic() >= deadline:
                print("[nhl_update] lineup archive: time budget spent, stopping")
                break
            if fails >= MAX_CONSEC_FAIL:
                print(f"[nhl_update] lineup archive: {fails} games in a row failed, stopping")
                break
            # VERIFIED 2026-07-31 on a FINAL game: `landing` carries NEITHER
            # scratches NOR the dressed roster. Scratches live in `right-rail`
            # (gameInfo.{home,away}Team.scratches) and the dressed roster in
            # `play-by-play` (rosterSpots). Fetching landing alone -- the
            # original implementation -- archived only a goalie-stats widget,
            # i.e. none of the information this archive exists to capture.
            payload, rail, pbp = None, None, None
            for name, url, required in (("landing", LANDING, True),
                                        ("right-rail", RIGHT_RAIL, False),
                                        ("play-by-play", PLAY_BY_PLAY, False)):
                try:
                    d = get(url.format(u["id"]))
                except Exception as ex:  # noqa: BLE001
                    print(f"[nhl_update] {name} fetch failed for {u['id']}: {ex}")
                    if required:
                        d = None
                        break
                    d = None
                if name == "landing":
                    payload = d
                elif name == "right-rail":
                    rail = d
                else:
                    pbp = d
                time.sleep(0.15)
            if payload is None:
                fails += 1
                continue
            fails = 0
            sub = _lineup_subset(payload, rail, pbp)
            key = (u["id"], lineup_fingerprint(sub))
            if not sub or key in seen:
                continue
            seen.add(key)
            # append-only jsonl: a partial line at worst, never a truncation
            fh.write(json.dumps({
                "gid": u["id"],
                "t_fetch": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"),
                "home": u.get("home"), "away": u.get("away"),
                "state": payload.get("gameState"),
                "lineup": sub}) + "\n")
            n += 1
            time.sleep(0.2)
    print(f"[nhl_update] {n} lineups archived")
    return n


def collect_week(data: dict, seen: set, today_iso: str) -> tuple[list, list]:
    """One /schedule/{date} answer -> (new finals for the spine, unplayed games).

    Finals (OFF/FINAL) not yet in `seen` become spine rows (and are added to
    `seen`). Unplayed games dated today (US-Eastern) or later are the slate;
    a game in progress (LIVE/CRIT) is kept whatever its date, so a game that
    started before midnight is not lost from the slate until it is final.
    """
    new_rows, upcoming = [], []
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
            elif state in UNPLAYED_STATES and gdate and (
                    gdate >= today_iso or state in ("LIVE", "CRIT")):
                upcoming.append({
                    "id": gid, "d": gdate, "season": g.get("season"),
                    "playoff": 1 if gtype == 3 else 0,
                    "home": (h.get("abbrev") or "").strip(),
                    "away": (a.get("abbrev") or "").strip(),
                    "t": (g.get("startTimeUTC") or ""),
                    "state": state})
    return new_rows, upcoming


def main() -> int:
    t0 = time.monotonic()
    rows = list(csv.DictReader(open(SPINE, encoding="utf-8")))
    seen = {int(r["game_id"]) for r in rows}
    last_date = max(r["date"] for r in rows)
    today = et_today()
    start = min(dt.date.fromisoformat(last_date) - dt.timedelta(days=3), today)
    end = today + dt.timedelta(days=10)

    new_rows, upcoming = [], []
    cursor = start.isoformat()
    n_req = 0
    while cursor <= end.isoformat() and n_req < 12:
        if time.monotonic() - t0 >= SCHEDULE_BUDGET_S:
            print(f"[nhl_update] schedule walk stopped at {cursor}: time budget spent")
            break
        try:
            data = get(API.format(cursor))
        except Exception as ex:  # noqa: BLE001
            print(f"[nhl_update] fetch failed at {cursor}: {ex}")
            break
        n_req += 1
        nr, up = collect_week(data, seen, today.isoformat())
        new_rows += nr
        upcoming += up
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
    # temp + os.replace, like the spine: a killed run never leaves half a slate
    with open(UPCOMING + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(sorted(ded.values(), key=lambda u: (u["d"], u["id"])), fh, indent=0)
    os.replace(UPCOMING + ".tmp", UPCOMING)
    print(f"[nhl_update] +{len(new_rows)} finals, {len(ded)} upcoming "
          f"({n_req} requests)")
    # pre-game snapshots only: an in-progress game's roster is not a lineup
    # announcement (the archive's purpose), and it was never archived before
    archive_lineups([u for u in ded.values() if u.get("state") not in ("LIVE", "CRIT")],
                    today.isoformat(), deadline=t0 + ARCHIVE_BUDGET_S)
    try:
        import nhl_site_fetch
        left = UPDATE_BUDGET_S - (time.monotonic() - t0)
        if left < 15:
            print(f"[nhl_update] site fetch skipped: {left:.0f}s of the time budget left; "
                  f"display caches kept")
        else:
            nhl_site_fetch.main(budget_s=min(nhl_site_fetch.BUDGET_S, left))
    except Exception as ex:  # noqa: BLE001  (display data never fails the spine step)
        print(f"[nhl_update] site fetch failed: {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
