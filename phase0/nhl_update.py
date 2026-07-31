"""Incremental NHL refresh: append new final scores to the spine and capture the
upcoming slate — the light daily step (a handful of API calls, not the 500-request
full pull).

  data/nhl_games.csv      += completed games since the last row (deduped)
  data/nhl_upcoming.json   = scheduled games in the next 10 days (for serving
                             predictions in-season; empty file in the offseason)
  data/nhl_lineup_archive.jsonl += game-day pre-game roster/scratch snapshots
                             from the gamecenter landing endpoint (see
                             archive_lineups)

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
LANDING = "https://api-web.nhle.com/v1/gamecenter/{}/landing"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
SPINE = "data/nhl_games.csv"
UPCOMING = "data/nhl_upcoming.json"
LINEUP_ARCHIVE = "data/nhl_lineup_archive.jsonl"


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=45).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _lineup_subset(payload):
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
    return out


def _pid_set(obj):
    """All player ids in a lineup subset (dedup key: same set = same line)."""
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


def archive_lineups(upcoming, today_iso):
    """Snapshot game-day pre-game rosters to data/nhl_lineup_archive.jsonl.

    Why this exists: ledger row 10 (scratch-absence) was the most promising
    null — its deployable version needs ANNOUNCED lineups valued with RAPM
    weights, and nobody archives announced lineups historically. This logger
    captures them ourselves from October on so the row-10 reopen condition can
    eventually be tested. One JSON line per (game, fetch) with whatever roster
    info the landing payload exposes; a (gid, player-id-set) already archived
    is skipped, so the 6x/day CI cadence stores only genuine changes.
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
            seen.add((rec.get("gid"), frozenset(_pid_set(rec.get("lineup")))))
    n = 0
    with open(LINEUP_ARCHIVE, "a", encoding="utf-8") as fh:
        for u in todays:
            try:
                payload = get(LANDING.format(u["id"]))
            except Exception as ex:  # noqa: BLE001
                print(f"[nhl_update] landing fetch failed for {u['id']}: {ex}")
                continue
            sub = _lineup_subset(payload)
            key = (u["id"], frozenset(_pid_set(sub)))
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
    archive_lineups(list(ded.values()), today.isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
