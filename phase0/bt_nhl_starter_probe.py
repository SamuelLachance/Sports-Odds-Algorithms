"""Live timing probe: WHEN does each public source name tonight's starting goalies?

Breakthrough program, serving-feasibility task (documents/
breakthrough_program_prereg_2026_09_24.md). The round-1 anatomy puts the NHL gap
to the close on nights a team starts a non-#1 goalie, so a goalie-aware model is
only servable if the confirmed starter can be read programmatically BEFORE puck
drop. This script measures that, on live games, instead of assuming it.

For each game id it polls, on a light schedule (every SLOW_MIN minutes until
FAST_FROM_MIN before puck drop, every FAST_MIN minutes from then until
STOP_AFTER_MIN after the scheduled start):

  api-web  gamecenter/{id}/landing       matchup.goalieComparison / goalieSeasonStats
  api-web  gamecenter/{id}/boxscore      playerByGameStats.*.goalies[].starter
  api-web  gamecenter/{id}/play-by-play  rosterSpots (dressed roster, goalies)
  api-web  gamecenter/{id}/right-rail    gameInfo.*.scratches
  nhl.com  scores/htmlreports/{season}/RO{gg}.HTM  official Playing Roster
           ("* Starting Lineup in Bold" -> the starting goalie)

and appends one compact JSON line per (game, poll) to OUT (default
data/bt_nhl_starter_probe.jsonl): minutes to scheduled puck drop, gameState,
and what each source says about goalies at that moment. Raw payloads are kept
only when --raw-dir is given (scratch space, never the repo).

Market-blind: no odds endpoint is touched. Standard library only.
Request volume: 5 requests per game per poll; a 7-hour watch of one game is
~30 polls (~150 requests), spaced minutes apart.

    python phase0/bt_nhl_starter_probe.py 2026010040 2026010042 ... [--raw-dir D]
    python phase0/bt_nhl_starter_probe.py --once 2026010049 ...   (single snapshot)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import html as htmlmod
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
GC = "https://api-web.nhle.com/v1/gamecenter/{}/{}"
RO = "https://www.nhl.com/scores/htmlreports/{}/RO{}.HTM"
OUT = "data/bt_nhl_starter_probe.jsonl"

SLOW_MIN = 30          # minutes between polls far from puck drop
FAST_FROM_MIN = 100    # switch to the fast cadence this many minutes before
FAST_MIN = 5
STOP_AFTER_MIN = 20    # last poll this many minutes after the scheduled start
REQ_TIMEOUT = 20
KEYWORDS = ("probab", "starter", "starting", "confirm", "projected", "expected")


def fetch(url):
    """(status, bytes|None). 404 is an answer (report not posted yet), not an error."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as ex:
        return ex.code, None
    except Exception as ex:  # noqa: BLE001
        return f"ERR {type(ex).__name__}", None
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return status, raw


def keyword_paths(obj, path=""):
    """Every JSON key path whose key name mentions probable/starter/confirmed/..."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            if any(w in k.lower() for w in KEYWORDS):
                hits.append(p)
            hits += keyword_paths(v, p)
    elif isinstance(obj, list):
        for v in obj:
            hits += keyword_paths(v, path + "[]")
    return sorted(set(hits))


def x_landing(d):
    m = d.get("matchup") or {}
    gc = m.get("goalieComparison") or {}
    out = {"state": d.get("gameState"),
           "matchup_keys": sorted(m.keys()),
           "gc_context": [gc.get("contextLabel"), gc.get("contextSeason")],
           "gc_leaders": {s: [g.get("playerId") for g in (gc.get(s) or {}).get("leaders") or []]
                          for s in ("homeTeam", "awayTeam")},
           "gss_ids": [g.get("playerId") for g in
                       (m.get("goalieSeasonStats") or {}).get("goalies") or []],
           "has_summary": bool(d.get("summary")),
           "kw": keyword_paths(d)}
    return out


def x_box(d):
    pg = d.get("playerByGameStats") or {}
    return {"state": d.get("gameState"),
            "goalies": {s: [[g.get("playerId"), g.get("starter"), g.get("toi")]
                            for g in (pg.get(s) or {}).get("goalies") or []]
                        for s in ("homeTeam", "awayTeam")},
            "n_skaters": {s: sum(len((pg.get(s) or {}).get(k) or [])
                                 for k in ("forwards", "defense"))
                          for s in ("homeTeam", "awayTeam")},
            "kw": keyword_paths(d)}


def x_pbp(d):
    rs = d.get("rosterSpots") or []
    hid = (d.get("homeTeam") or {}).get("id")
    goalies = {"homeTeam": [], "awayTeam": []}
    for p in rs:
        if p.get("positionCode") == "G":
            goalies["homeTeam" if p.get("teamId") == hid else "awayTeam"].append(p.get("playerId"))
    return {"state": d.get("gameState"), "n_roster": len(rs), "goalies": goalies,
            "n_plays": len(d.get("plays") or []), "kw": keyword_paths(d)}


def x_rail(d):
    gi = d.get("gameInfo") or {}
    return {"n_scratch": {s: len((gi.get(s) or {}).get("scratches") or [])
                         for s in ("homeTeam", "awayTeam")},
            "n_refs": len(gi.get("referees") or []),
            "keys": sorted(d.keys()), "kw": keyword_paths(d)}


def x_ro(raw):
    """Official Playing Roster: dressed goalies per team and the bold (starting) one."""
    t = raw.decode("utf-8", errors="replace")
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    status_words = [w for w in ("Final", "In Progress", "End of Period") if w in t]
    teams, cur, scratch = [], None, False
    out = {"teams": [], "status_words": status_words,
           "starting_lineup_note": "Starting Lineup in Bold" in t}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S):
        cells = [htmlmod.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)]
        if not cells:
            continue
        if len(cells) == 1 and cells[0].startswith("Scratches"):
            scratch = True
            continue
        if len(cells) >= 3 and cells[1] == "G":
            teams.append({"num": cells[0], "name": cells[2], "bold": "bold" in row,
                          "scratch": scratch})
    # the report prints visitor then home in two column blocks per row; we only need
    # whether ANY dressed goalie is bold and how many are
    dressed = [g for g in teams if not g["scratch"]]
    out["dressed_goalies"] = [[g["num"], g["name"], g["bold"]] for g in dressed]
    out["n_bold_goalies"] = sum(g["bold"] for g in dressed)
    return out


def poll_game(gid, start_utc, raw_dir=None):
    now = dt.datetime.now(dt.timezone.utc)
    rec = {"gid": gid, "t_fetch": now.isoformat(timespec="seconds"),
           "min_to_start": round((start_utc - now).total_seconds() / 60.0, 1)}
    for ep, fn in (("landing", x_landing), ("boxscore", x_box),
                   ("play-by-play", x_pbp), ("right-rail", x_rail)):
        st, raw = fetch(GC.format(gid, ep))
        if raw is None:
            rec[ep] = {"status": st}
        else:
            try:
                d = json.loads(raw)
                rec[ep] = {"status": st, "sha": hashlib.sha1(raw).hexdigest()[:10], **fn(d)}
            except Exception as ex:  # noqa: BLE001
                rec[ep] = {"status": st, "parse_error": str(ex)}
            if raw_dir:
                p = os.path.join(raw_dir, f"{gid}_{ep}_{now:%H%M}.json")
                open(p, "wb").write(raw)
        time.sleep(0.4)
    season = str(gid)[:4]
    season = f"{season}{int(season) + 1}"
    st, raw = fetch(RO.format(season, str(gid)[4:]))
    if raw is None:
        rec["ro"] = {"status": st}
    else:
        rec["ro"] = {"status": st, "sha": hashlib.sha1(raw).hexdigest()[:10], **x_ro(raw)}
        if raw_dir:
            open(os.path.join(raw_dir, f"{gid}_RO_{now:%H%M}.htm"), "wb").write(raw)
    time.sleep(0.4)
    return rec


def start_times(gids):
    """gid -> scheduled start (UTC) from the landing endpoint (one request per game)."""
    out = {}
    for gid in gids:
        st, raw = fetch(GC.format(gid, "landing"))
        d = json.loads(raw)
        out[gid] = dt.datetime.fromisoformat(d["startTimeUTC"].replace("Z", "+00:00"))
        time.sleep(0.3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gids", nargs="+", type=int)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--raw-dir")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if a.raw_dir:
        os.makedirs(a.raw_dir, exist_ok=True)
    starts = start_times(a.gids)
    nxt = {g: dt.datetime.now(dt.timezone.utc) for g in a.gids}
    while nxt:
        now = dt.datetime.now(dt.timezone.utc)
        due = [g for g, t in nxt.items() if t <= now]
        for g in due:
            rec = poll_game(g, starts[g], a.raw_dir)
            with open(a.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pb, bx, ro = rec["play-by-play"], rec["boxscore"], rec["ro"]
            print(f"{rec['t_fetch']} {g} T{-rec['min_to_start']:+.0f}m "
                  f"state={rec['landing'].get('state')} pbp_roster={pb.get('n_roster')} "
                  f"pbp_G={pb.get('goalies')} box_G={bx.get('goalies')} "
                  f"RO={ro.get('status')} RO_bold={ro.get('n_bold_goalies')}", flush=True)
            mins = (starts[g] - dt.datetime.now(dt.timezone.utc)).total_seconds() / 60.0
            if a.once or mins < -STOP_AFTER_MIN:
                del nxt[g]
            else:
                step = FAST_MIN if mins <= FAST_FROM_MIN else SLOW_MIN
                # never sleep past the start of the fast window
                t_next = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=step)
                fast_start = starts[g] - dt.timedelta(minutes=FAST_FROM_MIN)
                if mins > FAST_FROM_MIN and t_next > fast_start:
                    t_next = fast_start
                nxt[g] = t_next
        if nxt:
            wait = (min(nxt.values()) - dt.datetime.now(dt.timezone.utc)).total_seconds()
            time.sleep(max(5.0, min(wait, 300.0)))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
