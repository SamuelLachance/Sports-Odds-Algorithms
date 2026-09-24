"""Regular-season timing evidence from the Internet Archive: when does the
official NHL Playing Roster (RO) report first mark the starting goalies?

Today's live probe (bt_nhl_starter_probe.py) can only watch preseason games.
The Wayback Machine holds sporadic game-day captures of the RO report from
2022-23..2025-26 regular seasons and playoffs; a capture made before puck drop
shows what the report said at that moment, and its footer carries the report's
own generation stamp (venue-local time). This script parses a directory of such
captures (fetched once, politely, via web.archive.org/web/{ts}id_/{url}), joins
each to its scheduled start (api-web /v1/schedule/{date}, cached next to the
captures) and reports, per pre-game capture, minutes before the scheduled start
and how many teams already had a bold (starting) goalie.

Structure only: no model metric, no outcome, no odds. Reading a TEST-era
report's layout is not a TEST metric (rule 4 concerns model evaluation).

    python -X utf8 phase0/bt_nhl_starter_wayback.py --snap-dir DIR
    -> data/bt_nhl_starter_wayback.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_nhl_starter_feed import parse_playing_roster, starters_from_report  # noqa: E402

OUT = "data/bt_nhl_starter_wayback.json"
SCHED = "https://api-web.nhle.com/v1/schedule/{}"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}


def _offset(s):
    sign = -1 if s.startswith("-") else 1
    h, m = s.lstrip("+-").split(":")
    return dt.timedelta(hours=sign * int(h), minutes=sign * int(m))


def schedule_for(date, cache_dir):
    fn = os.path.join(cache_dir, f"sched_{date}.json")
    if not os.path.exists(fn):
        req = urllib.request.Request(SCHED.format(date), headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=20).read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        open(fn, "wb").write(raw)
        time.sleep(0.4)
    d = json.load(open(fn, encoding="utf-8"))
    out = {}
    for wk in d.get("gameWeek", []):
        for g in wk.get("games", []):
            out[str(g["id"])] = g
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", required=True)
    a = ap.parse_args(argv)
    rows = []
    for fn in sorted(glob.glob(os.path.join(a.snap_dir, "*.htm"))):
        gid, ts = os.path.basename(fn)[:-4].split("_")
        p = parse_playing_roster(open(fn, "rb").read())
        if p["started"] or p["final"]:
            continue                       # captured after puck drop: not pre-game evidence
        s = starters_from_report(p)
        cap = dt.datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
        gen_local = dt.datetime.strptime(p["gen_local"], "%Y-%m-%d-%H.%M.%S") if p["gen_local"] else None
        date = gen_local.date().isoformat() if gen_local else cap.date().isoformat()
        g = schedule_for(date, a.snap_dir).get(gid)
        if g is None:
            continue
        start = dt.datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
        gen_utc = None
        if gen_local:
            # stamp is venue-local: pick the venue offset, check it against the capture
            gv = (gen_local - _offset(g["venueUTCOffset"])).replace(tzinfo=dt.timezone.utc)
            ge = (gen_local - _offset(g["easternUTCOffset"])).replace(tzinfo=dt.timezone.utc)
            gen_utc = gv if gv <= cap else ge
        rows.append({
            "gid": int(gid), "type": int(str(gid)[4:6]),
            "matchup": f"{g['awayTeam']['abbrev']}@{g['homeTeam']['abbrev']}",
            "capture_utc": cap.isoformat(), "gen_local": p["gen_local"],
            "start_utc": g["startTimeUTC"], "venue_offset": g["venueUTCOffset"],
            "gen_min_before_start": round((start - gen_utc).total_seconds() / 60, 1) if gen_utc else None,
            "capture_min_before_start": round((start - cap).total_seconds() / 60, 1),
            "n_tables": p["n_tables"],
            "dressed_goalies": {k: len(s[k]["dressed_goalies"]) for k in ("away", "home")},
            "bold_goalie": {k: s[k]["n_bold_goalies"] for k in ("away", "home")},
            "bold_skaters": {k: sum(x["bold"] for x in p[k]["dressed"] if x["pos"] != "G")
                             for k in ("away", "home")},
            "wayback": f"https://web.archive.org/web/{ts}/https://www.nhl.com/scores/htmlreports/"
                       f"{str(gid)[:4]}{int(str(gid)[:4]) + 1}/RO{str(gid)[4:]}.HTM",
        })
    rows.sort(key=lambda r: -(r["gen_min_before_start"] or 0))
    sides = [(r["gen_min_before_start"], r["bold_goalie"][k]) for r in rows for k in ("away", "home")
             if r["gen_min_before_start"] is not None]
    bins = {}
    for lo, hi in ((60, 1e9), (30, 60), (20, 30), (15, 20), (10, 15), (5, 10), (0, 5), (-1e9, 0)):
        sel = [b for m, b in sides if lo <= m < hi]
        bins[f"[{lo if lo > -1e8 else '-inf'},{hi if hi < 1e8 else 'inf'}) min before"] = {
            "team_sides": len(sel), "with_bold_goalie": sum(1 for b in sel if b == 1)}
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "protocol": {"source": "Internet Archive captures of nhl.com RO reports, "
                                  "2022-23..2025-26; pre-game = header shows no start time "
                                  "and no Final", "no_model_metric": True, "market_use": "none"},
           "n_pregame_captures": len(rows), "by_minutes_before_start": bins, "captures": rows}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(bins, indent=1))
    for r in rows:
        print(r["gid"], r["matchup"], "gen T-%s" % r["gen_min_before_start"],
              "cap T-%s" % r["capture_min_before_start"], "bold G", r["bold_goalie"],
              "bold sk", r["bold_skaters"], "tables", r["n_tables"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
