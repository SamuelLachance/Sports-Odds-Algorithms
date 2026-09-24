"""Summarise data/bt_nhl_starter_probe.jsonl: when did each source first name
the starting goalies, relative to the scheduled puck drop?

Reads only the probe log written by phase0/bt_nhl_starter_probe.py (no network).
For every probed game and every signal it reports the first poll at which the
signal was present (minutes before the scheduled start; negative = after) and
the last poll at which it was still absent, so the true appearance time is
bracketed by the polling cadence.

Signals
  pbp_roster      play-by-play rosterSpots non-empty (dressed game roster)
  pbp_goalies     play-by-play rosterSpots lists goalies for both teams
  box_goalies     boxscore playerByGameStats lists goalies for both teams
  box_starter     boxscore marks exactly one goalie per team starter=true
  ro_posted       nhl.com Playing Roster report (RO) returns 200
  ro_bold_goalie  RO shows a bold (starting-lineup) goalie for both teams
  landing_kw      landing has any key mentioning probable/starter/confirmed
  gc_changed      landing goalieComparison leader list differs from first poll
  state_PRE       landing gameState == PRE
  state_LIVE      landing gameState in LIVE/CRIT

    python -X utf8 phase0/bt_nhl_starter_timing.py
    -> data/bt_nhl_starter_timing.json
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict

PROBE = "data/bt_nhl_starter_probe.jsonl"
OUT = "data/bt_nhl_starter_timing.json"


def signals(rec, first_gc):
    lp = rec.get("landing") or {}
    pb = rec.get("play-by-play") or {}
    bx = rec.get("boxscore") or {}
    ro = rec.get("ro") or {}
    pg, bg = pb.get("goalies") or {}, bx.get("goalies") or {}
    both = lambda d: bool(d.get("homeTeam")) and bool(d.get("awayTeam"))  # noqa: E731
    starters = {s: [x for x in (bg.get(s) or []) if x[1] is True] for s in ("homeTeam", "awayTeam")}
    return {
        "pbp_roster": (pb.get("n_roster") or 0) > 0,
        "pbp_goalies": both(pg),
        "box_goalies": both(bg),
        "box_starter": all(len(v) == 1 for v in starters.values()),
        "ro_posted": ro.get("status") == 200,
        "ro_bold_goalie": (ro.get("n_bold_goalies") or 0) >= 2,
        "landing_kw": bool(lp.get("kw")),
        "gc_changed": first_gc is not None and lp.get("gc_leaders") not in (None, first_gc),
        "state_PRE": lp.get("state") == "PRE",
        "state_LIVE": lp.get("state") in ("LIVE", "CRIT"),
    }


def main():
    recs = defaultdict(list)
    for line in open(PROBE, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs[r["gid"]].append(r)
    out = {}
    for gid, rs in sorted(recs.items()):
        rs.sort(key=lambda r: r["t_fetch"])
        first_gc = (rs[0].get("landing") or {}).get("gc_leaders")
        per = {}
        for r in rs:
            sg = signals(r, first_gc)
            for k, v in sg.items():
                e = per.setdefault(k, {"first_present_min_before": None,
                                       "last_absent_min_before": None})
                if v and e["first_present_min_before"] is None:
                    e["first_present_min_before"] = r["min_to_start"]
                if not v and e["first_present_min_before"] is None:
                    e["last_absent_min_before"] = r["min_to_start"]
        last = rs[-1]
        out[str(gid)] = {
            "n_polls": len(rs),
            "first_poll_min_before": rs[0]["min_to_start"],
            "last_poll_min_before": last["min_to_start"],
            "signals": per,
            "final_view": {"pbp_goalies": (last.get("play-by-play") or {}).get("goalies"),
                           "box_goalies": (last.get("boxscore") or {}).get("goalies"),
                           "ro_dressed_goalies": (last.get("ro") or {}).get("dressed_goalies"),
                           "gc_leaders_first": first_gc,
                           "gc_leaders_last": (last.get("landing") or {}).get("gc_leaders")},
        }
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "note": "minutes before the SCHEDULED start (negative = after); the true "
                   "appearance time lies between last_absent and first_present",
           "games": out}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for gid, g in out.items():
        print(gid, g["n_polls"], "polls")
        for k, e in g["signals"].items():
            print(f"   {k:15s} absent@{e['last_absent_min_before']}  present@{e['first_present_min_before']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
