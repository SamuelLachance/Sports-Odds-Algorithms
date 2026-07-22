"""Parse Retrosheet event files into per-pitcher-appearance FIP components.

FIP needs only HR, BB, HBP, SO and innings (outs) — all directly readable from the
event stream, and none of them requires reconstructing earned runs, which is the
part of ERA that events make painful. FIP is also defense-independent, which is
exactly the property a pitcher-quality signal should have.

Output: data/retro_events/appearances.csv with one row per (game, pitcher):
    game_id, date, season, team, opp, is_home, pitcher, is_starter,
    outs, bf, H, HR, BB, HBP, SO

Correctness is not assumed. validate_events.py cross-checks the per-pitcher sums
against the game-log team totals (team SO/BB/HR/H allowed == opponent batting),
so any parser error shows up as a mismatch on thousands of games.
"""

from __future__ import annotations

import csv
import glob
import re
import sys

OUT = "data/retro_events/appearances.csv"

# --- event-string classification ------------------------------------------
# The event is the last comma-field of a play record. Its leading token before
# any '/', '.', or modifier decides the batting outcome.
_HR = re.compile(r"^(HR|H)(\d|/|$)")          # "HR", "HR/F8", or bare "H" (inside-park in old data)
_SINGLE = re.compile(r"^S(\d|/|$)")
_DOUBLE = re.compile(r"^D(\d|/|G|$)")          # D, D7, DGR (ground-rule)
_TRIPLE = re.compile(r"^T(\d|/|$)")
_WALK = re.compile(r"^(W|IW|I)(\+|\.|/|;|$)")  # unintentional / intentional
_HBP = re.compile(r"^HP")
_K = re.compile(r"^K")                          # strikeout (may be "K23", "K+WP", "K+SB2")


def batting_outcome(ev: str) -> str:
    """Return one of: HR, H (non-HR hit), BB, HBP, SO, OUT, NB (no batter / other)."""
    e = ev.strip()
    if not e:
        return "NB"
    # Non-batter events (baserunning only): stolen base, caught stealing, etc.
    if e[:2] in ("SB", "CS", "PO", "PB", "WP", "DI", "OA", "BK") or e[0] in ("+",):
        return "NB"
    if e.startswith("NP"):        # no play (substitution marker)
        return "NB"
    if _HBP.match(e):
        return "HBP"
    if _HR.match(e):
        return "HR"
    if _SINGLE.match(e) or _DOUBLE.match(e) or _TRIPLE.match(e):
        return "H"
    if _WALK.match(e):
        return "BB"
    if _K.match(e):
        return "SO"
    if e.startswith(("E",)):      # reached on error — a batter event, not an out, not a hit
        return "REACH"
    if e.startswith(("FC", "FLE")):  # fielder's choice / foul error
        return "OUT"
    # fielded outs begin with a fielder digit (e.g. "8", "63", "6/P"), also SF/SH
    if e[0].isdigit() or e.startswith(("SF", "SH")):
        return "OUT"
    return "OTHER"


def outs_on_play(ev: str, outcome: str) -> int:
    """Best-effort out count for a play. Validated in aggregate, not per-play."""
    e = ev.strip()
    if outcome == "SO":
        # dropped-third-strike that reaches base: "K+...B-..." with no out — rare
        n = 1
        if "+" in e and any(t in e for t in ("PB", "WP", "E")):
            # batter may reach; conservatively still count the K as an out only
            # if not explicitly reaching. Retrosheet marks reach via 'K+...B-1'
            if re.search(r"K\+.*B-", e):
                n = 0
        return n
    if outcome in ("HR", "H", "BB", "HBP", "REACH", "NB", "OTHER"):
        # baserunning outs can still occur on these (e.g. runner thrown out),
        # captured below via the advance string.
        base = 0
    else:  # OUT
        base = 1
        if re.search(r"[GLF]?DP", e) or re.search(r"\(1\)|\(2\)|\(3\)", e.split("/")[0]):
            base = 2
        if "TP" in e:
            base = 3
    # additional outs recorded on the bases: "CS2(24)", "(B)" etc. in advances
    extra = len(re.findall(r"\(\d\)", e)) if base == 0 else max(0, len(re.findall(r"\(\d\)", e)) - (base - 1))
    return base + extra


def parse_file(path: str, rows: list):
    game_id = date = season = None
    vis = home = None
    cur_pitcher = {0: None, 1: None}       # team 0 (vis), 1 (home)
    starter = {0: None, 1: None}
    stat: dict = {}                        # pitcher_id -> counters

    def flush():
        if game_id is None:
            return
        for pid, s in stat.items():
            team = home if s["side"] == 1 else vis
            opp = vis if s["side"] == 1 else home
            rows.append([game_id, date, season, team, opp, s["side"],
                         pid, int(pid == starter[s["side"]]),
                         s["outs"], s["bf"], s["H"], s["HR"], s["BB"], s["HBP"], s["SO"]])

    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        rec = f[0]
        if rec == "id":
            flush()
            game_id = f[1]
            season = int(f[1][3:7])
            date = None
            vis = home = None
            cur_pitcher = {0: None, 1: None}
            starter = {0: None, 1: None}
            stat = {}
        elif rec == "info":
            if f[1] == "visteam":
                vis = f[2]
            elif f[1] == "hometeam":
                home = f[2]
            elif f[1] == "date":
                date = f[2].replace("/", "")
        elif rec in ("start", "sub"):
            pid, side, pos = f[1], int(f[3]), f[5]
            if pos == "1":                 # pitcher
                cur_pitcher[side] = pid
                if rec == "start":
                    starter[side] = pid
                stat.setdefault(pid, dict(side=side, outs=0, bf=0, H=0, HR=0, BB=0, HBP=0, SO=0))
        elif rec == "play":
            bat_side = int(f[2])
            ev = f[6] if len(f) > 6 else ""
            pit_side = 1 - bat_side
            pid = cur_pitcher[pit_side]
            if pid is None:
                continue
            s = stat.setdefault(pid, dict(side=pit_side, outs=0, bf=0, H=0, HR=0, BB=0, HBP=0, SO=0))
            oc = batting_outcome(ev)
            if oc != "NB":
                s["bf"] += 1
            if oc == "HR":
                s["HR"] += 1
                s["H"] += 1
            elif oc == "H":
                s["H"] += 1
            elif oc == "BB":
                s["BB"] += 1
            elif oc == "HBP":
                s["HBP"] += 1
            elif oc == "SO":
                s["SO"] += 1
            s["outs"] += outs_on_play(ev, oc)
    flush()


def main():
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    rows = []
    for i, path in enumerate(files):
        parse_file(path, rows)
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(files)} files, {len(rows):,} appearances", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "team", "opp", "is_home",
                    "pitcher", "is_starter", "outs", "bf", "H", "HR", "BB", "HBP", "SO"])
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows):,} appearances", flush=True)


if __name__ == "__main__":
    main()
