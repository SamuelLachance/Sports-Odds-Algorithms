"""Parse Retrosheet events into plate appearances + starting lineups.

For the batter-vs-pitcher rating idea: every plate appearance is one batter vs one
pitcher with a discrete outcome. We emit that atomic matchup plus each game's
batting order, so a per-PA rating system can be walked chronologically and a
lineup-vs-starter feature computed per game.

Outputs:
  data/retro_events/pa.csv       game_id,date,season,bat_side,batter,pitcher,on_base
  data/retro_events/lineups.csv  game_id,side,order,batter   (starters, order 1-9)

on_base = the batter reached (hit, walk, HBP, reached-on-error) vs was retired.
It is deliberately a simple binary — a clean 1v1 win/loss for the rating system.
Validated by comparing per-game PA counts to the game-log at-bats+walks tallies.
"""

from __future__ import annotations

import csv
import glob

from parse_events import batting_outcome   # reuse the validated classifier

PA_OUT = "data/retro_events/pa.csv"
LINE_OUT = "data/retro_events/lineups.csv"

# outcomes that put the batter on base
_ON_BASE = {"H", "HR", "BB", "HBP", "REACH"}
# outcomes that are a completed PA the batter did not reach on (an out)
_OUT = {"SO", "OUT"}


def parse_file(path: str, pa_rows: list, line_rows: list):
    game_id = date = season = None
    cur_pitcher = {0: None, 1: None}
    recorded_start = False

    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        rec = f[0]
        if rec == "id":
            game_id = f[1]
            season = int(f[1][3:7])
            date = None
            cur_pitcher = {0: None, 1: None}
            recorded_start = False
        elif rec == "info" and f[1] == "date":
            date = f[2].replace("/", "")
        elif rec in ("start", "sub"):
            pid, side, order, pos = f[1], int(f[3]), f[4], f[5]
            if pos == "1":
                cur_pitcher[side] = pid
            if rec == "start" and order not in ("0",):
                # batting order 1-9 = the lineup; capture starters only
                line_rows.append([game_id, side, int(order), pid])
        elif rec == "play":
            bat_side = int(f[2])
            batter = f[3]
            ev = f[6] if len(f) > 6 else ""
            oc = batting_outcome(ev)
            if oc in _ON_BASE or oc in _OUT:
                pit = cur_pitcher[1 - bat_side]
                if pit is not None:
                    pa_rows.append([game_id, date, season, bat_side, batter, pit,
                                    1 if oc in _ON_BASE else 0])


def main():
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    pa, line = [], []
    for i, path in enumerate(files):
        parse_file(path, pa, line)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}  PAs={len(pa):,}", flush=True)
    with open(PA_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "bat_side", "batter", "pitcher", "on_base"])
        w.writerows(pa)
    with open(LINE_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "side", "order", "batter"])
        w.writerows(line)
    obp = sum(r[6] for r in pa) / len(pa) if pa else 0
    print(f"wrote {PA_OUT}: {len(pa):,} PAs (on-base rate {obp:.3f})", flush=True)
    print(f"wrote {LINE_OUT}: {len(line):,} lineup slots", flush=True)


if __name__ == "__main__":
    main()
