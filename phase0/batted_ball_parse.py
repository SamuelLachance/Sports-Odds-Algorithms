"""Parse batted-ball type per pitcher-start (ground/fly/pop), for SIERA.

SIERA uses net ground-ball rate = (GB - FB - PU)/PA plus K% and BB%, with squares
and interactions. We need the batted-ball breakdown a pitcher allowed. From the
Retrosheet trajectory modifiers on non-HR balls in play: /G ground, /F fly, /P pop
(line drives /L are neither GB nor FB in netGB). Output:
  data/retro_events/battedball.csv  game_id,pitcher,gb,fb,pu
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

from parse_events import batting_outcome

OUT = "data/retro_events/battedball.csv"


def parse_file(path: str, agg: dict):
    game_id = None
    cur = {0: None, 1: None}
    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        rec = f[0]
        if rec == "id":
            game_id = f[1]
            cur = {0: None, 1: None}
        elif rec in ("start", "sub"):
            if f[5] == "1":
                cur[int(f[3])] = f[1]
        elif rec == "play":
            bat_side = int(f[2])
            ev = f[6] if len(f) > 6 else ""
            oc = batting_outcome(ev)
            if oc not in ("H", "OUT", "REACH"):     # non-HR balls in play only
                continue
            pit = cur[1 - bat_side]
            if pit is None:
                continue
            a = agg[(game_id, pit)]
            if "/G" in ev:
                a[0] += 1
            elif "/F" in ev or "/SF" in ev:
                a[1] += 1
            elif "/P" in ev:
                a[2] += 1


def main():
    agg: dict = defaultdict(lambda: [0, 0, 0])   # gb, fb, pu
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    for i, path in enumerate(files):
        parse_file(path, agg)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "pitcher", "gb", "fb", "pu"])
        for (gid, pit), v in agg.items():
            w.writerow([gid, pit] + v)
    tot = [sum(v[k] for v in agg.values()) for k in range(3)]
    print(f"wrote {OUT}: {len(agg):,} pitcher-games  GB={tot[0]:,} FB={tot[1]:,} PU={tot[2]:,}", flush=True)


if __name__ == "__main__":
    main()
