"""Parse per-batter-game wOBA components from Retrosheet events.

wOBA weights each offensive event by its run value (unlike our on-base rating,
which treats a walk and a homer the same, plus ISO for power). We need the event
breakdown per batter to build a wOBA rating and test whether it beats/adds to the
on-base + power pair. Output: data/retro_events/woba.csv
  game_id,batter,pa,bb,hbp,s1,s2,s3,hr
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

from parse_events import batting_outcome, _DOUBLE, _TRIPLE   # validated classifiers

OUT = "data/retro_events/woba.csv"


def parse_file(path: str, agg: dict):
    game_id = None
    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        rec = f[0]
        if rec == "id":
            game_id = f[1]
        elif rec == "play":
            batter = f[3]
            ev = f[6] if len(f) > 6 else ""
            oc = batting_outcome(ev)
            if oc not in ("HR", "H", "BB", "HBP", "REACH", "SO", "OUT"):
                continue
            a = agg[(game_id, batter)]
            a[0] += 1                                   # PA
            if oc == "BB":
                a[1] += 1
            elif oc == "HBP":
                a[2] += 1
            elif oc == "HR":
                a[6] += 1
            elif oc == "H":
                if _DOUBLE.match(ev.strip()):
                    a[4] += 1
                elif _TRIPLE.match(ev.strip()):
                    a[5] += 1
                else:
                    a[3] += 1                           # single


def main():
    agg: dict = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])   # pa,bb,hbp,s1,s2,s3,hr
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    for i, path in enumerate(files):
        parse_file(path, agg)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "batter", "pa", "bb", "hbp", "s1", "s2", "s3", "hr"])
        for (gid, bat), v in agg.items():
            w.writerow([gid, bat] + v)
    tot = [sum(v[k] for v in agg.values()) for k in range(7)]
    print(f"wrote {OUT}: {len(agg):,} batter-games", flush=True)
    print(f"  totals PA={tot[0]:,} BB={tot[1]:,} HBP={tot[2]:,} 1B={tot[3]:,} 2B={tot[4]:,} 3B={tot[5]:,} HR={tot[6]:,}", flush=True)


if __name__ == "__main__":
    main()
