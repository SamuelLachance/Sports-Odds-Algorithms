"""Parse Retrosheet events into per-batter-game power lines (total bases).

pa.csv recorded only a binary on_base outcome, which is blind to power. This emits
each batter-game's PA / H / TB so a per-batter isolated-power rating can be built:
    ISO-per-PA = (TB - H) / PA = extra bases per plate appearance
which isolates power *beyond* the on-base ability the TrueSkill feature captures
(a walk and a home run are identical to on_base; here they differ by 4 bases).

Output: data/retro_events/power.csv   game_id,batter,pa,h,tb
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

from parse_events import batting_outcome, _DOUBLE, _TRIPLE   # reuse validated classifier

OUT = "data/retro_events/power.csv"


def total_bases(ev: str, oc: str) -> int:
    e = ev.strip()
    if oc == "HR":
        return 4
    if oc == "H":
        if _DOUBLE.match(e):
            return 2
        if _TRIPLE.match(e):
            return 3
        return 1            # single
    return 0


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
            # count every completed batter PA (hit, out, BB, HBP, reach-on-error)
            if oc in ("HR", "H", "BB", "HBP", "REACH", "SO", "OUT"):
                a = agg[(game_id, batter)]
                a[0] += 1                              # PA
                if oc in ("HR", "H"):
                    a[1] += 1                          # H
                    a[2] += total_bases(ev, oc)        # TB


def main():
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    agg: dict = defaultdict(lambda: [0, 0, 0])   # (game_id, batter) -> PA, H, TB
    for i, path in enumerate(files):
        parse_file(path, agg)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}  rows={len(agg):,}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "batter", "pa", "h", "tb"])
        for (gid, bat), (pa, h, tb) in agg.items():
            w.writerow([gid, bat, pa, h, tb])
    tot_pa = sum(v[0] for v in agg.values())
    tot_h = sum(v[1] for v in agg.values())
    tot_tb = sum(v[2] for v in agg.values())
    print(f"wrote {OUT}: {len(agg):,} batter-games, {tot_pa:,} PA", flush=True)
    print(f"  league SLG/PA={tot_tb/tot_pa:.3f}  ISO/PA={(tot_tb-tot_h)/tot_pa:.3f}", flush=True)


if __name__ == "__main__":
    main()
