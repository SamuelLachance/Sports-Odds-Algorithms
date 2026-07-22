"""Parse fly balls allowed per pitcher-start, for xFIP.

xFIP replaces a pitcher's actual home runs with EXPECTED home runs = (fly balls) x
(league HR/FB), because HR/FB is mostly luck for a pitcher. We need his fly-ball
count. Retrosheet events carry a batted-ball trajectory modifier (/F fly, /G ground,
/L line, /P pop, /SF sac fly), coded on ~96% of balls in play. A fly ball here is a
non-HR batted ball with a /F or /SF modifier (HRs are added back as fly balls in the
eval). Output: data/retro_events/flyballs.csv  game_id,pitcher,fb
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

from parse_events import batting_outcome   # reuse the validated classifier

OUT = "data/retro_events/flyballs.csv"


def parse_file(path: str, fb: dict):
    game_id = None
    cur = {0: None, 1: None}
    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        rec = f[0]
        if rec == "id":
            game_id = f[1]
            cur = {0: None, 1: None}
        elif rec in ("start", "sub"):
            if f[5] == "1":                       # position 1 = pitcher
                cur[int(f[3])] = f[1]
        elif rec == "play":
            bat_side = int(f[2])
            ev = f[6] if len(f) > 6 else ""
            oc = batting_outcome(ev)
            if oc in ("H", "OUT", "REACH") and ("/F" in ev or "/SF" in ev):
                pit = cur[1 - bat_side]
                if pit is not None:
                    fb[(game_id, pit)] += 1


def main():
    fb: dict = defaultdict(int)
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    for i, path in enumerate(files):
        parse_file(path, fb)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "pitcher", "fb"])
        for (gid, pit), n in fb.items():
            w.writerow([gid, pit, n])
    print(f"wrote {OUT}: {len(fb):,} pitcher-games, {sum(fb.values()):,} fly balls", flush=True)


if __name__ == "__main__":
    main()
