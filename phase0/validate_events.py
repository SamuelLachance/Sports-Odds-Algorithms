"""Validate the event parser: per-pitcher sums must equal game-log team totals.

For each game, the sum over a team's pitchers of (SO, BB, H, HR) allowed must
equal the OPPONENT's batting totals recorded in the game log. The game log also
carries team strikeouts-by-pitchers and walks-by-pitchers directly. Any parser
bug in outcome classification shows up here as a mismatch rate.

Game-log field positions (0-indexed) for the stats we can check:
  visitor pitching: SO=45? -- we instead use the opponent BATTING totals, which
  are unambiguous: visitor batting H=24, HR=27, BB=31, SO=32
                    home    batting H=52, HR=55, BB=59, SO=60
So home pitchers allowed == visitor batting; visitor pitchers allowed == home batting.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

# game-log batting field indices (Retrosheet spec: vis block 21-48, home 49-76)
VB = {"H": 22, "HR": 25, "BB": 30, "SO": 32}     # visitor batting
HB = {"H": 50, "HR": 53, "BB": 58, "SO": 60}     # home batting


def load_gamelog_targets(year: int) -> dict:
    """game_id -> {'home':{allowed by home pitchers}, 'vis':{...}}."""
    out = {}
    path = f"data/retrosheet/gl{year}.txt"
    for r in csv.reader(open(path, encoding="latin-1")):
        if len(r) < 61:
            continue
        # Event id is HOMEteam + date + game-number. Game number is field 1.
        gid = r[6] + r[0] + r[1]
        # home pitchers allowed == visitor batting; vis pitchers allowed == home batting
        out[gid] = {
            "home": {k: int(r[VB[k]]) for k in VB},
            "vis": {k: int(r[HB[k]]) for k in HB},
        }
    return out


def main():
    ap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # gid -> side -> stat
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        side = "home" if r["is_home"] == "1" else "vis"
        for k in ("H", "HR", "BB", "SO"):
            ap[r["game_id"]][side][k] += int(r[k])

    years = sorted({int(g[3:7]) for g in ap})
    targets = {}
    for y in years:
        targets.update(load_gamelog_targets(y))

    checked = 0
    mism = defaultdict(int)
    examples = defaultdict(list)
    for gid, sides in ap.items():
        t = targets.get(gid)
        if not t:
            continue
        checked += 1
        for side in ("home", "vis"):
            for k in ("H", "HR", "BB", "SO"):
                if sides[side][k] != t[side][k]:
                    mism[k] += 1
                    if len(examples[k]) < 4:
                        examples[k].append((gid, side, sides[side][k], t[side][k]))

    print(f"games cross-checked against game logs: {checked:,}")
    print(f"(team-side comparisons per stat: {checked*2:,})\n")
    for k in ("SO", "BB", "H", "HR"):
        rate = 100.0 * mism[k] / (checked * 2) if checked else 0
        print(f"  {k}: {mism[k]:>6,} mismatched sides  ({rate:.2f}%)")
        for ex in examples[k]:
            print(f"       e.g. {ex[0]} {ex[1]}: parsed={ex[2]} gamelog={ex[3]}")


if __name__ == "__main__":
    main()
