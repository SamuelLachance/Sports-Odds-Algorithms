"""Parse per-player baserunning value from Retrosheet events (no Statcast).

Baserunning is the third leg of offense our model is blind to (we have on-base and
power). We walk each half-inning's base state to credit the right runner for:
  SB / CS      stolen bases and caught stealing
  extra bases  advancing more than the batted ball forces (1st-to-3rd on a single,
               scoring from 1st on a double, ...)
  out_on_base  thrown out trying to advance
  gidp         batter grounds into a double play (slow-runner penalty)
Output: data/retro_events/baserun.csv  game_id,player,pa,sb,cs,xb,oob,gidp
Base-state parsing is approximate on edge cases; the resulting rate is EB-shrunk,
so minor imprecision washes out. Validated by the season SB total.
"""

from __future__ import annotations

import csv
import glob
import re
from collections import defaultdict

OUT = "data/retro_events/baserun.csv"
_BASE = {"1": 1, "2": 2, "3": 3, "H": 4}


def batter_dest(core: str) -> int:
    """Where the batter ends up (0 = out/no advance handled by advances)."""
    e = core
    if re.match(r"^(S)(\d|/|$)", e):
        return 1
    if re.match(r"^D(\d|/|G|$)", e):
        return 2
    if re.match(r"^T(\d|/|$)", e):
        return 3
    if re.match(r"^(HR|H)(\d|/|$)", e):
        return 4
    if re.match(r"^(W|IW|I|HP)(\+|\.|/|;|$)", e):
        return 1
    if e[:1] == "E" or e.startswith(("FC", "C/")):
        return 1
    return 0


def min_adv(dest: int, frm: int) -> int:
    """Bases a runner on `frm` is forced/expected to take on a batter reaching `dest`.
    Single(1): 1 base each; Double(2): 2 bases; Triple(3): 3 bases; walk-force handled
    by caller. Used to flag EXTRA bases beyond this."""
    return {1: 1, 2: 2, 3: 3, 4: 4}.get(dest, 0)


def process(gid, batter, ev, bases, agg):
    e = ev.strip()
    core = e.split(".", 1)[0]
    advs = e.split(".", 1)[1].split(";") if "." in e else []
    dest = batter_dest(core.split("/")[0])

    def cr(pid, k):
        agg[(gid, pid)][k] += 1

    # stolen bases / caught stealing (may be several, and combined with a K/W)
    for tok in core.split(";"):
        t = tok.split("/")[0]
        mS = re.match(r"^SB([23H])", t)
        mC = re.match(r"^(?:PO)?CS([23H])", t)
        if mS:
            b = mS.group(1); frm = _BASE[b] - 1
            r = bases.get(frm)
            if r:
                cr(r, 1)
                bases[frm] = None
                if b != "H":
                    bases[_BASE[b]] = r
        elif mC:
            b = mC.group(1); frm = _BASE[b] - 1
            r = bases.get(frm)
            if r:
                cr(r, 2)
                bases[frm] = None

    # explicit runner advances / outs
    for a in advs:
        a = a.strip()
        m = re.match(r"^([B123])([-X])([123H])", a)
        if not m:
            continue
        frm_ch, kind, to_ch = m.group(1), m.group(2), m.group(3)
        if frm_ch == "B":
            if kind == "-" and to_ch != "H":
                bases[_BASE[to_ch]] = batter
            continue
        frm = int(frm_ch)
        r = bases.get(frm)
        if not r:
            continue
        if kind == "X":                       # thrown out advancing
            agg[(gid, r)][4] += 1
            bases[frm] = None
        else:                                 # advanced
            extra = (_BASE[to_ch] - frm) - min_adv(dest, frm)
            if extra > 0:
                agg[(gid, r)][3] += extra
            bases[frm] = None
            if to_ch != "H":
                bases[_BASE[to_ch]] = r

    # place batter if reached and not already placed via a B- advance
    if dest in (1, 2, 3) and bases.get(dest) is None and batter not in bases.values():
        bases[dest] = batter

    # GIDP: batter grounds into a double play
    if ("GDP" in ev or "DP" in core) and re.search(r"\(\d\)", core):
        agg[(gid, batter)][5] += 1


def parse_file(path, agg):
    game_id = None
    bases = {1: None, 2: None, 3: None}
    cur = None
    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        if f[0] == "id":
            game_id = f[1]
            bases = {1: None, 2: None, 3: None}
            cur = None
        elif f[0] == "play":
            key = (f[1], f[2])
            if key != cur:
                bases = {1: None, 2: None, 3: None}
                cur = key
            batter = f[3]
            ev = f[6] if len(f) > 6 else ""
            if ev.startswith("NP") or ev == "":
                continue
            agg[(game_id, batter)][0] += 1     # (rough) PA counter for non-NP plays
            process(game_id, batter, ev, bases, agg)


def main():
    agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # (game,player) -> pa,sb,cs,xb,oob,gidp
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    for i, p in enumerate(files):
        parse_file(p, agg)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "player", "pa", "sb", "cs", "xb", "oob", "gidp"])
        for (gid, pid), v in agg.items():
            w.writerow([gid, pid] + v)
    tot = [sum(v[k] for v in agg.values()) for k in range(6)]
    print(f"wrote {OUT}: {len(agg):,} players", flush=True)
    print(f"  totals SB={tot[1]:,} CS={tot[2]:,} extraB={tot[3]:,} outOnBase={tot[4]:,} GIDP={tot[5]:,}", flush=True)
    print(f"  ~per season: SB={tot[1]//26:,} (MLB ~2500-3000/yr is the sanity check)", flush=True)


if __name__ == "__main__":
    main()
