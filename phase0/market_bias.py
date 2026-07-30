"""Where is the OPENING line soft? Test specific market biases instead of blanket
decorrelation, and see if our model's edge concentrates into a vig-clearing subset.

A. Direct bias (model-free): calibration of the opening line by favourite strength
   (favourite-longshot bias) and home team actual-vs-implied (home bias).
B. Our value bets segmented by favourite/underdog x home/away -> flat yield + CLV.
C. Targeted strategies (fade favourites / fade home / big disagreement) with CIs.

recbp (at-open) model prob, 2016-2021, opening + closing moneylines. Odds = eval only.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

ODDS = "data/odds_mlb.csv"
PROBS = "data/model_probs.csv"


def load(years=range(2016, 2022)):
    odds = defaultdict(list)
    for r in csv.DictReader(open(ODDS)):
        odds[(r["date"], r["home"], r["away"])].append(r)
    used = defaultdict(int)
    rows = []
    mp = [r for r in csv.DictReader(open(PROBS)) if int(r["season"]) in years]
    for r in sorted(mp, key=lambda r: (r["date"], r["home"], r["away"], r["dh"])):
        k = (r["date"], r["home"], r["away"])
        lst = odds.get(k, [])
        if used[k] >= len(lst) or r["recbp_p"] == "":
            continue
        o = lst[used[k]]; used[k] += 1
        ho, ao = float(o["home_open"]), float(o["away_open"])
        hc, ac = float(o["home_close"]), float(o["away_close"])
        mh = (1 / ho) / (1 / ho + 1 / ao)          # no-vig opening home prob
        rows.append(dict(p=float(r["recbp_p"]), ho=ho, ao=ao, hc=hc, ac=ac,
                         mh=mh, y=int(r["y"])))
    return rows


def ci(x, n=10000, seed=7):
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def part_a(rows):
    print("=== A. DIRECT market bias in the OPENING line (model-free) ===")
    print("A1. Favourite-longshot: bucket by favourite no-vig implied prob; actual fav win% vs implied")
    print(f"{'fav implied':>14} {'n':>6} {'implied':>8} {'actual':>8} {'act-impl':>9}")
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.72), (0.72, 1.01)]
    for lo, hi in buckets:
        seg = []
        for r in rows:
            fav_p = max(r["mh"], 1 - r["mh"])       # favourite's implied prob
            if lo <= fav_p < hi:
                fav_won = (r["y"] == 1) if r["mh"] >= 0.5 else (r["y"] == 0)
                seg.append((fav_p, 1.0 if fav_won else 0.0))
        if not seg:
            continue
        impl = np.mean([s[0] for s in seg]); act = np.mean([s[1] for s in seg])
        print(f"{lo:.2f}-{hi:.2f}     {len(seg):>6} {impl:>8.3f} {act:>8.3f} {act-impl:>+9.3f}"
              + ("  <-favs OVERvalued" if act - impl < -0.008 else "  <-favs undervalued" if act - impl > 0.008 else ""))
    hi_impl = np.mean([r["mh"] for r in rows]); hi_act = np.mean([r["y"] for r in rows])
    print(f"A2. Home bias: home implied {hi_impl:.3f} vs actual {hi_act:.3f} -> "
          f"{'home OVERvalued' if hi_act < hi_impl - 0.005 else 'home UNDERvalued' if hi_act > hi_impl + 0.005 else 'fair'} "
          f"(act-impl {hi_act-hi_impl:+.3f})")


def value_bets(rows, phi=0.0, min_edge=0.0):
    bets = []
    for r in rows:
        p = r["p"]
        if abs(p - 0.5) < phi:
            continue
        opts = []
        if p * (r["ho"] - 1) - (1 - p) > min_edge:
            opts.append((p * (r["ho"] - 1) - (1 - p), 1, r["ho"], r["hc"], r["y"] == 1))
        if (1 - p) * (r["ao"] - 1) - p > min_edge:
            opts.append(((1 - p) * (r["ao"] - 1) - p, 0, r["ao"], r["ac"], r["y"] == 0))
        if not opts:
            continue
        ev, side, oo, oc, win = max(opts)
        bet_impl = (1 / oo) / (1 / r["ho"] + 1 / r["ao"])   # our side's no-vig implied
        bets.append(dict(side=side, oo=oo, oc=oc, win=win, is_fav=bet_impl >= 0.5,
                         is_home=side == 1, profit=(oo - 1) if win else -1.0,
                         clv=1 / oc - 1 / oo))
    return bets


def seg_report(bets, name):
    y, lo, hi = ci([b["profit"] for b in bets])
    clv = np.mean([b["clv"] for b in bets]) if bets else 0
    beat = np.mean([1.0 if b["oo"] > b["oc"] else 0.0 for b in bets]) if bets else 0
    sig = "PROFIT" if lo > 0 else ("LOSS" if hi < 0 else "n.s.")
    print(f"  {name:<26} n={len(bets):>5}  yield {y*100:>+6.2f}% [{lo*100:>+6.2f},{hi*100:>+6.2f}] "
          f" CLV {clv*100:>+5.2f}pp beat {beat*100:4.1f}%  {sig}")


def part_bc(rows):
    print("\n=== B. Our value bets by segment (phi=0.10) ===")
    bets = value_bets(rows, phi=0.10)
    seg_report(bets, "ALL value bets")
    seg_report([b for b in bets if b["is_fav"]], "on FAVOURITES")
    seg_report([b for b in bets if not b["is_fav"]], "on UNDERDOGS")
    seg_report([b for b in bets if b["is_home"]], "on HOME")
    seg_report([b for b in bets if not b["is_home"]], "on AWAY")
    seg_report([b for b in bets if not b["is_fav"] and not b["is_home"]], "AWAY UNDERDOGS")
    seg_report([b for b in bets if not b["is_fav"] and b["is_home"]], "HOME UNDERDOGS")
    print("\n=== C. Targeted strategies ===")
    for phi in (0.0, 0.10, 0.15):
        b = value_bets(rows, phi=phi)
        seg_report([x for x in b if not x["is_fav"]], f"fade-fav (dogs) phi={phi}")


def main():
    rows = load()
    print(f"joined games 2016-2021: {len(rows)}\n")
    part_a(rows)
    part_bc(rows)


if __name__ == "__main__":
    main()
