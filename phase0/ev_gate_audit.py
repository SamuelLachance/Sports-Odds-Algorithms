"""Favorite-longshot audit of the symmetric %EV gate + expected-vs-realized
hit-rate telemetry (sanctioned queue ranks 1-2, from Thaler-Ziemba / Kaunitz).

Question 1 (Thaler): does the ">= X% EV vs opening" rule over-select longshots?
EV thresholding is symmetric in probability space, but model error is not: a
+15% EV claim on a 5.0 longshot needs only a ~3pp probability overestimate,
while on a 1.5 favorite it needs ~10pp. If longshot edges are noise-inflated,
the longshot bucket shows realized hits << expected hits and worse ROI.

Question 2 (Kaunitz eq-8): the expected-vs-realized calibration z-score
  z = (realized_wins - sum p_i) / sqrt(sum p_i (1-p_i))
detects a broken edge at far smaller n than P&L significance ever can. Built
here as a reusable telemetry function for the live CLV pilot.

Data: opening/closing MLB odds (2010-2021) x shipped model probs. Odds are
evaluation-layer only (constitution).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

BUCKETS = [(1.0, 1.7, "big fav"), (1.7, 2.2, "coin flip"),
           (2.2, 3.0, "dog"), (3.0, 99.0, "longshot")]


def load(prob_col="full_p", years=range(2010, 2022)):
    odds = defaultdict(list)
    for r in csv.DictReader(open("data/odds_mlb.csv")):
        odds[(r["date"], r["home"], r["away"])].append(r)
    used = defaultdict(int)
    rows = []
    mp = [r for r in csv.DictReader(open("data/model_probs.csv"))
          if int(r["season"]) in years]
    for r in sorted(mp, key=lambda r: (r["date"], r["home"], r["away"], r["dh"])):
        k = (r["date"], r["home"], r["away"])
        lst = odds.get(k, [])
        if used[k] >= len(lst) or r[prob_col] == "":
            continue
        o = lst[used[k]]
        used[k] += 1
        rows.append({"date": r["date"], "season": int(r["season"]),
                     "p": float(r[prob_col]),
                     "ho": float(o["home_open"]), "ao": float(o["away_open"]),
                     "hc": float(o["home_close"]), "ac": float(o["away_close"]),
                     "y": int(r["y"])})
    return rows


def bets_at(rows, min_edge):
    """All positive-EV opening bets at the given gate. One bet per game max
    (the higher-EV side), matching the badge/pilot behavior."""
    out = []
    for g in rows:
        p = g["p"]
        cands = []
        ev_h = p * (g["ho"] - 1) - (1 - p)
        ev_a = (1 - p) * (g["ao"] - 1) - p
        if ev_h > min_edge:
            cands.append((ev_h, g["ho"], g["hc"], p, g["y"] == 1))
        if ev_a > min_edge:
            cands.append((ev_a, g["ao"], g["ac"], 1 - p, g["y"] == 0))
        if cands:
            ev, o_open, o_close, p_side, win = max(cands)
            out.append({"season": g["season"], "ev": ev, "odds": o_open,
                        "p": p_side, "win": win,
                        "clv": (1.0 / o_open) < (1.0 / o_close) - 1e-12})
    return out


def exp_vs_real(bets):
    """Kaunitz eq-8 telemetry: is the model's claimed win prob on the SELECTED
    set matching reality? Returns (n, expected, realized, z)."""
    if not bets:
        return 0, 0.0, 0, 0.0
    exp = sum(b["p"] for b in bets)
    var = sum(b["p"] * (1 - b["p"]) for b in bets)
    real = sum(b["win"] for b in bets)
    z = (real - exp) / np.sqrt(var) if var > 0 else 0.0
    return len(bets), exp, real, float(z)


def roi(bets):
    if not bets:
        return 0.0
    return float(np.mean([(b["odds"] - 1) if b["win"] else -1.0 for b in bets]))


def main():
    rows = load()
    print(f"{len(rows):,} games with opening odds + model prob (2010-2021)")
    res = {}
    for gate in (0.15, 0.20):
        bets = bets_at(rows, gate)
        n, exp, real, z = exp_vs_real(bets)
        print(f"\n=== gate >= {gate:.0%} EV vs opening: {n:,} bets ===")
        print(f"  overall: expected {exp:.0f} wins, realized {real} "
              f"(z {z:+.2f}), ROI {roi(bets):+.2%}, CLV+ {np.mean([b['clv'] for b in bets]):.1%}")
        gr = {"n": n, "expected": round(exp, 1), "realized": real, "z": round(z, 2),
              "roi": round(roi(bets), 4), "buckets": {}}
        print(f"  {'bucket':<10} {'n':>5} {'share':>6} {'E[hit]':>7} {'real':>5} "
              f"{'z':>6} {'ROI':>8} {'CLV+':>6}")
        for lo, hi, name in BUCKETS:
            sub = [b for b in bets if lo <= b["odds"] < hi]
            nn, ee, rr, zz = exp_vs_real(sub)
            if nn == 0:
                continue
            print(f"  {name:<10} {nn:>5} {nn/max(n,1):>6.1%} {ee/max(nn,1):>7.1%} "
                  f"{rr/max(nn,1):>5.1%} {zz:>+6.2f} {roi(sub):>+8.2%} "
                  f"{np.mean([b['clv'] for b in sub]):>6.1%}")
            gr["buckets"][name] = {"n": nn, "share": round(nn / max(n, 1), 3),
                                   "exp_rate": round(ee / max(nn, 1), 4),
                                   "real_rate": round(rr / max(nn, 1), 4),
                                   "z": round(zz, 2), "roi": round(roi(sub), 4),
                                   "clv_pos": round(float(np.mean([b["clv"] for b in sub])), 3)}
        # per-season telemetry (the pilot monitor in action)
        print("  per-season z (broken-edge detector):", end=" ")
        seas_z = {}
        for s in sorted({b["season"] for b in bets}):
            _, _, _, zz = exp_vs_real([b for b in bets if b["season"] == s])
            seas_z[s] = round(zz, 2)
        print(" ".join(f"{s}:{z:+.1f}" for s, z in seas_z.items()))
        gr["per_season_z"] = seas_z
        res[f"gate_{int(gate*100)}"] = gr
    json.dump(res, open("data/ev_gate_audit.json", "w"), indent=1)
    print("\nwrote data/ev_gate_audit.json")


if __name__ == "__main__":
    main()
