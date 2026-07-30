"""One-command paper-pilot health report — consolidates the telemetry born in
the EV-gate audit (calibration z, odds buckets), the luck-gap eval (fade/ride
arms) and the shade audit (segment exposure) into a single markdown + json.

Run: python phase0/pilot_report.py
Out: data/pilot_report.md + data/pilot_report.json

This is the report the live CLV pilot will emit per league once live grading
starts; today it runs on the historical MLB backtest (2010-2021 vs opening).
All odds usage is evaluation-layer (constitution).
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "phase0")
from ev_gate_audit import BUCKETS, exp_vs_real, load, roi  # noqa: E402
from luck_gap_eval import build_luck  # noqa: E402


def fmt_pct(x):
    return f"{x:+.2%}"


def main():
    rows = load()
    luck = build_luck(set(range(2010, 2022)))
    # attach luck gap to each game row (home minus away) for fade/ride arms
    import csv
    from collections import defaultdict
    odds = defaultdict(list)
    for r in csv.DictReader(open("data/odds_mlb.csv")):
        odds[(r["date"], r["home"], r["away"])].append(r)
    used = defaultdict(int)
    keyed = []
    mp = [r for r in csv.DictReader(open("data/model_probs.csv"))
          if 2010 <= int(r["season"]) <= 2021]
    for r in sorted(mp, key=lambda r: (r["date"], r["home"], r["away"], r["dh"])):
        k = (r["date"], r["home"], r["away"])
        lst = odds.get(k, [])
        if used[k] >= len(lst) or r["full_p"] == "":
            continue
        used[k] += 1
        lk = luck.get(k, (0.0, 0.0))
        keyed.append(lk[0] - lk[1])
    assert len(keyed) == len(rows), "row alignment drift between load() and rejoin"

    md = ["# Paper-pilot health report (MLB backtest, 2010-2021 vs opening)\n"]
    out = {}
    for gate in (0.15, 0.20):
        bets = []
        for g, gap in zip(rows, keyed):
            p = g["p"]
            cands = []
            ev_h = p * (g["ho"] - 1) - (1 - p)
            ev_a = (1 - p) * (g["ao"] - 1) - p
            if ev_h > gate:
                cands.append((ev_h, g["ho"], g["hc"], p, g["y"] == 1, -gap))
            if ev_a > gate:
                cands.append((ev_a, g["ao"], g["ac"], 1 - p, g["y"] == 0, gap))
            if cands:
                ev, o_o, o_c, p_s, win, opp_luck = max(cands)
                bets.append({"p": p_s, "win": win, "odds": o_o, "season": g["season"],
                             "clv": (1 / o_o) < (1 / o_c) - 1e-12,
                             "arm": "fade" if opp_luck > 0.01 else
                                    ("ride" if opp_luck < -0.01 else "flat")})
        n, e, r_, z = exp_vs_real(bets)
        clv = float(np.mean([b["clv"] for b in bets])) if bets else 0.0
        g_out = {"n": n, "roi": round(roi(bets), 4), "clv_pos": round(clv, 3),
                 "z": round(z, 2), "buckets": {}, "arms": {}, "per_season_z": {}}
        md.append(f"\n## Gate >= {gate:.0%} EV\n")
        md.append(f"- **{n} bets** · ROI {fmt_pct(roi(bets))} · CLV+ {clv:.1%} · "
                  f"calibration z **{z:+.2f}** "
                  f"({'HEALTHY' if z > -2 else 'OVERCONFIDENT on selection'})\n")
        md.append("\n| segment | n | ROI | CLV+ | z |\n|---|---|---|---|---|\n")
        for lo, hi, name in BUCKETS:
            sub = [b for b in bets if lo <= b["odds"] < hi]
            if not sub:
                continue
            nn, ee, rr, zz = exp_vs_real(sub)
            cc = float(np.mean([b["clv"] for b in sub]))
            md.append(f"| odds {name} | {nn} | {fmt_pct(roi(sub))} | {cc:.1%} | {zz:+.2f} |\n")
            g_out["buckets"][name] = {"n": nn, "roi": round(roi(sub), 4),
                                      "clv_pos": round(cc, 3), "z": round(zz, 2)}
        for arm in ("fade", "ride", "flat"):
            sub = [b for b in bets if b["arm"] == arm]
            if not sub:
                continue
            nn, ee, rr, zz = exp_vs_real(sub)
            cc = float(np.mean([b["clv"] for b in sub]))
            md.append(f"| {arm}-luck | {nn} | {fmt_pct(roi(sub))} | {cc:.1%} | {zz:+.2f} |\n")
            g_out["arms"][arm] = {"n": nn, "roi": round(roi(sub), 4),
                                  "clv_pos": round(cc, 3), "z": round(zz, 2)}
        seas_line = []
        for s in sorted({b["season"] for b in bets}):
            _, _, _, zz = exp_vs_real([b for b in bets if b["season"] == s])
            g_out["per_season_z"][s] = round(zz, 2)
            seas_line.append(f"{s}:{zz:+.1f}")
        md.append(f"\nPer-season calibration z: {' '.join(seas_line)}\n")
        out[f"gate_{int(gate*100)}"] = g_out

    md.append("\n---\n*Bucket lesson: mid-dogs were the weak segment at 15%; "
              "fade-luck arms outperform ride-luck (explainable-disagreement "
              "mechanism). NFL segment exposure: home-favorite bets carry the "
              "Levitt shade headwind (48.3% cover).*\n")
    open("data/pilot_report.md", "w", encoding="utf-8").write("".join(md))
    json.dump(out, open("data/pilot_report.json", "w"), indent=1)
    print("".join(md))
    print("wrote data/pilot_report.md + .json")


if __name__ == "__main__":
    main()
