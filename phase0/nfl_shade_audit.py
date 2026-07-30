"""Levitt segment shade audit on the NFL closing spread (1999-2025) +
spread-implied vs moneyline cross-market consistency (queue ranks 4 & 6).

Levitt 2004: books shade lines toward public biases (favorites, home teams) but
cap the shade below the 52.38% ATS breakeven. Measured here per segment
(home/visiting x favorite/underdog), by era, and early-season (weeks 1-4) vs
later. Second check: the spread-implied win prob (margin ~ N(spread, 13.45))
vs the de-vigged closing ML — where the two markets disagree, which one is
right? Odds/spreads are evaluation-layer only.
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np

BREAKEVEN = 0.5238
SIGMA = 13.45          # NFL margin sd around the spread (standard)


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    rows = []
    for r in csv.DictReader(open("data/nfl_games.csv")):
        if r["home_score"] == "" or r["spread_line"] in ("", None):
            continue
        try:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            sp = float(r["spread_line"])          # + = home favored
            wk = int(r["week"]); season = int(r["season"])
        except ValueError:
            continue
        margin = hs - as_
        if margin == sp:
            continue                              # push
        rows.append({"season": season, "week": wk, "sp": sp,
                     "home_cover": margin > sp, "home_win": margin > 0,
                     "margin": margin,
                     "ml_h": r.get("home_moneyline"), "ml_a": r.get("away_moneyline")})
    print(f"{len(rows):,} spread-graded games (pushes excluded), "
          f"{rows[0]['season']}-{rows[-1]['season']}")

    def seg(r):
        if abs(r["sp"]) < 0.5:
            return None                            # pick'em
        fav_home = r["sp"] > 0
        return ("home fav" if fav_home else "home dog",
                "vis dog" if fav_home else "vis fav")

    def report(rows, label):
        segs = {"home fav": [], "home dog": [], "vis fav": [], "vis dog": []}
        for r in rows:
            s = seg(r)
            if s is None:
                continue
            hf_or_hd, _ = s
            segs[hf_or_hd].append(r["home_cover"])
            other = "vis dog" if hf_or_hd == "home fav" else "vis fav"
            segs[other].append(not r["home_cover"])
        out = {}
        print(f"  {label}")
        for name in ("home fav", "vis fav", "home dog", "vis dog"):
            v = segs[name]
            p, lo, hi = wilson(sum(v), len(v))
            beats = "BEATS BREAKEVEN" if lo > BREAKEVEN else (
                "SIG<50%" if hi < 0.5 else "")
            print(f"    {name:<9} cover {p:.3f} [{lo:.3f},{hi:.3f}] n={len(v):<5} {beats}")
            out[name] = {"cover": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                         "n": len(v)}
        return out

    res = {}
    res["all"] = report(rows, "ALL 1999-2025")
    res["modern"] = report([r for r in rows if r["season"] >= 2015], "2015-2025")
    res["early"] = report([r for r in rows if r["week"] <= 4 and r["season"] >= 2015],
                          "2015-2025 weeks 1-4")

    # ---- spread-implied vs ML cross-market consistency ----
    def imp(ml):
        if ml in (None, ""):
            return None
        v = float(ml)
        return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))

    xs = []
    for r in rows:
        ph, pa = imp(r["ml_h"]), imp(r["ml_a"])
        if not ph or not pa:
            continue
        p_ml = ph / (ph + pa)
        p_sp = 1.0 - 0.5 * math.erfc(r["sp"] / (SIGMA * math.sqrt(2)))
        xs.append({"p_ml": p_ml, "p_sp": p_sp, "y": 1 if r["home_win"] else 0,
                   "d": p_ml - p_sp})
    d = np.array([x["d"] for x in xs])
    print(f"\n  cross-market: n={len(xs):,}  mean|ml-spread| {np.abs(d).mean():.4f}  "
          f"p95 {np.percentile(np.abs(d),95):.4f}")
    # on big-disagreement games, who's right? (LL of each price)
    big = [x for x in xs if abs(x["d"]) > 0.04]
    y = np.array([x["y"] for x in big])
    eps = 1e-9
    ll_ml = float(np.mean(-(y * np.log(np.clip([x["p_ml"] for x in big], eps, 1-eps))
                            + (1-y) * np.log(np.clip([1-x["p_ml"] for x in big], eps, 1-eps)))))
    ll_sp = float(np.mean(-(y * np.log(np.clip([x["p_sp"] for x in big], eps, 1-eps))
                            + (1-y) * np.log(np.clip([1-x["p_sp"] for x in big], eps, 1-eps)))))
    print(f"  disagreement>4pp games (n={len(big)}): LL moneyline {ll_ml:.5f} vs "
          f"spread-implied {ll_sp:.5f} -> {'ML sharper' if ll_ml < ll_sp else 'SPREAD sharper'}")
    res["cross_market"] = {"n": len(xs), "mean_abs_diff": round(float(np.abs(d).mean()), 4),
                           "n_disagree_4pp": len(big),
                           "ll_ml": round(ll_ml, 5), "ll_spread": round(ll_sp, 5)}
    json.dump(res, open("data/nfl_shade_audit.json", "w"), indent=1)
    print("\nwrote data/nfl_shade_audit.json")


if __name__ == "__main__":
    main()
