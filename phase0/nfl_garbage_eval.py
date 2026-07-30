"""Garbage-time filtering for the team EPA rating (idea from the nflfastR EP/WP paper).

Rebuild f_epa with plays filtered by non-vegas WP (market-free, game-state only):
keep plays with t <= wp <= 1-t for t in {0.05, 0.10, 0.20}; league rate recomputed
per filter. DEV CV selects vs current (unfiltered); ONE TEST look; keep if better.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # tuned F dict + cv_ll/llv  # noqa: S102

# filtered per-game-team aggregates
THRESH = (0.05, 0.10, 0.20)
aggf = {t: defaultdict(lambda: defaultdict(lambda: [0.0, 0])) for t in THRESH}
with open("data/nfl_plays_wp.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    next(rd)
    for r in rd:
        gid, pos, dfn, epa, wp = r[0], r[1], r[2], float(r[3]), r[4]
        try:
            w = float(wp)
        except ValueError:
            continue
        t_ = PBP_FIX.get(pos, pos)
        for th in THRESH:
            if th <= w <= 1 - th:
                a = aggf[th][gid][t_]
                a[0] += epa; a[1] += 1

def lg_of(agg_):
    num = den = 0.0
    for gid, tm in agg_.items():
        if int(gid[:4]) in DEV_YEARS:
            for t_, (s_, n_) in tm.items():
                num += s_; den += n_
    return num / den

def b_epa_agg(agg_, lg_):
    off = defaultdict(lambda: [0.0, 0.0]); dfa2 = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games)); prev = None
    dec, pn, sd = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
    def rate(st_):
        return (st_[0] + pn * lg_) / (st_[1] + pn)
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in list(off.values()) + list(dfa2.values()):
                st_[0] *= (1 - sd); st_[1] *= (1 - sd)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = (rate(off[h]) - rate(dfa2[h])) - (rate(off[a]) - rate(dfa2[a]))
        tm = agg_.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                s_, n_ = tm.get(t_off, (0.0, 0))
                o = off[t_off]; o[0] = dec * o[0] + s_; o[1] = dec * o[1] + n_
                dd = dfa2[opp]; dd[0] = dec * dd[0] + s_; dd[1] = dec * dd[1] + n_
    return f

X_cur = X_of(F)
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
best = None
for th in THRESH:
    lg_ = lg_of(aggf[th])
    fe = b_epa_agg(aggf[th], lg_)
    F2 = dict(F); F2["epa"] = fe
    s = cv_ll(X_of(F2))
    tag = ""
    if best is None or s < best[0]:
        best = (s, th, fe); tag = "  <-- best"
    print(f"  wp filter [{th},{1-th:.2f}]  (lg {lg_:+.4f})  DEV CV {s:.5f} "
          f"({s-base_cv:+.5f}){tag}")

s, th, fe = best
if s >= base_cv:
    print("\nno DEV improvement -> garbage-time filter rejected on DEV, no TEST look")
    json.dump({"verdict": "dev_rejected", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "thresh": th},
              open("data/nfl_garbage.json", "w"), indent=1)
    raise SystemExit(0)

F2 = dict(F); F2["epa"] = fe
Xn = X_of(F2)
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"\nmain TEST (filter {th}): current {llv(yt, p0).mean():.5f} -> "
      f"{llv(yt, p1).mean():.5f}  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])"
      f"  -> {'ADOPT' if better else 'keep current'}")
json.dump({"thresh": th, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_garbage.json", "w"), indent=1)
