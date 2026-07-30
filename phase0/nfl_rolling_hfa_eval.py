"""Rolling (adaptive) HFA — DEV-ONLY evaluation for the bundle.

The Elo core's home-field advantage becomes an online-learned state:
    after each non-neutral game:  hfa += k_h * (y - p)
initialized at the tuned static value. Leak-free (updates trail predictions).
DEV can only reward within-2001-2015 drift; the 2020 empty-stadium payoff sits in
TEST and comes free if the mechanism is sound. NO TEST look here.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, load_games  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
lg_qb = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]
BP = base["params"]                       # k, hfa, regress


def run_elo_adaptive(kh):
    R = {}
    hfa = BP["hfa"]
    ps = np.zeros(len(games))
    hfas = np.zeros(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - BP["regress"])
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        h = 0.0 if g["neutral"] else hfa
        p = 1.0 / (1.0 + 10 ** (-((rh + h) - ra) / 400.0))
        ps[i] = p
        hfas[i] = hfa
        d = BP["k"] * (g["y"] - p)
        R[g["home"]] += d
        R[g["away"]] -= d
        if not g["neutral"]:
            hfa += kh * (g["y"] - p)
    return ps, hfas


# QB feature (frozen shipped params)
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=SP["delta"],
          decay=SP["decay"], prior_db=SP["prior_db"], season_decay=SP["season_decay"])
qd, prev = np.zeros(len(games)), None
y = np.array([g["y"] for g in games])
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qd[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
    m.predict(g)
    m.update(g, 0.5, qbw)
eps = 1e-12
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))


def lg_(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()


ps0, _ = run_elo_adaptive(0.0)
base_cv = cv_ll(np.column_stack([lg_(ps0), qd]))
print(f"static-HFA blend DEV CV {base_cv:.5f}\n")
print(f"{'k_h':>6}  {'DEV CV':>9}  {'delta':>9}  {'hfa@2015':>9}")
best = (base_cv, 0.0)
for kh in (0.1, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0):
    ps, hfas = run_elo_adaptive(kh)
    s = cv_ll(np.column_stack([lg_(ps), qd]))
    end_dev = hfas[dev][-1]
    tag = "  <-- best" if s < best[0] else ""
    print(f"{kh:>6}  {s:>9.5f}  {s-base_cv:>+9.5f}  {end_dev:>9.1f}{tag}")
    if s < best[0]:
        best = (s, kh)

s, kh = best
print(f"\nDEV verdict: k_h={kh}  ({s-base_cv:+.5f}). "
      f"{'PARKED for bundle.' if kh > 0 else 'adaptive HFA does not help on DEV.'}")
if kh > 0:
    ps, hfas = run_elo_adaptive(kh)
    yrs = {}
    for i, g in enumerate(games):
        yrs.setdefault(g["season"], hfas[i])
    trail = {sn: round(v, 1) for sn, v in sorted(yrs.items()) if sn in
             (2001, 2005, 2010, 2015, 2020, 2021, 2025)}
    print(f"  learned HFA at season starts: {trail}")
json.dump({"dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "delta_cv": round(s - base_cv, 5), "k_h": kh},
          open("data/nfl_rolling_hfa.json", "w"), indent=1)
