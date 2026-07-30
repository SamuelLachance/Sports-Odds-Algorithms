"""Coaching features (DEV-ONLY — parked for the pre-registered bundle TEST look).

Two as-of features from games.csv coach columns (1999+, zero new data):
  REGIME  exp(-games_under_current_HC / kappa), home minus away — a new-coach signal
          that Elo's uniform season regression can't see (incl. mid-season firings).
  RATING  per-coach EWMA of (result - p_elo) residuals, EB-shrunk, travels with the
          coach's NAME across teams (Reid effect). diff home-away.
Tuned by 5-fold CV on DEV over the frozen blend [elo_logit, qb_diff]. NO TEST LOOK HERE.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
lg_qb = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g["hc"], g["ac"] = r["home_coach"].strip(), r["away_coach"].strip()

# frozen base signals
pe, y = run_elo(games, score_from=1999, **base["params"])
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=SP["delta"],
          decay=SP["decay"], prior_db=SP["prior_db"], season_decay=SP["season_decay"])
qd, prev = np.zeros(len(games)), None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qd[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
    m.predict(g)
    m.update(g, 0.5, qbw)
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
BASE = np.column_stack([lgt, qd])


def coach_features(kappa, decay, prior_n):
    tenure = {}                     # team -> [coach, games_under]
    crat = {}                       # coach -> [resid_sum, n]
    f_reg = np.zeros(len(games))
    f_rat = np.zeros(len(games))
    for i, g in enumerate(games):
        vals = []
        for team, coach in ((g["home"], g["hc"]), (g["away"], g["ac"])):
            t = tenure.get(team)
            if t is None or t[0] != coach:
                tenure[team] = t = [coach, 0]
            vals.append(math.exp(-t[1] / kappa))
        f_reg[i] = vals[0] - vals[1]
        rh = crat.get(g["hc"], (0.0, 0.0))
        ra = crat.get(g["ac"], (0.0, 0.0))
        f_rat[i] = rh[0] / (rh[1] + prior_n) - ra[0] / (ra[1] + prior_n)
        # update AFTER prediction
        resid = g["y"] - pe[i]                     # home-perspective residual vs Elo
        for coach, sgn in ((g["hc"], 1.0), (g["ac"], -1.0)):
            s, n = crat.get(coach, (0.0, 0.0))
            crat[coach] = (decay * s + sgn * resid, decay * n + 1.0)
        tenure[g["home"]][1] += 1
        tenure[g["away"]][1] += 1
    return f_reg, f_rat


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


base_cv = cv_ll(BASE)
print(f"frozen blend DEV CV {base_cv:.5f}\n")

rng = np.random.default_rng(20260723)
best = {"regime": None, "rating": None, "both": None}
t0 = time.time()
for i in range(150):
    cand = dict(
        kappa=float(np.exp(rng.uniform(np.log(2.0), np.log(60.0)))),
        decay=float(rng.uniform(0.90, 1.0)),
        prior_n=float(np.exp(rng.uniform(np.log(5.0), np.log(300.0)))),
    )
    f_reg, f_rat = coach_features(**cand)
    for kind, cols in (("regime", [f_reg]), ("rating", [f_rat]), ("both", [f_reg, f_rat])):
        s = cv_ll(np.column_stack([BASE] + cols))
        if best[kind] is None or s < best[kind][0]:
            best[kind] = (s, cand)
    if (i + 1) % 30 == 0:
        b = min(v[0] for v in best.values())
        print(f"  ...{i+1}/150  best CV {b:.5f} ({b-base_cv:+.5f})  ({time.time()-t0:.0f}s)",
              flush=True)

print()
for kind, (s, cand) in best.items():
    print(f"  {kind:<7} CV {s:.5f} ({s-base_cv:+.5f})  "
          f"{({k: round(v, 3) for k, v in cand.items()})}")
winner = min(best, key=lambda k: best[k][0])
s, cand = best[winner]
print(f"\nDEV verdict: {winner} ({s-base_cv:+.5f}). PARKED for the bundle — no TEST look.")
json.dump({"winner": winner, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "delta_cv": round(s - base_cv, 5), "params": cand,
           "all": {k: {"cv": round(v[0], 5), "params": v[1]} for k, v in best.items()}},
          open("data/nfl_coach.json", "w"), indent=1)
