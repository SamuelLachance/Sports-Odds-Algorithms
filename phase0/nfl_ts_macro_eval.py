"""Macro (team-vs-team) TrueSkill with football tuning, two questions on DEV:

  Q1 ADD:     does ts_diff add to the frozen blend [elo_logit, qb_diff]?
              (gate: DEV CV improvement >= 0.0020 to earn the ONE TEST look)
  Q2 REPLACE: is TS a better CORE than Elo?  [ts_diff_hfa, qb_diff] vs blend (DEV only)

Game-level Gaussian 1v1 updates (closed-form TrueSkill) with: tunable beta (football
noise), weekly tau, offseason sigma-widen + mu-regress (roster churn), and additive
home offset h inside the update (neutral sites h=0). Ties (15) skipped in updates.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

MU0, SIG0 = 25.0, 25.0 / 3.0
GATE = 0.0020
_SQ2PI = math.sqrt(2 * math.pi)


def _pdf(x): return math.exp(-0.5 * x * x) / _SQ2PI
def _cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
lg_qb = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]

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
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)
BASE = np.column_stack([lgt, qd])


def ts_macro(beta, tau, widen, mu_reg, h):
    mu, var = {}, {}
    f_diff = np.zeros(len(games))        # mu_h - mu_a (blend carries HFA)
    f_hfa = np.zeros(len(games))         # + h when not neutral (replacement variant)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in mu:
                mu[t] = MU0 + (mu[t] - MU0) * (1.0 - mu_reg)
                var[t] = min(var[t] + widen * widen, SIG0 * SIG0)
        prev = g["season"]
        hT, aT = g["home"], g["away"]
        for t in (hT, aT):
            mu.setdefault(t, MU0); var.setdefault(t, SIG0 * SIG0)
        hh = 0.0 if g["neutral"] else h
        f_diff[i] = mu[hT] - mu[aT]
        f_hfa[i] = mu[hT] + hh - mu[aT]
        if g["y"] == 0.5:
            continue
        # winner/loser with home offset inside the update
        if g["y"] == 1.0:
            w, l, off = hT, aT, hh          # home won; its perf includes +h
        else:
            w, l, off = aT, hT, -hh
        vw = var[w] + tau * tau
        vl = var[l] + tau * tau
        c2 = 2 * beta * beta + vw + vl
        c = math.sqrt(c2)
        t_ = (mu[w] + off - mu[l]) / c
        den = _cdf(t_)
        v = _pdf(t_) / den if den > 1e-12 else -t_
        wq = min(max(v * (v + t_), 0.0), 1.0)
        mu[w] += vw / c * v
        mu[l] -= vl / c * v
        var[w] = vw * (1 - vw / c2 * wq)
        var[l] = vl * (1 - vl / c2 * wq)
    return f_diff, f_hfa


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
print(f"frozen blend DEV CV {base_cv:.5f}   (ADD gate: <= {base_cv - GATE:.5f})\n")

rng = np.random.default_rng(20260723)
bestA, bestR, t0 = None, None, time.time()
for i in range(150):
    cand = dict(
        beta=float(np.exp(rng.uniform(np.log(2.0), np.log(25.0)))),
        tau=float(np.exp(rng.uniform(np.log(0.001), np.log(1.0)))),
        widen=float(rng.uniform(0.0, 6.0)),
        mu_reg=float(rng.uniform(0.0, 0.7)),
        h=float(rng.uniform(0.0, 3.0)),
    )
    f_diff, f_hfa = ts_macro(**cand)
    sA = cv_ll(np.column_stack([BASE, f_diff]))               # Q1 add
    sR = cv_ll(np.column_stack([f_hfa, qd]))                  # Q2 replace elo core
    if bestA is None or sA < bestA[0]:
        bestA = (sA, cand)
    if bestR is None or sR < bestR[0]:
        bestR = (sR, cand)
    if (i + 1) % 30 == 0:
        print(f"  ...{i+1}/150  ADD best {bestA[0]:.5f} ({bestA[0]-base_cv:+.5f})"
              f"   REPLACE best {bestR[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\nQ1 ADD:     best CV {bestA[0]:.5f} ({bestA[0]-base_cv:+.5f})  "
      f"{({k: round(v, 3) for k, v in bestA[1].items()})}")
print(f"Q2 REPLACE: best CV {bestR[0]:.5f} vs blend {base_cv:.5f} "
      f"({bestR[0]-base_cv:+.5f})  -> {'TS core better' if bestR[0] < base_cv else 'Elo core stands'}")
json.dump({"add": {"cv": round(bestA[0], 5), "params": bestA[1]},
           "replace": {"cv": round(bestR[0], 5), "params": bestR[1]},
           "base_cv": round(base_cv, 5)},
          open("data/nfl_ts_macro.json", "w"), indent=1)
if bestA[0] > base_cv - GATE:
    print(f"\nADD gate NOT met -> no TEST look; frozen model stands at 0.62806.")
    sys.exit(0)

f_diff, _ = ts_macro(**bestA[1])
Xw = np.column_stack([BASE, f_diff])
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=3000).fit(BASE[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=3000).fit(Xw[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)
rng2 = np.random.default_rng(7)
n = len(d)
bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST (ONE look): {llv(yt, pb).mean():.5f} -> {llv(yt, pw).mean():.5f}  "
      f"delta {d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] -> {sig}")
