"""Draft-pedigree QB prior: a rookie's replacement prior depends on where he was drafted.

rep_i = lg + delta * (1 - relief_i),  relief_i = c * exp(-(pick_i - 1) / scale)
  pick 1 with c=.6, scale=30 -> 60% relief (prior well above street level)
  late rounds -> relief ~0; undrafted/unknown -> relief 0 (full replacement)
Draft pick is known at draft time -> strictly as-of, leak-free.

Base = shipped rep-prior model (delta=-0.2482, V2 dynamics, TEST 0.62806). Search
(delta, c, scale) on DEV 5-fold CV, dynamics frozen; CV winner gets ONE TEST look.
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
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
lg = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]                     # delta, decay, prior_db, season_decay
DYN = dict(decay=SP["decay"], prior_db=SP["prior_db"], season_decay=SP["season_decay"])

picks = {}
for r in csv.DictReader(open("data/nfl_draft_picks.csv", encoding="utf-8")):
    if r["position"] == "QB" and r["gsis_id"] and r["pick"]:
        picks[r["gsis_id"]] = int(r["pick"])
print(f"QB draft picks mapped: {len(picks)}")

pe, y = run_elo(games, score_from=1999, **base["params"])
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)


def qb_feature(delta, c, scale):
    rep_map = {qid: lg + delta * (1.0 - c * math.exp(-(pk - 1) / scale))
               for qid, pk in picks.items()}
    m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg,
              rep_delta=delta, rep_map=rep_map, **DYN)
    out, prev = np.zeros(len(games)), None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        out[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
        m.predict(g)
        m.update(g, 0.5, qbw)
    return out


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X):
    idx = np.where(dev)[0]
    scores = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        scores[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return scores.mean()


qd0 = qb_feature(SP["delta"], 0.0, 30.0)          # c=0 -> no pedigree = shipped model
base_cv = cv_ll(np.column_stack([lgt, qd0]))
print(f"shipped rep-prior blend DEV CV {base_cv:.5f}")

rng = np.random.default_rng(20260723)
best, t0 = None, time.time()
for i in range(200):
    cand = dict(
        delta=float(rng.uniform(-0.35, -0.05)),
        c=float(rng.uniform(0.0, 1.0)),
        scale=float(np.exp(rng.uniform(np.log(3.0), np.log(150.0)))),
    )
    s = cv_ll(np.column_stack([lgt, qb_feature(**cand)]))
    if best is None or s < best[0]:
        best = (s, cand)
    if (i + 1) % 50 == 0:
        print(f"  ...{i+1}/200  best CV {best[0]:.5f} ({best[0]-base_cv:+.5f})"
              f"  ({time.time()-t0:.0f}s)", flush=True)

s, cand = best
print(f"\nbest pedigree params {({k: round(v, 4) for k, v in cand.items()})}")
print(f"DEV CV: shipped {base_cv:.5f} -> pedigree {s:.5f} ({s-base_cv:+.5f})")
if s >= base_cv:
    print("no DEV improvement -> rejected on DEV, TEST not touched")
    sys.exit(0)

qd_w = qb_feature(**cand)
Xb = np.column_stack([lgt, qd0])
Xw = np.column_stack([lgt, qd_w])
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=3000).fit(Xb[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=3000).fit(Xw[fit], y[fit])
pb = cb.predict_proba(Xb[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)
rng2 = np.random.default_rng(7)
n = len(d)
bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST 2016-2025 (n={n}, ONE look):")
print(f"  shipped rep-prior blend   LL {llv(yt, pb).mean():.5f}")
print(f"  + draft pedigree          LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
json.dump({"params": cand, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_base": round(float(llv(yt, pb).mean()), 5),
           "test_new": round(float(llv(yt, pw).mean()), 5),
           "delta_ll": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "verdict": sig},
          open("data/nfl_qb_pedigree.json", "w"), indent=1)
