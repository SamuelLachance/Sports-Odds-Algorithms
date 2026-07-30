"""NFL feature test: rest/bye — MLB increment protocol on the stacked blend.

Base blend (the model): y ~ elo_logit + qb_diff, fit on DEV (2001-2015, ties dropped
from fit only). Candidate rest features, selected by 5-fold CV on DEV ONLY:
  (a) rest_diff  = clip(home_rest - away_rest, +-7)
  (b) bye_diff   = 1[home_rest>=13] - 1[away_rest>=13]
  (c) short_diff = 1[home_rest<=5]  - 1[away_rest<=5]
  (d) all three
Winner gets ONE TEST (2016-2025) look, paired bootstrap vs the base blend.
"""
from __future__ import annotations

import csv
import json
import sys

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
P, lg = rep["qb_elo"]["params"], rep["qb_elo"]["lg_rate"]

# ---- rest columns joined by chronological order (same sort as load_games) ----
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
assert len(raw) == len(games)
hr = np.array([int(r["home_rest"]) for r in raw], float)
ar = np.array([int(r["away_rest"]) for r in raw], float)

# ---- frozen base signals: plain-Elo prob + as-of QB diff ----
pe, y_all = run_elo(games, score_from=1999, **base["params"])
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, decay=P["decay"],
          prior_db=P["prior_db"], season_decay=P["season_decay"], lg_rate=lg)
qb_diff, prev = [], None
for g in games:
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qb_diff.append(m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"]))
    m.predict(g)
    m.update(g, 0.5, qbw)
qb_diff = np.array(qb_diff)

eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
rest_diff = np.clip(hr - ar, -7, 7)
bye_diff = (hr >= 13).astype(float) - (ar >= 13).astype(float)
short_diff = (hr <= 5).astype(float) - (ar <= 5).astype(float)

seasons = np.array([g["season"] for g in games])
y = y_all
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)

BASE = np.column_stack([lgt, qb_diff])
CAND = {
    "a rest_diff": [rest_diff],
    "b bye_diff": [bye_diff],
    "c short_diff": [short_diff],
    "d all three": [rest_diff, bye_diff, short_diff],
}

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def cv_ll(X):
    """5-fold CV log loss on DEV (ties excluded from fits, included in scoring)."""
    idx = np.where(dev)[0]
    scores = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        scores[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return scores.mean()

print(f"DEV 5-fold CV (n={dev.sum()}):")
base_cv = cv_ll(BASE)
print(f"  base blend (elo+qb)   {base_cv:.5f}")
results = {}
for name, cols in CAND.items():
    X = np.column_stack([BASE] + cols)
    results[name] = cv_ll(X)
    print(f"  + {name:<14} {results[name]:.5f}  ({results[name]-base_cv:+.5f})")

winner = min(results, key=results.get)
print(f"\nDEV winner: {winner} (CV {results[winner]:.5f} vs base {base_cv:.5f})")
if results[winner] >= base_cv:
    print("no DEV improvement -> feature rejected on DEV, TEST not touched")
    sys.exit(0)

# ---- ONE TEST look: winner vs base, both fit on full DEV ----
Xw = np.column_stack([BASE] + CAND[winner])
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=3000).fit(BASE[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=3000).fit(Xw[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)          # >0 => rest feature helps
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST 2016-2025 (n={n}, ONE look):")
print(f"  base blend        LL {llv(yt, pb).mean():.5f}")
print(f"  + {winner:<14} LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
print(f"  rest coefs: {[round(c, 5) for c in cw.coef_[0][2:]]}")
