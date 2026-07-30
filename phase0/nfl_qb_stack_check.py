"""Architecture check: integrated QB-Elo engine vs 'two features trained together'.

Variant A (shipped): QB adj inside the Elo engine (affects updates). TEST 0.63296.
Variant B (stacked): plain-Elo prob + as-of QB-rate diff as TWO FEATURES in a
logistic regression fit on DEV — the 'use them together as features' architecture.
Same QB-state dynamics (decay/prior/season_decay from the tuned engine), same split.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, ll, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
qbrep = json.load(open("data/nfl_qb_elo.json"))
P = qbrep["qb_elo"]["params"]
lg = qbrep["qb_elo"]["lg_rate"]

# ---- plain-Elo prob per game (full walk, frozen baseline params) ----
pe_all, y_all = run_elo(games, score_from=1999, **base["params"])
assert len(pe_all) == len(games)

# ---- as-of QB rate diff per game (same dynamics as tuned engine, beta irrelevant) ----
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, decay=P["decay"],
          prior_db=P["prior_db"], season_decay=P["season_decay"], lg_rate=lg)
qb_diff, prev = [], None
for g in games:
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qb_diff.append(m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"]))
    m.predict(g)                   # seeds team dict; k=0 so ratings never move
    m.update(g, 0.5, qbw)          # QB EWMAs advance
qb_diff = np.array(qb_diff)

seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)
eps = 1e-12
lgt = np.log(np.clip(pe_all, eps, 1 - eps) / np.clip(1 - pe_all, eps, 1 - eps))
X = np.column_stack([lgt, qb_diff])
y = y_all

fit_mask = dev & (y != 0.5)            # sklearn needs discrete labels; ties only excluded from FIT
clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit_mask], y[fit_mask])
pb = clf.predict_proba(X[test])[:, 1]
yt = y[test]

def v(p): p = np.clip(p, eps, 1 - eps); return -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
ll_stack = float(v(pb).mean())
print(f"A integrated engine (shipped)   TEST LL {qbrep['qb_elo']['test_ll']:.5f}")
print(f"B stacked two-feature logistic  TEST LL {ll_stack:.5f}")
print(f"  stacked coefs: elo_logit {clf.coef_[0][0]:+.4f}, qb_diff {clf.coef_[0][1]:+.5f}")

# paired bootstrap A vs B
pq = None
from nfl_qb_elo import run_qb  # noqa: E402
pq, yq = run_qb(games, qbw, score_from=TEST_YEARS.start, score_to=TEST_YEARS.stop - 1,
                lg_rate=lg, **P)
assert np.array_equal(yq, yt)
d = v(pb) - v(pq)                     # >0 => integrated better
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  integrated - stacked delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
      f"  -> {'integrated SIG better' if lo > 0 else ('stacked SIG better' if hi < 0 else 'inside noise')}")
