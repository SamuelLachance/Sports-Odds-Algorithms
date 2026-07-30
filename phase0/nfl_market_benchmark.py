"""Benchmark the shipped NFL model against the DE-VIGGED CLOSING moneyline.

Market data (games.csv closing moneylines) is used for EVALUATION ONLY — it never
enters the model. Same protocol as the MLB benchmark: no-vig prob = normalize the
two implied probs to sum to 1; paired per-game LL on the TEST games where both a
model prob and a closing ML exist.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
lg = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]

# ---- closing moneylines, joined by chronological order ----
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
def imp(ml):
    if not ml:
        return None
    v = float(ml)
    return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))
mkt = np.full(len(games), np.nan)
for i, r in enumerate(raw):
    ph, pa = imp(r["home_moneyline"]), imp(r["away_moneyline"])
    if ph and pa:
        mkt[i] = ph / (ph + pa)                     # no-vig home prob

# ---- shipped model probs on TEST ----
pe, y = run_elo(games, score_from=1999, **base["params"])
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg, rep_delta=SP["delta"],
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
X = np.column_stack([lgt, qd])
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)
fit = dev & (y != 0.5)
clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
pm = np.full(len(games), np.nan)
pm[test] = clf.predict_proba(X[test])[:, 1]

# ---- paired comparison on TEST games with a closing line ----
mask = test & ~np.isnan(mkt)
print(f"TEST games with closing ML: {int(mask.sum())}/{int(test.sum())}")
by_season = {}
for s in range(TEST_YEARS.start, TEST_YEARS.stop):
    ms = mask & (seasons == s)
    by_season[s] = int(ms.sum())
print("  per season:", by_season)

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

yt = y[mask]
lm, lk = llv(yt, pm[mask]), llv(yt, mkt[mask])
d = lk - lm                                   # >0 => model better than close
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmodel (shipped blend)     LL {lm.mean():.5f}")
print(f"closing ML no-vig         LL {lk.mean():.5f}")
print(f"model - close gap {lm.mean()-lk.mean():+.5f}   paired CI on (close-model) "
      f"[{lo:+.5f}, {hi:+.5f}]")
print("  ->", "model SIG better" if lo > 0 else ("close SIG better" if hi < 0 else "inside noise"))
# agreement diagnostics
corr = np.corrcoef(pm[mask], mkt[mask])[0, 1]
mad = np.abs(pm[mask] - mkt[mask]).mean()
print(f"\ncorr(model, close) {corr:.3f}   mean|diff| {mad:.3f} prob-pts")
