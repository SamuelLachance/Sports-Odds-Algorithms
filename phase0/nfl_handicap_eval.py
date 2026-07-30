"""Handicapper's composite (the pasted NBA-tool structure, ported to NFL):
  f1 season record diff       (win% last 17 games)
  f2 home/away split diff     (home team's HOME win% vs away team's AWAY win%)
  f3 same, last 10 respective venue games
  f4 last-10 form diff
  f5 avg points diff          (PF/game, last 17)
  f6 avg points last-10 diff
  composite = mean of DEV-z-scored six (the tool's 'Total')
Variants: composite / six individual / both. DEV tryout -> ONE TEST look if it clears.
Prior: every ingredient is a tested-channel re-summary — expect tryout rejection.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict, deque

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
hs = np.array([int(r["home_score"]) for r in raw])
as_ = np.array([int(r["away_score"]) for r in raw])

res = defaultdict(lambda: deque(maxlen=17))       # (win, pf)
resH = defaultdict(lambda: deque(maxlen=10))      # home games only
resA = defaultdict(lambda: deque(maxlen=10))      # away games only
resHs = defaultdict(lambda: deque(maxlen=17))     # season-ish venue records
resAs = defaultdict(lambda: deque(maxlen=17))

def wp(dq, k=None):
    it = list(dq)[-k:] if k else list(dq)
    return sum(w for w, _ in it) / len(it) if it else 0.5

def pf(dq, k=None):
    it = list(dq)[-k:] if k else list(dq)
    return sum(p_ for _, p_ in it) / len(it) if it else 21.0

F6 = np.zeros((len(games), 6))
for i, g in enumerate(games):
    h, a = g["home"], g["away"]
    F6[i, 0] = wp(res[h]) - wp(res[a])
    F6[i, 1] = wp(resHs[h]) - wp(resAs[a])
    F6[i, 2] = wp(resH[h]) - wp(resA[a])
    F6[i, 3] = wp(res[h], 10) - wp(res[a], 10)
    F6[i, 4] = pf(res[h]) - pf(res[a])
    F6[i, 5] = pf(res[h], 10) - pf(res[a], 10)
    hw = 1.0 if g["y"] == 1 else (0.5 if g["y"] == 0.5 else 0.0)
    res[h].append((hw, hs[i])); res[a].append((1 - hw, as_[i]))
    resH[h].append((hw, hs[i])); resHs[h].append((hw, hs[i]))
    resA[a].append((1 - hw, as_[i])); resAs[a].append((1 - hw, as_[i]))

mu6 = F6[dev].mean(0); sd6 = F6[dev].std(0)
Z6 = (F6 - mu6) / sd6
comp = Z6.mean(1)

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

X_cur = X_of(F)
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
CAND = {"composite": [comp.reshape(-1, 1)],
        "six individual": [Z6],
        "composite+six": [comp.reshape(-1, 1), Z6]}
best = None
for name, cols in CAND.items():
    X = np.column_stack([X_cur] + cols)
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, name, X); tag = "  <-- best"
    print(f"  + {name:<16} DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, name, Xn = best
if s >= base_cv:
    print("\nno DEV improvement -> handicapper composite rejected at tryouts, no TEST look")
    json.dump({"verdict": "dev_rejected", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "variant": name},
              open("data/nfl_handicap.json", "w"), indent=1)
    raise SystemExit(0)

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
print(f"\nmain TEST ({name}): current {llv(yt, p0).mean():.5f} -> {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"variant": name, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_handicap.json", "w"), indent=1)
