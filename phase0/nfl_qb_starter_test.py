"""Leak correction, re-measured on TEST: the shipped NFL blend with the QB feature
keyed on the game's STARTER (first passer) instead of nflverse's post-game
primary passer. See nfl_qb_elo.apply_starters.

This is not a candidate search - it removes post-game information from a shipped
feature, so its TEST number is expected to get WORSE, and whichever way it goes
it is the honest figure for the shipped model. Protocol identical to the serve's
own TEST reproduction (walk-forward 2016-2025, recency hl=3, C=100).

Usage:  python phase0/nfl_qb_starter_test.py listed|starter   (writes per-game
losses), then  python phase0/nfl_qb_starter_test.py compare
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

MODE = sys.argv[1]
OUT = "data/nfl_qb_starter_test.json"
if MODE == "compare":
    a, b = np.load("data/nfl_qb_losses_listed.npy"), np.load("data/nfl_qb_losses_starter.npy")
    sea = np.load("data/nfl_qb_losses_seasons.npy")
    d = b - a                                     # positive = starter version is worse
    rng = np.random.default_rng(20260924)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    per = {int(s): [round(float(a[sea == s].mean()), 5), round(float(b[sea == s].mean()), 5)]
           for s in np.unique(sea)}
    res = {"n": int(len(d)), "listed_ll": round(float(a.mean()), 5), "starter_ll": round(float(b.mean()), 5),
           "cost": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)], "per_season": per}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    raise SystemExit(0)

os.environ["NFL_QB_STARTER"] = "1" if MODE == "starter" else "0"
T0 = time.time()
_src = open("phase0/nfl_season_serve.py", encoding="utf-8").read()
exec(compile(_src.split("# ---------------- protocol check")[0], "serve<prelude>", "exec"))  # noqa: S102
from sklearn.linear_model import LogisticRegression  # noqa: E402

SEAS = np.array([g["season"] for g in games])  # noqa: F821
Y = np.array([g["y"] for g in games], float)  # noqa: F821
vec, sea = [], []
for s_ in range(2016, 2026):
    tr = (SEAS < s_) & (Y != 0.5)
    te = SEAS == s_
    w = 0.5 ** ((s_ - 1 - SEAS[tr]) / 3.0)
    m = LogisticRegression(C=100.0, max_iter=5000).fit(X14[tr], Y[tr], sample_weight=w)  # noqa: F821
    p = np.clip(m.predict_proba(X14[te])[:, 1], 1e-12, 1 - 1e-12)  # noqa: F821
    vec.append(-(Y[te] * np.log(p) + (1 - Y[te]) * np.log(1 - p))); sea.append(np.full(te.sum(), s_))
v = np.concatenate(vec)
np.save(f"data/nfl_qb_losses_{MODE}.npy", v)
np.save("data/nfl_qb_losses_seasons.npy", np.concatenate(sea))
print(f"{MODE}: TEST LL {v.mean():.5f} on {len(v)} games")
