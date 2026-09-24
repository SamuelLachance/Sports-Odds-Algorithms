"""Breakthrough program (NFL, ideas leg) - DEV-only cache of the shipped blend.

Reproduces the shipped main-split walk-forward (error_map.nfl_league, verbatim
protocol: expanding refit, recency HL=3, C=100, ties dropped from fits) on DEV
2006-2015 and caches, for DEV rows ONLY:
  per-game OOF prediction, outcome, X14 row, gid/season/week/teams/QBs.

MARKET-BLIND: no odds are read here. TEST (season >= 2016) rows are never fit,
never predicted, never scored and never written. Feature walks inside
build_main() run over the whole spine (they are as-of walks), but only DEV rows
leave this process.

Output: data/bt_nfl_ideas_dev.json (metadata) + data/bt_nfl_ideas_dev_X.npy
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
T0 = time.time()
import nfl_depth_eval as N  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

TEST_ERA = 2016
assert N.MAIN_HI < N.MAIN_TEST_ERA == TEST_ERA

X14, ex = N.build_main()
games, seasons, y_all = N.games, N.seasons, N.y
print(f"[{time.time()-T0:.0f}s] X14 built {X14.shape}", flush=True)

pred = np.full(len(y_all), np.nan)
for s_ in range(N.MAIN_LO, N.MAIN_HI + 1):
    assert s_ < TEST_ERA
    tr = (seasons < s_) & (y_all != 0.5)
    te = seasons == s_
    assert seasons[tr].max() < s_ and seasons[te].max() < TEST_ERA
    w = 0.5 ** ((s_ - 1 - seasons[tr]) / N.HL_BASE)
    lm = LogisticRegression(C=N.C_BASE, max_iter=5000).fit(X14[tr], y_all[tr], sample_weight=w)
    pred[te] = lm.predict_proba(X14[te])[:, 1]

dev = (seasons >= N.MAIN_LO) & (seasons <= N.MAIN_HI)
assert seasons[dev].max() < TEST_ERA
ll = N.llv(y_all[dev], pred[dev]).mean()
print(f"DEV 2006-2015 walk-forward LL {ll:.6f} (recorded 0.62292) n={int(dev.sum())}", flush=True)
assert abs(ll - 0.62292) < 5e-4

# DEV-era training rows (1999-2005) are also cached (features only + outcome) so
# downstream screens can refit the walk-forward with extra columns.
keep = seasons < TEST_ERA
idx = np.where(keep)[0]
meta = []
for i in idx:
    g = games[i]
    meta.append({k: g.get(k) for k in ("gid", "season", "week", "type", "home", "away",
                                         "home_qb", "away_qb", "hrest", "arest", "neutral",
                                         "kick", "y")})
    meta[-1]["pred"] = None if np.isnan(pred[i]) else float(pred[i])
np.save("data/bt_nfl_ideas_dev_X.npy", X14[idx])
with open("data/bt_nfl_ideas_dev.json", "w", encoding="utf-8") as fh:
    json.dump({"n": len(idx), "dev_ll": float(ll), "games": meta,
               "qh": ex["qh"][idx].tolist(), "qa": ex["qa"][idx].tolist(),
               # the retired team off/def TrueSkill edge (ledger row 6; col 7 before
               # row 63) - kept only as a DEV-era stand-in for an opponent-adjusted
               # play-level rating when checking redundancy of new candidates
               "tse_old": np.asarray(N.F["tse"])[idx].tolist()}, fh)
print(f"[{time.time()-T0:.0f}s] cached {len(idx)} pre-2016 rows", flush=True)
