"""Replacement-level QB prior: fix the z=+3.5 unproven-QB bias, MLB protocol.

The diagnostic showed the model treats unknown QBs as league-average; reality says
replacement level. Fix: EB shrinkage target lg+delta (delta<0), never-seen QBs get the
full prior. Selection on DEV 5-fold CV only:
  V1 isolation: frozen QB dynamics, fine delta grid       (attribution)
  V2 joint:     random search over delta + QB dynamics    (best feature)
CV winner gets ONE TEST look vs the base blend (0.63059), paired bootstrap.
Also re-checks the unproven-away-QB bias slice on DEV after the fix.
"""
from __future__ import annotations

import csv
import json
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
P, lg = rep["qb_elo"]["params"], rep["qb_elo"]["lg_rate"]
FROZEN = dict(decay=P["decay"], prior_db=P["prior_db"], season_decay=P["season_decay"])

pe, y = run_elo(games, score_from=1999, **base["params"])
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)


def qb_feature(delta, decay, prior_db, season_decay):
    m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, decay=decay, prior_db=prior_db,
              season_decay=season_decay, lg_rate=lg, rep_delta=delta)
    out, dbh, prev = np.zeros(len(games)), np.zeros(len(games)), None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        out[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
        dbh[i] = m.Q.get(g["away_qb"], [0, 0])[1]     # away-QB known dropbacks (for slice)
        m.predict(g)
        m.update(g, 0.5, qbw)
    return out, dbh


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X, want_probs=False):
    idx = np.where(dev)[0]
    scores = np.zeros(len(idx))
    probs = np.full(len(games), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        pv = clf.predict_proba(X[iva])[:, 1]
        scores[va] = llv(y[iva], pv)
        probs[iva] = pv
    return (scores.mean(), probs) if want_probs else scores.mean()


# ---- V1: isolation sweep, frozen dynamics ----
print("V1 delta sweep (frozen dynamics), DEV 5-fold CV:")
qd0, dba0 = qb_feature(0.0, **FROZEN)
base_cv = cv_ll(np.column_stack([lgt, qd0]))
print(f"  delta +0.000 (current)  CV {base_cv:.5f}")
best1 = (base_cv, 0.0)
for delta in np.arange(-0.20, 0.021, 0.02):
    if abs(delta) < 1e-9:
        continue
    qd, _ = qb_feature(float(delta), **FROZEN)
    s = cv_ll(np.column_stack([lgt, qd]))
    tag = "  <-- best" if s < best1[0] else ""
    print(f"  delta {delta:+.3f}          CV {s:.5f}{tag}")
    if s < best1[0]:
        best1 = (s, float(delta))

# ---- V2: joint search ----
rng = np.random.default_rng(20260723)
best2, t0 = None, time.time()
for i in range(200):
    cand = dict(
        delta=float(rng.uniform(-0.25, 0.0)),
        decay=float(rng.uniform(0.80, 1.0)),
        prior_db=float(np.exp(rng.uniform(np.log(20.0), np.log(1500.0)))),
        season_decay=float(rng.uniform(0.0, 0.6)),
    )
    qd, _ = qb_feature(**cand)
    s = cv_ll(np.column_stack([lgt, qd]))
    if best2 is None or s < best2[0]:
        best2 = (s, cand)
print(f"\nV2 joint search (200 trials, {time.time()-t0:.0f}s): "
      f"CV {best2[0]:.5f}  {({k: round(v, 4) for k, v in best2[1].items()})}")

# ---- pick CV winner ----
if best2[0] < best1[0]:
    win_cv, win_kind = best2[0], "V2 joint"
    qd_w, dba_w = qb_feature(**best2[1])
    win_params = best2[1]
else:
    win_cv, win_kind = best1[0], f"V1 delta={best1[1]:+.3f}"
    qd_w, dba_w = qb_feature(best1[1], **FROZEN)
    win_params = {"delta": best1[1], **FROZEN}
print(f"\nCV winner: {win_kind}  (CV {win_cv:.5f} vs current {base_cv:.5f}, "
      f"{win_cv-base_cv:+.5f})")
if win_cv >= base_cv:
    print("no DEV improvement -> rejected on DEV, TEST not touched")
    sys.exit(0)

# ---- bias-slice recheck on DEV (out-of-fold) ----
_, p_new = cv_ll(np.column_stack([lgt, qd_w]), want_probs=True)
_, p_old = cv_ll(np.column_stack([lgt, qd0]), want_probs=True)
sl = dev & (dba_w < 150)
for tag, pp in [("before", p_old), ("after ", p_new)]:
    pm, am = pp[sl].mean(), y[sl].mean()
    sd = np.sqrt((pp[sl] * (1 - pp[sl])).sum()) / sl.sum()
    print(f"unproven-away-QB slice {tag}: n={sl.sum()}  pred {pm:.3f}  actual {am:.3f}"
          f"  bias {am-pm:+.3f}  z {((am-pm)/sd):+.1f}")

# ---- ONE TEST look ----
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
print(f"  base blend (delta=0)      LL {llv(yt, pb).mean():.5f}")
print(f"  replacement-prior blend   LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
json.dump({"winner": win_kind, "params": win_params, "dev_cv": round(win_cv, 5),
           "dev_cv_base": round(base_cv, 5),
           "test_base": round(float(llv(yt, pb).mean()), 5),
           "test_new": round(float(llv(yt, pw).mean()), 5),
           "delta_ll": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "verdict": sig},
          open("data/nfl_qb_replacement.json", "w"), indent=1)
