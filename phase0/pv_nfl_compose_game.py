"""Player-value program (pv), NFL, COMPOSE step 4: game-level INDICATION on the shipped harness. DEV ONLY.

Not the pre-registered screen (the lead engineer owns that). Uses the cached shipped 14-feature DEV
matrix (data/bt_nfl_ideas_dev_X.npy, rows 1999-2015 only) and the shipped walk-forward protocol
verbatim (expanding refit per season, recency half-life 3 seasons, C=100, ties dropped from fits,
DEV 2006-2015 scored, new columns z-scored on the train fold), exactly as
bt_nfl_ideas_quick3.wf / pv_nfl_passing_game.wf.

Features come from data/pv_nfl_compose_game_features.csv (all pre-game, as-of weights). Games of
the spine without a pbp value (3 in 1999-2000, never scored) get 0.

Arms, declared before the harness was run (the team-level validation of step 3 had been seen):
  G0   cached X14                                    harness sanity: must reproduce 0.622937
  G0x  drop col 1 (QB)                               known effect: about -0.0061
  G1   + LV diff (composed lineup value, QB incl.)   PRIMARY
  G2   + [LV-QV diff, QV diff]                       lineup and starting QB separately
  G3   + LVX diff (expected roster, no day-of info)
  G4   col 11 (roster_quality, rate6) REPLACED by LV diff
  G5   + TS diff (11v11 shared-credit analog)        comparator
  G6   + LV diff + TS diff
  G7   + the 8 unit aggregate diffs (nominal EPA/game; the logistic fit weighs the units itself)
Paired game bootstrap (4000) of per-game log-loss differences vs G0 (> 0: arm better).
Bar (pre-registered, NFL): gain >= +0.00150 with 95% CI lower bound > 0.
Output: data/pv_nfl_compose_game.json. No odds read. No season >= 2016 row exists.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260928)
UNITS = ["QB", "REC", "RUSH", "PROT", "PR", "RF", "BALL", "COV"]

D = json.load(open(os.path.join(DATA, "bt_nfl_ideas_dev.json"), encoding="utf-8"))
G = D["games"]
X14 = np.load(os.path.join(DATA, "bt_nfl_ideas_dev_X.npy"))
N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
F = pd.read_csv(os.path.join(DATA, "pv_nfl_compose_game_features.csv"))
assert F.season.max() < TEST_ERA
F = F.set_index("game_id")


rowsF = [F.loc[g["gid"]] if g["gid"] in F.index else None for g in G]


def diff(a, b):
    return np.array([0.0 if r is None else float(r[a] - r[b]) for r in rowsF])


LV = diff("lv_home", "lv_away"); QV = diff("qv_home", "qv_away"); NQ = LV - QV
LVX = diff("lvx_home", "lvx_away"); TS = diff("ts_home", "ts_away")
AU = [diff(f"A_{u}_home", f"A_{u}_away") for u in UNITS]
nomiss = sum(r is None for r in rowsF)
dv = SEAS >= DEV_LO
assert all(rowsF[i] is not None for i in np.where(dv)[0]), "a DEV game has no feature row"
print(f"spine {N} games, {nomiss} without a pbp feature row (all pre-2006); DEV corr LV vs shipped cols: "
      + ", ".join(f"c{k} {np.corrcoef(LV[dv], X14[dv, k])[0, 1]:+.2f}" for k in range(14) if X14[dv, k].std() > 0))


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wf(extra=None, drop=None, replace=None):
    X = X14.copy()
    if replace:
        for c, v in replace.items():
            X[:, c] = v
    if drop:
        X = np.delete(X, drop, axis=1)
    per, vec, coefs = {}, [], {}
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X
        if extra is not None:
            E = np.column_stack(extra) if isinstance(extra, (list, tuple)) else extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
        coefs[s_] = m.coef_[0][X.shape[1]:].round(4).tolist()
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), coefs


arms = {
    "G0 cached X14": wf(),
    "G0x drop col 1 (QB)": wf(drop=[1]),
    "G1 + LV diff (PRIMARY)": wf(LV),
    "G2 + [LV-QV, QV]": wf([NQ, QV]),
    "G3 + LVX diff (no day-of info)": wf(LVX),
    "G4 col 11 (rq) -> LV diff": wf(replace={11: LV}),
    "G5 + TS diff (11v11 analog)": wf(TS),
    "G6 + LV diff + TS diff": wf([LV, TS]),
    "G7 + 8 unit aggregate diffs": wf(AU),
}
b_ll, b_per, b_vec, _ = arms["G0 cached X14"]
print(f"harness sanity: G0 {b_ll:.6f} (recorded 0.622937)")
assert abs(b_ll - 0.622937) < 5e-5
OUT = {"n_dev_games": int(len(b_vec)), "baseline_dev_ll": b_ll, "arms": {},
       "bar": "gain >= +0.00150 and bootstrap 95% CI lower bound > 0 (NFL pre-registered)"}
for k, (ll, per, vec, coefs) in arms.items():
    d = b_vec - vec                                   # > 0: arm better than G0
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    OUT["arms"][k] = {"dev_ll": round(ll, 6), "gain_vs_G0": round(float(d.mean()), 6),
                      "ci": [round(float(lo), 6), round(float(hi), 6)], "seasons_better": f"{pos}/10",
                      "per_season_gain": {str(s): round(b_per[s] - per[s], 5) for s in per},
                      "added_coefs_last_fold": coefs[DEV_HI],
                      "clears_bar": bool(d.mean() >= 0.0015 and lo > 0)}
    print(f"  {k:<34} LL {ll:.6f}  gain {d.mean():+.6f}  CI[{lo:+.6f},{hi:+.6f}]  seasons+ {pos}/10  "
          f"last-fold added coefs {coefs[DEV_HI]}")
json.dump(OUT, open(os.path.join(DATA, "pv_nfl_compose_game.json"), "w"), indent=1)
print("wrote data/pv_nfl_compose_game.json")
