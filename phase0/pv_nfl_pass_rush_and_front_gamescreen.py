"""pv NFL pass_rush_and_front -- step 6: DEV-only game-level INDICATION on the shipped harness.

Same walk-forward as bt_nfl_ideas_quick3.wf(): the shipped 14 columns
(data/bt_nfl_ideas_dev_X.npy, 1999-2015), each DEV season 2006-2015 scored by a
logistic blend fit on all earlier seasons (half-life 3 seasons, C=100, ties
dropped from fits); new columns z-scored on the train fold. Harness sanity:
baseline DEV LL must reproduce 0.62294 and dropping the QB column must cost
about +0.0061 (movcore sanity arm; ledger row 3 +0.0069).

Columns (all PRE-GAME, from pv_nfl_pass_rush_and_front_team_games.csv):
  PRIMARY (pre-declared) F  = front block, two columns:
     pass-rush matchup  [log D_pr(home def) + log O_pr(away off)] - [log D_pr(away def) + log O_pr(home off)]
     run-front matchup  [log D_st(home def) + log O_st(away off)] - [same, away]
     (scorer factor cancels: both sides play in the same stadium)
  diagnostics (not candidates): F_sk sack matchup only; F_agg pass-rush matchup
  with the PLAYER aggregate in place of the unit factor; F_prot protection only
  (log O_pr home off - log O_pr away off).
Games before 2006 carry pressure = sacks (QB hits not logged 2003-2005) -- the
blend's training rows see a sack-only version of the column for those years.
Output: data/pv_nfl_pass_rush_and_front_gamescreen.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA, DEV_LO, DEV_HI, HL, C = 2016, 2006, 2015, 3.0, 100.0
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
RNG = np.random.default_rng(20260927)

D = json.load(open(os.path.join(DATA, "bt_nfl_ideas_dev.json"), encoding="utf-8"))
G = D["games"]
X14 = np.load(os.path.join(DATA, "bt_nfl_ideas_dev_X.npy"))
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
TG = pd.read_csv(os.path.join(DATA, "pv_nfl_pass_rush_and_front_team_games.csv"))
assert TG.season.max() < TEST_ERA
T = TG.set_index(["game_id", "defteam"])


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def col(fn):
    v = np.full(len(G), np.nan)
    for i, g in enumerate(G):
        h, a = FR.get(g["home"], g["home"]), FR.get(g["away"], g["away"])
        if (g["gid"], h) in T.index and (g["gid"], a) in T.index:
            v[i] = fn(T.loc[(g["gid"], h)], T.loc[(g["gid"], a)])
    return v


lg = np.log
F_pr = col(lambda H, A: (lg(H.D_pr) + lg(H.Oopp_pr)) - (lg(A.D_pr) + lg(A.Oopp_pr)))
F_st = col(lambda H, A: (lg(H.D_st) + lg(H.Oopp_st)) - (lg(A.D_st) + lg(A.Oopp_st)))
F_sk = col(lambda H, A: (lg(H.D_sk) + lg(H.Oopp_sk)) - (lg(A.D_sk) + lg(A.Oopp_sk)))
F_agg = col(lambda H, A: (lg(max(H.agg_pr, 1e-3)) + lg(H.Oopp_pr)) - (lg(max(A.agg_pr, 1e-3)) + lg(A.Oopp_pr)))
# home row (defteam=home) holds the AWAY offence's factor; protection diff = home offence - away offence
F_prot = col(lambda H, A: lg(A.Oopp_pr) - lg(H.Oopp_pr))
miss = int(np.isnan(F_pr).sum())
for v in (F_pr, F_st, F_sk, F_agg, F_prot):
    v[np.isnan(v)] = 0.0                       # 3 schedule games absent from pbp (1999-2000): neutral


def wf(extra=None, drop=None):
    X = X14.copy()
    if drop is not None:
        X = np.delete(X, drop, axis=1)
    per, vec, coefs = {}, [], {}
    for s_ in range(DEV_LO, DEV_HI + 1):
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_
        Xs = X
        if extra is not None:
            Ex = np.column_stack(extra)
            mu, sd = Ex[tr].mean(0), Ex[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (Ex - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
        coefs[s_] = m.coef_[0][X.shape[1]:].round(4).tolist()
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), coefs


b_ll, b_per, b_vec, _ = wf()
assert abs(b_ll - 0.62294) < 5e-4, b_ll
OUT = {"baseline_dev_ll": round(b_ll, 6), "games_without_pbp_neutral": miss, "results": {}}


def rep(nm, res):
    ll, per, vec, coefs = res
    d = b_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    print(f"  {nm:<44} {ll:.5f} gain {b_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10", flush=True)
    OUT["results"][nm] = {"ll": round(ll, 6), "gain": round(b_ll - ll, 6), "ci95": [round(lo, 6), round(hi, 6)],
                          "seasons_pos": pos, "per_season_gain": {str(s): round(b_per[s] - per[s], 5) for s in per},
                          "added_coefs_by_fold": {str(k): v for k, v in coefs.items()}}


print(f"baseline DEV LL {b_ll:.5f}")
rep("SANITY drop QB column (expect ~ -0.0061)", wf(drop=1))
rep("PRIMARY F front block (pass-rush + run matchups)", wf([F_pr, F_st]))
rep("diag F_pr pass-rush matchup only", wf([F_pr]))
rep("diag F_st run-front matchup only", wf([F_st]))
rep("diag F_sk sack matchup only", wf([F_sk]))
rep("diag F_agg player-aggregate pass-rush matchup", wf([F_agg]))
rep("diag F_prot protection (QB+OL) only", wf([F_prot]))
OUT["bar"] = "NFL TEST look requires DEV gain >= +0.00150 with bootstrap CI lower bound > 0 (prereg 2026-09-24)"
OUT["primary_clears_bar"] = bool(OUT["results"]["PRIMARY F front block (pass-rush + run matchups)"]["gain"] >= 0.0015
                                 and OUT["results"]["PRIMARY F front block (pass-rush + run matchups)"]["ci95"][0] > 0)
json.dump(OUT, open(os.path.join(DATA, "pv_nfl_pass_rush_and_front_gamescreen.json"), "w"), indent=1)
print("primary clears bar:", OUT["primary_clears_bar"])
