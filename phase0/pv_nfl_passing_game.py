"""Game-level INDICATION for the NFL passing rating (pv_nfl_passing_build.py). DEV ONLY.

Not the pre-registered screen. Uses the cached shipped 14-feature DEV matrix
(data/bt_nfl_ideas_dev_X.npy, 1999-2015 rows only) and the shipped walk-forward
protocol verbatim (expanding refit per season, recency HL=3, C=100, ties dropped
from fits, DEV 2006-2015 scored), and asks whether the passing composite, keyed
on each side's STARTER (first dropback passer), beats the shipped QB feature
(col 1) keyed on the starter too (the leak-free comparator; the cached col 1 uses
the post-game primary passer).

Arms (declared before running; no knob search):
  B0   cached X14                                   (harness sanity: 0.622937)
  B0x  drop col 1                                   (known effect: ~ -0.0061)
  B1   col 1 = shipped QbElo, starter-keyed         (the served feature; reference)
  B2   col 1 = passing composite diff (replace)
  B3   B1 + passing composite diff as a 15th column (add)
  B4   col 1 = opp-adjusted EPA Kalman diff (replace; ablation of the composite)
  -- declared after B0-B4 were run (the MLB 'player inside the team rating' step):
  B5   B1 + starter-aware pass matchup from the 3-state filter (add):
         (q_home_starter + cast_home + d_away) - (q_away_starter + cast_away + d_home)
  B6   B1 with col 12 (team pass_net) REPLACED by that starter-aware matchup
Paired game bootstrap (4000) vs B1. No odds read. No season >= 2016 row exists.
Output: data/pv_nfl_passing_game.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260927)

D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
GID = [g["gid"] for g in G]

V = pd.read_csv("data/pv_nfl_passing_values.csv")
T = pd.read_csv("data/pv_nfl_passing_team.csv")
assert V.season.max() < TEST_ERA
starter = {(g, t): s for g, t, s in zip(T.game_id, T.posteam, T.starter)}
V = V[V.is_starter == 1]
V["comp_f"] = V.passing_composite.fillna(V.epa_q)          # 1999 (warm-up only) has no composite yet
V["u3"] = V.epa3_q + V.epa3_cast + V.epa3_opp_def          # expected pass EPA/db of this side tonight
val = {(g, sd): (c, e, u) for g, sd, c, e, u in zip(V.game_id, V.side, V.comp_f, V.epa_q, V.u3)}
SS = json.load(open("data/pv_nfl_passing_qbelo_starter_sides.json"))

comp = np.zeros(N); epa = np.zeros(N); qbs = np.zeros(N); u3 = np.zeros(N); miss = 0
for i, gid in enumerate(GID):
    h, a = val.get((gid, "home")), val.get((gid, "away"))
    if h is None or a is None:
        miss += 1
    else:
        comp[i] = h[0] - a[0]; epa[i] = h[1] - a[1]; u3[i] = h[2] - a[2]
    qbs[i] = SS[gid][0] - SS[gid][1]
print(f"games {N}, sides without a pbp starter value: {miss} "
      f"(DEV rows missing: {sum(1 for i, g in enumerate(GID) if SEAS[i] >= DEV_LO and (val.get((g, 'home')) is None or val.get((g, 'away')) is None))})")


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wf(X):
    per, vec = {}, []
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(X[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec)


def col1(v):
    X = X14.copy(); X[:, 1] = v; return X


arms = {
    "B0 cached X14 (listed QB)": X14,
    "B0x drop col 1": np.delete(X14, 1, axis=1),
    "B1 col1 = shipped QbElo starter-keyed (reference)": col1(qbs),
    "B2 col1 = passing composite (replace)": col1(comp),
    "B3 B1 + passing composite (add)": np.column_stack([col1(qbs), comp]),
    "B4 col1 = opp-adj EPA Kalman (replace)": col1(epa),
    "B5 B1 + starter-aware pass matchup (add)": np.column_stack([col1(qbs), u3]),
    "B6 B1, col12 pass_net -> starter-aware matchup": (lambda X: (X.__setitem__((slice(None), 12), u3), X)[1])(col1(qbs)),
}
dv = SEAS >= DEV_LO
print("corr(u3, col12 pass_net) on DEV rows: %.3f" % np.corrcoef(u3[dv], X14[dv, 12])[0, 1])
res = {k: wf(X) for k, X in arms.items()}
b0 = res["B0 cached X14 (listed QB)"][0]
print(f"harness sanity: B0 {b0:.6f} (recorded 0.622937)")
assert abs(b0 - 0.622937) < 5e-5
ref_ll, ref_per, ref_vec = res["B1 col1 = shipped QbElo starter-keyed (reference)"]
OUT = {"n_dev_games": int(len(ref_vec)), "arms": {}}
for k, (ll, per, vec) in res.items():
    d = ref_vec - vec                                   # > 0: arm better than B1
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if ref_per[s] - per[s] > 0)
    OUT["arms"][k] = {"dev_ll": round(ll, 6), "gain_vs_B1": round(float(d.mean()), 6),
                      "ci": [round(float(lo), 6), round(float(hi), 6)], "seasons_better": f"{pos}/10",
                      "per_season_gain": {str(s): round(ref_per[s] - per[s], 5) for s in per}}
    print(f"  {k:<52} LL {ll:.6f}  gain vs B1 {d.mean():+.6f}  CI[{lo:+.6f},{hi:+.6f}]  seasons+ {pos}/10")
json.dump(OUT, open("data/pv_nfl_passing_game.json", "w"), indent=1)
print("wrote data/pv_nfl_passing_game.json")
