"""ORACLE WIN-GRAPH CENTRALITY (Balreira 2014, Or(w+,w+) wins-only variant).

Markov-centrality ranking on the within-season win graph, screened as a 15th
feature over the main 14-feature model. Margins deliberately EXCLUDED.

Mechanism (paper pp. 7-10): random walker moves from a team to the teams that
BEAT it (loser -> winner, ties 0.5 both ways), or up to an Oracle node with
weight u_i = w+_i = wins_i + 1; from the Oracle it moves down to team i with
weight d_i = w+_i. Column-stochastic matrix (columns = source node, Eq 2.2:
p_{n+1,i} = 1/(losses_i + 1) for Or(e,e), generalized here to Or(w+,w+)),
power-iterated to the stationary vector; drop oracle component, renormalize.
Feature: log(r_home / r_away) pre-game; 0.0 when either team unseen that
season (covers week 1). New season resets everything.

DEV 5-fold CV screen (2001-2015), gate delta >= 0.0020. NO TEST LOOK.
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

# ---- rebuild the shipped 14-col matrix (passrun cols + v7 ratings col) ----
from collections import defaultdict
aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
        a = aggc[r[ix["game_id"]]][t_]
        if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
            a[0] += float(r[ix["epa"]]); a[1] += 1
        else:
            a[2] += float(r[ix["epa"]]); a[3] += 1
ps = pn = rs = rn = 0.0
for gid, tm in aggc.items():
    if int(gid[:4]) in DEV_YEARS:
        for t_, (a_, b_, c_, d_) in tm.items():
            ps += a_; pn += b_; rs += c_; rn += d_
LGP, LGR = ps / pn, rs / rn
dec_, pn_, sd_ = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
    prev = g["season"]
    def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
    h, a = g["home"], g["away"]
    f_pass[i] = (crate(offP[h], LGP) - crate(dfaP[h], LGP)) - (crate(offP[a], LGP) - crate(dfaP[a], LGP))
    f_run[i] = (crate(offR[h], LGR) - crate(dfaR[h], LGR)) - (crate(offR[a], LGR) - crate(dfaR[a], LGR))
    tm = aggc.get(g["gid"])
    if tm:
        for t_off, opp in ((h, a), (a, h)):
            pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
            o = offP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
            o = offR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
            d2 = dfaP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
            d2 = dfaR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN

X14 = np.column_stack([X_of(F), f_pass, f_run])
X14[:, 7] = np.load("data/nfl_v7_feature.npy")
cur_cv = cv_ll(X14)
print(f"[{time.time()-T0:.0f}s] current model DEV CV {cur_cv:.5f}", flush=True)
assert abs(cur_cv - 0.62300) < 0.0007, f"repro FAILED: {cur_cv:.5f} vs 0.62300"

# ---------------- Oracle Or(w+,w+) feature ----------------
def oracle_walk(beats, wins, order):
    """Stationary vector of the Or(w+,w+) chain over teams in `order`.

    Column i (source) = games team i LOST (row = winner) + up-weight w+_i at
    the oracle row; oracle column (source) = down-weights w+_j on team rows.
    """
    T = len(order)
    idx = {t: k for k, t in enumerate(order)}
    O = np.zeros((T + 1, T + 1))
    for (wt, lt), c in beats.items():
        O[idx[wt], idx[lt]] += c            # loser column -> winner row
    wp = np.array([wins[t] + 1.0 for t in order])
    O[T, :T] = wp                           # up moves: team col -> oracle row
    O[:T, T] = wp                           # down moves: oracle col -> team rows
    P = O / O.sum(axis=0, keepdims=True)    # column-stochastic (cols all > 0)
    r = np.full(T + 1, 1.0 / (T + 1))
    for _ in range(50):
        r = P @ r
    r = r[:T]
    return r / r.sum(), idx

oracle_feat = np.zeros(len(games))
cur_season = None
seen = wins = beats = None
n_games_season = 0
for i, g in enumerate(games):
    if g["season"] != cur_season:
        cur_season = g["season"]
        seen = []                            # teams in order of first appearance
        wins = defaultdict(float)            # win count (ties = 0.5)
        beats = defaultdict(float)           # (winner, loser) -> count
    h, a = g["home"], g["away"]
    if h in wins and a in wins:              # both seen this season -> rateable
        r, idx = oracle_walk(beats, wins, seen)
        oracle_feat[i] = np.log(r[idx[h]] / r[idx[a]])
    # update the season graph with this game's result (post-hoc, no leak)
    for t in (h, a):
        if t not in wins:
            wins[t] = 0.0
            seen.append(t)
    yv = g["y"]
    if yv == 1.0:
        beats[(h, a)] += 1.0; wins[h] += 1.0
    elif yv == 0.0:
        beats[(a, h)] += 1.0; wins[a] += 1.0
    else:                                    # tie: 0.5 both ways
        beats[(h, a)] += 0.5; beats[(a, h)] += 0.5
        wins[h] += 0.5; wins[a] += 0.5
print(f"[{time.time()-T0:.0f}s] oracle feature built", flush=True)

nz_all = int((oracle_feat != 0.0).sum())
nz_dev = int(((oracle_feat != 0.0) & dev).sum())
print(f"coverage: nonzero {nz_all}/{len(games)} overall, {nz_dev}/{int(dev.sum())} dev",
      flush=True)

# ---------------- screens ----------------
X15 = np.column_stack([X14, oracle_feat])
cv15 = cv_ll(X15)

wk4 = np.array([g["week"] >= 4 for g in games])
feat_wk4 = np.where(wk4, oracle_feat, 0.0)
cv15b = cv_ll(np.column_stack([X14, feat_wk4]))

# containment diagnostics
m = dev & (oracle_feat != 0.0)
corr_elo = float(np.corrcoef(oracle_feat[m], X14[m, 0])[0, 1])
cv_elo = cv_ll(X14[:, [0]])
cv_elo_or = cv_ll(np.column_stack([X14[:, 0], oracle_feat]))

# full-dev coefficient sign (informative only)
fitm = np.where(dev & (y != 0.5))[0]
clf = LogisticRegression(C=1e6, max_iter=5000).fit(X15[fitm], y[fitm])
coef_or = float(clf.coef_[0][14])
coef_elo = float(clf.coef_[0][0])

print("\nORACLE Or(w+,w+) win-graph centrality (DEV 5-fold, gate +0.0020):", flush=True)
print(f"  base 14-col          CV {cur_cv:.5f}", flush=True)
print(f"  +oracle (15-col)     CV {cv15:.5f}  (delta {cur_cv-cv15:+.5f})", flush=True)
print(f"  +oracle wk4+ only    CV {cv15b:.5f}  (delta {cur_cv-cv15b:+.5f})", flush=True)
print("\ncontainment diagnostics:", flush=True)
print(f"  corr(oracle, elo logit) on dev nonzero rows: {corr_elo:+.4f}  (n={int(m.sum())})",
      flush=True)
print(f"  elo alone            CV {cv_elo:.5f}", flush=True)
print(f"  elo + oracle (2-col) CV {cv_elo_or:.5f}  (delta vs elo {cv_elo-cv_elo_or:+.5f})",
      flush=True)
print(f"  full-dev fit coefs: elo {coef_elo:+.4f}, oracle {coef_or:+.4f}", flush=True)

out = {
    "screen": "oracle_wingraph_centrality_Or(w+,w+)",
    "base_cv": round(float(cur_cv), 5),
    "variants": {
        "oracle15": {"cv": round(float(cv15), 5), "delta": round(float(cur_cv - cv15), 5)},
        "oracle15_wk4plus": {"cv": round(float(cv15b), 5), "delta": round(float(cur_cv - cv15b), 5)},
    },
    "gate": 0.0020,
    "gate_met": bool(max(cur_cv - cv15, cur_cv - cv15b) >= 0.0020),
    "diagnostics": {
        "corr_oracle_elo_dev_nonzero": round(corr_elo, 4),
        "n_dev_nonzero": nz_dev,
        "n_dev": int(dev.sum()),
        "nonzero_all": nz_all,
        "n_games": len(games),
        "cv_elo_only": round(float(cv_elo), 5),
        "cv_elo_plus_oracle": round(float(cv_elo_or), 5),
        "delta_over_bare_elo": round(float(cv_elo - cv_elo_or), 5),
        "fulldev_coef_elo": round(coef_elo, 4),
        "fulldev_coef_oracle": round(coef_or, 4),
    },
}
json.dump(out, open("data/nfl_oracle_screen.json", "w"), indent=1)
print("\nwrote data/nfl_oracle_screen.json", flush=True)
