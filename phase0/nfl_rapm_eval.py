"""RAPM: ridge plus-minus over all 22 on-field players (participation 2016-2025).

Ratings: epa ~ sum(beta_off of 11 offense) - sum(beta_def of 11 defense), sparse ridge,
refit at each season start on ALL prior seasons (as-of, leak-free). Team feature per
game: sum over this game's actives (= pre-kickoff inactives list at serve time) of
EWMA participation share x rating, offense + defense, home-away net.

SECONDARY PROTOCOL #2 (locked): ratings usable 2017+, DEV 2018-2022 (CV screen,
gate -0.0020), TEST 2023-2025 scored once if gate met. Power is poor (n~850) — stated.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

XBASE = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts])
GATE = 0.0020

# ---- load rapm plays ----
rows = []
with open("data/nfl_rapm_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    header = next(rd)
    ix = {c: i for i, c in enumerate(header)}
    for r in rd:
        rows.append((r[ix["game_id"]], r[ix["offense_players"]],
                     r[ix["defense_players"]], float(r[ix["epa"]])))
print(f"rapm plays: {len(rows):,}")

play_season = np.array([int(r[0][:4]) for r in rows])
pid_ix = {}
def pix(pid, role):
    key = (pid, role)
    if key not in pid_ix:
        pid_ix[key] = len(pid_ix)
    return pid_ix[key]

# pre-index all plays (rows: play, cols: player-role, +1 off / -1 def)
rr, cc, vv, yy = [], [], [], []
for n, (gid, offp, defp, epa) in enumerate(rows):
    yy.append(epa)
    for pid in offp.split(";"):
        rr.append(n); cc.append(pix(pid, "O")); vv.append(1.0)
    for pid in defp.split(";"):
        rr.append(n); cc.append(pix(pid, "D")); vv.append(-1.0)
A = sparse.csr_matrix((vv, (rr, cc)), shape=(len(rows), len(pid_ix)))
Y = np.array(yy)
print(f"design {A.shape[0]:,} x {A.shape[1]:,}")

def fit_ratings(upto_season, lam):
    m = play_season < upto_season
    rg = Ridge(alpha=lam, fit_intercept=True, solver="sparse_cg", tol=1e-4)
    rg.fit(A[m], Y[m])
    return rg.coef_

# ---- per-game participation shares + actives ----
gp = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # gid -> pid -> [off_ct, def_ct, _]
team_plays = defaultdict(lambda: defaultdict(int))          # gid -> team? (not needed: use counts)
game_off = defaultdict(int)
for gid, offp, defp, epa in rows:
    game_off[gid] += 1
    for pid in offp.split(";"):
        gp[gid][pid][0] += 1
    for pid in defp.split(";"):
        gp[gid][pid][1] += 1

# map gid -> (home, away) from games; walk games chronologically
def rapm_feature(lam):
    ratings = {}
    cur_fit_season = None
    share = {}                    # pid -> {"O": [ewma, w], "D": [ewma, w]}
    team_of, roster = {}, defaultdict(set)
    f = np.zeros(len(games))
    for i, g in enumerate(games):
        s = g["season"]
        if s < 2017:
            # warm shares only
            pass
        elif cur_fit_season != s:
            ratings = dict(zip(pid_ix.keys(),
                               fit_ratings(s, lam))) if s != cur_fit_season else ratings
            cur_fit_season = s
        gid = g["gid"]
        tbl = gp.get(gid)
        if tbl is not None and s >= 2017:
            # feature BEFORE updating shares: actives = players in tbl
            # attribute actives to teams via roster (last-seen team); newcomers skipped
            val = {g["home"]: 0.0, g["away"]: 0.0}
            for pid, (oc, dc, _) in tbl.items():
                t = team_of.get(pid)
                if t not in val:
                    continue
                sh = share.get(pid)
                if not sh:
                    continue
                for role, st in sh.items():
                    if st[1] <= 0:
                        continue
                    w = st[0] / st[1]
                    b = ratings.get((pid, role))
                    if b is not None:
                        val[t] += w * b * (11.0 if role == "O" else 11.0)
            f[i] = val[g["home"]] - val[g["away"]]
        # update shares/rosters AFTER
        if tbl is not None:
            n_off = max(game_off.get(gid, 1), 1)
            for pid, (oc, dc, _) in tbl.items():
                sh = share.setdefault(pid, {"O": [0.0, 0.0], "D": [0.0, 0.0]})
                sh["O"][0] = 0.8 * sh["O"][0] + oc / n_off
                sh["O"][1] = 0.8 * sh["O"][1] + 1.0
                sh["D"][0] = 0.8 * sh["D"][0] + dc / n_off
                sh["D"][1] = 0.8 * sh["D"][1] + 1.0
            # team attach: majority side unknown -> use game teams; pick by co-occurrence
        # roster attach via snap tables is cleaner, but participation lacks team; infer:
        # players in this game belong to home or away; assign by whichever roster they
        # were in, else defer to next game (two-game co-occurrence resolves)
        if tbl is not None:
            for pid in tbl:
                if pid not in team_of:
                    team_of[pid] = None
            # resolve: players seen in a previous game of ONE of these teams keep team;
            # unresolved get assigned by intersection at their NEXT appearance
            for pid in tbl:
                if team_of.get(pid) is None:
                    cand = prev_teams.get(pid)
                    if cand:
                        both = cand & {g["home"], g["away"]}
                        if len(both) == 1:
                            team_of[pid] = both.pop()
                prev_teams.setdefault(pid, set()).update({g["home"], g["away"]})
    return f

prev_teams = {}

def llv(yy_, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy_ * np.log(p) + (1 - yy_) * np.log(1 - p))

def cv_ll(X, mask):
    idx = np.where(mask)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fitm], y[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()

dev3 = (seasons >= 2018) & (seasons <= 2022)
test3 = (seasons >= 2023) & (seasons <= 2025)
base_cv = cv_ll(XBASE, dev3)
print(f"secondary#2 split: DEV {int(dev3.sum())}  TEST {int(test3.sum())}")
print(f"current model DEV CV {base_cv:.5f}   (gate: <= {base_cv - GATE:.5f})\n")

best = None
for lam in (100.0, 300.0, 1000.0, 3000.0):
    prev_teams = {}
    t0 = time.time()
    fr = rapm_feature(lam)
    s = cv_ll(np.column_stack([XBASE, fr]), dev3)
    tag = ""
    if best is None or s < best[0]:
        best = (s, lam, fr); tag = "  <-- best"
    print(f"  lambda {lam:>6.0f}  DEV CV {s:.5f} ({s-base_cv:+.5f})  ({time.time()-t0:.0f}s){tag}",
          flush=True)

s, lam, fr = best
gate_met = s <= base_cv - GATE
print(f"\nDEV verdict: lambda={lam:.0f} ({s-base_cv:+.5f}) -> "
      f"{'GATE MET -> ONE TEST look' if gate_met else 'gate not met -> parked'}")
json.dump({"dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5), "lambda": lam,
           "gate_met": bool(gate_met)}, open("data/nfl_rapm.json", "w"), indent=1)

if gate_met:
    Xw = np.column_stack([XBASE, fr])
    fitm = dev3 & (y != 0.5)
    cb = LogisticRegression(C=1e6, max_iter=5000).fit(XBASE[fitm], y[fitm])
    cw = LogisticRegression(C=1e6, max_iter=5000).fit(Xw[fitm], y[fitm])
    pb = cb.predict_proba(XBASE[test3])[:, 1]
    pw = cw.predict_proba(Xw[test3])[:, 1]
    yt = y[test3]
    d = llv(yt, pb) - llv(yt, pw)
    rng2 = np.random.default_rng(7)
    n = len(d)
    bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
    print(f"\nsecondary TEST 2023-2025 (n={n}, ONE look):")
    print(f"  current model LL {llv(yt, pb).mean():.5f}")
    print(f"  + RAPM        LL {llv(yt, pw).mean():.5f}")
    print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
