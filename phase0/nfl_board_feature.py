"""Test the FINAL board rating as the model's roster-quality input.

roster_quality variant: offensive actives valued by the board rating (per-play
opponent-adjusted EPA residual, decayed, EB to position replacement) — defenders keep
the weekly composite (board covers offense only). vs the current model (weekly
composite for everyone). ONE TEST compare; adopt if better (Samuel's rule).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # noqa: S102
exec("#" + full_src.split("# pass 1")[1].split("# ---------------- screens")[0])  # noqa: S102
# base features incl current f_rq; Z tables; composite machinery

epaP = json.load(open("data/nfl_epa_rating.json"))["params"]

plays_by_game = defaultdict(list)
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        plays_by_game[r[ix["game_id"]]].append(
            (PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]]),
             PBP_FIX.get(r[ix["defteam"]], r[ix["defteam"]]),
             float(r[ix["epa"]]), r[ix["passer_player_id"]],
             r[ix["rusher_player_id"]], r[ix["receiver_player_id"]]))

posmap = {}
for (pid, s, w), d in OFFW.items():
    posmap[pid] = d["pos"]

DECAY_D = 0.998
S_DECAY = 0.30
P_EB = 150.0
REP_D = 0.5

# ---- pass A: DEV-era position norms (identical to the board build) ----
dfa = defaultdict(lambda: [0.0, 0.0])
def drate(t):
    st = dfa[t]
    return (st[0] + epaP["prior_n"] * LG_EPA) / (st[1] + epaP["prior_n"])

P_ST = defaultdict(lambda: [0.0, 0.0])
dev_snap = defaultdict(list)
prev = None
for g in games:
    s = g["season"]
    if prev is not None and s != prev:
        for st in P_ST.values():
            st[0] *= (1 - S_DECAY); st[1] *= (1 - S_DECAY)
        for st in dfa.values():
            st[0] *= (1 - epaP["season_decay"]); st[1] *= (1 - epaP["season_decay"])
    prev = s
    gte = defaultdict(lambda: [0.0, 0])
    for (pt, dt, epa, pa, ru, re) in plays_by_game.get(g["gid"], ()):
        resid = epa - drate(dt)
        for pid in (pa, ru, re):
            if pid:
                st = P_ST[pid]
                st[0] = DECAY_D * st[0] + resid
                st[1] = DECAY_D * st[1] + 1.0
        gte[pt][0] += epa; gte[pt][1] += 1
    for t, (sm, n) in gte.items():
        opp = g["away"] if t == g["home"] else g["home"]
        st = dfa[opp]
        st[0] = epaP["decay"] * st[0] + sm
        st[1] = epaP["decay"] * st[1] + n
    if 2001 <= s <= 2015 and g["week"] == 10:
        for pid, st in P_ST.items():
            if st[1] >= 100:
                dev_snap[posmap.get(pid, "?")].append(st[0] / st[1])
NORM = {}
for pos, vals in dev_snap.items():
    v = np.array(vals)
    if len(v) > 50:
        NORM[pos] = (float(v.mean()), float(v.std()))
print("norms:", {p: (round(m, 3), round(sd, 3)) for p, (m, sd) in NORM.items()})

def board_z(pid):
    pos = posmap.get(pid)
    if pos not in NORM:
        return None
    st = P_ST.get(pid)
    if st is None or st[1] < 10:
        return None
    mu_p, sd_p = NORM[pos]
    rep = mu_p - REP_D * sd_p
    rate = (st[0] + P_EB * rep) / (st[1] + P_EB)
    return (rate - mu_p) / sd_p

# ---- pass B: full walk with the feature ----
P_ST.clear(); dfa.clear()
STATE.clear()
share4, pos4, team4 = {}, {}, {}
roster4 = defaultdict(set)
f_bq = np.zeros(len(games))
done4 = set()
prev = None

def rqual(team, gid):
    tbl = snaps.get((gid, team))
    if tbl is None:
        return 0.0
    tot = 0.0
    for pid in tbl:
        st = share4.get(pid)
        if not st or st[1] <= 0:
            continue
        sh = st[0] / st[1]
        p = pos4.get(pid, "")
        g_ = pfr2gsis.get(pid)
        if p in DEFPOS:
            r_ = rate_of(g_) if g_ else None
            if r_ is not None:
                tot += sh * max(-2.0, min(2.0, r_))
        else:
            z = board_z(g_) if g_ else None
            if z is not None:
                tot += sh * max(-2.0, min(2.0, z))
    return tot

for i, g in enumerate(games):
    s, w = g["season"], g["week"]
    if prev is not None and s != prev:
        for st in P_ST.values():
            st[0] *= (1 - S_DECAY); st[1] *= (1 - S_DECAY)
        for st in dfa.values():
            st[0] *= (1 - epaP["season_decay"]); st[1] *= (1 - epaP["season_decay"])
    prev = s
    if s >= 2013:
        f_bq[i] = rqual(g["home"], g["gid"]) - rqual(g["away"], g["gid"])
    gte = defaultdict(lambda: [0.0, 0])
    for (pt, dt, epa, pa, ru, re) in plays_by_game.get(g["gid"], ()):
        resid = epa - drate(dt)
        for pid in (pa, ru, re):
            if pid:
                st = P_ST[pid]
                st[0] = DECAY_D * st[0] + resid
                st[1] = DECAY_D * st[1] + 1.0
        gte[pt][0] += epa; gte[pt][1] += 1
    for t, (sm, n) in gte.items():
        opp = g["away"] if t == g["home"] else g["home"]
        st = dfa[opp]
        st[0] = epaP["decay"] * st[0] + sm
        st[1] = epaP["decay"] * st[1] + n
    for team in (g["home"], g["away"]):
        for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
            if (pid, s, w) in done4:
                continue
            done4.add((pid, s, w))
            apply_week(pid, s, w)
    for team in (g["home"], g["away"]):
        tbl = snaps.get((g["gid"], team))
        if not tbl:
            continue
        for pid, (pos, op, dp) in tbl.items():
            pct = dp if pos in DEFPOS else op
            st = share4.setdefault(pid, [0.0, 0.0])
            st[0] = snapP["decay"] * st[0] + pct
            st[1] = snapP["decay"] * st[1] + 1.0
            pos4[pid] = pos
            if team4.get(pid) != team:
                if team4.get(pid) in roster4:
                    roster4[team4.get(pid)].discard(pid)
                team4[pid] = team
                roster4[team].add(pid)

# ---- screens ----
def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def cv_ll(X):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fitm], y[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()

X_cur = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, f_rq])
X_rep = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, f_bq])
X_add = np.column_stack([X_cur, f_bq])
print(f"\nDEV CV: current {cv_ll(X_cur):.5f}  board-replace {cv_ll(X_rep):.5f}  "
      f"board-add {cv_ll(X_add):.5f}")
variants = {"replace": X_rep, "add": X_add}
best_name = min(variants, key=lambda k: cv_ll(variants[k]))
Xn = variants[best_name]
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
print(f"main TEST ({best_name}): current {llv(yt, p0).mean():.5f} -> "
      f"{llv(yt, p1).mean():.5f}  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])"
      f"  -> {'ADOPT' if better else 'keep current'}")
json.dump({"variant": best_name, "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_board_feature.json", "w"), indent=1)
