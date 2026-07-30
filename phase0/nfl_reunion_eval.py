"""THE REUNION: current 14-feature model + every market-blind rejected feature that
ever showed a positive delta. One TEST look, adopt if better (Samuel's rule).

Included (original tuned params): SRS (ridge-Massey, ridge=2 cap=28), close-game luck
(<=8, .95), injury playing-hurt qsh (2009+), non-QB missing production, coach rating,
macro-TrueSkill add. RAPM included but inert (zero across all of DEV -> coef unlearnable).
Excluded: nfelo (market). Prior: SRS+qsh were TEST-toxic individually; expect a loss.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

# ---------- current model: + pass/run channels ----------
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
def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
    prev = g["season"]
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
X_base = np.column_stack([X_of(F), f_pass, f_run])

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
margins = np.array([int(r["home_score"]) - int(r["away_score"]) for r in raw])

# ---------- 1. SRS (ridge=2, cap=28) ----------
f_srs = np.zeros(len(games))
by_season = defaultdict(list)
for i, g in enumerate(games):
    by_season[g["season"]].append(i)
for s_, idxs in by_season.items():
    teams = sorted({t for i in idxs for t in (games[i]["home"], games[i]["away"])})
    ti = {t: j for j, t in enumerate(teams)}
    n_ = len(teams)
    N = np.zeros(n_); A = np.zeros((n_, n_)); marg = np.zeros(n_)
    cur, r_ = None, np.zeros(n_)
    for i in idxs:
        g = games[i]
        if g["date"] != cur:
            r_ = np.linalg.solve(np.diag(N) - A + 2.0 * np.eye(n_), marg)
            cur = g["date"]
        f_srs[i] = r_[ti[g["home"]]] - r_[ti[g["away"]]]
        mg = max(-28, min(28, int(margins[i])))
        ih, ia = ti[g["home"]], ti[g["away"]]
        N[ih] += 1; N[ia] += 1; A[ih, ia] += 1; A[ia, ih] += 1
        marg[ih] += mg; marg[ia] -= mg

# ---------- 2. close-game luck (<=8, decay .95) ----------
f_cl = np.zeros(len(games))
stc = defaultdict(lambda: [0.0, 0.0])
for i, g in enumerate(games):
    h, a = g["home"], g["away"]
    f_cl[i] = stc[h][0] / (stc[h][1] + 4.0) - stc[a][0] / (stc[a][1] + 4.0)
    if abs(margins[i]) <= 8 and g["y"] != 0.5:
        for team, res in ((h, g["y"]), (a, 1.0 - g["y"])):
            s2 = stc[team]
            s2[0] = 0.95 * s2[0] + (res - 0.5)
            s2[1] = 0.95 * s2[1] + 1.0

# ---------- 3. injury qsh (2009+) ----------
inj = defaultdict(dict)
for yr in range(2009, 2026):
    try:
        fh = open(f"data/inj_{yr}.csv", encoding="utf-8")
    except FileNotFoundError:
        continue
    for r in csv.DictReader(fh):
        t = PBP_FIX.get(r["team"], r["team"])
        try:
            key = (int(r["season"]), int(r["week"]), t)
        except ValueError:
            continue
        inj[key][r["gsis_id"]] = r.get("report_status") or ""
QD = {"Questionable", "Doubtful"}
share5 = {}
f_qsh = np.zeros(len(games))
for i, g in enumerate(games):
    s_, w_ = g["season"], g["week"]
    if s_ >= 2009:
        for side, team in ((1, g["home"]), (-1, g["away"])):
            tbl = snaps.get((g["gid"], team))
            it = inj.get((s_, w_, team), {})
            if tbl is None or not it:
                continue
            for pid in tbl:
                st2 = share5.get(pid)
                if not st2 or st2[1] <= 0:
                    continue
                g_ = pfr2gsis.get(pid)
                if g_ and it.get(g_) in QD:
                    f_qsh[i] += side * (st2[0] / st2[1])
    for team in (g["home"], g["away"]):
        tbl = snaps.get((g["gid"], team))
        if not tbl:
            continue
        for pid, (pos, op, dp) in tbl.items():
            pct = dp if pos in DEFPOS else op
            st2 = share5.setdefault(pid, [0.0, 0.0])
            st2[0] = snapP["decay"] * st2[0] + pct
            st2[1] = snapP["decay"] * st2[1] + 1.0

# ---------- 4. missing production (tuned params) ----------
mpP = json.load(open("data/nfl_missing_prod.json"))["params"]
SKL = ("RB", "WR", "TE", "FB")
pw = {}; played = defaultdict(set)
for (pid, s2, w2), d3 in OFFW.items():
    if d3["pos"] in SKL:
        pw[(pid, s2, w2)] = (d3["repa"] + d3["cepa"], d3["car"] + d3["tgt"], d3["team"], d3["pos"])
        played[(d3["team"], s2, w2)].add(pid)
pos_num = defaultdict(float); pos_den = defaultdict(float)
for (pid, s2, w2), (epa2, opp2, t2, pos2_) in pw.items():
    if s2 in DEV_YEARS:
        p3 = "RB" if pos2_ in ("RB", "FB") else pos2_
        pos_num[p3] += epa2; pos_den[p3] += opp2
POS_LG = {p3: pos_num[p3] / pos_den[p3] for p3 in pos_num}
rating2, usage2, team_of2, pos_of2 = {}, {}, {}, {}
team_roster2 = defaultdict(set)
f_mp = np.zeros(len(games))
def pval(pid):
    pos3 = "RB" if pos_of2.get(pid) in ("RB", "FB") else pos_of2.get(pid, "WR")
    rep = POS_LG.get(pos3, 0.0) + mpP["rep_delta"]
    st3 = rating2.get(pid)
    q = rep if (not st3 or st3[1] <= 0) else (st3[0] + mpP["prior_opp"] * rep) / (st3[1] + mpP["prior_opp"])
    u = usage2.get(pid)
    upg = (u[0] / u[1]) if u and u[1] > 0 else 0.0
    return (q - rep) * upg
for i, g in enumerate(games):
    s2, w2 = g["season"], g["week"]
    def tmiss(team):
        cands = [(usage2[pid][0] / usage2[pid][1], pid) for pid in team_roster2.get(team, ())
                 if pid in usage2 and usage2[pid][1] > 0]
        cands.sort(reverse=True)
        return sum(pval(pid) for _, pid in cands[:int(mpP["K"])]
                   if pid not in played.get((team, s2, w2), ()))
    f_mp[i] = tmiss(g["home"]) - tmiss(g["away"])
    for team in (g["home"], g["away"]):
        for pid in played.get((team, s2, w2), ()):
            epa2, opp2, t2, pos3 = pw[(pid, s2, w2)]
            st3 = rating2.setdefault(pid, [0.0, 0.0])
            st3[0] = mpP["decay"] * st3[0] + epa2
            st3[1] = mpP["decay"] * st3[1] + opp2
            u = usage2.setdefault(pid, [0.0, 0.0])
            u[0] = mpP["usage_decay"] * u[0] + opp2
            u[1] = mpP["usage_decay"] * u[1] + 1.0
            if team_of2.get(pid) != team:
                if team_of2.get(pid) in team_roster2:
                    team_roster2[team_of2[pid]].discard(pid)
                team_of2[pid] = team
                team_roster2[team].add(pid)
            pos_of2[pid] = pos3

# ---------- 5. coach rating (kappa irrelevant; decay .993 prior 29) ----------
for g, r in zip(games, raw):
    g["hc"], g["ac"] = r["home_coach"].strip(), r["away_coach"].strip()
crat = {}
f_co = np.zeros(len(games))
for i, g in enumerate(games):
    rh = crat.get(g["hc"], (0.0, 0.0)); ra = crat.get(g["ac"], (0.0, 0.0))
    f_co[i] = rh[0] / (rh[1] + 29.0) - ra[0] / (ra[1] + 29.0)
    resid = g["y"] - pe[i]
    for coach, sgn in ((g["hc"], 1.0), (g["ac"], -1.0)):
        s3, n3 = crat.get(coach, (0.0, 0.0))
        crat[coach] = (0.993 * s3 + sgn * resid, 0.993 * n3 + 1.0)

# ---------- 6. macro TS add (tuned add-config) ----------
import sys as _s
_s.path.insert(0, ".")
from mlbwp.trueskill import TrueSkill1v1  # noqa: E402
tsm = TrueSkill1v1(beta=2.029, tau=0.397, mu0=25.0, sig0=25.0 / 3.0)
f_mts = np.zeros(len(games)); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for k2 in list(tsm.mu.keys()):
            tsm.mu[k2] = 25.0 + (tsm.mu[k2] - 25.0) * (1 - 0.078)
            tsm.var[k2] = min(tsm.var[k2] + 2.376 ** 2, (25.0 / 3.0) ** 2)
    prev = g["season"]
    f_mts[i] = tsm.mu[g["home"]] - tsm.mu[g["away"]]
    if g["y"] != 0.5:
        w2, l2 = (g["home"], g["away"]) if g["y"] == 1 else (g["away"], g["home"])
        tsm.update(w2, l2)

# ---------- assemble + ONE TEST look ----------
def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

EX = np.column_stack([f_srs, f_cl, f_qsh, f_mp, f_co, f_mts])
X_mega = np.column_stack([X_base, EX])
print(f"DEV CV: current {cv_ll(X_base):.5f}  reunion {cv_ll(X_mega):.5f}")
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_base[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(X_mega[fitm], y[fitm])
p0 = c0.predict_proba(X_base[test])[:, 1]
p1 = c1.predict_proba(X_mega[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"main TEST: current {llv(yt, p0).mean():.5f} -> reunion {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_reunion.json", "w"), indent=1)
