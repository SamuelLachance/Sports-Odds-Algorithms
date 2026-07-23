"""TEST: the 11v11 participation TrueSkill as the MODEL's player rating.

Samuel's ask: replace the ratings inside the model (roster_quality's weekly
composite rate6) with the new per-play TrueSkill engine, retrain, test.

Construction (leak-free by design — feature BEFORE the game's plays update):
  walk games chronologically; for each game
    1. rq_ts = share-weighted sum over gameday actives of clamp(mu_p - 25, +-3),
       home minus away  (identical population/weights to the shipped rq walk;
       only the player value changes: TrueSkill mu instead of rate6 z)
    2. THEN update TrueSkill from that game's 11v11 plays (same engine params
       as the ratings build: beta=sig0, weekly tau, garbage down-weight,
       offseason shrink)
    3. THEN update the snap-share walk (same as shipped)
  zero before 2016 (no participation data — same convention as snap absence
  pre-2013).

Variants, each retrained with the adopted method (walk-forward + recency
hl=3, C=100) and scored on the locked TEST 2016-2025:
  A REPLACE  X14 with rq -> rq_ts
  B ADD      X15 = X14 + rq_ts
Paired bootstrap vs the current model. Adopt per the point-estimate rule.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from math import erf, exp, pi, sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

# ---- pass/run channels (identical to training_eval / serve) ----
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
X14 = np.column_stack([X_of(F), f_pass, f_run])
print(f"[{time.time()-T0:.0f}s] X14 built", flush=True)

# ---- 11v11 TrueSkill engine (identical params to phase0/nfl_trueskill_players.py) ----
MU0, SIG0 = 25.0, 25.0 / 3.0
SIG0_2 = SIG0 * SIG0
BETA = SIG0
TAU2 = 0.10 ** 2
SEASON_SHRINK = 1.0 / 3.0
SEASON_WIDEN2 = 1.5 ** 2
SIG2_FLOOR = 0.5 ** 2
GARBAGE_W = 0.25
V_CLAMP = 3.0

def pdf(x): return exp(-0.5 * x * x) / sqrt(2 * pi)
def cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
def v_win(t):
    c = cdf(t)
    return -t if c < 1e-12 else pdf(t) / c

wp_of = {}
with open("data/nfl_play_ctx.csv") as fh:
    rd = csv.reader(fh); next(rd)
    for gid, pid_, wp in rd:
        wp_of[(gid, pid_)] = float(wp)
plays_by_gid = defaultdict(list)
with open("data/nfl_rapm_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); next(rd)
    for gid, pid_, off, dfn, epa in rd:
        plays_by_gid[gid].append((int(float(pid_)), off, dfn, float(epa)))
for v in plays_by_gid.values():
    v.sort()
print(f"[{time.time()-T0:.0f}s] plays loaded for {len(plays_by_gid)} games", flush=True)

ts_mu, ts_s2, ts_lastwk = {}, {}, {}

def ts_update_game(gid):
    season, week = gid[:4], gid[5:7]
    wk = (season, week)
    for pid_, off, dfn, epa in plays_by_gid.get(gid, ()):
        O = off.split(";"); D = dfn.split(";")
        if not (6 <= len(O) <= 12 and 6 <= len(D) <= 12):
            continue
        for p in O + D:
            if p not in ts_mu:
                ts_mu[p] = MU0; ts_s2[p] = SIG0_2
            elif ts_lastwk.get(p) != wk:
                ts_s2[p] = min(ts_s2[p] + TAU2, SIG0_2)
            ts_lastwk[p] = wk
        wgt = 1.0
        wp = wp_of.get((gid, str(pid_))) or wp_of.get((gid, f"{pid_}.0"))
        if wp is not None and (wp < 0.05 or wp > 0.95):
            wgt = GARBAGE_W
        c2 = (len(O) + len(D)) * BETA * BETA + sum(ts_s2[p] for p in O) + sum(ts_s2[p] for p in D)
        c = sqrt(c2)
        t = (sum(ts_mu[p] for p in O) - sum(ts_mu[p] for p in D)) / c
        if epa > 0:
            win, lose, tt = O, D, t
        else:
            win, lose, tt = D, O, -t
        v = v_win(tt); w = v * (v + tt)
        for p in win:
            ts_mu[p] += wgt * (ts_s2[p] / c) * v
            ts_s2[p] = max(ts_s2[p] * (1.0 - wgt * (ts_s2[p] / c2) * w), SIG2_FLOOR)
        for p in lose:
            ts_mu[p] -= wgt * (ts_s2[p] / c) * v
            ts_s2[p] = max(ts_s2[p] * (1.0 - wgt * (ts_s2[p] / c2) * w), SIG2_FLOOR)

def ts_season_boundary():
    for p in ts_mu:
        ts_mu[p] = MU0 + (ts_mu[p] - MU0) * (1.0 - SEASON_SHRINK)
        ts_s2[p] = min(ts_s2[p] + SEASON_WIDEN2, SIG0_2)

# ---- the walk: feature (pre-game) -> TS play updates -> share updates ----
share2, pos2, team2 = {}, {}, {}
roster2 = defaultdict(set)
rq_ts = np.zeros(len(games))
prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev and g["season"] >= 2016:
        ts_season_boundary()
    prev = g["season"]
    if g["season"] >= 2016:
        for side, team in ((1, g["home"]), (-1, g["away"])):
            tbl = snaps.get((g["gid"], team))
            if tbl is None:
                continue
            tot = 0.0
            for pid in tbl:
                st = share2.get(pid)
                if not st or st[1] <= 0:
                    continue
                g_ = pfr2gsis.get(pid)
                mu_ = ts_mu.get(g_) if g_ else None
                if mu_ is not None:
                    tot += (st[0] / st[1]) * max(-V_CLAMP, min(V_CLAMP, mu_ - MU0))
            rq_ts[i] += side * tot
        ts_update_game(g["gid"])
    for team in (g["home"], g["away"]):
        tbl = snaps.get((g["gid"], team))
        if not tbl:
            continue
        for pid, (pos, op, dp) in tbl.items():
            pct = dp if pos in DEFPOS else op
            st = share2.setdefault(pid, [0.0, 0.0])
            st[0] = snapP["decay"] * st[0] + pct
            st[1] = snapP["decay"] * st[1] + 1.0
            pos2[pid] = pos
            if team2.get(pid) != team:
                if team2.get(pid) in roster2:
                    roster2[team2.get(pid)].discard(pid)
                team2[pid] = team
                roster2[team].add(pid)
print(f"[{time.time()-T0:.0f}s] rq_ts walk done | nonzero {int((rq_ts!=0).sum())} "
      f"| corr w/ shipped rq {np.corrcoef(rq_ts[test], X14[test,11])[0,1]:.3f}", flush=True)

# ---- retrain + one TEST look each, adopted training method ----
HL, BC = 3.0, 100.0
def wf_probs(X):
    p_wf = np.zeros(int(test.sum()))
    tidx = np.where(test)[0]
    pos_ = {gi: j for j, gi in enumerate(tidx)}
    for s_ in sorted(np.unique(seasons[test])):
        tr = (seasons < s_) & (y != 0.5)
        te = test & (seasons == s_)
        w_ = 0.5 ** ((s_ - 1 - seasons[tr]) / HL)
        m_ = LogisticRegression(C=BC, max_iter=5000).fit(X[tr], y[tr], sample_weight=w_)
        pp = m_.predict_proba(X[te])[:, 1]
        for k2, gi in enumerate(np.where(te)[0]):
            p_wf[pos_[gi]] = pp[k2]
    return p_wf

X_rep = X14.copy(); X_rep[:, 11] = rq_ts
X_add = np.column_stack([X14, rq_ts])
yt = y[test]
p_base = wf_probs(X14)
p_rep = wf_probs(X_rep)
p_add = wf_probs(X_add)
ll_base = float(llv(yt, p_base).mean())
rng = np.random.default_rng(7)
def paired(p_new):
    d = llv(yt, p_base) - llv(yt, p_new)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
print(f"\nmain TEST (n={len(yt)}), walk-forward + recency (the adopted training):")
print(f"  current model (rate6 composite)   LL {ll_base:.5f}")
rows = {}
for tag, pp in (("A REPLACE rq -> 11v11 TS", p_rep), ("B ADD rq_ts as 15th", p_add)):
    ll_ = float(llv(yt, pp).mean())
    d, lo, hi = paired(pp)
    rows[tag] = {"ll": round(ll_, 5), "delta": round(d, 5),
                 "ci": [round(lo, 5), round(hi, 5)]}
    print(f"  {tag:<33} LL {ll_:.5f}  (delta {d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}])")
best = min(rows.values(), key=lambda r: r["ll"])
verdict = "ADOPT" if best["ll"] < ll_base else "keep current"
print(f"\n-> {verdict}")
json.dump({"base_ll": round(ll_base, 5), "variants": rows, "verdict": verdict,
           "spec": "player value = clamp(mu-25,+-3), share-weighted actives, zero pre-2016"},
          open("data/nfl_ts_model_test.json", "w"), indent=1)
