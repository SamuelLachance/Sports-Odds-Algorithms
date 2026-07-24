"""Special-teams unit EPA rating as a 15th feature — DEV screen.

Punt/kickoff (+ FG/XP in variant 2) per-game team EPA aggregated from nflverse
pbp (data/nfl_st_games.csv, built by phase0/nfl_st_pull.py). Per-team EWMA
states with the SAME dynamics family as the shipped EPA channel
(decay=epaP["decay"], prior_n=epaP["prior_n"]/2, season_decay=
epaP["season_decay"], league rate = DEV-years mean ST epa/play). As-of walk:
feature is computed from PRE-game states, then states update with the game.

DEV 5-fold CV screen (2001-2015), gate delta >= 0.0020. NO TEST look.
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
assert abs(cur_cv - 0.62300) < 0.0007, f"repro check FAILED: {cur_cv:.5f}"

# ---- load ST per-game per-team aggregates ----
# stg[gid][team] = [pk_s, pk_n, pk_ag_s, pk_ag_n, fx_s, fx_n, fx_ag_s, fx_ag_n]
stg = defaultdict(dict)
with open("data/nfl_st_games.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); next(rd)
    for r in rd:
        t_ = PBP_FIX.get(r[1], r[1])
        stg[r[0]][t_] = [float(r[2]), int(r[3]), float(r[4]), int(r[5]),
                         float(r[6]), int(r[7]), float(r[8]), int(r[9])]
print(f"[{time.time()-T0:.0f}s] ST agg loaded: {len(stg)} games", flush=True)

# league rates over DEV years (epa per play)
s_pk = n_pk = s_all = n_all = 0.0
for gid, tms in stg.items():
    if int(gid[:4]) in DEV_YEARS:
        for a in tms.values():
            s_pk += a[0]; n_pk += a[1]
            s_all += a[0] + a[4]; n_all += a[1] + a[5]
LG_PK = s_pk / n_pk
LG_ALL = s_all / n_all
print(f"league ST epa/play  pk={LG_PK:+.4f}  pk+fx={LG_ALL:+.4f}", flush=True)


def build_feat(use_fx: bool) -> np.ndarray:
    lg = LG_ALL if use_fx else LG_PK
    fo = defaultdict(lambda: [0.0, 0.0])   # ST epa for (as posteam)
    ag = defaultdict(lambda: [0.0, 0.0])   # ST epa allowed (as defteam)
    feat = np.zeros(len(games)); prev = None

    def crate2(st_):
        return (st_[0] + (pn_ / 2) * lg) / (st_[1] + pn_ / 2)

    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in list(fo.values()) + list(ag.values()):
                st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
        prev = g["season"]
        h, a = g["home"], g["away"]
        feat[i] = ((crate2(fo[h]) - crate2(ag[h]))
                   - (crate2(fo[a]) - crate2(ag[a])))
        tms = stg.get(g["gid"])
        if tms:
            for t_ in (h, a):
                r = tms.get(t_)
                if r is None:
                    continue
                fS, fN = r[0], r[1]
                aS, aN = r[2], r[3]
                if use_fx:
                    fS += r[4]; fN += r[5]
                    aS += r[6]; aN += r[7]
                o = fo[t_]; o[0] = dec_ * o[0] + fS; o[1] = dec_ * o[1] + fN
                d2 = ag[t_]; d2[0] = dec_ * d2[0] + aS; d2[1] = dec_ * d2[1] + aN
    return feat


# ---- coverage diagnostics ----
cov_all = sum(1 for g in games if g["gid"] in stg)
devi = np.where(dev)[0]
cov_dev = sum(1 for i in devi if games[i]["gid"] in stg)
print(f"coverage: {cov_all}/{len(games)} all games, {cov_dev}/{len(devi)} dev games",
      flush=True)

out = {"base": round(cur_cv, 5), "coverage_all": cov_all, "n_games": len(games),
       "coverage_dev": cov_dev, "n_dev": int(len(devi)),
       "league_rate_pk": round(LG_PK, 5), "league_rate_pkfx": round(LG_ALL, 5),
       "variants": {}}

print("\nST unit-EPA 15th feature (DEV 5-fold, gate delta >= 0.0020):", flush=True)
for name, use_fx in (("punt_kick", False), ("punt_kick_fg_xp", True)):
    f_st = build_feat(use_fx)
    X15 = np.column_stack([X14, f_st])
    s = cv_ll(X15)
    d = cur_cv - s
    corr_epa = float(np.corrcoef(f_st[devi], X14[devi, 2])[0, 1])
    fitm = devi[y[devi] != 0.5]
    clf = LogisticRegression(C=1e6, max_iter=5000).fit(X15[fitm], y[fitm])
    coef = float(clf.coef_[0][-1])
    out["variants"][name] = {
        "cv": round(s, 5), "delta": round(d, 5),
        "corr_with_epa_net": round(corr_epa, 4),
        "coef_full_dev": round(coef, 4),
        "feat_sd_dev": round(float(f_st[devi].std()), 5),
    }
    print(f"  {name:16s} CV {s:.5f}  delta {d:+.5f}  "
          f"corr(EPAnet) {corr_epa:+.3f}  coef {coef:+.3f}  "
          f"sd {f_st[devi].std():.4f}", flush=True)

json.dump(out, open("data/nfl_st_screen.json", "w"), indent=1)
print(f"\n[{time.time()-T0:.0f}s] wrote data/nfl_st_screen.json", flush=True)
