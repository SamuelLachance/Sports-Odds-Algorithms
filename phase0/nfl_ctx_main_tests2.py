"""Context-audit Phase B, part 2: the two heavy main-model reference-frame fixes.

  B5 QB running league anchor   Q states accumulate EXCESS EPA/db vs a running
                                league rate (EWMA, DEV-seeded) instead of the
                                frozen 2001-15 lg; pedigree priors in excess space
  B6 rate6 running z-centers    weekly composites centered on as-of season-lagged
                                metric means (3-season EWMA, DEV-seeded) instead
                                of frozen 2001-15 Z means; DEV sd kept
Gate -0.0020 on DEV 5-fold CV.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

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
X14[:, 7] = np.load("data/nfl_v7_feature.npy")
cur_cv = cv_ll(X14)
print(f"[{time.time()-T0:.0f}s] current model DEV CV {cur_cv:.5f}", flush=True)

# ================= B5: QB running league anchor =================
rep_x = {qid: pedP["delta"] * (1.0 - pedP["c"] * math.exp(-(pk - 1) / pedP["scale"]))
         for qid, pk in picks.items()}
def qd_running(lam=0.99975):
    Q = {}
    LS, LN = lg_qb, 1.0                    # running league EPA/db, DEV-seeded
    out = np.zeros(len(games)); prev = None
    def adj(qid):
        r_ = rep_x.get(qid, pedP["delta"])
        s = Q.get(qid)
        if not s or s[1] <= 0:
            return r_
        return (s[0] + SP["prior_db"] * r_) / (s[1] + SP["prior_db"])
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for s in Q.values():
                s[0] *= (1.0 - SP["season_decay"]); s[1] *= (1.0 - SP["season_decay"])
        prev = g["season"]
        out[i] = adj(g["home_qb"]) - adj(g["away_qb"])
        lg_t = LS / LN
        for qid in (g["home_qb"], g["away_qb"]):
            line = qbw.get((qid, g["season"], g["week"]))
            if line is None:
                continue
            s = Q.setdefault(qid, [0.0, 0.0])
            s[0] = SP["decay"] * s[0] + (line[0] - lg_t * line[1])   # EXCESS epa
            s[1] = SP["decay"] * s[1] + line[1]
            LS = lam * LS + line[0] / max(line[1], 1.0) if line[1] >= 10 else LS
            LN = lam * LN + 1.0 if line[1] >= 10 else LN
    return out

print("\nB5 QB running league anchor (DEV CV, gate -0.0020):", flush=True)
for lam in (0.99975, 0.999):
    qd_r = qd_running(lam)
    X = X14.copy(); X[:, 1] = qd_r
    s = cv_ll(X)
    print(f"  lam {lam}: CV {s:.5f}  ({cur_cv-s:+.5f})", flush=True)

# ================= B6: rate6 running z-centers =================
# per-season per-(bucket, metric) means from the weekly tables
MS = defaultdict(lambda: [0.0, 0.0])       # (b, k, season) -> [val_sum, w_sum]
for (pid, s, w), o in OFFW.items():
    b = bucket(o["pos"])
    if b == "QB":
        db = o["att"] + o["sk"]
        if db > 0:
            MS[("QB", "dak", s)][0] += o["dak"] * db; MS[("QB", "dak", s)][1] += db
            MS[("QB", "negsk", s)][0] += -o["sk"]; MS[("QB", "negsk", s)][1] += db
        MS[("QB", "repa", s)][0] += o["repa"]; MS[("QB", "repa", s)][1] += 1.0
    else:
        opp = o["car"] + o["tgt"]
        if opp > 0:
            MS[("SK", "eff", s)][0] += o["repa"] + o["cepa"]; MS[("SK", "eff", s)][1] += opp
        if o["rec"] > 0:
            MS[("SK", "yacr", s)][0] += o["yac"]; MS[("SK", "yacr", s)][1] += o["rec"]
        MS[("SK", "wopr", s)][0] += o["wopr"]; MS[("SK", "wopr", s)][1] += 1.0
for (pid, s, w), d in DEFW.items():
    b = bucket(d["pos"])
    MS[(b, "rush", s)][0] += d["sack"] + 0.5 * d["tfl"] + 0.5 * d["hit"]; MS[(b, "rush", s)][1] += 1.0
    MS[(b, "ball", s)][0] += 3.0 * d["int"] + d["pd"]; MS[(b, "ball", s)][1] += 1.0
    MS[(b, "tak", s)][0] += d["solo"]; MS[(b, "tak", s)][1] += 1.0

M_RUN = {}                                  # (b, k, season) -> as-of center (EWMA of PAST seasons)
LAM_S = 0.7937                              # 3-season half-life
for b in Z:
    for k in Z[b]:
        mu0 = Z[b][k][0]
        cs, cw = mu0, 1.0
        for s in range(1999, 2026):
            M_RUN[(b, k, s)] = cs / cw
            m = MS.get((b, k, s))
            if m and m[1] > 0:
                cs = LAM_S * cs + m[0] / m[1]
                cw = LAM_S * cw + 1.0

def rate6_run(pid, season):
    st = STATE.get(pid)
    if st is None:
        return None
    b = st.get("bucket")
    if b not in Z:
        return None
    if st.get(PRIMARY[b] + "_w", 0.0) < 5.0:
        return None
    tot, any_ = 0.0, False
    for k, wt in HANDW[b]:
        w = st.get(k + "_w", 0.0)
        if w <= 1 or k not in Z[b]:
            continue
        mu_c = M_RUN.get((b, k, season), Z[b][k][0])
        sd = Z[b][k][1]
        v = (st.get(k, 0.0) + 6.0 * mu_c) / (w + 6.0)
        tot += wt * (v - mu_c) / sd
        any_ = True
    return tot if any_ else None

def rq_running():
    STATE.clear()
    share2, pos2, team2 = {}, {}, {}
    roster2 = defaultdict(set)
    done2 = set()
    frq = np.zeros(len(games))
    for i, g in enumerate(games):
        s, w = g["season"], g["week"]
        if s >= 2013:
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
                    r_ = rate6_run(g_, s) if g_ else None
                    if r_ is not None:
                        tot += (st[0] / st[1]) * max(-1.5, min(1.5, r_))
                frq[i] += side * tot
        for team in (g["home"], g["away"]):
            for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
                if (pid, s, w) in done2:
                    continue
                done2.add((pid, s, w))
                apply_week(pid, s, w)
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
    return frq

frq_r = rq_running()
X = X14.copy(); X[:, 11] = frq_r
s_b6 = cv_ll(X)
print(f"\nB6 rate6 running z-centers: CV {s_b6:.5f}  ({cur_cv-s_b6:+.5f})", flush=True)
json.dump({"cur": round(cur_cv, 5), "b6": round(cur_cv - s_b6, 5)},
          open("data/nfl_ctx_main2.json", "w"), indent=1)
print("done")
