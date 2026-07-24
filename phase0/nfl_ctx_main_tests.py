"""Context-audit Phase B: main-model reference-frame fixes, DEV screens.

  B2  absence ghost rosters  expire roster membership (20 team-games + boundary
                             purge) so 'missing' means missing, not retired
  B9  run/pass/epa anchors   running league means (EWMA ~2.6-season half-life)
                             instead of frozen 2001-15 constants
  B10 fumble keep-rate       running p-hat instead of the hard-coded 0.5 split
  B7  home coordination      is_home blend column + thfa centered on the
                             league-wide home-residual EWMA
Gate -0.0020 on DEV 5-fold CV; gate-clearers earn a TEST look.
"""
from __future__ import annotations

import csv
import json
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

def passrun(run_anchor="fixed", lam=0.999):
    offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
    dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
    LPs, LPn, LRs, LRn = LGP, 1.0, LGR, 1.0      # running anchors seeded at DEV means
    ws_p = ws_r = 1.0
    f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in (list(offP.values()) + list(offR.values())
                        + list(dfaP.values()) + list(dfaR.values())):
                st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
        prev = g["season"]
        lp = LPs / ws_p if run_anchor == "running" else LGP
        lr = LRs / ws_r if run_anchor == "running" else LGR
        def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
        h, a = g["home"], g["away"]
        f_pass[i] = (crate(offP[h], lp) - crate(dfaP[h], lp)) - (crate(offP[a], lp) - crate(dfaP[a], lp))
        f_run[i] = (crate(offR[h], lr) - crate(dfaR[h], lr)) - (crate(offR[a], lr) - crate(dfaR[a], lr))
        tm = aggc.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
                o = offP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
                o = offR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
                d2 = dfaP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
                d2 = dfaR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN
                if run_anchor == "running" and pN > 0:
                    LPs = lam * LPs + (pS / pN); ws_p = lam * ws_p + 1.0
                if run_anchor == "running" and rN > 0:
                    LRs = lam * LRs + (rS / rN); ws_r = lam * ws_r + 1.0
    return f_pass, f_run

f_pass0, f_run0 = passrun("fixed")
X14 = np.column_stack([X_of(F), f_pass0, f_run0])
X14[:, 7] = np.load("data/nfl_v7_feature.npy")
cur_cv = cv_ll(X14)
print(f"[{time.time()-T0:.0f}s] current model DEV CV {cur_cv:.5f}", flush=True)

# ---------- B2: ghost-roster absence (lean walk, no ratings needed) ----------
def absence(sn_decay, sn_thresh, expire):
    share2, pos2, team2 = {}, {}, {}
    roster2 = defaultdict(set)
    last_tg = {}
    tg = defaultdict(int)
    fol = np.zeros(len(games)); fde = np.zeros(len(games)); fsk = np.zeros(len(games))
    for i, g in enumerate(games):
        s = g["season"]
        if s >= 2013:
            for side, team in ((1, g["home"]), (-1, g["away"])):
                tbl = snaps.get((g["gid"], team))
                if tbl is None:
                    continue
                m = [0.0, 0.0, 0.0]
                for pid in roster2.get(team, ()):
                    if expire and tg[team] - last_tg.get(pid, -99) > expire:
                        continue                       # ghost: long gone, not "missing"
                    st = share2.get(pid)
                    if not st or st[1] <= 0:
                        continue
                    sh = st[0] / st[1]
                    if sh < sn_thresh or pid in tbl:
                        continue
                    p = pos2.get(pid, "")
                    if p in OLPOS:
                        m[0] += sh
                    elif p in DEFPOS:
                        m[1] += sh
                    elif p in SKPOS:
                        m[2] += sh
                fol[i] += side * m[0]; fde[i] += side * m[1]; fsk[i] += side * m[2]
        for team in (g["home"], g["away"]):
            tg[team] += 1
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEFPOS else op
                st = share2.setdefault(pid, [0.0, 0.0])
                st[0] = sn_decay * st[0] + pct
                st[1] = sn_decay * st[1] + 1.0
                pos2[pid] = pos
                last_tg[pid] = tg[team]
                if team2.get(pid) != team:
                    if team2.get(pid) in roster2:
                        roster2[team2.get(pid)].discard(pid)
                    team2[pid] = team
                    roster2[team].add(pid)
    return fol, fde, fsk

print("\nB2 ghost-roster absence (DEV CV, gate -0.0020):", flush=True)
best_b2 = (0.0, None, None)
for cfg in ((snapP["decay"], snapP["thresh"], 20), (snapP["decay"], snapP["thresh"], 12),
            (0.90, snapP["thresh"], 20), (snapP["decay"], 0.60, 20)):
    fol, fde, fsk = absence(*cfg)
    X = X14.copy(); X[:, 8] = fol; X[:, 9] = fde; X[:, 10] = fsk
    s = cv_ll(X)
    d = cur_cv - s
    print(f"  decay {cfg[0]} thresh {cfg[1]} expire {cfg[2]}: CV {s:.5f}  ({d:+.5f})", flush=True)
    if d > best_b2[0]:
        best_b2 = (d, cfg, (fol, fde, fsk))
print(f"  -> best {best_b2[1]}: {best_b2[0]:+.5f} "
      f"{'GATE MET' if best_b2[0] >= 0.0020 else 'below gate'}", flush=True)

# ---------- B9: running league anchors (pass/run/epa) ----------
f_pass_r, f_run_r = passrun("running")
X = X14.copy(); X[:, 12] = f_pass_r; X[:, 13] = f_run_r
s_b9 = cv_ll(X)
print(f"\nB9 running run/pass anchors: CV {s_b9:.5f}  ({cur_cv-s_b9:+.5f})", flush=True)

# ---------- B10: fumble running keep-rate ----------
def luck_running(lam=0.999):
    luck = defaultdict(lambda: [0.0, 0.0])
    KS, TS_ = 0.513, 1.0
    d_l, pn_l, sd_l = luckP["decay"], luckP["prior_n"], luckP["season_decay"]
    f = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st in luck.values():
                st[0] *= (1 - sd_l); st[1] *= (1 - sd_l)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = luck[h][0] / (luck[h][1] + pn_l) - luck[a][0] / (luck[a][1] + pn_l)
        gt = tv.get(g["gid"])
        if gt and h in gt and a in gt:
            p_hat = KS / TS_
            _, fh, flh = gt[h]; _, fa, fla = gt[a]
            tot = fh + fa
            rec_h = (fh - flh) + fla
            exp_h = p_hat * fh + (1 - p_hat) * fa
            for team, rec, ex in ((h, rec_h, exp_h), (a, tot - rec_h, tot - exp_h)):
                s_ = luck[team]
                s_[0] = d_l * s_[0] + (rec - ex)
                s_[1] = d_l * s_[1] + tot
            if tot > 0:
                KS = lam * KS + ((fh - flh) + (fa - fla)) / tot; TS_ = lam * TS_ + 1.0
    return f

f_luck_r = luck_running()
X = X14.copy(); X[:, 6] = f_luck_r
s_b10 = cv_ll(X)
print(f"B10 fumble running keep-rate: CV {s_b10:.5f}  ({cur_cv-s_b10:+.5f})", flush=True)

# ---------- B7: is_home column + thfa league-centered ----------
def thfa_centered():
    st = {}
    ls0, ls1 = 0.0, 0.0
    f = np.zeros(len(games))
    for i, g in enumerate(games):
        if not g["neutral"]:
            lg_r = ls0 / (ls1 + 180.0)
            s = st.get(g["home"], (0.0, 0.0))
            f[i] = s[0] / (s[1] + 180.0)
            resid = (g["y"] - pe[i]) - lg_r
            st[g["home"]] = (0.991 * s[0] + resid, 0.991 * s[1] + 1.0)
            ls0 = 0.991 * ls0 + (g["y"] - pe[i]); ls1 = 0.991 * ls1 + 1.0
    return f

f_home = np.array([0.0 if g["neutral"] else 1.0 for g in games])
f_thfa_c = thfa_centered()
X = np.column_stack([X14, f_home])
s_b7a = cv_ll(X)
X2 = X14.copy(); X2[:, 4] = f_thfa_c
X2 = np.column_stack([X2, f_home])
s_b7b = cv_ll(X2)
print(f"B7 is_home column only: CV {s_b7a:.5f}  ({cur_cv-s_b7a:+.5f})", flush=True)
print(f"B7 is_home + centered thfa: CV {s_b7b:.5f}  ({cur_cv-s_b7b:+.5f})", flush=True)

json.dump({"cur": round(cur_cv, 5), "b2": round(best_b2[0], 5),
           "b9": round(cur_cv - s_b9, 5), "b10": round(cur_cv - s_b10, 5),
           "b7a": round(cur_cv - s_b7a, 5), "b7b": round(cur_cv - s_b7b, 5)},
          open("data/nfl_ctx_main.json", "w"), indent=1)
print("wrote data/nfl_ctx_main.json")
