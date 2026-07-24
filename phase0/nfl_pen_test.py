"""T2: penalty-discipline EWMA as a MAIN-model feature. DEV screen, gate -0.0020."""
from __future__ import annotations
import csv, json, time
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])
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
X14[:, 7] = np.load("data/nfl_v6_feature.npy")
cur_cv = cv_ll(X14)
print(f"[{time.time()-T0:.0f}s] current model DEV CV {cur_cv:.5f}", flush=True)

pens = defaultdict(dict)
for r in csv.DictReader(open("data/nfl_penalties.csv")):
    t_ = PBP_FIX.get(r["penalty_team"], r["penalty_team"])
    pens[r["game_id"]][t_] = (float(r["n"]), float(r["yds"]))

def pen_feature(decay, prior_g, season_decay, use_yds):
    st = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for s_ in st.values():
                s_[0] *= (1 - season_decay); s_[1] *= (1 - season_decay)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = st[h][0] / (st[h][1] + prior_g) - st[a][0] / (st[a][1] + prior_g)
        gp = pens.get(g["gid"])
        if gp is not None:
            for t_ in (h, a):
                n_, y_ = gp.get(t_, (0.0, 0.0))
                s_ = st[t_]
                s_[0] = decay * s_[0] + (y_ if use_yds else n_)
                s_[1] = decay * s_[1] + 1.0
    return f

print("\nT2 penalty discipline (DEV CV, gate -0.0020):", flush=True)
best = (0.0, None)
for cfg in ((0.95, 6.0, 0.3, True), (0.90, 10.0, 0.3, True),
            (0.95, 6.0, 0.3, False), (0.98, 6.0, 0.5, True)):
    fp_ = pen_feature(*cfg)
    s = cv_ll(np.column_stack([X14, fp_]))
    d = cur_cv - s
    tag = "yds" if cfg[3] else "count"
    print(f"  decay {cfg[0]} prior {cfg[1]:.0f} sd {cfg[2]} ({tag}): CV {s:.5f}  ({d:+.5f})", flush=True)
    if d > best[0]:
        best = (d, cfg)
print(f"  -> best {best[1]} improvement {best[0]:+.5f} "
      f"{'GATE MET' if best[0] >= 0.0020 else 'below gate - no TEST look'}", flush=True)
json.dump({"best_gain": round(best[0], 5), "cfg": list(best[1]) if best[1] else None},
          open("data/nfl_pen_test.json", "w"), indent=1)
