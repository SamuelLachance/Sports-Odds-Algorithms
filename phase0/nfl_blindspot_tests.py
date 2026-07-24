"""Last untested channels on the MAIN model (current 14 feats, v6 @ col 7).

  T1 kicker skill    FG points-over-expected-by-distance EWMA, as-of, per team
  T3 weekly refits   walk-forward retrained per WEEK instead of per season
  T4 whispers        divisional flag, rematch count, wind x pass-strength

Discipline: T1/T4 are feature screens -> DEV 5-fold CV, gate -0.0020, ONE TEST
look only if the gate is met. T3 is a training procedure -> single TEST report
(row-47 precedent). Penalties (T2) run separately once the 1999-2025 pull lands.
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

# ================= T1: kicker skill (FG points over expected) =================
kicks = defaultdict(lambda: defaultdict(list))
dev_d, dev_m = [], []
for r in csv.DictReader(open("data/nfl_fgs.csv")):
    t_ = PBP_FIX.get(r["posteam"], r["posteam"])
    d_ = float(r["kick_distance"]); m_ = int(r["made"])
    kicks[r["game_id"]][t_].append((d_, m_))
    if int(r["game_id"][:4]) in DEV_YEARS:
        dev_d.append(d_); dev_m.append(m_)
mk = LogisticRegression(C=1e6, max_iter=2000).fit(
    np.array(dev_d).reshape(-1, 1), np.array(dev_m))
def p_make(d_): return float(mk.predict_proba([[d_]])[0, 1])
PMAKE = {d_: p_make(d_) for d_ in range(15, 70)}

def kicker_feature(decay, prior_n, season_decay):
    st = defaultdict(lambda: [0.0, 0.0])           # [poe pts, attempts]
    f = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for s_ in st.values():
                s_[0] *= (1 - season_decay); s_[1] *= (1 - season_decay)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = st[h][0] / (st[h][1] + prior_n) - st[a][0] / (st[a][1] + prior_n)
        gk = kicks.get(g["gid"])
        if gk:
            for t_, lst in gk.items():
                s_ = st[t_]
                for d_, m_ in lst:
                    s_[0] = decay * s_[0] + 3.0 * (m_ - PMAKE.get(int(d_), p_make(d_)))
                    s_[1] = decay * s_[1] + 1.0
    return f

print("\nT1 kicker skill (DEV CV, gate -0.0020):", flush=True)
best_t1 = (0.0, None)
for cfg in ((0.95, 10.0, 0.3), (0.90, 20.0, 0.3), (0.98, 10.0, 0.5)):
    fk = kicker_feature(*cfg)
    s = cv_ll(np.column_stack([X14, fk]))
    d = cur_cv - s
    print(f"  decay {cfg[0]} prior {cfg[1]:.0f} sd {cfg[2]}: CV {s:.5f}  ({d:+.5f})", flush=True)
    if d > best_t1[0]:
        best_t1 = (d, cfg)
print(f"  -> best {best_t1[1]} improvement {best_t1[0]:+.5f} "
      f"{'GATE MET' if best_t1[0] >= 0.0020 else 'below gate - no TEST look'}", flush=True)

# ================= T4: whispers =================
DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
div_of = {t: d for d, ts in DIVS.items() for t in ts}
wind_of = {}
for r in csv.DictReader(open("data/nfl_games.csv")):
    if r["home_score"] != "" and r.get("wind"):
        try:
            wind_of[r["game_id"]] = float(r["wind"])
        except ValueError:
            pass
f_div = np.zeros(len(games)); f_rem = np.zeros(len(games)); f_wnd = np.zeros(len(games))
met = defaultdict(int); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        met.clear()
    prev = g["season"]
    if g["season"] >= 2002 and div_of.get(g["home"]) == div_of.get(g["away"]):
        f_div[i] = 1.0
    pair = tuple(sorted((g["home"], g["away"])))
    f_rem[i] = float(met[pair])
    met[pair] += 1
    f_wnd[i] = (wind_of.get(g["gid"], 0.0) / 10.0) * f_pass[i]
print("\nT4 whispers (DEV CV, gate -0.0020):", flush=True)
for nm, col in (("divisional flag", f_div), ("rematch count", f_rem),
                ("wind x pass-strength", f_wnd)):
    s = cv_ll(np.column_stack([X14, col]))
    print(f"  {nm:<22} CV {s:.5f}  ({cur_cv-s:+.5f})", flush=True)
s = cv_ll(np.column_stack([X14, f_div, f_rem, f_wnd]))
print(f"  {'bundle (all three)':<22} CV {s:.5f}  ({cur_cv-s:+.5f})", flush=True)

# ================= T3: weekly in-season refits (TEST report) =================
HL, BC = 3.0, 100.0
def wf_yearly(X):
    p = np.zeros(int(test.sum()))
    pos_ = {gi: j for j, gi in enumerate(np.where(test)[0])}
    for s_ in sorted(np.unique(seasons[test])):
        tr = (seasons < s_) & (y != 0.5)
        te = test & (seasons == s_)
        w_ = 0.5 ** ((s_ - 1 - seasons[tr]) / HL)
        m_ = LogisticRegression(C=BC, max_iter=5000).fit(X[tr], y[tr], sample_weight=w_)
        pp = m_.predict_proba(X[te])[:, 1]
        for k2, gi in enumerate(np.where(te)[0]):
            p[pos_[gi]] = pp[k2]
    return p

weeks_arr = np.array([g["week"] for g in games])
def wf_weekly(X):
    p = np.zeros(int(test.sum()))
    pos_ = {gi: j for j, gi in enumerate(np.where(test)[0])}
    for s_ in sorted(np.unique(seasons[test])):
        for w_k in sorted(np.unique(weeks_arr[test & (seasons == s_)])):
            tr = ((seasons < s_) | ((seasons == s_) & (weeks_arr < w_k))) & (y != 0.5)
            te = test & (seasons == s_) & (weeks_arr == w_k)
            w_ = 0.5 ** ((s_ - 1 - seasons[tr]) / HL)
            m_ = LogisticRegression(C=BC, max_iter=5000).fit(X[tr], y[tr], sample_weight=w_)
            pp = m_.predict_proba(X[te])[:, 1]
            for k2, gi in enumerate(np.where(te)[0]):
                p[pos_[gi]] = pp[k2]
    return p

yt = y[test]
p_y = wf_yearly(X14)
p_w = wf_weekly(X14)
ll_y = float(llv(yt, p_y).mean()); ll_w = float(llv(yt, p_w).mean())
d = llv(yt, p_y) - llv(yt, p_w)
rng = np.random.default_rng(7)
bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nT3 weekly refits (main TEST, n={len(yt)}):")
print(f"  yearly refit (current)   LL {ll_y:.5f}")
print(f"  weekly refit             LL {ll_w:.5f}  (delta {float(d.mean()):+.5f}  CI [{lo:+.5f}, {hi:+.5f}])")
print(f"  -> {'ADOPT weekly' if ll_w < ll_y else 'keep yearly'}")
json.dump({"t1_best": {"cfg": best_t1[1], "dev_gain": round(best_t1[0], 5)},
           "t3": {"yearly": round(ll_y, 5), "weekly": round(ll_w, 5),
                  "delta": round(float(d.mean()), 5),
                  "ci": [round(float(lo), 5), round(float(hi), 5)]}},
          open("data/nfl_blindspots.json", "w"), indent=1, default=list)
print("wrote data/nfl_blindspots.json")
