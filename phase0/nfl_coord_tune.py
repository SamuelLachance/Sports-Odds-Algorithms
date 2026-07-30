"""A+B: coordinate-descent re-tune of every component's knobs vs the CURRENT model,
plus the blend's ridge sweep. DEV-CV gated; ONE TEST look at the end.

Components swept (cheap -> expensive): Elo core, QB feature, per-team HFA, fumble
luck, rest clip, EPA rating, snap absence, TS edge. Rating system keeps its fresh
tune (decay .95, prior 6, clamp 1.5, floor 5). Baseline must reproduce ~0.62315.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

T0 = time.time()
full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # loaders + big_test features (current params)  # noqa: S102

# ---------------- tuned roster_quality (decay .95, prior 6, clamp 1.5, floor 5) ----------------
PRIMARY = {"QB": "dak", "SK": "eff", "DL": "rush", "LB": "rush", "DB": "ball"}
HANDW = {"QB": [("dak", 0.50), ("negsk", 0.25), ("repa", 0.25)],
         "SK": [("eff", 0.40), ("yacr", 0.15), ("wopr", 0.45)],
         "DL": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
         "LB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
         "DB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)]}
DECAY = 0.95
exec("#" + full_src.split("# pass 1")[1].split("# pass 2")[0])  # pass 1 only -> Z  # noqa: S102

def rate6(pid):
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
        mu, sd = Z[b][k]
        v = (st.get(k, 0.0) + 6.0 * mu) / (w + 6.0)
        tot += wt * (v - mu) / sd
        any_ = True
    return tot if any_ else None

def rq_and_absence(sn_decay, sn_thresh):
    """One walk -> (f_rq tuned clamp 1.5, f_ol, f_def, f_sk with given snap params)."""
    STATE.clear()
    share2, pos2, team2 = {}, {}, {}
    roster2 = defaultdict(set)
    done2 = set()
    frq = np.zeros(len(games))
    fol = np.zeros(len(games)); fde = np.zeros(len(games)); fsk = np.zeros(len(games))
    for i, g in enumerate(games):
        s, w = g["season"], g["week"]
        if s >= 2013:
            for side, team in ((1, g["home"]), (-1, g["away"])):
                tbl = snaps.get((g["gid"], team))
                if tbl is None:
                    continue
                m = [0.0, 0.0, 0.0]
                for pid in roster2.get(team, ()):
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
                tot = 0.0
                for pid in tbl:
                    st = share2.get(pid)
                    if not st or st[1] <= 0:
                        continue
                    g_ = pfr2gsis.get(pid)
                    r_ = rate6(g_) if g_ else None
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
                st[0] = sn_decay * st[0] + pct
                st[1] = sn_decay * st[1] + 1.0
                pos2[pid] = pos
                if team2.get(pid) != team:
                    if team2.get(pid) in roster2:
                        roster2[team2.get(pid)].discard(pid)
                    team2[pid] = team
                    roster2[team].add(pid)
    return frq, fol, fde, fsk

F = {}
F["rq"], F["ol"], F["de"], F["sk"] = rq_and_absence(snapP["decay"], snapP["thresh"])
F["lgt"], F["qd"], F["epa"], F["early"] = lgt, qd_ped, f_epa, f_early
F["thfa"], F["rest"], F["luck"], F["tse"] = f_thfa, f_rest, f_luck, f_ts

def X_of(F_):
    return np.column_stack([F_["lgt"], F_["qd"], F_["epa"], F_["early"], F_["thfa"],
                            F_["rest"], F_["luck"], F_["tse"], F_["ol"], F_["de"],
                            F_["sk"], F_["rq"]])

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def cv_ll(X, C=1e6):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=C, max_iter=5000).fit(X[fitm], y[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()

X_CUR0 = X_of(F)                      # frozen baseline snapshot (current model)
cur_cv = cv_ll(X_CUR0)
print(f"[{time.time()-T0:.0f}s] baseline DEV CV {cur_cv:.5f}", flush=True)
CHANGES = {}

def try_knob(name, feat_keys, builder, grid):
    """coordinate sweep: builder(cfg) -> dict of feature arrays for feat_keys."""
    global cur_cv
    best_cfg, best_feats = None, None
    for cfg in grid:
        feats = builder(cfg)
        F2 = dict(F); F2.update(feats)
        s = cv_ll(X_of(F2))
        if s < cur_cv - 1e-6:
            cur_cv = s
            best_cfg, best_feats = cfg, feats
    if best_cfg is not None:
        F.update(best_feats)
        CHANGES[name] = best_cfg
        print(f"[{time.time()-T0:.0f}s] {name}: UPDATED {best_cfg}  DEV CV {cur_cv:.5f}", flush=True)
    else:
        print(f"[{time.time()-T0:.0f}s] {name}: no improvement", flush=True)

# ---- 1. Elo core ----
from nfl_elo import run_elo as _run_elo  # noqa: E402
def b_elo(cfg):
    k, hfa, reg = cfg
    pe2, _ = _run_elo(games, score_from=1999, k=k, hfa=hfa, regress=reg)
    return {"lgt": np.log(np.clip(pe2, eps, 1 - eps) / np.clip(1 - pe2, eps, 1 - eps))}
bp = base["params"]
try_knob("elo", ["lgt"], b_elo,
         [(k, h, r) for k in (30.0, bp["k"], 50.0) for h in (45.0, bp["hfa"], 60.0)
          for r in (0.35, bp["regress"], 0.52) if (k, h, r) != (bp["k"], bp["hfa"], bp["regress"])])

# ---- 2. QB feature ----
def b_qb(cfg):
    dec, pdb, sdec, dlt = cfg
    m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=dlt,
              rep_map={qid: lg_qb + dlt * (1.0 - pedP["c"] * np.exp(-(pk - 1) / pedP["scale"]))
                       for qid, pk in picks.items()},
              decay=dec, prior_db=pdb, season_decay=sdec)
    out, prev = np.zeros(len(games)), None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        out[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
        m.predict(g)
        m.update(g, 0.5, qbw)
    return {"qd": out}
sp = SP
qb_grid = []
for dec in (0.72, sp["decay"], 0.88):
    qb_grid.append((dec, sp["prior_db"], sp["season_decay"], pedP["delta"]))
for pdb in (400.0, 1000.0):
    qb_grid.append((sp["decay"], pdb, sp["season_decay"], pedP["delta"]))
for sd in (0.25, 0.55):
    qb_grid.append((sp["decay"], sp["prior_db"], sd, pedP["delta"]))
for dl in (-0.20, -0.32):
    qb_grid.append((sp["decay"], sp["prior_db"], sp["season_decay"], dl))
try_knob("qb", ["qd"], b_qb, qb_grid)

# ---- 3. per-team HFA ----
def b_thfa(cfg):
    dcy, pn = cfg
    st_ = {}
    f = np.zeros(len(games))
    for i, g in enumerate(games):
        if not g["neutral"]:
            s_ = st_.get(g["home"], (0.0, 0.0))
            f[i] = s_[0] / (s_[1] + pn)
            st_[g["home"]] = (dcy * s_[0] + (g["y"] - pe[i]), dcy * s_[1] + 1.0)
    return {"thfa": f}
try_knob("team_hfa", ["thfa"], b_thfa,
         [(d, p) for d in (0.97, 0.991, 1.0) for p in (90.0, 180.0, 350.0)
          if (d, p) != (0.991, 180.0)])

# ---- 4. fumble luck ----
def b_luck(cfg):
    dcy, pn, sd = cfg
    luck = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in luck.values():
                st_[0] *= (1 - sd); st_[1] *= (1 - sd)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = luck[h][0] / (luck[h][1] + pn) - luck[a][0] / (luck[a][1] + pn)
        gt = tv.get(g["gid"])
        if gt and h in gt and a in gt:
            _, fh, flh = gt[h]; _, fa, fla = gt[a]
            tot = fh + fa; rec_h = (fh - flh) + fla
            for team, rec in ((h, rec_h), (a, tot - rec_h)):
                s_ = luck[team]
                s_[0] = dcy * s_[0] + (rec - 0.5 * tot)
                s_[1] = dcy * s_[1] + tot
    return {"luck": f}
lp = luckP
try_knob("fumble_luck", ["luck"], b_luck,
         [(d, p, s) for d in (0.87, lp["decay"], 0.97) for p in (2.0, lp["prior_n"], 10.0)
          for s in (lp["season_decay"], 0.3) if (d, p, s) != (lp["decay"], lp["prior_n"], lp["season_decay"])])

# ---- 5. rest clip ----
raw_rest = np.array([g["hrest"] - g["arest"] for g in games], float)
def b_rest(cfg):
    return {"rest": np.clip(raw_rest, -cfg, cfg)}
try_knob("rest_clip", ["rest"], b_rest, [4.0, 10.0, 14.0])

# ---- 6. EPA rating ----
def b_epa(cfg):
    dec, pn, sd = cfg
    off = defaultdict(lambda: [0.0, 0.0]); dfa2 = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games)); prev = None
    def rate(st_):
        return (st_[0] + pn * LG_EPA) / (st_[1] + pn)
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in list(off.values()) + list(dfa2.values()):
                st_[0] *= (1 - sd); st_[1] *= (1 - sd)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = (rate(off[h]) - rate(dfa2[h])) - (rate(off[a]) - rate(dfa2[a]))
        tm = agg.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                s_, n_ = tm.get(t_off, (0.0, 0))
                o = off[t_off]; o[0] = dec * o[0] + s_; o[1] = dec * o[1] + n_
                dd = dfa2[opp]; dd[0] = dec * dd[0] + s_; dd[1] = dec * dd[1] + n_
    return {"epa": f}
ep = epaP
try_knob("epa_rating", ["epa"], b_epa,
         [(d, p, s) for d in (0.82, ep["decay"], 0.93) for p in (350.0, ep["prior_n"], 1500.0)
          for s in (0.5, ep["season_decay"], 0.85)
          if (d, p, s) != (ep["decay"], ep["prior_n"], ep["season_decay"])])

# ---- 7. snap absence (re-walk; also rebuilds rq with same share walk) ----
def b_snap(cfg):
    dcy, th = cfg
    frq, fol, fde, fsk = rq_and_absence(dcy, th)
    return {"rq": frq, "ol": fol, "de": fde, "sk": fsk}
try_knob("snap_share", ["rq", "ol", "de", "sk"], b_snap,
         [(d, t) for d in (0.70, snapP["decay"], 0.90) for t in (0.60, snapP["thresh"], 0.85)
          if (d, t) != (snapP["decay"], snapP["thresh"])])

# ---- 8. TS edge (expensive; one-at-a-time) ----
import sys as _sys
_sys.path.insert(0, ".")
from mlbwp.trueskill import TrueSkill1v1  # noqa: E402
def b_ts(cfg):
    beta, tau, widen, mreg = cfg
    ts_ = TrueSkill1v1(beta=beta, tau=tau, mu0=25.0, sig0=25.0 / 3.0)
    f = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for k2 in list(ts_.mu.keys()):
                ts_.mu[k2] = 25.0 + (ts_.mu[k2] - 25.0) * (1 - mreg)
                ts_.var[k2] = min(ts_.var[k2] + widen ** 2, (25.0 / 3.0) ** 2)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = ((ts_.mu[h + "_OFF"] - ts_.mu[a + "_DEF"])
                - (ts_.mu[a + "_OFF"] - ts_.mu[h + "_DEF"]))
        for pos_, dfn, success in plays.get(g["gid"], ()):
            if success:
                ts_.update(pos_ + "_OFF", dfn + "_DEF")
            else:
                ts_.update(dfn + "_DEF", pos_ + "_OFF")
    return {"tse": f}
tp = tsP
ts_grid = ([(b_, tp["tau"], tp["widen"], tp["mu_reg"]) for b_ in (10.0, 20.0)]
           + [(tp["beta"], t_, tp["widen"], tp["mu_reg"]) for t_ in (0.05, 0.2)]
           + [(tp["beta"], tp["tau"], w_, tp["mu_reg"]) for w_ in (1.4, 3.0)]
           + [(tp["beta"], tp["tau"], tp["widen"], m_) for m_ in (0.45, 0.75)])
try_knob("ts_edge", ["tse"], b_ts, ts_grid)

# ---- B: blend ridge sweep ----
bestC, bestS = 1e6, cur_cv
for C in (100.0, 30.0, 10.0, 3.0, 1.0):
    s = cv_ll(X_of(F), C=C)
    if s < bestS - 1e-6:
        bestS, bestC = s, C
print(f"[{time.time()-T0:.0f}s] blend C: best {bestC}  DEV CV {bestS:.5f}", flush=True)
print(f"\nknob changes: {CHANGES}")
print(f"final DEV CV {bestS:.5f}  (baseline was the start value)")

# ---- ONE TEST look ----
X_cur0 = X_CUR0
Xn = X_of(F)
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur0[fitm], y[fitm])
c1 = LogisticRegression(C=bestC, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur0[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"\nmain TEST: current {llv(yt, p0).mean():.5f} -> tuned {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"changes": {k: list(v) if isinstance(v, tuple) else v for k, v in CHANGES.items()},
           "blend_C": bestC, "dev_cv": round(bestS, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_coord_tune.json", "w"), indent=1)
