"""ML-learned composite weights: which metric mix best predicts FUTURE performance?

Stage 1 (player level, ~150k rows, DEV era only): per bucket, ridge-fit
    future_primary_value(next 4 appearances)  ~  current z-components
  QB   [dak, negsk, repa]   -> future att-weighted dakota
  SK   [eff, yacr, wopr]    -> future EPA/opportunity
  DEF  [rush, ball, tak]    -> future fixed-currency value/game
Stage 2: rebuild roster_quality with learned weights; ONE TEST compare vs the
current 12-feature model (old hand weights); adopt if better (Samuel's rule).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 2")[0])  # loaders, engine defs, Z tables  # noqa: S102

HAND = {"QB": {"dak": 0.50, "negsk": 0.25, "repa": 0.25},
        "SK": {"eff": 0.40, "yacr": 0.15, "wopr": 0.45},
        "DL": {"rush": 0.45, "ball": 0.35, "tak": 0.20},
        "LB": {"rush": 0.45, "ball": 0.35, "tak": 0.20},
        "DB": {"rush": 0.45, "ball": 0.35, "tak": 0.20}}
COMPS = {"QB": ["dak", "negsk", "repa"], "SK": ["eff", "yacr", "wopr"],
         "DL": ["rush", "ball", "tak"], "LB": ["rush", "ball", "tak"],
         "DB": ["rush", "ball", "tak"]}

def week_value(pid, s, w, b):
    if b == "QB":
        o = OFFW.get((pid, s, w))
        if o is None:
            return None
        db = o["att"] + o["sk"]
        return (o["dak"], db) if db > 0 else None
    if b == "SK":
        o = OFFW.get((pid, s, w))
        if o is None:
            return None
        opp = o["car"] + o["tgt"]
        return ((o["repa"] + o["cepa"]) / opp, opp) if opp > 0 else None
    d = DEFW.get((pid, s, w))
    if d is None:
        return None
    return (d["sack"] + 0.5 * d["tfl"] + 0.5 * d["hit"]
            + 3.0 * d["int"] + d["pd"] + 0.15 * d["solo"], 1.0)

# per-player appearance sequences
appear = defaultdict(list)
for (pid, s, w) in list(OFFW) + list(DEFW):
    appear[pid].append((s, w))
for pid in appear:
    appear[pid] = sorted(set(appear[pid]))

# stage 1: training rows via independent per-player replays
ROWS = defaultdict(lambda: ([], []))          # bucket -> (X, y)
for pid, seq in appear.items():
    STATE.pop(pid, None)
    vals = []
    b = None
    for (s, w) in seq:
        o = OFFW.get((pid, s, w)); d = DEFW.get((pid, s, w))
        bb = bucket((o or d)["pos"])
        b = bb
        vals.append(week_value(pid, s, w, bb))
    # replay: capture z-vector before each appearance, target = next-4 weighted value
    STATE.pop(pid, None)
    for t, (s, w) in enumerate(seq):
        if s <= 2014 and b in Z:
            st = STATE.get(pid)
            if st is not None and st.get("bucket") == b:
                x = []
                ok = True
                for k in COMPS[b]:
                    wgt = st.get(k + "_w", 0.0)
                    if wgt <= 3 or k not in Z[b]:
                        ok = False
                        break
                    mu, sd = Z[b][k]
                    x.append(((st[k] + 6.0 * mu) / (wgt + 6.0) - mu) / sd)
                if ok:
                    fw = [vals[j] for j in range(t + 1, min(t + 5, len(seq)))
                          if vals[j] is not None and seq[j][0] <= 2015]
                    if len(fw) >= 2:
                        num = sum(v * wt for v, wt in fw)
                        den = sum(wt for _, wt in fw)
                        if den > 0:
                            ROWS[b][0].append(x)
                            ROWS[b][1].append(num / den)
        apply_week(pid, s, w)

LEARNED = {}
print(f"{'bucket':<6}{'n rows':>8}   hand weights        ->  learned weights (|w| normalized)")
for b, (X, yv) in ROWS.items():
    X = np.array(X); yv = np.array(yv)
    yz = (yv - yv.mean()) / yv.std()
    rg = Ridge(alpha=100.0).fit(X, yz)
    wraw = rg.coef_
    wn = wraw / max(np.abs(wraw).sum(), 1e-9)
    LEARNED[b] = dict(zip(COMPS[b], wn))
    hd = [HAND[b][k] for k in COMPS[b]]
    print(f"{b:<6}{len(yv):>8}   {np.round(hd, 2)}  ->  {np.round(wn, 3)}"
          f"   (R2 {rg.score(X, yz):.3f})")

# stage 2: rebuild roster quality with old vs learned weights in ONE walk
def make_rate(weights):
    def rate(pid):
        st = STATE.get(pid)
        if st is None:
            return None
        b = st.get("bucket")
        if b not in Z:
            return None
        tot, any_ = 0.0, False
        for k, wt in weights[b].items():
            wgt = st.get(k + "_w", 0.0)
            if wgt <= 1 or k not in Z[b]:
                continue
            mu, sd = Z[b][k]
            v = (st[k] + 6.0 * mu) / (wgt + 6.0)
            tot += wt * (v - mu) / sd
            any_ = True
        return tot if any_ else None
    return rate

rate_old, rate_new = make_rate(HAND), make_rate(LEARNED)
STATE.clear()
share, pos_of, team_of = {}, {}, {}
roster = defaultdict(set)
f_ol = np.zeros(len(games)); f_def = np.zeros(len(games)); f_sk = np.zeros(len(games))
f_rq_old = np.zeros(len(games)); f_rq_new = np.zeros(len(games))
done = set()

def unit_missing(team, gid):
    tbl = snaps.get((gid, team))
    if tbl is None:
        return 0.0, 0.0, 0.0
    m = [0.0, 0.0, 0.0]
    for pid in roster.get(team, ()):
        st = share.get(pid)
        if not st or st[1] <= 0:
            continue
        sh = st[0] / st[1]
        if sh < snapP["thresh"] or pid in tbl:
            continue
        p = pos_of.get(pid, "")
        if p in OLPOS:
            m[0] += sh
        elif p in DEFPOS:
            m[1] += sh
        elif p in SKPOS:
            m[2] += sh
    return m[0], m[1], m[2]

def roster_q(team, gid, rate):
    tbl = snaps.get((gid, team))
    if tbl is None:
        return 0.0
    tot = 0.0
    for pid in tbl:
        st = share.get(pid)
        if not st or st[1] <= 0:
            continue
        g_ = pfr2gsis.get(pid)
        r_ = rate(g_) if g_ else None
        if r_ is not None:
            tot += (st[0] / st[1]) * max(-2.0, min(2.0, r_))
    return tot

for i, g in enumerate(games):
    s, w = g["season"], g["week"]
    if s >= 2013:
        h, a = g["home"], g["away"]
        mh = unit_missing(h, g["gid"]); ma = unit_missing(a, g["gid"])
        f_ol[i], f_def[i], f_sk[i] = mh[0] - ma[0], mh[1] - ma[1], mh[2] - ma[2]
        f_rq_old[i] = roster_q(h, g["gid"], rate_old) - roster_q(a, g["gid"], rate_old)
        f_rq_new[i] = roster_q(h, g["gid"], rate_new) - roster_q(a, g["gid"], rate_new)
    for team in (g["home"], g["away"]):
        for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
            if (pid, s, w) in done:
                continue
            done.add((pid, s, w))
            apply_week(pid, s, w)
    for team in (g["home"], g["away"]):
        tbl = snaps.get((g["gid"], team))
        if not tbl:
            continue
        for pid, (pos, op, dp) in tbl.items():
            pct = dp if pos in DEFPOS else op
            st = share.setdefault(pid, [0.0, 0.0])
            st[0] = snapP["decay"] * st[0] + pct
            st[1] = snapP["decay"] * st[1] + 1.0
            pos_of[pid] = pos
            if team_of.get(pid) != team:
                if team_of.get(pid) in roster:
                    roster[team_of[pid]].discard(pid)
                team_of[pid] = team
                roster[team].add(pid)

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

X12_old = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                           f_ol, f_def, f_sk, f_rq_old])
X12_new = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                           f_ol, f_def, f_sk, f_rq_new])
fitm = dev & (y != 0.5)
co = LogisticRegression(C=1e6, max_iter=5000).fit(X12_old[fitm], y[fitm])
cn = LogisticRegression(C=1e6, max_iter=5000).fit(X12_new[fitm], y[fitm])
po = co.predict_proba(X12_old[test])[:, 1]
pn = cn.predict_proba(X12_new[test])[:, 1]
yt = y[test]
d = llv(yt, po) - llv(yt, pn)
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmain TEST 2016-2025 (ONE compare):")
print(f"  current model (hand weights)   LL {llv(yt, po).mean():.5f}")
print(f"  ML-learned weights             LL {llv(yt, pn).mean():.5f}")
print(f"  delta {d.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")
json.dump({"learned": {b: {k: round(v, 4) for k, v in ws.items()} for b, ws in LEARNED.items()},
           "test_old": round(float(llv(yt, po).mean()), 5),
           "test_new": round(float(llv(yt, pn).mean()), 5),
           "delta": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)]},
          open("data/nfl_ml_weights.json", "w"), indent=1)
