"""Non-QB 'missing production' feature (DEV-ONLY screen, gate -0.0020).

Rate every RB/WR/TE/FB like the QB: (rushing_epa+receiving_epa)/(carries+targets)
EWMA, EB-shrunk toward position replacement (pos_lg + rep_delta). Expected starters =
top-K by usage EWMA on each team; the feature is the rated value of expected starters
ABSENT this week (no stat line = inactive; serve-time equivalent: the 90-min inactives
list — the MLB lineup-card precedent). f = home_missing - away_missing.
Screened against the CURRENT 8-feature model.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

XBASE = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts])
GATE = 0.0020
SKILL = ("RB", "WR", "TE", "FB")


def f2(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


# weekly skill lines: pw[(sid,season,week)] -> (epa, opp, team, pos); played[(team,s,w)] -> set
pw = {}
played = defaultdict(set)
for path, team_col in (("data/nfl_player_stats.csv", "recent_team"),
                       ("data/nfl_player_stats_2025.csv", "team")):
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("position") not in SKILL:
            continue
        t = PBP_FIX.get(r[team_col], r[team_col])
        s, w = int(r["season"]), int(r["week"])
        key = (r["player_id"], s, w)
        epa = f2(r.get("rushing_epa")) + f2(r.get("receiving_epa"))
        opp = f2(r.get("carries")) + f2(r.get("targets"))
        if key in pw:
            e0, o0, _, _ = pw[key]
            pw[key] = (e0 + epa, o0 + opp, t, r["position"])
        else:
            pw[key] = (epa, opp, t, r["position"])
        played[(t, s, w)].add(r["player_id"])

# attach week to games
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g["week"] = int(r["week"])

# DEV-era position league rates
pos_num = defaultdict(float); pos_den = defaultdict(float)
for (pid, s, w), (epa, opp, t, pos) in pw.items():
    if s in DEV_YEARS:
        p = "RB" if pos in ("RB", "FB") else pos
        pos_num[p] += epa; pos_den[p] += opp
POS_LG = {p: pos_num[p] / pos_den[p] for p in pos_num}
print("DEV pos EPA/opp:", {p: round(v, 4) for p, v in POS_LG.items()})


def missing_feature(decay, prior_opp, usage_decay, K, rep_delta):
    rating = {}                 # pid -> [epa_sum, opp_sum]
    usage = {}                  # pid -> [opp_ewma_sum, wk_count]
    team_of = {}                # pid -> last team
    pos_of = {}
    f = np.zeros(len(games))
    done_weeks = set()

    def player_val(pid):
        pos = "RB" if pos_of.get(pid) in ("RB", "FB") else pos_of.get(pid, "WR")
        lgr = POS_LG.get(pos, 0.0)
        rep = lgr + rep_delta
        st = rating.get(pid)
        if not st or st[1] <= 0:
            q = rep
        else:
            q = (st[0] + prior_opp * rep) / (st[1] + prior_opp)
        u = usage.get(pid)
        upg = (u[0] / u[1]) if u and u[1] > 0 else 0.0
        return (q - rep) * upg          # EPA/game above replacement

    def team_missing(team, s, w):
        cands = [(usage[pid][0] / usage[pid][1], pid) for pid in team_roster.get(team, ())
                 if pid in usage and usage[pid][1] > 0]
        cands.sort(reverse=True)
        miss = 0.0
        for _, pid in cands[:K]:
            if pid not in played.get((team, s, w), ()):
                miss += player_val(pid)
        return miss

    team_roster = defaultdict(set)
    for i, g in enumerate(games):
        s, w = g["season"], g["week"]
        f[i] = team_missing(g["home"], s, w) - team_missing(g["away"], s, w)
        wk = (s, w, g["home"], g["away"])
        # update AFTER the game with both teams' actual lines this week
        for team in (g["home"], g["away"]):
            for pid in played.get((team, s, w), ()):
                line = pw.get((pid, s, w))
                if line is None:
                    continue
                epa, opp, t, pos = line
                st = rating.setdefault(pid, [0.0, 0.0])
                st[0] = decay * st[0] + epa
                st[1] = decay * st[1] + opp
                u = usage.setdefault(pid, [0.0, 0.0])
                u[0] = usage_decay * u[0] + opp
                u[1] = usage_decay * u[1] + 1.0
                if team_of.get(pid) != team:
                    if team_of.get(pid) in team_roster:
                        team_roster[team_of[pid]].discard(pid)
                    team_of[pid] = team
                    team_roster[team].add(pid)
                pos_of[pid] = pos
    return f


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


base_cv = cv_ll(XBASE)
print(f"current model DEV CV {base_cv:.5f}   (gate: <= {base_cv - GATE:.5f})\n")

rng = np.random.default_rng(20260723)
best, t0 = None, time.time()
for i in range(100):
    cand = dict(
        decay=float(rng.uniform(0.70, 1.0)),
        prior_opp=float(np.exp(rng.uniform(np.log(10.0), np.log(400.0)))),
        usage_decay=float(rng.uniform(0.50, 0.95)),
        K=int(rng.choice([4, 5, 6, 7])),
        rep_delta=float(rng.uniform(-0.6, 0.0)),
    )
    fm = missing_feature(**cand)
    s = cv_ll(np.column_stack([XBASE, fm]))
    if best is None or s < best[0]:
        best = (s, cand)
    if (i + 1) % 25 == 0:
        print(f"  ...{i+1}/100  best CV {best[0]:.5f} ({best[0]-base_cv:+.5f})"
              f"  ({time.time()-t0:.0f}s)", flush=True)

s, cand = best
print(f"\nbest: {({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cand.items()})}")
print(f"DEV CV: current {base_cv:.5f} -> +missing_prod {s:.5f} ({s-base_cv:+.5f})")
verdict = "GATE MET -> earns the TEST look" if s <= base_cv - GATE else "gate NOT met -> parked"
print(verdict)
json.dump({"dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "delta_cv": round(s - base_cv, 5), "params": cand, "verdict": verdict},
          open("data/nfl_missing_prod.json", "w"), indent=1)
