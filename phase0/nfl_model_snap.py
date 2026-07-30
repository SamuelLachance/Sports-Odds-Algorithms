"""Incorporate snap-count absence (ol/def/skill diffs, tuned params) into the model.

Feature live 2013+ (zeros before; coefficient identified by 2013+ rows only).
Main-protocol scoring: fit DEV 2001-2015, evaluate TEST 2016-2025 vs the current
8-feature model. Production coefficients refit on all completed seasons.
Updates data/nfl_model.json if the new model scores better (Samuel's rule).
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

snapP = json.load(open("data/nfl_snap_absence.json"))["params"]   # decay, thresh
OL = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}
DEFP = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB", "EDGE", "DL"}
SKILL = {"RB", "WR", "TE", "FB", "HB"}

snaps = defaultdict(dict)
# discover snap tables on disk (snap_2026.csv is picked up the day it lands);
# 2013 floor kept: snap_2012.csv exists but was never read (feature is live 2013+)
SNAP_FILES = [p for p in sorted(glob.glob("data/snap_20??.csv"))
              if int(p[-8:-4]) >= 2013]
for path_ in SNAP_FILES:
    for r in csv.DictReader(open(path_)):
        t = PBP_FIX.get(r["team"], r["team"])
        try:
            op, dp = float(r["offense_pct"] or 0), float(r["defense_pct"] or 0)
        except ValueError:
            continue
        snaps[(r["game_id"], t)][r["pfr_player_id"]] = (r["position"], op, dp)


def absence_features(decay, thresh):
    share, pos_of, team_of = {}, {}, {}
    roster = defaultdict(set)
    f_ol = np.zeros(len(games)); f_def = np.zeros(len(games)); f_sk = np.zeros(len(games))

    def unit_missing(team, gid):
        tbl = snaps.get((gid, team))
        if tbl is None:
            return 0.0, 0.0, 0.0
        m = [0.0, 0.0, 0.0]
        for pid in roster.get(team, ()):
            st = share.get(pid)
            if not st or st[1] <= 0:
                continue
            s = st[0] / st[1]
            if s < thresh or pid in tbl:
                continue
            p = pos_of.get(pid, "")
            if p in OL:
                m[0] += s
            elif p in DEFP:
                m[1] += s
            elif p in SKILL:
                m[2] += s
        return m[0], m[1], m[2]

    for i, g in enumerate(games):
        if g["season"] < 2013:
            continue
        h, a = g["home"], g["away"]
        oh = unit_missing(h, g["gid"]); oa = unit_missing(a, g["gid"])
        f_ol[i], f_def[i], f_sk[i] = oh[0] - oa[0], oh[1] - oa[1], oh[2] - oa[2]
        for team in (h, a):
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEFP else op
                st = share.setdefault(pid, [0.0, 0.0])
                st[0] = decay * st[0] + pct
                st[1] = decay * st[1] + 1.0
                pos_of[pid] = pos
                if team_of.get(pid) != team:
                    if team_of.get(pid) in roster:
                        roster[team_of[pid]].discard(pid)
                    team_of[pid] = team
                    roster[team].add(pid)
    return f_ol, f_def, f_sk


f_ol, f_def, f_sk = absence_features(**snapP)
X8 = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts])
X11 = np.column_stack([X8, f_ol, f_def, f_sk])


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


fit = dev & (y != 0.5)
c8 = LogisticRegression(C=1e6, max_iter=5000).fit(X8[fit], y[fit])
c11 = LogisticRegression(C=1e6, max_iter=5000).fit(X11[fit], y[fit])
p8 = c8.predict_proba(X8[test])[:, 1]
p11 = c11.predict_proba(X11[test])[:, 1]
yt = y[test]
d = llv(yt, p8) - llv(yt, p11)
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"main TEST 2016-2025 (n={n}):")
print(f"  current model (8f)   LL {llv(yt, p8).mean():.5f}")
print(f"  + snap absence (11f) LL {llv(yt, p11).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

if llv(yt, p11).mean() < llv(yt, p8).mean():
    allfit = (seasons >= DEV_SCORE_FROM) & (y != 0.5)
    call = LogisticRegression(C=1e6, max_iter=5000).fit(X11[allfit], y[allfit])
    spec = json.load(open("data/nfl_model.json"))
    spec["features"] = spec["features"] + ["ol_absence", "def_absence", "skill_absence"]
    spec["component_params"]["snap_absence"] = {**snapP, "data_from": 2013,
                                               "zero_before": True}
    spec["prev_test_ll"] = spec["test_ll"]
    spec["test_ll"] = round(float(llv(yt, p11).mean()), 5)
    spec["blend_dev_fit"] = {"coef": [round(float(c), 6) for c in c11.coef_[0]],
                             "intercept": round(float(c11.intercept_[0]), 6)}
    spec["blend_production"] = {"coef": [round(float(c), 6) for c in call.coef_[0]],
                                "intercept": round(float(call.intercept_[0]), 6),
                                "fit": "2001-2025 all completed, ties dropped"}
    json.dump(spec, open("data/nfl_model.json", "w"), indent=1)
    print("\nsnap coefs (production):",
          {k: round(float(c), 4) for k, c in zip(("ol", "def", "skill"), call.coef_[0][8:])})
    print("data/nfl_model.json UPDATED -> model now includes snap absence")
else:
    print("\nnew model NOT better on main TEST -> model unchanged (Samuel's rule)")
