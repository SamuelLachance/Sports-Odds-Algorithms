"""Breakthrough program 2026-09-24 -- NFL gap anatomy, step 6 (what IS the margin channel?).

The era split shows the only mechanism whose gap survives once player availability
is live (2013-2015) is 'season point differential disagrees with our model'. Ledger
row 10 (SRS, ridge-Massey margin) failed TEST, so before anyone builds on this we
decompose the channel on DEV only:
  M1 season-to-date PD/g                       (the anatomy variable)
  M2 capped margin (|m|<=14) season-to-date
  M3 EWMA margin, carry-over across seasons (decay .9/game, season keep .5), cap 21
  M4 opponent-adjusted PD/g (own PD/g minus mean PD/g of opponents faced, as-of)
  M5 NON-EPA margin: season-to-date mean of (game margin - k*game EPA diff), k fit
     on PRIOR seasons only -> points our EPA rating cannot see (finishing, ST,
     return/defensive TDs, luck)
  M6 EPA-implied margin: season-to-date mean of k*game EPA diff (the part EPA sees)
For each: corr with the market's out-of-span residual r_perp (where the market's
extra information lies) and LOSO outcome gain (market-blind) on all DEV and on the
2013-2015 era where player columns are live. Odds = evaluation only.
Output data/bt_nfl_anatomy5.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

src4 = open("phase0/bt_nfl_anatomy4.py", encoding="utf-8").read()
exec(src4.split("lock_bye_h = ")[0])  # frame + A-D + fit_logit/loso helpers  # noqa: S102

# per-game offensive EPA per team (nfl_plays.csv; raw codes may be pbp codes -> map)
FIX = {"JAC": "JAX", "WSH": "WAS", "LA": "STL", "SD": "SD", "OAK": "OAK", "STL": "STL", "LAC": "SD", "LV": "OAK"}
gepa = defaultdict(lambda: defaultdict(float))
with open("data/nfl_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); next(rd)
    for gid_, _, pos_, _, epa_ in rd:
        if int(gid_[:4]) > 2015:
            continue
        gepa[gid_][FIX.get(pos_, pos_)] += float(epa_)

# walk all games <= 2015 in order; per team as-of states
seq = sorted(raw_all, key=lambda r: (r["gameday"], r["game_id"]))
# k (points per EPA) fit on PRIOR seasons only: pooled OLS of margin on EPA diff through s-1
per_season_pairs = defaultdict(list)
for r in seq:
    e = gepa.get(r["game_id"])
    if not e or r["home_team"] not in e or r["away_team"] not in e:
        continue
    per_season_pairs[int(r["season"])].append((e[r["home_team"]] - e[r["away_team"]],
                                               int(r["home_score"]) - int(r["away_score"])))
def k_for(s):
    xs, ys_ = [], []
    for s_ in range(1999, s):
        for a, b in per_season_pairs[s_]:
            xs.append(a); ys_.append(b)
    if len(xs) < 200:
        return 1.0
    xs = np.array(xs); ys_ = np.array(ys_)
    return float((xs @ ys_) / (xs @ xs))
KS = {s: k_for(s) for s in range(1999, 2016)}

st = defaultdict(lambda: {"m": [], "mc": [], "opp": [], "ne": [], "ee": []})
ew = defaultdict(lambda: [0.0, 0.0])
pd_now = {}
feat = {}
prev_season = None
for r in seq:
    s = int(r["season"]); gid_ = r["game_id"]
    if prev_season is not None and s != prev_season:
        for v in ew.values():
            v[0] *= 0.5; v[1] *= 0.5
    prev_season = s
    h, a = r["home_team"], r["away_team"]
    def ssum(t, key):
        L = [x for (ss, x) in st[t][key] if ss == s]
        return (sum(L) / len(L)) if len(L) >= 3 else np.nan
    def oppadj(t):
        L = [(x, o) for (ss, x, o) in st[t]["opp"] if ss == s]
        if len(L) < 3:
            return np.nan
        own = sum(x for x, _ in L) / len(L)
        opp_pd = [pd_now.get((o, s), 0.0) for _, o in L]
        return own + sum(opp_pd) / len(opp_pd)     # beating good opponents counts more
    feat[gid_] = {
        "M1": ssum(h, "m") - ssum(a, "m"),
        "M2": ssum(h, "mc") - ssum(a, "mc"),
        "M3": (ew[h][0] / ew[h][1] if ew[h][1] > 0 else 0.0) - (ew[a][0] / ew[a][1] if ew[a][1] > 0 else 0.0),
        "M4": oppadj(h) - oppadj(a),
        "M5": ssum(h, "ne") - ssum(a, "ne"),
        "M6": ssum(h, "ee") - ssum(a, "ee"),
    }
    hs, as_ = int(r["home_score"]), int(r["away_score"])
    e = gepa.get(gid_, {})
    ed = (e.get(h, np.nan) - e.get(a, np.nan)) if e else np.nan
    k = KS[s]
    for t, o, mg, sg in ((h, a, hs - as_, 1.0), (a, h, as_ - hs, -1.0)):
        st[t]["m"].append((s, mg))
        st[t]["mc"].append((s, max(-14, min(14, mg))))
        st[t]["opp"].append((s, mg, o))
        if not np.isnan(ed):
            st[t]["ne"].append((s, mg - k * sg * ed))
            st[t]["ee"].append((s, k * sg * ed))
        cm = max(-21, min(21, mg))
        ew[t][0] = 0.9 * ew[t][0] + cm; ew[t][1] = 0.9 * ew[t][1] + 1.0
        L = [x for (ss, x) in st[t]["m"] if ss == s]
        pd_now[(t, s)] = sum(L) / len(L)

# r_perp from the anatomy frame (market logit residual after projecting on our span)
Xspan = np.column_stack([np.ones(n)] + [np.array([g["X"][c] for g in G]) for c in
                         ("lgt", "qd", "epa", "early", "thfa", "rest", "luck", "ol", "de", "sk", "rq", "pass", "run")]
                        + [logit(p)])
bproj = np.linalg.lstsq(Xspan, logit(pm), rcond=None)[0]
r_perp = logit(pm) - Xspan @ bproj
late_era = SEA >= 2013
res = {"k_points_per_epa_by_season": {s: round(v, 3) for s, v in KS.items() if s >= 2006}}
rows_ = []
for key, lab in (("M1", "season-to-date PD/g"), ("M2", "capped (14) PD/g"), ("M3", "EWMA capped-21 margin, carry-over"),
                 ("M4", "opponent-adjusted PD/g"), ("M5", "NON-EPA margin (margin - k*EPA diff)"),
                 ("M6", "EPA-implied margin (k*EPA diff)")):
    v = np.array([feat[g["gid"]][key] for g in G], float)
    ok = ~np.isnan(v)
    v0 = np.where(ok, v, 0.0)
    g_all, wall = loso_free(v0)
    c_perp = float(np.corrcoef(v[ok], r_perp[ok])[0, 1])
    c_perp_late = float(np.corrcoef(v[ok & late_era], r_perp[ok & late_era])[0, 1])
    c_out = float(np.corrcoef(v[ok], (y - p)[ok])[0, 1])
    c_out_late = float(np.corrcoef(v[ok & late_era], (y - p)[ok & late_era])[0, 1])
    rows_.append({"var": key, "label": lab, "n_live": int(ok.sum()),
                  "corr_r_perp": round(c_perp, 4), "corr_r_perp_2013_15": round(c_perp_late, 4),
                  "corr_outcome_resid": round(c_out, 4), "corr_outcome_resid_2013_15": round(c_out_late, 4),
                  "loso_gain_all": round(float(g_all.mean()), 5), "loso_ci_all": boot_ci(g_all),
                  "loso_gain_2013_15": round(float(g_all[late_era].mean()), 5), "loso_ci_2013_15": boot_ci(g_all[late_era]),
                  "coef_all_dev": round(float(wall[2]), 4)})
    print(f"{key} {lab:<40} corr_rperp={c_perp:+.3f} (13-15 {c_perp_late:+.3f}) corr_out={c_out:+.3f} "
          f"(13-15 {c_out_late:+.3f}) loso={g_all.mean():+.5f} {rows_[-1]['loso_ci_all']} "
          f"13-15 {g_all[late_era].mean():+.5f}")
res["variants"] = rows_
# joint: EPA-seen vs non-EPA parts together
v5 = np.nan_to_num(np.array([feat[g["gid"]]["M5"] for g in G], float))
v6 = np.nan_to_num(np.array([feat[g["gid"]]["M6"] for g in G], float))
Xb = np.column_stack([np.ones(n), logit(p)]); Xp = np.column_stack([np.ones(n), logit(p), v5, v6])
lb = np.zeros(n); lpp = np.zeros(n)
for s in range(2006, 2016):
    te = SEA == s
    wb = fit_logit(Xb[~te], y[~te]); wp = fit_logit(Xp[~te], y[~te])
    lb[te] = llv(y[te], 1 / (1 + np.exp(-(Xb[te] @ wb)))); lpp[te] = llv(y[te], 1 / (1 + np.exp(-(Xp[te] @ wp))))
res["joint_M5_M6"] = {"loso_gain": round(float((lb - lpp).mean()), 5), "ci": boot_ci(lb - lpp),
                      "coefs_all_dev": [round(float(c), 4) for c in fit_logit(Xp, y)]}
# market projection weights (measuring device): how does the close load on the two parts beyond our logit?
A_ = np.column_stack([np.ones(n), logit(p), v5, v6])
res["close_projection_on[1,logit_p,nonEPA,EPAimplied]"] = [round(float(c), 4) for c in np.linalg.lstsq(A_, logit(pm), rcond=None)[0]]
print(res["joint_M5_M6"], res["close_projection_on[1,logit_p,nonEPA,EPAimplied]"])
json.dump(res, open("data/bt_nfl_anatomy5.json", "w"), indent=1)
print("wrote data/bt_nfl_anatomy5.json")
