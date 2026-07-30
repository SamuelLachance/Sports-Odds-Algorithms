"""THE BIG TEST: all near-miss features combined, one pre-specified TEST look.

Components (all with their previously-tuned params, NO re-tuning):
  qb-ped   qb_diff with draft-pedigree replacement prior     (TEST near-miss +0.00100)
  epa      off/def EPA-per-play net rating                   (TEST near-miss +0.00154)
  ts       off/def per-play TrueSkill edge                   (TEST near-miss +0.00094)
  rest     rest_diff clipped +-7                             (TEST near-miss +0.00048)
  early    west-team-at-early-ET-kick flag                   (DEV -0.00056)
  thfa     per-team HFA residual EWMA                        (DEV -0.00050)
  luck     fumble-recovery luck EWMA                         (DEV -0.00037)

Variant chosen on DEV 5-fold CV ONLY:
  V1 = base + epa + early + thfa            (the pre-registered lean set)
  V2 = V1 + rest + luck
  V3 = V2 with qb -> pedigree-qb
  V4 = V3 + ts                              (everything)
Winner gets the ONE TEST look vs frozen blend (0.62806), paired bootstrap.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.trueskill import TrueSkill1v1  # noqa: E402
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo, FRANCHISE  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

MU0, SIG0 = 25.0, 25.0 / 3.0
PBP_FIX = {"JAC": "JAX", "WSH": "WAS", **FRANCHISE}

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
epaP = json.load(open("data/nfl_epa_rating.json"))["params"]
tsP = json.load(open("data/nfl_ts_report.json"))["ts_params"]
pedP = json.load(open("data/nfl_qb_pedigree.json"))["params"]
luckP = json.load(open("data/nfl_turnover.json"))["params"]
lg_qb = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g["gid"] = r["game_id"]
    g["hrest"], g["arest"] = int(r["home_rest"]), int(r["away_rest"])
    try:
        g["kick"] = int(r["gametime"].split(":")[0]) + int(r["gametime"].split(":")[1]) / 60.0
    except (ValueError, IndexError, AttributeError):
        g["kick"] = 13.0

# ---------- base signals ----------
pe, y = run_elo(games, score_from=1999, **base["params"])
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)


def qb_feature(rep_map=None, delta=SP["delta"]):
    m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=delta,
              rep_map=rep_map, decay=SP["decay"], prior_db=SP["prior_db"],
              season_decay=SP["season_decay"])
    out, prev = np.zeros(len(games)), None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        out[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
        m.predict(g)
        m.update(g, 0.5, qbw)
    return out


qd = qb_feature()                                             # rep-prior (shipped)
picks = {}
for r in csv.DictReader(open("data/nfl_draft_picks.csv", encoding="utf-8")):
    if r["position"] == "QB" and r["gsis_id"] and r["pick"]:
        picks[r["gsis_id"]] = int(r["pick"])
rep_map = {qid: lg_qb + pedP["delta"] * (1.0 - pedP["c"] * math.exp(-(pk - 1) / pedP["scale"]))
           for qid, pk in picks.items()}
qd_ped = qb_feature(rep_map=rep_map, delta=pedP["delta"])     # pedigree prior

# ---------- EPA rating ----------
agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
for r in csv.reader(open("data/nfl_plays.csv")):
    if r[0] == "game_id":
        continue
    gid, _, pos, _, epa = r
    t = PBP_FIX.get(pos, pos)
    a = agg[gid][t]
    a[0] += float(epa); a[1] += 1
num = den = 0.0
for gid, tm in agg.items():
    if int(gid[:4]) in DEV_YEARS:
        for t, (s, n) in tm.items():
            num += s; den += n
LG_EPA = num / den

def epa_net():
    off = defaultdict(lambda: [0.0, 0.0]); dfa = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games)); prev = None
    d, pn, sd = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
    def rate(st): return (st[0] + pn * LG_EPA) / (st[1] + pn)
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st in list(off.values()) + list(dfa.values()):
                st[0] *= (1 - sd); st[1] *= (1 - sd)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = (rate(off[h]) - rate(dfa[h])) - (rate(off[a]) - rate(dfa[a]))
        tm = agg.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                s, n = tm.get(t_off, (0.0, 0))
                o = off[t_off]; o[0] = d * o[0] + s; o[1] = d * o[1] + n
                dd = dfa[opp]; dd[0] = d * dd[0] + s; dd[1] = d * dd[1] + n
    return f

f_epa = epa_net()

# ---------- TrueSkill edge ----------
plays = defaultdict(list)
for r in csv.reader(open("data/nfl_plays.csv")):
    if r[0] == "game_id":
        continue
    gid, _, pos, dfn, epa = r
    plays[gid].append((PBP_FIX.get(pos, pos), PBP_FIX.get(dfn, dfn), float(epa) > 0))

def ts_edge():
    ts = TrueSkill1v1(beta=tsP["beta"], tau=tsP["tau"], mu0=MU0, sig0=SIG0)
    f = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for k in list(ts.mu.keys()):
                ts.mu[k] = MU0 + (ts.mu[k] - MU0) * (1 - tsP["mu_reg"])
                ts.var[k] = min(ts.var[k] + tsP["widen"] ** 2, SIG0 * SIG0)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = ((ts.mu[h + "_OFF"] - ts.mu[a + "_DEF"])
                - (ts.mu[a + "_OFF"] - ts.mu[h + "_DEF"]))
        for pos, dfn, success in plays.get(g["gid"], ()):
            if success:
                ts.update(pos + "_OFF", dfn + "_DEF")
            else:
                ts.update(dfn + "_DEF", pos + "_OFF")
    return f

f_ts = ts_edge()

# ---------- rest / early / per-team HFA / fumble luck ----------
f_rest = np.array([max(-7, min(7, g["hrest"] - g["arest"])) for g in games], float)

TZ = {"BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "BAL": 0, "CIN": 0, "CLE": 0, "PIT": 0,
      "IND": 0, "JAX": 0, "ATL": 0, "CAR": 0, "TB": 0, "WAS": 0, "NYG": 0, "PHI": 0,
      "DET": 0, "CHI": 1, "GB": 1, "MIN": 1, "DAL": 1, "HOU": 1, "TEN": 1, "KC": 1,
      "NO": 1, "DEN": 2, "ARI": 2, "SEA": 3, "SF": 3, "LAC": 3, "LV": 3, "LA": 3}
def tz_of(t, s): return 1 if (t == "LA" and s < 2016) else TZ[t]
f_early = np.zeros(len(games))
for i, g in enumerate(games):
    if not g["neutral"]:
        if tz_of(g["away"], g["season"]) - tz_of(g["home"], g["season"]) >= 2 and g["kick"] <= 14.0:
            f_early[i] = 1.0

def team_hfa(decay=0.991, prior_n=180.0):
    st = {}; f = np.zeros(len(games))
    for i, g in enumerate(games):
        if not g["neutral"]:
            s = st.get(g["home"], (0.0, 0.0))
            f[i] = s[0] / (s[1] + prior_n)
            st[g["home"]] = (decay * s[0] + (g["y"] - pe[i]), decay * s[1] + 1.0)
    return f
f_thfa = team_hfa()

tv = defaultdict(dict)
for r in csv.DictReader(open("data/nfl_turnovers.csv")):
    t = PBP_FIX.get(r["posteam"], r["posteam"])
    tv[r["game_id"]][t] = (float(r["int_thrown"]), float(r["fumbles"]), float(r["fumbles_lost"]))

def fumble_luck():
    luck = defaultdict(lambda: [0.0, 0.0]); f = np.zeros(len(games)); prev = None
    d, pn, sd = luckP["decay"], luckP["prior_n"], luckP["season_decay"]
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st in luck.values():
                st[0] *= (1 - sd); st[1] *= (1 - sd)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f[i] = luck[h][0] / (luck[h][1] + pn) - luck[a][0] / (luck[a][1] + pn)
        gt = tv.get(g["gid"])
        if gt and h in gt and a in gt:
            _, fh, flh = gt[h]; _, fa, fla = gt[a]
            tot = fh + fa; rec_h = (fh - flh) + fla
            for team, rec in ((h, rec_h), (a, tot - rec_h)):
                s = luck[team]
                s[0] = d * s[0] + (rec - 0.5 * tot)
                s[1] = d * s[1] + tot
    return f
f_luck = fumble_luck()

# ---------- variants, DEV CV selection ----------
def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def cv_ll(X):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fit], y[fit])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()

BASE = np.column_stack([lgt, qd])
V = {
    "V1 lean (epa+early+thfa)":      np.column_stack([lgt, qd, f_epa, f_early, f_thfa]),
    "V2 +rest+luck":                 np.column_stack([lgt, qd, f_epa, f_early, f_thfa, f_rest, f_luck]),
    "V3 +pedigree-qb":               np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck]),
    "V4 everything (+ts)":           np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts]),
}
base_cv = cv_ll(BASE)
print(f"frozen blend DEV CV {base_cv:.5f}\n")
best = None
for name, X in V.items():
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, name, X); tag = "  <-- best"
    print(f"  {name:<28} DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, name, Xw = best
print(f"\nDEV winner: {name}  ({s-base_cv:+.5f})")

# ---------- ONE TEST look ----------
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=5000).fit(BASE[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=5000).fit(Xw[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST 2016-2025 (n={n}, ONE look):")
print(f"  frozen blend    LL {llv(yt, pb).mean():.5f}")
print(f"  {name:<15} LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
json.dump({"winner": name, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_base": round(float(llv(yt, pb).mean()), 5),
           "test_new": round(float(llv(yt, pw).mean()), 5),
           "delta_ll": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "verdict": sig},
          open("data/nfl_big_test.json", "w"), indent=1)
