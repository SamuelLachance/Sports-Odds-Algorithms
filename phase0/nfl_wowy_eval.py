"""WOWY player value: the counterfactual 'could the team survive without him?'

For each team-game (2013+), two unit rows:
  OFF: y = off EPA/play (game) - as-of off rating   DEF: y = EPA allowed - as-of def rating
Design: each absent expected starter j contributes -share_j (own off row / opp def hurt
mirrored) with ONE coefficient impact_j >= his unit's drop per full-time absence.
Two-level, walk-forward (refit each season on all PRIOR seasons — leak-free):
  level 1: position-bucket coefficients (QB RB WR TE OL DL LB DB) — the VALUE TABLE
  level 2: individual deviations (players with >=4 prior absences), ridge-shrunk
Feature: value-weighted absence diff (non-QB; QB channel already modeled).
One TEST compare vs the current 12-feature model; adopt if better (Samuel's rule).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # loaders: games+gid+week, snaps, crosswalk, agg, epaP  # noqa: S102

B8 = {}
for p in ("QB",):
    B8[p] = "QB"
for p in ("RB", "FB", "HB"):
    B8[p] = "RB"
B8["WR"] = "WR"; B8["TE"] = "TE"
for p in OLPOS:
    B8[p] = "OL"
for p in ("DE", "DT", "NT", "EDGE", "DL"):
    B8[p] = "DL"
for p in ("LB", "ILB", "OLB", "MLB"):
    B8[p] = "LB"
for p in ("CB", "S", "FS", "SS", "DB"):
    B8[p] = "DB"
BUCKETS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]

# ---- as-of unit EPA ratings (same params as the shipped epa feature) ----
off_st = defaultdict(lambda: [0.0, 0.0]); dfa_st = defaultdict(lambda: [0.0, 0.0])
def unit_rate(st):
    return (st[0] + epaP["prior_n"] * LG_EPA) / (st[1] + epaP["prior_n"])

# ---- absence machinery (share EWMAs, rosters) ----
share, pos_of, team_of = {}, {}, {}
roster = defaultdict(set)

def absent_list(team, gid):
    tbl = snaps.get((gid, team))
    if tbl is None:
        return []
    out = []
    for pid in roster.get(team, ()):
        st = share.get(pid)
        if not st or st[1] <= 0:
            continue
        sh = st[0] / st[1]
        if sh < snapP["thresh"] or pid in tbl:
            continue
        b = B8.get(pos_of.get(pid, ""), None)
        if b:
            out.append((pid, b, sh))
    return out

# ---- walk: collect training rows + compute the feature with as-of coefs ----
train_rows = []                      # (season, y, [(pid,bucket,weight)])  weight signed
coef_pos = {}                        # bucket -> impact (unit EPA/play per full absence)
coef_dev = {}                        # pid -> deviation
prior_abs = defaultdict(int)         # pid -> absence instances seen so far
fit_season = None
f_val = np.zeros(len(games))

def refit(upto_season):
    rows = [r for r in train_rows if r[0] < upto_season]
    if len(rows) < 300:
        return {}, {}
    players = sorted({pid for _, _, terms in rows for (pid, b, w) in terms
                      if prior_abs_at_fit.get(pid, 0) >= 4})
    pix = {p: len(BUCKETS) + j for j, p in enumerate(players)}
    ncol = len(BUCKETS) + len(players)
    A = np.zeros((len(rows) + len(players), ncol))
    Y = np.zeros(len(rows) + len(players))
    bix = {b: j for j, b in enumerate(BUCKETS)}
    for n, (_, yv, terms) in enumerate(rows):
        Y[n] = yv
        for (pid, b, w) in terms:
            A[n, bix[b]] += w
            if pid in pix:
                A[n, pix[pid]] += w
    lam = 3.0                          # ridge on player deviations only
    for j, p in enumerate(players):
        A[len(rows) + j, pix[p]] = lam
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return ({b: float(sol[bix[b]]) for b in BUCKETS},
            {p: float(sol[pix[p]]) for p in players})

prior_abs_at_fit = {}
for i, g in enumerate(games):
    s = g["season"]
    if s >= 2014 and s != fit_season:
        prior_abs_at_fit = dict(prior_abs)
        coef_pos, coef_dev = refit(s)
        fit_season = s
    h, a = g["home"], g["away"]
    gid = g["gid"]
    if s >= 2013 and coef_pos:
        # feature (pre-game info only): value of absent expected starters, non-QB
        def team_cost(team):
            c = 0.0
            for (pid, b, sh) in absent_list(team, gid):
                if b == "QB":
                    continue
                c += sh * max(0.0, coef_pos.get(b, 0.0) + coef_dev.get(pid, 0.0))
            return c
        f_val[i] = team_cost(h) - team_cost(a)
    if s >= 2013:
        ah, aa = absent_list(h, gid), absent_list(a, gid)
        th, ta = agg.get(gid, {}).get(h), agg.get(gid, {}).get(a)
        if th and ta and th[1] > 10 and ta[1] > 10:
            for team, opp, mine, theirs, tm_agg in ((h, a, ah, aa, th), (a, h, aa, ah, ta)):
                y_off = tm_agg[0] / tm_agg[1] - unit_rate(off_st[team])
                y_def = tm_agg[0] / tm_agg[1] - unit_rate(dfa_st[opp]) if False else None
                terms = ([(pid, b, -sh) for (pid, b, sh) in mine if b in ("QB", "RB", "WR", "TE", "OL")]
                         + [(pid, b, +sh) for (pid, b, sh) in theirs if b in ("DL", "LB", "DB")])
                if terms:
                    train_rows.append((s, y_off, terms))
            for (pid, b, sh) in ah + aa:
                prior_abs[pid] += 1
    # updates AFTER: unit EPA states + share/rosters
    tm = agg.get(gid)
    if tm:
        for t_off, opp in ((h, a), (a, h)):
            sm, n_ = tm.get(t_off, (0.0, 0))
            o = off_st[t_off]; o[0] = epaP["decay"] * o[0] + sm; o[1] = epaP["decay"] * o[1] + n_
            d = dfa_st[opp]; d[0] = epaP["decay"] * d[0] + sm; d[1] = epaP["decay"] * d[1] + n_
    if g["season"] != games[i - 1]["season"] if i else False:
        pass
    for team in (h, a):
        tbl = snaps.get((gid, team))
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
    # season boundary decay for unit states
    if i + 1 < len(games) and games[i + 1]["season"] != s:
        for stx in list(off_st.values()) + list(dfa_st.values()):
            stx[0] *= (1 - epaP["season_decay"]); stx[1] *= (1 - epaP["season_decay"])

# ---- THE VALUE TABLE (final fit, all data) ----
prior_abs_at_fit = dict(prior_abs)
coef_pos_f, coef_dev_f = refit(9999)
print("POSITION VALUE TABLE (unit EPA/play lost when a full-time starter is out;")
print("x ~62 plays = points/game):")
for b in BUCKETS:
    v = coef_pos_f.get(b, 0.0)
    print(f"  {b:<4} {v:+.4f} EPA/play  ~ {v*62:+.2f} pts/game")
top = sorted(coef_dev_f.items(), key=lambda kv: -(coef_pos_f.get(B8.get(pos_of.get(kv[0], ''), ''), 0) + kv[1]))[:12]
print("\ntop individual MVP-impact (pos value + personal deviation):")
for pid, dv in top:
    b = B8.get(pos_of.get(pid, ""), "?")
    tot = coef_pos_f.get(b, 0) + dv
    print(f"  {names.get(pfr2gsis.get(pid, ''), pid):<24} {b:<3} {tot*62:+.2f} pts/game "
          f"(n_abs {prior_abs.get(pid, 0)})")

# ---- screen: value-weighted absence into the model ----

# rebuild current model features via the pass-2 section is heavy; reuse spec approach:
# current 12-feature X = big-test 8 + snap absence 3 + roster quality — rebuild roster
# quality requires the rating walk; instead compare on the 11-feature + f_rq from spec?
# Pragmatic: load the shipped spec's DEV-fit and rebuild X12 exactly as adopted.
print("\n(building current-model features for the compare...)")
exec("#" + full_src.split("# pass 1")[1].split("# ---------------- screens")[0])  # noqa: S102
# the above re-runs pass1+pass2 of the player system -> provides f_rq (roster quality)
# and fresh f_ol/f_def/f_sk consistent with the shipped build
X12 = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                       f_ol, f_def, f_sk, f_rq])
X13 = np.column_stack([X12, f_val])

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X12[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(X13[fitm], y[fitm])
p0 = c0.predict_proba(X12[test])[:, 1]
p1 = c1.predict_proba(X13[test])[:, 1]
yt = y[test]
d = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmain TEST 2016-2025 (ONE compare):")
print(f"  current model (12f)      LL {llv(yt, p0).mean():.5f}")
print(f"  + WOWY value absence     LL {llv(yt, p1).mean():.5f}")
print(f"  delta {d.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")
json.dump({"pos_values_epa_play": {b: round(coef_pos_f.get(b, 0), 5) for b in BUCKETS},
           "test_current": round(float(llv(yt, p0).mean()), 5),
           "test_wowy": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)]},
          open("data/nfl_wowy.json", "w"), indent=1)
