"""MLB-grade NFL player ratings: per-play TrueSkill duels vs defense units.

Exactly the MLB architecture (same TrueSkill1v1 class): every dropback/target/carry
is a duel — offensive player vs opposing DEFENSE UNIT entity, win = EPA > 0.
Opponent-adjusted by construction; (mu, sigma) native; display = mu - 2*sigma
(conservative, kills the Milton problem structurally); season boundary regress+widen.

Outputs: top-10 QB conservative leaderboard + master offense table
data/nfl_player_ts_2025.csv. Model screen: offense side of roster_quality replaced
by TS ratings (defenders keep the weekly composite). ONE TEST compare, keep if better.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # loaders + base features + engine defs  # noqa: S102
exec("#" + full_src.split("# pass 1")[1].split("# ---------------- screens")[0])  # noqa: S102
# gives current f_rq + f_ol/f_def/f_sk + STATE (weekly composites) + Z + share walk end-state

import sys
sys.path.insert(0, ".")
from mlbwp.trueskill import TrueSkill1v1  # noqa: E402

MU0, SIG0 = 25.0, 25.0 / 3.0

# ---- duel plays grouped per game ----
# per-role median EPA thresholds (DEV era) so pass and run duels are both ~50/50
_raw = []
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        _raw.append((r[ix["game_id"]], PBP_FIX.get(r[ix["defteam"]], r[ix["defteam"]]),
                     float(r[ix["epa"]]), r[ix["passer_player_id"]],
                     r[ix["rusher_player_id"]], r[ix["receiver_player_id"]]))
_dev_pass = [e for (g, _, e, pa, ru, re) in _raw if pa and int(g[:4]) <= 2015]
_dev_run = [e for (g, _, e, pa, ru, re) in _raw if ru and int(g[:4]) <= 2015]
MED_PASS = float(np.median(_dev_pass))
MED_RUN = float(np.median(_dev_run))
print(f"role medians (DEV): pass {MED_PASS:+.3f}  run {MED_RUN:+.3f}")
duels = defaultdict(list)          # gid -> [(off_pid, def_unit, win), ...]
for (gid, dt, epa, pa, ru, re) in _raw:
    dunit = dt + "_DEF"
    if pa:
        duels[gid].append((pa, dunit, epa > MED_PASS))
    if re:
        duels[gid].append((re, dunit, epa > MED_PASS))
    if ru:
        duels[gid].append((ru, dunit, epa > MED_RUN))
del _raw
print(f"duel games: {len(duels):,}  duels: {sum(len(v) for v in duels.values()):,}")

# position map for offensive players (from weekly stats)
posmap = {}
for (pid, s, w), d in OFFW.items():
    posmap[pid] = d["pos"]

TSP = dict(beta=14.0, tau=0.08, widen=2.0, mu_reg=0.5)   # unit-TS-informed defaults


def run_player_ts(beta, tau, widen, mu_reg, want_feature=True):
    ts = TrueSkill1v1(beta=beta, tau=tau, mu0=MU0, sig0=SIG0)
    f = np.zeros(len(games))
    prev = None
    # share walk replicated for as-of roster quality (offense TS + defender composite)
    share3, pos3, team3 = {}, {}, {}
    roster3 = defaultdict(set)

    def rqual(team, gid):
        tbl = snaps.get((gid, team))
        if tbl is None:
            return 0.0
        tot = 0.0
        for pid in tbl:
            st = share3.get(pid)
            if not st or st[1] <= 0:
                continue
            sh = st[0] / st[1]
            p = pos3.get(pid, "")
            g_ = pfr2gsis.get(pid)
            if p in DEFPOS:
                r_ = rate_of(g_) if g_ else None        # weekly composite (defenders)
                if r_ is not None:
                    tot += sh * max(-2.0, min(2.0, r_))
            else:
                if g_ and ts.n.get(g_, 0) >= 30:        # evidence floor in duels
                    z = (ts.mu[g_] - MU0) / SIG0
                    tot += sh * max(-2.0, min(2.0, z * 2.0))
        return tot

    done3 = set()
    for i, g in enumerate(games):
        s, w = g["season"], g["week"]
        if prev is not None and s != prev:
            for k in list(ts.mu.keys()):
                ts.mu[k] = MU0 + (ts.mu[k] - MU0) * (1 - mu_reg * (0.4 if k.endswith("_DEF") else 1.0))
                ts.var[k] = min(ts.var[k] + widen ** 2, SIG0 * SIG0)
        prev = s
        if want_feature and s >= 2013:
            f[i] = rqual(g["home"], g["gid"]) - rqual(g["away"], g["gid"])
        for (opid, dunit, win) in duels.get(g["gid"], ()):
            if win:
                ts.update(opid, dunit)
            else:
                ts.update(dunit, opid)
        for team in (g["home"], g["away"]):
            for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
                if (pid, s, w) in done3:
                    continue
                done3.add((pid, s, w))
                apply_week(pid, s, w)
        for team in (g["home"], g["away"]):
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEFPOS else op
                st = share3.setdefault(pid, [0.0, 0.0])
                st[0] = snapP["decay"] * st[0] + pct
                st[1] = snapP["decay"] * st[1] + 1.0
                pos3[pid] = pos
                if team3.get(pid) != team:
                    if team3.get(pid) in roster3:
                        roster3[team3.get(pid)].discard(pid)
                    team3[pid] = team
                    roster3[team].add(pid)
    return ts, f


STATE.clear()
ts, f_tsq = run_player_ts(**TSP)

# ---- conservative leaderboard (mu - 2*sigma), 2025-active offense ----
ACT = ({p for (p, s, w) in OFFW if s == 2025})
rows_out = []
for pid in ts.mu:
    if pid.endswith("_DEF") or pid not in ACT:
        continue
    n = ts.n.get(pid, 0)
    if n < 30:
        continue
    cons = ts.mu[pid] - 2.0 * np.sqrt(ts.var[pid])
    rows_out.append((cons, ts.mu[pid], np.sqrt(ts.var[pid]), n, pid, posmap.get(pid, "?")))
rows_out.sort(reverse=True)
with open("data/nfl_player_ts_2025.csv", "w", newline="", encoding="utf-8") as fh:
    wcsv = csv.writer(fh)
    wcsv.writerow(["player", "gsis_id", "pos", "mu", "sigma", "conservative", "n_duels"])
    for cons, mu, sg, n, pid, pos in rows_out:
        wcsv.writerow([names.get(pid, pid), pid, pos, round(mu, 2), round(sg, 2),
                       round(cons, 2), n])
qbs = [(c, m, sg, n, pid) for c, m, sg, n, pid, pos in rows_out if pos == "QB" and n >= 500][:10]
print("\nTOP-10 QB (conservative mu-2sigma, opponent-adjusted per-play TrueSkill):")
for i, (c, m, sg, n, pid) in enumerate(qbs, 1):
    print(f"  {i:>2} {names.get(pid, pid):<24} cons {c:5.2f}  (mu {m:5.2f} sigma {sg:4.2f}, "
          f"n {n:,})")

# ---- model screen ----
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

X_cur = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, f_rq])
X_new = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, f_tsq])
print(f"\nDEV CV: current {cv_ll(X_cur):.5f}  ts-roster {cv_ll(X_new):.5f}")
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(X_new[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(X_new[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"main TEST: current {llv(yt, p0).mean():.5f} -> ts-roster {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"params": TSP, "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_player_ts.json", "w"), indent=1)
