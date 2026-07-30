"""Low-sample fix: shrink player ratings toward REPLACEMENT level, not position mean.

The Joe Milton problem = the same wrong-prior bug the QB rep-level fix solved (SIG win):
mean-centered EB lets thin samples sit at 'average or better'. Fix: EB center =
mu - delta*sd (replacement), prior weight P raised. Small samples start low and climb.

Screens (P, delta) grid on DEV CV; winner gets ONE TEST compare vs the current model
(mean-centered ratings inside roster_quality). Keep if better (Samuel's rule).
Also re-emits the master table (data/nfl_player_ratings_2025.csv) with the fix + n_eff.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # loaders + engine defs  # noqa: S102
exec("#" + full_src.split("# pass 1")[1].split("# ---------------- screens")[0])  # noqa: S102
# ^ pass 1 (Z tables) + pass 2 (shares, f_ol/f_def/f_sk, f_rq with CURRENT mean-centered
#   rating). STATE now holds end-of-walk player states; share/pos_of/roster populated.

HAND = {"QB": [("dak", 0.50), ("negsk", 0.25), ("repa", 0.25)],
        "SK": [("eff", 0.40), ("yacr", 0.15), ("wopr", 0.45)],
        "DL": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
        "LB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
        "DB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)]}


def make_rate_rep(P, delta):
    def rate(pid):
        st = STATE.get(pid)
        if st is None:
            return None
        b = st.get("bucket")
        if b not in Z:
            return None
        tot, any_ = 0.0, False
        for k, wt in HAND[b]:
            w = st.get(k + "_w", 0.0)
            if k not in Z[b]:
                continue
            mu, sd = Z[b][k]
            center = mu - delta * sd                     # replacement level
            v = (st.get(k, 0.0) + P * center) / (w + P)  # w=0 -> starts at replacement
            tot += wt * (v - mu) / sd
            any_ = True
        return tot if any_ else None
    return rate


# roster quality with a given rate fn, using the FINAL share/roster states is WRONG
# (needs as-of walk). Rebuild the walk for both variants at once.
def build_rq(rates):
    STATE.clear()
    share2, pos2, team2 = {}, {}, {}
    roster2 = defaultdict(set)
    outs = [np.zeros(len(games)) for _ in rates]
    done2 = set()

    def rq(team, gid, rate):
        tbl = snaps.get((gid, team))
        if tbl is None:
            return 0.0
        tot = 0.0
        for pid in tbl:
            st = share2.get(pid)
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
            for j, rate in enumerate(rates):
                outs[j][i] = rq(g["home"], g["gid"], rate) - rq(g["away"], g["gid"], rate)
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
                st[0] = snapP["decay"] * st[0] + pct
                st[1] = snapP["decay"] * st[1] + 1.0
                pos2[pid] = pos
                if team2.get(pid) != team:
                    if team2.get(pid) in roster2:
                        roster2[team2.get(pid)].discard(pid)
                    team2[pid] = team
                    roster2[team].add(pid)
    return outs


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


GRID = [(15.0, 0.35), (30.0, 0.50), (60.0, 0.75)]
rates = [make_rate_rep(P, d) for (P, d) in GRID]
outs = build_rq(rates)
X_cur = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, f_rq])
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
best = None
for (P, d), fr in zip(GRID, outs):
    X = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, fr])
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, P, d, fr); tag = "  <-- best"
    print(f"  rep-prior P={P:.0f} delta={d}   DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, P, d, fr = best
Xn = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                      f_ol, f_def, f_sk, fr])
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmain TEST 2016-2025 (ONE compare):")
print(f"  current (mean-centered)         LL {llv(yt, p0).mean():.5f}")
print(f"  rep-centered P={P:.0f} d={d}        LL {llv(yt, p1).mean():.5f}")
print(f"  delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}]")
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print("  ->", "ADOPT rep-centered ratings" if better else "keep current")

# ---- corrected master table with the winning (or default 30/.5) rep prior ----
Pt, dt = (P, d) if better else (30.0, 0.50)
rate_t = make_rate_rep(Pt, dt)
ACTIVE25 = ({p for (p, s, w) in OFFW if s == 2025} | {p for (p, s, w) in DEFW if s == 2025})
PRIMARY = {"QB": "dak", "SK": "eff", "DL": "rush", "LB": "rush", "DB": "ball"}
rows_out = []
for pid in list(STATE):
    r_ = rate_t(pid)
    if r_ is None or pid not in ACTIVE25:
        continue
    st = STATE[pid]
    b_ = st.get("bucket")
    neff = st.get(PRIMARY.get(b_, "rush") + "_w", 0.0)
    if neff < 8.0:
        continue
    rows_out.append((r_, pid, b_, neff))
rows_out.sort(reverse=True)
by_b = defaultdict(list)
for r_, pid, b_, ne in rows_out:
    by_b[b_].append((r_, pid, ne))
with open("data/nfl_player_ratings_2025.csv", "w", newline="", encoding="utf-8") as fh:
    wcsv = csv.writer(fh)
    wcsv.writerow(["player", "gsis_id", "bucket", "rating_0_100", "tier", "n_eff"])
    for b_, lst in by_b.items():
        vals = np.array([x[0] for x in lst])
        for r_, pid, ne in lst:
            pct = (vals < r_).mean() * 100
            tier = ("S" if pct >= 95 else "A" if pct >= 80 else "B" if pct >= 55
                    else "C" if pct >= 25 else "D")
            wcsv.writerow([names.get(pid, pid), pid, b_, round(pct, 1), tier, round(ne, 1)])
print(f"\nrewrote data/nfl_player_ratings_2025.csv (rep prior P={Pt:.0f} delta={dt})")
qbs = [(r_, pid, ne) for r_, pid, ne in by_b["QB"]][:10]
print("\ncorrected top-10 QB:")
for i, (r_, pid, ne) in enumerate(qbs, 1):
    print(f"  {i:>2} {names.get(pid, pid):<24} n_eff {ne:.0f}")
json.dump({"grid": GRID, "winner": [P, d], "dev_cv_base": round(base_cv, 5),
           "dev_cv": round(s, 5), "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_lowsample_fix.json", "w"), indent=1)
