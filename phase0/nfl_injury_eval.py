"""Injury-report features: the forward-looking 'playing hurt' signal (2009+).

Snap absence already knows who's OUT. The report adds what absence can't see:
  F1 qsh   share-sum of ACTIVES who carried Questionable/Doubtful into the game
  F2 dnp   share-sum of ACTIVES whose practice week was DNP or Limited
  F3 load  raw injury-report count (team health burden)
All diffs home-away, zeros pre-2009. Main protocol; DEV CV selects; ONE TEST look.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # loaders + tuned F dict + cv_ll/llv  # noqa: S102

# ---- injury lookup: (season, week, franchise) -> {gsis: (report, practice)} ----
inj = defaultdict(dict)
for yr in range(2009, 2027):
    try:
        fh = open(f"data/inj_{yr}.csv", encoding="utf-8")
    except FileNotFoundError:
        continue
    for r in csv.DictReader(fh):
        t = PBP_FIX.get(r["team"], r["team"])
        try:
            key = (int(r["season"]), int(r["week"]), t)
        except ValueError:
            continue
        inj[key][r["gsis_id"]] = (r.get("report_status") or "",
                                  r.get("practice_status") or "")
print(f"injury tables: {len(inj):,} team-weeks")

QD = {"Questionable", "Doubtful"}
BADPRAC = ("Did Not Participate", "Limited Participation")

# ---- lean walk: share EWMAs + per-game injury features ----
share5, pos5 = {}, {}
f_qsh = np.zeros(len(games)); f_dnp = np.zeros(len(games)); f_load = np.zeros(len(games))
for i, g in enumerate(games):
    s, w = g["season"], g["week"]
    if s >= 2009:
        for side, team in ((1, g["home"]), (-1, g["away"])):
            tbl = snaps.get((g["gid"], team))
            it = inj.get((s, w, team), {})
            f_load[i] += side * len(it)
            if tbl is None or not it:
                continue
            for pid in tbl:
                st = share5.get(pid)
                if not st or st[1] <= 0:
                    continue
                sh = st[0] / st[1]
                g_ = pfr2gsis.get(pid)
                rec = it.get(g_) if g_ else None
                if rec is None:
                    continue
                rs, ps = rec
                if rs in QD:
                    f_qsh[i] += side * sh
                if any(ps.startswith(b) for b in BADPRAC):
                    f_dnp[i] += side * sh
    for team in (g["home"], g["away"]):
        tbl = snaps.get((g["gid"], team))
        if not tbl:
            continue
        for pid, (pos, op, dp) in tbl.items():
            pct = dp if pos in DEFPOS else op
            st = share5.setdefault(pid, [0.0, 0.0])
            st[0] = snapP["decay"] * st[0] + pct
            st[1] = snapP["decay"] * st[1] + 1.0
            pos5[pid] = pos

nz = int((np.abs(f_qsh) > 0).sum())
print(f"games with playing-hurt signal: {nz} | mean|qsh| {np.abs(f_qsh[f_qsh!=0]).mean():.2f}")

X_cur = X_of(F)
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
CAND = {
    "qsh": [f_qsh], "dnp": [f_dnp], "load": [f_load],
    "qsh+dnp": [f_qsh, f_dnp], "all": [f_qsh, f_dnp, f_load],
}
best = None
for name, cols in CAND.items():
    X = np.column_stack([X_cur] + cols)
    sc = cv_ll(X)
    tag = ""
    if best is None or sc < best[0]:
        best = (sc, name, X); tag = "  <-- best"
    print(f"  + {name:<8} DEV CV {sc:.5f} ({sc-base_cv:+.5f}){tag}")

sc, name, Xn = best
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
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"\nmain TEST ({name}): current {llv(yt, p0).mean():.5f} -> {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"variant": name, "dev_cv_base": round(base_cv, 5), "dev_cv": round(sc, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_injury.json", "w"), indent=1)

# ---- secondary protocol: modern-tag era only (fit 2016-2020, test 2021-2025) ----
dev2 = (seasons >= 2016) & (seasons <= 2020)
test2 = (seasons >= 2021) & (seasons <= 2025)
Xq = np.column_stack([X_cur, f_qsh])
f2m = dev2 & (y != 0.5)
b0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[f2m], y[f2m])
b1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xq[f2m], y[f2m])
q0 = b0.predict_proba(X_cur[test2])[:, 1]
q1 = b1.predict_proba(Xq[test2])[:, 1]
yt2 = y[test2]
d2 = llv(yt2, q0) - llv(yt2, q1)
bs2 = d2[np.random.default_rng(7).integers(0, len(d2), size=(10000, len(d2)))].mean(axis=1)
lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
print(f"secondary (modern tag, fit 16-20, test 21-25 n={len(yt2)}):")
print(f"  base {llv(yt2, q0).mean():.5f} -> +qsh {llv(yt2, q1).mean():.5f}"
      f"  (delta {d2.mean():+.5f}  CI [{lo2:+.5f}, {hi2:+.5f}])")
