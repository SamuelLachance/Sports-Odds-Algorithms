"""Test a CBB-style Simple Rating System (run-differential + strength-of-schedule)
as an incremental feature over the frozen MLB model, on the locked split.

SRS is the CBB.py algorithm ported to baseball: each team's rating solves the Massey
fixed point  r_i = avg_margin_i + avg(r of opponents faced)  -- i.e. schedule-adjusted
run differential. Computed AS-OF each game date (only games strictly earlier in the
same season) so it never sees the outcome it is used to predict. Ridge-regularized so
early-season ratings shrink toward 0.

Increment protocol (same as every prior feature test): fit  y ~ base_logit + srs_diff
on DEV (2000-2015), evaluate log loss on TEST (2016-2024 ex-2020), paired clustered
bootstrap on the per-game log-loss delta vs the base-only model.
"""
import csv, glob, os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

RIDGE = 1.0          # shrink early-season ratings toward 0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 1. parse gamelogs -> per-game (season, iso_date, home, away, home_margin) ----
games = []
for fp in sorted(glob.glob(os.path.join(HERE, "data", "retrosheet", "gl*.txt"))):
    for row in csv.reader(open(fp)):
        if len(row) < 11:
            continue
        d = row[0]; iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        season = int(d[:4]); away = row[3]; home = row[6]
        try:
            vs, hs = int(row[9]), int(row[10])
        except ValueError:
            continue
        games.append((season, iso, home, away, hs - vs))

# ---- 2. as-of SRS per (season, date) via ridge Massey solve ----
by_season = defaultdict(list)
for g in games:
    by_season[g[0]].append(g)

srs_lookup = {}          # (season, date, team) -> rating as-of that date
for season, gl in by_season.items():
    gl.sort(key=lambda x: x[1])
    teams = sorted({t for _, _, h, a, _ in gl for t in (h, a)})
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    N = np.zeros(n)                 # games played
    M = np.zeros((n, n))            # -A off-diagonal, games-count on diagonal built below
    A = np.zeros((n, n))            # pairwise game counts
    marg = np.zeros(n)             # cumulative margin (team perspective)
    dates = sorted({d for _, d, _, _, _ in gl})
    date_games = defaultdict(list)
    for g in gl:
        date_games[g[1]].append(g)
    for d in dates:
        # snapshot BEFORE adding today's games -> as-of ratings exclude date d
        diagN = np.diag(N)
        Mmat = diagN - A + RIDGE * np.eye(n)
        r = np.linalg.solve(Mmat, marg)
        for t, i in ti.items():
            srs_lookup[(season, d, t)] = r[i]
        # now fold today's games into the cumulative state
        for _, _, h, a, hm in date_games[d]:
            ih, ia = ti[h], ti[a]
            N[ih] += 1; N[ia] += 1
            A[ih, ia] += 1; A[ia, ih] += 1
            marg[ih] += hm; marg[ia] -= hm

# ---- 3. attach srs_diff to model_probs ----
rows = []
matched = 0
for r in csv.DictReader(open(os.path.join(HERE, "data", "model_probs.csv"))):
    s = int(r["season"])
    if r["recbp_p"] == "" or r["full_p"] == "":
        continue
    sh = srs_lookup.get((s, r["date"], r["home"]))
    sa = srs_lookup.get((s, r["date"], r["away"]))
    if sh is None or sa is None:
        continue
    matched += 1
    rows.append((s, float(r["recbp_p"]), float(r["full_p"]), sh - sa, int(r["y"])))

S = np.array([x[0] for x in rows])
prec = np.array([x[1] for x in rows]); pful = np.array([x[2] for x in rows])
srs = np.array([x[3] for x in rows]); y = np.array([x[4] for x in rows])
print(f"matched {matched} games with as-of SRS  ({matched/ max(1,len(rows)):.0%} of usable model rows)")
print(f"SRS diff: mean {srs.mean():+.3f}  sd {srs.std():.3f}  (run units, home-away)")

eps = 1e-9
def lg(p): p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def LL(p): p = np.clip(p, eps, 1 - eps); return -(y2 * np.log(p) + (1 - y2) * np.log(1 - p))

dev = S <= 2015
test = (S >= 2016) & (S != 2020)
srs_z = (srs - srs[dev].mean()) / srs[dev].std()   # standardize on DEV only

rng = np.random.default_rng(7)
def evaluate(base_p, name):
    global y2
    Xb = lg(base_p).reshape(-1, 1)
    Xa = np.column_stack([lg(base_p), srs_z])
    base = LogisticRegression(C=1e6).fit(Xb[dev], y[dev])
    aug = LogisticRegression(C=1e6).fit(Xa[dev], y[dev])
    pb = base.predict_proba(Xb[test])[:, 1]
    pa = aug.predict_proba(Xa[test])[:, 1]
    y2 = y[test]
    llb, lla = LL(pb).mean(), LL(pa).mean()
    delta = LL(pb) - LL(pa)                       # >0 means SRS helps
    n = len(delta)
    bs = delta[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    coef = aug.coef_[0][1]
    sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
    print(f"\n{name} base (n_test={n})")
    print(f"  base-only LL         {llb:.5f}")
    print(f"  base + SRS feature   {lla:.5f}   (delta {llb-lla:+.5f})")
    print(f"  SRS coef in stack    {coef:+.4f}")
    print(f"  bootstrap 95% CI on delta LL [{lo:+.5f}, {hi:+.5f}]  -> {sig}")

evaluate(prec, "recbp (no-lineup)")
evaluate(pful, "full (with-lineup)")
