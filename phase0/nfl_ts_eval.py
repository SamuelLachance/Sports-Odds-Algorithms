"""NFL TrueSkill: per-play offense-unit vs defense-unit ratings as a blend feature.

The MLB-PA analog: every pass/run snap is a 1v1 duel — offense unit vs defense unit,
success = EPA > 0. Reuses mlbwp.trueskill.TrueSkill1v1 verbatim. Entities "TEAM_OFF"/
"TEAM_DEF"; game feature (as-of, before the game's plays update anything):
    ts_edge = (mu[homeOFF] - mu[awayDEF]) - (mu[awayOFF] - mu[homeDEF])
Season boundary: var += widen^2, mu regressed toward mu0 (roster turnover).

Protocol: TS hyperparams (beta, tau, widen, mu_regress) tuned by 5-fold CV of the
stacked blend on DEV ONLY (base = elo_logit + qb_diff); winner gets ONE TEST look,
paired bootstrap vs the base blend. Base model to beat: 0.63059.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.trueskill import TrueSkill1v1  # noqa: E402
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo  # noqa: E402
from nfl_elo import FRANCHISE  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

MU0, SIG0 = 25.0, 25.0 / 3.0
PBP_FIX = {"JAC": "JAX", "WSH": "WAS", **FRANCHISE}

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
P, lg = rep["qb_elo"]["params"], rep["qb_elo"]["lg_rate"]

# ---- attach game_id to each blend game (same chronological order) ----
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
assert len(raw) == len(games)
for g, r in zip(games, raw):
    g["gid"] = r["game_id"]

# ---- plays grouped per game: [(off, def, success), ...] ----
plays = defaultdict(list)
bad_codes = set()
for r in csv.reader(open("data/nfl_plays.csv")):
    if r[0] == "game_id":
        continue
    gid, _, pos, dfn, epa = r
    pos = PBP_FIX.get(pos, pos); dfn = PBP_FIX.get(dfn, dfn)
    plays[gid].append((pos, dfn, float(epa) > 0.0))
have = sum(1 for g in games if g["gid"] in plays)
print(f"plays for {have}/{len(games)} games "
      f"({sum(len(v) for v in plays.values()):,} plays)")

# ---- frozen base signals (identical to rest eval) ----
pe, y = run_elo(games, score_from=1999, **base["params"])
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, decay=P["decay"],
          prior_db=P["prior_db"], season_decay=P["season_decay"], lg_rate=lg)
qb_diff, prev = [], None
for g in games:
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qb_diff.append(m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"]))
    m.predict(g)
    m.update(g, 0.5, qbw)
qb_diff = np.array(qb_diff)
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)
BASE = np.column_stack([lgt, qb_diff])


def ts_edges(beta, tau, widen, mu_reg):
    """Walk all games chronologically; return as-of ts_edge per game."""
    ts = TrueSkill1v1(beta=beta, tau=tau, mu0=MU0, sig0=SIG0)
    out = np.zeros(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for k in list(ts.mu.keys()):
                ts.mu[k] = MU0 + (ts.mu[k] - MU0) * (1.0 - mu_reg)
                ts.var[k] = min(ts.var[k] + widen * widen, SIG0 * SIG0)
        prev = g["season"]
        h, a = g["home"], g["away"]
        out[i] = ((ts.mu[h + "_OFF"] - ts.mu[a + "_DEF"])
                  - (ts.mu[a + "_OFF"] - ts.mu[h + "_DEF"]))
        for pos, dfn, success in plays.get(g["gid"], ()):
            if success:
                ts.update(pos + "_OFF", dfn + "_DEF")
            else:
                ts.update(dfn + "_DEF", pos + "_OFF")
    return out


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X):
    idx = np.where(dev)[0]
    scores = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        scores[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return scores.mean()


base_cv = cv_ll(BASE)
print(f"base blend DEV CV {base_cv:.5f}")

rng = np.random.default_rng(20260723)
N = 40
t0 = time.time()
best = None
for i in range(N):
    cand = dict(
        beta=float(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),
        tau=float(np.exp(rng.uniform(np.log(0.001), np.log(0.3)))),
        widen=float(rng.uniform(0.0, 6.0)),
        mu_reg=float(rng.uniform(0.0, 0.6)),
    )
    e = ts_edges(**cand)
    s = cv_ll(np.column_stack([BASE, e]))
    if best is None or s < best[0]:
        best = (s, cand, e)
    if (i + 1) % 10 == 0:
        print(f"  ...{i+1}/{N}  best CV {best[0]:.5f} ({best[0]-base_cv:+.5f})"
              f"  ({time.time()-t0:.0f}s)", flush=True)

s, cand, e = best
print(f"\nbest TS params {'{'}{', '.join(f'{k}={v:.4f}' for k, v in cand.items())}{'}'}")
print(f"DEV CV: base {base_cv:.5f} -> +ts {s:.5f} ({s-base_cv:+.5f})")
if s >= base_cv:
    print("no DEV improvement -> rejected on DEV, TEST not touched")
    sys.exit(0)

# ---- ONE TEST look ----
X = np.column_stack([BASE, e])
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=3000).fit(BASE[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
pw = cw.predict_proba(X[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)
rng2 = np.random.default_rng(7)
n = len(d)
bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST 2016-2025 (n={n}, ONE look):")
print(f"  base blend LL {llv(yt, pb).mean():.5f}")
print(f"  + ts_edge  LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
print(f"  ts coef {cw.coef_[0][2]:+.5f}")
json.dump({"ts_params": cand, "dev_cv_base": round(base_cv, 5), "dev_cv_ts": round(s, 5),
           "test_base": round(float(llv(yt, pb).mean()), 5),
           "test_ts": round(float(llv(yt, pw).mean()), 5),
           "delta": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "verdict": sig},
          open("data/nfl_ts_report.json", "w"), indent=1)
