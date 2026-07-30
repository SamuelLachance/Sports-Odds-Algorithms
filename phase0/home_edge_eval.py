"""Does a TEAM-SPECIFIC home-field edge add over the full model (0.67322)?

The model uses one global home-field constant; but home edge is not uniform (Coors
altitude is the famous outlier). For each team we accumulate its home-game residual
vs the model's prediction (which already includes the global HFA) -- how much it
over/under-performs at home beyond what the model expects -- and its road residual,
both empirical-Bayes shrunk toward 0 and read AS-OF (career, leak-safe). Feature:
home_resid(home team) - road_resid(away team), added to the full model.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.siera import SieraRater  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
PRIOR = 150.0        # games of EB shrink toward the global HFA (residual mean 0)


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    games = load_games()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, FZ.load_lineups(), lgiso)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = json.load(open("mlbwp/artifacts/blend.json"))

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    hres = defaultdict(float); hn = defaultdict(int)   # home residual sum, count
    rres = defaultdict(float); rn = defaultdict(int)   # road residual sum, count
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        h, a = g["home"], g["away"]
        he = hres[h] / (hn[h] + PRIOR)       # home team's home edge (as-of)
        re = rres[a] / (rn[a] + PRIOR)        # away team's road edge (as-of)
        p = m.predict(h, a, g["home_sp"], g["away_sp"])
        sd = sr.edge(g["home_sp"], g["away_sp"])
        gid = g["game_id"]
        if gid in ts and pdiff.get(gid) is not None:
            rows.append((g["season"], p, g["y"], ts[gid], bpd[gid], pdiff[gid], sd, he - re))

        # advance: residual vs the model's prediction p (incl. global HFA)
        resid = g["y"] - p                    # + = home won more than predicted
        hres[h] += resid; hn[h] += 1
        rres[a] += -resid; rn[a] += 1
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)

    aa = np.array(rows, float)
    seas = aa[:, 0]; lf = logit(aa[:, 1]); y = aa[:, 2]
    dm = np.isin(seas, list(DEV)); tm = np.isin(seas, TEST)

    def z(c, mu, sd):
        return (aa[:, c] - B[mu]) / B[sd]
    tsz = z(3, "ts_mu", "ts_sd"); bpz = z(4, "bp_mu", "bp_sd")
    pwz = z(5, "pw_mu", "pw_sd"); siz = z(6, "si_mu", "si_sd")
    he = aa[:, 7]; hez = (he - he[dm].mean()) / (he[dm].std() or 1.0)
    yt = y[tm]

    def fit(cols):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        return mdl, mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
    base = [lf, tsz, bpz, pwz, siz]
    _, p_full = fit(base)
    mdl, p_a = fit(base + [hez])
    d, lo, hi = boot(ll_vec(p_full, yt), ll_vec(p_a, yt))
    sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
    # show the biggest home edges for face validity
    edges = sorted(((t, hres[t] / (hn[t] + PRIOR)) for t in hres if hn[t] > 500), key=lambda x: -x[1])[:5]
    print(f"games: {len(y):,}   top home edges (resid/game): " +
          ", ".join(f"{t} {e:+.3f}" for t, e in edges))
    print(f"\n=== LOCKED TEST  n={len(yt):,}  full LL={ll(p_full, yt):.5f} (=0.67322) ===")
    print(f"  + team home-field edge   LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig}  coef={mdl.coef_[0][-1]:+.4f}")


if __name__ == "__main__":
    main()
