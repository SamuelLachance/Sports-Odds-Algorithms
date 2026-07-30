"""Does late-season stakes / tanking add over the full model (0.67322)?

Orthogonal-to-talent hypothesis: a team late in the season that is out of the race
rests regulars and coasts (low effort); a team fighting for a playoff spot plays
all-out. We walk standings as-of each game, find the playoff bubble (win% of the
Nth-best team in the league; N = 4 pre-2012, 5 through 2021, 6 from 2022), and per
team derive:
  intensity  late-season 'in a race' bump (peaks at the bubble)
  out        late + clearly below the bubble (tanking)
  clinch     late + clearly above the bubble (coasting)
Home-minus-away differentials added to [logit_xfip + ts + bp + power + siera].
"""

from __future__ import annotations

import csv
import glob
import math
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


def read_leagues():
    lg, total = {}, defaultdict(int)
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        yr = int(path[-8:-4])
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            try:
                vr, hr = int(r[9]), int(r[10])
            except ValueError:
                continue
            if vr == hr:
                continue
            gid = r[6] + r[0] + r[1]
            lg[gid] = (r[7], r[4])            # home league, away league
            total[(yr, r[6])] += 1
            total[(yr, r[3])] += 1
    return lg, total


def spots(season):
    return 4 if season <= 2011 else (6 if season >= 2022 else 5)


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
    lg_map, total = read_leagues()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, FZ.load_lineups(), lgiso)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = __import__("json").load(open("mlbwp/artifacts/blend.json"))

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    W = defaultdict(int); L = defaultdict(int); GP = defaultdict(int); league_of = {}
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        s = g["season"]; gid = g["game_id"]
        hlg, alg = lg_map.get(gid, (None, None))
        league_of[(s, g["home"])] = hlg
        league_of[(s, g["away"])] = alg

        def bubble(lgc):
            wps = [W[k] / (W[k] + L[k]) for k, l in league_of.items()
                   if k[0] == s and l == lgc and (W[k] + L[k]) > 0]
            wps.sort(reverse=True)
            N = spots(s)
            return wps[N - 1] if len(wps) >= N else (wps[-1] if wps else 0.5)

        def feat(team, lgc):
            w, ll_, gp = W[(s, team)], L[(s, team)], GP[(s, team)]
            if w + ll_ == 0:
                return 0.0, 0.0, 0.0
            wp = w / (w + ll_)
            margin = wp - bubble(lgc)
            gr = max(0, total[(s, team)] - gp)
            late = max(0.0, 1.0 - gr / 45.0)
            intensity = late * math.exp(-((margin / 0.05) ** 2))
            out = late if margin < -0.05 else 0.0
            clinch = late if margin > 0.05 else 0.0
            return intensity, out, clinch
        hi, ho, hc = feat(g["home"], hlg)
        ai, ao, ac = feat(g["away"], alg)

        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        sd = sr.edge(g["home_sp"], g["away_sp"])
        if gid in ts and pdiff.get(gid) is not None:
            rows.append((s, p, g["y"], ts[gid], bpd[gid], pdiff[gid], sd,
                         hi - ai, ao - ho, ac - hc))

        # advance
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)
        hw = int(g["y"] == 1.0)
        W[(s, g["home"])] += hw; L[(s, g["home"])] += 1 - hw; GP[(s, g["home"])] += 1
        W[(s, g["away"])] += 1 - hw; L[(s, g["away"])] += hw; GP[(s, g["away"])] += 1

    a = np.array(rows, float)
    seas = a[:, 0]; lf = logit(a[:, 1]); y = a[:, 2]
    dm = np.isin(seas, list(DEV)); tm = np.isin(seas, TEST)

    def z(col, mu, sd):
        v = a[:, col]
        return (v - B[mu]) / B[sd]

    def zc(col):
        v = a[:, col]
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    tsz = z(3, "ts_mu", "ts_sd"); bpz = z(4, "bp_mu", "bp_sd")
    pwz = z(5, "pw_mu", "pw_sd"); siz = z(6, "si_mu", "si_sd")
    intd, outd, clid = zc(7), zc(8), zc(9)
    yt = y[tm]

    def fit(cols):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        return mdl, mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
    base_cols = [lf, tsz, bpz, pwz, siz]
    _, p_full = fit(base_cols)
    print(f"games: {len(y):,}")
    print(f"\n=== LOCKED TEST  n={len(yt):,}  full LL={ll(p_full, yt):.5f} (=0.67322) ===")
    for name, feats in [("intensity (in a race)", [intd]),
                        ("out/tanking", [outd]),
                        ("clinch/coasting", [clid]),
                        ("all stakes signals", [intd, outd, clid])]:
        mdl, p_a = fit(base_cols + feats)
        d, lo, hi = boot(ll_vec(p_full, yt), ll_vec(p_a, yt))
        sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
        coefs = [round(c, 4) for c in mdl.coef_[0][5:]]
        print(f"  + {name:24s} LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig:5s} coef={coefs}")


if __name__ == "__main__":
    main()
