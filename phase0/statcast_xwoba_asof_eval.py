"""THE clean test: does AS-OF (leak-free, within-season) lineup xwOBA add over the
full model in the Statcast era?

Accumulates each batter's xwOBA career-as-of from per-batted-ball components
(data/statcast_bbe_batter.csv), reading STRICTLY the days before each game (no leak),
EB-shrunk to league -- exactly parallel to the power/on-base features. Lineup mean
home-minus-away, increment over the full model. Fit 2015-2019, test 2021-2024.

Run phase0/statcast_bbe_pull.py first. Arg 'pitcher' tests as-of xwOBA-AGAINST for
the starting pitchers instead (home_sp - away_sp).
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
TRAIN = {2015, 2016, 2017, 2018, 2019}
TEST = {2021, 2022, 2023, 2024}
LG_XWOBA = 0.318
PRIOR = 100.0                     # denom (~PA) of EB shrink toward league xwOBA


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def load_bbe(path, m2r):
    """date(YYYYMMDD) -> list[(retro_id, xnum, denom)], mlbam mapped to retro."""
    by_date = defaultdict(list)
    for r in csv.DictReader(open(path)):
        retro = m2r.get(int(r["mlbam"]))
        if retro:
            d = r["date"].replace("-", "")
            by_date[d].append((retro, float(r["xnum"]), float(r["denom"])))
    return by_date


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "batter"
    path = f"data/statcast_bbe_{which}.csv"
    m2r = {int(k): v for k, v in json.load(open("mlbwp/artifacts/mlbam_to_retro.json")).items()}
    bbe = load_bbe(path, m2r)
    bbe_dates = sorted(bbe)

    games = load_games()
    lu = FZ.load_lineups()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lgiso)
    brdiff, _ = FZ.baserun_diffs(games, FZ.load_baserun(), lu)
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = json.load(open("mlbwp/artifacts/blend.json"))
    fb = B["full"]

    def z(v, mu, sd_):
        return (v - B[mu]) / B[sd_]

    xnum = defaultdict(float); xden = defaultdict(float)   # career-as-of accumulators

    def xw(pid):
        return (xnum[pid] + PRIOR * LG_XWOBA) / (xden[pid] + PRIOR)

    def lineup(gid, side):
        bs = lu.get((gid, side), [])
        v = [xw(b) for b in bs]
        return sum(v) / len(v) if v else None

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rows = []
    prev = None; bp = 0
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        gid = g["game_id"]; s = g["season"]; gd = g["date"]
        # flush all Statcast days strictly before this game's date (leak-safe as-of)
        while bp < len(bbe_dates) and bbe_dates[bp] < gd:
            for pid, xn, dn in bbe[bbe_dates[bp]]:
                xnum[pid] += xn; xden[pid] += dn
            bp += 1

        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        lf = logit(np.array([p]))[0]
        sd = sr.edge(g["home_sp"], g["away_sp"])
        ts, pw, br = tsf.get(gid), pdiff.get(gid), brdiff.get(gid)
        if (s in TRAIN or s in TEST) and ts is not None and pw is not None and br is not None:
            fl = (fb["b0"] + fb["b1"] * lf + fb["b2"] * z(ts, "ts_mu", "ts_sd")
                  + fb["b3"] * z(bpd[gid], "bp_mu", "bp_sd") + fb["b4"] * z(pw, "pw_mu", "pw_sd")
                  + fb["b5"] * z(sd, "si_mu", "si_sd") + fb["b6"] * z(br, "br_mu", "br_sd"))
            if which == "pitcher":
                feat = xw(g["home_sp"]) - xw(g["away_sp"])   # lower home xwOBA-against = better
            else:
                h, a = lineup(gid, 1), lineup(gid, 0)
                feat = (h - a) if (h is not None and a is not None) else None
            if feat is not None:
                rows.append((s, fl, g["y"], feat))
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fbb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fbb, pu, bf)

    a = np.array(rows, float)
    seas = a[:, 0]; fl = a[:, 1]; y = a[:, 2]; fe = a[:, 3]
    dm = np.isin(seas, list(TRAIN)); tm = np.isin(seas, list(TEST))
    fz = (fe - fe[dm].mean()) / (fe[dm].std() or 1.0)
    yt = y[tm]

    def fit(cols):
        X = np.column_stack([fl] + cols)
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(X[dm], y[dm])
        return mdl, mdl.predict_proba(X[tm])[:, 1]
    _, p_base = fit([])
    mdl, p_x = fit([fz])
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_x, yt))
    sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
    print(f"AS-OF xwOBA ({which}) | train 2015-19 (n={int(dm.sum()):,}) test 2021-24 (n={len(yt):,})")
    print(f"  base full model            LL={ll(p_base, yt):.5f}")
    print(f"  + AS-OF xwOBA (leak-free)  LL={ll(p_x, yt):.5f}  d={d:+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}] {sig}  coef={mdl.coef_[0][-1]:+.3f}")


if __name__ == "__main__":
    main()
