"""Does PARK x pitcher fly-ball profile add over the full model (0.67322)?

xFIP normalizes HR by LEAGUE HR/FB, but HR/FB is park-dependent -- a fly-ball
pitcher in a bandbox is riskier than in a pitcher's park. Both starters pitch in
the same park, so the net win-prob effect is:
    park_HR_deviation  x  (away starter fly-ball% - home starter fly-ball%)
Park HR/FB is accumulated per home park as-of (leak-safe, EB-shrunk to league);
each starter's fly-ball share comes from the SIERA rater (as-of). Added to the full
model. Bootstrap.
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
PARK_PRIOR = 800.0    # fly balls of EB shrink of a park's HR/FB to league
FB_PRIOR = 120.0      # batted balls of shrink of a pitcher's fly-ball share to league


def per_game_totals():
    ghr = defaultdict(int); gfb = defaultdict(int)
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        ghr[r["game_id"]] += int(r["HR"])
    for r in csv.DictReader(open("data/retro_events/flyballs.csv")):
        gfb[r["game_id"]] += int(r["fb"])
    return ghr, gfb


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
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = json.load(open("mlbwp/artifacts/blend.json"))
    lg_hrfb = B.get("bp_lg_core") and 0.12247 or 0.12247
    lg_hrfb = json.load(open("mlbwp/artifacts/ratings.json")).get("league_hrfb", 0.12247)
    lg_fb_share = slg["fb"] / (slg["gb"] + slg["fb"] + slg["pu"])
    ghr, gfb = per_game_totals()

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    phr = defaultdict(float); pfb = defaultdict(float)   # park HR, park FB_total (as-of)
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        park = g["home"]
        park_hrfb = (phr[park] + PARK_PRIOR * lg_hrfb) / (pfb[park] + PARK_PRIOR)
        park_dev = park_hrfb / lg_hrfb - 1.0

        def fb_share(pid):
            e = sr.E[pid]
            bip = e[2] + e[3] + e[4]
            return (e[3] + FB_PRIOR * lg_fb_share) / (bip + FB_PRIOR)
        feat = park_dev * (fb_share(g["away_sp"]) - fb_share(g["home_sp"]))

        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        sd = sr.edge(g["home_sp"], g["away_sp"])
        gid = g["game_id"]
        if gid in tsf and pdiff.get(gid) is not None:
            rows.append((g["season"], p, g["y"], tsf[gid], bpd[gid], pdiff[gid], sd, feat))

        # advance
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)
        hr_g = ghr.get(gid, 0); fb_g = gfb.get(gid, 0)
        phr[park] += hr_g; pfb[park] += fb_g + hr_g

    aa = np.array(rows, float)
    seas = aa[:, 0]; lf = logit(aa[:, 1]); y = aa[:, 2]
    dm = np.isin(seas, list(DEV)); tm = np.isin(seas, TEST)

    def z(c, mu, sd):
        return (aa[:, c] - B[mu]) / B[sd]
    tsz = z(3, "ts_mu", "ts_sd"); bpz = z(4, "bp_mu", "bp_sd")
    pwz = z(5, "pw_mu", "pw_sd"); siz = z(6, "si_mu", "si_sd")
    pk = aa[:, 7]; pkz = (pk - pk[dm].mean()) / (pk[dm].std() or 1.0)
    yt = y[tm]

    def fit(cols):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        return mdl, mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
    base = [lf, tsz, bpz, pwz, siz]
    _, p_full = fit(base)
    mdl, p_a = fit(base + [pkz])
    d, lo, hi = boot(ll_vec(p_full, yt), ll_vec(p_a, yt))
    sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
    top = sorted(((k, (phr[k] + PARK_PRIOR * lg_hrfb) / (pfb[k] + PARK_PRIOR) / lg_hrfb - 1)
                  for k in phr if pfb[k] > 2000), key=lambda x: -x[1])
    print(f"games: {len(y):,}  park HR factor (dev from league): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in top[:3]) + " ... "
          + ", ".join(f"{k} {v:+.2f}" for k, v in top[-3:]))
    print(f"\n=== LOCKED TEST  n={len(yt):,}  full LL={ll(p_full, yt):.5f} (=0.67322) ===")
    print(f"  + park x pitcher fly-ball  LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig}  coef={mdl.coef_[0][-1]:+.4f}")


if __name__ == "__main__":
    main()
