"""Does BASERUNNING / speed add over the full model (0.67322)?

Per-batter career baserunning runs (wSB + extra bases - outs on base - GIDP, run-
valued), rate per PA, EB-shrunk to 0 (league-average baserunner adds 0), averaged
over the actual lineup, home-minus-away, as-of. The third leg of offense the model
is blind to (we have on-base + power). Added to [xfip + ts + bp + power + siera].
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
PRIOR_PA = 200.0
# run values: SB +0.20, CS -0.41, extra base +0.19, out-on-base -0.41, GIDP -0.25
RV = (0.20, -0.41, 0.19, -0.41, -0.25)


def load_baserun():
    by_game = defaultdict(list)
    for r in csv.DictReader(open("data/retro_events/baserun.csv")):
        pa = int(r["pa"])
        brv = (RV[0] * int(r["sb"]) + RV[1] * int(r["cs"]) + RV[2] * int(r["xb"])
               + RV[3] * int(r["oob"]) + RV[4] * int(r["gidp"]))
        by_game[r["game_id"]].append((r["player"], pa, brv))
    return by_game


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
    br = load_baserun()
    lu = FZ.load_lineups()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lgiso)
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = json.load(open("mlbwp/artifacts/blend.json"))

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    car = defaultdict(lambda: [0.0, 0.0])   # player -> [brv_sum, pa_sum]
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]

        def rate(pid):
            c = car[pid]
            return c[0] / (c[1] + PRIOR_PA)

        def lineup(side):
            bs = lu.get((g["game_id"], side), [])
            v = [rate(b) for b in bs]
            return sum(v) / len(v) if v else None
        h, a = lineup(1), lineup(0)
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        sd = sr.edge(g["home_sp"], g["away_sp"])
        gid = g["game_id"]
        if gid in tsf and pdiff.get(gid) is not None and h is not None and a is not None:
            rows.append((g["season"], p, g["y"], tsf[gid], bpd[gid], pdiff[gid], sd, h - a))
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)
        for pid, pa, brv in br.get(gid, []):
            car[pid][0] += brv; car[pid][1] += pa

    aa = np.array(rows, float)
    seas = aa[:, 0]; lf = logit(aa[:, 1]); y = aa[:, 2]
    dm = np.isin(seas, list(DEV)); tm = np.isin(seas, TEST)

    def z(c, mu, sd):
        return (aa[:, c] - B[mu]) / B[sd]
    tsz = z(3, "ts_mu", "ts_sd"); bpz = z(4, "bp_mu", "bp_sd")
    pwz = z(5, "pw_mu", "pw_sd"); siz = z(6, "si_mu", "si_sd")
    brd = aa[:, 7]; brz = (brd - brd[dm].mean()) / (brd[dm].std() or 1.0)
    yt = y[tm]

    def fit(cols):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        return mdl, mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
    base = [lf, tsz, bpz, pwz, siz]
    _, p_full = fit(base)
    mdl, p_a = fit(base + [brz])
    d, lo, hi = boot(ll_vec(p_full, yt), ll_vec(p_a, yt))
    sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
    # face validity: top baserunners with a real sample
    top = sorted(((p, car[p][0] / (car[p][1] + PRIOR_PA)) for p in car if car[p][1] > 2000),
                 key=lambda x: -x[1])[:5]
    print(f"games: {len(y):,}  top baserunners (runs/PA): " + ", ".join(f"{p} {v:+.3f}" for p, v in top))
    print(f"\n=== LOCKED TEST  n={len(yt):,}  full LL={ll(p_full, yt):.5f} (=0.67322) ===")
    print(f"  + lineup baserunning   LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig}  coef={mdl.coef_[0][-1]:+.4f}")


if __name__ == "__main__":
    main()
