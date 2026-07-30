"""Does SIERA add over the xFIP model (0.67368)?

SIERA estimates a pitcher from K%, BB% and net ground-ball rate with squares and
interactions -- it adds ground balls + non-linearity that xFIP lacks. We build each
starter's rolling SIERA (EWMA of his SO/BB/GB/FB/PU/PA, EB-shrunk to league rates,
per the published FanGraphs formula), take the home-minus-away differential as-of,
and add it over the current full model [logit_xfip + ts_edge_z + bp_z + power_z].
Head-to-head: recalibrated SIERA-diff-only pitcher signal vs the xFIP core. Bootstrap.
"""

from __future__ import annotations

import csv
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
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
DECAY = PARAMS["decay"]
PRIOR = 120.0        # PA-equivalent shrink of each rate to league


def load_starter_stats():
    """(game_id, pitcher) -> (SO, BB, BF) for starters."""
    out = {}
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        if r["is_starter"] == "1":
            out[(r["game_id"], r["pitcher"])] = (int(r["SO"]), int(r["BB"]), int(r["bf"]))
    return out


def load_bb():
    out = {}
    for r in csv.DictReader(open("data/retro_events/battedball.csv")):
        out[(r["game_id"], r["pitcher"])] = (int(r["gb"]), int(r["fb"]), int(r["pu"]))
    return out


def siera(so, bb, netgb):
    """FanGraphs SIERA from per-PA rates (constant drops out of a differential)."""
    return (6.145 - 16.986 * so + 11.434 * bb - 1.858 * netgb
            + 7.653 * so * so - 6.664 * netgb * netgb
            + 10.130 * so * netgb - 5.195 * bb * netgb)


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
    games = load_games()          # xFIP baseline
    ss = load_starter_stats()
    bb = load_bb()
    # league rates
    tS = tB = tG = tF = tP = tPA = 0
    for k, (so, b, bf) in ss.items():
        g, f, p = bb.get(k, (0, 0, 0))
        tS += so; tB += b; tPA += bf; tG += g; tF += f; tP += p
    lg = dict(so=tS / tPA, bb=tB / tPA, gb=tG / tPA, fb=tF / tPA, pu=tP / tPA)
    lg_netgb = lg["gb"] - lg["fb"] - lg["pu"]

    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    E = defaultdict(lambda: dict(SO=0.0, BB=0.0, GB=0.0, FB=0.0, PU=0.0, PA=0.0))
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]

        def sp_siera(pid):
            s = E[pid]
            pa = s["PA"] + PRIOR
            so = (s["SO"] + PRIOR * lg["so"]) / pa
            bbr = (s["BB"] + PRIOR * lg["bb"]) / pa
            netgb = ((s["GB"] - s["FB"] - s["PU"]) + PRIOR * lg_netgb) / pa
            return siera(so, bbr, netgb)
        sd = sp_siera(g["home_sp"]) - sp_siera(g["away_sp"])
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in FZ_ts and g["game_id"] in pdiff and pdiff[g["game_id"]] is not None:
            rows.append((g["season"], p, g["y"], FZ_ts[g["game_id"]], bpd[g["game_id"]],
                         pdiff[g["game_id"]], sd))
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, b, bf = ss.get((g["game_id"], pid), (0, 0, 0))
            gg, ff, pp = bb.get((g["game_id"], pid), (0, 0, 0))
            s = E[pid]
            for key, val in (("SO", so), ("BB", b), ("PA", bf), ("GB", gg), ("FB", ff), ("PU", pp)):
                s[key] = DECAY * s[key] + val

    arr = np.array(rows, dtype=float)
    s = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]
    tse, bpe, pwe, sie = arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
    dm = np.isin(s, list(DEV)); tm = np.isin(s, TEST)

    def z(v):
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    tsz, bpz, pwz, siz = z(tse), z(bpe), z(pwe), z(sie)
    yt = y[tm]

    def fit(cols):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        return mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]

    p_full = fit([lf, tsz, bpz, pwz])
    p_a = fit([lf, tsz, bpz, pwz, siz])
    print(f"games: {len(y):,}  league SIERA-ish netGB%={lg_netgb:.3f}")
    print(f"\n=== LOCKED TEST  n={len(yt):,}  full [xfip+ts+bp+power] LL={ll(p_full, yt):.5f} (=0.67368) ===")
    d, lo, hi = boot(ll_vec(p_full, yt), ll_vec(p_a, yt))
    sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
    print(f"  + SIERA differential   LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig}")


# module-level joins (loaded once)
bpd = FZ.bullpen_diffs(load_games(), *FZ.reliever_aggs())
_pw, _lgiso = FZ.load_power()
pdiff, _ = FZ.power_diffs(load_games(), _pw, FZ.load_lineups(), _lgiso)
FZ_ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

if __name__ == "__main__":
    main()
