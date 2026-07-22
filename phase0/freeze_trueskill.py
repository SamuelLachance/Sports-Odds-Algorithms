"""Freeze the TrueSkill ratings and the blend the live model needs.

Writes:
  mlbwp/artifacts/ts_ratings.json  {retro_id: mu}  (players with a real sample)
  mlbwp/artifacts/blend.json       the DEV-fit logistic blend:
    recal  — logistic(y ~ logit_fip)                  used when lineups are unknown
    full   — logistic(y ~ logit_fip + ts_edge_z)      used when lineups are posted
    ts_mu, ts_sd — standardisation of ts_edge from DEV

The full blend is the 0.67509-holdout model; recal is the 0.67558 fallback that
needs no lineups. Both are fit on DEV 2000-2015 only.
"""

from __future__ import annotations

import csv
import json
import os
import sys

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
from trueskill_pa import build_feature  # noqa: E402

ART = "mlbwp/artifacts"
EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
MIN_N = 50


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    # 1. TrueSkill ratings (walks all PAs, also refreshes ts_feature.csv)
    ts = build_feature()
    ratings = {p: round(ts.mu[p], 3) for p in ts.mu if ts.n[p] >= MIN_N}
    json.dump(ratings, open(f"{ART}/ts_ratings.json", "w"))
    print(f"froze {len(ratings)} TrueSkill ratings")

    # 2. FIP-Elo per-game probability
    games = load_games()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rec, last = {}, None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        rec[g["game_id"]] = (g["season"], m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"]), g["y"])
        m.update(g)

    # 3. join ts_edge, fit blend on DEV
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    j = [(s, p, y, tsf[gid]) for gid, (s, p, y) in rec.items() if gid in tsf]
    seasons = np.array([x[0] for x in j])
    lf = logit(np.array([x[1] for x in j]))
    y = np.array([x[2] for x in j], float)
    tse = np.array([x[3] for x in j])
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)
    ts_mu, ts_sd = float(tse[dmask].mean()), float(tse[dmask].std())
    tsz = (tse - ts_mu) / ts_sd

    recal = LogisticRegression(C=1e6, max_iter=1000).fit(lf[dmask].reshape(-1, 1), y[dmask])
    full = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])

    p_recal = recal.predict_proba(lf[tmask].reshape(-1, 1))[:, 1]
    p_full = full.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    yt = y[tmask]

    blend = {
        "recal": {"b0": float(recal.intercept_[0]), "b1": float(recal.coef_[0][0])},
        "full": {"b0": float(full.intercept_[0]),
                 "b1": float(full.coef_[0][0]), "b2": float(full.coef_[0][1])},
        "ts_mu": ts_mu, "ts_sd": ts_sd,
        "holdout": {"recal_ll": round(ll(p_recal, yt), 5), "full_ll": round(ll(p_full, yt), 5),
                    "n": int(len(yt))},
    }
    json.dump(blend, open(f"{ART}/blend.json", "w"), indent=1)
    print(f"froze blend: recal LL={blend['holdout']['recal_ll']}  full LL={blend['holdout']['full_ll']}")
    print(f"  recal: p=sigmoid({blend['recal']['b0']:.3f} + {blend['recal']['b1']:.3f}*logit_fip)")
    print(f"  full:  + {blend['full']['b2']:.4f}*ts_edge_z   (ts_mu={ts_mu:.3f}, ts_sd={ts_sd:.3f})")


if __name__ == "__main__":
    main()
