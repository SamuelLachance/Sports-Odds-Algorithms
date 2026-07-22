"""Freeze the TrueSkill ratings and the blend the live model needs.

Writes:
  mlbwp/artifacts/ts_ratings.json  {retro_id: mu}  (players with a real sample)
  mlbwp/artifacts/blend.json       the DEV-fit logistic blend:
    recal  — logistic(y ~ logit_fip)                            no bullpen, no lineups
    recbp  — logistic(y ~ logit_fip + bp_z)                     bullpen, no lineups
    full   — logistic(y ~ logit_fip + ts_edge_z + bp_z)         bullpen + lineups
    ts_mu, ts_sd — standardisation of ts_edge from DEV
    bp_mu, bp_sd — standardisation of the bullpen differential from DEV
    bp_lg_core, bp_prior_outs — the empirical-Bayes prior for a team's bullpen FIP

bp_z is the home-minus-away season-to-date reliever FIP-core differential: the
model rates the STARTER via FIP but left the ~40% of relief innings to team Elo;
the bullpen term recovers that (locked-holdout tested, phase0/bullpen_eval.py:
+0.00017 nats, CI [+0.00004,+0.00029]). recbp is the default served tier; full is
the best when lineups post. All coefficients are fit on DEV 2000-2015 only.
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
from mlbwp.train import PARAMS  # noqa: E402
from trueskill_pa import build_feature  # noqa: E402

ART = "mlbwp/artifacts"
APPEAR = "data/retro_events/appearances.csv"
EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
MIN_N = 50
BP_PRIOR_OUTS = 200.0    # empirical-Bayes shrink strength (~67 IP) for bullpen FIP


def reliever_aggs():
    """(game_id, team) -> (fip_num, outs) for relievers only; plus league FIP-core.

    fip_num = 13*HR + 3*(BB+HBP) - 2*SO over the team's relievers in that game.
    Must match mlbwp.serve.Predictor.bp_fip exactly (train/serve parity)."""
    per_game = defaultdict(lambda: [0.0, 0])
    lg_num = lg_outs = 0
    with open(APPEAR, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["is_starter"] == "1":
                continue
            outs = int(r["outs"])
            hr, bb, hbp, so = (int(r[k]) for k in ("HR", "BB", "HBP", "SO"))
            num = 13 * hr + 3 * (bb + hbp) - 2 * so
            a = per_game[(r["game_id"], r["team"])]
            a[0] += num
            a[1] += outs
            lg_num += num
            lg_outs += outs
    lg_core = lg_num / lg_outs * 3.0 if lg_outs else 1.0
    return per_game, lg_core


def bullpen_diffs(games, per_game, lg_core):
    """game_id -> home-minus-away season-to-date bullpen FIP-core differential,
    read AS-OF before each game (leak-safe). Sign matches phase0/bullpen_eval.py:
    away_fip - home_fip, so a positive value favours the home team (lower = better)."""
    cum = defaultdict(lambda: [0.0, 0])   # (season, team) -> cum fip_num, outs
    prior = BP_PRIOR_OUTS * lg_core / 3.0
    out = {}
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            cum.clear()
        prev = g["season"]

        def fip(team):
            c = cum[(g["season"], team)]
            return (c[0] + prior) / (c[1] + BP_PRIOR_OUTS) * 3.0
        out[g["game_id"]] = fip(g["away"]) - fip(g["home"])
        for team in (g["home"], g["away"]):
            num, outs = per_game.get((g["game_id"], team), (0.0, 0))
            c = cum[(g["season"], team)]
            c[0] += num
            c[1] += outs
    return out


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    # 1. TrueSkill ratings (walks all PAs, also refreshes ts_feature.csv).
    # The TS layer is unchanged when only the blend is being re-frozen; set
    # SKIP_TS=1 to reuse the existing ts_ratings.json + ts_feature.csv and skip
    # the multi-minute PA walk.
    if os.environ.get("SKIP_TS") == "1" and os.path.isfile(f"{ART}/ts_ratings.json") \
            and os.path.isfile("data/retro_events/ts_feature.csv"):
        print("SKIP_TS=1: reusing existing ts_ratings.json + ts_feature.csv")
    else:
        ts = build_feature()
        ratings = {p: round(ts.mu[p], 3) for p in ts.mu if ts.n[p] >= MIN_N}
        json.dump(ratings, open(f"{ART}/ts_ratings.json", "w"))
        print(f"froze {len(ratings)} TrueSkill ratings")

    # 2. FIP-Elo per-game probability + bullpen differential (same game walk)
    games = load_games()
    per_game, bp_lg_core = reliever_aggs()
    bpd = bullpen_diffs(games, per_game, bp_lg_core)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rec, last = {}, None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        rec[g["game_id"]] = (g["season"], m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"]), g["y"])
        m.update(g)

    # 3. join ts_edge + bullpen diff, fit the three blend tiers on DEV
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    j = [(s, p, y, tsf[gid], bpd[gid]) for gid, (s, p, y) in rec.items() if gid in tsf]
    seasons = np.array([x[0] for x in j])
    lf = logit(np.array([x[1] for x in j]))
    y = np.array([x[2] for x in j], float)
    tse = np.array([x[3] for x in j])
    bpe = np.array([x[4] for x in j])
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)
    ts_mu, ts_sd = float(tse[dmask].mean()), float(tse[dmask].std())
    bp_mu, bp_sd = float(bpe[dmask].mean()), float(bpe[dmask].std())
    tsz = (tse - ts_mu) / ts_sd
    bpz = (bpe - bp_mu) / bp_sd

    recal = LogisticRegression(C=1e6, max_iter=1000).fit(lf[dmask].reshape(-1, 1), y[dmask])
    recbp = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], bpz[dmask]]), y[dmask])
    full = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask], bpz[dmask]]), y[dmask])

    yt = y[tmask]
    p_recal = recal.predict_proba(lf[tmask].reshape(-1, 1))[:, 1]
    p_recbp = recbp.predict_proba(np.column_stack([lf[tmask], bpz[tmask]]))[:, 1]
    p_full = full.predict_proba(np.column_stack([lf[tmask], tsz[tmask], bpz[tmask]]))[:, 1]

    blend = {
        "recal": {"b0": float(recal.intercept_[0]), "b1": float(recal.coef_[0][0])},
        "recbp": {"b0": float(recbp.intercept_[0]),
                  "b1": float(recbp.coef_[0][0]), "b3": float(recbp.coef_[0][1])},
        "full": {"b0": float(full.intercept_[0]), "b1": float(full.coef_[0][0]),
                 "b2": float(full.coef_[0][1]), "b3": float(full.coef_[0][2])},
        "ts_mu": ts_mu, "ts_sd": ts_sd,
        "bp_mu": bp_mu, "bp_sd": bp_sd,
        "bp_lg_core": round(bp_lg_core, 4), "bp_prior_outs": BP_PRIOR_OUTS,
        "holdout": {"recal_ll": round(ll(p_recal, yt), 5),
                    "recbp_ll": round(ll(p_recbp, yt), 5),
                    "full_ll": round(ll(p_full, yt), 5), "n": int(len(yt))},
    }
    json.dump(blend, open(f"{ART}/blend.json", "w"), indent=1)
    h = blend["holdout"]
    print(f"froze blend: recal={h['recal_ll']}  recbp={h['recbp_ll']}  full={h['full_ll']}")
    print(f"  recal: sigmoid({blend['recal']['b0']:.3f} + {blend['recal']['b1']:.3f}*logit_fip)")
    print(f"  recbp: + {blend['recbp']['b3']:.4f}*bp_z")
    print(f"  full:  + {blend['full']['b2']:.4f}*ts_z + {blend['full']['b3']:.4f}*bp_z")
    print(f"  bp: lg_core={bp_lg_core:.3f} prior_outs={BP_PRIOR_OUTS:.0f} mu={bp_mu:.4f} sd={bp_sd:.4f}")


if __name__ == "__main__":
    main()
