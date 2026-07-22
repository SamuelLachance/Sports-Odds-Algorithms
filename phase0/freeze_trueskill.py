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
from mlbwp.trueskill import MU0, SIG0, BETA, TAU  # noqa: E402
from mlbwp.siera import SieraRater  # noqa: E402

ART = "mlbwp/artifacts"
APPEAR = "data/retro_events/appearances.csv"
POWER = "data/retro_events/power.csv"
LINEUPS = "data/retro_events/lineups.csv"
EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
MIN_N = 50
MIN_PA = 20              # freeze a batter's power rating once he has this many career PA
BP_PRIOR_OUTS = 200.0    # empirical-Bayes shrink strength (~67 IP) for bullpen FIP
PWR_PRIOR_PA = 150.0     # empirical-Bayes shrink strength (PA) for a batter's power rate
SIERA_PRIOR = 120.0      # PA-equivalent shrink of a starter's rate to league (SIERA)
SIERA_MIN_PA = 60        # freeze a starter's SIERA state once his EWMA PA reaches this


def load_siera_inputs():
    """(game_id,pitcher)->(SO,BB,BF) for starters, and ->(GB,FB,PU) batted balls."""
    ss, bbl = {}, {}
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        if r["is_starter"] == "1":
            ss[(r["game_id"], r["pitcher"])] = (int(r["SO"]), int(r["BB"]), int(r["bf"]))
    for r in csv.DictReader(open("data/retro_events/battedball.csv")):
        bbl[(r["game_id"], r["pitcher"])] = (int(r["gb"]), int(r["fb"]), int(r["pu"]))
    return ss, bbl


def league_siera_rates(games, ss, bbl):
    tS = tB = tPA = tG = tF = tP = 0
    for g in games:
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((g["game_id"], pid), (0, 0, 0))
            gb, fb, pu = bbl.get((g["game_id"], pid), (0, 0, 0))
            tS += so; tB += bb; tPA += bf; tG += gb; tF += fb; tP += pu
    return dict(so=tS / tPA, bb=tB / tPA, gb=tG / tPA, fb=tF / tPA, pu=tP / tPA)


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


def load_power():
    """(game_id -> list[(batter, pa, h, tb)]) and league ISO-per-PA."""
    by_game = defaultdict(list)
    tot_pa = tot_h = tot_tb = 0
    with open(POWER, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pa, h, tb = int(r["pa"]), int(r["h"]), int(r["tb"])
            by_game[r["game_id"]].append((r["batter"], pa, h, tb))
            tot_pa += pa; tot_h += h; tot_tb += tb
    return by_game, (tot_tb - tot_h) / tot_pa


def load_lineups():
    """(game_id, side) -> [batters]; Retrosheet side 0 = visitor/away, 1 = home."""
    lu = defaultdict(list)
    with open(LINEUPS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lu[(r["game_id"], int(r["side"]))].append(r["batter"])
    return lu


def power_diffs(games, power, lu, lg_iso):
    """game_id -> home-minus-away lineup isolated-power differential, as-of before
    each game (leak-safe), AND the final per-batter career (PA,H,TB) through the
    last season (frozen for serving). Sign matches phase0/power_eval.py:
    home_lineup_ISO - away_lineup_ISO, so positive favours the home lineup."""
    car = defaultdict(lambda: [0, 0, 0])   # batter -> career PA,H,TB (as-of)
    prior = PWR_PRIOR_PA * lg_iso
    out = {}
    for g in games:
        def bat_iso(b):
            c = car[b]
            return (c[2] - c[1] + prior) / (c[0] + PWR_PRIOR_PA)

        def lineup(side):
            bs = lu.get((g["game_id"], side), [])
            v = [bat_iso(b) for b in bs]
            return sum(v) / len(v) if v else None
        h, a = lineup(1), lineup(0)
        out[g["game_id"]] = (h - a) if (h is not None and a is not None) else None
        for b, pa, hh, tb in power.get(g["game_id"], []):
            c = car[b]; c[0] += pa; c[1] += hh; c[2] += tb
    return out, car


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
        # Freeze (mu, var) per player so the live server can reconstruct the engine
        # and replay this season's PAs through the identical TrueSkill.update.
        ratings = {p: [round(ts.mu[p], 4), round(ts.var[p], 6)]
                   for p in ts.mu if ts.n[p] >= MIN_N}
        json.dump(ratings, open(f"{ART}/ts_ratings.json", "w"))
        print(f"froze {len(ratings)} TrueSkill ratings (mu, var)")

    # 2. FIP-Elo probability + bullpen + power + SIERA differentials (one walk)
    games = load_games()
    per_game, bp_lg_core = reliever_aggs()
    bpd = bullpen_diffs(games, per_game, bp_lg_core)
    power, pwr_lg_iso = load_power()
    pdiff, career = power_diffs(games, power, load_lineups(), pwr_lg_iso)
    ss, bbl = load_siera_inputs()
    siera_lg = league_siera_rates(games, ss, bbl)
    sr = SieraRater(decay=PARAMS["decay"], prior=SIERA_PRIOR, lg=siera_lg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rec, sdiff, last = {}, {}, None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        rec[g["game_id"]] = (g["season"], m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"]), g["y"])
        sdiff[g["game_id"]] = sr.edge(g["home_sp"], g["away_sp"])
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((g["game_id"], pid), (0, 0, 0))
            gb, fb, pu = bbl.get((g["game_id"], pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)

    # 3. join features + fit the blend tiers on DEV. SIERA (like bullpen) needs no
    # lineup, so it joins BOTH the no-lineup 'recbp' tier and the 'full' tier.
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    j = [(s, p, y, tsf[gid], bpd[gid], pdiff.get(gid), sdiff[gid])
         for gid, (s, p, y) in rec.items() if gid in tsf and pdiff.get(gid) is not None]
    seasons = np.array([x[0] for x in j])
    lf = logit(np.array([x[1] for x in j]))
    y = np.array([x[2] for x in j], float)
    tse = np.array([x[3] for x in j]); bpe = np.array([x[4] for x in j])
    pwe = np.array([x[5] for x in j]); sie = np.array([x[6] for x in j])
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)
    ts_mu, ts_sd = float(tse[dmask].mean()), float(tse[dmask].std())
    bp_mu, bp_sd = float(bpe[dmask].mean()), float(bpe[dmask].std())
    pw_mu, pw_sd = float(pwe[dmask].mean()), float(pwe[dmask].std())
    si_mu, si_sd = float(sie[dmask].mean()), float(sie[dmask].std())
    tsz = (tse - ts_mu) / ts_sd
    bpz = (bpe - bp_mu) / bp_sd
    pwz = (pwe - pw_mu) / pw_sd
    siz = (sie - si_mu) / si_sd

    recal = LogisticRegression(C=1e6, max_iter=1000).fit(lf[dmask].reshape(-1, 1), y[dmask])
    recbp = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], bpz[dmask], siz[dmask]]), y[dmask])
    full = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask], bpz[dmask], pwz[dmask], siz[dmask]]), y[dmask])

    yt = y[tmask]
    p_recal = recal.predict_proba(lf[tmask].reshape(-1, 1))[:, 1]
    p_recbp = recbp.predict_proba(np.column_stack([lf[tmask], bpz[tmask], siz[tmask]]))[:, 1]
    p_full = full.predict_proba(np.column_stack([lf[tmask], tsz[tmask], bpz[tmask], pwz[tmask], siz[tmask]]))[:, 1]

    blend = {
        "recal": {"b0": float(recal.intercept_[0]), "b1": float(recal.coef_[0][0])},
        "recbp": {"b0": float(recbp.intercept_[0]), "b1": float(recbp.coef_[0][0]),
                  "b3": float(recbp.coef_[0][1]), "b5": float(recbp.coef_[0][2])},
        "full": {"b0": float(full.intercept_[0]), "b1": float(full.coef_[0][0]),
                 "b2": float(full.coef_[0][1]), "b3": float(full.coef_[0][2]),
                 "b4": float(full.coef_[0][3]), "b5": float(full.coef_[0][4])},
        "ts_mu": ts_mu, "ts_sd": ts_sd,
        "ts_params": {"beta": BETA, "tau": TAU, "mu0": MU0, "sig0": SIG0},
        "bp_mu": bp_mu, "bp_sd": bp_sd,
        "pw_mu": pw_mu, "pw_sd": pw_sd,
        "si_mu": si_mu, "si_sd": si_sd,
        "bp_lg_core": round(bp_lg_core, 4), "bp_prior_outs": BP_PRIOR_OUTS,
        "pwr_lg_iso": round(pwr_lg_iso, 5), "pwr_prior_pa": PWR_PRIOR_PA,
        "siera_prior": SIERA_PRIOR, "siera_lg": {k: round(v, 6) for k, v in siera_lg.items()},
        "holdout": {"recal_ll": round(ll(p_recal, yt), 5),
                    "recbp_ll": round(ll(p_recbp, yt), 5),
                    "full_ll": round(ll(p_full, yt), 5), "n": int(len(yt))},
    }
    json.dump(blend, open(f"{ART}/blend.json", "w"), indent=1)

    # 4. freeze per-batter power counts and per-starter SIERA EWMA state
    pr = {b: [c[0], c[1], c[2]] for b, c in career.items() if c[0] >= MIN_PA}
    json.dump(pr, open(f"{ART}/power_ratings.json", "w"))
    srr = {pid: [round(x, 4) for x in e] for pid, e in sr.E.items() if e[5] >= SIERA_MIN_PA}
    json.dump(srr, open(f"{ART}/siera_ratings.json", "w"))

    h = blend["holdout"]
    print(f"froze blend: recal={h['recal_ll']}  recbp={h['recbp_ll']}  full={h['full_ll']}")
    print(f"  full:  + {blend['full']['b2']:.4f}*ts_z + {blend['full']['b3']:.4f}*bp_z "
          f"+ {blend['full']['b4']:.4f}*pw_z + {blend['full']['b5']:.4f}*siera_z")
    print(f"  froze {len(pr)} power ratings, {len(srr)} SIERA states (prior {SIERA_PRIOR:.0f}, "
          f"lg netGB%={siera_lg['gb']-siera_lg['fb']-siera_lg['pu']:.3f})")


if __name__ == "__main__":
    main()
