"""Player-value program (pv), NFL, COMPOSE step 2: walk-forward unit weights -> points per player. DEV ONLY.

Weights. For every season s the eight unit weights (QB, REC, RUSH, PROT, PR, RF, BALL, COV) are
re-estimated on seasons 2000..s-1 ONLY (1999 is the components' burn-in) by predicting each
game's point differential (home - away) from the home-minus-away LINEUP aggregates of
pv_nfl_compose_build (the players who actually played in the snap era, the depth/usage proxy
before), plus a home-field intercept (0 at neutral sites):
      margin = h*home + sum_k w_k * dA_k + e
Every A_k is already in nominal EPA per game, so the natural prior is w_k = 1 ("a nominal EPA is a
point"); the fit is ridge-shrunk TOWARD 1 with strength lambda_k = 267 * Var_train(dA_k), i.e. a
prior worth one season of games per unit (pre-declared, not tuned). A unit with no variance in the
training window (REC before 2006: receiver ratings start with air yards in 2006) keeps w = 1.
Robustness fit (reported, not used): the same regression on the EPA differential.

Outputs, all as-of (season s rows use the weights fit on seasons < s):
  data/pv_nfl_compose_weights.json           weights per fold (+ EPA-target variant, + a game-
                                             bootstrap of the 2015 fold)
  data/pv_nfl_compose_game_features.csv      one row per game 1999-2015 (dev = 2006-2015):
      lv_{home,away}      lineup value, points/game vs an average lineup (QB included)
      qv_{home,away}      the starting QB's value (points/game)  -- the QB separately
      lvx_{home,away}     lineup value with NO day-of information (expected roster)
      miss_{home,away}    lv - lvx (2013-2015: value of absent regulars net of replacement; 0 before)
      A_<unit>_{home,away} unit aggregates, nominal EPA/game, for downstream refits
      ts_{home,away}, tsx_{..}  the 11v11 shared-credit analog (pv_nfl_compose_tsdev), EPA/play net
  data/pv_nfl_compose_game_outcomes.csv      margin, epa_margin per game (OUTCOMES, validation only)
  data/pv_nfl_compose_player_values.parquet  one row per player-game (pre-game): points/game by
      unit and total, vs an average player at his position (regulars' as-of baseline)
  data/pv_nfl_compose_player_seasons.csv     one row per player-season: value at his last game
      and mean over his games, position, team, games
TEST discipline: inputs are the < 2016 outputs of the build scripts; asserts everywhere.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
UNITS = ["QB", "REC", "RUSH", "PROT", "PR", "RF", "BALL", "COV"]
TRAIN_LO = 2000
LAM_GAMES = 267.0
RNG = np.random.default_rng(20260924)


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def game_frame(TG, var="lin"):
    h = TG[TG.side == "home"].set_index("game_id")
    a = TG[TG.side == "away"].set_index("game_id").loc[h.index]
    G = pd.DataFrame({"season": h.season, "week": h.week, "home": h.team, "away": a.team,
                      "home_ind": 1.0 - h.neutral, "margin": h.pts_for - h.pts_against,
                      "epa_margin": h.epa_off - a.epa_off}, index=h.index)
    for u in UNITS:
        G[f"d_{u}"] = h[f"{u}_{var}"] - a[f"{u}_{var}"]
    return G


def fit_weights(G, s, target="margin"):
    tr = (G.season >= TRAIN_LO) & (G.season < s)
    if tr.sum() == 0:
        return {u: 1.0 for u in UNITS}, 0.0, 0
    X = G.loc[tr, [f"d_{u}" for u in UNITS]].to_numpy(float)
    y = G.loc[tr, target].to_numpy(float)
    h = G.loc[tr, "home_ind"].to_numpy(float)
    w0 = np.ones(len(UNITS))
    z = y - X @ w0                                    # residual after the prior
    var = X.var(0)
    lam = LAM_GAMES * var
    Z = np.column_stack([h, X])
    P = np.diag(np.concatenate([[0.0], lam]))
    live = np.concatenate([[True], var > 1e-10])
    Zl, Pl = Z[:, live], P[np.ix_(live, live)]
    beta = np.linalg.solve(Zl.T @ Zl + Pl, Zl.T @ z)
    full = np.zeros(len(live)); full[live] = beta
    w = w0 + full[1:]
    return dict(zip(UNITS, w.tolist())), float(full[0]), int(tr.sum())


def main():
    TG = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_team_games.parquet"))
    assert TG.season.max() < TEST_ERA
    assert TG[[f"{u}_{v}" for u in UNITS for v in ("lin", "exp")]].notna().all().all()
    G = game_frame(TG, "lin")
    Gx = game_frame(TG, "exp")
    seasons = sorted(G.season.unique())
    W, We, H = {}, {}, {}
    for s in seasons:
        W[s], H[s], n = fit_weights(G, s)
        We[s], _, _ = fit_weights(G, s, "epa_margin")
        if s >= 2006:
            log(f"fold {s} (train {TRAIN_LO}-{s-1}, n {n}): "
                + " ".join(f"{u} {W[s][u]:+.2f}" for u in UNITS) + f" | hfa {H[s]:+.2f}")
    # bootstrap of the last fold's weights (games resampled) -- sampling uncertainty of the weights
    tr = (G.season >= TRAIN_LO) & (G.season < 2015)
    Gt = G[tr]
    boots = []
    for _ in range(400):
        idx = RNG.integers(0, len(Gt), len(Gt))
        Gb = Gt.iloc[idx].copy(); Gb["season"] = 2014
        boots.append([fit_weights(Gb, 2015)[0][u] for u in UNITS])
    boots = np.array(boots)
    ci15 = {u: [round(float(np.percentile(boots[:, j], 2.5)), 3), round(float(np.percentile(boots[:, j], 97.5)), 3)]
            for j, u in enumerate(UNITS)}
    log("2015-fold weight 95% CI (game bootstrap):", ci15)

    # ------------------------------------------------ team-game values
    for u in UNITS:
        TG[f"w_{u}"] = TG.season.map(lambda s, u=u: W[s][u])
    TG["lv"] = sum(TG[f"w_{u}"] * TG[f"{u}_lin"] for u in UNITS)
    TG["lvx"] = sum(TG[f"w_{u}"] * TG[f"{u}_exp"] for u in UNITS)
    TG["qv"] = TG.w_QB * TG.QB_lin
    TG["miss"] = TG.lv - TG.lvx
    # centre on the as-of league mean of team lineup values (display only; cancels in home-away)
    for c in ("lv", "lvx"):
        m = TG.groupby("season")[c].mean()
        prior = {s: (m[[x for x in m.index if x < s]].mean() if s > m.index.min() else m[s]) for s in m.index}
        TG[c] = TG[c] - TG.season.map(prior)
    TS = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_tsdev_team_games.parquet"))
    assert TS.season.max() < TEST_ERA
    TG = TG.merge(TS[["game_id", "team", "ts_lin", "ts_exp"]], on=["game_id", "team"], how="left")

    h = TG[TG.side == "home"].set_index("game_id")
    a = TG[TG.side == "away"].set_index("game_id").loc[h.index]
    F = pd.DataFrame({"game_id": h.index, "season": h.season.values, "week": h.week.values,
                      "season_type": h.season_type.values, "home": h.team.values, "away": a.team.values,
                      "neutral": h.neutral.values, "dev": (h.season >= 2006).astype(int).values,
                      "snap_era": h.snap_era.values,
                      "lv_home": h.lv.values, "lv_away": a.lv.values,
                      "qv_home": h.qv.values, "qv_away": a.qv.values,
                      "lvx_home": h.lvx.values, "lvx_away": a.lvx.values,
                      "miss_home": h.miss.values, "miss_away": a.miss.values,
                      "starter_home": h.starter.values, "starter_away": a.starter.values,
                      "ts_home": h.ts_lin.values, "ts_away": a.ts_lin.values,
                      "tsx_home": h.ts_exp.values, "tsx_away": a.ts_exp.values})
    for u in UNITS:
        F[f"A_{u}_home"] = h[f"{u}_lin"].values; F[f"A_{u}_away"] = a[f"{u}_lin"].values
    assert F.season.max() < TEST_ERA
    F.to_csv(os.path.join(DATA, "pv_nfl_compose_game_features.csv"), index=False, float_format="%.6g")
    # outcomes live in their own file so the feature file can never carry a same-game result
    O = pd.DataFrame({"game_id": h.index, "season": h.season.values,
                      "margin": (h.pts_for - h.pts_against).values, "epa_margin": (h.epa_off - a.epa_off).values})
    O.to_csv(os.path.join(DATA, "pv_nfl_compose_game_outcomes.csv"), index=False, float_format="%.6g")
    log(f"wrote data/pv_nfl_compose_game_features.csv ({len(F)} games, DEV {int(F.dev.sum())})")

    # ------------------------------------------------ player values (points/game, pre-game)
    PG = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_player_games.parquet"))
    assert PG.season.max() < TEST_ERA
    PL = pd.read_csv(os.path.join(DATA, "nfl_players.csv"), low_memory=False,
                     usecols=["gsis_id", "display_name", "position", "position_group"])
    POS = dict(zip(PL.gsis_id, PL.position)); PGRP = dict(zip(PL.gsis_id, PL.position_group))
    NAME = dict(zip(PL.gsis_id, PL.display_name))
    PR = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_player_games.parquet"),
                         columns=["player_id", "bucket", "season"])
    PR = PR[PR.season < TEST_ERA]
    BK = dict(zip(PR.player_id, PR.bucket))
    PG["pos"] = PG.player_id.map(POS); PG["pgrp"] = PG.player_id.map(PGRP)
    PG["bucket"] = PG.player_id.map(BK).fillna(PG.pgrp)
    PG = PG[PG.is_starter_qb == 0].copy()             # the starting QB is valued through the QB unit
    # as-of baselines: regulars' (a >= 0.5) mean contribution per bucket, seasons strictly before
    for u, key in (("REC", "pgrp"), ("RUSH", None), ("PR", "bucket"), ("RF", "bucket"), ("BALL", "bucket"),
                   ("COV", None)):
        c = f"c_{u}"
        if key is None:
            PG[f"b_{u}"] = 0.0
            continue
        reg = PG[(PG.a >= 0.5) & PG[c].notna()]
        mb = reg.groupby(["season", key])[c].mean()
        base = {}
        for (s_, k_) in set(zip(PG.season, PG[key])):
            prior = [(y, k_) for y in range(s_ - 3, s_) if (y, k_) in mb.index]
            base[(s_, k_)] = float(mb.loc[prior].mean()) if prior else (float(mb.loc[(s_, k_)]) if (s_, k_) in mb.index else 0.0)
        PG[f"b_{u}"] = [base[(s_, k_)] for s_, k_ in zip(PG.season, PG[key])]
    PG["p_total"] = 0.0
    for u in ("REC", "RUSH", "PR", "RF", "BALL", "COV"):
        w = PG.season.map(lambda s, u=u: W[s][u])
        PG[f"p_{u}"] = w * (PG[f"c_{u}"] - PG[f"b_{u}"])
        PG["p_total"] += PG[f"p_{u}"].fillna(0.0)
    # QBs: the starter's value from the team-game table
    Q = TG[["game_id", "season", "team", "side", "starter", "qv"]].rename(columns={"starter": "player_id"})
    Q = Q[Q.player_id.notna()].copy()
    Q["pos"] = "QB"; Q["pgrp"] = "QB"; Q["bucket"] = "QB"; Q["a"] = 1.0; Q["absent"] = 0
    Q["p_QB"] = Q.qv; Q["p_total"] = Q.qv
    keep = ["game_id", "season", "team", "side", "player_id", "pos", "pgrp", "bucket", "a", "absent", "p_total"]
    V = pd.concat([PG[keep + [f"p_{u}" for u in ("REC", "RUSH", "PR", "RF", "BALL", "COV")]],
                   Q[keep + ["p_QB"]]], ignore_index=True)
    V["name"] = V.player_id.map(NAME)
    assert V.season.max() < TEST_ERA
    V.to_parquet(os.path.join(DATA, "pv_nfl_compose_player_values.parquet"), index=False)
    # player-seasons: games where he was a regular member (a >= 0.5) and not absent
    R = V[(V.a >= 0.5) & (V.absent == 0)].copy()
    order = TG[["game_id", "game_date"]].drop_duplicates().set_index("game_id").game_date
    R["date"] = R.game_id.map(order)
    R = R.sort_values("date")
    S = R.groupby(["player_id", "season"]).agg(name=("name", "first"), pos=("pos", "first"), pgrp=("pgrp", "first"),
                                               team=("team", "last"), games=("p_total", "size"),
                                               value_mean=("p_total", "mean"), value_last=("p_total", "last")).reset_index()
    S.to_csv(os.path.join(DATA, "pv_nfl_compose_player_seasons.csv"), index=False, float_format="%.4g")
    json.dump({"design": {"target": "game point differential (home - away)", "train": f"{TRAIN_LO}..s-1",
                          "prior": "w = 1 (nominal EPA = 1 point)", "lambda": "267 * Var_train(dA_k)",
                          "units": UNITS},
               "weights_by_fold": {str(s): W[s] for s in seasons},
               "hfa_by_fold": {str(s): H[s] for s in seasons},
               "weights_by_fold_epa_target": {str(s): We[s] for s in seasons},
               "fold2015_weight_ci95_game_bootstrap": ci15},
              open(os.path.join(DATA, "pv_nfl_compose_weights.json"), "w"), indent=1)
    log(f"wrote player values ({len(V)} rows), player seasons ({len(S)}), weights json")


if __name__ == "__main__":
    main()
