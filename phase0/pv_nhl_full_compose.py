"""Player-value program, NHL -- FULL (DEV + TEST) COMPOSE: panel -> lineup values ->
per-game features, under the pre-registered TEST protocol
(data/pv_nhl_serve_prereg.json -> test_protocol).

Wraps the unmodified phase0/pv_nhl_compose_panel.py, pv_nhl_compose_core.py and
pv_nhl_compose_fit.py.build_features, reading the full component files
(data/pv_nhl_full_*; built by pv_nhl_full_{creation,finishing,duels,defense_onice}.py)
and writing only data/pv_nhl_full_compose_*.

What is frozen (never fit on TEST data):
  * the declared spec of pv_nhl_compose_core (components, PRIOR_W, PRIOR_TAU,
    CENTRE_HL, DEBUT_MIN, FIRST_FIT_SEASON);
  * the goalie design rule: G2 was REJECTED on DEV (data/pv_nhl_compose_weights.json
    -> goalie_rule), so the goalie value is the 1v1 saving rating (adopt=False, no
    rho offsets);
  * compose weights: DEV seasons keep their DEV walk-forward weights; EVERY TEST
    season uses W[2018-19] = the ridge fit on DEV regular-season games 2011-12..2017-18
    (C.walk_forward_weights with seasons=[20182019] on the DEV rows only).
What keeps walking forward (state, not fitted): the position-group centring means
(decayed, strictly earlier dates) and every component rating.

TEST outcome hygiene: the panel's team-game labels (goals by strength, goal
differential) are computed for DEV games only; TEST team-game rows carry the game
facts (date, season, type, home, away) and NaN labels. Nothing is scored.

    python phase0/pv_nhl_full_compose.py
Outputs:
  data/pv_nhl_full_compose_{skater,goalie,team}_games.parquet   panel
  data/pv_nhl_full_compose_weights.json                          weights used per season
  data/pv_nhl_full_compose_game_features.csv                     per game, DEV + TEST
  data/pv_nhl_full_compose_player_values.parquet, _goalie_values.csv
  data/pv_nhl_full_serve_lv_last.csv                             S1 input: each side's LV
                                                                 from its PREVIOUS game
"""
from __future__ import annotations

import ast
import copy
import json
import os
import time

import numpy as np
import pandas as pd

import pv_nhl_full_common as FC
import pv_nhl_compose_panel as P
import pv_nhl_compose_core as C
import pv_nhl_compose_fit as CF
from nhl_features_eval import TEAM_FIX

FULL_INPUTS = {"pv_nhl_creation_wf.parquet", "pv_nhl_creation_pg.parquet",
               "pv_nhl_finishing_and_saving_player_games.parquet",
               "pv_nhl_discrete_duels.csv", "pv_nhl_defense_onice_values.csv",
               "pv_nhl_finishing_and_saving_team_games.csv",
               "pv_nhl_compose_skater_games.parquet", "pv_nhl_compose_goalie_games.parquet",
               "pv_nhl_compose_team_games.parquet"}


def D_full(p):
    return FC.full_path(p) if p in FULL_INPUTS else FC.D(p)


# ------------------------------------------------------------------ panel ---
_ORIG_TEAM_GOALS = P.team_goals


def team_goals_full():
    """DEV: the original labels (computed exactly as the DEV run, DEV loaders).
    TEST: one row per team-game with the game facts and NaN labels."""
    P.DEV_MAX_GID = FC.DEV_MAX_GID
    try:
        tg_dev = _ORIG_TEAM_GOALS()
    finally:
        P.DEV_MAX_GID = FC.NO_GUARD
    games = pd.read_csv(FC.D("nhl_games.csv"), usecols=["game_id", "date", "season", "type",
                                                         "home", "away"])
    games = games[games.game_id >= FC.DEV_MAX_GID].rename(columns={"game_id": "gid"})
    rows = []
    for side in ("H", "A"):
        t = games.copy()
        t["side"] = side
        rows.append(t)
    tt = pd.concat(rows, ignore_index=True)
    tt["team"] = np.where(tt.side == "H", tt.home, tt.away)
    tt["opp"] = np.where(tt.side == "H", tt.away, tt.home)
    tt = tt.reindex(columns=tg_dev.columns)          # labels -> NaN
    lab = [c for c in tg_dev.columns if c not in ("gid", "date", "season", "type", "home",
                                                   "away", "side", "team", "opp")]
    assert tt[lab].isna().all().all()
    return pd.concat([tg_dev, tt.astype({c: float for c in lab})], ignore_index=True)


def patch_panel():
    P.D = D_full
    P.DEV_MAX_GID = FC.NO_GUARD
    # P.load_events stays the DEV-only loader: the panel reads the event archive only
    # inside team_goals (labels), which must never see a TEST game
    P.team_goals = team_goals_full
    P.OUT_SK = FC.full_path("pv_nhl_compose_skater_games.parquet")
    P.OUT_GK = FC.full_path("pv_nhl_compose_goalie_games.parquet")
    P.OUT_TG = FC.full_path("pv_nhl_compose_team_games.parquet")


# ---------------------------------------------------------------- weights ---
_ORIG_WFW = C.walk_forward_weights
FROZEN = {}


def walk_forward_weights_frozen(G, comps=C.COMPS, target="gd_noen", prior=None, tau=None,
                                seasons=C.SEASONS):
    assert list(seasons) == list(C.SEASONS)
    Gd = G[G.gid < FC.DEV_MAX_GID]
    assert int(Gd.season.max()) <= 20172018
    W = _ORIG_WFW(Gd, comps, target, prior, tau, seasons)           # DEV, as in DEV
    wf = _ORIG_WFW(Gd, comps, target, prior, tau, [FC.FREEZE_SEASON])[FC.FREEZE_SEASON]
    assert wf["n"] == int(((Gd.season >= C.FIRST_FIT_SEASON)
                           & (Gd.season <= 20172018)).sum())
    FROZEN["w"] = wf
    for s in FC.TEST_SEASONS:
        W[s] = copy.deepcopy(wf)
    return W


def patch_core():
    C.D = D_full
    C.DEV_MAX_GID = FC.NO_GUARD
    C.walk_forward_weights = walk_forward_weights_frozen
    CF.DEV_MAX_GID = FC.NO_GUARD


def lv_last_fn():
    """pv_nhl_serve_screen.lv_last, taken verbatim from its source (not re-typed)."""
    src = open(os.path.join(FC.HERE, "pv_nhl_serve_screen.py"), encoding="utf-8").read()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
          and n.name == "lv_last"][0]
    ns = {"pd": pd, "TEAM_FIX": TEAM_FIX}
    exec(compile(ast.Module([fn], []), "pv_nhl_serve_screen.py:lv_last", "exec"), ns)
    return ns["lv_last"]


def r6(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 6)


def main():
    t0 = time.time()
    patch_panel()
    P.main()
    print(f"panel {time.time()-t0:.0f}s", flush=True)

    patch_core()
    sk, gk, tg = C.load_panel()
    # G2 goalie rule: frozen at the DEV decision (rejected)
    wdev = json.load(open(FC.D("pv_nhl_compose_weights.json")))
    assert wdev["goalie_rule"].startswith("G2 REJECTED")
    F, W, L, G, st, pv, x, m = CF.build_features(sk, gk, tg, rho=None, adopt=False)
    F.to_csv(FC.full_path("pv_nhl_compose_game_features.csv"), index=False,
             float_format="%.6f")
    wjson = {"protocol": "data/pv_nhl_serve_prereg.json -> test_protocol",
             "prior": {"w": C.PRIOR_W, "tau": C.PRIOR_TAU},
             "goalie_rule": wdev["goalie_rule"] + " (DEV decision, frozen)",
             "frozen_test_weights": {"source_season": FC.FREEZE_SEASON,
                                     "fit_on": "DEV regular season 2011-12..2017-18",
                                     "n_train_games": FROZEN["w"]["n"],
                                     "intercept": FROZEN["w"]["b"],
                                     "w": FROZEN["w"]["w"], "se": FROZEN["w"]["se"],
                                     "applied_to": FC.TEST_SEASONS},
             "walk_forward": {str(s): {"n_train_games": v["n"], "intercept": r6(v["b"]),
                                       "w": {c: r6(val) for c, val in v["w"].items()}}
                              for s, v in W.items()}}
    json.dump(wjson, open(FC.full_path("pv_nhl_compose_weights.json"), "w"), indent=1)
    pvo = pd.concat([sk[["gid", "date", "season", "gtype", "team", "side", "pid", "grp"]],
                     m.rename(columns={"ev": "m_ev", "pp": "m_pp", "all": "m_all"}),
                     x.add_prefix("x_"), pv], axis=1)
    pvo.to_parquet(FC.full_path("pv_nhl_compose_player_values.parquet"), index=False)
    st[["gid", "side", "g_pid", "season", "sav_mu", "sav_sd", "lg_xg_pre", "goalie", "gv"]
       ].to_csv(FC.full_path("pv_nhl_compose_goalie_values.csv"), index=False,
                float_format="%.6f")
    # S1 input (pre-registered construction), from the written file exactly as the
    # screen reads its feature file
    Fr = pd.read_csv(FC.full_path("pv_nhl_compose_game_features.csv"))
    LL = lv_last_fn()(Fr)
    out = Fr[["gid", "date", "season", "gtype", "home", "away"]].merge(
        LL.reset_index(), on="gid", how="left")
    out.to_csv(FC.D("pv_nhl_full_serve_lv_last.csv"), index=False)
    print(f"features {len(F):,} games ({int((F.gid >= FC.DEV_MAX_GID).sum()):,} TEST) "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
