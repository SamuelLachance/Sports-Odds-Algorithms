"""Player-value program, NHL -- COMPOSE step 1: the player-game panel (DEV ONLY).

Joins the four validated component files into one row per dressed player per DEV
game, every value a PRE-GAME walk-forward snapshot written by its own component
(each component's truncation / permutation audit passed there):

  creation   data/pv_nhl_creation_wf.parquet           EB-shrunk individual rates
             (5v5 = "ev", power play = "pp"), expected 5v5 / PP minutes, and the
             incumbent walk-forward xG stint-RAPM value (rapm_net_v, rapmel fits)
  finishing  data/pv_nhl_finishing_and_saving_player_games.parquet
             1v1 shooter finishing logit (skaters) / goalie saving logit (goalies)
  duels      data/pv_nhl_discrete_duels.csv            fo_g60, pen_g60, dd_g60,
             expected all-situations TOI
  defense    data/pv_nhl_defense_onice_values.csv       d_xga (xG-against RAPM,
             xG/60 prevented, corrected event window; regular season only)

plus SAME-GAME realised facts used ONLY as labels / accumulator inputs:
  data/pv_nhl_creation_pg.parquet (TOI and on-ice goals by strength) and the
  event archive (team goals by strength, empty-net and shootout goals excluded).

Outputs (DEV, gid < 2018000000 asserted):
  data/pv_nhl_compose_skater_games.parquet   one row per skater-game
  data/pv_nhl_compose_goalie_games.parquet   one row per goalie-game (starter flag,
                                             saving logit, realised xGA / GA)
  data/pv_nhl_compose_team_games.parquet     one row per team-game: realised goals
                                             by strength (labels), league xGA/game
                                             of strictly earlier dates

Market-blind: no odds are read anywhere.

    python phase0/pv_nhl_compose_panel.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pv_nhl_io import DEV_MAX_GID, load_events, load_rosters  # noqa: E402

D = lambda p: os.path.join(ROOT, "data", p)  # noqa: E731
OUT_SK = D("pv_nhl_compose_skater_games.parquet")
OUT_GK = D("pv_nhl_compose_goalie_games.parquet")
OUT_TG = D("pv_nhl_compose_team_games.parquet")

CRE_COLS = ["gid", "pid", "date", "season", "gtype", "team", "side", "pos", "grp",
            "n_prior_games", "prior_min_ev", "prior_min_pp", "prior_min_all",
            "exp_min_ev", "exp_min_pp",
            "ixg_ev_hat", "g_ev_hat", "a1_ev_hat", "a2_ev_hat", "cre_ev", "cre_evx",
            "ixg_pp_hat", "g_pp_hat", "a1_pp_hat", "a2_pp_hat", "cre_pp",
            "on_gf_ev_hat", "on_xgf_ev_hat", "rel_gf_hat", "rel_xgf_hat",
            "rapm_off_s", "rapm_net_v", "rapm_cutoff"]
PG_COLS = ["gid", "pid", "toi_s", "toi_ev", "toi_pp", "toi_sh",
           "on_gf_ev", "on_ga_ev", "on_xgf_ev", "on_xga_ev",
           "on_gf_pp", "on_ga_pp", "on_gf_sh", "on_ga_sh",
           "tm_gf_ev", "tm_ga_ev", "tm_xgf_ev", "tm_xga_ev", "tm_toi_ev",
           "tm_gf_pp", "tm_ga_pp", "tm_gf_sh", "tm_ga_sh",
           "g_ev", "a1_ev", "a2_ev", "ixg_ev", "g_pp", "a1_pp", "a2_pp"]
DD_COLS = ["gid", "pid", "fo_g60", "pen_g60", "tkgv_g60", "dd_g60", "exp_toi_s"]
DF_COLS = ["gid", "pid", "fit", "known", "new", "m5", "d_xga", "d_ga", "d_gax"]


def dev(df, name):
    assert len(df) and int(df.gid.max()) < DEV_MAX_GID, f"TEST gid in {name}"
    return df


def team_goals():
    """Realised goals per team-game by strength (labels only).

    Excluded: shootout goals (ptype SO) and goals into an EMPTY net (defending
    goalie not in net, g_ag == 0). Strength from the acting (scoring) side:
    s5 = 5v5 with both goalies in; pp = more skaters; sh = fewer; oth = rest
    (4v4, 3v3 OT, ...)."""
    ev = dev(load_events(columns=["season", "gtype", "ptype", "ev", "team", "is_home",
                                  "sk_for", "sk_ag", "g_ag", "a_g", "h_g", "xg"]),
             "events")
    g = ev[(ev.ev == "goal") & (ev.ptype != "SO")].copy()
    g["en"] = (g.g_ag == 0)
    g["st"] = np.select([(g.sk_for == 5) & (g.sk_ag == 5) & (g.a_g == 1) & (g.h_g == 1),
                         g.sk_for > g.sk_ag, g.sk_for < g.sk_ag], ["s5", "pp", "sh"], "oth")
    g.loc[g.en, "st"] = "en"
    g["side"] = np.where(g.is_home == 1, "H", "A")
    tab = g.groupby(["gid", "side", "st"]).size().unstack("st", fill_value=0)
    for c in ("s5", "pp", "sh", "oth", "en"):
        if c not in tab:
            tab[c] = 0
    tab = tab[["s5", "pp", "sh", "oth", "en"]].add_prefix("gf_")
    # unblocked non-shootout xG per side (MoneyPuck), all strengths, non-EN
    sh = ev[(ev.ptype != "SO") & ev.ev.isin(["goal", "shot-on-goal", "missed-shot"])
            & (ev.g_ag == 1)].copy()
    sh["side"] = np.where(sh.is_home == 1, "H", "A")
    xg = sh.groupby(["gid", "side"]).xg.sum().rename("xgf_all")
    games = pd.read_csv(D("nhl_games.csv"), usecols=["game_id", "date", "season", "type",
                                                     "home", "away"])
    games = games[games.game_id < DEV_MAX_GID].rename(columns={"game_id": "gid"})
    rows = []
    for side in ("H", "A"):
        t = games.copy()
        t["side"] = side
        rows.append(t)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg.merge(tab.reset_index(), on=["gid", "side"], how="left")
    tg = tg.merge(xg.reset_index(), on=["gid", "side"], how="left")
    for c in [c for c in tg.columns if c.startswith("gf_")] + ["xgf_all"]:
        tg[c] = tg[c].fillna(0.0)
    opp = tg[["gid", "side"] + [c for c in tg.columns if c.startswith("gf_")]
             + ["xgf_all"]].copy()
    opp["side"] = np.where(opp.side == "H", "A", "H")
    opp = opp.rename(columns={c: "ga_" + c[3:] for c in opp.columns if c.startswith("gf_")}
                     ).rename(columns={"xgf_all": "xga_all"})
    tg = tg.merge(opp, on=["gid", "side"], how="left")
    tg["gf_noen"] = tg.gf_s5 + tg.gf_pp + tg.gf_sh + tg.gf_oth
    tg["ga_noen"] = tg.ga_s5 + tg.ga_pp + tg.ga_sh + tg.ga_oth
    tg["gd_noen"] = tg.gf_noen - tg.ga_noen
    tg["gd_s5"] = tg.gf_s5 - tg.ga_s5
    tg["gd_st"] = (tg.gf_pp + tg.gf_sh) - (tg.ga_pp + tg.ga_sh)
    tg["team"] = np.where(tg.side == "H", tg.home, tg.away)
    tg["opp"] = np.where(tg.side == "H", tg.away, tg.home)
    return dev(tg, "team_games")


def main():
    t0 = time.time()
    cre = dev(pd.read_parquet(D("pv_nhl_creation_wf.parquet"), columns=CRE_COLS), "creation")
    pg = dev(pd.read_parquet(D("pv_nhl_creation_pg.parquet"), columns=PG_COLS), "pg")
    fs = dev(pd.read_parquet(D("pv_nhl_finishing_and_saving_player_games.parquet")), "fs")
    dd = dev(pd.read_csv(D("pv_nhl_discrete_duels.csv"), usecols=DD_COLS), "duels")
    df = dev(pd.read_csv(D("pv_nhl_defense_onice_values.csv"), usecols=DF_COLS), "defense")
    print(f"loaded {time.time()-t0:.0f}s: cre {len(cre):,} pg {len(pg):,} fs {len(fs):,} "
          f"dd {len(dd):,} def {len(df):,}", flush=True)

    fsk = fs[fs.grp != "G"][["gid", "pid", "mu", "sd"]].rename(
        columns={"mu": "fin_mu", "sd": "fin_sd"})
    sk = cre.merge(fsk, on=["gid", "pid"], how="left", validate="1:1")
    sk = sk.merge(dd, on=["gid", "pid"], how="left", validate="1:1")
    sk = sk.merge(df, on=["gid", "pid"], how="left", validate="1:1")
    sk = sk.merge(pg, on=["gid", "pid"], how="left", validate="1:1")
    cov = {c: float(sk[c].notna().mean()) for c in ["fin_mu", "dd_g60", "d_xga", "toi_ev",
                                                     "exp_min_ev", "rapm_net_v"]}
    cov_reg = {c: float(sk.loc[sk.gtype == 2, c].notna().mean()) for c in ["d_xga"]}
    print("coverage", cov, "d_xga regular season", cov_reg, flush=True)
    assert cov["fin_mu"] > 0.999 and cov["dd_g60"] > 0.999 and cov["toi_ev"] > 0.999
    sk = sk.sort_values(["date", "gid", "side", "pid"]).reset_index(drop=True)
    sk.to_parquet(OUT_SK, index=False)

    # goalies: every dressed goalie-game with the pre-game saving logit
    gk = fs[fs.grp == "G"][["gid", "gi", "date", "season", "gtype", "team", "side_h", "pid",
                            "g_start", "g_net", "toi_s", "mu", "sd", "fa_n", "fa_xg",
                            "fa_xrc", "fa_g"]].copy()
    gk["side"] = np.where(gk.side_h == 1, "H", "A")
    gk = gk.rename(columns={"mu": "sav_mu", "sd": "sav_sd"})
    gk = gk.sort_values(["gi", "side", "pid"]).reset_index(drop=True)
    assert int((gk.g_start == 1).sum()) == 2 * gk.gid.nunique(), "one starter per side"
    gk.to_parquet(OUT_GK, index=False)

    tg = team_goals()
    # league recalibrated xGA per team-game of strictly earlier dates (the saving scale)
    ftg = pd.read_csv(D("pv_nhl_finishing_and_saving_team_games.csv"),
                      usecols=["gid", "side_h", "lg_xg_pre"])
    ftg["side"] = np.where(ftg.side_h == 1, "H", "A")
    tg = tg.merge(ftg[["gid", "side", "lg_xg_pre"]], on=["gid", "side"], how="left")
    ev_games = set(sk.gid.unique())
    tg = tg[tg.gid.isin(ev_games)].reset_index(drop=True)
    tg.to_parquet(OUT_TG, index=False)
    print(f"team-games {len(tg):,}  gd_noen mean(H) "
          f"{tg[tg.side=='H'].gd_noen.mean():.3f}  EN goals {int(tg.gf_en.sum())}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
