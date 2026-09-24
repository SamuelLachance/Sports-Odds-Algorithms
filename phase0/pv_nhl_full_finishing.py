"""Player-value program, NHL -- FULL (DEV + TEST) run of component FINISHING_AND_SAVING.

Runs phase0/pv_nhl_finishing_and_saving.py (unmodified) through 2025-26 with the
hyperparameters FROZEN from its DEV fit (data/pv_nhl_finishing_and_saving_params.json
-> params: per-group S0 / RHO_G / RHO_S, and the fixed CELL_SD0 / CELL_LAM; never
re-tuned). The online state (player / team / rink beliefs and the xG recalibration
cells) keeps filtering through TEST exactly as serving would.

One DEV-reproduction detail. prep() assigns each player ONE entity group (G / F / D)
from his majority roster position over every game it loads. With TEST rosters loaded,
2 DEV players' majority would flip (8474722, 8477502: D -> F), which would change DEV
values. Here every player seen in DEV keeps the group of his DEV majority (TEST roster
rows of those players are presented to prep() with their DEV-majority position; the
real per-game position is restored in the output); players first seen in TEST get the
unchanged majority rule over their own rows. Position is a pre-game roster label, not
an outcome.

Outputs (never the DEV files):
  data/pv_nhl_full_finishing_and_saving_player_games.parquet
  data/pv_nhl_full_finishing_and_saving_entity_games.parquet
  data/pv_nhl_full_finishing_and_saving_shots.parquet
  data/pv_nhl_full_finishing_and_saving_final_state.parquet
  data/pv_nhl_full_finishing_and_saving_team_games.csv

    python phase0/pv_nhl_full_finishing.py
"""
from __future__ import annotations

import json
import time

import numpy as np

import pv_nhl_full_common as FC
import pv_nhl_io as IO
import pv_nhl_finishing_and_saving as FS

_REAL_POS = {}


def rosters_dev_groups(allow_test=True):
    ro = IO.load_rosters(allow_test=True)
    dev = ro[ro.gid < FC.DEV_MAX_GID]
    maj = dev.groupby("pid").pos.agg(lambda x: x.value_counts().index[0])
    test = (ro.gid >= FC.DEV_MAX_GID) & ro.pid.isin(maj.index)
    _REAL_POS["frame"] = ro[["gid", "pid", "pos"]].copy()
    ro = ro.copy()
    ro.loc[test, "pos"] = ro.loc[test, "pid"].map(maj).to_numpy()
    return ro


def main():
    t0 = time.time()
    FS.load_rosters = rosters_dev_groups
    D = FS.prep(allow_test=True)
    # the DEV group of every DEV player is unchanged (asserted)
    ro_dev = IO.load_rosters()
    maj_dev = ro_dev.groupby("pid").pos.agg(lambda x: x.value_counts().index[0]).map(FS._group)
    grp_of = dict(zip(D["pids"], np.array(FS.GROUPS)[D["e_group"][:D["P"]]]))
    assert all(grp_of[p] == g for p, g in maj_dev.items() if p in grp_of)
    cfg = {g: tuple(v) for g, v in json.load(open(FS.OUT_PAR))["params"].items()}
    print("frozen DEV params", cfg, flush=True)
    pg, eg, sh, final, c = FS.build_outputs(D, cfg, allow_test=True)
    # restore the real per-game roster position label (cosmetic; not used downstream)
    rp = _REAL_POS["frame"].rename(columns={"pos": "pos_real"})
    pg = pg.merge(rp, on=["gid", "pid"], how="left")
    pg["pos"] = np.where(pg.pos_real.notna(), pg.pos_real, pg.pos)
    pg = pg.drop(columns="pos_real")
    for df, name in ((pg, "pv_nhl_finishing_and_saving_player_games.parquet"),
                     (eg, "pv_nhl_finishing_and_saving_entity_games.parquet"),
                     (sh, "pv_nhl_finishing_and_saving_shots.parquet"),
                     (final, "pv_nhl_finishing_and_saving_final_state.parquet")):
        df.to_parquet(FC.full_path(name), index=False)
    tg = FS.build_team_games(pg, eg)
    # build_team_games also attaches realised post-game LABELS (goals, goals above
    # xG). They are never a feature; for TEST rows they are blanked so no TEST
    # residual exists on disk (compose reads only lg_xg_pre from this file).
    te = tg.gid >= FC.DEV_MAX_GID
    for c in ("gf", "ga", "fin_real", "sav_real"):
        tg[c] = tg[c].astype(float)
        tg.loc[te, c] = np.nan
    tg.to_csv(FC.full_path("pv_nhl_finishing_and_saving_team_games.csv"), index=False)
    print(f"wrote {len(pg)} player-games, {len(eg)} entity-games, {len(sh)} shots "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
