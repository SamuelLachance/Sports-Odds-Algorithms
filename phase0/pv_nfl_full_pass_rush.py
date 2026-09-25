"""Player-value program (pv), NFL FULL-SPAN extension: PASS RUSH + RUN FRONT. DISPLAY ONLY.

Extends pv_nfl_pass_rush_and_front_credits.py / pv_nfl_pass_rush_and_front.py (DEV <= 2015)
through the latest pulled season with FROZEN DEV hyper-parameters (see pv_nfl_full_common):

  stage credits  the unmodified credits script, executed inside a sandbox project root
                 (its ROOT/DATA derive from __file__) with --through <latest>: it reads
                 hard-linked copies of the events / games / players / snap tables and
                 writes its fixed-name outputs into the sandbox; they are then moved to
                 data/pv_nfl_full_front_{credits,units,snaps,olpen}.parquet. The DEV
                 outputs in data/ are never touched. The script prints nothing past 2015.
  stage ratings  the imported build() with its module DEFAULT parameters (the DEV-tuned
                 K / decay / season-carry values, pv_nfl_pass_rush_and_front_tune.json)
                 over every season. Positions: nfl_players.csv as the DEV build read it,
                 plus the weekly roster for ids it does not know (this year's rookies) --
                 identity metadata that only picks the empirical-Bayes prior bucket.

Read-out. build() writes a player's PRE-game state for each game he is present in or
expected for. To read every player's state AFTER his last game, one zero-exposure
"read-out" game per team is appended after the last real date (same season, 0
dropbacks, 0 runs, 0 credits) with every recently credited player listed present:
their pre-game rows ARE the current states, and the league, unit and player states
those rows read were built from real games only (a zero-exposure game adds nothing to
any numerator or denominator before its rows are written). The same rows give each
offence's protection factors (sacks / pressure / run stops ALLOWED vs an average unit,
opponent- and scorer-adjusted: QB + line together -- nobody records which lineman lost).

Outputs
  data/pv_nfl_full_front_current.parquet        per player: th_pr/sk/ht/st/tf (share of an
      average unit's production vs an average opponent), league rates, exposure COUNTS
  data/pv_nfl_full_front_units_current.parquet  per team: D_* (defence) and O_* (offence
      protection) factors, league rates
No realised sack / hit / stop count is stored.

--verify-dev: runs the ratings stage on the DEV credits (<= 2015) and checks it reproduces
data/pv_nfl_pass_rush_and_front_player_games.parquet (a DEV-only check).
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

import pv_nfl_full_common as C

CREDITS = "pv_nfl_pass_rush_and_front_credits"
BUILD = "pv_nfl_pass_rush_and_front"
FULL = {"credits": "pv_nfl_full_front_credits.parquet", "units": "pv_nfl_full_front_units.parquet",
        "snaps": "pv_nfl_full_front_snaps.parquet", "olpen": "pv_nfl_full_front_olpen.parquet"}
DEV_NAMES = {"credits": "pv_nfl_pass_rush_and_front_credits.parquet",
             "units": "pv_nfl_pass_rush_and_front_units.parquet",
             "snaps": "pv_nfl_pass_rush_and_front_snaps.parquet",
             "olpen": "pv_nfl_pass_rush_and_front_olpen.parquet"}
OUT_P = "pv_nfl_full_front_current.parquet"
OUT_U = "pv_nfl_full_front_units_current.parquet"
FAKE_DATE = "9999-12-31"
RECENT = 3              # read out players credited in the last RECENT seasons


def stage_credits(last: int) -> None:
    inputs = ["pv_nfl_events.parquet", "nfl_games.csv", "nfl_players.csv"] + \
             [f"snap_{y}.csv" for y in range(2012, last + 1)]
    with C.sandbox(inputs) as root:
        C.log(f"credits: sandbox run through {last} (silent past 2015)")
        C.exec_script(CREDITS, ["--through", last], fake_root=root)
        for k, dev in DEV_NAMES.items():
            src = os.path.join(root, "data", dev)
            assert os.path.exists(src), f"credits wrote no {dev}"
            dst = os.path.join(C.DATA, FULL[k])
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
    C.log("credits: wrote " + ", ".join(FULL.values()))


def inputs_for(B, files: dict, through: int):
    """the build's load_inputs(), reading the given tables; positions per module doc."""
    pg = pd.read_parquet(os.path.join(C.DATA, files["credits"]))
    u = pd.read_parquet(os.path.join(C.DATA, files["units"]))
    sn = pd.read_parquet(os.path.join(C.DATA, files["snaps"]))
    pg = pg[pg.season <= through].copy()
    u = u[u.season <= through].copy()
    sn = sn[sn.season <= through].copy()
    assert pg.season.max() <= through and u.season.max() <= through
    pg["pr"] = pg.sack + pg.hit
    pl = pd.read_csv(os.path.join(C.DATA, "nfl_players.csv"), usecols=["gsis_id", "display_name", "position"])
    names = dict(zip(pl.gsis_id, pl.display_name))
    pos = C.positions(through) if through >= C.TEST_ERA else dict(zip(pl.gsis_id, pl.position))
    g = pd.read_csv(os.path.join(C.DATA, "nfl_games.csv"), usecols=["game_id", "home_team", "season"])
    g = g[g.season <= through]
    stad = {r.game_id: C.fr(r.home_team) for r in g.itertuples()}
    return pg, u, sn, pos, names, stad


def with_readout(pg, u, last: int):
    """append one zero-exposure read-out game per team after the last real date."""
    teams = sorted(set(u.defteam))
    pgs = pg.sort_values(["game_date", "game_id"], kind="mergesort")
    latest = pgs.groupby("player_id").agg(team=("team", "last"), season=("season", "max"))
    latest = latest[(latest.season >= last - RECENT + 1) & latest.team.isin(teams)]
    fu = pd.DataFrame({"game_id": [f"ZZREADOUT_{t}" for t in teams], "defteam": teams, "posteam": teams,
                       "season": last, "week": 99, "season_type": "REG", "game_date": FAKE_DATE,
                       "n_db": 0, "n_run": 0, "sacks": 0, "hits_ns": 0, "tfl_plays": 0, "stop_plays": 0,
                       "rsucc": 0, "n_plays": 0, "pressure": 0, "pts_for": np.nan})
    fp = pd.DataFrame({"game_id": "ZZREADOUT_" + latest.team, "team": latest.team,
                       "player_id": latest.index, "appeared": 1})
    for c in ("hit", "rtk", "sack", "stop", "tfl", "pen_presnap", "pen_rtp", "pr"):
        fp[c] = 0.0
    fp["season"] = last; fp["week"] = 99; fp["game_date"] = FAKE_DATE
    u2 = pd.concat([u, fu[u.columns]], ignore_index=True)
    pg2 = pd.concat([pg, fp[pg.columns]], ignore_index=True)
    return pg2, u2


def exposure_counts(pg, u, last: int) -> pd.DataFrame:
    """per player, games present (credited or appeared) and the team dropbacks / designed
    runs in those games -- sample sizes, no outcome."""
    x = pg.merge(u[["game_id", "defteam", "n_db", "n_run"]].rename(columns={"defteam": "team"}),
                 on=["game_id", "team"], how="left")
    out = {}
    for yr, tag in ((last, "cur"), (last - 1, "prev")):
        s = x[x.season == yr].groupby("player_id").agg(g=("game_id", "nunique"), db=("n_db", "sum"),
                                                        run=("n_run", "sum"))
        s.columns = [f"g_{tag}", f"db_{tag}", f"run_{tag}"]
        out[tag] = s
    tot = x.groupby("player_id").agg(g_career=("game_id", "nunique"))
    return out["cur"].join(out["prev"], how="outer").join(tot, how="outer").fillna(0)


def stage_ratings(last: int) -> None:
    B = C.import_component(BUILD)
    pg, u, sn, pos, names, stad = inputs_for(B, FULL, last)
    cnt = exposure_counts(pg, u, last)
    pg2, u2 = with_readout(pg, u, last)
    C.log(f"ratings: walk 1999-{last} with the DEV defaults, + {u2.game_id.str.startswith('ZZREADOUT_').sum()} "
          f"read-out games")
    PGo, TGo, _ = B.build(through=last, inputs=(pg2, u2, sn, pos, names, stad))
    ro = PGo[PGo.game_id.str.startswith("ZZREADOUT_")].drop_duplicates("player_id")
    ru = TGo[TGo.game_id.str.startswith("ZZREADOUT_")]
    assert len(ru) == u2.game_id.str.startswith("ZZREADOUT_").sum()
    P = ro[["player_id", "team", "bucket", "th_pr", "th_sk", "th_ht", "th_st", "th_tf", "neff_pr", "neff_st",
            "n_games"]].copy()
    Lr = ru.iloc[0]
    for k in ("pr", "sk", "ht", "st", "tf"):
        P[f"L_{k}"] = float(Lr[f"L_{k}"])
    P = P.merge(cnt, left_on="player_id", right_index=True, how="left").fillna(
        {c: 0 for c in cnt.columns})
    P["name"] = P.player_id.map(names)
    U = ru[["defteam", "D_pr", "D_sk", "D_ht", "D_st", "D_tf", "Oopp_pr", "Oopp_sk", "Oopp_ht", "Oopp_st",
            "Oopp_tf", "L_pr", "L_sk", "L_ht", "L_st", "L_tf"]].rename(columns={"defteam": "team"}).copy()
    U.columns = [c.replace("Oopp_", "O_") for c in U.columns]
    C.write_parquet(P, OUT_P)
    C.write_parquet(U, OUT_U)
    C.record_provenance(BUILD, C.sha256_file(B.__file__), {"front": {"through_season": last}})
    C.log(f"wrote {OUT_P} ({len(P):,} players) and {OUT_U} ({len(U)} teams), state as of season {last}")


def verify_dev() -> None:
    B = C.import_component(BUILD)
    pg, u, sn, pos, names, stad = inputs_for(B, DEV_NAMES, C.TEST_ERA - 1)
    PGo, _, _ = B.build(through=C.TEST_ERA - 1, inputs=(pg, u, sn, pos, names, stad))
    dev = pd.read_parquet(os.path.join(C.DATA, "pv_nfl_pass_rush_and_front_player_games.parquet"))
    cols = ["th_pr", "th_sk", "th_ht", "th_st"]
    m = PGo.merge(dev, on=["game_id", "team", "player_id"], suffixes=("", "_dev"))
    assert len(m) == len(dev) == len(PGo), (len(m), len(dev), len(PGo))
    diffs = {c: float(np.nanmax(np.abs(m[c] - m[c + "_dev"]))) for c in cols}
    for c, v in diffs.items():
        C.log(f"  verify DEV {c}: max |diff| {v:.2e}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["credits", "ratings", "all"], default="all")
    ap.add_argument("--verify-dev", action="store_true")
    a = ap.parse_args()
    if a.verify_dev:
        verify_dev()
        return
    last = C.events_span()["season"]
    if a.stage in ("credits", "all"):
        stage_credits(last)
    if a.stage in ("ratings", "all"):
        stage_ratings(last)


if __name__ == "__main__":
    main()
