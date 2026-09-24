"""Player-value program (pv), NFL, component `pass_rush_and_front` -- step 1: CREDITS.

Reduces data/pv_nfl_events.parquet (DEV + warm-up seasons ONLY, <= 2015 by
default) to the individually-credited defensive-front events and the unit-level
exposures they are rated against. No rating, no statistic about the future.

Per play (scrimmage plays: play_type in pass/run, aborted plays kept):
  dropback       qb_dropback == 1 (attempts, sacks, scrambles)
  designed run   play_type == run, not a scramble, not a kneel
  sack credit    sack_credit ('id:1' or 'id:0.5;id:0.5') -> weight 1 / 0.5
  QB hit         qb_hit_1/2_player_id on NON-sack plays, 1/n_hitters each
                 (on sack plays the hit credit duplicates the sacker 99.8%)
  run TFL        designed run with yards_gained < 0: every credited defensive
                 tackler (solo 1/2, tackle_with_assist 1/2, assist 1..4 whose
                 *_team == defteam) shares 1 credit equally. Derived from the
                 tackler ids in EVERY season, because tackle_for_loss_*_id is
                 empty 2003-2007 (scorer drift, see events notes).
  run stop       designed run that FAILED by the yards rule (1st down: < 40% of
                 ydstogo, 2nd: < 60%, 3rd/4th: < 100%; a TD is never a
                 failure) -- tacklers share 1 credit. TFL is a subset.
  appearance     any defensive credit on a scrimmage play for defteam: tackle,
                 assist, sack, hit, pass defensed, INT, forced fumble,
                 fumble recovery, defensive penalty. Used only as an exposure
                 proxy for seasons without snap counts (pre-2013).
  OL discipline  offensive holding / false start charged to an offensive
                 player (penalty_player_id, penalty_team == posteam), incl.
                 no_play rows -- the only OL-individual events in the pbp.

Unit level per (game, defteam): dropbacks, designed runs, sacks (flag),
non-sack hits (flag), pressure = sack+hit, run TFL plays, run-stop plays,
yards-rule run successes allowed, plus the offence's points from the schedule.

Outputs (walk-forward-neutral: facts about each game only)
  data/pv_nfl_pass_rush_and_front_credits.parquet   player x game credits
  data/pv_nfl_pass_rush_and_front_units.parquet     defteam x game unit facts
  data/pv_nfl_pass_rush_and_front_snaps.parquet     2013-2015 snap shares (gsis)

Locked split: default --through 2015. Building TEST seasons requires the
explicit --through flag (lead engineer, serving only); nothing about a season
>= 2016 is printed either way.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}

TACKLE_COLS = [("solo_tackle_1_player_id", "solo_tackle_1_team"),
               ("solo_tackle_2_player_id", "solo_tackle_2_team"),
               ("tackle_with_assist_1_player_id", "tackle_with_assist_1_team"),
               ("tackle_with_assist_2_player_id", "tackle_with_assist_2_team"),
               ("assist_tackle_1_player_id", "assist_tackle_1_team"),
               ("assist_tackle_2_player_id", "assist_tackle_2_team"),
               ("assist_tackle_3_player_id", "assist_tackle_3_team"),
               ("assist_tackle_4_player_id", "assist_tackle_4_team")]
OTHER_DEF = [("pass_defense_1_player_id", None), ("pass_defense_2_player_id", None),
             ("interception_player_id", None),
             ("forced_fumble_player_1_player_id", "forced_fumble_player_1_team"),
             ("forced_fumble_player_2_player_id", "forced_fumble_player_2_team"),
             ("fumble_recovery_1_player_id", "fumble_recovery_1_team"),
             ("qb_hit_1_player_id", None), ("qb_hit_2_player_id", None),
             ("sack_player_id", None), ("half_sack_1_player_id", None), ("half_sack_2_player_id", None)]
OL_PEN = {"Offensive Holding", "False Start", "Offensive Offside", "Illegal Use of Hands",
          "Chop Block", "Illegal Block Above the Waist", "Offensive Pass Interference"}
OL_PEN_CORE = {"Offensive Holding", "False Start"}


def load(through: int) -> pd.DataFrame:
    tc = [c for pair in TACKLE_COLS for c in pair]
    oc = [c for pair in OTHER_DEF for c in pair if c]
    cols = sorted(set(["game_id", "play_id", "season", "season_type", "week", "game_date", "home_team",
                       "away_team", "posteam", "defteam", "play_type", "down", "ydstogo", "yards_gained",
                       "qb_dropback", "qb_scramble", "qb_kneel", "sack", "qb_hit", "touchdown", "aborted_play",
                       "penalty", "penalty_type", "penalty_team", "penalty_player_id", "sack_credit"] + tc + oc))
    df = pd.read_parquet(os.path.join(DATA, "pv_nfl_events.parquet"), columns=cols,
                         filters=[("season", "<=", through)])
    assert df.season.max() <= through
    return df


def yards_fail(df: pd.DataFrame) -> np.ndarray:
    yg = df.yards_gained.fillna(0).to_numpy(float)
    togo = df.ydstogo.fillna(10).to_numpy(float)
    dn = df.down.fillna(1).to_numpy(float)
    need = np.where(dn == 1, 0.4 * togo, np.where(dn == 2, 0.6 * togo, togo))
    ok = (yg >= need) | (df.touchdown.fillna(0).to_numpy() == 1)
    return ~ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=TEST_ERA - 1)
    a = ap.parse_args()
    quiet = a.through >= TEST_ERA
    df = load(a.through)
    df["defteam"] = df.defteam.map(lambda t: FR.get(t, t))
    df["posteam"] = df.posteam.map(lambda t: FR.get(t, t))
    scrim = df.play_type.isin(["pass", "run"])
    s = df[scrim].copy()
    s["is_db"] = (s.qb_dropback == 1).astype(int)
    s["is_run"] = ((s.play_type == "run") & (s.qb_scramble.fillna(0) != 1) & (s.qb_kneel.fillna(0) != 1)).astype(int)
    s["is_sack"] = (s.sack == 1).astype(int)
    s["is_hit_ns"] = ((s.qb_hit == 1) & (s.sack != 1)).astype(int)
    s["is_tfl"] = ((s.is_run == 1) & (s.yards_gained < 0)).astype(int)
    s["is_stop"] = ((s.is_run == 1) & yards_fail(s)).astype(int)
    s["is_rsucc"] = ((s.is_run == 1) & (s.is_stop == 0)).astype(int)

    # ------------------------------------------------ player credit rows ----
    rows = []   # (game_id, play_id, team, player, kind, credit)
    # sacks
    sk = s[s.is_sack.eq(1) & s.sack_credit.notna()][["game_id", "play_id", "defteam", "sack_credit"]]
    for g, p, t, c in sk.itertuples(index=False):
        for part in str(c).split(";"):
            pid, w = part.rsplit(":", 1)
            rows.append((g, p, t, pid, "sack", float(w)))
    # non-sack hits
    h = s[s.is_hit_ns.eq(1)][["game_id", "play_id", "defteam", "qb_hit_1_player_id", "qb_hit_2_player_id"]]
    for g, p, t, h1, h2 in h.itertuples(index=False):
        ids = [x for x in (h1, h2) if isinstance(x, str)]
        for pid in ids:
            rows.append((g, p, t, pid, "hit", 1.0 / len(ids)))
    # tacklers on designed runs (TFL / stop / any tackle)
    r = s[s.is_run.eq(1)]
    tk_ids = np.full((len(r), len(TACKLE_COLS)), None, dtype=object)
    for j, (pc, tc) in enumerate(TACKLE_COLS):
        ok = r[pc].notna() & (r[tc].map(lambda t: FR.get(t, t)) == r.defteam)
        tk_ids[:, j] = np.where(ok, r[pc], None)
    gid, pid_, dt = r.game_id.to_numpy(), r.play_id.to_numpy(), r.defteam.to_numpy()
    tfl, stop = r.is_tfl.to_numpy(), r.is_stop.to_numpy()
    for i in range(len(r)):
        ids = list(dict.fromkeys(x for x in tk_ids[i] if x is not None))
        if not ids:
            continue
        w = 1.0 / len(ids)
        for x in ids:
            rows.append((gid[i], pid_[i], dt[i], x, "rtk", w))
            if tfl[i]:
                rows.append((gid[i], pid_[i], dt[i], x, "tfl", w))
            if stop[i]:
                rows.append((gid[i], pid_[i], dt[i], x, "stop", w))
    cr = pd.DataFrame(rows, columns=["game_id", "play_id", "team", "player_id", "kind", "credit"])

    # ------------------------------------------------ appearance (defence) ----
    app = []
    allc = TACKLE_COLS + OTHER_DEF
    for pc, tc in allc:
        sub = s[s[pc].notna()]
        if tc:
            sub = sub[sub[tc].map(lambda t: FR.get(t, t)) == sub.defteam]
        app.append(sub[["game_id", "defteam", pc]].rename(columns={pc: "player_id", "defteam": "team"}))
    pen = df[(df.penalty == 1) & df.penalty_player_id.notna()].copy()
    pen["penalty_team"] = pen.penalty_team.map(lambda t: FR.get(t, t))
    dpen = pen[pen.penalty_team == pen.defteam]
    app.append(dpen[["game_id", "defteam", "penalty_player_id"]].rename(
        columns={"penalty_player_id": "player_id", "defteam": "team"}))
    app = pd.concat(app).drop_duplicates()
    app["appeared"] = 1

    # player x game table
    pg = cr.pivot_table(index=["game_id", "team", "player_id"], columns="kind", values="credit",
                        aggfunc="sum", fill_value=0.0).reset_index()
    for k in ("sack", "hit", "rtk", "tfl", "stop"):
        if k not in pg:
            pg[k] = 0.0
    pg = app.merge(pg, on=["game_id", "team", "player_id"], how="outer")
    pg[["sack", "hit", "rtk", "tfl", "stop"]] = pg[["sack", "hit", "rtk", "tfl", "stop"]].fillna(0.0)
    pg["appeared"] = pg.appeared.fillna(1).astype(int)
    # defensive penalties by type (pre-snap / roughing) for front discipline
    dpen2 = dpen.assign(pre=dpen.penalty_type.isin(["Defensive Offside", "Neutral Zone Infraction",
                                                     "Encroachment"]).astype(float),
                        rtp=(dpen.penalty_type == "Roughing the Passer").astype(float))
    dp = dpen2.groupby(["game_id", "defteam", "penalty_player_id"])[["pre", "rtp"]].sum().reset_index().rename(
        columns={"defteam": "team", "penalty_player_id": "player_id", "pre": "pen_presnap", "rtp": "pen_rtp"})
    pg = pg.merge(dp, on=["game_id", "team", "player_id"], how="left").fillna({"pen_presnap": 0.0, "pen_rtp": 0.0})

    # ------------------------------------------------ OL discipline rows ----
    open_ = pen[(pen.penalty_team == pen.posteam) & pen.penalty_type.isin(OL_PEN)]
    ol = open_.assign(core=open_.penalty_type.isin(OL_PEN_CORE).astype(float),
                      hold=(open_.penalty_type == "Offensive Holding").astype(float),
                      fs=(open_.penalty_type == "False Start").astype(float))
    ol = ol.groupby(["game_id", "posteam", "penalty_player_id"])[["core", "hold", "fs"]].sum().reset_index().rename(
        columns={"posteam": "team", "penalty_player_id": "player_id", "core": "olpen_core", "hold": "olpen_hold",
                 "fs": "olpen_fs"})

    # ------------------------------------------------ unit table ----
    u = s.groupby(["game_id", "defteam", "posteam"]).agg(
        season=("season", "first"), week=("week", "first"), season_type=("season_type", "first"),
        game_date=("game_date", "first"), n_db=("is_db", "sum"), n_run=("is_run", "sum"),
        sacks=("is_sack", "sum"), hits_ns=("is_hit_ns", "sum"), tfl_plays=("is_tfl", "sum"),
        stop_plays=("is_stop", "sum"), rsucc=("is_rsucc", "sum"), n_plays=("play_id", "size")).reset_index()
    u["pressure"] = u.sacks + u.hits_ns
    # points (schedule; points are the outcome, not an input to any rating here)
    gm = pd.read_csv(os.path.join(DATA, "nfl_games.csv"), usecols=["game_id", "season", "home_team", "away_team",
                                                                  "home_score", "away_score"])
    gm = gm[gm.season <= a.through]
    gm["home_team"] = gm.home_team.map(lambda t: FR.get(t, t))
    gm["away_team"] = gm.away_team.map(lambda t: FR.get(t, t))
    pts = pd.concat([gm.rename(columns={"home_team": "posteam", "home_score": "pts_for"})[["game_id", "posteam", "pts_for"]],
                     gm.rename(columns={"away_team": "posteam", "away_score": "pts_for"})[["game_id", "posteam", "pts_for"]]])
    u = u.merge(pts, on=["game_id", "posteam"], how="left")
    # game order key
    pg = pg.merge(u[["game_id", "defteam", "season", "week", "game_date"]].rename(columns={"defteam": "team"}),
                  on=["game_id", "team"], how="left")
    miss = pg.season.isna().sum()
    pg = pg[pg.season.notna()].copy()
    pg["season"] = pg.season.astype(int)
    ol = ol.merge(u[["game_id", "posteam", "season", "week", "game_date"]].rename(columns={"posteam": "team"}),
                  on=["game_id", "team"], how="left")

    # ------------------------------------------------ snaps 2013+ (gsis) ----
    xw = pd.read_csv(os.path.join(DATA, "nfl_players.csv"), usecols=["gsis_id", "pfr_id"]).dropna()
    sn = []
    for yr in range(2012, min(a.through, TEST_ERA - 1 if not quiet else a.through) + 1):
        f = os.path.join(DATA, f"snap_{yr}.csv")
        if os.path.exists(f):
            t = pd.read_csv(f)
            if len(t):
                sn.append(t)
    snaps = pd.concat(sn).merge(xw, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    snaps["team"] = snaps.team.map(lambda t: FR.get(t, t))
    snaps = snaps.rename(columns={"gsis_id": "player_id"})[
        ["game_id", "season", "week", "team", "player_id", "position", "offense_snaps", "offense_pct",
         "defense_snaps", "defense_pct", "st_snaps"]]

    pg.to_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_credits.parquet"), index=False)
    u.to_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_units.parquet"), index=False)
    ol.to_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_olpen.parquet"), index=False)
    snaps.to_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"), index=False)
    if quiet:
        print("built (TEST seasons included; no statistics printed)")
        return
    dev = u[u.season.between(2006, 2015)]
    print(f"player-game rows {len(pg):,} (dropped {miss} without unit row) | unit rows {len(u):,} | "
          f"OL-pen rows {len(ol):,} | snap rows {len(snaps):,}")
    print(dev.groupby("season")[["n_db", "n_run", "sacks", "hits_ns", "tfl_plays", "stop_plays"]].sum())
    c = pg[pg.season.between(2006, 2015)].groupby("season")[["sack", "hit", "tfl", "stop"]].sum()
    print("credited totals\n", c.round(0))


if __name__ == "__main__":
    sys.exit(main())
