"""Player-value program (pv), NFL receiving_rushing, STEP 1: walk-forward
situation-expectation models for every target and every designed carry.

The MLB principle applied to skill players: rate a receiver/rusher only on
what HE controls, relative to what an average player would have produced in
the same situation. This step builds the "same situation" baselines:

  targets (non-sack, non-2pt pass plays with target_player_id; 2006+ because
  air_yards / pass_location / yards_after_catch start in 2006):
      xcomp  P(complete | air_yards, pass_location, down, ydstogo, yardline,
             shotgun, no_huddle, qtr, clock, score)
      xyac   E[yards_after_catch | complete, same]        (YAC clipped -5..40)
      xepa_t E[epa | same]                                 (epa per target)
  designed carries (play_type run, not a QB scramble, not a 2pt try; 1999+):
      xryd   E[clip(yards, -5, 20) | down, ydstogo, yardline, goal_to_go,
             run_location, run_gap, shotgun, no_huddle, qtr, clock, score]
      xsucc  P(epa > 0 | same)
      xepa_r E[epa | same]

WALK-FORWARD: the models are refit at every (season, week) block on plays
strictly before that block (window: current season to date + the 3 previous
seasons). A block with too little prior data gets NaN expectations (no
residual, so no rating update) -- this only affects the first weeks of 2006
(targets) and 1999 (carries), which are warm-up.

LOCKED SPLIT: by default only seasons <= 2015 are READ (pyarrow filter), so no
TEST statistic can be computed. `--through YYYY` extends the walk for the lead
engineer's single TEST look / serving; nothing TEST-side is printed.

nflfastR model outputs (cp, cpoe, xyac_epa, xpass, ...) are NOT used as
baselines: they are pooled fits that include post-2015 seasons. `epa` itself
is used as an outcome, with the same caveat the shipped features accept.

Output: data/pv_nfl_receiving_rushing_plays.parquet (one row per target or
carry with the context, outcomes and walk-forward expectations).
"""
from __future__ import annotations

import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EVENTS = os.path.join(DATA, "pv_nfl_events.parquet")
DEV_LAST = 2015

ap = argparse.ArgumentParser()
ap.add_argument("--through", type=int, default=DEV_LAST,
                help="last season to walk (default 2015 = DEV only; >2015 is for the lead engineer)")
ap.add_argument("--out", default=os.path.join(DATA, "pv_nfl_receiving_rushing_plays.parquet"))
args = ap.parse_args()
THROUGH = args.through
QUIET_TEST = THROUGH > DEV_LAST  # never print anything once TEST seasons are in memory

COLS = ["game_id", "play_id", "season", "season_type", "week", "game_date", "posteam", "defteam",
        "play_type", "qtr", "down", "ydstogo", "yardline_100", "goal_to_go", "half_seconds_remaining",
        "game_seconds_remaining", "score_differential", "shotgun", "no_huddle", "yards_gained", "epa",
        "success", "first_down", "touchdown", "pass_touchdown", "rush_touchdown", "qb_scramble", "sack",
        "complete_pass", "interception", "fumble", "fumble_lost", "fumbled_1_player_id", "two_point_attempt",
        "air_yards", "yards_after_catch", "pass_location", "run_location", "run_gap", "passer_id",
        "target_player_id", "rusher_player_id", "rusher_id", "td_player_id"]

t0 = time.time()
ev = pd.read_parquet(EVENTS, columns=COLS, filters=[("season", "<=", THROUGH)])
assert int(ev.season.max()) <= THROUGH
for c in ["qtr", "down", "ydstogo", "yardline_100", "goal_to_go", "half_seconds_remaining",
          "game_seconds_remaining", "score_differential", "shotgun", "no_huddle", "yards_gained",
          "success", "first_down", "touchdown", "pass_touchdown", "rush_touchdown", "qb_scramble", "sack",
          "complete_pass", "interception", "fumble", "fumble_lost", "two_point_attempt", "air_yards",
          "yards_after_catch", "week", "season"]:
    ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("float64")
ev["epa"] = ev["epa"].astype("float64")

# game order: date then game_id (a game's plays are all in one block)
gorder = ev[["game_id", "game_date", "season", "week"]].drop_duplicates("game_id")
gorder = gorder.sort_values(["game_date", "game_id"]).reset_index(drop=True)
gorder["gidx"] = np.arange(len(gorder))
ev = ev.merge(gorder[["game_id", "gidx"]], on="game_id", how="left")

# ------------------------------------------------------------------ targets
tg = ev[(ev.play_type == "pass") & (ev.sack != 1) & (ev.two_point_attempt != 1)
        & ev.target_player_id.notna() & (ev.season >= 2006)].copy()
tg["kind"] = "T"
tg["player_id"] = tg["target_player_id"].astype(str)
tg["yac_c"] = tg["yards_after_catch"].clip(-5, 40)
# ------------------------------------------------------------------ carries
ru = ev[(ev.play_type == "run") & (ev.qb_scramble != 1) & (ev.two_point_attempt != 1)
        & ev.rusher_player_id.notna()].copy()
ru["kind"] = "R"
ru["player_id"] = ru["rusher_player_id"].astype(str)
ru["yds_c"] = ru["yards_gained"].clip(-5, 20)
ru["succ"] = (ru["epa"] > 0).astype(float)
ru.loc[ru.epa.isna(), "succ"] = np.nan

LOC = {"left": 0, "middle": 1, "right": 2}
GAP = {"end": 0, "tackle": 1, "guard": 2}
for d in (tg, ru):
    d["ploc"] = d["pass_location"].map(LOC).astype("float64")
    d["rloc"] = d["run_location"].map(LOC).astype("float64")
    d["rgap"] = d["run_gap"].map(GAP).astype("float64")
    d["blk"] = (d["season"].astype(int) * 100 + d["week"].astype(int))

FT_T = ["air_yards", "ploc", "down", "ydstogo", "yardline_100", "goal_to_go", "shotgun", "no_huddle",
        "qtr", "half_seconds_remaining", "score_differential"]
FT_R = ["rloc", "rgap", "down", "ydstogo", "yardline_100", "goal_to_go", "shotgun", "no_huddle",
        "qtr", "half_seconds_remaining", "score_differential"]
P_BIN = dict(objective="binary", learning_rate=0.08, num_leaves=15, min_data_in_leaf=300,
             feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             verbose=-1, num_threads=16, seed=7)
P_REG = dict(P_BIN, objective="regression")
NROUND = 160
MIN_TRAIN = 3000


def walk(d, feats, specs, label):
    """specs: list of (out_col, target_col, row_mask_fn, params)."""
    out = {o: np.full(len(d), np.nan) for o, *_ in specs}
    blks = np.sort(d["blk"].unique())
    seas = d["season"].to_numpy()
    bl = d["blk"].to_numpy()
    X = d[feats].to_numpy(dtype=np.float64)
    nfit = 0
    prep = [(o, d[ycol].to_numpy(dtype=np.float64), mfn(d), prm) for o, ycol, mfn, prm in specs]
    for b in blks:
        s = b // 100
        te = bl == b
        tr = (bl < b) & (seas >= s - 3)
        for o, y, fm, prm in prep:
            m = tr & ~np.isnan(y) & fm
            if m.sum() < MIN_TRAIN:
                continue
            bst = lgb.train(prm, lgb.Dataset(X[m], y[m], free_raw_data=True), NROUND)
            out[o][te] = bst.predict(X[te])
            nfit += 1
    for o in out:
        d[o] = out[o]
    if not QUIET_TEST:
        print(f"  {label}: {len(blks)} blocks, {nfit} fits, {time.time()-t0:.0f}s", flush=True)
    return d


ALL = lambda d: np.ones(len(d), bool)  # noqa: E731
COMP = lambda d: (d["complete_pass"] == 1).to_numpy()  # noqa: E731
tg = walk(tg, FT_T, [("xcomp", "complete_pass", ALL, P_BIN),
                     ("xyac", "yac_c", COMP, P_REG),
                     ("xepa", "epa", ALL, P_REG)], "targets")
ru = walk(ru, FT_R, [("xyds", "yds_c", ALL, P_REG),
                     ("xsucc", "succ", ALL, P_BIN),
                     ("xepa", "epa", ALL, P_REG)], "carries")

keep = ["kind", "game_id", "gidx", "play_id", "season", "season_type", "week", "game_date", "posteam",
        "defteam", "player_id", "passer_id", "down", "ydstogo", "yardline_100", "yards_gained", "epa",
        "first_down", "touchdown", "fumble_lost", "fumbled_1_player_id", "complete_pass", "interception",
        "air_yards", "yards_after_catch", "yac_c", "yds_c", "succ", "xcomp", "xyac", "xepa", "xyds", "xsucc"]
for d in (tg, ru):
    for c in keep:
        if c not in d:
            d[c] = np.nan
out = pd.concat([tg[keep], ru[keep]], ignore_index=True)
out = out.sort_values(["gidx", "play_id", "kind"]).reset_index(drop=True)
out.to_parquet(args.out, index=False)
if not QUIET_TEST:
    dv = out[out.season <= DEV_LAST]
    t = dv[dv.kind == "T"]; r = dv[dv.kind == "R"]
    print(f"targets {len(t):,} (xcomp coverage {t.xcomp.notna().mean():.3f}) | carries {len(r):,} "
          f"(xyds coverage {r.xyds.notna().mean():.3f})")
    # calibration sanity on DEV, walk-forward predictions
    tt = t[t.xcomp.notna()]
    print(f"  xcomp mean {tt.xcomp.mean():.4f} vs actual {tt.complete_pass.mean():.4f}; "
          f"brier {((tt.complete_pass-tt.xcomp)**2).mean():.4f} vs const {tt.complete_pass.var():.4f}")
    cc = tt[tt.complete_pass == 1]
    print(f"  xyac mean {cc.xyac.mean():.3f} vs {cc.yac_c.mean():.3f}; R2 "
          f"{1-((cc.yac_c-cc.xyac)**2).mean()/cc.yac_c.var():.3f}")
    print(f"  xepa(t) R2 {1-((tt.epa-tt.xepa)**2).mean()/tt.epa.var():.3f}")
    rr = r[r.xyds.notna()]
    print(f"  xyds mean {rr.xyds.mean():.3f} vs {rr.yds_c.mean():.3f}; R2 {1-((rr.yds_c-rr.xyds)**2).mean()/rr.yds_c.var():.3f}; "
          f"xsucc brier {((rr.succ-rr.xsucc)**2).mean():.4f} vs {rr.succ.var():.4f}; xepa(r) R2 "
          f"{1-((rr.epa-rr.xepa)**2).mean()/rr.epa.var():.3f}")
    print(f"wrote {args.out}  {time.time()-t0:.0f}s")
