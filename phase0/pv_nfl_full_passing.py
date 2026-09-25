"""Player-value program (pv), NFL FULL-SPAN extension: PASSING. DISPLAY ONLY.

Extends pv_nfl_passing_build.py (DEV <= 2015) through the latest pulled season with
FROZEN DEV hyper-parameters (see pv_nfl_full_common):

  * the QB-game table is the build's own aggregate() (imported), on every season;
  * each component (opp-adjusted EPA/dropback, success, sack rate, CPOE) runs through
    the build's joint QB x pass-defence Kalman kernel (imported, numba) with the
    parameters the DEV run fitted on its warm-up windows, read from
    data/pv_nfl_passing_hyper.json -- no re-fit;
  * the passing composite uses the LAST DEV fold's weights (season 2015, trained
    2007-2014), frozen: EPA/dropback above league, opponent-neutral.

Output data/pv_nfl_full_passing_current.parquet: one row per QB, his component ratings
as of his NEXT game (the kernel's own season transition, (1-rq)^gap, applied), the
composite, and per-season dropback COUNTS. No realised EPA or success is stored.

--verify-dev: runs the same code on DEV seasons only and checks it reproduces the DEV
run's data/pv_nfl_passing_values.csv (component ratings; composite on the 2015 fold).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import pv_nfl_full_common as C

OUT = "pv_nfl_full_passing_current.parquet"
COMPS = ("epa", "succ", "sack", "cpoe")
DEV_FOLD = "2015"


def load_plays(B, through: int) -> pd.DataFrame:
    """the build's load_plays without its locked-split filter (upper bound `through`)."""
    tb = ds.dataset(os.path.join(C.DATA, "pv_nfl_events.parquet")).to_table(
        columns=B.COLS, filter=ds.field("season") <= through)
    df = tb.to_pandas()
    assert len(df) and int(df.season.max()) <= through
    return df.sort_values(["game_date", "game_id", "play_id"], kind="mergesort").reset_index(drop=True)


def qb_games(B, df):
    """the build's main(), table part: QB-game rows in walk order + filter arrays."""
    qg, tg = B.aggregate(df)
    gm = df.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                   game_date=("game_date", "first"), stype=("season_type", "first"),
                                   home=("home_team", "first"), away=("away_team", "first"))
    qg = qg.merge(gm, left_on="game_id", right_index=True)
    qg = qg.sort_values(["game_date", "game_id", "posteam", "first_play"], kind="mergesort").reset_index(drop=True)
    qids = {q: i for i, q in enumerate(pd.unique(qg.qb))}
    tids = {t: i for i, t in enumerate(sorted(set(qg.defteam) | set(qg.posteam)))}
    dates = {d_: i for i, d_ in enumerate(sorted(set(qg.game_date)))}
    blk = pd.factorize(qg.game_id + "|" + qg.posteam)[0]
    assert np.all(np.diff(blk) >= 0)
    A = {"date_i": qg.game_date.map(dates).to_numpy(np.int64), "blk": blk.astype(np.int64),
         "season": qg.season.to_numpy(np.int64), "qb": qg.qb.map(qids).to_numpy(np.int64),
         "dfn": qg.defteam.map(tids).to_numpy(np.int64), "nQ": len(qids), "nD": len(tids)}
    return qg, tg, A


def component_inputs(qg):
    """the build's comps dict (ybar, n) for the kept components."""
    ndb = qg.n_db.to_numpy(float)
    ncp = qg.n_cp.to_numpy(float)
    return {
        "epa": (qg.s_epa / ndb, ndb),
        "succ": (qg.s_succ / ndb, ndb),
        "sack": (qg.n_sack / ndb, ndb),
        "cpoe": (np.where(ncp > 0, qg.s_cpoe / np.maximum(ncp, 1), 0.0), ncp),
    }


def run(B, hyper, qg, A):
    """kernel walk per component with the frozen DEV parameters; returns out arrays."""
    OUTC = {}
    for c, (y_, n_) in component_inputs(qg).items():
        y_ = np.nan_to_num(np.asarray(y_, float)); n_ = np.asarray(n_, float)
        h = hyper[c]
        p = [h[k] for k in B.PNAMES]
        OUTC[c] = B.run_filter(A, y_, n_, p, h["sig2"])
    return OUTC


def composite(hyper, Q):
    w = hyper["composite_weights_by_season"][DEV_FOLD]
    assert w["components"] == list(COMPS)
    return sum(w["qb_coef"][c] * Q[c] for c in COMPS)


def verify_dev(B, hyper):
    df = load_plays(B, C.TEST_ERA - 1)
    qg, _, A = qb_games(B, df)
    OUTC = run(B, hyper, qg, A)
    dev = pd.read_csv(os.path.join(C.DATA, "pv_nfl_passing_values.csv"))
    mine = pd.DataFrame({"game_id": qg.game_id, "qb_id": qg.qb, "season": qg.season,
                         **{f"{c}_q": OUTC[c][:, 0] for c in COMPS}})
    mine["comp"] = composite(hyper, {c: OUTC[c][:, 0] for c in COMPS})
    m = mine.merge(dev, on=["game_id", "qb_id"], suffixes=("", "_dev"))
    assert len(m) == len(dev), (len(m), len(dev))
    diffs = {f"{c}_q": float(np.abs(m[f"{c}_q"] - m[f"{c}_q_dev"]).max()) for c in COMPS}
    f = m[m.season == int(DEV_FOLD)]
    diffs["composite_2015"] = float(np.abs(f.comp - f.passing_composite).max())
    for k, v in diffs.items():
        C.log(f"  verify DEV {k}: max |diff| {v:.2e}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dev", action="store_true")
    a = ap.parse_args()
    os.chdir(C.ROOT)                       # the build reads data/... relative paths
    B = C.import_component("pv_nfl_passing_build")
    hyper = C.read_json("pv_nfl_passing_hyper.json")
    if a.verify_dev:
        verify_dev(B, hyper)
        return
    last = C.events_span()["season"]
    df = load_plays(B, last)
    qg, tg, A = qb_games(B, df)
    C.log(f"passing: {len(qg):,} QB-game rows through {last}")
    OUTC = run(B, hyper, qg, A)
    # post-game state of each QB's last game (a zero-weight obs carries no update:
    # its post state is its pre state), then the kernel's season transition to `last`
    ymap = component_inputs(qg)
    last_ix = qg.groupby("qb").cumcount(ascending=False).to_numpy() == 0
    rows = pd.DataFrame({"qb_id": qg.qb, "team": qg.posteam.map(C.fr), "season": qg.season,
                         "game_date": qg.game_date})[last_ix].reset_index(drop=True)
    idx = np.where(last_ix)[0]
    gap = (last - qg.season.to_numpy()[idx]).astype(float)
    Q, SD = {}, {}
    for c in COMPS:
        h = hyper[c]
        o = OUTC[c]
        n_ = np.asarray(ymap[c][1], float)[idx]
        m = np.where(n_ > 0, o[idx, 5], o[idx, 0])
        v = np.where(n_ > 0, o[idx, 6], o[idx, 1])
        Q[c] = m * (1.0 - h["rq"]) ** gap
        SD[c] = np.sqrt(v + h["tsq"] * gap)
        rows[f"{c}_q"] = Q[c]
        rows[f"{c}_sd"] = SD[c]
    rows["passing_composite"] = composite(hyper, Q)
    cnt = qg.groupby(["qb", "season"]).n_db.sum()
    for yr, tag in ((last, "cur"), (last - 1, "prev")):
        rows[f"db_{tag}"] = [int(cnt.get((q, yr), 0)) for q in rows.qb_id]
    rows["db_career"] = rows.qb_id.map(qg.groupby("qb").n_db.sum()).astype(int)
    rows["last_season"] = rows.season.astype(int)
    rows = rows.drop(columns=["season"])
    C.write_parquet(rows, OUT)
    C.record_provenance("pv_nfl_passing_build", C.sha256_file(B.__file__),
                        {"passing": {"through_season": last, "weights_fold": DEV_FOLD}})
    C.log(f"wrote {OUT}: {len(rows):,} passers (state as of season {last})")


if __name__ == "__main__":
    main()
