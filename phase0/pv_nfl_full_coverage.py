"""Player-value program (pv), NFL FULL-SPAN extension: COVERAGE. DISPLAY ONLY.

Extends pv_nfl_coverage.py (DEV <= 2015) through the latest pulled season with FROZEN DEV
hyper-parameters (see pv_nfl_full_common):

  * tables: the component's own load(all_seasons=True) / build_tables() (imported);
  * engine: its Engine / run() (imported) with its module-constant dynamics and the
    empirical-Bayes prior strengths, efficiency priors, R6 z-tables and EPA linear
    weights the DEV run set on the warm-up seasons, READ from
    data/pv_nfl_coverage_priors.json -- the warm-up method-of-moments pass is not re-run;
  * positions: nfl_players.csv as the DEV run read it, plus the weekly roster for ids it
    does not know (this year's rookies) -- identity metadata (the position group).

What public play-by-play can attribute to a pass defender is limited (the component's
docstring): interceptions, passes defensed, the tackler after a catch and coverage
penalties. It cannot see who was covering on most completions or on untargeted snaps.

Output data/pv_nfl_full_coverage_current.parquet: one row per defender, the engine's
reading of him after the last real date (the same Engine.player() that writes a PRE-game
row), plus exposure COUNTS (games credited, opponent attempts in them). th_ball = passes
defensed + interceptions over expectation for his position group vs the opposing QB;
epap = EPA saved per attributed target over the QB's expectation (shrunk); at_rate =
attributed targets per opponent attempt. No realised count is stored.

--verify-dev: runs the same code on DEV seasons only and checks it reproduces the DEV
run's data/pv_nfl_coverage_player_games.parquet pre-game values.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import pv_nfl_full_common as C

OUT = "pv_nfl_full_coverage_current.parquet"
KEEP = ("th_pd", "th_int", "th_ball", "th_tc", "th_cpen", "cv", "epap", "dsp", "at_rate", "cv_eff", "at_n", "games")


def frozen(M):
    pri = C.read_json("pv_nfl_coverage_priors.json")
    prior_a = {(g, k): pri["gamma_prior_strength"][f"{g}|{k}"]["a"] for g in M.GROUPS for k in M.KINDS}
    eff = {(g, m): pri["efficiency_prior_strength"][f"{g}|{m}"] for g in M.GROUPS for m in ("ds", "epa", "pm")}
    r6z = {g: {k: tuple(v) for k, v in d.items()} for g, d in pri["r6_ztables_warmup"].items()}
    return prior_a, eff, r6z, pri["linear_weights_defence_positive"]


def posmap(M, E, season):
    pos = C.positions(season)
    m = {p: M.pos_group(pos.get(p)) for p in E.pid.unique()}
    return {p: g for p, g in m.items() if g is not None}


def verify_dev(M):
    df = M.load(all_seasons=False)
    Gm, T, E = M.build_tables(df)
    prior_a, eff, r6z, lw = frozen(M)
    _, out = M.run(Gm, T, E, prior_a, lw, M.posmap_from(E), r6z, seasons_max=C.TEST_ERA - 1, eff_prior=eff)
    P = pd.DataFrame(out["player"])
    dev = pd.read_parquet(os.path.join(C.DATA, "pv_nfl_coverage_player_games.parquet"))
    m = P.merge(dev, on=["game_id", "team", "player_id"], suffixes=("", "_dev"))
    assert len(m) == len(dev) == len(P), (len(m), len(dev), len(P))
    diffs = {c: float(np.nanmax(np.abs(m[c] - m[c + "_dev"])))
             for c in ("pre_th_ball", "pre_epap", "pre_cv", "pre_cv_eff", "pre_at_rate")}
    for c, v in diffs.items():
        C.log(f"  verify DEV {c}: max |diff| {v:.2e}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dev", action="store_true")
    a = ap.parse_args()
    os.chdir(C.ROOT)                       # the component reads data/... relative paths
    M = C.import_component("pv_nfl_coverage")
    if a.verify_dev:
        verify_dev(M)
        return
    last = C.events_span()["season"]
    df = M.load(all_seasons=True)
    Gm, T, E = M.build_tables(df)
    assert int(Gm.season.max()) == last
    prior_a, eff, r6z, lw = frozen(M)
    pm = posmap(M, E, last)
    C.log(f"coverage: engine walk 1999-{last} with the DEV priors ({len(pm):,} mapped defenders)")
    eng, _ = M.run(Gm, T, E, prior_a, lw, pm, r6z, seasons_max=last, collect=False, eff_prior=eff)
    # exposure counts (sample size only): games credited and opponent attempts in them
    x = E[["game_id", "defteam", "pid"]].drop_duplicates().merge(
        T[["game_id", "defteam", "att", "season"]], on=["game_id", "defteam"], how="left")
    team = x.merge(Gm[["game_id", "game_date"]], on="game_id").sort_values(["game_date", "game_id"]) \
        .groupby("pid").defteam.last()
    cnt = {}
    for yr, tag in ((last, "cur"), (last - 1, "prev")):
        s = x[x.season == yr].groupby("pid").agg(g=("game_id", "nunique"), att=("att", "sum"))
        cnt[tag] = s.rename(columns={"g": f"g_{tag}", "att": f"att_{tag}"})
    recent = x[x.season >= last - 2].pid.unique()
    rows = []
    for pid in recent:
        r = eng.player(pid) if pid in pm else None
        if r is None:
            continue
        rec = {"player_id": pid, "group": r["group"], "team": C.fr(team.get(pid, ""))}
        rec.update({k: float(r[k]) for k in KEEP})
        g = r["group"]
        rec["mu_pd"] = eng.mu_rate(g, "pd"); rec["mu_int"] = eng.mu_rate(g, "int")
        rows.append(rec)
    out = pd.DataFrame(rows)
    for tag in ("cur", "prev"):
        out = out.merge(cnt[tag], left_on="player_id", right_index=True, how="left")
    out = out.fillna({c: 0 for c in ("g_cur", "att_cur", "g_prev", "att_prev")})
    C.write_parquet(out, OUT)
    C.record_provenance("pv_nfl_coverage", C.sha256_file(M.__file__), {"coverage": {"through_season": last}})
    C.log(f"wrote {OUT}: {len(out):,} defenders (state as of season {last})")


if __name__ == "__main__":
    main()
