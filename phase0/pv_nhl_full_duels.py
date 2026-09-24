"""Player-value program, NHL -- FULL (DEV + TEST) run of component DISCRETE_DUELS.

Uses the unmodified phase0/pv_nhl_duels.py / pv_nhl_duels_build.py functions and runs
the per-player-game build through 2025-26 with the DEV hyperparameters FROZEN
(data/pv_nhl_discrete_duels_params.json: faceoff EKF sigma0 / q_season / position
priors / lr, and the EB-EWMA lam / delta / n0 per rate -- never re-tuned).

Goal values (league-level regression weights: goals per faceoff win by context, per
penalty by type, per takeaway/giveaway by zone). The DEV design fits season s on
seasons < s. Under the TEST protocol no weight may be fit on TEST-season data, so:
  * DEV seasons: exactly the DEV walk-forward values (recomputed from DEV data only and
    checked equal to data/pv_nhl_discrete_duels_values.json);
  * every TEST season: the value the DEV walk-forward assigns to 2018-19, i.e. the fit
    on DEV 2010-11..2017-18 only (computed here on DEV-only frames, FROZEN).
The faceoff Bradley-Terry ratings, the player EB-EWMA rates, league means and arena
factors keep walking forward through TEST (serving behaviour).

Missing shift charts (TEST only, pv_nhl_full_common.noshift_games): the roster TOI of
such a game is blank (read as 0 s), so its counts would enter the per-60 rates with no
minutes. Those rows contribute neither counts nor minutes to any TOI-denominated
accumulator and do not advance its per-game decay (_ewma_prior_obs); the league TOI
mean skips them (league_mean_prior_obs). Faceoff WIN/LOSS ratings (1v1, no minutes)
still update. On DEV (no missing game) every operation is the original one.

Outputs (never the DEV files):
  data/pv_nhl_full_discrete_duels.csv             one row per dressed skater per game
  data/pv_nhl_full_discrete_duels_fo.parquet      per-faceoff pre-event prob
  data/pv_nhl_full_discrete_duels_values.json     goal values used per season (+ freeze note)
  data/pv_nhl_full_discrete_duels_build_meta.json

    python phase0/pv_nhl_full_duels.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from numba import njit

import pv_nhl_full_common as FC
import pv_nhl_duels as D
import pv_nhl_duels_build as B

_OBS = {"arr": None}
_ORIG_LMP = D.league_mean_prior


@njit(cache=True)
def _ewma_prior_obs_kernel(pidx, seas, x, toi, lam, delta, obs):
    n = len(pidx)
    S = np.zeros(n)
    T = np.zeros(n)
    cs = 0.0
    ct = 0.0
    for i in range(n):
        if i == 0 or pidx[i] != pidx[i - 1]:
            cs = 0.0
            ct = 0.0
        elif seas[i] != seas[i - 1]:
            cs *= delta
            ct *= delta
        S[i] = cs
        T[i] = ct
        if obs[i]:
            cs = lam * cs + x[i]
            ct = lam * ct + toi[i]
    return S, T


def ewma_prior_obs(pidx, seas, x, toi, lam, delta):
    obs = _OBS["arr"]
    assert obs is not None and len(obs) == len(pidx)
    return _ewma_prior_obs_kernel(pidx, seas, np.asarray(x, np.float64),
                                  np.asarray(toi, np.float64), lam, delta, obs)


def league_mean_prior_obs(pg, xcol, group):
    if "_obs" in pg.columns:
        pg = pg.assign(toi_h=pg.toi_h.to_numpy(np.float64) * pg["_obs"].to_numpy(np.float64))
    return _ORIG_LMP(pg, xcol, group)


def zero_cols(pg):
    cols = (["toi_s", "toi_h", "fo_n", "fo_w", "fo_val", "fo_real",
             "pen_take_all", "pen_take_n", "pen_take_g",
             "pen_draw_all", "pen_draw_n", "pen_draw_g",
             "tk_n", "gv_n", "tk_adj", "gv_adj", "tk_g", "gv_g", "goals", "assists"]
            + [f"fo_n_{c}" for c in D.FO_CATS]
            + [f"pen_take_{c}" for c in D.PEN_CATS] + [f"pen_draw_{c}" for c in D.PEN_CATS]
            + [f"{c}_adj" for c in D.TK_CATS])
    return [c for c in cols if c in pg.columns]


def jsonable(v):
    return {str(k): x for k, x in v.items() if not str(k).startswith("_")}


def main():
    t0 = time.time()
    # ---- goal values: DEV walk-forward + the 2018-19 value (fit on DEV only)
    ev_d, ro_d, games_d = D.load_base(False)
    dev_seasons = sorted(ev_d.season.unique().tolist())
    assert dev_seasons == FC.DEV_SEASONS
    vals_dev, _, _, _ = D.walk_forward_values(ev_d, games_d, dev_seasons + [FC.FREEZE_SEASON])
    assert vals_dev[FC.FREEZE_SEASON]["fit_on"] == "20102011..20172018"
    ref = json.load(open(D.OUT + "_values.json"))
    same = all(json.dumps(ref[str(s)], sort_keys=True) == json.dumps(vals_dev[s], sort_keys=True)
               for s in dev_seasons)
    print("DEV goal values identical to data/pv_nhl_discrete_duels_values.json:", same,
          flush=True)
    assert same
    del ev_d, ro_d, games_d
    print(f"values {time.time()-t0:.0f}s", flush=True)

    # ---- full data
    ev, ro, games = D.load_base(True)
    seasons = sorted(ev.season.unique().tolist())
    assert seasons == FC.ALL_SEASONS, seasons
    hi, ai, _ = D.team_season_index(ev, games)
    f = D.fo_frame(ev, hi, ai)
    p = D.pen_frame(ev, hi, ai)
    t = D.tk_frame(ev, hi, ai)
    vals = {s: vals_dev[s] for s in FC.DEV_SEASONS}
    for s in FC.TEST_SEASONS:
        v = json.loads(json.dumps(vals_dev[FC.FREEZE_SEASON]))
        v["fit_on"] = "20102011..20172018 (FROZEN: DEV walk-forward value for 20182019)"
        vals[s] = v
    json.dump(jsonable(vals), open(FC.full_path("pv_nhl_discrete_duels_values.json"), "w"),
              indent=1)
    vals["_pen_frame"] = p
    vals["_tk_frame"] = t
    pids = sorted(set(ro.pid.astype(np.int64)))
    pid_index = {p_: i for i, p_ in enumerate(pids)}
    print(f"frames {time.time()-t0:.0f}s", flush=True)

    # ---- build (pv_nhl_duels_build.run 'build' branch, frozen DEV params)
    prm = json.load(open(B.PARAMS_JSON))
    fo_prm = {k: prm["fo"][k] for k in ("sigma0", "q_season", "mu_c", "mu_w", "mu_d", "lr")}
    arr, fs, gpos, r, ptr, rop = B.fo_setup(f, games, ro, pid_index)
    pw, th, sm, sv, sn, sbar = D.run_fo(arr, ptr, rop, len(pid_index), fo_prm)
    fo_out = pd.DataFrame({"gid": fs.gid.values, "season": fs.season.values,
                           "cat": fs.cat.values, "win_pid": fs.fo_win.astype(np.int64).values,
                           "lose_pid": fs.fo_lose.astype(np.int64).values,
                           "is_home_win": fs.is_home.values, "zone_win": fs.zone.values,
                           "p_winner": pw})
    fo_out.to_parquet(FC.full_path("pv_nhl_discrete_duels_fo.parquet"), index=False)
    snap = pd.DataFrame({"gid": r.gid.values, "pid": r.pid.astype(np.int64).values,
                         "fo_m": sm, "fo_v": sv, "fo_nprev": sn, "fo_mbar": sbar})
    pg = D.player_games(ev, ro, games, vals, fs)
    pg = pg.merge(snap, on=["gid", "pid"], how="left")
    pg = B.pg_prepare(pg, games)
    ns = FC.noshift_games()
    obs = ~pg.gid.isin(ns).to_numpy()
    pgz = pg.copy()
    pgz["_obs"] = obs.astype(np.float64)
    for c in zero_cols(pgz):
        v = pgz[c].to_numpy().copy()
        v[~obs] = 0
        pgz[c] = v
    _OBS["arr"] = obs
    D._ewma_prior = ewma_prior_obs
    D.league_mean_prior = league_mean_prior_obs
    V = B.compute_values(pgz, prm["rates"], fo_prm, snap, vals)
    out = pd.concat([pg[["gid", "date", "season", "team", "side", "pid", "pos", "toi_s",
                         "fo_n", "fo_w", "fo_val", "fo_real", "pen_take_all", "pen_take_n",
                         "pen_take_g", "pen_draw_all", "pen_draw_n", "pen_draw_g",
                         "tk_n", "gv_n", "tk_adj", "gv_adj", "tk_g", "gv_g", "goals",
                         "assists", "fo_nprev"]], V], axis=1)
    out = out.sort_values(["date", "gid", "side", "pid"]).reset_index(drop=True)
    out.to_csv(FC.full_path("pv_nhl_discrete_duels.csv"), index=False, float_format="%.6g")
    json.dump({"fo_theta_end": th.tolist(), "rows": int(len(out)),
               "rows_without_shift_chart": int((~obs).sum()),
               "frozen_params": prm["fo"] | {"rates": prm["rates"]},
               "goal_values": "DEV walk-forward for DEV seasons; 20182019 DEV fit frozen "
                              "for every TEST season"},
              open(FC.full_path("pv_nhl_discrete_duels_build_meta.json"), "w"), indent=1)
    print(f"build done rows={len(out)} no-shift rows={(~obs).sum()} {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
