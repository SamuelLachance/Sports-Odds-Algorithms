"""pv_nhl_duels_build -- driver for phase0/pv_nhl_duels.py (DISCRETE_DUELS).

  values : walk-forward goal values per season -> data/pv_nhl_discrete_duels_values.json
  tune   : DEV-A (2011-12..2014-15) grids for the faceoff EKF and the EB-EWMA
           rates; DEV-B (2015-16..2017-18) is never used to choose anything.
           -> data/pv_nhl_discrete_duels_params.json
  build  : per-player-game walk-forward values with the tuned params
           -> data/pv_nhl_discrete_duels.csv, data/pv_nhl_discrete_duels_fo.parquet

Run via `python phase0/pv_nhl_duels.py <cmd>`.
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import pv_nhl_duels as D

VALS_JSON = D.OUT + "_values.json"
PARAMS_JSON = D.OUT + "_params.json"


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _setup(allow_test, cutoff=None):
    ev, ro, games, vals, f, p, t, pid_index = D.prepare(allow_test, cutoff)
    vals_out = {str(k): v for k, v in vals.items()}
    tag = "" if not allow_test else "_ALLTEST"
    if cutoff is None:
        with open(D.OUT + tag + "_values.json", "w") as fh:
            json.dump(vals_out, fh, indent=1)
    vals["_pen_frame"] = p
    vals["_tk_frame"] = t
    return ev, ro, games, vals, f, p, t, pid_index


def fo_setup(f, games, ro, pid_index):
    arr, fs, gpos = D.fo_inputs(f, games, ro, pid_index)
    r, ptr, rop = D.roster_arrays(ro[ro.pos != "G"], gpos, pid_index)
    return arr, fs, gpos, r, ptr, rop


def tune_fo(arr, fs, ptr, rop, n_players):
    tune_mask = fs.season.isin(D.TUNE_SEASONS).values
    hold_mask = fs.season.isin(D.HOLD_SEASONS).values
    res = []
    base = dict(sigma0=1e-4, q_season=0.0, mu_c=0.0, mu_w=0.0, mu_d=0.0, lr=1e-4)
    pw, th, *_ = D.run_fo(arr, ptr, rop, n_players, base)
    ctx_only = {"tune_ll": D.fo_ll(pw, tune_mask), "hold_ll": D.fo_ll(pw, hold_mask),
                "theta": th.tolist()}
    print("context-only", ctx_only, flush=True)
    grid = itertools.product([0.10, 0.15, 0.20, 0.25, 0.32], [0.0, 0.1, 0.3],
                             [0.0, -0.05], [-0.1, -0.25, -0.4], [-0.3])
    for s0, qf, mc, mw, md in grid:
        prm = dict(sigma0=s0, q_season=qf * s0 * s0, mu_c=mc, mu_w=mw, mu_d=md, lr=1e-4)
        pw, th, *_ = D.run_fo(arr, ptr, rop, n_players, prm)
        res.append({**prm, "q_frac": qf, "tune_ll": D.fo_ll(pw, tune_mask)})
    res.sort(key=lambda r: r["tune_ll"])
    best = dict(res[0])
    print("fo best (DEV-A)", best, flush=True)
    return best, ctx_only, res[:10]


def pg_prepare(pg, games):
    gorder = games.sort_values(["date", "gid"]).gid.values
    gpos = {g: i for i, g in enumerate(gorder)}
    pg = pg.copy()
    pg["g"] = pg.gid.map(gpos)
    pids = sorted(set(pg.pid))
    pmap = {p: i for i, p in enumerate(pids)}
    pg["pidx"] = pg.pid.map(pmap)
    pg["sidx"] = pg.season.map(D.season_idx)
    pg["toi_h"] = pg.toi_s / 3600.0
    pg["pgrp"] = np.where(pg.pos == "D", "D", "F")
    pg["fgrp"] = np.where(pg.pos == "C", "C", np.where(pg.pos == "D", "D", "W"))
    pg["all"] = "all"
    pg = pg.sort_values(["pidx", "g"], kind="mergesort").reset_index(drop=True)
    return pg


RATE_STATS = {  # stat -> prior group
    "fo_n": "fgrp", "pen_take_n": "pgrp", "pen_draw_n": "pgrp",
    "tk_adj": "pgrp", "gv_adj": "pgrp",
}


def tune_rates(pg):
    m = pg.season.isin(D.TUNE_SEASONS).values & (pg.toi_h.values > 0)
    out = {}
    for st, grp in RATE_STATS.items():
        best = None
        rows = []
        mu0 = D.league_mean_prior(pg, st, grp)
        for lam, dl, n0 in itertools.product([0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 1.0],
                                              [1.0, 0.7, 0.4, 0.2],
                                              [0.5, 1, 3, 6, 12, 25, 50]):
            r, mu, S, T = D.eb_rate(pg, st, lam, dl, n0, grp, mu=mu0)
            pred = r * pg.toi_h.values
            dev = D.poisson_dev(pg[st].values[m], pred[m])
            rows.append((dev, lam, dl, n0))
            if best is None or dev < best[0]:
                best = (dev, lam, dl, n0)
        # reference: the position-mean-only prediction (n0 -> infinity)
        r, mu, S, T = D.eb_rate(pg, st, 1.0, 1.0, 1e9, grp, mu=mu0)
        dev_mu = D.poisson_dev(pg[st].values[m], (mu * pg.toi_h.values)[m])
        # reference: raw career rate, no shrink (n0 tiny)
        r, mu, S, T = D.eb_rate(pg, st, 1.0, 1.0, 1e-3, grp, mu=mu0)
        dev_raw = D.poisson_dev(pg[st].values[m], (r * pg.toi_h.values)[m])
        out[st] = {"dev": best[0], "lam": best[1], "delta": best[2], "n0": best[3],
                   "dev_position_mean_only": dev_mu, "dev_raw_career_rate": dev_raw}
        print(st, out[st], flush=True)
    return out


def _beta(pg, vals, kind, cat):
    """Goal value of `cat` for each row's season: fit on seasons < s (time of USE)."""
    m = {int(k): v[kind][cat] for k, v in vals.items() if not str(k).startswith("_")}
    return pg.season.map(m).values.astype(float)


def _eb_valued(pg, vals, kind, cols_cats, prm, group):
    """sum_c beta_c(s) * EB-rate(count_c): the player's goal-valued rate per hour,
    plus the matching all-skater league mean (walk-forward)."""
    tot = np.zeros(len(pg))
    mu_all = np.zeros(len(pg))
    for col, cat in cols_cats:
        r, _, _, _ = D.eb_rate(pg, col, prm["lam"], prm["delta"], prm["n0"], group)
        b = _beta(pg, vals, kind, cat)
        tot += b * r
        mu_all += b * D.league_mean_prior(pg, col, "all")
    return tot, mu_all


def compute_values(pg, rp, fo_prm, snap, vals):
    """All walk-forward per-row values (pg sorted by pidx, g). Goal weights are
    applied at the time of use: row in season s uses weights fit on seasons < s
    (2010-11 rows: declared priors) times EB-shrunk per-category rates built from
    the player's strictly earlier games."""
    V = pd.DataFrame(index=pg.index)
    # ---------------- faceoffs
    prior_m = np.where(pg.fgrp == "C", fo_prm["mu_c"],
                       np.where(pg.fgrp == "D", fo_prm["mu_d"], fo_prm["mu_w"]))
    m = np.where(np.isnan(pg.fo_m.values), prior_m, pg.fo_m.values)
    V["fo_rating"] = m - pg.fo_mbar.values
    V["fo_sd"] = np.sqrt(np.where(np.isnan(pg.fo_v.values), fo_prm["sigma0"] ** 2,
                                  pg.fo_v.values))
    V["fo_pavg"] = _sig(V.fo_rating.values)
    c = rp["fo_n"]
    V["fo_n60"], _, _, _ = D.eb_rate(pg, "fo_n", c["lam"], c["delta"], c["n0"], "fgrp")
    V["fo_val60"], _ = _eb_valued(pg, vals, "fo", [(f"fo_n_{k}", k) for k in D.FO_CATS],
                                  c, "fgrp")
    V["fo_g60"] = (V.fo_pavg - 0.5) * V.fo_val60
    # ---------------- penalties (goal-valued, relative to the all-skater mean)
    ct, cd = rp["pen_take_n"], rp["pen_draw_n"]
    V["pen_take60"], mu_t = _eb_valued(pg, vals, "pen",
                                       [(f"pen_take_{k}", k) for k in D.PEN_CATS], ct, "pgrp")
    V["pen_draw60"], mu_d = _eb_valued(pg, vals, "pen",
                                       [(f"pen_draw_{k}", k) for k in D.PEN_CATS], cd, "pgrp")
    V["pen_take_n60"], _, _, _ = D.eb_rate(pg, "pen_take_n", ct["lam"], ct["delta"], ct["n0"], "pgrp")
    V["pen_draw_n60"], _, _, _ = D.eb_rate(pg, "pen_draw_n", cd["lam"], cd["delta"], cd["n0"], "pgrp")
    V["pen_g60"] = (V.pen_draw60 - mu_d) - (V.pen_take60 - mu_t)
    # ---------------- takeaways / giveaways (arena-adjusted, goal-valued by zone)
    ck, cg = rp["tk_adj"], rp["gv_adj"]
    V["tk_g60"], mu_tk = _eb_valued(pg, vals, "tkgv",
                                    [(f"{k}_adj", k) for k in D.TK_CATS if k.startswith("tk")],
                                    ck, "pgrp")
    V["gv_g60"], mu_gv = _eb_valued(pg, vals, "tkgv",
                                    [(f"{k}_adj", k) for k in D.TK_CATS if k.startswith("gv")],
                                    cg, "pgrp")
    V["tk_adj60"], _, _, _ = D.eb_rate(pg, "tk_adj", ck["lam"], ck["delta"], ck["n0"], "pgrp")
    V["gv_adj60"], _, _, _ = D.eb_rate(pg, "gv_adj", cg["lam"], cg["delta"], cg["n0"], "pgrp")
    V["tkgv_g60"] = (V.tk_g60 - mu_tk) + (V.gv_g60 - mu_gv)
    # ---------------- sums
    V["dd_g60"] = V.fo_g60 + V.pen_g60
    V["dd_all_g60"] = V.dd_g60 + V.tkgv_g60
    # ---------------- expected TOI per game (EWMA over prior games, position prior)
    S, T = D._ewma_prior(pg.pidx.values, pg.sidx.values, pg.toi_s.values.astype(float),
                         np.ones(len(pg)), 0.9, 1.0)
    pg1 = pg.assign(one=1.0, toi_h=1.0)       # league mean TOI per game by position
    mu_toi = D.league_mean_prior(pg1.assign(x=pg.toi_s.values), "x", "pgrp")
    V["exp_toi_s"] = (S + 3.0 * mu_toi) / (T + 3.0)
    V["n_prev_games"] = pg.groupby("pidx").cumcount().values
    return V


def run(cmd, allow_test=False, cutoff=None):
    t0 = time.time()
    ev, ro, games, vals, f, p, t, pid_index = _setup(allow_test, cutoff)
    if cmd == "values":
        return
    n_players = len(pid_index)
    arr, fs, gpos, r, ptr, rop = fo_setup(f, games, ro, pid_index)
    if cmd == "tune":
        assert not allow_test, "tuning is DEV-only"
        fo_best, ctx_only, fo_top = tune_fo(arr, fs, ptr, rop, n_players)
        pg = D.player_games(ev, ro, games, vals, fs)
        pg = pg_prepare(pg, games)
        rp = tune_rates(pg)
        with open(PARAMS_JSON, "w") as fh:
            json.dump({"fo": fo_best, "fo_context_only": ctx_only, "fo_top10": fo_top,
                       "rates": rp, "tuned_on": D.TUNE_SEASONS,
                       "note": "DEV-A only; DEV-B (2015-16..2017-18) untouched by tuning"},
                      fh, indent=1)
        print(f"tune done {time.time()-t0:.0f}s")
        return
    # ------------------------------------------------------------- build
    prm = json.load(open(PARAMS_JSON))
    fo_prm = {k: prm["fo"][k] for k in ("sigma0", "q_season", "mu_c", "mu_w", "mu_d", "lr")}
    pw, th, sm, sv, sn, sbar = D.run_fo(arr, ptr, rop, n_players, fo_prm)
    fo_out = pd.DataFrame({"gid": fs.gid.values, "season": fs.season.values,
                           "cat": fs.cat.values, "win_pid": fs.fo_win.astype(np.int64).values,
                           "lose_pid": fs.fo_lose.astype(np.int64).values,
                           "is_home_win": fs.is_home.values, "zone_win": fs.zone.values,
                           "p_winner": pw})
    tag = "" if not allow_test else "_ALLTEST"
    if cmd != "audit":
        fo_out.to_parquet(D.OUT + tag + "_fo.parquet", index=False)
    snap = pd.DataFrame({"gid": r.gid.values, "pid": r.pid.astype(np.int64).values,
                         "fo_m": sm, "fo_v": sv, "fo_nprev": sn, "fo_mbar": sbar})
    pg = D.player_games(ev, ro, games, vals, fs)
    pg = pg.merge(snap, on=["gid", "pid"], how="left")
    pg = pg_prepare(pg, games)
    V = compute_values(pg, prm["rates"], fo_prm, snap, vals)
    out = pd.concat([pg[["gid", "date", "season", "team", "side", "pid", "pos", "toi_s",
                         "fo_n", "fo_w", "fo_val", "fo_real", "pen_take_all", "pen_take_n",
                         "pen_take_g", "pen_draw_all", "pen_draw_n", "pen_draw_g",
                         "tk_n", "gv_n", "tk_adj", "gv_adj", "tk_g", "gv_g", "goals",
                         "assists", "fo_nprev"]], V],
                    axis=1)
    out = out.sort_values(["date", "gid", "side", "pid"]).reset_index(drop=True)
    if not allow_test:
        assert out.gid.max() < D.DEV_MAX_GID
    if cmd == "audit":
        full = pd.read_csv(D.OUT + ".csv")
        full = full[full.date < cutoff].sort_values(["date", "gid", "side", "pid"]
                                                    ).reset_index(drop=True)
        assert len(full) == len(out), (len(full), len(out))
        num = [c for c in V.columns]
        assert (full.gid.values == out.gid.values).all() and (full.pid.values == out.pid.values).all()
        rep = {"cutoff": cutoff, "rows_compared": int(len(out)),
               "note": "saved CSV holds 6 significant digits, so a relative diff ~5e-6 is identity",
               "max_rel_diff": {c: float(np.nanmax(np.abs(full[c].values - out[c].values)
                                                   / np.maximum(1.0, np.abs(out[c].values))))
                                for c in num}}
        fo_full = pd.read_parquet(D.OUT + "_fo.parquet")
        fo_full = fo_full[fo_full.gid.isin(set(fs.gid))]
        rep["fo_pw_max_abs_diff"] = float(np.max(np.abs(np.sort(fo_full.p_winner.values)
                                                        - np.sort(pw))))
        rep["pass"] = bool(all(v < 1e-5 for v in rep["max_rel_diff"].values())
                           and rep["fo_pw_max_abs_diff"] < 1e-9)
        ap = D.OUT + "_audit.json"
        allrep = json.load(open(ap)) if os.path.exists(ap) else {}
        allrep[cutoff] = rep
        json.dump(allrep, open(ap, "w"), indent=1)
        print(json.dumps(rep, indent=1))
        return
    out.to_csv(D.OUT + tag + ".csv", index=False, float_format="%.6g")
    json.dump({"fo_theta_end": th.tolist(), "rows": int(len(out))},
              open(D.OUT + tag + "_build_meta.json", "w"), indent=1)
    print(f"build done rows={len(out)} {time.time()-t0:.0f}s")
