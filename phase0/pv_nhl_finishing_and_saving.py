"""Player-value program, NHL -- component FINISHING_AND_SAVING (shooter vs goalie).

The plate-appearance analogue.  Every unblocked, non-shootout shot at a goalie in
net is a 1v1 between the shooter and the goalie on one outcome (goal / no goal),
with the shot's xG as the prior expectation:

    logit P(goal) = logit(xG) + c[cell]                      league recalibration
                    + a_shooter - b_goalie                    the 1v1 (players)
                    + o_team(shooting) - d_team(defending)    team context
                    + k_rink(home arena)                      scorer / building

  xG        MoneyPuck xGoal (shot quantity & location & type)
  c[cell]   league RECALIBRATION of xG, one offset per (shot type x strength x
            shooter F/D) cell, walk-forward (online Newton, updated only at date
            boundaries).  Absorbs season-level drift (DEV goals/xG 0.971..1.009 by
            season) and xG mis-calibration by shot type (tip-ins 0.94, snaps 1.05),
            so a player is graded against what the league converts from that shot.
  a_s       shooter finishing (goals above expected, logit; includes accuracy)
  b_g       goalie saving (goals prevented, logit; + = better)
  o_t, d_t  team finishing / team save environment not carried by the individual
            (pre-shot passing the xG cannot see; screens; defensive coverage).  A
            DEV check found teammate goalies' same-season GSAx correlate +0.13, i.e.
            part of "goalie GSAx" is the team in front of him -- without d_t that
            is credited to whoever is in net.
  k_r       rink: home-arena scorer bias in shot recording (the QA found missed-
            shot recording home/road ratios of 0.71..1.40 by arena) changes the
            recorded xG per real chance, for BOTH teams' shots in that building.

Why a Bayesian 1v1 and not EB goals-above-expected (GSAx):
  * GSAx is not opponent-adjusted: the 1v1 grades each side against the other's
    current estimate (batter-vs-pitcher TrueSkill in MLB).
  * EB shrinkage falls out: every entity carries a Gaussian belief N(mu, v) with
    prior N(0, S0[group]^2), S0 = true-talent spread of its group, fitted on DEV
    by prequential likelihood (never on TEST).
  * Process over results: evidence decays -- per-appearance AR(1) toward the
    prior (RHO_G) and a season-boundary step (RHO_S), the analogue of the MLB FIP
    EWMA.

Update (assumed-density / Laplace, batched per game so every prediction inside a
game uses the PRE-GAME state):
    for each shot with participants e (sign s_e): z = z_rc + sum_e s_e mu_e,
        p = sigmoid(z), h = p(1-p), r = y - p,
        V_e' = sum of the OTHER participants' variances,
        G_e += s_e r / (1 + h V_e'),  H_e += h / (1 + h V_e')
      (1/(1 + h V') marginalises the others' uncertainty -- the Glicko g(RD) term)
    at game end, per entity: mu += v G / (1 + v H);  v = v / (1 + v H)
    cells: accumulated over the whole DATE, P = LAM P + (1-LAM) P0 + H_day,
           c += (G_day - (1-LAM) P0 c) / P.

Walk-forward guarantees (asserted / audited):
  * games processed in (date, gid) order; a player appears in at most one game per
    date; entity state is read before the game and written after it;
  * cell offsets change only between dates;
  => every value written for game i is a function of games on EARLIER dates.
  phase0/pv_nhl_finishing_and_saving_validate.py permutes all outcomes after a
  cutoff and checks every earlier value is bitwise unchanged.

DEV only by default.  `--allow-test` extends the run through TEST seasons for the
lead engineer's single TEST look / serving, with hyperparameters frozen from the
DEV fit in data/pv_nhl_finishing_and_saving_params.json (never re-fit).

Outputs (DEV run):
  data/pv_nhl_finishing_and_saving_player_games.parquet  one row per dressed player
      per game: pre-game mu/sd (walk-forward) + the observed post-game line
  data/pv_nhl_finishing_and_saving_entity_games.parquet  same for team / rink terms
  data/pv_nhl_finishing_and_saving_shots.parquet         per-shot pre-game terms
  data/pv_nhl_finishing_and_saving_final_state.parquet   state after the last game
  data/pv_nhl_finishing_and_saving_params.json           hyperparameters + grid
  data/pv_nhl_finishing_and_saving_team_games.csv        per team-game pre-game lineup
      values in goals (skaters' finishing, starting goalie's saving, team context)

Market-blind: nothing here reads odds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pv_nhl_io import DEV_MAX_GID, load_events, load_rosters  # noqa: E402

OUT_PG = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_player_games.parquet")
OUT_EG = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_entity_games.parquet")
OUT_SH = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_shots.parquet")
OUT_FIN = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_final_state.parquet")
OUT_PAR = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_params.json")
OUT_TG = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_team_games.csv")

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"]
GROUPS = ["G", "F", "D", "TO", "TD", "RK"]      # goalie, fwd, def, team-off, team-def, rink
GI = {g: i for i, g in enumerate(GROUPS)}
EVAL_FROM = 20112012          # 2010-11 is warm-up (left-censored careers)
TUNE_TO = 20142015            # hyperparameters chosen on 2011-12..2014-15 prequential LL
XG_CLIP = (1e-4, 0.995)
OFF = 1e-6                    # an S0 this small switches a group off

# fixed (not searched): cell-offset prior sd and per-date forgetting (half-life ~1 season)
CELL_SD0 = 0.10
CELL_LAM = 0.5 ** (1.0 / 190.0)


# ------------------------------------------------------------------ prep ---
def _group(pos):
    return "G" if pos == "G" else ("D" if pos == "D" else "F")


def prep(allow_test=False, verbose=True):
    t0 = time.time()
    cols = ["season", "gtype", "period", "ptype", "game_sec", "sort", "ev", "team",
            "is_home", "shooter", "goalie", "xg", "sk_for", "sk_ag", "shot_type", "dist"]
    ev = load_events(columns=cols, allow_test=allow_test)
    ro = load_rosters(allow_test=allow_test)
    gm = pd.read_csv(os.path.join(ROOT, "data/nhl_games.csv"),
                     usecols=["game_id", "date", "season", "type", "home", "away"])
    if not allow_test:
        gm = gm[gm.game_id < DEV_MAX_GID]
        assert int(gm.game_id.max()) < DEV_MAX_GID
    gm = gm[gm.game_id.isin(set(ro.gid.unique()))]
    gm = gm.sort_values(["date", "game_id"]).reset_index(drop=True)
    gm["gi"] = np.arange(len(gm))
    gi_of = dict(zip(gm.game_id, gm.gi))

    # ---- shots: unblocked, non-shootout, goalie in net, xG present
    s = ev[ev.ev.isin(["goal", "shot-on-goal", "missed-shot"])
           & (ev.ptype != "SO") & ev.xg.notna() & ev.goalie.notna()
           & ev.shooter.notna()].copy()
    s["y"] = (s.ev == "goal").astype(np.int8)
    s["onnet"] = s.ev.isin(["goal", "shot-on-goal"]).astype(np.int8)
    s["gi"] = s.gid.map(gi_of)
    assert s.gi.notna().all()
    s["gi"] = s.gi.astype(np.int64)

    # ---- players & groups (majority roster position)
    pos = ro.groupby("pid").pos.agg(lambda x: x.value_counts().index[0])
    grp = pos.map(_group)
    s = s[s.shooter.map(grp).fillna("F") != "G"]         # 28 goalie shots dropped
    pids = np.array(sorted(set(ro.pid.astype(np.int64)) | set(s.shooter.astype(np.int64))
                           | set(s.goalie.astype(np.int64))))
    P = len(pids)
    pidx = {p: i for i, p in enumerate(pids)}
    teams = sorted(set(gm.home) | set(gm.away))
    T = len(teams)
    tidx = {t: i for i, t in enumerate(teams)}
    e_group = np.concatenate([
        np.array([GI[grp.get(p, "F")] for p in pids], dtype=np.int64),
        np.full(T, GI["TO"]), np.full(T, GI["TD"]), np.full(T, GI["RK"])]).astype(np.int64)
    e_TO, e_TD, e_RK = P, P + T, P + 2 * T

    # ---- cells: shot type x strength x shooter F/D
    st = np.where(s.sk_for > s.sk_ag, 1, np.where(s.sk_for < s.sk_ag, 2, 0))
    ty = s.shot_type.map({t: i for i, t in enumerate(SHOT_TYPES)}).fillna(len(SHOT_TYPES))
    fd = (s.shooter.map(grp) == "D").astype(int).values
    s["cell"] = (ty.astype(int).values * 3 + st) * 2 + fd
    n_cells = (len(SHOT_TYPES) + 1) * 3 * 2
    xg = s.xg.clip(*XG_CLIP)
    s["z0"] = np.log(xg / (1 - xg))
    s = s.sort_values(["gi", "sort"]).reset_index(drop=True)
    hm = gm.set_index("gi").home
    aw = gm.set_index("gi").away
    s_home = hm.loc[s.gi].values
    s_away = aw.loc[s.gi].values
    # team resolution check: acting team must be the game's home/away code
    assert (np.where(s.is_home == 1, s_home, s_away) == s.team.values).all()
    shoot_t = np.array([tidx[x] for x in s.team.values])
    def_t = np.array([tidx[x] for x in np.where(s.is_home == 1, s_away, s_home)])
    rink_t = np.array([tidx[x] for x in s_home])
    s_ent = np.column_stack([
        s.shooter.astype(np.int64).map(pidx).values,
        s.goalie.astype(np.int64).map(pidx).values,
        e_TO + shoot_t, e_TD + def_t, e_RK + rink_t]).astype(np.int64)
    s_sgn = np.tile(np.array([1.0, -1.0, 1.0, -1.0, 1.0]), (len(s), 1))

    # ---- roster entries per game = dressed players UNION shot participants
    ro = ro[ro.gid.isin(gi_of)].copy()
    ro["gi"] = ro.gid.map(gi_of).astype(np.int64)
    ro["pid"] = ro.pid.astype(np.int64)
    ro["side_h"] = (ro.side == "H").astype(np.int8)
    extra = []
    for col in ("shooter", "goalie"):
        e = s[["gid", "gi", col, "is_home"]].rename(columns={col: "pid"})
        if col == "goalie":          # the defending side
            e = e.assign(is_home=1 - e.is_home)
        e = e.assign(side_h=e.is_home.astype(np.int8))
        extra.append(e[["gid", "gi", "pid", "side_h"]])
    ex = pd.concat(extra).drop_duplicates(["gi", "pid"])
    ex["pid"] = ex.pid.astype(np.int64)
    have = set(zip(ro.gi, ro.pid))
    ex = ex[[(a, b) not in have for a, b in zip(ex.gi, ex.pid)]]
    if len(ex):
        ex["season"] = gm.set_index("gi").loc[ex.gi, "season"].values
        ex["pos"] = ex.pid.map(pos).fillna("F")
        ex["team"] = np.where(ex.side_h == 1, hm.loc[ex.gi].values, aw.loc[ex.gi].values)
        ro = pd.concat([ro, ex[["gid", "gi", "pid", "side_h", "season", "pos", "team"]]],
                       ignore_index=True)
    ro["ei"] = ro.pid.map(pidx).astype(np.int64)
    ro["kind"] = "P"
    dd = ro.merge(gm[["gi", "date"]], on="gi")
    assert not dd.duplicated(["date", "pid"]).any(), "player in two games on one date"

    # team / rink entity appearances: 5 per game
    te = []
    for kind, base, col in (("TO", e_TO, "home"), ("TD", e_TD, "home"), ("TO", e_TO, "away"),
                            ("TD", e_TD, "away"), ("RK", e_RK, "home")):
        te.append(pd.DataFrame(dict(gid=gm.game_id, gi=gm.gi, season=gm.season,
                                    side_h=np.int8(1 if col == "home" else 0),
                                    team=gm[col].values, kind=kind,
                                    ei=base + np.array([tidx[x] for x in gm[col]]))))
    te = pd.concat(te, ignore_index=True)
    ent = pd.concat([ro, te], ignore_index=True)
    ent = ent.sort_values(["gi", "kind", "side_h", "ei"]).reset_index(drop=True)

    n_games = len(gm)
    date_codes = pd.factorize(gm.date)[0].astype(np.int64)
    ar = np.arange(n_games)
    D = dict(
        gm=gm, s=s, ent=ent, pids=pids, teams=teams, P=P, T=T, e_group=e_group,
        n_cells=n_cells, g_date=date_codes, g_season=gm.season.values.astype(np.int64),
        g_s0=np.searchsorted(s.gi.values, ar, "left").astype(np.int64),
        g_s1=np.searchsorted(s.gi.values, ar, "right").astype(np.int64),
        g_r0=np.searchsorted(ent.gi.values, ar, "left").astype(np.int64),
        g_r1=np.searchsorted(ent.gi.values, ar, "right").astype(np.int64),
        s_ent=s_ent, s_sgn=s_sgn, s_z0=s.z0.values.astype(np.float64),
        s_cell=s.cell.values.astype(np.int64), s_y=s.y.values.astype(np.float64),
        r_ei=ent.ei.values.astype(np.int64))
    if verbose:
        print(f"prep: {n_games} games, {len(s)} shots, {len(ent)} entity-games, "
              f"{P} players, {T} teams, {n_cells} cells, {time.time()-t0:.1f}s", flush=True)
    return D


# ---------------------------------------------------------------- kernel ---
@njit(cache=True)
def _kernel(e_group, g_date, g_season, g_s0, g_s1, g_r0, g_r1,
            s_ent, s_sgn, s_z0, s_cell, s_y, r_ei, n_cells,
            S0, RHOG, RHOS, cell_sd0, cell_lam):
    n_ent = e_group.shape[0]
    n_games = g_date.shape[0]
    n_sh, K = s_ent.shape
    n_ro = r_ei.shape[0]
    mu = np.zeros(n_ent)
    v = np.empty(n_ent)
    for k in range(n_ent):
        v[k] = S0[e_group[k]] ** 2
    last_season = np.full(n_ent, -1, np.int64)
    last_date = np.full(n_ent, -1, np.int64)
    c = np.zeros(n_cells)
    Pc = np.full(n_cells, 1.0 / cell_sd0 ** 2)
    cg = np.zeros(n_cells)
    ch = np.zeros(n_cells)
    accg = np.zeros(n_ent)
    acch = np.zeros(n_ent)
    r_mu = np.empty(n_ro)
    r_v = np.empty(n_ro)
    s_zrc = np.empty(n_sh)          # logit of the recalibrated baseline
    s_term = np.empty((n_sh, K))    # signed pre-game contribution of each participant
    s_var = np.empty((n_sh, K))
    cur_date = -1
    pr = (1.0 - cell_lam) / cell_sd0 ** 2
    for gi in range(n_games):
        d = g_date[gi]
        if d != cur_date:
            for k in range(n_cells):
                Pc[k] = cell_lam * Pc[k] + pr + ch[k]
                c[k] += (cg[k] - pr * c[k]) / Pc[k]
                cg[k] = 0.0
                ch[k] = 0.0
            cur_date = d
        season = g_season[gi]
        # ---- pre-game drift + snapshot
        for r in range(g_r0[gi], g_r1[gi]):
            k = r_ei[r]
            if last_date[k] == d:
                raise ValueError("entity twice on one date")
            gr = e_group[k]
            s02 = S0[gr] ** 2
            if last_season[k] != season:
                if last_season[k] >= 0:
                    rs = RHOS[gr]
                    mu[k] = rs * mu[k]
                    v[k] = rs * rs * v[k] + (1.0 - rs * rs) * s02
                last_season[k] = season
            rg = RHOG[gr]
            mu[k] = rg * mu[k]
            v[k] = rg * rg * v[k] + (1.0 - rg * rg) * s02
            last_date[k] = d
            r_mu[r] = mu[k]
            r_v[r] = v[k]
        # ---- shots (pre-game states; accumulate)
        for j in range(g_s0[gi], g_s1[gi]):
            cc = s_cell[j]
            zb = s_z0[j] + c[cc]
            z = zb
            vt = 0.0
            for q in range(K):
                e = s_ent[j, q]
                t = s_sgn[j, q] * mu[e]
                s_term[j, q] = t
                s_var[j, q] = v[e]
                z += t
                vt += v[e]
            s_zrc[j] = zb
            p = 1.0 / (1.0 + np.exp(-z))
            h = p * (1.0 - p)
            res = s_y[j] - p
            for q in range(K):
                e = s_ent[j, q]
                w = 1.0 / (1.0 + h * (vt - v[e]))
                accg[e] += s_sgn[j, q] * res * w
                acch[e] += h * w
            cg[cc] += res
            ch[cc] += h
        # ---- post-game update
        for r in range(g_r0[gi], g_r1[gi]):
            k = r_ei[r]
            if acch[k] > 0.0:
                mu[k] += v[k] * accg[k] / (1.0 + v[k] * acch[k])
                v[k] = v[k] / (1.0 + v[k] * acch[k])
                accg[k] = 0.0
                acch[k] = 0.0
    return r_mu, r_v, s_zrc, s_term, s_var, mu, v, c


def run(D, S0, RHOG, RHOS, cell_sd0=CELL_SD0, cell_lam=CELL_LAM, y=None):
    return _kernel(D["e_group"], D["g_date"], D["g_season"], D["g_s0"], D["g_s1"],
                   D["g_r0"], D["g_r1"], D["s_ent"], D["s_sgn"], D["s_z0"], D["s_cell"],
                   D["s_y"] if y is None else y, D["r_ei"], D["n_cells"],
                   np.asarray(S0, float), np.asarray(RHOG, float),
                   np.asarray(RHOS, float), float(cell_sd0), float(cell_lam))


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def full_p(out):
    return sig(out[2] + out[3].sum(axis=1))


# ---------------------------------------------------------------- tuning ---
def _vec(d):
    """dict of per-group (S0, RHOG, RHOS) -> three arrays in GROUPS order."""
    S0 = [d[g][0] for g in GROUPS]
    RG = [d[g][1] for g in GROUPS]
    RS = [d[g][2] for g in GROUPS]
    return S0, RG, RS


def tune(D, verbose=True):
    """Coordinate grid search of the prequential (pre-game) shot log-loss on DEV
    2011-12..2014-15 (TUNE window).  The untouched DEV seasons 2015-16..2017-18 are
    scored for every grid point as a check only (never used to choose)."""
    seas = D["gm"].season.values[D["s"].gi.values]
    y = D["s_y"]
    wt = (seas >= EVAL_FROM) & (seas <= TUNE_TO)
    wv = seas > TUNE_TO
    grid = []
    cfg = {"G": (OFF, 1.0, 1.0), "F": (OFF, 1.0, 1.0), "D": (OFF, 1.0, 1.0),
           "TO": (OFF, 1.0, 1.0), "TD": (OFF, 1.0, 1.0), "RK": (OFF, 1.0, 1.0)}
    t0 = time.time()

    def score(c):
        out = run(D, *_vec(c))
        L = ll(y, full_p(out))
        return float(L[wt].mean()), float(L[wv].mean())

    def stage(name, groups, spaces):
        nonlocal cfg
        best = None
        for combo in itertools.product(*spaces):
            c = dict(cfg)
            for g in groups:
                c[g] = combo
            lt, lv = score(c)
            row = dict(stage=name, groups="+".join(groups), S0=combo[0], RHOG=combo[1],
                       RHOS=combo[2], ll_tune=lt, ll_check=lv)
            grid.append(row)
            if best is None or lt < best[0]:
                best = (lt, lv, c)
        cfg = best[2]
        if verbose:
            print(f"{name:10s} {groups} -> {[cfg[g] for g in groups]}  tune {best[0]:.7f} "
                  f"check {best[1]:.7f}  ({time.time()-t0:.0f}s)", flush=True)

    lt0, lv0 = score(cfg)
    grid.append(dict(stage="recal_only", groups="", S0=0, RHOG=1, RHOS=1, ll_tune=lt0, ll_check=lv0))
    gsp = ([0.03, 0.05, 0.07, 0.10, 0.13, 0.16], [1.0, 0.999, 0.998, 0.997, 0.995, 0.99],
           [1.0, 0.9, 0.75, 0.6, 0.45])
    fsp = ([0.10, 0.13, 0.16, 0.20, 0.25], [1.0, 0.9995, 0.999, 0.998, 0.997],
           [1.0, 0.95, 0.9, 0.8])
    dsp = ([0.06, 0.10, 0.15, 0.20, 0.25, 0.32], [1.0, 0.9995, 0.999, 0.998, 0.997],
           [1.0, 0.95, 0.9, 0.8])
    tsp = ([OFF, 0.02, 0.03, 0.05, 0.08], [1.0, 0.998, 0.995, 0.99], [1.0, 0.8, 0.6, 0.4])
    rsp = ([OFF, 0.02, 0.03, 0.05, 0.08, 0.12], [1.0, 0.995, 0.99], [1.0, 0.9, 0.7, 0.5])
    stage("goalie", ["G"], gsp)
    stage("fwd", ["F"], fsp)
    stage("def", ["D"], dsp)
    stage("team_off", ["TO"], tsp)
    stage("team_def", ["TD"], tsp)
    stage("rink", ["RK"], rsp)
    stage("team_off2", ["TO"], tsp)       # second pass: context given the rink
    stage("team_def2", ["TD"], tsp)
    stage("rink2", ["RK"], rsp)
    stage("goalie2", ["G"], gsp)          # second pass: players given the context
    stage("fwd2", ["F"], fsp)
    stage("def2", ["D"], dsp)
    return cfg, grid


# ------------------------------------------------------------------ main ---
def build_outputs(D, cfg, allow_test=False):
    S0, RG, RS = _vec(cfg)
    r_mu, r_v, s_zrc, s_term, s_var, mu, v, c = run(D, S0, RG, RS)
    s = D["s"]
    z_sh, z_go, z_to, z_td, z_rk = (s_term[:, q] for q in range(5))
    sh = pd.DataFrame(dict(
        gid=s.gid.values, gi=s.gi.values, season=s.season.values, gtype=s.gtype.values,
        sort=s.sort.values, team=s.team.values, is_home=s.is_home.values,
        shooter=s.shooter.astype(np.int64).values, goalie=s.goalie.astype(np.int64).values,
        cell=s.cell.values, onnet=s.onnet.values, y=s.y.values, xg=s.xg.values,
        z_rc=s_zrc, t_sh=z_sh, t_go=z_go, t_to=z_to, t_td=z_td, t_rk=z_rk,
        v_sh=s_var[:, 0], v_go=s_var[:, 1]))
    sh["p_rc"] = sig(sh.z_rc)
    sh["p_sh"] = sig(sh.z_rc + sh.t_sh)
    sh["p_go"] = sig(sh.z_rc + sh.t_go)
    sh["p_ctx"] = sig(sh.z_rc + sh.t_to + sh.t_td + sh.t_rk)
    sh["p_fs"] = sig(sh.z_rc + s_term.sum(axis=1))
    ent = D["ent"].copy()
    ent["mu"] = r_mu
    ent["sd"] = np.sqrt(r_v)
    pl = ent[ent.kind == "P"].copy()
    pl["grp"] = np.array(GROUPS)[D["e_group"][pl.ei.values]]
    # observed post-game lines (labels for validation; never features)
    a = sh.groupby(["gi", "shooter"]).agg(sh_n=("y", "size"), sh_xg=("xg", "sum"),
                                         sh_xrc=("p_rc", "sum"), sh_xctx=("p_ctx", "sum"),
                                         sh_g=("y", "sum"), sh_on=("onnet", "sum")).reset_index()
    b = sh.groupby(["gi", "goalie"]).agg(fa_n=("y", "size"), fa_xg=("xg", "sum"),
                                        fa_xrc=("p_rc", "sum"), fa_xctx=("p_ctx", "sum"),
                                        fa_g=("y", "sum"), fa_on=("onnet", "sum")).reset_index()
    pg = pl.merge(a.rename(columns={"shooter": "pid"}), on=["gi", "pid"], how="left")
    pg = pg.merge(b.rename(columns={"goalie": "pid"}), on=["gi", "pid"], how="left")
    for col in ["sh_n", "sh_xg", "sh_xrc", "sh_xctx", "sh_g", "sh_on",
                "fa_n", "fa_xg", "fa_xrc", "fa_xctx", "fa_g", "fa_on"]:
        pg[col] = pg[col].fillna(0.0)
    pg = pg.merge(D["gm"][["gi", "date"]], on="gi", how="left")
    pg["pid"] = pg.pid.astype(np.int64)
    pg["gtype"] = (pg.gid // 10000 % 10).astype(int)
    keep = ["gid", "gi", "date", "season", "gtype", "team", "side_h", "pid", "pos", "grp",
            "g_start", "g_net", "toi_s", "mu", "sd",
            "sh_n", "sh_xg", "sh_xrc", "sh_xctx", "sh_g", "sh_on",
            "fa_n", "fa_xg", "fa_xrc", "fa_xctx", "fa_g", "fa_on"]
    pg = pg[keep].sort_values(["gi", "side_h", "pid"]).reset_index(drop=True)
    eg = ent[ent.kind != "P"][["gid", "gi", "season", "side_h", "team", "kind", "mu", "sd"]]
    eg = eg.merge(D["gm"][["gi", "date"]], on="gi", how="left").reset_index(drop=True)
    if not allow_test:
        for df in (pg, sh, eg):
            assert int(df.gid.max()) < DEV_MAX_GID
    T = D["T"]
    names = list(D["pids"]) + [f"{k}:{t}" for k in ("TO", "TD", "RK") for t in D["teams"]]
    final = pd.DataFrame(dict(entity=[str(x) for x in names],
                              grp=np.array(GROUPS)[D["e_group"]], mu=mu, sd=np.sqrt(v)))
    final["pid"] = np.r_[D["pids"], np.full(3 * T, -1)]
    return pg, eg, sh, final, c


def build_team_games(pg, eg):
    """Per team-game PRE-GAME lineup values in goals/game (+ realised post-game labels).

    fin_pre  sum over dressed skaters of ixG/GP (walk-forward EWMA, half-life 40 games,
             10-game prior at the league position mean of strictly-earlier dates)
             x (exp(mu_finishing) - 1)
    sav_pre  starting goalie: league xG per team-game (strictly earlier dates)
             x (1 - exp(-mu_saving))
    fs_pre   fin_pre + sav_pre;  env_off/env_def: team-context terms, same scale;
             fs_env_pre = fs_pre + env_off_pre + env_def_pre.
    Realised labels (fin_real = GF - xGF_rc, sav_real = xGA_rc - GA) are post-game
    and exist for validation only."""
    pg = pg.sort_values("gi").copy()
    sk = (pg.grp.values != "G")
    rate = np.full(len(pg), np.nan)
    st = {}
    prior = {"F": 0.35, "D": 0.17}
    lsum = {"F": [0.0, 0], "D": [0.0, 0]}
    lam = 0.5 ** (1 / 40)
    cur, pend = None, []
    for i, (pid, grp, x, date, isk) in enumerate(zip(pg.pid.values, pg.grp.values,
                                                     pg.sh_xrc.values, pg.date.values, sk)):
        if not isk:
            continue
        if date != cur:                         # league means: strictly earlier dates
            for g_, x_ in pend:
                lsum[g_][0] += x_
                lsum[g_][1] += 1
            pend = []
            for g_ in ("F", "D"):
                if lsum[g_][1] > 1000:
                    prior[g_] = lsum[g_][0] / lsum[g_][1]
            cur = date
        s = st.get(pid)
        if s is None:
            s = st[pid] = [prior[grp] * 10, 10.0]
        rate[i] = s[0] / s[1]
        s[0] = lam * s[0] + x
        s[1] = lam * s[1] + 1
        pend.append((grp, x))
    pg["xg_rate_pre"] = rate
    pg["fin_pre"] = np.where(sk, pg.xg_rate_pre * (np.exp(pg.mu) - 1), 0.0)
    gs = pg[(pg.grp == "G") & (pg.g_start == 1)][["gi", "side_h", "pid", "mu", "sd"]]
    gs = gs.rename(columns={"pid": "g_pid", "mu": "g_mu", "sd": "g_sd"})
    tg = pg[sk].groupby(["gi", "gid", "date", "season", "gtype", "side_h"]).agg(
        fin_pre=("fin_pre", "sum"), n_sk=("pid", "size"), gf=("sh_g", "sum"),
        xgf_rc=("sh_xrc", "sum"), xgf=("sh_xg", "sum")).reset_index()
    tg = tg.merge(gs, on=["gi", "side_h"], how="left")
    e = eg.pivot_table(index=["gi", "side_h"], columns="kind", values="mu").reset_index()
    e = e.rename(columns={"TO": "to_mu", "TD": "td_mu", "RK": "rk_mu"})
    tg = tg.merge(e[["gi", "side_h", "to_mu", "td_mu"]], on=["gi", "side_h"], how="left")
    tm = eg[eg.kind == "TO"][["gi", "side_h", "team"]]
    tg = tg.merge(tm, on=["gi", "side_h"], how="left")
    daily = tg.groupby("date").xgf_rc.agg(["sum", "size"]).sort_index()
    lgx = (daily["sum"].cumsum().shift(1) / daily["size"].cumsum().shift(1)).fillna(2.5)
    tg["lg_xg_pre"] = tg.date.map(lgx)
    tg["sav_pre"] = tg.lg_xg_pre * (1 - np.exp(-tg.g_mu.fillna(0)))
    tg["fs_pre"] = tg.fin_pre + tg.sav_pre
    tg["env_off_pre"] = tg.lg_xg_pre * (np.exp(tg.to_mu) - 1)
    tg["env_def_pre"] = tg.lg_xg_pre * (1 - np.exp(-tg.td_mu))
    tg["fs_env_pre"] = tg.fs_pre + tg.env_off_pre + tg.env_def_pre
    opp = tg[["gi", "side_h", "gf", "xgf_rc"]].copy()
    opp["side_h"] = 1 - opp.side_h
    tg = tg.merge(opp.rename(columns={"gf": "ga", "xgf_rc": "xga_rc"}), on=["gi", "side_h"],
                  how="left")
    tg["fin_real"] = tg.gf - tg.xgf_rc
    tg["sav_real"] = tg.xga_rc - tg.ga
    return tg.sort_values(["gi", "side_h"]).reset_index(drop=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="re-run the DEV grid search")
    ap.add_argument("--allow-test", action="store_true",
                    help="LEAD ENGINEER ONLY: extend through TEST seasons with frozen params")
    args = ap.parse_args()
    t0 = time.time()
    D = prep(allow_test=args.allow_test)
    if args.tune:
        assert not args.allow_test, "tuning is DEV-only"
        cfg, grid = tune(D)
        json.dump(dict(params=cfg, groups=GROUPS, fixed=dict(CELL_SD0=CELL_SD0, CELL_LAM=CELL_LAM),
                       tune_window=[EVAL_FROM, TUNE_TO], check_window=[TUNE_TO + 10001, 20172018],
                       grid=grid), open(OUT_PAR, "w"), indent=1)
    cfg = {g: tuple(v) for g, v in json.load(open(OUT_PAR))["params"].items()}
    print("params", cfg, flush=True)
    pg, eg, sh, final, c = build_outputs(D, cfg, allow_test=args.allow_test)
    sfx = "_alltest" if args.allow_test else ""
    for df, path in ((pg, OUT_PG), (eg, OUT_EG), (sh, OUT_SH), (final, OUT_FIN)):
        df.to_parquet(path.replace(".parquet", f"{sfx}.parquet"), index=False)
    tg = build_team_games(pg, eg)
    if not args.allow_test:
        assert int(tg.gid.max()) < DEV_MAX_GID
    tg.to_csv(OUT_TG.replace(".csv", f"{sfx}.csv"), index=False)
    print(f"wrote {len(pg)} player-games, {len(eg)} entity-games, {len(sh)} shots "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
