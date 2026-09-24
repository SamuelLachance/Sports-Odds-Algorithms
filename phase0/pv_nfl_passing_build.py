"""Player value, NFL component 1: PASSING - the QB on the events he controls. DEV ONLY.

The MLB principle applied to the quarterback:
  * rate the QB only on his own dropbacks (attempts, sacks, scrambles), with the
    per-play outcome he controls: qb_epa (EPA that stops at a receiver's
    post-catch fumble), success, CPOE (completion over expected) and sacks taken;
  * every dropback is a 1v1 duel QB vs the opposing PASS DEFENCE, both sides
    rated jointly and opponent-adjusted (a two-state Gaussian Kalman / continuous
    TrueSkill: obs = mu_league + q_QB + d_DEF + noise, noise var sig2 / n);
  * walk-forward, empirical-Bayes: a new QB starts at a replacement-level prior
    (m0q, v0q), talent drifts per appearance (tq) and across seasons (tsq, rq),
    and the hyper-parameters are type-II maximum likelihood on one-step-ahead
    predictions in a WARM-UP window only (1999-2005; CPOE, which starts in 2006,
    uses 2007-2008 with 2006 as burn-in);
  * the component ratings are combined into one 'passing' composite (EPA/dropback
    units, opponent-neutral) by a season-by-season walk-forward regression of the
    QB's next-game EPA/dropback on his pre-game component ratings, controlling for
    the opponent's pre-game defence ratings (FIP analog: stable, QB-controlled
    components get the weight their predictive value earns).

Also reproduces the shipped QbElo feature (X14 col 1) on the same QB-games, both
keyed on nflverse's listed QB (post-game primary passer, as cached) and on the
starter (nfl_qb_elo.apply_starters, as served since ade264f2).

Constitution: no odds read; seasons >= 2016 are filtered at load (pyarrow filter)
and hard-asserted absent; every value written for a game uses only games dated
strictly before it (league means fold at date boundaries; a game's own updates
are applied after all of its pre-game values are recorded).

Outputs
  data/pv_nfl_passing_values.csv    per player per game, PRE-game walk-forward values
  data/pv_nfl_passing_outcomes.csv  per player per game, POST-game outcomes (validation only)
  data/pv_nfl_passing_team.csv      per team-game: starter, unit outcomes (validation only)
  data/pv_nfl_passing_hyper.json    fitted hyper-parameters, composite weights by fold
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit
from scipy.optimize import minimize

T0 = time.time()
TEST_ERA = 2016
PQ = "data/pv_nfl_events.parquet"
LAM = 0.5 ** (1.0 / 25.0)     # league-mean EWMA per game-date (~1.2 seasons half-life)


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


# ======================================================================= data
COLS = ["game_id", "play_id", "season", "season_type", "week", "game_date", "home_team",
        "away_team", "posteam", "defteam", "play_type", "qb_dropback", "qb_scramble",
        "pass_attempt", "sack", "complete_pass", "interception", "epa", "qb_epa", "success",
        "cpoe", "passer_id", "rusher_id", "two_point_attempt", "fixed_drive", "touchdown",
        "td_team", "field_goal_result", "extra_point_result", "two_point_conv_result"]


def load_plays():
    tb = ds.dataset(PQ).to_table(columns=COLS, filter=ds.field("season") < TEST_ERA)
    df = tb.to_pandas()
    assert len(df) and int(df.season.max()) < TEST_ERA, "TEST season present"
    df = df.sort_values(["game_date", "game_id", "play_id"], kind="mergesort").reset_index(drop=True)
    return df


def qb_positions():
    pos = {}
    with open("data/nfl_players.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pos[r["gsis_id"]] = r["position"]
    return pos


def aggregate(df):
    """QB-game and team-game tables."""
    pt = df.play_type.fillna("")
    db = ((df.qb_dropback == 1) & pt.isin(["pass", "run"]) & (df.two_point_attempt != 1)
          & df.passer_id.notna() & df.qb_epa.notna()).fillna(False).to_numpy(bool)
    att = db & ((df.pass_attempt == 1) & (df.sack == 0)).fillna(False).to_numpy(bool)
    pos = qb_positions()
    is_qb = df.rusher_id.map(lambda x: pos.get(x) == "QB" if isinstance(x, str) else False).to_numpy(bool)
    rush = ((pt == "run") & (df.qb_dropback != 1) & (df.two_point_attempt != 1)
            & df.epa.notna()).fillna(False).to_numpy(bool) & is_qb

    d = df.loc[db, ["game_id", "posteam", "defteam", "passer_id", "play_id", "qb_epa", "success",
                    "sack", "interception", "qb_scramble", "cpoe", "pass_attempt"]].copy()
    d["att"] = att[db].astype(float)
    d["cp_n"] = (d["att"] > 0) & d.cpoe.notna()
    d["cpoe_f"] = np.where(d.cp_n, d.cpoe.astype(float) / 100.0, 0.0)
    d["int_att"] = np.where(d.att > 0, d.interception.fillna(0).astype(float), 0.0)
    d["qb_epa2"] = d.qb_epa ** 2
    d["cpoe_f2"] = d.cpoe_f ** 2
    g = d.groupby(["game_id", "posteam", "passer_id"], sort=False)
    qg = pd.DataFrame({
        "defteam": g.defteam.first(), "first_play": g.play_id.min(),
        "n_db": g.size().astype(float), "s_epa": g.qb_epa.sum(), "s_epa2": g.qb_epa2.sum(),
        "s_succ": g.success.sum().astype(float), "n_sack": g.sack.sum().astype(float),
        "n_scr": g.qb_scramble.sum().astype(float), "n_att": g.att.sum(),
        "n_int": g.int_att.sum(), "n_cp": g.cp_n.sum().astype(float),
        "s_cpoe": g.cpoe_f.sum(), "s_cpoe2": g.cpoe_f2.sum(),
    }).reset_index().rename(columns={"passer_id": "qb"})
    r = df.loc[rush, ["game_id", "posteam", "rusher_id", "epa"]]
    rg = r.groupby(["game_id", "posteam", "rusher_id"]).epa.agg(["size", "sum"]).reset_index()
    rg.columns = ["game_id", "posteam", "qb", "n_rush", "s_rush"]
    qg = qg.merge(rg, on=["game_id", "posteam", "qb"], how="left")
    qg[["n_rush", "s_rush"]] = qg[["n_rush", "s_rush"]].fillna(0.0)

    # ---- team-game: starter (first dropback passer), unit outcomes
    first = d.sort_values("play_id").groupby(["game_id", "posteam"]).passer_id.first()
    tdb = d.groupby(["game_id", "posteam"]).agg(u_db=("qb_epa", "size"), u_epa=("qb_epa", "sum"))
    # offensive points: offensive TDs (pass/run, scored by posteam), made FGs, and the
    # conversion that follows an OFFENSIVE touchdown of the same team
    pts = {}
    drives = {}
    cur_game, last_td = None, None
    sc = df[["game_id", "posteam", "play_type", "touchdown", "td_team", "field_goal_result",
             "extra_point_result", "two_point_conv_result", "two_point_attempt", "fixed_drive"]].copy()
    for c in ("touchdown", "two_point_attempt", "fixed_drive"):
        sc[c] = sc[c].fillna(-1).astype(int)
    for c in ("posteam", "play_type", "td_team", "field_goal_result", "extra_point_result",
              "two_point_conv_result"):
        sc[c] = sc[c].astype(object).where(sc[c].notna(), "")
    for row in sc.itertuples(index=False):
        gid, pos_, ptyp, tdf, tdt, fgr, xpr, tpr, tpa, drv = row
        if gid != cur_game:
            cur_game, last_td = gid, None
        if not isinstance(pos_, str) or not pos_:
            continue
        key = (gid, pos_)
        pts.setdefault(key, 0.0)
        if ptyp not in ("kickoff", "extra_point") and tpa != 1 and drv >= 0:
            drives.setdefault(key, set()).add(drv)
        if tpa == 1 or ptyp == "extra_point":
            if last_td == pos_:
                if ptyp == "extra_point" and xpr == "good":
                    pts[key] += 1.0
                elif tpa == 1 and tpr == "success":
                    pts[key] += 2.0
            continue
        if tdf == 1:
            if ptyp in ("pass", "run") and tdt == pos_:
                pts[key] += 6.0
                last_td = pos_
            else:
                last_td = None
        if ptyp == "field_goal" and fgr == "made":
            pts[key] += 3.0
    tg = pd.DataFrame({"starter": first}).join(tdb, how="outer").reset_index()
    tg["off_pts"] = [pts.get((a, b), 0.0) for a, b in zip(tg.game_id, tg.posteam)]
    tg["drives"] = [float(len(drives.get((a, b), ()))) for a, b in zip(tg.game_id, tg.posteam)]
    return qg, tg


# ============================================================ Kalman kernel
@njit(cache=True)
def kalman(date_i, blk, season, qb, dfn, ybar, n, llm, nQ, nD,
           v0q, m0q, tq, tsq, rq, v0d, td, tsd, rd, sig2, lam, rec):
    """Joint QB x defence Gaussian filter. Obs sorted by (date, block); a block is one
    (game, offence) side. Pre-game values for all obs of a block are taken before any
    of the block's updates. League mean folds only at date boundaries."""
    M = len(ybar)
    mq = np.zeros(nQ); vq = np.zeros(nQ); lq = -np.ones(nQ, np.int64)
    md = np.zeros(nD); vd = np.zeros(nD); ld = -np.ones(nD, np.int64)
    gq = -np.ones(nQ, np.int64)                   # last block a QB was drifted in
    gd = -np.ones(nD, np.int64)
    S = 0.0; W = 0.0; dS = 0.0; dW = 0.0; cur = -1
    out = np.zeros((M, 7)) if rec else np.zeros((1, 7))
    ll = 0.0
    j = 0
    while j < M:
        if date_i[j] != cur:
            S = lam * S + dS; W = lam * W + dW; dS = 0.0; dW = 0.0; cur = date_i[j]
        mu = S / W if W > 0 else 0.0
        k = j
        while k < M and blk[k] == blk[j]:
            k += 1
        # transitions (once per block per entity)
        for t in range(j, k):
            q = qb[t]; s = season[t]
            if lq[q] < 0:
                mq[q] = m0q; vq[q] = v0q; lq[q] = s
            elif s > lq[q]:
                gap = s - lq[q]
                mq[q] = mq[q] * (1.0 - rq) ** gap
                vq[q] += tsq * gap
                lq[q] = s
            if gq[q] != blk[t]:
                vq[q] += tq; gq[q] = blk[t]
            dd = dfn[t]
            if ld[dd] < 0:
                md[dd] = 0.0; vd[dd] = v0d; ld[dd] = s
            elif s > ld[dd]:
                gap = s - ld[dd]
                md[dd] = md[dd] * (1.0 - rd) ** gap
                vd[dd] += tsd * gap
                ld[dd] = s
            if gd[dd] != blk[t]:
                vd[dd] += td; gd[dd] = blk[t]
        # pre-game record + likelihood
        pmq = np.zeros(k - j); pvq = np.zeros(k - j); pmd = np.zeros(k - j); pvd = np.zeros(k - j)
        for t in range(j, k):
            pmq[t - j] = mq[qb[t]]; pvq[t - j] = vq[qb[t]]
            pmd[t - j] = md[dfn[t]]; pvd[t - j] = vd[dfn[t]]
            if rec:
                out[t, 0] = mq[qb[t]]; out[t, 1] = vq[qb[t]]
                out[t, 2] = md[dfn[t]]; out[t, 3] = vd[dfn[t]]; out[t, 4] = mu
            if n[t] > 0 and llm[t]:
                Sv = pvq[t - j] + pvd[t - j] + sig2 / n[t]
                r = ybar[t] - mu - pmq[t - j] - pmd[t - j]
                ll += -0.5 * (math.log(2.0 * math.pi * Sv) + r * r / Sv)
        # updates (sequential within the block)
        for t in range(j, k):
            if n[t] <= 0:
                continue
            q = qb[t]; dd = dfn[t]
            Sv = vq[q] + vd[dd] + sig2 / n[t]
            r = ybar[t] - mu - mq[q] - md[dd]
            gq_ = vq[q] / Sv; gd_ = vd[dd] / Sv
            mq[q] += gq_ * r; md[dd] += gd_ * r
            vq[q] -= gq_ * vq[q]; vd[dd] -= gd_ * vd[dd]
            dS += ybar[t] * n[t]; dW += n[t]
            if rec:
                out[t, 5] = mq[q]; out[t, 6] = vq[q]
        j = k
    return ll, out


PNAMES = ["v0q", "m0q", "tq", "tsq", "rq", "v0d", "td", "tsd", "rd"]


def unpack(th, scale):
    th = np.clip(np.asarray(th, float), -40.0, 12.0)
    v0q, m0q, tq, tsq, rq, v0d, td, tsd, rd = th
    s2 = scale ** 2
    return (math.exp(v0q) * s2, m0q * scale, math.exp(tq) * s2, math.exp(tsq) * s2,
            1 / (1 + math.exp(-rq)), math.exp(v0d) * s2, math.exp(td) * s2,
            math.exp(tsd) * s2, 1 / (1 + math.exp(-rd)))


def fit_component(A, ybar, n, sig2, fit_mask, scale, adj=True, tag=""):
    """type-II ML of the filter hyper-parameters on fit_mask obs.
    adj=False freezes the defence at 0 (v0d=td=tsd=0): the un-adjusted control arm."""
    th0 = np.array([0.0, -0.5, math.log(0.01), math.log(0.25), -2.0,
                    math.log(0.25), math.log(0.0025), math.log(0.25), 0.0])
    free = list(range(9)) if adj else [0, 1, 2, 3, 4]
    last = int(np.where(fit_mask)[0].max()) + 1          # the walk stops at the window's end
    A = {k: (v[:last] if isinstance(v, np.ndarray) else v) for k, v in A.items()}
    ybar, n, fit_mask = ybar[:last], n[:last], fit_mask[:last]

    def full(x):
        th = th0.copy(); th[free] = x
        p = list(unpack(th, scale))
        if not adj:
            p[5] = p[6] = p[7] = 0.0
        return p

    def nll(x):
        ll, _ = kalman(A["date_i"], A["blk"], A["season"], A["qb"], A["dfn"], ybar, n, fit_mask,
                       A["nQ"], A["nD"], *full(x), sig2, LAM, False)
        return -ll
    best = None
    for bump in (0.0, 1.0):
        x0 = th0[free] + bump * np.array([1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])[free]
        r = minimize(nll, x0, method="Nelder-Mead",
                     options={"maxiter": 6000, "maxfev": 6000, "xatol": 1e-4, "fatol": 1e-4})
        r = minimize(nll, r.x, method="Nelder-Mead",       # restart from the optimum
                     options={"maxiter": 6000, "maxfev": 6000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    p = full(best.x)
    log(f"  {tag:<10} nll {best.fun:.2f}  " + " ".join(f"{k}={v:.4g}" for k, v in zip(PNAMES, p)))
    return p, float(best.fun)


def run_filter(A, ybar, n, p, sig2):
    _, out = kalman(A["date_i"], A["blk"], A["season"], A["qb"], A["dfn"], ybar, n,
                    np.zeros(len(ybar), np.bool_), A["nQ"], A["nD"], *p, sig2, LAM, True)
    return out


def pooled_sig2(s, s2, nn, mask):
    """pooled within-QB-game per-play variance."""
    m = mask & (nn > 1)
    return float((s2[m] - s[m] ** 2 / nn[m]).sum() / (nn[m] - 1).sum())


# ================================== 3-state variant: QB x supporting cast x defence
@njit(cache=True)
def kalman3(date_i, blk, season, qb, cst, dfn, ybar, n, llm, nQ, nT,
            v0q, m0q, tq, tsq, rq, v0c, tc, tsc, rc, v0d, td, tsd, rd, sig2, lam, rec):
    """obs = mu + q_QB + c_CAST(offence team without its QB) + d_DEF + noise.
    The QB's rating travels with him; the cast rating stays with the team (the FIP
    move: strip the teammates out of the QB's number). Same timing rules as kalman()."""
    M = len(ybar)
    mq = np.zeros(nQ); vq = np.zeros(nQ); lq = -np.ones(nQ, np.int64); gq = -np.ones(nQ, np.int64)
    mc = np.zeros(nT); vc = np.zeros(nT); lc = -np.ones(nT, np.int64); gc = -np.ones(nT, np.int64)
    md = np.zeros(nT); vd = np.zeros(nT); ld = -np.ones(nT, np.int64); gd = -np.ones(nT, np.int64)
    S = 0.0; W = 0.0; dS = 0.0; dW = 0.0; cur = -1
    out = np.zeros((M, 9)) if rec else np.zeros((1, 9))
    ll = 0.0
    j = 0
    while j < M:
        if date_i[j] != cur:
            S = lam * S + dS; W = lam * W + dW; dS = 0.0; dW = 0.0; cur = date_i[j]
        mu = S / W if W > 0 else 0.0
        k = j
        while k < M and blk[k] == blk[j]:
            k += 1
        for t in range(j, k):
            s = season[t]
            q = qb[t]
            if lq[q] < 0:
                mq[q] = m0q; vq[q] = v0q; lq[q] = s
            elif s > lq[q]:
                mq[q] = mq[q] * (1.0 - rq) ** (s - lq[q]); vq[q] += tsq * (s - lq[q]); lq[q] = s
            if gq[q] != blk[t]:
                vq[q] += tq; gq[q] = blk[t]
            c = cst[t]
            if lc[c] < 0:
                mc[c] = 0.0; vc[c] = v0c; lc[c] = s
            elif s > lc[c]:
                mc[c] = mc[c] * (1.0 - rc) ** (s - lc[c]); vc[c] += tsc * (s - lc[c]); lc[c] = s
            if gc[c] != blk[t]:
                vc[c] += tc; gc[c] = blk[t]
            dd = dfn[t]
            if ld[dd] < 0:
                md[dd] = 0.0; vd[dd] = v0d; ld[dd] = s
            elif s > ld[dd]:
                md[dd] = md[dd] * (1.0 - rd) ** (s - ld[dd]); vd[dd] += tsd * (s - ld[dd]); ld[dd] = s
            if gd[dd] != blk[t]:
                vd[dd] += td; gd[dd] = blk[t]
        for t in range(j, k):
            q = qb[t]; c = cst[t]; dd = dfn[t]
            if rec:
                out[t, 0] = mq[q]; out[t, 1] = vq[q]; out[t, 2] = md[dd]; out[t, 3] = vd[dd]
                out[t, 4] = mu; out[t, 7] = mc[c]; out[t, 8] = vc[c]
            if n[t] > 0 and llm[t]:
                Sv = vq[q] + vc[c] + vd[dd] + sig2 / n[t]
                r = ybar[t] - mu - mq[q] - mc[c] - md[dd]
                ll += -0.5 * (math.log(2.0 * math.pi * Sv) + r * r / Sv)
        # NB: pre-game values of a block are identical for all its obs (one QB each; the
        # shared cast/defence are only updated below), so the likelihood above is pre-game.
        for t in range(j, k):
            if n[t] <= 0:
                continue
            q = qb[t]; c = cst[t]; dd = dfn[t]
            Sv = vq[q] + vc[c] + vd[dd] + sig2 / n[t]
            r = ybar[t] - mu - mq[q] - mc[c] - md[dd]
            kq = vq[q] / Sv; kc = vc[c] / Sv; kd = vd[dd] / Sv
            mq[q] += kq * r; mc[c] += kc * r; md[dd] += kd * r
            vq[q] -= kq * vq[q]; vc[c] -= kc * vc[c]; vd[dd] -= kd * vd[dd]
            dS += ybar[t] * n[t]; dW += n[t]
            if rec:
                out[t, 5] = mq[q]; out[t, 6] = vq[q]
        j = k
    return ll, out


PNAMES3 = ["v0q", "m0q", "tq", "tsq", "rq", "v0c", "tc", "tsc", "rc", "v0d", "td", "tsd", "rd"]


def unpack3(th, scale):
    th = np.clip(np.asarray(th, float), -40.0, 12.0)
    s2 = scale ** 2
    e = math.exp
    lg = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    return (e(th[0]) * s2, th[1] * scale, e(th[2]) * s2, e(th[3]) * s2, lg(th[4]),
            e(th[5]) * s2, e(th[6]) * s2, e(th[7]) * s2, lg(th[8]),
            e(th[9]) * s2, e(th[10]) * s2, e(th[11]) * s2, lg(th[12]))


def fit_component3(A, ybar, n, sig2, fit_mask, scale, p2, tag=""):
    """type-II ML, warm window only; started from the fitted 2-state solution with a
    small cast share (and a second start with a large one)."""
    last = int(np.where(fit_mask)[0].max()) + 1
    B = {k: (v[:last] if isinstance(v, np.ndarray) else v) for k, v in A.items()}
    yb, nn, fm = ybar[:last], n[:last], fit_mask[:last]
    s2 = scale ** 2
    lgt = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))  # noqa: E731
    lv = lambda v: math.log(max(v, 1e-12) / s2)  # noqa: E731
    base = [lv(p2[0]), p2[1] / scale, lv(p2[2]), lv(p2[3]), lgt(p2[4]),
            lv(0.1 * p2[0]), lv(0.001 * s2), lv(0.05 * s2), 0.0,
            lv(p2[5]), lv(p2[6]), lv(p2[7]), lgt(p2[8])]

    def nll(th):
        ll, _ = kalman3(B["date_i"], B["blk"], B["season"], B["qb"], B["cst"], B["dfn"], yb, nn, fm,
                        B["nQ"], B["nT"], *unpack3(th, scale), sig2, LAM, False)
        return -ll
    best = None
    for cshare in (0.1, 0.5):
        x0 = np.array(base, float)
        x0[0] = lv((1 - cshare) * p2[0]); x0[5] = lv(cshare * p2[0])
        r = minimize(nll, x0, method="Nelder-Mead",
                     options={"maxiter": 9000, "maxfev": 9000, "xatol": 1e-4, "fatol": 1e-4})
        r = minimize(nll, r.x, method="Nelder-Mead",
                     options={"maxiter": 9000, "maxfev": 9000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    p = list(unpack3(best.x, scale))
    log(f"  {tag:<10} nll {best.fun:.2f}  " + " ".join(f"{k}={v:.4g}" for k, v in zip(PNAMES3, p)))
    return p, float(best.fun)


# ============================================================ shipped QbElo
def shipped_qbelo(qg):
    """pre-game shipped QbElo value (EPA/db, beta=1) for every QB-game obs, two walks:
    listed-QB keyed (the cached X14 col 1) and starter keyed (served since ade264f2)."""
    sys.path.insert(0, "phase0")
    import nfl_qb_elo as QE
    qbw = {k: v for k, v in QE.load_qb_weeks().items() if k[1] < TEST_ERA}
    pedP = json.load(open("data/nfl_qb_pedigree.json"))["params"]
    lg_qb = json.load(open("data/nfl_qb_elo.json"))["qb_elo"]["lg_rate"]
    SP = json.load(open("data/nfl_qb_replacement.json"))["params"]
    picks = {}
    for r in csv.DictReader(open("data/nfl_draft_picks.csv", encoding="utf-8")):
        if r["position"] == "QB" and r["gsis_id"] and r["pick"]:
            picks[r["gsis_id"]] = int(r["pick"])
    rep_map = {q: lg_qb + pedP["delta"] * (1.0 - pedP["c"] * math.exp(-(pk - 1) / pedP["scale"]))
               for q, pk in picks.items()}
    by_gid = qg.groupby("game_id").qb.apply(list).to_dict()
    res = {}
    for mode in ("listed", "starter"):
        games = QE.load_games_qb(starter=False)
        raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
        raw.sort(key=lambda r: (r["gameday"],))
        for g, r in zip(games, raw):
            g["gid"] = r["game_id"]
        keep = [k for k, g in enumerate(games) if g["season"] < TEST_ERA]
        assert keep == list(range(len(keep)))            # season-monotone prefix
        games = games[:len(keep)]; raw = raw[:len(keep)]
        if mode == "starter":                             # nfl_qb_elo.apply_starters, pre-2016 only
            first = {k: v for k, v in QE.first_passers().items() if int(k[0][:4]) < TEST_ERA}
            nchg = QE.apply_starters(games, raw, qbw, first)
            log(f"starter walk: {nchg} pre-2016 sides use the first passer")
        m = QE.QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=pedP["delta"],
                     rep_map=rep_map, decay=SP["decay"], prior_db=SP["prior_db"],
                     season_decay=SP["season_decay"])
        val, side, prev = {}, {}, None
        for g in games:
            if prev is not None and g["season"] != prev:
                m.new_season()
            prev = g["season"]
            for q in by_gid.get(g["gid"], []):
                val[(g["gid"], q)] = m.qb_adj(q)
            side[g["gid"]] = (m.qb_adj(g["home_qb"]), m.qb_adj(g["away_qb"]), g["home_qb"], g["away_qb"])
            m.predict(g)
            m.update(g, 0.5, qbw)
        res[mode] = (val, side)
    return res


# ===================================================================== main
def main():
    df = load_plays()
    log(f"plays {len(df):,} seasons {int(df.season.min())}-{int(df.season.max())}")
    qg, tg = aggregate(df)
    gm = df.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                   game_date=("game_date", "first"), stype=("season_type", "first"),
                                   home=("home_team", "first"), away=("away_team", "first"))
    qg = qg.merge(gm, left_on="game_id", right_index=True)
    qg["side"] = np.where(qg.posteam == qg.home, "home", "away")
    tg = tg.merge(gm, left_on="game_id", right_index=True)
    assert int(qg.season.max()) < TEST_ERA and int(tg.season.max()) < TEST_ERA
    qg = qg.sort_values(["game_date", "game_id", "posteam", "first_play"], kind="mergesort").reset_index(drop=True)
    log(f"QB-game obs {len(qg):,}; team-games {len(tg):,}")

    # arrays for the filter
    qids = {q: i for i, q in enumerate(pd.unique(qg.qb))}
    tids = {t: i for i, t in enumerate(sorted(set(qg.defteam) | set(qg.posteam)))}
    dates = {d_: i for i, d_ in enumerate(sorted(set(qg.game_date)))}
    blk_key = qg.game_id + "|" + qg.posteam
    blk = pd.factorize(blk_key)[0]
    assert np.all(np.diff(blk) >= 0)
    A = {"date_i": qg.game_date.map(dates).to_numpy(np.int64), "blk": blk.astype(np.int64),
         "season": qg.season.to_numpy(np.int64), "qb": qg.qb.map(qids).to_numpy(np.int64),
         "dfn": qg.defteam.map(tids).to_numpy(np.int64), "nQ": len(qids), "nD": len(tids),
         "cst": qg.posteam.map(tids).to_numpy(np.int64), "nT": len(tids)}
    seas = A["season"]
    warm = (seas >= 2000) & (seas <= 2005)                # 1999 = burn-in
    cwin = (seas >= 2007) & (seas <= 2008)                # CPOE: 2006 = burn-in

    ndb = qg.n_db.to_numpy(float); natt = qg.n_att.to_numpy(float); ncp = qg.n_cp.to_numpy(float)
    nall = ndb + qg.n_rush.to_numpy(float)
    comps = {
        # name: (ybar, n, sig2, fit window, scale)
        "epa": (qg.s_epa / ndb, ndb, pooled_sig2(qg.s_epa.values, qg.s_epa2.values, ndb, warm), warm, 0.1),
        "succ": (qg.s_succ / ndb, ndb, None, warm, 0.03),
        "sack": (qg.n_sack / ndb, ndb, None, warm, 0.015),
        "int": (np.where(natt > 0, qg.n_int / np.maximum(natt, 1), 0.0), natt, None, warm, 0.007),
        "cpoe": (np.where(ncp > 0, qg.s_cpoe / np.maximum(ncp, 1), 0.0), ncp,
                 pooled_sig2(qg.s_cpoe.values, qg.s_cpoe2.values, ncp, cwin), cwin, 0.03),
        "epax": ((qg.s_epa + qg.s_rush) / nall, nall, None, warm, 0.1),
    }
    # binary components: sig2 = p(1-p) at the warm-window rate
    for c in ("succ", "sack", "int"):
        y_, n_, _, w_, sc = comps[c]
        p_ = float((np.asarray(y_)[w_] * n_[w_]).sum() / n_[w_].sum())
        comps[c] = (y_, n_, p_ * (1 - p_), w_, sc)
    # epax: pooled variance approximated by the dropback epa variance (runs are lower-variance)
    y_, n_, _, w_, sc = comps["epax"]
    comps["epax"] = (y_, n_, comps["epa"][2], w_, sc)

    hyper = {"lam_per_date": LAM, "fit_window": {"default": "2000-2005 (1999 burn-in)",
                                                  "cpoe": "2007-2008 (2006 burn-in)"}}
    OUTC = {}
    for c, (y_, n_, sig2, w_, sc) in comps.items():
        y_ = np.nan_to_num(np.asarray(y_, float)); n_ = np.asarray(n_, float)
        p, nll = fit_component(A, y_, n_, sig2, w_, sc, adj=True, tag=c)
        hyper[c] = {"sig2": sig2, "nll": nll, **dict(zip(PNAMES, p))}
        OUTC[c] = run_filter(A, y_, n_, p, sig2)
        if c == "epa":           # un-adjusted control arm (defence frozen at 0)
            p0, nll0 = fit_component(A, y_, n_, sig2, w_, sc, adj=False, tag="epa_raw")
            hyper["epa_raw"] = {"sig2": sig2, "nll": nll0, **dict(zip(PNAMES, p0))}
            OUTC["epa_raw"] = run_filter(A, y_, n_, p0, sig2)

    # ---------------- 3-state EPA: QB x supporting cast x defence
    y_, n_, sig2, w_, sc = comps["epa"]
    y_ = np.nan_to_num(np.asarray(y_, float)); n_ = np.asarray(n_, float)
    pe = [hyper["epa"][k] for k in PNAMES]
    p3, nll3 = fit_component3(A, y_, n_, sig2, w_, sc, pe, tag="epa3")
    hyper["epa3"] = {"sig2": sig2, "nll": nll3, **dict(zip(PNAMES3, p3))}
    _, O3 = kalman3(A["date_i"], A["blk"], A["season"], A["qb"], A["cst"], A["dfn"], y_, n_,
                    np.zeros(len(y_), np.bool_), A["nQ"], A["nT"], *p3, sig2, LAM, True)

    # ---------------- walk-forward composite: next-game EPA/db on pre-game components
    target = np.asarray(comps["epa"][0], float) - OUTC["epa"][:, 4]     # EPA/db above league
    Q_ = {c: OUTC[c][:, 0] for c in ("epa", "succ", "sack", "cpoe")}
    D_ = {c: OUTC[c][:, 2] for c in ("epa", "succ", "sack", "cpoe")}
    comp = np.full(len(qg), np.nan)
    wts = {}
    for s in range(2000, TEST_ERA):
        use_c = s >= 2008
        cs = ["epa", "succ", "sack"] + (["cpoe"] if use_c else [])
        lo = 2007 if use_c else 2000
        tr = (seas >= lo) & (seas < s) & (ndb > 0)
        if s == 2000:
            tr = (seas == 1999) & (ndb > 0)
        te = seas == s
        X = np.column_stack([Q_[c] for c in cs] + [D_[c] for c in cs] + [np.ones(len(qg))])
        W = ndb[tr]
        beta = np.linalg.solve((X[tr] * W[:, None]).T @ X[tr] + 1e-9 * np.eye(X.shape[1]),
                               (X[tr] * W[:, None]).T @ target[tr])
        k = len(cs)
        comp[te] = X[te][:, :k] @ beta[:k]
        wts[s] = {"components": cs, "qb_coef": dict(zip(cs, np.round(beta[:k], 4).tolist())),
                  "def_coef": dict(zip(cs, np.round(beta[k:2 * k], 4).tolist())),
                  "train": f"{lo if s > 2000 else 1999}-{s-1}", "n_train": int(tr.sum())}
    hyper["composite_weights_by_season"] = wts
    # composite3: same recipe, EPA component from the cast-stripped 3-state filter and
    # the team's cast rating as a control (so the QB weights are net of teammates)
    Q_["epa"] = O3[:, 0]; D_["epa"] = O3[:, 2]
    comp3 = np.full(len(qg), np.nan); wts3 = {}
    for s in range(2000, TEST_ERA):
        use_c = s >= 2008
        cs = ["epa", "succ", "sack"] + (["cpoe"] if use_c else [])
        lo = 2007 if use_c else 2000
        tr = ((seas >= lo) & (seas < s) & (ndb > 0)) if s > 2000 else ((seas == 1999) & (ndb > 0))
        te = seas == s
        X = np.column_stack([Q_[c] for c in cs] + [D_[c] for c in cs] + [O3[:, 7], np.ones(len(qg))])
        W = ndb[tr]
        beta = np.linalg.solve((X[tr] * W[:, None]).T @ X[tr] + 1e-9 * np.eye(X.shape[1]),
                               (X[tr] * W[:, None]).T @ target[tr])
        k = len(cs)
        comp3[te] = X[te][:, :k] @ beta[:k]
        wts3[s] = {"components": cs, "qb_coef": dict(zip(cs, np.round(beta[:k], 4).tolist())),
                   "cast_coef": round(float(beta[2 * k]), 4)}
    hyper["composite3_weights_by_season"] = wts3
    log("composite3 weights 2015 fold:", wts3[2015])
    log("composite weights 2015 fold:", wts[2015]["qb_coef"])

    # ---------------- shipped QbElo on the same obs
    ship = shipped_qbelo(qg)
    qbelo_l = np.array([ship["listed"][0].get((g, q), np.nan) for g, q in zip(qg.game_id, qg.qb)])
    qbelo_s = np.array([ship["starter"][0].get((g, q), np.nan) for g, q in zip(qg.game_id, qg.qb)])
    log(f"QbElo join: listed {np.isfinite(qbelo_l).mean():.4f} starter {np.isfinite(qbelo_s).mean():.4f}")
    # sanity: listed-walk side values reproduce the cached X14 col-1 halves
    D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
    side_l = ship["listed"][1]
    dif = [abs(side_l[g["gid"]][0] - qh) + abs(side_l[g["gid"]][1] - qa)
           for g, qh, qa in zip(D["games"], D["qh"], D["qa"]) if g["gid"] in side_l]
    log(f"QbElo reproduction vs cached qh/qa: n={len(dif)} max abs diff {max(dif):.2e}")
    assert len(dif) == len(D["games"]) and max(dif) < 1e-9
    hyper["qbelo_repro_max_abs_diff"] = max(dif)
    side_s = ship["starter"][1]
    json.dump({g: [a, b, c, d_] for g, (a, b, c, d_) in side_s.items()},
              open("data/pv_nfl_passing_qbelo_starter_sides.json", "w"))

    names = {}
    with open("data/pv_nfl_players_seen.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            names[r["player_id"]] = r["name"]
    st = tg.set_index(["game_id", "posteam"]).starter
    is_st = np.array([st.get((g, t)) == q for g, t, q in zip(qg.game_id, qg.posteam, qg.qb)])

    vals = pd.DataFrame({
        "game_id": qg.game_id, "season": qg.season, "week": qg.week, "game_date": qg.game_date,
        "season_type": qg.stype, "team": qg.posteam, "opp": qg.defteam, "side": qg.side,
        "qb_id": qg.qb, "qb_name": qg.qb.map(names), "is_starter": is_st.astype(int),
        "passing_composite": comp,
        "epa_q": OUTC["epa"][:, 0], "epa_q_var": OUTC["epa"][:, 1],
        "epa_raw_q": OUTC["epa_raw"][:, 0],
        "succ_q": OUTC["succ"][:, 0], "sack_q": OUTC["sack"][:, 0], "int_q": OUTC["int"][:, 0],
        "cpoe_q": OUTC["cpoe"][:, 0], "epax_q": OUTC["epax"][:, 0],
        "passing_composite3": comp3, "epa3_q": O3[:, 0], "epa3_q_var": O3[:, 1],
        "epa3_cast": O3[:, 7], "epa3_cast_var": O3[:, 8], "epa3_opp_def": O3[:, 2],
        "opp_def_epa": OUTC["epa"][:, 2], "opp_def_epa_var": OUTC["epa"][:, 3],
        "opp_def_succ": OUTC["succ"][:, 2], "opp_def_sack": OUTC["sack"][:, 2],
        "opp_def_cpoe": OUTC["cpoe"][:, 2],
        "lg_epa": OUTC["epa"][:, 4], "lg_cpoe": OUTC["cpoe"][:, 4], "lg_sack": OUTC["sack"][:, 4],
        "lg_succ": OUTC["succ"][:, 4],
        "qbelo_listed": qbelo_l, "qbelo_starter": qbelo_s,
    })
    vals.to_csv("data/pv_nfl_passing_values.csv", index=False, float_format="%.6g")
    outc = pd.DataFrame({
        "game_id": qg.game_id, "team": qg.posteam, "qb_id": qg.qb,
        "n_db": qg.n_db, "s_epa": qg.s_epa, "s_succ": qg.s_succ, "n_sack": qg.n_sack,
        "n_att": qg.n_att, "n_int": qg.n_int, "n_cp": qg.n_cp, "s_cpoe": qg.s_cpoe,
        "n_scr": qg.n_scr, "n_rush": qg.n_rush, "s_rush": qg.s_rush,
        # post-game state (after this game's update): used ONLY as 'end of season'
        # ratings for next-season validation
        "post_epa_q": OUTC["epa"][:, 5], "post_epa_raw_q": OUTC["epa_raw"][:, 5],
        "post_succ_q": OUTC["succ"][:, 5], "post_sack_q": OUTC["sack"][:, 5],
        "post_cpoe_q": OUTC["cpoe"][:, 5], "post_epa3_q": O3[:, 5],
    })
    outc.to_csv("data/pv_nfl_passing_outcomes.csv", index=False, float_format="%.6g")
    tg.to_csv("data/pv_nfl_passing_team.csv", index=False)
    json.dump(hyper, open("data/pv_nfl_passing_hyper.json", "w"), indent=1)
    log("wrote data/pv_nfl_passing_values.csv, _outcomes.csv, _team.csv, _hyper.json")


if __name__ == "__main__":
    main()
