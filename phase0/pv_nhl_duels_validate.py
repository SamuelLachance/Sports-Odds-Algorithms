"""pv_nhl_duels_validate -- player-level (then unit/game-level) validation of the
DISCRETE_DUELS component on DEV only.

Inputs : data/pv_nhl_discrete_duels.csv (walk-forward per-player-game values,
         phase0/pv_nhl_duels.py build), data/pv_nhl_discrete_duels_fo.parquet,
         data/pv_nhl_discrete_duels_params.json,
         data/nhl_shifts.csv (DEV rows only -> on-ice goals),
         data/bt_nhl_rapmel_values.csv (walk-forward xG-RAPM = the CURRENT player
         rating: on-ice 5v5 net xG/60, EB-shrunk, pre-season / Jan-1 cutoffs),
         data/bt_nhl_rapmel_feats.csv (walk-forward RAPM lineup power per game).
Output : data/pv_nhl_discrete_duels_validation.json (+ _onice.csv cache)

Sections
  R  reliability: split-half (odd/even gid within season) and year-over-year
     correlations of RAW per-player rates, with n; arena-adjusted vs raw TK/GV.
  F  faceoff-level predictive LL: EKF ratings vs context-only, DEV-A (tuning)
     and DEV-B (never used for any choice).
  P  predictive validity, player-season: PRE-SEASON walk-forward values ->
     that season's realised (a) duel goals, (b) on-ice goal differential /60
     (all situations and 5v5), (c) points/60; vs naive last-season rate and vs
     the xG-RAPM; incremental R^2 with player-clustered bootstrap CIs.
  U  unit (team-game): tonight's dressed lineup sum of duel goals ->
     realised team special-teams goal differential, faceoff share, and total goal
     differential, beside the shipped features and the RAPM lineup power.
  G  game-level DEV LOSO log-loss screen vs the shipped blend (harness sanity
     S1/S2 reproduced) -- informative only; NO TEST look is taken here.
  V  face validity lists.

DEV only: every input is asserted gid < 2018000000 / season <= 2017-18.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_nhl_duels as D  # noqa: E402
from pv_nhl_io import DEV_MAX_GID, load_events  # noqa: E402

ROOT = D.ROOT
OUTJ = D.OUT + "_validation.json"
ONICE = D.OUT + "_onice.csv"
EVAL = D.DEV_SEASONS[1:]
RNG = np.random.default_rng(20260924)


def dpath(*a):
    return os.path.join(ROOT, "data", *a)


# ================================================================== loading
def load_pg():
    pg = pd.read_csv(D.OUT + ".csv")
    assert pg.gid.max() < DEV_MAX_GID and pg.season.max() <= 20172018
    return pg


def onice_goals(pg):
    if os.path.exists(ONICE):
        oi = pd.read_csv(ONICE)
        assert oi.gid.max() < DEV_MAX_GID
        return oi
    t0 = time.time()
    ev = load_events(columns=["period", "ptype", "per_sec", "ev", "team", "h_sk", "a_sk",
                              "h_g", "a_g"])
    gl = ev[(ev.ev == "goal") & (ev.ptype != "SO")].copy()
    gl["is5"] = ((gl.h_sk == 5) & (gl.a_sk == 5) & (gl.h_g == 1) & (gl.a_g == 1)).astype(int)
    gl = gl[["gid", "period", "per_sec", "team", "is5"]].rename(columns={"team": "gteam"})
    parts = []
    for ch in pd.read_csv(dpath("nhl_shifts.csv"), chunksize=2_000_000,
                          usecols=["game_id", "player_id", "period", "start_s", "end_s"]):
        ch = ch[ch.game_id < DEV_MAX_GID]
        if len(ch):
            parts.append(ch)
    sh = pd.concat(parts).rename(columns={"game_id": "gid", "player_id": "pid"})
    team = pg[["gid", "pid", "team"]]
    sh = sh.merge(team, on=["gid", "pid"], how="inner")      # skaters only
    out = []
    for s in sorted(pg.season.unique()):
        gids = pg.gid[pg.season == s].unique()
        a = sh[sh.gid.isin(gids)]
        b = gl[gl.gid.isin(gids)]
        m = a.merge(b, on=["gid", "period"])
        m = m[(m.start_s < m.per_sec) & (m.per_sec <= m.end_s)]
        m["gf"] = (m.team == m.gteam).astype(int)
        m["ga"] = 1 - m.gf
        m["gf5"] = m.gf * m.is5
        m["ga5"] = m.ga * m.is5
        # a player can have two overlapping shift rows at the goal second: dedupe
        m = m.drop_duplicates(["gid", "pid", "period", "per_sec", "gteam"])
        out.append(m.groupby(["gid", "pid"])[["gf", "ga", "gf5", "ga5"]].sum().reset_index())
    oi = pd.concat(out)
    t5 = []
    for s in D.DEV_SEASONS:
        f = dpath(f"bt_nhl_rapmel_toi5_{s}.csv")
        t5.append(pd.read_csv(f)[["gid", "pid", "sec5"]])
    t5 = pd.concat(t5)
    oi = pg[["gid", "pid"]].merge(oi, on=["gid", "pid"], how="left").merge(
        t5, on=["gid", "pid"], how="left").fillna(0)
    assert oi.gid.max() < DEV_MAX_GID
    oi.to_csv(ONICE, index=False)
    print(f"on-ice built {time.time()-t0:.0f}s", flush=True)
    return oi


def rapm_preseason(pg):
    v = pd.read_csv(dpath("bt_nhl_rapmel_values.csv"))
    first = pg.groupby("season").date.min()
    rows = []
    cuts = sorted(v.cutoff_date.unique())
    for s, d0 in first.items():
        c = [x for x in cuts if x < d0]
        if not c:
            continue
        cut = c[-1]
        vv = v[v.cutoff_date == cut][["pid", "v"]].assign(season=s, rapm_cut=cut)
        rows.append(vv)
    return pd.concat(rows)


# ================================================================ helpers
def wcorr(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cx, cy = x - mx, y - my
    return float(np.sum(w * cx * cy) / np.sqrt(np.sum(w * cx * cx) * np.sum(w * cy * cy)))


def wls_r2(y, Xs, w):
    X = np.column_stack([np.ones(len(y))] + Xs)
    W = np.sqrt(w)
    b = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
    r = y - X @ b
    my = np.average(y, weights=w)
    return float(1 - np.sum(w * r * r) / np.sum(w * (y - my) ** 2)), b


def sb(r):
    return float(2 * r / (1 + r)) if r > -1 else float("nan")


# ============================================================ R reliability
def season_rates(pg, oi=None):
    agg = {"toi_s": "sum", "fo_n": "sum", "fo_w": "sum", "fo_real": "sum",
           "pen_take_n": "sum", "pen_draw_n": "sum", "pen_take_g": "sum",
           "pen_draw_g": "sum", "tk_n": "sum", "gv_n": "sum", "tk_adj": "sum",
           "gv_adj": "sum", "tk_g": "sum", "gv_g": "sum", "goals": "sum", "assists": "sum"}
    return agg


def rates_from(a):
    h = a.toi_s / 3600.0
    r = pd.DataFrame(index=a.index)
    r["toi_min"] = a.toi_s / 60.0
    r["fo_n"] = a.fo_n
    r["fo_pct"] = a.fo_w / a.fo_n.replace(0, np.nan)
    r["fo_g60"] = a.fo_real / h
    r["pen_draw60"] = a.pen_draw_n / h
    r["pen_take60"] = a.pen_take_n / h
    r["pen_net60"] = (a.pen_draw_n - a.pen_take_n) / h
    r["pen_g60"] = (a.pen_draw_g - a.pen_take_g) / h
    r["tk60_raw"] = a.tk_n / h
    r["gv60_raw"] = a.gv_n / h
    r["tk60_adj"] = a.tk_adj / h
    r["gv60_adj"] = a.gv_adj / h
    r["tkgv60_raw"] = (a.tk_n - a.gv_n) / h
    r["tkgv60_adj"] = (a.tk_adj - a.gv_adj) / h
    r["tkgv_g60"] = (a.tk_g + a.gv_g) / h
    r["dd_g60"] = r.fo_g60 + r.pen_g60
    r["dd_all_g60"] = r.dd_g60 + r.tkgv_g60
    r["pts60"] = (a.goals + a.assists) / h
    return r


REL_STATS = ["fo_pct", "fo_g60", "pen_draw60", "pen_take60", "pen_net60", "pen_g60",
             "tk60_raw", "tk60_adj", "gv60_raw", "gv60_adj", "tkgv60_raw", "tkgv60_adj",
             "tkgv_g60", "dd_g60", "dd_all_g60", "pts60"]


def reliability(pg):
    agg = season_rates(pg)
    pg = pg.assign(half=(pg.gid % 2))
    a = pg.groupby(["season", "pid", "half"]).agg(agg)
    r = rates_from(a).reset_index()
    pos = pg.groupby("pid").pos.agg(lambda x: x.mode().iloc[0])
    out = {"split_half": {}, "yoy": {}, "min_toi_min_per_half": 250,
           "min_toi_min_per_season": 500, "min_fo_per_half": 100,
           "min_fo_per_season": 200}
    A = r[r.half == 0].set_index(["season", "pid"])
    B = r[r.half == 1].set_index(["season", "pid"])
    J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    okt = (J.toi_min_a >= 250) & (J.toi_min_b >= 250)
    for st in REL_STATS:
        ok = okt.copy()
        if st.startswith("fo"):
            ok &= (J.fo_n_a >= 100) & (J.fo_n_b >= 100)
        x, y = J.loc[ok, st + "_a"], J.loc[ok, st + "_b"]
        rr = wcorr(x, y)
        out["split_half"][st] = {"r_half": round(rr, 3), "r_full_SB": round(sb(rr), 3),
                                 "n": int(ok.sum())}
        for grp in ("F", "D"):
            if st.startswith("fo") and grp == "D":
                continue
            pid_ok = J.index.get_level_values("pid").map(pos)
            gm = ok & ((pid_ok == "D") if grp == "D" else (pid_ok != "D"))
            if gm.sum() > 30:
                out["split_half"][st][f"r_half_{grp}"] = round(wcorr(J.loc[gm, st + "_a"],
                                                                     J.loc[gm, st + "_b"]), 3)
    # year over year
    s_all = pg.groupby(["season", "pid"]).agg(agg)
    R = rates_from(s_all).reset_index()
    seas = D.DEV_SEASONS
    nxt = {seas[i]: seas[i + 1] for i in range(len(seas) - 1)}
    R1 = R.assign(season=R.season.map(nxt)).dropna(subset=["season"])
    R1["season"] = R1.season.astype(int)
    Y = R1.set_index(["season", "pid"]).join(R.set_index(["season", "pid"]),
                                             lsuffix="_prev", rsuffix="_cur", how="inner")
    okt = (Y.toi_min_prev >= 500) & (Y.toi_min_cur >= 500)
    team = pg.groupby(["season", "pid"]).team.agg(lambda x: x.mode().iloc[0])
    moved = []
    for (s, p) in Y.index:
        prev = [x for x, y_ in nxt.items() if y_ == s][0]
        moved.append(team.get((prev, p)) != team.get((s, p)))
    Y["moved"] = moved
    for st in REL_STATS:
        ok = okt.copy()
        if st.startswith("fo"):
            ok &= (Y.fo_n_prev >= 200) & (Y.fo_n_cur >= 200)
        rr = wcorr(Y.loc[ok, st + "_prev"], Y.loc[ok, st + "_cur"])
        mv = ok & Y.moved
        out["yoy"][st] = {"r": round(rr, 3), "n": int(ok.sum())}
        ppos = Y.index.get_level_values("pid").map(pos)
        for grp in ("F", "D"):
            if st.startswith("fo") and grp == "D":
                continue
            gm = ok & ((ppos == "D") if grp == "D" else (ppos != "D"))
            out["yoy"][st][f"r_{grp}"] = round(wcorr(Y.loc[gm, st + "_prev"],
                                                     Y.loc[gm, st + "_cur"]), 3)
            out["yoy"][st][f"n_{grp}"] = int(gm.sum())
            gmv = gm & Y.moved
            if gmv.sum() > 30:
                out["yoy"][st][f"r_team_changers_{grp}"] = round(
                    wcorr(Y.loc[gmv, st + "_prev"], Y.loc[gmv, st + "_cur"]), 3)
        if mv.sum() > 30:
            out["yoy"][st]["r_team_changers"] = round(
                wcorr(Y.loc[mv, st + "_prev"], Y.loc[mv, st + "_cur"]), 3)
            out["yoy"][st]["n_team_changers"] = int(mv.sum())
    return out


# ============================================================ F faceoff LL
def faceoff_ll():
    fo = pd.read_parquet(D.OUT + "_fo.parquet")
    assert fo.gid.max() < DEV_MAX_GID
    prm = json.load(open(D.OUT + "_params.json"))
    ll = lambda m: float(-np.log(np.clip(fo.p_winner[m], 1e-12, 1)).mean())
    a = fo.season.isin(D.TUNE_SEASONS)
    b = fo.season.isin(D.HOLD_SEASONS)
    res = {"rating_ll_devA": round(ll(a), 5), "rating_ll_devB": round(ll(b), 5),
           "context_only_ll_devA": round(prm["fo_context_only"]["tune_ll"], 5),
           "context_only_ll_devB": round(prm["fo_context_only"]["hold_ll"], 5),
           "coin_ll": round(float(np.log(2)), 5), "n_devA": int(a.sum()), "n_devB": int(b.sum())}
    res["gain_devB_vs_context"] = round(res["context_only_ll_devB"] - res["rating_ll_devB"], 5)
    # calibration on DEV-B: orient each faceoff to the home-side taker
    bb = fo[b]
    ph = np.where(bb.is_home_win == 1, bb.p_winner, 1 - bb.p_winner)
    yh = (bb.is_home_win == 1).astype(int).values
    bins = np.quantile(ph, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, ph, "right") - 1, 0, 9)
    res["calibration_devB_deciles"] = [
        {"p_mean": round(float(ph[idx == k].mean()), 4), "y_mean": round(float(yh[idx == k].mean()), 4)}
        for k in range(10)]
    return res


# ======================================================= P player-season
PRED_COLS = ["fo_g60", "pen_g60", "tkgv_g60", "dd_g60", "dd_all_g60", "fo_rating",
             "pen_draw_n60", "pen_take_n60", "tk_adj60", "gv_adj60"]


def player_season_table(pg, oi):
    pg = pg.merge(oi, on=["gid", "pid"], how="left")
    pg = pg.sort_values(["pid", "date", "gid"])
    first = pg.groupby(["season", "pid"]).head(1).set_index(["season", "pid"])
    agg = season_rates(pg)
    agg.update({"gf": "sum", "ga": "sum", "gf5": "sum", "ga5": "sum", "sec5": "sum"})
    S = pg.groupby(["season", "pid"]).agg(agg)
    R = rates_from(S)
    R["onice_gd60"] = (S.gf - S.ga) / (S.toi_s / 3600)
    R["onice_gd60_5v5"] = (S.gf5 - S.ga5) / (S.sec5.replace(0, np.nan) / 3600)
    R["toi5_min"] = S.sec5 / 60
    T = R.join(first[PRED_COLS].add_prefix("wf_")).join(
        first[["pos", "n_prev_games"]])
    # naive: last season's raw realised rates
    seas = D.DEV_SEASONS
    nxt = {seas[i]: seas[i + 1] for i in range(len(seas) - 1)}
    L = R[["dd_g60", "dd_all_g60", "pen_g60", "fo_g60", "tkgv_g60", "onice_gd60", "toi_min"]]
    L = L.reset_index()
    L["season"] = L.season.map(nxt)
    L = L.dropna(subset=["season"])
    L["season"] = L.season.astype(int)
    T = T.join(L.set_index(["season", "pid"]).add_prefix("last_"))
    return T.reset_index()


def boot_delta_r2(y, base, add, w, groups, B=400):
    """Player-clustered bootstrap of the R^2 gain from adding `add` to `base`."""
    ug = np.unique(groups)
    gi = {g: np.nonzero(groups == g)[0] for g in ug}
    out = []
    for _ in range(B):
        pick = RNG.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([gi[g] for g in pick])
        r0, _ = wls_r2(y[ix], [c[ix] for c in base], w[ix])
        r1, _ = wls_r2(y[ix], [c[ix] for c in base] + [c[ix] for c in add], w[ix])
        out.append(r1 - r0)
    return [round(float(np.percentile(out, 2.5)), 5), round(float(np.percentile(out, 97.5)), 5)]


def predictive(T, rap):
    T = T.merge(rap[["season", "pid", "v"]], on=["season", "pid"], how="left")
    T = T[T.season.isin(EVAL)]
    res = {"filter": "season TOI >= 500 min, eval seasons 2011-12..2017-18, TOI-weighted",
           "outcomes": {}}
    base = T[(T.toi_min >= 500)].copy()
    res["n_player_seasons"] = int(len(base))
    res["n_with_rapm"] = int(base.v.notna().sum())
    outcomes = {
        "realised_duel_goals60 (fo+pen)": "dd_g60",
        "realised_duel_goals60 (fo+pen+tkgv)": "dd_all_g60",
        "realised_pen_goals60": "pen_g60",
        "realised_fo_goals60": "fo_g60",
        "realised_tkgv_goals60": "tkgv_g60",
        "onice_GD60_all": "onice_gd60",
        "onice_GD60_5v5": "onice_gd60_5v5",
        "points60": "pts60",
    }
    preds = {"wf_dd_g60": "walk-forward DD (fo+pen)",
             "wf_dd_all_g60": "walk-forward DD (fo+pen+tkgv)",
             "wf_pen_g60": "walk-forward PEN", "wf_fo_g60": "walk-forward FO",
             "wf_tkgv_g60": "walk-forward TKGV",
             "last_dd_g60": "naive last-season DD",
             "last_dd_all_g60": "naive last-season DD+TKGV",
             "last_pen_g60": "naive last-season PEN",
             "last_fo_g60": "naive last-season FO",
             "last_tkgv_g60": "naive last-season TKGV",
             "v": "xG-RAPM (current rating)",
             "last_onice_gd60": "naive last-season on-ice GD60"}
    base["pgrp"] = np.where(base.pos == "D", "D", "F")
    num = [c for c in list(outcomes.values()) + list(preds) if c in base.columns]

    def group_frame(sub, grp):
        if grp == "F":
            return sub[sub.pgrp == "F"]
        if grp == "D":
            return sub[sub.pgrp == "D"]
        if grp == "ALL":
            return sub
        # POSADJ: demean every predictor and outcome within (season, F/D), TOI-weighted
        out = sub.copy()
        for c in num:
            ok = np.isfinite(out[c])
            w = out.toi_min.where(ok, 0)
            m = (out[c].fillna(0) * w).groupby([out.season, out.pgrp]).transform("sum") /                 w.groupby([out.season, out.pgrp]).transform("sum")
            out[c] = out[c] - m
        return out

    plan = [("DEV_all", EVAL, ("POSADJ", "F", "D", "ALL")),
            ("DEV_A", D.TUNE_SEASONS, ("POSADJ",)),
            ("DEV_B", D.HOLD_SEASONS, ("POSADJ", "F", "D"))]
    for split, seas, groups in plan:
        for grp in groups:
            sub = group_frame(base[base.season.isin(seas)], grp)
            for oname, ocol in outcomes.items():
                key = f"{split}|{grp}|{oname}"
                d = sub[np.isfinite(sub[ocol])]
                rec = {"n": int(len(d))}
                for pc in preds:
                    dd_ = d[np.isfinite(d[pc])]
                    if len(dd_) > 50 and dd_[pc].std() > 0:
                        ww = dd_.toi5_min.values if ocol == "onice_gd60_5v5" else dd_.toi_min.values
                        rec["r|" + preds[pc]] = round(wcorr(dd_[pc], dd_[ocol], ww), 4)
                c = d[np.isfinite(d.v) & np.isfinite(d.wf_dd_g60)]
                if len(c) > 100 and split != "DEV_A" and grp != "ALL":
                    y = c[ocol].values
                    ww = c.toi5_min.values if ocol == "onice_gd60_5v5" else c.toi_min.values
                    r_v, _ = wls_r2(y, [c.v.values], ww)
                    r_dd, _ = wls_r2(y, [c.wf_dd_g60.values], ww)
                    r_both, b_ = wls_r2(y, [c.v.values, c.wf_dd_g60.values], ww)
                    rec["common_n"] = int(len(c))
                    rec["R2_rapm"] = round(r_v, 5)
                    rec["R2_dd"] = round(r_dd, 5)
                    rec["R2_rapm+dd"] = round(r_both, 5)
                    rec["dR2_dd_over_rapm"] = round(r_both - r_v, 5)
                    rec["coef_rapm_dd"] = [round(float(b_[1]), 4), round(float(b_[2]), 4)]
                    rec["dR2_dd_over_rapm_CI"] = boot_delta_r2(
                        y, [c.v.values], [c.wf_dd_g60.values], ww, c.pid.values, B=300)
                    if grp != "D" or True:
                        r_all, b3 = wls_r2(y, [c.v.values, c.wf_dd_g60.values,
                                               c.wf_tkgv_g60.values], ww)
                        rec["dR2_tkgv_over_rapm+dd"] = round(r_all - r_both, 5)
                        rec["coef_rapm_dd_tkgv"] = [round(float(x), 4) for x in b3[1:]]
                        rec["dR2_tkgv_over_rapm+dd_CI"] = boot_delta_r2(
                            y, [c.v.values, c.wf_dd_g60.values], [c.wf_tkgv_g60.values], ww,
                            c.pid.values, B=300)
                res["outcomes"][key] = rec
    return res


# ================================================================== U units
def lineup_sums(pg):
    h = pg.exp_toi_s / 3600.0
    x = pg.assign(fo_u=pg.fo_g60 * h, pen_u=pg.pen_g60 * h, tkgv_u=pg.tkgv_g60 * h,
                  dd_u=pg.dd_g60 * h, dda_u=pg.dd_all_g60 * h)
    L = x.groupby(["gid", "side"])[["fo_u", "pen_u", "tkgv_u", "dd_u", "dda_u"]].sum()
    L = L.unstack("side")
    out = pd.DataFrame(index=L.index)
    for c in ["fo_u", "pen_u", "tkgv_u", "dd_u", "dda_u"]:
        out[c + "_diff"] = L[(c, "H")] - L[(c, "A")]
    return out


def lineup_deviation(pg, win=10):
    """dDD: tonight's lineup duel goals minus the mean of the team's previous `win`
    dressed lineups, every lineup re-valued with the latest values known before
    tonight (each player's most recent pre-game value). Pure lineup-change signal
    -- the part a team rating cannot already know."""
    from collections import deque
    x = pg.assign(u=pg.dd_g60 * pg.exp_toi_s / 3600.0,
                  ua=pg.dd_all_g60 * pg.exp_toi_s / 3600.0)
    x = x.sort_values(["date", "gid", "side"])
    latest, latest_a = {}, {}
    hist = {}
    rows = []
    for (gid, side), grp in x.groupby(["gid", "side"], sort=False):
        team = grp.team.iloc[0]
        pids = grp.pid.values
        for p_, u, ua in zip(pids, grp.u.values, grp.ua.values):
            latest[p_] = u
            latest_a[p_] = ua
        cur, cur_a = float(grp.u.sum()), float(grp.ua.sum())
        h = hist.setdefault(team, deque(maxlen=win))
        if h:
            base = np.mean([sum(latest[q] for q in r) for r in h])
            base_a = np.mean([sum(latest_a[q] for q in r) for r in h])
        else:
            base, base_a = cur, cur_a
        rows.append((gid, side, cur - base, cur_a - base_a))
        h.append(tuple(pids))
    d = pd.DataFrame(rows, columns=["gid", "side", "ddev", "ddev_a"]).set_index(["gid", "side"])
    d = d.unstack("side")
    return pd.DataFrame({"ddev_diff": d[("ddev", "H")] - d[("ddev", "A")],
                         "ddev_a_diff": d[("ddev_a", "H")] - d[("ddev_a", "A")]})


def unit_outcomes():
    ev = load_events(columns=["ptype", "ev", "is_home", "h_sk", "a_sk", "h_g", "a_g",
                              "pen_code"])
    ev = ev[ev.ptype != "SO"]
    g = ev[ev.ev == "goal"]
    h = g.is_home == 1
    st = np.where(g.h_sk > g.a_sk, "hPP", np.where(g.h_sk < g.a_sk, "aPP", "EV"))
    # special-teams GD from home side: goals scored with a skater advantage or
    # short-handed (both count), empty-net states excluded from ST by h_g/a_g
    en = (g.h_g == 0) | (g.a_g == 0)
    stg = (~en) & (st != "EV")
    y = pd.DataFrame({"gid": g.gid.values, "gd": np.where(h, 1, -1),
                      "st_gd": np.where(stg, np.where(h, 1, -1), 0)})
    Y = y.groupby("gid").sum()
    fo = ev[ev.ev == "faceoff"]
    F = fo.groupby("gid").is_home.agg(["sum", "size"])
    Y["fo_share_h"] = F["sum"] / F["size"]
    pen = ev[(ev.ev == "penalty") & ev.pen_code.isin(["MIN", "BEN", "MAJ"])]
    P = pen.groupby("gid").is_home.agg(["sum", "size"])
    Y["pen_diff_h"] = (P["size"] - P["sum"]) - P["sum"]         # away pens - home pens
    return Y


def unit_level(pg):
    import nhl_depth_eval as Hd
    ctx = Hd.Ctx()
    gids = np.array([g["game_id"] for g in ctx.games])
    L = lineup_sums(pg).join(lineup_deviation(pg))
    Y = unit_outcomes()
    feats = pd.read_csv(dpath("bt_nhl_rapmel_feats.csv")).set_index("game_id")
    m = ctx.mask
    G = pd.DataFrame({"gid": gids[m], "season": ctx.seasons[m]})
    G = G.join(L, on="gid").join(Y, on="gid")
    G["P_diff"] = G.gid.map(feats.P_home - feats.P_away).values
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    G["elo_logit"] = X0[:, 1]
    G["xg_diff"] = X0[:, 5]
    assert G.season.max() <= 20172018
    res = {"n_games": int(len(G)), "missing_lineup": int(G.dd_u_diff.isna().sum())}
    G = G.dropna(subset=["dd_u_diff", "gd", "P_diff"])
    def ols(ycol, xcols):
        """OLS with HC1 heteroskedasticity-robust t (one row per game)."""
        X = np.column_stack([np.ones(len(G))] + [G[c].values for c in xcols])
        yv = G[ycol].values.astype(float)
        ok = np.isfinite(yv) & np.all(np.isfinite(X), axis=1)
        X, yv = X[ok], yv[ok]
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ yv
        e = yv - X @ b
        n, k = X.shape
        V = XtXi @ (X.T * (e * e)) @ X @ XtXi * n / (n - k)
        t = b / np.sqrt(np.diag(V))
        r2 = 1 - (e @ e) / np.sum((yv - yv.mean()) ** 2)
        return {"coef": dict(zip(["const"] + xcols, [round(float(c), 4) for c in b])),
                "t": dict(zip(["const"] + xcols, [round(float(c), 2) for c in t])),
                "r2": round(float(r2), 5), "n": int(n)}
    res["sd"] = {c: round(float(G[c].std()), 4) for c in
                 ["fo_u_diff", "pen_u_diff", "tkgv_u_diff", "dd_u_diff", "P_diff", "gd", "st_gd"]}
    res["pen_lineup -> pen_diff_h (count)"] = ols("pen_diff_h", ["pen_u_diff"])
    res["pen_lineup -> special-teams GD"] = ols("st_gd", ["pen_u_diff"])
    res["pen_lineup -> ST GD | shipped"] = ols("st_gd", ["pen_u_diff", "elo_logit", "xg_diff"])
    res["fo_lineup -> home FO share"] = ols("fo_share_h", ["fo_u_diff"])
    res["GD ~ shipped"] = ols("gd", ["elo_logit", "xg_diff"])
    res["GD ~ shipped + P"] = ols("gd", ["elo_logit", "xg_diff", "P_diff"])
    res["GD ~ shipped + P + DD"] = ols("gd", ["elo_logit", "xg_diff", "P_diff", "dd_u_diff"])
    res["GD ~ shipped + P + fo + pen + tkgv"] = ols(
        "gd", ["elo_logit", "xg_diff", "P_diff", "fo_u_diff", "pen_u_diff", "tkgv_u_diff"])
    res["GD ~ shipped + P + dDD(lineup deviation)"] = ols(
        "gd", ["elo_logit", "xg_diff", "P_diff", "ddev_diff"])
    res["sd"]["ddev_diff"] = round(float(G.ddev_diff.std()), 4)
    res["GD ~ DD alone"] = ols("gd", ["dd_u_diff"])
    res["GD ~ P alone"] = ols("gd", ["P_diff"])
    return res, ctx, L, feats


def game_screen(ctx, L, feats):
    import nhl_depth_eval as Hd
    gids = np.array([g["game_id"] for g in ctx.games])
    m = ctx.mask
    y = ctx.yv()
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = ctx.folds.loso_pred(X0, y)
    ll0 = ctx.folds.pooled(y, p0)
    Xnx = np.delete(X0, 5, axis=1)
    pnx = ctx.folds.loso_pred(Xnx, y)
    drop_xg = ctx.folds.pooled(y, pnx) - ll0
    Lm = L.reindex(gids[m])
    P = pd.Series(feats.P_home - feats.P_away).reindex(gids[m]).fillna(0).values
    res = {"S1_shipped_dev_loso_ll": round(ll0, 6), "S1_target": 0.672398,
           "S1_pass": abs(ll0 - 0.672398) < 2e-6,
           "S2_drop_xg_cost": round(drop_xg, 5), "S2_target": 0.00303,
           "S2_pass": abs(drop_xg - 0.00303) < 0.0002, "n": int(m.sum()),
           "lineup_missing": int(Lm.dd_u_diff.isna().sum()), "arms": {}}
    arms = {"+DD(fo+pen)": ["dd_u_diff"], "+DD_all(fo+pen+tkgv)": ["dda_u_diff"],
            "+fo,pen,tkgv separate": ["fo_u_diff", "pen_u_diff", "tkgv_u_diff"],
            "+pen": ["pen_u_diff"], "+fo": ["fo_u_diff"], "+tkgv": ["tkgv_u_diff"],
            "+dDD lineup deviation": ["ddev_diff"],
            "+dDD_all lineup deviation": ["ddev_a_diff"]}
    for name, cols in arms.items():
        X = np.column_stack([X0] + [Lm[c].fillna(0).values for c in cols])
        p = ctx.folds.loso_pred(X, y)
        res["arms"][name] = Hd.fold_stats(y, ctx.folds, p0, p)
    # beside the RAPM lineup power (does DD add to the rapmel feature?)
    XP = np.column_stack([X0, P])
    pP = ctx.folds.loso_pred(XP, y)
    XPD = np.column_stack([XP, Lm.dd_u_diff.fillna(0).values])
    pPD = ctx.folds.loso_pred(XPD, y)
    res["arms"]["+P (rapmel lineup power, reference)"] = Hd.fold_stats(y, ctx.folds, p0, pP)
    res["arms"]["+P+DD vs +P"] = Hd.fold_stats(y, ctx.folds, pP, pPD)
    return res


# ================================================================== V face
def face_validity(pg):
    pl = pd.read_csv(dpath("pv_nhl_players.csv"))
    name = (pl.first + " " + pl["last"]).set_axis(pl.pid)
    last = pg.sort_values(["pid", "date"]).groupby("pid").tail(1)
    career_toi = pg.groupby("pid").toi_s.sum() / 60
    last = last.assign(career_toi_min=last.pid.map(career_toi) - last.toi_s / 60,
                       name=last.pid.map(name))
    last = last[(last.career_toi_min >= 3000) & (last.season >= 20162017)]
    def lst(col, asc, k=15, extra=()):
        d = last.sort_values(col, ascending=asc).head(k)
        cols = ["name", "pos", col] + list(extra)
        return [{c: (round(float(r[c]), 4) if isinstance(r[c], (float, np.floating)) else
                     (int(r[c]) if isinstance(r[c], (np.integer,)) else r[c])) for c in cols}
                for _, r in d.iterrows()]
    return {
        "snapshot": "pre-game walk-forward value at each player's last DEV game "
                    "(2016-17 or 2017-18), career TOI >= 3000 min",
        "n_players": int(len(last)),
        "top_dd_g60": lst("dd_g60", False, extra=("fo_g60", "pen_g60")),
        "bottom_dd_g60": lst("dd_g60", True, extra=("fo_g60", "pen_g60")),
        "top_fo_rating": lst("fo_rating", False, extra=("fo_nprev", "fo_g60")),
        "bottom_fo_rating_min2000fo": [x for x in lst("fo_rating", True, k=400,
                                       extra=("fo_nprev", "fo_g60"))
                                       if x["fo_nprev"] >= 2000][:15],
        "top_pen_g60": lst("pen_g60", False, extra=("pen_draw_n60", "pen_take_n60")),
        "bottom_pen_g60": lst("pen_g60", True, extra=("pen_draw_n60", "pen_take_n60")),
        "top_tkgv_g60": lst("tkgv_g60", False, extra=("tk_adj60", "gv_adj60")),
        "bottom_tkgv_g60": lst("tkgv_g60", True, extra=("tk_adj60", "gv_adj60")),
    }


def magnitudes(pg):
    """Spread of the pre-season walk-forward values among regulars (>= 500 min)."""
    x = pg.sort_values(["pid", "date"])
    first = x.groupby(["season", "pid"]).head(1).set_index(["season", "pid"])
    toi = x.groupby(["season", "pid"]).toi_s.sum() / 60
    f = first.join(toi.rename("toi_min"))
    f = f[(f.toi_min >= 500) & (f.index.get_level_values("season") >= 20112012)]
    out = {}
    for grp, m in (("C", f.pos == "C"), ("W", f.pos.isin(["L", "R"])), ("D", f.pos == "D")):
        g = f[m]
        out[grp] = {c: {"sd": round(float(g[c].std()), 4), "p05": round(float(g[c].quantile(.05)), 4),
                        "p95": round(float(g[c].quantile(.95)), 4)}
                    for c in ("fo_g60", "pen_g60", "tkgv_g60", "dd_g60")}
        out[grp]["n"] = int(len(g))
    return out


def main():
    t0 = time.time()
    pg = load_pg()
    out = {"component": "NHL discrete_duels", "protocol": {
        "dev_only": True, "max_gid": int(pg.gid.max()), "max_season": int(pg.season.max()),
        "tuning_seasons": D.TUNE_SEASONS, "untouched_by_tuning": D.HOLD_SEASONS,
        "test_seasons_touched": 0, "market_blind": True}}
    out["F_faceoff_ll"] = faceoff_ll()
    print("F", out["F_faceoff_ll"], flush=True)
    out["R_reliability"] = reliability(pg)
    print("R done", flush=True)
    oi = onice_goals(pg)
    T = player_season_table(pg, oi)
    rap = rapm_preseason(pg)
    out["P_predictive"] = predictive(T, rap)
    print("P done", flush=True)
    out["V_face"] = face_validity(pg)
    out["magnitudes_goals_per60"] = magnitudes(pg)
    ures, ctx, L, feats = unit_level(pg)
    out["U_unit"] = ures
    print("U done", flush=True)
    out["G_game_screen_DEV"] = game_screen(ctx, L, feats)
    print("G done", flush=True)
    vals = json.load(open(D.OUT + "_values.json"))
    out["goal_values_walkforward"] = {k: {kk: vv for kk, vv in v.items() if kk in ("fo", "pen", "tkgv", "fit_on")}
                                      for k, v in vals.items()}
    with open(OUTJ, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
