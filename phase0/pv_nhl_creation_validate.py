"""Player-value program, NHL -- component "creation", STEP 3: player-level validation.

DEV ONLY. Every number here comes from seasons 2010-11..2017-18 (gid < 2018000000).

  reliability   raw per-60 rates, regular season:
                  split-half  odd vs even games of the player's season
                              (players >= 250 EV min in each half; PP: >= 50 PP min)
                  year-over-year  season s vs s+1 (>= 500 EV min both; PP >= 100)
  predict       season-start WALK-FORWARD values (the pv_nhl_creation_rate value at
                the player's first regular-season game of season s, s = 2011-12 ..
                2017-18) against what the player did in season s:
                  T_on_gf   on-ice 5v5 goals for /60        (the owner's question)
                  T_rel_gf  on-ice minus off-ice 5v5 GF/60  (team context removed)
                  T_pts     individual 5v5 points /60
                  T_on_xgf  on-ice 5v5 xGF/60               (the proxy, for reference)
                TOI-weighted Pearson r, player-cluster bootstrap 95% CIs, paired
                delta-r vs the incumbent walk-forward stint-RAPM offence value, team
                changers only, incremental R^2, mid-season (Jan 1 -> end) replication.
  lineup        team-game check: the dressed lineup's TOI-weighted creation vs the
                team's 5v5 goals in that game, partial of a walk-forward team EWMA.
  face          top / bottom lists at the start of 2017-18.

    python phase0/pv_nhl_creation_validate.py            (all; ~2-4 min)
Writes data/pv_nhl_creation_validation.json and data/pv_nhl_creation_wf.parquet
(the deliverable per-player-per-game walk-forward values, all DEV games, chosen
hyper-parameters).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_creation_rate import build, load_pg, EV_IND, EV_ON, PP_ALL, ALL_SIT, OFF  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "data/pv_nhl_creation_validation.json")
OUT_WF = os.path.join(ROOT, "data/pv_nhl_creation_wf.parquet")
EVAL = [20112012, 20122013, 20132014, 20142015, 20152016, 20162017, 20172018]
RNG = np.random.default_rng(20260924)
NBOOT = 1000


def wcorr(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(vx * vy))


def wr2(y, X, w):
    X = np.column_stack([np.ones(len(y))] + list(X))
    sw = np.sqrt(w)
    b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    res = y - X @ b
    return 1 - np.average(res ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2,
                                                            weights=w)


# ------------------------------------------------------------ reliability ---
def season_rates(g):
    m = g.toi_ev / 3600.0
    mp = (g.toi_pp / 3600.0).where(g.toi_pp > 0)
    o = pd.DataFrame(index=g.index)
    for c in ("ixg_ev", "icf_ev", "iff_ev", "isf_ev", "g_ev", "a1_ev", "a2_ev",
              "ireb_ev", "irush_ev", "irc_ev", "on_gf_ev", "on_xgf_ev", "on_cf_ev"):
        o[c] = g[c] / m
    o["pts_ev (cre_ev raw)"] = (g.g_ev + g.a1_ev + g.a2_ev) / m
    o["pts_ev+ixg (cre_evx raw)"] = (g.g_ev + g.a1_ev + g.a2_ev + g.ixg_ev) / m
    o["fin_ev (g-ixg)"] = (g.g_ev - g.ixg_ev) / m
    o["rel_gf"] = g.on_gf_ev / m - g.off_gf / (g.off_toi / 3600.0)
    o["rel_xgf"] = g.on_xgf_ev / m - g.off_xgf / (g.off_toi / 3600.0)
    for c in ("ixg_pp", "g_pp", "a1_pp", "on_gf_pp", "on_xgf_pp"):
        o[c] = g[c] / mp
    o["pts_pp (cre_pp raw)"] = (g.g_pp + g.a1_pp + g.a2_pp) / mp
    o["pdrawn"] = g.pdrawn / (g.toi_all / 3600.0)
    o["min_ev"], o["min_pp"] = g.toi_ev / 60.0, g.toi_pp / 60.0
    return o


SUMC = ["toi_ev", "toi_pp", "toi_all", "off_toi", "ixg_ev", "icf_ev", "iff_ev", "isf_ev",
        "g_ev", "a1_ev", "a2_ev", "ireb_ev", "irush_ev", "irc_ev", "on_gf_ev", "on_xgf_ev",
        "on_cf_ev", "off_gf", "off_xgf", "ixg_pp", "g_pp", "a1_pp", "a2_pp", "on_gf_pp",
        "on_xgf_pp", "pdrawn"]


def reliability(pg):
    r = pg[pg.gtype == 2].sort_values(["pid", "date"]).copy()
    r["half"] = r.groupby(["pid", "season"]).cumcount() % 2
    grp = r.groupby(["season", "pid"]).grp.agg(lambda s: s.mode().iloc[0])
    H = r.groupby(["season", "pid", "half"])[SUMC].sum()
    R = season_rates(H)
    h0 = R.xs(0, level="half")
    h1 = R.xs(1, level="half")
    j = h0.join(h1, lsuffix="_0", rsuffix="_1", how="inner").join(grp)
    Sg = r.groupby(["season", "pid"])[SUMC].sum()
    SR = season_rates(Sg).join(grp).reset_index()
    nxt = {s: s + 10001 for s in EVAL + [20102011]}
    SR2 = SR.copy()
    SR2["season"] = SR2.season.map(nxt)
    yy = SR.merge(SR2, on=["season", "pid"], suffixes=("_1", "_0"))
    yy = yy[yy.grp_1 == yy.grp_0].rename(columns={"grp_1": "grp"})
    metrics = [c for c in R.columns if not c.startswith("min_")]
    out = {}
    for g in ("F", "D"):
        rows = {}
        for c in metrics:
            pp = c.endswith("_pp") or c.startswith("pts_pp")
            mcol, lo_h, lo_y = ("min_pp", 50, 100) if pp else ("min_ev", 250, 500)
            q = j[(j.grp == g) & (j[f"{mcol}_0"] >= lo_h) & (j[f"{mcol}_1"] >= lo_h)]
            q = q[np.isfinite(q[f"{c}_0"]) & np.isfinite(q[f"{c}_1"])]
            rh = float(np.corrcoef(q[f"{c}_0"], q[f"{c}_1"])[0, 1])
            T = float(np.median(q[f"{mcol}_0"]))
            y = yy[(yy.grp == g) & (yy[f"{mcol}_0"] >= lo_y) & (yy[f"{mcol}_1"] >= lo_y)]
            y = y[np.isfinite(y[f"{c}_0"]) & np.isfinite(y[f"{c}_1"])]
            ry = float(np.corrcoef(y[f"{c}_0"], y[f"{c}_1"])[0, 1])
            rows[c] = {"split_half_r": round(rh, 3), "split_half_n": int(len(q)),
                       "spearman_brown_full_season": round(2 * rh / (1 + rh), 3),
                       "median_half_minutes": round(T),
                       "k_minutes_implied": (round(T * (1 - rh) / rh) if rh > 0 else None),
                       "yoy_r": round(ry, 3), "yoy_n": int(len(y))}
        out[g] = rows
    return out


# ---------------------------------------------------------------- predict ---
PREDS = ["cre_ev", "cre_evx", "g_ev_hat", "a1_ev_hat", "a2_ev_hat", "ixg_ev_hat",
         "icf_ev_hat", "fin_ev_hat", "irush_ev_hat", "ireb_ev_hat", "irc_ev_hat",
         "on_gf_ev_hat", "on_xgf_ev_hat", "on_cf_ev_hat", "rel_gf_hat", "rel_xgf_hat",
         "rapm_off", "rapm_off_s", "rapm_net_v"]
TGTS = ["T_on_gf", "T_rel_gf", "T_pts", "T_on_xgf"]


def outcomes(pg, start_date=None):
    r = pg[pg.gtype == 2]
    if start_date is not None:
        r = r.merge(start_date, on=["season", "pid"])
        r = r[r.date >= r.start]
    S = r.groupby(["season", "pid"])[SUMC].sum()
    m = S.toi_ev / 3600.0
    o = pd.DataFrame(index=S.index)
    o["T_on_gf"] = S.on_gf_ev / m
    o["T_on_xgf"] = S.on_xgf_ev / m
    o["T_rel_gf"] = S.on_gf_ev / m - S.off_gf / (S.off_toi / 3600.0)
    o["T_pts"] = (S.g_ev + S.a1_ev + S.a2_ev) / m
    o["T_min_ev"] = S.toi_ev / 60.0
    mp = S.toi_pp / 3600.0
    o["T_pp_on_gf"] = (S.on_gf_pp / mp).where(S.toi_pp >= 100 * 60)
    o["T_pp_pts"] = ((S.g_pp + S.a1_pp + S.a2_pp) / mp).where(S.toi_pp >= 100 * 60)
    o["T_min_pp"] = S.toi_pp / 60.0
    # main team of the season (most EV minutes)
    tm = r.groupby(["season", "pid", "team"]).toi_ev.sum().reset_index()
    tm = tm.sort_values("toi_ev").groupby(["season", "pid"]).team.last()
    o["team_s"] = tm
    return o.reset_index()


def season_start_values(wf, mid=False):
    w = wf[(wf.gtype == 2) & wf.season.isin(EVAL)].sort_values(["pid", "date"])
    if mid:
        w = w[w.date.str[5:7].isin(["01", "02", "03", "04"])]
    first = w.groupby(["season", "pid"]).head(1)
    return first


def build_panel(wf, pg, mid=False):
    v = season_start_values(wf, mid)
    sd = v[["season", "pid", "date"]].rename(columns={"date": "start"})
    o = outcomes(pg, sd if mid else None)
    P = v.merge(o, on=["season", "pid"], how="inner")
    # previous season's main team (team changers)
    prev = outcomes(pg)[["season", "pid", "team_s"]].copy()
    prev["season"] = prev.season + 10001
    P = P.merge(prev.rename(columns={"team_s": "team_prev"}), on=["season", "pid"], how="left")
    P["changer"] = P.team_prev.notna() & (P.team_prev != P.team_s)
    return P


def sample(P, grp, min_ev=400, min_prior=200):
    return P[(P.grp == grp) & (P.T_min_ev >= min_ev) & (P.prior_min_ev >= min_prior)
             & P.rapm_off.notna()]


def boot_table(q, preds, tgt, ref="rapm_off", nb=NBOOT):
    pids = q.pid.unique()
    idx = {p: np.where(q.pid.to_numpy() == p)[0] for p in pids}
    x = {p: q[p].to_numpy(float) for p in preds}
    y, w = q[tgt].to_numpy(float), q.T_min_ev.to_numpy(float)
    base = {p: wcorr(x[p], y, w) for p in preds}
    bs = {p: [] for p in preds}
    for _ in range(nb):
        pick = RNG.choice(len(pids), len(pids), replace=True)
        ii = np.concatenate([idx[pids[k]] for k in pick])
        for p in preds:
            bs[p].append(wcorr(x[p][ii], y[ii], w[ii]))
    out = {}
    for p in preds:
        b = np.array(bs[p])
        d = b - np.array(bs[ref])
        out[p] = {"r": round(base[p], 4),
                  "ci": [round(float(np.percentile(b, 2.5)), 4),
                         round(float(np.percentile(b, 97.5)), 4)],
                  f"delta_vs_{ref}": round(base[p] - base[ref], 4),
                  "delta_ci": [round(float(np.percentile(d, 2.5)), 4),
                               round(float(np.percentile(d, 97.5)), 4)]}
    return out


def predict_block(P, label, nb=NBOOT, full=True):
    res = {}
    for g in ("F", "D"):
        q = sample(P, g)
        blk = {"n_player_seasons": int(len(q)), "n_players": int(q.pid.nunique())}
        for t in TGTS:
            if full:
                blk[t] = boot_table(q, PREDS if t == "T_on_gf" else
                                    ["cre_ev", "cre_evx", "g_ev_hat", "a1_ev_hat",
                                     "ixg_ev_hat", "on_gf_ev_hat", "on_xgf_ev_hat",
                                     "rel_gf_hat", "rapm_off"], t, nb=nb)
            else:
                blk[t] = {p: round(wcorr(q[p].to_numpy(float), q[t].to_numpy(float),
                                         q.T_min_ev.to_numpy(float)), 4) for p in PREDS}
        res[g] = blk
    res["label"] = label
    return res


def incremental(P):
    out = {}
    for g in ("F", "D"):
        q = sample(P, g)
        w = q.T_min_ev.to_numpy(float)
        o = {}
        for t in ("T_on_gf", "T_rel_gf"):
            y = q[t].to_numpy(float)
            col = lambda c: q[c].to_numpy(float)  # noqa: E731
            rows = {}
            for name, base in (("rapm_off", ["rapm_off"]),
                               ("on_xgf_ev_hat", ["on_xgf_ev_hat"]),
                               ("on_gf_ev_hat", ["on_gf_ev_hat"]),
                               ("rapm_off+on_xgf+on_gf", ["rapm_off", "on_xgf_ev_hat",
                                                          "on_gf_ev_hat"])):
                b = wr2(y, [col(c) for c in base], w)
                rows[name] = {"R2_base": round(b, 4),
                              "R2_plus_cre_ev": round(wr2(y, [col(c) for c in base + ["cre_ev"]], w), 4),
                              "R2_plus_cre_evx": round(wr2(y, [col(c) for c in base + ["cre_evx"]], w), 4)}
            rows["cre_ev_alone"] = round(wr2(y, [col("cre_ev")], w), 4)
            rows["cre_ev+rapm_off"] = round(wr2(y, [col("cre_ev"), col("rapm_off")], w), 4)
            o[t] = rows
        out[g] = o
    return out


def pp_block(P):
    out = {}
    for g in ("F", "D"):
        q = P[(P.grp == g) & (P.T_min_pp >= 100) & (P.prior_min_pp >= 50)]
        w = q.T_min_pp.to_numpy(float)
        o = {"n": int(len(q))}
        for t in ("T_pp_on_gf", "T_pp_pts"):
            y = q[t].to_numpy(float)
            o[t] = {p: round(wcorr(q[p].to_numpy(float), y, w), 4)
                    for p in ("cre_pp", "g_pp_hat", "a1_pp_hat", "ixg_pp_hat",
                              "on_gf_pp_hat", "on_xgf_pp_hat", "cre_ev", "rapm_off")}
        out[g] = o
    return out


# ---------------------------------------------------------------- lineup ---
def lineup_check(wf, pg):
    """Team-game: TOI-weighted creation of the dressed lineup vs the team's 5v5
    goals in that game; partial of the team's own walk-forward EWMA of 5v5 GF/60
    and xGF/60 (half-life 40 team games)."""
    w = wf[wf.gtype == 2].copy()
    em = w.exp_min_ev.fillna(np.nan)
    fill = w.groupby("grp").exp_min_ev.transform(lambda s: s[w.loc[s.index, "prior_min_ev"] < 300].median())
    w["em"] = np.where(np.isfinite(em) & (w.prior_min_ev >= 60), em, fill)
    cols = ["cre_ev", "cre_evx", "ixg_ev_hat", "on_gf_ev_hat", "on_xgf_ev_hat", "rapm_off_s"]
    w["rapm_off_s"] = w.rapm_off_s.fillna(0.0)
    agg = {}
    for c in cols:
        w[c + "_w"] = w[c] * w.em
    g = w.groupby(["gid", "team"]).agg(**{c: (c + "_w", "sum") for c in cols},
                                       em=("em", "sum"), season=("season", "first"),
                                       date=("date", "first"))
    for c in cols:
        g[c] = g[c] / g.em * 5.0
    tg = pg[pg.gtype == 2].groupby(["gid", "team"]).agg(tm_gf=("tm_gf_ev", "first"),
                                                       tm_xgf=("tm_xgf_ev", "first"),
                                                       tm_toi=("tm_toi_ev", "first"))
    g = g.join(tg).reset_index().sort_values(["team", "date", "gid"])
    lam = 0.5 ** (1 / 40.0)
    for c in ("tm_gf", "tm_xgf", "tm_toi"):
        x = g[c].to_numpy(float)
        k = g.groupby("team").cumcount().to_numpy().astype(float)
        wn = lam ** (-k)
        cs = pd.Series(x * wn).groupby(g.team.to_numpy()).cumsum().to_numpy() - x * wn
        g["ew_" + c] = np.where(k > 0, cs * lam ** (k - 1), np.nan)
    g["ew_gf60"] = g.ew_tm_gf / g.ew_tm_toi * 3600
    g["ew_xgf60"] = g.ew_tm_xgf / g.ew_tm_toi * 3600
    g["y"] = g.tm_gf / g.tm_toi * 3600
    q = g[g.season.isin(EVAL) & (g.tm_toi > 1200) & g.ew_gf60.notna()].copy()
    wt = q.tm_toi.to_numpy(float)
    y = q.y.to_numpy(float)
    res = {"n_team_games": int(len(q)), "target": "team 5v5 GF/60 in the game (TOI-weighted)"}
    base = [q.ew_gf60.to_numpy(float), q.ew_xgf60.to_numpy(float)]
    r2b = wr2(y, base, wt)
    res["R2_team_ewma_only"] = round(r2b, 5)
    for c in cols:
        x = q[c].to_numpy(float)
        res[c] = {"r": round(wcorr(x, y, wt), 4),
                  "R2_alone": round(wr2(y, [x], wt), 5),
                  "R2_team_ewma_plus": round(wr2(y, base + [x], wt), 5),
                  "delta_R2_over_team_ewma": round(wr2(y, base + [x], wt) - r2b, 5)}
    both = [q.cre_ev.to_numpy(float), q.rapm_off_s.to_numpy(float)]
    res["R2_team_ewma+cre_ev+rapm_off_s"] = round(wr2(y, base + both, wt), 5)
    res["R2_team_ewma+rapm_off_s"] = round(wr2(y, base + [q.rapm_off_s.to_numpy(float)], wt), 5)
    return res


# ------------------------------------------------------------ leak audit ---
def walk_forward_audit(H, prior, cut="2015-01-15"):
    """Blank every count of games ON the cut date and drop every game AFTER it;
    every value for games dated <= cut must be bit-identical to the full build
    (a value may use only strictly earlier games; the league prior mean only
    strictly earlier dates)."""
    full = build(H, prior, reg_only=False)
    d = load_pg(reg_only=False)
    d = d[d.date <= cut].copy()
    cnt = EV_IND + EV_ON + PP_ALL + ALL_SIT + OFF + ["toi_ev", "toi_pp", "toi_all", "off_toi"]
    d.loc[d.date == cut, cnt] = 0.0
    tr = build(H, prior, reg_only=False, d=d.reset_index(drop=True))
    key = ["gid", "pid"]
    f = full[full.date <= cut].set_index(key).sort_index()
    t = tr.set_index(key).sort_index()
    cols = [c for c in f.columns if c.endswith("_hat") or c.startswith("cre_")
            or c.startswith("prior_min") or c.startswith("exp_min") or c.startswith("rapm_")]
    cols = [c for c in cols if c != "rapm_cutoff"]
    a, b = f[cols].to_numpy(float), t.loc[f.index, cols].to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    nan_mismatch = int((np.isnan(a) ^ np.isnan(b)).sum())
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    rel = diff / np.maximum(1.0, np.abs(np.nan_to_num(a)))
    on_cut = (f.date == cut).to_numpy()
    # float64 closed-form EWMA: blanking a row changes cumsum rounding (~1e-10
    # relative); a real leak moves values by O(1e-2) or more
    return {"cut_date": cut, "rows_checked": int(len(f)),
            "rows_on_cut_date": int(on_cut.sum()),
            "columns_checked": len(cols), "nan_pattern_mismatches": nan_mismatch,
            "max_abs_diff": float(np.nanmax(diff)),
            "max_rel_diff": float(np.nanmax(rel)),
            "max_abs_diff_rows_on_cut_date": float(np.nanmax(diff[on_cut])),
            "tolerance_rel": 1e-6,
            "pass": bool(np.nanmax(rel) < 1e-6 and nan_mismatch == 0)}


# ------------------------------------------------------------------ face ---
def face(wf):
    names = pd.read_csv(os.path.join(ROOT, "data/pv_nhl_players.csv"))
    names["name"] = names["first"] + " " + names["last"]
    v = season_start_values(wf)
    v = v[v.season == 20172018].merge(names[["pid", "name"]], on="pid", how="left")
    out = {"as_of": "each player's first regular-season game of 2017-18 (walk-forward)"}
    cols = ["name", "team", "cre_ev", "g_ev_hat", "a1_ev_hat", "ixg_ev_hat", "rapm_off",
            "prior_min_ev"]
    for g, lo in (("F", 600), ("D", 600)):
        q = v[(v.grp == g) & (v.prior_min_ev >= lo)].copy()
        rnd = lambda df: [{k: (round(float(r[k]), 3) if isinstance(r[k], (float, np.floating)) else r[k])  # noqa: E731
                           for k in cols} for _, r in df.iterrows()]
        out[g] = {"n_eligible": int(len(q)), "min_prior_ev_minutes": lo,
                  "top_cre_ev": rnd(q.nlargest(15, "cre_ev")),
                  "bottom_cre_ev": rnd(q.nsmallest(10, "cre_ev")),
                  "top_rapm_off": rnd(q.nlargest(10, "rapm_off"))}
    return out


def main():
    t0 = time.time()
    pg = load_pg(reg_only=False)
    res = {"component": "NHL creation", "protocol": {
        "dev_only": True, "max_gid": DEV_MAX_GID, "eval_seasons": EVAL,
        "test_seasons_touched": 0, "market_inputs": 0}}
    res["reliability"] = reliability(pg)
    print(f"reliability done {time.time()-t0:.0f}s", flush=True)

    # hyper-parameter choice on DEV, head-to-head variant (regular-season accumulation,
    # like the RAPM incumbent); criterion = mean F/D r(cre_ev, T_on_gf)
    grid = {}
    for H in (80.0, 160.0, 320.0, float("inf")):
        for prior in ("all", "fringe"):
            wf = build(H, prior, reg_only=True)
            P = build_panel(wf, pg)
            rr = {g: wcorr(sample(P, g).cre_ev.to_numpy(float), sample(P, g).T_on_gf.to_numpy(float),
                           sample(P, g).T_min_ev.to_numpy(float)) for g in ("F", "D")}
            grid[f"H={H}|prior={prior}"] = {g: round(v, 4) for g, v in rr.items()}
            grid[f"H={H}|prior={prior}"]["mean"] = round((rr["F"] + rr["D"]) / 2, 4)
            print(H, prior, grid[f"H={H}|prior={prior}"], flush=True)
    best = max(grid, key=lambda k: grid[k]["mean"])
    H = float(best.split("|")[0][2:])
    prior = best.split("=")[-1]
    res["hyper_grid_cre_ev_vs_T_on_gf"] = grid
    res["chosen"] = {"half_life_games": H, "prior": prior}

    wf = build(H, prior, reg_only=True)
    P = build_panel(wf, pg)
    res["predict_season_start"] = predict_block(P, "season-start walk-forward -> season s")
    res["predict_team_changers"] = predict_block(P[P.changer], "team changers only", nb=500)
    res["incremental_R2"] = incremental(P)
    res["power_play"] = pp_block(P)
    Pm = build_panel(wf, pg, mid=True)
    res["predict_midseason"] = predict_block(Pm, "value at first game on/after Jan 1 -> rest of season",
                                             full=False)
    print(f"predict done {time.time()-t0:.0f}s", flush=True)

    # deliverable: all DEV games (regular season + playoffs accumulate)
    wfa = build(H, prior, reg_only=False)
    res["lineup_team_game"] = lineup_check(wfa, pg)
    res["face_validity"] = face(wfa)
    res["walk_forward_audit"] = walk_forward_audit(H, prior)
    print("audit", res["walk_forward_audit"], flush=True)
    for c in wfa.columns:
        if wfa[c].dtype == np.float64:
            wfa[c] = wfa[c].astype(np.float32)
    assert int(wfa.gid.max()) < DEV_MAX_GID
    wfa.to_parquet(OUT_WF, index=False)
    res["files"] = {"wf": OUT_WF, "rows": int(len(wfa)), "games": int(wfa.gid.nunique())}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"wrote {OUT_JSON} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
