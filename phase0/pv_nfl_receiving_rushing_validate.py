"""Player-value program (pv), NFL receiving_rushing, STEP 3: PLAYER-LEVEL validation, DEV only.

Reads the outputs of STEP 1/2 and asks, for receivers (WR/TE/RB targets) and
rushers (RB/FB designed carries), on DEV seasons 2006-2015 only:

 (1) RELIABILITY  - split-half (odd/even appearance within season, Spearman-
     Brown corrected) and year-over-year correlation of each season-level
     metric, with n.
 (2) PREDICTIVE VALIDITY (future RESULTS: yards, first downs, TDs, points)
     a. FIP-style: season-s metric -> season-(s+1) results, same player; and
        the same for players who CHANGED TEAM (new QB / line / scheme) - the
        test of whether a metric is the player's own.
     b. walk-forward per game: the PRE-GAME rating -> that game's results,
        out-of-sample R^2 of a walk-forward (fit on seasons < s) OLS, with a
        game-clustered bootstrap of the NEW-minus-comparator difference.
     c. unit level: pre-game usage-weighted aggregates of the offense's skill
        players -> the offense's POINTS, beyond an as-of team model (points
        EWMA + team EPA/play EWMA, both sides), walk-forward OLS.
     Comparators: naive results ratings (EB yards/target, yards/carry), the
     SHIPPED DEV-era skill rating (rate6 SK composite: 40% EPA/opportunity,
     15% YAC/rec, 45% WOPR, decay .90/wk, EB 6 wks - phase0/
     nfl_player_rating_system.py, reconstructed from the same weekly lines),
     and the shared-credit proxy (DEV analog of the 11v11 participation TS; the
     real 11v11 rating is built from 2016+ participation lists = TEST seasons,
     so it cannot be evaluated here without breaking the locked split).
 (3) FACE VALIDITY - top / bottom lists at the end of DEV seasons.

Hyper-parameters of STEP 2 were chosen on 2007-2010 events, so the tables are
also reported on 2011-2015 (selection-free).

Output: data/pv_nfl_receiving_rushing_validation.json (+ _player_games_dev_scored.parquet)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEV_LO, DEV_HI = 2006, 2015
OUT = {}
RNG = np.random.default_rng(20260924)

EV = pd.read_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_events.parquet"))
EV = EV[EV.season <= DEV_HI]
PG = pd.read_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_player_games.parquet"))
PG = PG[PG.season <= DEV_HI]
assert EV.season.max() <= DEV_HI and PG.season.max() <= DEV_HI
PLY = pd.read_csv(os.path.join(DATA, "nfl_players.csv"), usecols=["gsis_id", "display_name", "position",
                                                                    "position_group"], low_memory=False)
POSG = dict(zip(PLY.gsis_id, PLY.position_group))
POS = dict(zip(PLY.gsis_id, PLY.position))
NAME = dict(zip(PLY.gsis_id, PLY.display_name))
EV["pg"] = EV.player_id.map(POSG)
PG["pg"] = PG.player_id.map(POSG)
REC_PG = {"WR", "TE", "RB"}
RUSH_PG = {"RB"}  # includes FB (position_group RB)

# ---------------------------------------------------------------- rate6 (shipped DEV rating)
ST = pd.read_csv(os.path.join(DATA, "nfl_player_stats.csv"), low_memory=False,
                 usecols=["player_id", "position", "season", "week", "season_type", "carries", "targets",
                          "receptions", "rushing_epa", "receiving_epa", "receiving_yards_after_catch", "wopr"])
ST = ST[(ST.season <= DEV_HI) & ST.position.isin(["RB", "WR", "TE", "FB"])].fillna(0.0)
assert ST.season.max() <= DEV_HI
ST = ST.sort_values(["season", "week"])
DEC = 0.90


def rate6_walk():
    """Replicates nfl_player_rating_system.rate_of() for bucket SK: {(pid, season, week): pre-week composite}."""
    st = defaultdict(lambda: defaultdict(float))
    snaps_for_z = defaultdict(list)
    pre = {}
    for (s, w), d in ST.groupby(["season", "week"], sort=True):
        for r in d.itertuples(index=False):
            pre[(r.player_id, s, w)] = dict(st[r.player_id]) if r.player_id in st else None
            if 2001 <= s <= DEV_HI:
                q = st.get(r.player_id)
                if q:
                    for k in ("eff", "yacr", "wopr"):
                        if q.get(k + "_w", 0) > 3:
                            snaps_for_z[k].append(q[k] / q[k + "_w"])
        for r in d.itertuples(index=False):
            q = st[r.player_id]
            opp = r.carries + r.targets
            for k, v, wt in (("eff", r.rushing_epa + r.receiving_epa, opp),
                             ("yacr", r.receiving_yards_after_catch, r.receptions), ("wopr", r.wopr, 1.0)):
                if k == "eff" and opp <= 0:
                    continue
                if k == "yacr" and r.receptions <= 0:
                    continue
                q[k] = DEC * q[k] + v
                q[k + "_w"] = DEC * q[k + "_w"] + wt
    Z = {k: (float(np.mean(v)), float(np.std(v))) for k, v in snaps_for_z.items()}
    comp = {}
    for key, q in pre.items():
        if not q:
            comp[key] = np.nan
            continue
        tot = 0.0; used = False
        for k, wt in (("eff", 0.40), ("yacr", 0.15), ("wopr", 0.45)):
            wgt = q.get(k + "_w", 0.0)
            if wgt <= 1:
                continue
            mu, sd = Z[k]
            v = (q.get(k, 0.0) + 6.0 * mu) / (wgt + 6.0)
            tot += wt * (v - mu) / sd; used = True
        comp[key] = tot if used else np.nan
    return comp, Z


R6, Z6 = rate6_walk()
PG["rate6"] = [R6.get((p, s, w), np.nan) for p, s, w in zip(PG.player_id, PG.season, PG.week)]
OUT["rate6_join"] = float(PG.rate6.notna().mean())
print(f"rate6 pre-week composite joined to {OUT['rate6_join']:.3f} of player-games (NaN = no prior line)")


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


# ============================================================== season-level metrics
EV = EV[EV.season.between(DEV_LO, DEV_HI)].copy()
EV["recyds"] = np.where(EV.complete_pass == 1, EV.yards_gained, 0.0)
EV["recfd"] = np.where(EV.complete_pass == 1, EV.first_down, 0.0)
EV["rectd"] = np.where(EV.complete_pass == 1, EV.touchdown, 0.0)
EV["app"] = EV.groupby(["player_id", "season"]).gidx.rank(method="dense").astype(int)
EV["half"] = EV.app % 2

tgt = EV[(EV.kind == "T") & EV.pg.isin(REC_PG) & EV.r_comp.notna()]
car = EV[(EV.kind == "R") & EV.pg.isin(RUSH_PG) & EV.r_ryds.notna()]

REC_METRICS = {  # name -> (column, per-event mask) ; season value = mean over events
    "ypt (raw yds/target)": ("recyds", None),
    "catch% (raw)": ("complete_pass", None),
    "epa/target (raw)": ("epa", None),
    "yac/rec (raw)": ("yac_c", "comp"),
    "aDOT (air yds/target)": ("air_yards", None),
    "xYPT opp. quality (sit.)": ("x_ypt", None),
    "YPTOE execution (sit.)": ("r_yptoe", None),
    "xEPA/target opp. quality (sit.)": ("xepa", None),
    "CROE (sit.)": ("r_comp", None),
    "YACOE (sit.)": ("r_yac", "comp"),
    "EPAOE/target (sit.)": ("r_epat", None),
    "ypt adj (QB+def)": ("a_ypta", None),
    "xYPT adj (QB+def)": ("a_xypt", None),
    "YPTOE adj (QB+def)": ("a_yptoe", None),
    "xEPA/target adj (QB+def)": ("a_xepat", None),
    "CROE adj (QB+def)": ("a_comp", None),
    "YACOE adj (QB+def)": ("a_yac", "comp"),
    "EPAOE/target adj (QB+def)": ("a_epat", None),
}
RUSH_METRICS = {
    "ypc (raw)": ("yards_gained", None),
    "success (raw epa>0)": ("succ", None),
    "epa/carry (raw)": ("epa", None),
    "RYOE capped (sit.)": ("r_ryds", None),
    "SOE (sit.)": ("r_rsuc", None),
    "EPAOE/carry (sit.)": ("r_epar", None),
    "ypc adj (line+def)": ("a_ypca", None),
    "RYOE adj (line+def)": ("a_ryds", None),
    "SOE adj (line+def)": ("a_rsuc", None),
    "EPAOE/carry adj (line+def)": ("a_epar", None),
}


def season_table(ev, metrics, by=("player_id", "season")):
    g = ev.groupby(list(by))
    out = pd.DataFrame({"n": g.size()})
    for nm, (col, msk) in metrics.items():
        if msk == "comp":
            out[nm] = ev[ev.complete_pass == 1].groupby(list(by))[col].mean()
        else:
            out[nm] = g[col].mean()
    return out


pgs = PG[PG.season.between(DEV_LO, DEV_HI)]
EVall = pd.read_parquet(os.path.join(DATA, "pv_nfl_events.parquet"),
                        columns=["game_id", "season", "posteam", "defteam", "play_type", "epa"],
                        filters=[("season", ">=", DEV_LO), ("season", "<=", DEV_HI)])
assert EVall.season.max() <= DEV_HI
EVall = EVall[EVall.play_type.isin(["pass", "run"]) & EVall.epa.notna()]
team_game_epa = EVall.groupby(["game_id", "posteam"]).epa.mean()
pgs = pgs.assign(team_epa=[team_game_epa.get((g, t), np.nan) for g, t in zip(pgs.game_id, pgs.team)])
use = pgs.groupby(["player_id", "season"]).agg(tg=("y_tgt", "sum"), ttg=("team_tgt", "sum"),
                                               cr=("y_car", "sum"), tcr=("team_car", "sum"),
                                               g=("gidx", "size"), shared_raw=("team_epa", "mean"),
                                               team=("team", lambda t: t.value_counts().index[0]))
use["tshare"] = use.tg / use.ttg.clip(lower=1)
use["cshare"] = use.cr / use.tcr.clip(lower=1)


def rate6_season_of(stq):
    g = stq.groupby(["player_id", "season"]).agg(
        epa=("rushing_epa", "sum"), repa=("receiving_epa", "sum"), car=("carries", "sum"), tg=("targets", "sum"),
        yac=("receiving_yards_after_catch", "sum"), rec=("receptions", "sum"), wopr=("wopr", "mean"))
    return (0.40 * (((g.epa + g.repa) / (g.car + g.tg).clip(lower=1)) - Z6["eff"][0]) / Z6["eff"][1]
            + 0.15 * ((g.yac / g.rec.clip(lower=1)) - Z6["yacr"][0]) / Z6["yacr"][1]
            + 0.45 * (g.wopr - Z6["wopr"][0]) / Z6["wopr"][1]).rename("rate6_season")


ST_dev = ST[ST.season.between(DEV_LO, DEV_HI)].copy()
use = use.join(rate6_season_of(ST_dev), how="left")
pgs_h = pgs.assign(app=pgs.groupby(["player_id", "season"]).gidx.rank(method="dense").astype(int))
pgs_h["half"] = pgs_h.app % 2
HALF_OF_WEEK = dict(zip(zip(pgs_h.player_id, pgs_h.season, pgs_h.week), pgs_h.half))
ST_dev["half"] = [HALF_OF_WEEK.get(k, -1) for k in zip(ST_dev.player_id, ST_dev.season, ST_dev.week)]
HU = []
for h in (0, 1):
    q = pgs_h[pgs_h.half == h].groupby(["player_id", "season"]).agg(
        tg=("y_tgt", "sum"), ttg=("team_tgt", "sum"), cr=("y_car", "sum"), tcr=("team_car", "sum"),
        shared_raw=("team_epa", "mean"))
    q["tshare"] = q.tg / q.ttg.clip(lower=1); q["cshare"] = q.cr / q.tcr.clip(lower=1)
    q = q.join(rate6_season_of(ST_dev[ST_dev.half == h]), how="left")
    HU.append(q[["tshare", "cshare", "shared_raw", "rate6_season"]])


def reliability(ev, metrics, min_n, extra_cols, label):
    res = {}
    full = season_table(ev, metrics).join(use[extra_cols], how="left")
    halves = [season_table(ev[ev.half == h], metrics).join(HU[h][extra_cols], how="left") for h in (0, 1)]
    for nm in list(metrics) + list(extra_cols):
        j = halves[0][["n", nm]].join(halves[1][["n", nm]], lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.n_a >= min_n / 2) & (j.n_b >= min_n / 2)]
        r, n = corr(j[nm + "_a"], j[nm + "_b"])
        sh = {"r_half": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3), "n": n}
        f = full[full.n >= min_n][[nm]].dropna().reset_index()
        yy = f.merge(f.assign(season=f.season - 1), on=["player_id", "season"], suffixes=("_s", "_s1"))
        r, n = corr(yy[nm + "_s"], yy[nm + "_s1"])
        yy2 = yy[yy.season >= 2010]
        r2, n2 = corr(yy2[nm + "_s"], yy2[nm + "_s1"])
        res[nm] = {"split_half": sh, "yoy": {"r": round(r, 3), "n": n}, "yoy_2010_15": {"r": round(r2, 3), "n": n2}}
    print(f"\n(1) RELIABILITY - {label} (min {min_n} per season; split-half needs {min_n//2} per half)")
    print(f"   {'metric':<30} {'r_half':>7} {'SB':>6} {'n':>5} | {'YoY r':>6} {'n':>5} | {'YoY 10-15':>9} {'n':>5}")
    for nm, v in res.items():
        sh = v["split_half"]
        print(f"   {nm:<30} {sh['r_half']:>7.3f} {sh['spearman_brown']:>6.3f} {sh['n']:>5} | "
              f"{v['yoy']['r']:>6.3f} {v['yoy']['n']:>5} | {v['yoy_2010_15']['r']:>9.3f} {v['yoy_2010_15']['n']:>5}")
    return res, full


EXT_T = ["tshare", "shared_raw", "rate6_season"]
EXT_R = ["cshare", "shared_raw", "rate6_season"]
OUT["reliability_receiving"], REC_S = reliability(tgt, REC_METRICS, 50, EXT_T, "RECEIVERS (WR/TE/RB targets)")
OUT["reliability_rushing"], RUSH_S = reliability(car, RUSH_METRICS, 100, EXT_R, "RUSHERS (RB/FB designed carries)")


# ============================================================== (2a) FIP-style s -> s+1 results
def next_season_results(ev, kind):
    g = ev.groupby(["player_id", "season"])
    if kind == "T":
        o = pd.DataFrame({"n": g.size(), "ypt": g.recyds.mean(), "fd_t": g.recfd.mean(), "td_t": g.rectd.mean(),
                          "catch": g.complete_pass.mean()})
        o = o.join(use[["tg", "g", "team"]]); o["yds_pg"] = o.ypt * o.tg / o.g
    else:
        o = pd.DataFrame({"n": g.size(), "ypc": g.yards_gained.mean(), "fd_c": g.first_down.mean(),
                          "td_c": g.touchdown.mean(), "succ": g.succ.mean()})
        o = o.join(use[["cr", "g", "team"]]); o["yds_pg"] = o.ypc * o.cr / o.g
    return o


def fip_style(S, ev, kind, metrics, extra, min_n, label):
    nxt = next_season_results(ev, kind)
    outs = ["ypt", "fd_t", "td_t", "yds_pg"] if kind == "T" else ["ypc", "succ", "fd_c", "td_c", "yds_pg"]
    a = S[S.n >= min_n].join(use[["team"]]).reset_index()
    b = nxt[nxt.n >= min_n].reset_index()
    b["season"] -= 1
    j = a.merge(b.rename(columns={o: o + "_nx" for o in outs + ["team", "n"]}), on=["player_id", "season"])
    mov = j[j.team != j.team_nx]
    res = {"n_all": int(len(j)), "n_2010_15": int((j.season >= 2010).sum()), "n_team_changers": int(len(mov))}
    print(f"\n(2a) FIP-STYLE: season-s metric -> season-(s+1) RESULTS, {label} (both seasons >= {min_n}); "
          f"Pearson r, n={len(j)} (2010-15 pairs {res['n_2010_15']}); TEAM-CHANGERS n={len(mov)}")
    print(f"   {'metric (season s)':<30} " + " ".join(f"{o+'_s+1':>10}" for o in outs)
          + f" {'late '+outs[0]:>9} {'MOVERS '+outs[0]:>12} {'MOVERS '+outs[-1]:>13}")
    for nm in list(metrics) + extra:
        row = {o: round(corr(j[nm], j[o + "_nx"])[0], 3) for o in outs}
        jl = j[j.season >= 2010]
        row["late_" + outs[0]] = round(corr(jl[nm], jl[outs[0] + "_nx"])[0], 3)
        row["movers_" + outs[0]] = round(corr(mov[nm], mov[outs[0] + "_nx"])[0], 3)
        row["movers_" + outs[-1]] = round(corr(mov[nm], mov[outs[-1] + "_nx"])[0], 3)
        res[nm] = row
        print(f"   {nm:<30} " + " ".join(f"{row[o]:>10.3f}" for o in outs)
              + f" {row['late_'+outs[0]]:>9.3f} {row['movers_'+outs[0]]:>12.3f} {row['movers_'+outs[-1]]:>13.3f}")
    return res


OUT["fip_style_receiving"] = fip_style(REC_S, tgt, "T", REC_METRICS, EXT_T, 50, "RECEIVERS")
OUT["fip_style_rushing"] = fip_style(RUSH_S, car, "R", RUSH_METRICS, EXT_R, 100, "RUSHERS")

# ============================================================== (2b) walk-forward per-game
PG = PG[PG.season.between(DEV_LO + 1, DEV_HI)].copy()
PG["v_t"] = PG.mu_xypt + PG.mu_yptoe            # NEW value per target (yards above average)
PG["v_e"] = PG.mu_xepat + PG.mu_epat            # NEW value per target (EPA above average)
PG.to_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_player_games_dev_scored.parquet"), index=False)


def wf_pred(df, ycol, wcol, feats, lo=DEV_LO + 1):
    """walk-forward weighted OLS: fit on seasons < s, predict season s (s >= lo+1)."""
    p = np.full(len(df), np.nan)
    seas = df.season.to_numpy()
    yv = df[ycol].to_numpy(float); wv = df[wcol].to_numpy(float)
    Xa = np.column_stack([df[feats].to_numpy(float), np.ones(len(df))]) if feats else np.ones((len(df), 1))
    base = np.full(len(df), np.nan)
    for s in range(lo + 1, DEV_HI + 1):
        tr = seas < s; te = seas == s
        sw = np.sqrt(wv[tr])
        b = np.linalg.lstsq(Xa[tr] * sw[:, None], yv[tr] * sw, rcond=None)[0]
        p[te] = Xa[te] @ b
        base[te] = np.average(yv[tr], weights=wv[tr])
    return p, base


def r2(y, p, b, w, m):
    return 100 * (1 - np.sum(w[m] * (y[m] - p[m]) ** 2) / np.sum(w[m] * (y[m] - b[m]) ** 2))


def per_game_block(df, label, specs, outcomes, ref_pairs):
    res = {}
    print(f"\n(2b) WALK-FORWARD PER GAME, {label}: pre-game rating -> that game's results; OOS R^2 % "
          f"(walk-forward weighted OLS), 2008-15 | 2011-15")
    print(f"   {'rating (pre-game)':<36} " + " ".join(f"{o:>17}" for o in outcomes))
    se = {}
    seas = df.season.to_numpy()
    mall = seas >= DEV_LO + 2; mlate = seas >= 2011
    for nm, feats in specs.items():
        row = {}
        for o, (ycol, wcol) in outcomes.items():
            p, b = wf_pred(df, ycol, wcol, feats)
            y = df[ycol].to_numpy(float); w = df[wcol].to_numpy(float)
            row[o] = [round(r2(y, p, b, w, mall), 3), round(r2(y, p, b, w, mlate), 3)]
            se[(nm, o)] = w * (y - p) ** 2
        res[nm] = row
        print(f"   {nm:<36} " + " ".join(f"{row[o][0]:>8.3f}|{row[o][1]:>7.3f}" for o in outcomes))
    # game-clustered bootstrap of R^2-point differences (NEW minus comparator), 2008-15
    gid = df.game_id.to_numpy()
    ug, inv = np.unique(gid[mall], return_inverse=True)
    boots = {}
    for new, ref in ref_pairs:
        for o, (ycol, wcol) in outcomes.items():
            y = df[ycol].to_numpy(float); w = df[wcol].to_numpy(float)
            _, b = wf_pred(df, ycol, wcol, [])
            sst = (w * (y - b) ** 2)[mall]
            d = (se[(ref, o)] - se[(new, o)])[mall]
            dg = np.bincount(inv, weights=d); sg = np.bincount(inv, weights=sst)
            idx = RNG.integers(0, len(ug), size=(2000, len(ug)))
            bs = 100 * dg[idx].sum(1) / sg[idx].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boots[f"{new} vs {ref} | {o}"] = [round(100 * dg.sum() / sg.sum(), 3), round(lo, 3), round(hi, 3)]
    print("   R^2-point difference, NEW minus comparator [95% CI, game-clustered]:")
    for k, v in boots.items():
        print(f"     {k:<80} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")
    res["_bootstrap"] = boots
    return res


rec = PG[PG.pg.isin(REC_PG) & (PG.y_tgt > 0)].copy()
rec["ypt"] = rec.y_recyds / rec.y_tgt; rec["fdt"] = rec.y_recfd / rec.y_tgt; rec["tdt"] = rec.y_rectd / rec.y_tgt
rec["one"] = 1.0; rec["ept"] = rec.y_epat / rec.y_tgt
rec = rec.dropna(subset=["rate6"])  # identical row set for every rating (rate6 needs a prior weekly line)
OUT["per_game_receiving_n"] = int(len(rec))
REC_SPECS = {
    "naive EB yds/target": ["mu_rawt"],
    "shipped rate6 SK composite": ["rate6"],
    "shared-credit proxy (11v11 analog)": ["mu_shared"],
    "NEW xYPT opportunity quality": ["mu_xypt"],
    "NEW YPTOE execution": ["mu_yptoe"],
    "NEW ypt adj (QB+def, one number)": ["mu_ypta"],
    "NEW CROE": ["mu_comp"],
    "NEW YACOE": ["mu_yac"],
    "NEW EPAOE/target": ["mu_epat"],
    "NEW v_t = xYPT + YPTOE": ["v_t"],
    "NEW xEPA opportunity (EPA units)": ["mu_xepat"],
    "NEW v_e = xEPA + EPAOE": ["v_e"],
    "NEW xYPT, YPTOE (2 cols)": ["mu_xypt", "mu_yptoe"],
    "rate6 + NEW xYPT, YPTOE": ["rate6", "mu_xypt", "mu_yptoe"],
    "naive + NEW xYPT, YPTOE": ["mu_rawt", "mu_xypt", "mu_yptoe"],
}
REC_OUT = {"yds/target": ("ypt", "y_tgt"), "1st downs/target": ("fdt", "y_tgt"), "TD/target": ("tdt", "y_tgt"),
           "EPA/target (proxy)": ("ept", "y_tgt")}
OUT["per_game_receiving"] = per_game_block(
    rec, f"RECEIVERS n={len(rec):,} player-games", REC_SPECS, REC_OUT,
    [("NEW xYPT, YPTOE (2 cols)", "shipped rate6 SK composite"), ("NEW xYPT, YPTOE (2 cols)", "naive EB yds/target"),
     ("NEW xYPT, YPTOE (2 cols)", "shared-credit proxy (11v11 analog)"),
     ("NEW v_e = xEPA + EPAOE", "shipped rate6 SK composite"), ("NEW v_e = xEPA + EPAOE", "naive EB yds/target")])

ru = PG[PG.pg.isin(RUSH_PG) & (PG.y_car > 0)].copy()
ru["ypc"] = ru.y_rushyds / ru.y_car; ru["fdc"] = ru.y_rushfd / ru.y_car; ru["tdc"] = ru.y_rushtd / ru.y_car
ru["sc"] = ru.y_rsucc / ru.y_car
ru = ru.dropna(subset=["rate6"])
OUT["per_game_rushing_n"] = int(len(ru))
RUSH_SPECS = {
    "naive EB yds/carry": ["mu_rawr"],
    "shipped rate6 SK composite": ["rate6"],
    "shared-credit proxy (11v11 analog)": ["mu_shared"],
    "NEW RYOE": ["mu_ryds"],
    "NEW SOE": ["mu_rsuc"],
    "NEW EPAOE/carry": ["mu_epar"],
    "NEW ypc adj (line+def, one number)": ["mu_ypca"],
    "NEW RYOE, SOE (2 cols)": ["mu_ryds", "mu_rsuc"],
    "rate6 + NEW RYOE, SOE": ["rate6", "mu_ryds", "mu_rsuc"],
    "naive + NEW RYOE, SOE": ["mu_rawr", "mu_ryds", "mu_rsuc"],
}
RUSH_OUT = {"yds/carry": ("ypc", "y_car"), "1st downs/carry": ("fdc", "y_car"), "TD/carry": ("tdc", "y_car"),
            "success": ("sc", "y_car")}
OUT["per_game_rushing"] = per_game_block(
    ru, f"RUSHERS n={len(ru):,} player-games", RUSH_SPECS, RUSH_OUT,
    [("NEW RYOE, SOE (2 cols)", "shipped rate6 SK composite"), ("NEW RYOE, SOE (2 cols)", "naive EB yds/carry"),
     ("NEW RYOE, SOE (2 cols)", "shared-credit proxy (11v11 analog)")])

# production per game (usage x efficiency): receiving yards in the game
for a, b in (("u_rawt", "mu_rawt"), ("u_rate6", "rate6"), ("u_vt", "v_t"), ("u_xypt", "mu_xypt"),
             ("u_yptoe", "mu_yptoe"), ("u_shared", "mu_shared")):
    rec[a] = rec.pre_tsh * rec[b]
OUT["per_game_receiving_production"] = per_game_block(
    rec, "RECEIVING YARDS PER GAME (usage x efficiency)",
    {"usage only (pre-game target share)": ["pre_tsh"],
     "usage x naive yds/target": ["pre_tsh", "u_rawt"],
     "usage x rate6": ["pre_tsh", "u_rate6"],
     "rate6 alone": ["rate6"],
     "usage x shared proxy": ["pre_tsh", "u_shared"],
     "usage x NEW xYPT, YPTOE": ["pre_tsh", "u_xypt", "u_yptoe"],
     "usage x NEW v_t": ["pre_tsh", "u_vt"]},
    {"rec yds/game": ("y_recyds", "one")},
    [("usage x NEW xYPT, YPTOE", "usage x rate6"), ("usage x NEW xYPT, YPTOE", "usage x naive yds/target")])

# ============================================================== (2c) unit level: offense points
G = pd.read_csv(os.path.join(DATA, "nfl_games.csv"), usecols=["game_id", "season", "home_score", "away_score"])
G = G[G.season <= DEV_HI]
assert G.season.max() <= DEV_HI
pts = {}
for r in G.itertuples(index=False):
    pts[(r.game_id, "H")] = r.home_score; pts[(r.game_id, "A")] = r.away_score
EVH = pd.read_parquet(os.path.join(DATA, "pv_nfl_events.parquet"), columns=["game_id", "season", "home_team"],
                      filters=[("season", "<=", DEV_HI)]).drop_duplicates("game_id")
HOME = dict(zip(EVH.game_id, EVH.home_team))
PGu = PG[PG.pg.isin(REC_PG)].copy()
PGu["side"] = np.where(PGu.team == PGu.game_id.map(HOME), "H", "A")
PGu["rate6f"] = PGu.rate6.fillna(0.0)
TGT_PG, CAR_PG = 34.0, 26.0  # typical team targets / designed carries per game (scale only)
unit = PGu.assign(
    new_t=PGu.pre_tsh * TGT_PG * PGu.v_t, new_r=PGu.pre_csh * CAR_PG * PGu.mu_ypca,
    new_rr=PGu.pre_csh * CAR_PG * PGu.mu_ryds,
    new_te=PGu.pre_tsh * TGT_PG * PGu.v_e, new_re=PGu.pre_csh * CAR_PG * PGu.mu_epar,
    naive_t=PGu.pre_tsh * TGT_PG * PGu.mu_rawt, naive_r=PGu.pre_csh * CAR_PG * PGu.mu_rawr,
    r6=(PGu.pre_tsh + PGu.pre_csh) * PGu.rate6f,
    shared=(PGu.pre_tsh + PGu.pre_csh) * PGu.mu_shared,
).groupby(["game_id", "season", "gidx", "team", "side"])[
    ["new_t", "new_r", "new_rr", "new_te", "new_re", "naive_t", "naive_r", "r6", "shared"]].sum().reset_index()
unit["pts"] = [pts.get((g, s), np.nan) for g, s in zip(unit.game_id, unit.side)]
unit = unit.dropna(subset=["pts"])
unit = unit.merge(unit[["game_id", "team"]].rename(columns={"team": "opp"}), on="game_id")
unit = unit[unit.team != unit.opp].sort_values(["gidx", "team"]).reset_index(drop=True)
tepa = EVall.groupby(["game_id", "posteam"]).epa.mean().to_dict()
unit["epa_g"] = [tepa.get((g, t), np.nan) for g, t in zip(unit.game_id, unit.team)]
# as-of team model: EWMA (per game, decay .9, EB prior 4 games) of points for / against and EPA/play for / against
st = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # team -> [pts_for_sum, w, epa_for_sum, w] ; defense uses 'D:'
feats = {k: np.zeros(len(unit)) for k in ("off_pts", "def_pts", "off_epa", "def_epa")}
LGP, LGE, DECT, PRN = 21.5, 0.0, 0.9, 4.0
for g, d in unit.groupby("gidx", sort=True):
    for i, r in zip(d.index, d.itertuples()):
        o = st[r.team]; q = st["D:" + r.opp]
        feats["off_pts"][i] = (o[0] + PRN * LGP) / (o[1] + PRN); feats["off_epa"][i] = (o[2] + PRN * LGE) / (o[3] + PRN)
        feats["def_pts"][i] = (q[0] + PRN * LGP) / (q[1] + PRN); feats["def_epa"][i] = (q[2] + PRN * LGE) / (q[3] + PRN)
    for r in d.itertuples():
        for key in (r.team, "D:" + r.opp):
            s_ = st[key]
            s_[0] = DECT * s_[0] + r.pts; s_[1] = DECT * s_[1] + 1
            s_[2] = DECT * s_[2] + (r.epa_g if r.epa_g == r.epa_g else 0.0); s_[3] = DECT * s_[3] + 1
for k, v in feats.items():
    unit[k] = v
unit["home"] = (unit.side == "H").astype(float); unit["one"] = 1.0
BASE = ["off_pts", "def_pts", "off_epa", "def_epa", "home"]
print(f"\n(2c) UNIT LEVEL: offense POINTS in the game, n={len(unit):,} team-games; walk-forward OLS OOS R^2 %, "
      f"2008-15 | 2011-15; base = as-of points + team EPA/play EWMAs (off, opp def) + home")
UNIT_SPECS = {"as-of team model (base)": BASE,
              "base + naive skill aggregates": BASE + ["naive_t", "naive_r"],
              "base + shipped rate6 aggregate": BASE + ["r6"],
              "base + shared-credit aggregate": BASE + ["shared"],
              "base + NEW aggregates (v_t, ypc adj)": BASE + ["new_t", "new_r"],
              "base + NEW aggregates (v_t, RYOE)": BASE + ["new_t", "new_rr"],
              "base + NEW EPA aggregates (v_e, EPAOE/c)": BASE + ["new_te", "new_re"],
              "base + rate6 + NEW": BASE + ["r6", "new_t", "new_r"]}
yv = unit.pts.to_numpy(float); wv = np.ones(len(unit)); seas = unit.season.to_numpy()
mall = seas >= DEV_LO + 2; mlate = seas >= 2011
ures, useq = {}, {}
for nm, f in UNIT_SPECS.items():
    p, b = wf_pred(unit, "pts", "one", f)
    ures[nm] = [round(r2(yv, p, b, wv, mall), 3), round(r2(yv, p, b, wv, mlate), 3)]
    useq[nm] = (yv - p) ** 2
    print(f"   {nm:<42} {ures[nm][0]:>7.3f} | {ures[nm][1]:>7.3f}")
ug, inv = np.unique(unit.game_id.to_numpy()[mall], return_inverse=True)
ub = {}
for nm in UNIT_SPECS:
    if nm == "as-of team model (base)":
        continue
    d = (useq["as-of team model (base)"] - useq[nm])[mall]
    dg = np.bincount(inv, weights=d)
    bs = dg[RNG.integers(0, len(ug), size=(4000, len(ug)))].sum(1) / mall.sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ub[nm] = [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]
print("   MSE gain vs base, points^2 per team-game [95% CI, game-clustered]:")
for k, v in ub.items():
    print(f"     {k:<42} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]")
OUT["unit_points"] = {"r2": ures, "mse_gain_vs_base": ub, "n": int(len(unit)), "n_scored_2008_15": int(mall.sum())}


# ============================================================== (3) face validity
def face(season, kind, grp=None):
    d = PG[(PG.season == season)].sort_values("gidx").groupby("player_id").tail(1).copy()
    d["name"] = d.player_id.map(NAME); d["pos"] = d.player_id.map(POS)
    if kind == "T":
        d = d[(d.pg == grp) & (d.n_xepat >= 120)]
        d["epa_pg_over_avg"] = d.v_e * d.pre_tsh * TGT_PG
        cols = ["name", "team", "n_xepat", "mu_xepat", "mu_epat", "v_e", "mu_xypt", "mu_yptoe", "pre_tsh",
                "epa_pg_over_avg", "rate6"]
        d = d.sort_values("v_e", ascending=False)
    else:
        d = d[d.pg.isin(RUSH_PG) & (d.n_ryds >= 200)]
        d["yds_pg_over_avg"] = d.mu_ryds * d.pre_csh * CAR_PG
        cols = ["name", "pos", "team", "n_ryds", "mu_ryds", "mu_rsuc", "mu_ypca", "mu_rawr", "pre_csh",
                "yds_pg_over_avg", "rate6"]
        d = d.sort_values("mu_ryds", ascending=False)
    return d[cols]


FACE = {}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
for season in (2010, 2015):
    for kind, grp, lab in (("T", "WR", "WIDE RECEIVERS"), ("T", "TE", "TIGHT ENDS"), ("T", "RB", "RB AS RECEIVERS"),
                           ("R", None, "RUSHERS by RYOE (capped yds/carry over expected, line+def adjusted)")):
        d = face(season, kind, grp)
        k = 10 if kind == "T" else 12
        print(f"\n(3) FACE VALIDITY {season} (last pre-game state) - {lab}"
              + (" by v_e = xEPA + EPAOE (EPA/target above avg, QB+def adjusted)" if kind == "T" else "")
              + f"; n rated {len(d)}")
        print(d.head(k).round(3).to_string(index=False))
        print("   ... bottom:")
        print(d.tail(5).round(3).to_string(index=False))
        FACE[f"{season}_{kind}_{grp}"] = {"top": d.head(15).round(4).to_dict("records"),
                                          "bottom": d.tail(8).round(4).to_dict("records")}
OUT["face"] = FACE
json.dump(OUT, open(os.path.join(DATA, "pv_nfl_receiving_rushing_validation.json"), "w"), indent=1, default=str)
print("\nwrote data/pv_nfl_receiving_rushing_validation.json")
