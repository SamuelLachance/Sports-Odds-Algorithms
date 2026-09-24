"""Player-value program (pv), NFL, COMPOSE step 3: does the composed lineup value predict future team
performance better than the 11v11 roster rating? DEV ONLY (2006-2015; 1999-2005 only as training).

Comparators on DEV (the shipped col 7 v7 TrueSkill is identically 0 on every DEV game -- its
participation data start in 2016 -- so it cannot be scored without touching TEST):
  TS   pv_nfl_compose_tsdev: the 11v11 shared-credit TrueSkill analog on the SAME lineups
       (snap-share participation 2013-15, credited players before), lineup and expected variants
  RQ   shipped col 11 roster_quality (rate6 x pre-game snap share over the game's snap table),
       non-zero 2013-2015 only -- the shipped DEV-era roster rating
  BASE the shipped 14-column blend's information, as a walk-forward OLS margin predictor
Candidates:
  LV   composed lineup value (points/game, as-of weights), LVX (no day-of info), QV (QB only),
       LV-QV (non-QB lineup)

Tests
  A  game margin (home - away), every DEV game: Pearson r; walk-forward OOS MSE of one-variable
     calibrations (fit on seasons < s); gain beyond BASE; partial r beyond BASE; paired game bootstrap.
  B  NEXT PERIOD: team rating at its first game after week 4 / 8 / 12 vs its rest-of-regular-season
     margin per game (and EPA/play net), raw and partial beyond to-date margin and to-date EPA/play;
     and preseason (first game of season s, roster carried + actual starter) vs season-s margin.
     Team-season cluster bootstrap.
  C  snap era 2013-15: LV vs LVX vs RQ vs TS; does the day-of MISS term carry signal beyond LVX
     and BASE; persistence of absences (injury, not game script).
  D  leak probes: team rating vs this game's own margin vs its next game's margin.
  E  player level: year-over-year stability of player-season values by position (stayers /
     movers), and face validity (top / bottom by position).
Output: data/pv_nfl_compose_validation.json (+ printed tables). No odds read. No TEST season.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
RNG = np.random.default_rng(20260925)
NB = 2000
OUT = {}


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def r_(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def resid(y, Z):
    Z1 = np.column_stack([np.ones(len(y)), Z])
    b = np.linalg.lstsq(Z1, y, rcond=None)[0]
    return y - Z1 @ b


def boot_idx(n, clusters=None):
    if clusters is None:
        return RNG.integers(0, n, size=(NB, n))
    u, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(u))]
    out = []
    for _ in range(NB):
        pick = RNG.integers(0, len(u), len(u))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def main():
    F = pd.read_csv(os.path.join(DATA, "pv_nfl_compose_game_features.csv"))
    assert F.season.max() < TEST_ERA
    Oc = pd.read_csv(os.path.join(DATA, "pv_nfl_compose_game_outcomes.csv"))
    assert Oc.season.max() < TEST_ERA
    F = F.merge(Oc[["game_id", "margin", "epa_margin"]], on="game_id", how="left")
    D = json.load(open(os.path.join(DATA, "bt_nfl_ideas_dev.json"), encoding="utf-8"))
    X14 = np.load(os.path.join(DATA, "bt_nfl_ideas_dev_X.npy"))
    gidx = {g["gid"]: i for i, g in enumerate(D["games"])}
    assert max(g["season"] for g in D["games"]) < TEST_ERA
    F = F[F.game_id.isin(gidx)].reset_index(drop=True)
    XI = X14[[gidx[g] for g in F.game_id]]
    F["rq"] = XI[:, 11]
    F["d_lv"] = F.lv_home - F.lv_away; F["d_lvx"] = F.lvx_home - F.lvx_away
    F["d_qv"] = F.qv_home - F.qv_away; F["d_nq"] = F.d_lv - F.d_qv
    F["d_miss"] = F.miss_home - F.miss_away
    F["d_ts"] = F.ts_home - F.ts_away; F["d_tsx"] = F.tsx_home - F.tsx_away
    F["h"] = 1.0 - F.neutral
    S = F.season.to_numpy()
    y = F.margin.to_numpy(float)

    # ---------------------------------------------------------------- BASE: walk-forward OLS of margin on X14
    base = np.full(len(F), np.nan)
    for s in range(DEV_LO, DEV_HI + 1):
        tr = S < s; te = S == s
        Z = np.column_stack([np.ones(tr.sum()), XI[tr]])
        b = np.linalg.lstsq(Z, y[tr], rcond=None)[0]
        base[te] = np.column_stack([np.ones(te.sum()), XI[te]]) @ b
    F["base"] = base
    dev = (S >= DEV_LO)
    FD = F[dev].reset_index(drop=True)
    yD = FD.margin.to_numpy(float)
    log(f"DEV games {len(FD)}; BASE walk-forward margin r {r_(FD.base, yD):.4f}")

    # ---------------------------------------------------------------- A. game margin
    cands = {"LV (lineup value, pts)": "d_lv", "LVX (no day-of info)": "d_lvx", "QV (starter QB)": "d_qv",
             "LV - QV (non-QB lineup)": "d_nq", "TS (11v11 analog, lineup)": "d_ts",
             "TSX (11v11 analog, expected)": "d_tsx"}
    A = {}

    def wf_mse(cols, extra_base=False):
        """walk-forward OLS margin ~ h + cols (+ BASE prediction), train seasons 2000..s-1 -> DEV errors."""
        err = np.zeros(len(F)) * np.nan
        for s in range(DEV_LO, DEV_HI + 1):
            tr = (S >= 2000) & (S < s); te = S == s
            M = [F.h.to_numpy()] + [F[c].to_numpy(float) for c in cols]
            if extra_base:
                # BASE is itself walk-forward only on DEV rows; for training use in-sample OLS on X14
                M = M + [XI[:, j] for j in range(14)]
            Z = np.column_stack([np.ones(len(F))] + M)
            b = np.linalg.lstsq(Z[tr], y[tr], rcond=None)[0]
            err[te] = y[te] - Z[te] @ b
        return err[dev] ** 2

    e_h = wf_mse([])
    e_base = wf_mse([], extra_base=True)
    rows = {}
    for nm, c in cands.items():
        x = FD[c].to_numpy(float)
        e1 = wf_mse([c]); e2 = wf_mse([c], extra_base=True)
        rb = resid(yD, FD.base.to_numpy()); xb = resid(x, FD.base.to_numpy())
        rows[nm] = {"r_margin": round(r_(x, yD), 4), "oos_mse_gain_vs_hfa_only": round(float(e_h.mean() - e1.mean()), 3),
                    "oos_mse_gain_beyond_base": round(float(e_base.mean() - e2.mean()), 3),
                    "partial_r_beyond_base": round(r_(xb, rb), 4),
                    "seasons_r": {str(s): round(r_(FD[c][FD.season == s], FD.margin[FD.season == s]), 3)
                                  for s in range(DEV_LO, DEV_HI + 1)}}
        A[c] = (e1, e2, xb)
    rb = resid(yD, FD.base.to_numpy())
    # paired bootstraps: LV vs TS (and LVX vs TSX)
    bi = boot_idx(len(FD))
    pair = {}
    for a_, b_ in (("d_lv", "d_ts"), ("d_lvx", "d_tsx"), ("d_nq", "d_ts")):
        xa, xb_ = FD[a_].to_numpy(float), FD[b_].to_numpy(float)
        dr = [r_(xa[i], yD[i]) - r_(xb_[i], yD[i]) for i in bi]
        dm = [A[b_][0][i].mean() - A[a_][0][i].mean() for i in bi]
        dm2 = [A[b_][1][i].mean() - A[a_][1][i].mean() for i in bi]
        dp = [r_(A[a_][2][i], rb[i]) - r_(A[b_][2][i], rb[i]) for i in bi]
        pair[f"{a_} minus {b_}"] = {"r_diff": round(r_(xa, yD) - r_(xb_, yD), 4), "r_diff_ci": ci(dr),
                                    "oos_mse_gain_diff": round(float(A[b_][0].mean() - A[a_][0].mean()), 3),
                                    "oos_mse_gain_diff_ci": ci(dm),
                                    "beyond_base_mse_gain_diff": round(float(A[b_][1].mean() - A[a_][1].mean()), 3),
                                    "beyond_base_mse_gain_diff_ci": ci(dm2),
                                    "partial_r_diff": round(r_(A[a_][2], rb) - r_(A[b_][2], rb), 4),
                                    "partial_r_diff_ci": ci(dp),
                                    "seasons_a_better_r": int(sum(r_(FD[a_][FD.season == s], FD.margin[FD.season == s])
                                                                  > r_(FD[b_][FD.season == s], FD.margin[FD.season == s])
                                                                  for s in range(DEV_LO, DEV_HI + 1)))}
    # both in one model beyond BASE (who carries the signal)
    Zj = np.column_stack([FD.base, FD.d_lv, FD.d_ts])
    Zs = (Zj - Zj.mean(0)) / Zj.std(0)
    bj = np.linalg.lstsq(np.column_stack([np.ones(len(FD)), Zs]), yD, rcond=None)[0]
    bb = []
    for i in bi[:1000]:
        bb.append(np.linalg.lstsq(np.column_stack([np.ones(len(i)), Zs[i]]), yD[i], rcond=None)[0][1:])
    bb = np.array(bb)
    joint = {"coef_per_sd_pts": {"BASE": round(bj[1], 3), "LV": round(bj[2], 3), "TS": round(bj[3], 3)},
             "ci": {"BASE": ci(bb[:, 0]), "LV": ci(bb[:, 1]), "TS": ci(bb[:, 2])},
             "corr_LV_TS": round(r_(FD.d_lv, FD.d_ts), 3), "corr_LV_BASE": round(r_(FD.d_lv, FD.base), 3),
             "corr_TS_BASE": round(r_(FD.d_ts, FD.base), 3)}
    # robustness: the same composition with the EPA-differential-target weights (as-of folds)
    WJ = json.load(open(os.path.join(DATA, "pv_nfl_compose_weights.json")))["weights_by_fold_epa_target"]
    units = ["QB", "REC", "RUSH", "PROT", "PR", "RF", "BALL", "COV"]
    lve = np.array([sum(WJ[str(s)][u] * (a_h - a_a) for u, a_h, a_a in
                        zip(units, [r[f"A_{u}_home"] for u in units], [r[f"A_{u}_away"] for u in units]))
                    for s, r in zip(FD.season, FD.to_dict("records"))])
    rows["LV, EPA-target weights (robustness)"] = {
        "r_margin": round(r_(lve, yD), 4),
        "partial_r_beyond_base": round(r_(resid(lve, FD.base.to_numpy()), rb), 4),
        "r_with_LV": round(r_(lve, FD.d_lv), 4)}
    OUT["A_game_margin"] = {"n_games": int(len(FD)), "base_r": round(r_(FD.base, yD), 4),
                            "hfa_only_mse": round(float(e_h.mean()), 3), "base_mse": round(float(e_base.mean()), 3),
                            "candidates": rows, "paired": pair, "joint_beyond_base": joint}
    log("A. game margin (DEV 2006-2015, n=%d)" % len(FD))
    for nm, v in rows.items():
        if "oos_mse_gain_vs_hfa_only" not in v:
            log(f"   {nm:<32} {v}")
            continue
        log(f"   {nm:<32} r {v['r_margin']:+.4f}  OOS MSE gain vs HFA {v['oos_mse_gain_vs_hfa_only']:+7.3f}  "
            f"beyond BASE {v['oos_mse_gain_beyond_base']:+6.3f}  partial r {v['partial_r_beyond_base']:+.4f}")
    for k, v in pair.items():
        log(f"   {k:<22} r diff {v['r_diff']:+.4f} {v['r_diff_ci']} | OOS gain diff {v['oos_mse_gain_diff']:+.3f} "
            f"{v['oos_mse_gain_diff_ci']} | beyond BASE {v['beyond_base_mse_gain_diff']:+.3f} "
            f"{v['beyond_base_mse_gain_diff_ci']} | partial r diff {v['partial_r_diff']:+.4f} {v['partial_r_diff_ci']} "
            f"| seasons {v['seasons_a_better_r']}/10")
    log("   joint beyond BASE:", joint)

    # ---------------------------------------------------------------- B. next period
    TG = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_team_games.parquet"))
    assert TG.season.max() < TEST_ERA
    TSg = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_tsdev_team_games.parquet"))
    long = []
    for sd in ("home", "away"):
        o = "away" if sd == "home" else "home"
        sub = F[["game_id", "season", "week", "season_type", f"lv_{sd}", f"lvx_{sd}", f"qv_{sd}", f"ts_{sd}", f"tsx_{sd}"]].copy()
        sub.columns = ["game_id", "season", "week", "season_type", "lv", "lvx", "qv", "ts", "tsx"]
        sub["team"] = F[sd]
        sub["margin"] = F.margin * (1 if sd == "home" else -1)
        long.append(sub)
    L = pd.concat(long, ignore_index=True)
    ep = TG[["game_id", "team", "epa_off", "n_play"]]
    L = L.merge(ep, on=["game_id", "team"], how="left")
    opp = TG[["game_id", "team", "opp"]]
    L = L.merge(opp, on=["game_id", "team"], how="left")
    ep2 = TG[["game_id", "team", "epa_off", "n_play"]].rename(columns={"team": "opp", "epa_off": "epa_def", "n_play": "n_play_def"})
    L = L.merge(ep2, on=["game_id", "opp"], how="left")
    L["epa_net_pp"] = L.epa_off / L.n_play - L.epa_def / L.n_play_def
    L = L[(L.season_type == "REG")].sort_values(["team", "season", "week"])
    B = {}
    recs = []
    for (t, s), d in L[L.season.between(DEV_LO, DEV_HI)].groupby(["team", "season"]):
        for k in (4, 8, 12):
            past = d[d.week <= k]; fut = d[d.week > k]
            if len(past) < 3 or len(fut) < 3:
                continue
            g0 = fut.iloc[0]
            recs.append({"team": t, "season": s, "k": k, "lv": g0.lv, "lvx": g0.lvx, "qv": g0.qv, "ts": g0.ts,
                         "tsx": g0.tsx, "todate_margin": past.margin.mean(), "todate_epa": past.epa_net_pp.mean(),
                         "fut_margin": fut.margin.mean(), "fut_epa": fut.epa_net_pp.mean(), "n_fut": len(fut)})
    R = pd.DataFrame(recs)
    cl = (R.team + R.season.astype(str)).to_numpy()
    bi2 = boot_idx(len(R), cl)
    for tgt in ("fut_margin", "fut_epa"):
        yy = R[tgt].to_numpy(float)
        ctrl = R[["todate_margin", "todate_epa"]].to_numpy(float)
        ry = resid(yy, ctrl)
        res = {}
        for c in ("lv", "lvx", "qv", "ts", "tsx", "todate_margin"):
            x = R[c].to_numpy(float)
            res[c] = {"r": round(r_(x, yy), 4),
                      "partial_r_beyond_todate": round(r_(resid(x, ctrl), ry), 4) if c != "todate_margin" else None}
        for a_, b_ in (("lvx", "tsx"), ("lv", "ts")):
            xa, xb_ = R[a_].to_numpy(float), R[b_].to_numpy(float)
            ra, rbb = resid(xa, ctrl), resid(xb_, ctrl)
            d1 = [r_(xa[i], yy[i]) - r_(xb_[i], yy[i]) for i in bi2]
            d2 = [r_(ra[i], ry[i]) - r_(rbb[i], ry[i]) for i in bi2]
            res[f"{a_}-{b_}"] = {"r_diff": round(r_(xa, yy) - r_(xb_, yy), 4), "r_diff_ci": ci(d1),
                                 "partial_r_diff": round(r_(ra, ry) - r_(rbb, ry), 4), "partial_r_diff_ci": ci(d2)}
        by_k = {}
        for k in (4, 8, 12):
            m = R.k == k
            by_k[str(k)] = {c: round(r_(R.loc[m, c], R.loc[m, tgt]), 3) for c in ("lvx", "tsx", "todate_margin")}
        res["by_checkpoint_r"] = by_k
        B[tgt] = res
    # preseason: first game of season s -> season-s margin per game
    pre = []
    for (t, s), d in L[L.season.between(DEV_LO, DEV_HI)].groupby(["team", "season"]):
        g0 = d.iloc[0]
        pre.append({"team": t, "season": s, "lvx": g0.lvx, "lv": g0.lv, "tsx": g0.tsx, "ts": g0.ts, "qv": g0.qv,
                    "season_margin": d.margin.mean(), "season_epa": d.epa_net_pp.mean()})
    P = pd.DataFrame(pre)
    pres = {}
    bi3 = boot_idx(len(P))
    for tgt in ("season_margin", "season_epa"):
        yy = P[tgt].to_numpy(float)
        pres[tgt] = {c: round(r_(P[c], yy), 4) for c in ("lvx", "lv", "qv", "tsx", "ts")}
        d1 = [r_(P.lvx.to_numpy()[i], yy[i]) - r_(P.tsx.to_numpy()[i], yy[i]) for i in bi3]
        pres[tgt]["lvx-tsx"] = round(pres[tgt]["lvx"] - pres[tgt]["tsx"], 4); pres[tgt]["lvx-tsx_ci"] = ci(d1)
    B["preseason_week1_to_season"] = {"n_team_seasons": int(len(P)), **pres}
    # offseason roster change: (rating at the first game of s) - (rating at the last REG game of s-1)
    # vs (margin_s - margin_{s-1}), controlling for margin_{s-1} and the s-1 end rating. A rating that
    # measures PLAYERS should move with the roster; a rating that is team memory should not.
    last = L.groupby(["team", "season"]).tail(1).set_index(["team", "season"])
    first = L.groupby(["team", "season"]).head(1).set_index(["team", "season"])
    seas_m = L.groupby(["team", "season"]).margin.mean()
    rc = []
    for (t, s) in first.index:
        if not (DEV_LO + 1 <= s <= DEV_HI) or (t, s - 1) not in last.index:
            continue
        rec = {"team": t, "season": s, "dm": seas_m[(t, s)] - seas_m[(t, s - 1)], "m_prev": seas_m[(t, s - 1)]}
        for c in ("lvx", "tsx", "qv"):
            rec["d_" + c] = first.loc[(t, s), c] - last.loc[(t, s - 1), c]
            rec["end_" + c] = last.loc[(t, s - 1), c]
        rc.append(rec)
    RC = pd.DataFrame(rc)
    yy = RC.dm.to_numpy(float)
    roster = {"n_team_seasons": int(len(RC))}
    for c in ("lvx", "tsx", "qv"):
        ctrl = RC[["m_prev", "end_" + c]].to_numpy(float)
        roster[c] = {"r_delta_vs_dmargin": round(r_(RC["d_" + c], yy), 4),
                     "partial_r": round(r_(resid(RC["d_" + c].to_numpy(float), ctrl), resid(yy, ctrl)), 4),
                     "sd_delta": round(float(RC["d_" + c].std()), 4)}
    bi5 = boot_idx(len(RC))
    ca = RC[["m_prev", "end_lvx"]].to_numpy(float); cb = RC[["m_prev", "end_tsx"]].to_numpy(float)
    xa, xb_ = resid(RC.d_lvx.to_numpy(float), ca), resid(RC.d_tsx.to_numpy(float), cb)
    ya, yb = resid(yy, ca), resid(yy, cb)
    dd = [r_(xa[i], ya[i]) - r_(xb_[i], yb[i]) for i in bi5]
    roster["lvx_minus_tsx_partial_r"] = round(r_(xa, ya) - r_(xb_, yb), 4); roster["ci"] = ci(dd)
    # the non-QB part of the roster change (QB changes are the known big lever)
    RC["d_nqx"] = RC.d_lvx - RC.d_qv; RC["end_nqx"] = RC.end_lvx - RC.end_qv
    ctrl = RC[["m_prev", "end_nqx", "d_qv"]].to_numpy(float)
    roster["non_qb_delta_partial_r_beyond_qb_delta"] = round(r_(resid(RC.d_nqx.to_numpy(float), ctrl), resid(yy, ctrl)), 4)
    B["offseason_roster_change"] = roster
    B["n_checkpoints"] = int(len(R))
    OUT["B_next_period"] = B
    log(f"B. next period (n={len(R)} team-checkpoints, {R.groupby(['team','season']).ngroups} team-seasons)")
    for tgt in ("fut_margin", "fut_epa"):
        v = B[tgt]
        log(f"   {tgt}: " + " ".join(f"{c} r {v[c]['r']:+.3f}/p {v[c]['partial_r_beyond_todate']}" for c in
                                    ("lv", "lvx", "qv", "ts", "tsx")) + f" | todate_margin r {v['todate_margin']['r']:+.3f}")
        log(f"      LVX-TSX {v['lvx-tsx']}  LV-TS {v['lv-ts']}  by checkpoint {v['by_checkpoint_r']}")
    log("   preseason -> season:", B["preseason_week1_to_season"])
    log("   offseason roster change -> change in margin:", B["offseason_roster_change"])

    # ---------------------------------------------------------------- C. snap era
    SE = FD[FD.season >= 2013].reset_index(drop=True)
    ySE = SE.margin.to_numpy(float)
    C = {"n_games": int(len(SE))}
    for c in ("d_lv", "d_lvx", "d_miss", "d_ts", "d_tsx", "rq", "d_nq"):
        C[c] = {"r_margin": round(r_(SE[c], ySE), 4),
                "partial_r_beyond_base": round(r_(resid(SE[c].to_numpy(float), SE.base.to_numpy()),
                                                  resid(ySE, SE.base.to_numpy())), 4)}
    XS = XI[dev][FD.season.to_numpy() >= 2013]
    ctrl = np.column_stack([SE.base, SE.d_lvx])
    C["miss_partial_r_beyond_base_and_lvx"] = round(r_(resid(SE.d_miss.to_numpy(float), ctrl), resid(ySE, ctrl)), 4)
    ctrl2 = np.column_stack([SE.base, SE.d_lvx, XS[:, 8], XS[:, 9], XS[:, 10], XS[:, 11]])
    C["miss_partial_r_beyond_base_lvx_and_shipped_absence_rq"] = round(
        r_(resid(SE.d_miss.to_numpy(float), ctrl2), resid(ySE, ctrl2)), 4)
    bi4 = boot_idx(len(SE))
    for a_, b_ in (("d_lv", "rq"), ("d_nq", "rq"), ("d_lv", "d_ts"), ("d_miss", "rq")):
        xa, xb_ = SE[a_].to_numpy(float), SE[b_].to_numpy(float)
        rbS = resid(ySE, SE.base.to_numpy())
        ra, rbb = resid(xa, SE.base.to_numpy()), resid(xb_, SE.base.to_numpy())
        d1 = [r_(xa[i], ySE[i]) - r_(xb_[i], ySE[i]) for i in bi4]
        d2 = [r_(ra[i], rbS[i]) - r_(rbb[i], rbS[i]) for i in bi4]
        C[f"{a_} minus {b_}"] = {"r_diff": round(r_(xa, ySE) - r_(xb_, ySE), 4), "r_diff_ci": ci(d1),
                                 "partial_r_diff": round(r_(ra, rbS) - r_(rbb, rbS), 4), "partial_r_diff_ci": ci(d2)}
    # absence persistence: a regular absent now -> absent (or still not back) in the team's next game
    EM = pd.read_parquet(os.path.join(DATA, "pv_nfl_compose_lineups.parquet"))
    assert EM.season.max() < TEST_ERA
    EM = EM[EM.season >= 2013]
    order = TG[["game_id", "team", "game_date"]]
    EM = EM.merge(order, on=["game_id", "team"])
    nxt = {}
    for t, d in order[order.game_id.isin(EM.game_id)].sort_values("game_date").groupby("team"):
        gl = d.game_id.tolist()
        for i in range(len(gl) - 1):
            nxt[(gl[i], t)] = gl[i + 1]
    ab = EM[EM.absent == 1]
    SN = pd.read_parquet(os.path.join(DATA, "pv_nfl_pass_rush_and_front_snaps.parquet"))
    SN = SN[SN.season < TEST_ERA]
    played = set(zip(SN.game_id, SN.player_id))
    per = []
    for g, t, p in ab[["game_id", "team", "player_id"]].itertuples(index=False):
        n_ = nxt.get((g, t))
        if n_ is not None:
            per.append(int((n_, p) not in played))
    C["absent_regular_also_absent_next_game"] = round(float(np.mean(per)), 3) if per else None
    C["n_absent_regular_team_games"] = int(len(ab))
    OUT["C_snap_era"] = C
    log("C. snap era 2013-15:", json.dumps(C))

    # ---------------------------------------------------------------- D. leak probe
    Lp = L[L.season.between(DEV_LO, DEV_HI)].copy()
    Lp["next_margin"] = Lp.groupby(["team", "season"]).margin.shift(-1)
    Lp["prev_margin"] = Lp.groupby(["team", "season"]).margin.shift(1)
    m = Lp.next_margin.notna() & Lp.prev_margin.notna()
    Dd = {}
    for c in ("lv", "lvx", "ts", "qv"):
        Dd[c] = {"r_prev_game": round(r_(Lp.loc[m, c], Lp.loc[m, "prev_margin"]), 4),
                 "r_this_game": round(r_(Lp.loc[m, c], Lp.loc[m, "margin"]), 4),
                 "r_next_game": round(r_(Lp.loc[m, c], Lp.loc[m, "next_margin"]), 4)}
    OUT["D_leak_probe_team_rating_vs_own_margin"] = Dd
    log("D. leak probe (team rating vs own margin: previous / this / next game):", Dd)

    # ---------------------------------------------------------------- E. player level
    PS = pd.read_csv(os.path.join(DATA, "pv_nfl_compose_player_seasons.csv"))
    assert PS.season.max() < TEST_ERA
    PS = PS[PS.games >= 8]
    E = {}
    nx = PS.merge(PS, on="player_id", suffixes=("", "_n"))
    nx = nx[nx.season_n == nx.season + 1]
    nx = nx[nx.season.between(DEV_LO, DEV_HI - 1)]
    for grp in ("QB", "WR", "TE", "RB", "DL", "LB", "DB"):
        d = nx[nx.pgrp == grp]
        if len(d) < 30:
            continue
        st = d[d.team == d.team_n]; mv = d[d.team != d.team_n]
        E[grp] = {"n": int(len(d)), "yoy_r_mean_value": round(r_(d.value_mean, d.value_mean_n), 3),
                  "yoy_r_end_value": round(r_(d.value_last, d.value_mean_n), 3),
                  "stayers_r": round(r_(st.value_mean, st.value_mean_n), 3) if len(st) > 20 else None,
                  "movers_r": round(r_(mv.value_mean, mv.value_mean_n), 3) if len(mv) > 20 else None,
                  "n_movers": int(len(mv)),
                  "sd_value_pts_per_game": round(float(d.value_mean.std()), 3)}
    OUT["E_player_yoy"] = E
    log("E. player-season value YoY (>=8 games both seasons, DEV):")
    for g_, v in E.items():
        log(f"   {g_:<3} {v}")
    face = {}
    for s in (2010, 2015):
        face[str(s)] = {}
        for grp in ("QB", "WR", "TE", "RB", "DL", "LB", "DB"):
            d = PS[(PS.season == s) & (PS.pgrp == grp) & (PS.games >= 10)].sort_values("value_last", ascending=False)
            face[str(s)][grp] = {"top": [f"{n} ({t}) {v:+.2f}" for n, t, v in d[["name", "team", "value_last"]].head(8).itertuples(index=False)],
                                 "bottom": [f"{n} ({t}) {v:+.2f}" for n, t, v in d[["name", "team", "value_last"]].tail(5).itertuples(index=False)]}
    OUT["E_face_validity"] = face
    for s in ("2015",):
        for grp, v in face[s].items():
            log(f"   {s} {grp} top: {', '.join(v['top'])}")
            log(f"   {s} {grp} bottom: {', '.join(v['bottom'])}")
    # value scale by position (what one player is worth)
    sc = PS[PS.season.between(DEV_LO, DEV_HI)].groupby("pgrp").value_mean.describe(percentiles=[.1, .5, .9])
    OUT["E_value_scale_by_position"] = {k: {c: round(float(v), 3) for c, v in r.items()} for k, r in sc.iterrows()}
    log("   value scale (pts/game, player-seasons >= 8 games):\n" + sc.round(2).to_string())
    json.dump(OUT, open(os.path.join(DATA, "pv_nfl_compose_validation.json"), "w"), indent=1)
    log("wrote data/pv_nfl_compose_validation.json")


if __name__ == "__main__":
    main()
