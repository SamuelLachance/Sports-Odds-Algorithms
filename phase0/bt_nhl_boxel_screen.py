"""Breakthrough pick nhl_boxel, round 2 -- the DEV screen on real box scores.

DEV ONLY, MARKET-BLIND.  Binding protocol:
documents/breakthrough_program_prereg_2026_09_24.md.  This screen's own
pre-declaration was frozen BEFORE any box-score statistic was read:
data/bt_nhl_boxel_prereg.json (arms, grid, gates, bar, serving diagnostic).

Round 1 built the construction (phase0/bt_nhl_nhl_boxel.py: lineup_pass,
elo_cell, Evaluator, perturbation tests) but could only run its no-box parts,
because the ~9.4k box scores had not been pulled.  The pull is
phase0/bt_nhl_boxel_fetch.py -> data/bt_nhl_boxel_box.csv.  This file reuses the
round-1 construction UNCHANGED (its LF-normalised sha1 is asserted against the
prereg) and runs exactly the pre-declared arms:

  P   PRIMARY  lineup sum L + starting-goalie level G, both INSIDE the W/L Elo
               (round-1 C3), (beta, gamma) nested over {0..60} x {0..180}
  A2           L inside only (gamma = 0), beta nested          (round-1 C2)
  A3           G inside (gamma nested) + dLtil beside            (round-1 C4)
  C0, C1       references: shipped blend; G inside only (= gmar A1)
  D1           SERVING DIAGNOSTIC, never eligible: P's per-season cells and
               blend weights, tonight's feature computed from PROJECTED inputs

DAY-OF INFORMATION.  P's two new inputs are same-day facts: tonight's dressed
skater list (the box score's dressed list = the confirmed lineup, known at
warm-ups) and tonight's starting goalie (first-shot goalie = the confirmed
starter, usually known the morning of, always at warm-ups).  Their post-game
production/GA only ever enters the state AFTER the game is predicted.  A gain
on P is therefore a gain for a CONFIRMED-tier forecast issued after lineups
and starters are known; D1 measures what survives when the lineup is the
team's previous-game lineup and the goalie is projected (PROJECTED/EARLY).

Every scored mask passes nhl_depth_eval.assert_dev_only.  No TEST-era game
(game_id >= 2018000000, season >= 2018-19) is loaded, rated, joined or scored.

Usage:
  python phase0/bt_nhl_boxel_screen.py           # full screen (needs the pull)
  python phase0/bt_nhl_boxel_screen.py --smoke   # SYNTHETIC production on the real
                                                 # lineups: plumbing only; writes
                                                 # data/bt_nhl_boxel_screen_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, TEST_START, llv  # noqa: E402
from bt_nhl_anatomy_build import MAX_GID, load_goalie_games, load_starters  # noqa: E402
import bt_nhl_nhl_boxel as R1  # noqa: E402
import bt_nhl_gmar_gl as GM  # noqa: E402

PREREG = "data/bt_nhl_boxel_prereg.json"
OUT = "data/bt_nhl_boxel_screen.json"
OUT_SMOKE = "data/bt_nhl_boxel_screen_smoke.json"
assert DEV_END == 20172018 and TEST_START == 20182019 and MAX_GID == 2018000000

K_ELO, HA_ELO, REG_ELO = Hd.SHIPPED_ELO
BETAS, GAMMAS = R1.BETAS, R1.GAMMAS
BAR = 0.00100
r6 = R1.r6


def sha16(path):
    return hashlib.sha1(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16]


def check_prereg():
    pr = json.load(open(PREREG, encoding="utf-8"))
    want = pr["construction_frozen"]["module_sha1_16_lf_normalised"]
    got = {f: sha16(f) for f in want}
    assert got == want, f"construction changed after the prereg froze: {got} != {want}"
    assert pr["construction_frozen"]["grid"] == {"beta": BETAS, "gamma": GAMMAS}
    return {"prereg_sha1_16": sha16(PREREG), "frozen_at_utc": pr["frozen_at_utc"],
            "module_hashes_match": True}


# ----------------------------------------------------- projected lineup pass ---
def lineup_pass_proj(games, box, dfb):
    """R1.lineup_pass (onice=None) re-implemented line for line, plus a PROJECTED
    read: the same pre-game state, with tonight's roster replaced by the roster
    the team dressed in its previous game (strictly earlier date).  The actual
    read is asserted bit-identical to R1.lineup_pass by the caller.
    """
    MIN_SK, M0, T0 = R1.MIN_SK, R1.M0, R1.T0
    n = len(games)
    Lh, La = np.zeros(n), np.zeros(n)
    Lth, Lta = np.zeros(n), np.zeros(n)
    Lph, Lpa = np.zeros(n), np.zeros(n)
    S, T, N, M, FIRST, POS = {}, {}, {}, {}, {}, {}
    ps = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
    pool = {"F": [0.0, 0.0, 0], "D": [0.0, 0.0, 0]}
    ebuf = {"F": [0.0, 0.0, 0], "D": [0.0, 0.0, 0]}
    pool_max_season = None
    wf = [0, 0.0, 0.0]
    day_ps = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
    day_L = []
    last_absorbed = ""
    vbar = {"F": 0.0, "D": 0.0}
    vbar_fixed = False
    season_toi = defaultdict(float)
    cur_date = None
    prev = None
    last_roster = {}                 # team -> (date, [(pid, pc)]) of its previous game
    proj_empty = 0

    def pos_mean(pc):
        s_, h_ = ps[pc]
        return s_ / h_ if h_ > 0 else 0.0

    def ent_mu(pc):
        g_, h_, c_ = pool[pc]
        if c_ >= R1.ENT_MIN and h_ > 0:
            return g_ / h_
        return R1.ENT_FALLBACK * pos_mean(pc)

    def v_of(p, pc, date):
        f = FIRST.get(p, date)
        if f < R1.ENT_DATE:
            mu = pos_mean(pc)
        elif N.get(p, 0) < R1.ENT_MAXN:
            mu = ent_mu(pc)
        else:
            mu = pos_mean(pc)
        return (S.get(p, 0.0) + mu * T0) / (T.get(p, 0.0) + T0)

    def running_vbar(date):
        acc = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
        for p, w in season_toi.items():
            if w <= 0:
                continue
            pc = POS.get(p, "F")
            acc[pc][0] += w * v_of(p, pc, date)
            acc[pc][1] += w
        return {pc: (acc[pc][0] / acc[pc][1] if acc[pc][1] > 0 else pos_mean(pc))
                for pc in acc}

    for i, g in enumerate(games):
        s, date = g["season"], g["date"]
        if date != cur_date:
            if cur_date is not None:
                for pc in ("F", "D"):
                    ps[pc][0] += day_ps[pc][0]
                    ps[pc][1] += day_ps[pc][1]
                for x in day_L:
                    wf[0] += 1
                    dlt = x - wf[1]
                    wf[1] += dlt / wf[0]
                    wf[2] += dlt * (x - wf[1])
                last_absorbed = cur_date
            day_ps = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
            day_L = []
            new_season = prev is not None and s != prev
            if new_season:
                for p in S:
                    S[p] *= R1.SCARRY
                    T[p] *= R1.SCARRY
                for pc in ("F", "D"):
                    for k in range(3):
                        pool[pc][k] += ebuf[pc][k]
                    ebuf[pc] = [0.0, 0.0, 0]
                pool_max_season = prev
                vbar = running_vbar(date)
                vbar_fixed = True
                season_toi.clear()
            elif not vbar_fixed:
                vbar = running_vbar(date)
            cur_date = date
            prev = s
        assert last_absorbed < date
        assert pool_max_season is None or pool_max_season < s
        sd_L = np.sqrt(wf[2] / (wf[0] - 1)) if wf[0] >= R1.SD_MIN else None
        gid = g["game_id"]
        gb = box.get(gid, {}) if box is not None else {}
        tonight = {}
        for side, Larr, Ltarr, Lparr in (("home", Lh, Lth, Lph), ("away", La, Lta, Lpa)):
            sd_rec = gb.get(side)
            if sd_rec is not None and len(sd_rec["sk"]) >= MIN_SK:
                roster = [(r[0], r[1]) for r in sd_rec["sk"]]
            else:
                roster = []
                for p in dfb.get(gid, {}).get(side, []):
                    roster.append((p, POS.get(p, "F")))
            L = 0.0
            for p, pc in roster:
                L += (v_of(p, pc, date) - vbar[pc]) * M.get(p, M0[pc]) / 60.0
            Larr[i] = L
            Ltarr[i] = L / sd_L if sd_L else 0.0
            # PROJECTED read: previous game's dressed list, same state, same scale
            lr = last_roster.get(g[side])
            if lr is None:
                rp = []
                proj_empty += 1
            else:
                assert lr[0] < date, "projected roster not from a strictly earlier date"
                rp = lr[1]
            Lp = 0.0
            for p, pc in rp:
                Lp += (v_of(p, pc, date) - vbar[pc]) * M.get(p, M0[pc]) / 60.0
            Lparr[i] = Lp / sd_L if sd_L else 0.0
            tonight[side] = roster
        # ---------------------------- update (after the read) ----------------
        for side in ("home", "away"):
            day_L.append(Lh[i] if side == "home" else La[i])
            last_roster[g[side]] = (date, tonight[side])
            sd_rec = gb.get(side)
            if sd_rec is None or len(sd_rec["sk"]) < MIN_SK:
                continue
            for (p, pc, _pos, G_, A_, SOG, BLK, PIM, toi) in sd_rec["sk"]:
                gsl = R1.WG * G_ + R1.WA * A_ + R1.WSOG * SOG + R1.WBLK * BLK - R1.WPIM * PIM
                h = toi / 3600.0
                S[p] = R1.PDECAY * S.get(p, 0.0) + gsl
                T[p] = R1.PDECAY * T.get(p, 0.0) + h
                N[p] = N.get(p, 0) + 1
                M[p] = R1.M_DECAY * M.get(p, M0[pc]) + (1 - R1.M_DECAY) * toi / 60.0
                if p not in FIRST:
                    FIRST[p] = date
                POS[p] = pc
                day_ps[pc][0] += gsl
                day_ps[pc][1] += h
                if FIRST[p] >= R1.ENT_DATE and N[p] <= R1.ENT_FIRST:
                    ebuf[pc][0] += gsl
                    ebuf[pc][1] += h
                    ebuf[pc][2] += 1
                season_toi[p] += h
    return {"Lth": Lth, "Lta": Lta, "Lph": Lph, "Lpa": Lpa, "proj_empty_sides": proj_empty}


def elo_dual(games, beta, gamma, Lth, Lta, Gh, Ga, Lph, Lpa, Gph, Gpa):
    """R1.elo_cell plus a projected-input probability from the SAME pre-game
    ratings.  Ratings update on the actual-input probability only (history is
    rebuilt with actual lineups/starters, exactly as a served model would)."""
    R = {}
    n = len(games)
    out = np.empty(n)
    outp = np.empty(n)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - REG_ELO)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        E = (rh + beta * Lth[i] + gamma * Gh[i] + HA_ELO) - (ra + beta * Lta[i] + gamma * Ga[i])
        p = 1.0 / (1.0 + 10 ** (-E / 400.0))
        Ep = ((rh + beta * Lph[i] + gamma * Gph[i] + HA_ELO)
              - (ra + beta * Lpa[i] + gamma * Gpa[i]))
        outp[i] = 1.0 / (1.0 + 10 ** (-Ep / 400.0))
        out[i] = p
        R[g["home"]] += K_ELO * (g["y"] - p)
        R[g["away"]] += K_ELO * ((1 - g["y"]) - (1 - p))
    return out, outp


def projected_goalie(games, starters, ggames):
    proj = GM.project_starters(games, starters)
    r = GM.gl_elo(games, starters, ggames, 0.0, mode="new", starter_override=proj)
    return proj, r["G_h"], r["G_a"]


def proj_features(games, box, dfb, starters, ggames, beta=30, gamma=120):
    """Everything D1 reads, for the W6 perturbation test."""
    lpp = lineup_pass_proj(games, box, dfb)
    Gh, Ga, _ = R1.goalie_terms(games, starters, ggames)
    _, Gph, Gpa = projected_goalie(games, starters, ggames)
    pa, pp = elo_dual(games, beta, gamma, lpp["Lth"], lpp["Lta"], Gh, Ga,
                      lpp["Lph"], lpp["Lpa"], Gph, Gpa)
    return {"Lph": lpp["Lph"], "Lpa": lpp["Lpa"], "Gph": Gph, "Gpa": Gpa, "p_proj": pp}


def w6(games, box, dfb, starters, ggames):
    """W6: scramble EVERYTHING dated on/after each cutoff (results, GA/xGA, box
    production, dressed lists, starters): every projected feature strictly before
    the cutoff must be bit-identical."""
    base = proj_features(games, box, dfb, starters, ggames)
    dates = np.array([g["date"] for g in games])
    res = {}
    for ci, cut in enumerate(R1.CUTOFFS):
        pre = dates < cut
        pert = proj_features(*R1.perturb_inputs(games, box, dfb, starters, ggames, cut,
                                                seed=300 + ci, full=True))
        ok = all(np.array_equal(base[k][pre], pert[k][pre]) for k in base)
        changed = {k: bool(not np.array_equal(base[k][~pre], pert[k][~pre])) for k in base}
        res[cut] = {"pre_cutoff_bit_identical": bool(ok), "post_cutoff_changed": changed}
    return res


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    pre = check_prereg()
    ctx = Hd.Ctx()
    games = ctx.games
    assert max(g["season"] for g in games) <= DEV_END
    ids = {g["game_id"] for g in games}
    assert max(ids) < MAX_GID
    y = ctx.yv()
    folds = ctx.folds
    m = ctx.mask
    Hd.assert_dev_only(ctx.seasons[m])
    MH = R1.mask_hash(m)

    starters = load_starters(ids)
    ggames, goalie_ids = load_goalie_games(ids)
    dfb, dfb_unmatched = R1.load_dressed_fb(games, goalie_ids)

    rep = {"pick": "nhl_boxel", "round": 2, "prereg": pre,
           "mode": "SMOKE: synthetic production on real lineups; numbers carry no result"
           if args.smoke else "FULL",
           "protocol": {"doc": "documents/breakthrough_program_prereg_2026_09_24.md",
                        "harness": "phase0/nhl_depth_eval.py Ctx, LOSO over 2011-12..2017-18",
                        "test_seasons_touched": 0, "market_blind": True,
                        "bar": f"P nested gain >= {BAR} and 10k paired bootstrap CI lo > 0 "
                               "and every gate passes"}}

    # ---------------------------------------------------------- S1 / S2 ---
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    ll0 = folds.pooled(y, p0)
    s1 = {"ll": r6(ll0), "n": folds.N,
          "pass": bool(folds.N == 7929 and abs(ll0 - 0.672398) < 2e-6)}
    p_nx = folds.loso_pred(X0[:, :-1], y)
    cost = folds.pooled(y, p_nx) - ll0
    s2 = {"drop_xg_cost": r6(cost), "expected": 0.00303,
          "pass": bool(abs(cost - 0.00303) <= 0.0002)}
    print(f"S1 shipped DEV LOSO {ll0:.6f} n={folds.N}  S2 drop-xg cost {cost:+.5f}", flush=True)
    assert s1["pass"] and s2["pass"], "harness sanity failed"

    # ---------------------------------------------------------- box -------
    if args.smoke:
        box = R1.synth_box(games, dfb, starters)
        binfo = {"present": "SYNTHETIC"}
        coverage_ok = True
    else:
        box, binfo = R1.load_box(ids)
        assert box is not None, "data/bt_nhl_boxel_box.csv missing: run the fetch first"
        coverage_ok = bool(binfo["coverage"] >= 0.95)
    rep["box_info"] = binfo
    print(f"[{time.time()-t0:.0f}s] box: {binfo if args.smoke else {k: binfo[k] for k in ('rows', 'games', 'valid_sides', 'coverage')}}",
          flush=True)
    assert coverage_ok, f"box coverage {binfo.get('coverage')} < 0.95"

    Gh, Ga, ginfo = R1.goalie_terms(games, starters, ggames)
    assert ginfo["source"].startswith("bt_nhl_gmar_gl"), ginfo
    lp = R1.lineup_pass(games, box, dfb)
    Lth, Lta, dLh, dLa = lp["Lth"], lp["Lta"], lp["dLh"], lp["dLa"]
    lpp = lineup_pass_proj(games, box, dfb)
    assert np.array_equal(lpp["Lth"], Lth) and np.array_equal(lpp["Lta"], Lta), \
        "projected pass's actual read differs from R1.lineup_pass"
    Lph, Lpa = lpp["Lph"], lpp["Lpa"]
    proj_st, Gph, Gpa = projected_goalie(games, starters, ggames)
    print(f"[{time.time()-t0:.0f}s] lineup + goalie features built", flush=True)

    # ---------------------------------------------------------- S3 --------
    ref = Hd.run_elo_arr(games, K_ELO, HA_ELO, REG_ELO)
    s3d = float(np.max(np.abs(ref - R1.elo_cell(games, 0, 0, Lth, Lta, Gh, Ga))))
    base = ctx.base_cols()
    xgc = ctx.xg(*Hd.SHIPPED_XG)

    def X_of(p_elo, extra=None):
        cols = [R1.logit_masked(p_elo, m)] + base + [xgc]
        if extra is not None:
            cols.append(extra)
        return Hd.design(cols)

    def X_cell(b, gm, beside=False):
        return X_of(R1.elo_cell(games, b, gm, Lth, Lta, Gh, Ga),
                    (dLh - dLa)[m] if beside else None)

    design_eq = bool(np.array_equal(X_cell(0, 0), X0))
    s3 = {"max_abs_diff_vs_run_elo_arr": s3d, "design_00_equals_shipped": design_eq,
          "pass": bool(s3d <= 1e-12 and design_eq)}
    assert s3["pass"], s3

    # ---------------------------------------------------------- cells -----
    ev = R1.Evaluator(ctx)
    for b in BETAS:
        for gm in GAMMAS:
            ev.add(("C", b, gm), X_cell(b, gm))
    for gm in GAMMAS:
        ev.add(("B", 0, gm), X_cell(0, gm, beside=True))
    assert abs(ev.cells[("C", 0, 0)]["ll"] - ll0) < 1e-12
    print(f"[{time.time()-t0:.0f}s] {len(ev.cells)} cells evaluated", flush=True)

    keys = {"C1": [("C", 0, gm) for gm in GAMMAS],
            "A2": [("C", b, 0) for b in BETAS],
            "P": [("C", b, gm) for b in BETAS for gm in GAMMAS],
            "A3": [("B", 0, gm) for gm in GAMMAS]}
    pred, pick, best = {}, {}, {}
    for k, ks in keys.items():
        pred[k], pick[k], best[k] = ev.nested(ks)

    def picks(pk):
        return {str(s_): {"beta": c[1], "gamma": c[2]} for s_, c in pk.items()}

    def bestd(c):
        return {"beta": c[1], "gamma": c[2], "ll": r6(ev.cells[c]["ll"]),
                "gain": r6(ll0 - ev.cells[c]["ll"])}

    arms = {"C0_shipped_baseline": {"dev_ll": r6(ll0), "n": folds.N}}
    arms["C1_reference_G_inside_only"] = R1.arm_report(ev, p0, pred["C1"], {
        "nested_pick": picks(pick["C1"]), "optimistic_best": bestd(best["C1"])})
    arms["P_PRIMARY_L_and_G_inside"] = R1.arm_report(ev, p0, pred["P"], {
        "nested_pick": picks(pick["P"]), "optimistic_best": bestd(best["P"]),
        "picks_at_grid_edge": sorted({str(s_) for s_, c in pick["P"].items()
                                      if c[1] == BETAS[-1] or c[2] == GAMMAS[-1]}),
        "cell_ll": {f"b{c[1]}_g{c[2]}": r6(ev.cells[c]["ll"]) for c in keys["P"]}})
    arms["A2_L_inside_only"] = R1.arm_report(ev, p0, pred["A2"], {
        "nested_pick": picks(pick["A2"]), "optimistic_best": bestd(best["A2"])})
    arms["A3_G_inside_dL_beside"] = R1.arm_report(ev, p0, pred["A3"], {
        "nested_pick": picks(pick["A3"]), "optimistic_best": bestd(best["A3"])})
    lb = {k: llv(y, v) for k, v in pred.items()}
    secondary = {
        "lineup_increment_P_vs_C1": {"gain": r6((lb["C1"] - lb["P"]).mean()),
                                     "boot_ci_10k": R1.boot(lb["C1"] - lb["P"])},
        "inside_minus_beside_P_vs_A3": {"gain": r6((lb["A3"] - lb["P"]).mean()),
                                        "boot_ci_10k": R1.boot(lb["A3"] - lb["P"])}}
    late = np.isin(folds.seasons, R1.LATE)
    Hd.assert_dev_only(folds.seasons[late])
    d_inc = lb["C1"] - lb["P"]
    secondary["lineup_increment_P_vs_C1"]["late_dev_2015_18"] = {
        "gain": r6(d_inc[late].mean()), "boot_ci_10k": R1.boot(d_inc[late])}

    # ---------------------------------------------------------- D1 --------
    pD1 = np.empty(folds.N)
    pchk = np.empty(folds.N)
    dual_eq = True
    for s_ in folds.list:
        c = pick["P"][s_]
        pa, pp = elo_dual(games, c[1], c[2], Lth, Lta, Gh, Ga, Lph, Lpa, Gph, Gpa)
        dual_eq &= bool(np.array_equal(pa, R1.elo_cell(games, c[1], c[2], Lth, Lta, Gh, Ga)))
        Xa, Xp = X_of(pa), X_of(pp)
        tr = folds.seasons != s_
        w = Hd.fit_logit(Xa[tr], y[tr], Hd.BLEND_C)
        te = folds.idx[s_]
        pD1[te] = Hd._sig(Xp[te] @ w)
        pchk[te] = Hd._sig(Xa[te] @ w)
    d1_selfcheck = float(np.max(np.abs(pchk - pred["P"])))
    assert dual_eq and d1_selfcheck < 1e-6, (dual_eq, d1_selfcheck)
    gi = [(g["game_id"], g) for g, k in zip(games, m) if k]
    acc = [int(proj_st.get(gid, {}).get(f) == starters.get(gid, {}).get(f))
           for gid, _ in gi for f in (1, 0) if starters.get(gid, {}).get(f) is not None]
    lin_changed = float(np.mean(np.r_[Lph[m] != Lth[m], Lpa[m] != Lta[m]]))
    arms["D1_serving_projected_inputs_DIAGNOSTIC"] = R1.arm_report(ev, p0, pD1, {
        "note": "never eligible: P's per-season cells and blend weights (fit on actual "
                "day-of inputs), tonight's feature from the previous-game lineup and "
                "gmar project_starters",
        "selfcheck_actual_inputs_reproduce_P_max_abs_dp": d1_selfcheck,
        "projected_starter_correct_share": r6(np.mean(acc)),
        "share_sides_projected_L_differs": r6(lin_changed),
        "corr_projected_vs_actual_Ltil": r6(np.corrcoef(np.r_[Lph[m], Lpa[m]],
                                                         np.r_[Lth[m], Lta[m]])[0, 1])})
    d_d1 = llv(y, pD1) - lb["P"]
    secondary["P_minus_D1_value_of_confirmation"] = {
        "loss": r6(d_d1.mean()), "boot_ci_10k": R1.boot(d_d1),
        "share_of_P_gain_kept_by_D1": r6((ll0 - float(llv(y, pD1).mean()))
                                         / (ll0 - float(lb["P"].mean())))
        if ll0 != float(lb["P"].mean()) else None}
    for k, v in arms.items():
        if "gain" in v and not args.smoke:
            print(f"  {k:<42} LL {v['dev_ll']:.6f} gain {v['gain']:+.6f} CI {v['boot_ci_10k']} "
                  f"late {v['late_dev_2015_18']['gain']:+.6f} {v['late_dev_2015_18']['boot_ci_10k']}",
                  flush=True)

    # ---------------------------------------------------------- gates -----
    s4a, s4b = R1.s4a_s4b(games, box, dfb, starters)
    s4c = R1.s4c_check(games, starters, ggames, Gh, Ga)
    ka1, lla1, _ = R1.gmar_a1_ll()
    s4c.update({"gmar_json_arm": ka1, "gmar_A1_dev_ll": lla1,
                "C1_dev_ll": arms["C1_reference_G_inside_only"]["dev_ll"]})
    fv = R1.face_validity(ctx, lp["ts_L"])
    fv["pass"] = bool(fv["corr_meanL_vs_GF_per_game"] > 0)
    top = lp["diag"]["top10_v_at_2016_01_01"] or []
    try:
        names = json.load(open("data/nhl_player_names.json", encoding="utf-8"))
        for r in top:
            r["name"] = names.get(str(r["pid"]), {}).get("name")
    except Exception:  # noqa: BLE001
        pass
    fv["top10_v_at_2016_01_01"] = top
    print(f"[{time.time()-t0:.0f}s] S4a {s4a}  S4b {s4b}  S4c pass {s4c.get('pass')}  "
          f"S4d corr {fv['corr_meanL_vs_GF_per_game']}", flush=True)

    walk = {"W4": {"mask_hash_all_arms": MH,
                   "no_nan_masked": bool(not any(np.isnan(a[m]).any() for a in
                                                 (Lth, Lta, dLh, dLa, Gh, Ga, Lph, Lpa,
                                                  Gph, Gpa))),
                   "fallback_sides_nhl_dressed": lp["diag"]["fallback_sides"],
                   "fallback_players_without_history":
                       lp["diag"]["fallback_players_without_history"],
                   "projected_empty_sides": lpp["proj_empty_sides"],
                   "goalie_info": ginfo, "dressed_fb_unmatched_rows": dfb_unmatched}}
    try:
        walk["W2_W5"] = R1.w2_w5(games, box, dfb, starters, ggames, with_box=True)
        w25 = True
    except AssertionError as ex:
        walk["W2_W5"] = f"FAILED: {ex}"
        w25 = False
    walk["W6_projected"] = w6(games, box, dfb, starters, ggames)
    w6ok = all(v["pre_cutoff_bit_identical"] for v in walk["W6_projected"].values())
    print(f"[{time.time()-t0:.0f}s] W2/W5 {w25}  W6 {w6ok}", flush=True)

    gates = {"S1": s1["pass"], "S2": s2["pass"], "S3": s3["pass"], "S4a": s4a["pass"],
             "S4b": s4b["pass"], "S4c": bool(s4c.get("pass")), "S4d": fv["pass"],
             "coverage": coverage_ok, "W2_W5": w25, "W4": walk["W4"]["no_nan_masked"],
             "W6_projected_diag": w6ok}
    sanity = {"S1": s1, "S2": s2, "S3": s3, "S4a_dressed_vs_shift_charts": s4a,
              "S4b_box_starter_vs_first_shot": s4b, "S4c_C1_equals_gmar": s4c,
              "S4d_face_validity": fv}

    rep["close_diag_eval_only"] = R1.close_diag(ctx, {"C0": p0, "C1": pred["C1"],
                                                      "P": pred["P"], "D1": pD1})
    sib = {"rule": "within 0.0002 nats on DEV -> the simpler wins (gmar > boxel > rapmel)"}
    pll = arms["P_PRIMARY_L_and_G_inside"]["dev_ll"]
    for nm, f in (("nhl_gmar", "data/bt_nhl_gmar.json"), ("nhl_rapmel", "data/bt_nhl_rapmel.json")):
        try:
            v = json.load(open(f, encoding="utf-8"))["verdict"]
            sib[nm] = {"primary_dev_ll": v["candidate_dev_ll"], "boxel_P_dev_ll": pll,
                       "boxel_minus_sibling": r6(pll - v["candidate_dev_ll"])}
        except Exception as ex:  # noqa: BLE001
            sib[nm] = f"unavailable: {ex!r}"
    rep["sibling_comparison"] = sib

    P = arms["P_PRIMARY_L_and_G_inside"]
    gating = [g_ for g_ in gates if g_ != "W6_projected_diag"]
    all_ok = all(gates[g_] for g_ in gating)
    rep["verdict"] = {
        "primary_arm": "P (lineup sum L + starting-goalie level G inside the W/L Elo)",
        "baseline_dev_ll": r6(ll0), "candidate_dev_ll": P["dev_ll"],
        "nested_gain": P["gain"], "boot_ci_10k": P["boot_ci_10k"],
        "late_dev_2015_18": P["late_dev_2015_18"],
        "gates": gates, "failed_gates": [g_ for g_ in gating if not gates[g_]],
        "clears_bar": bool(P["gain"] >= BAR and P["boot_ci_10k"][0] > 0 and all_ok),
        "information_tier": "CONFIRMED only: needs tonight's dressed lineup and starting goalie"}
    rep.update({"arms": arms, "secondary": secondary, "sanity": sanity,
                "walk_forward": walk,
                "lineup_diag": {k: v for k, v in lp["diag"].items()
                                if k != "top10_v_at_2016_01_01"},
                "seconds": round(time.time() - t0, 1)})
    out = OUT_SMOKE if args.smoke else OUT
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, default=float)
    if args.smoke:
        print(json.dumps({"gates": gates, "failed": rep["verdict"]["failed_gates"]}, indent=1))
    else:
        print(json.dumps(rep["verdict"], indent=1))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
