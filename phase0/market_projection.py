"""MARKET PROJECTION — how much of the closing line lives INSIDE our feature space?

Third leg of the floor programme (documents/depth_program.md). error_map.py measured
the SIZE of the gap to the close; market_gap.py measured WHERE it lives and found it
DIFFUSE in every league. This file asks the remaining structural question:

    Take the closing line and project it onto the span of OUR OWN FEATURES.
    How much of it do our features already explain (R^2)?  And does the part they
    CANNOT explain — the residual — still predict outcomes once our own forecast is
    already in the equation, and by how much?

That decomposition separates the two hypotheses the NFL leg left open, and asks a
question of NHL that could not be asked before today: our NHL model ties the closing
line in aggregate log loss (+0.00171, n.s.). A tie is consistent with TWO very
different worlds — "we know the same things" (small residual, or a residual that
carries no outcome information) and "we know different things equally well" (large
residual that DOES predict outcomes, while we also hold information the close lacks).
Only the projection + residual test can tell them apart.

=============================== THIS IS A MEASURING DEVICE, NOT A MODEL =============
Every regression below has the CLOSING LINE on one side of the equals sign. That is a
DIAGNOSTIC / EVALUATION use of the odds — explicitly sanctioned, and identical in kind
to what error_map.py and market_gap.py already do. Concretely:

  * No model in this repo sees a closing line. No served artifact is touched, refitted
    or written by this file. No feature is created.
  * The projection `logit(p_close) ~ our features` is an INFORMATION-OVERLAP MEASUREMENT.
    Its coefficients are "what weight would a market-mimicking blend put on this
    feature", reported for comparison against our own weights. They are never used to
    fit, tune, select or serve anything.
  * The residual test `y ~ logit(p_ours) + residual` is a *bound* on what the close's
    orthogonal information is worth. It is deliberately an upper bound (it is fitted
    on the same DEV games, though also reported leave-one-season-out), and it is
    reported as a diagnostic ceiling, NOT as a candidate model. A model built this way
    would be market-derived and is forbidden by the programme's rules.

============================== ONE-LOOK PROTOCOL (hard-asserted) ====================
  MLB   DEV 2002-2015, market overlap 2010-2015.  TEST >= 2016 never loaded.
  NHL   DEV 2011-12 .. 2017-18.                   TEST >= 2018-19 never loaded.
  NFL   not re-run here (market_gap.py already carries it); no NFL number is produced.
No TEST-era model number is computed, printed or written by this file. 0 TEST looks.

============================== REUSE, NOT REBUILD ===================================
  phase0/error_map.py    mlb_league() / nhl_league() run unchanged; their slicing
                         labels are captured in flight by market_gap's existing spies.
  phase0/market_gap.py   Frame, league_block, heterogeneity, encompassing,
                         disagreement_table, fit_glm are imported and called directly,
                         so the NHL decomposition is literally the same instrument that
                         produced the MLB and NFL rows.
  phase0/nhl_floor.py    the NHL odds join is the same (date, home, away) join, with
                         the same SBR-score-vs-API result assertion, re-verified
                         against data/nhl_floor.json before anything is computed.
  Feature matrices are intercepted from the harnesses themselves (mlb_depth_eval.
  full_cols and nhl_depth_eval.Ctx), never re-derived, so this file cannot drift from
  the shipped models even by accident.

Usage:  python -X utf8 phase0/market_projection.py   ->  data/market_projection.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import error_map as EM      # noqa: E402  loaders + atomic_json + llv/logit/sig
import market_gap as MG     # noqa: E402  Frame / league_block / encompassing / fit_glm

OUT = "data/market_projection.json"
ODDS_NHL = "data/odds_nhl.csv"
NHL_FLOOR_REF = "data/nhl_floor.json"
EM_REF = "data/error_map.json"
B_BOOT = 2000
SEED = 11
T0 = time.time()

MLB_TEST_LO = 2016

r6, llv, logit, sig = EM.r6, EM.llv, EM.logit, EM.sig


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ================================================== feature capture (spies) ===
CAPF: dict[str, dict] = {}


def mlb_load():
    """error_map.mlb_league() unchanged, with the blend's feature columns captured.

    mlb_depth_eval.full_cols is the single place the shipped blend's design is
    assembled; wrapping it means the matrix measured here IS the matrix the model
    is fitted on, in the model's own column order.
    """
    import mlb_depth_eval as M
    real = M.full_cols
    names = ["lf"] + list(M.FEATS)

    def spy(cols):
        out = real(cols)
        CAPF["mlb"] = {"X": np.column_stack(out), "names": names}
        return out

    M.full_cols = spy
    try:
        hold, lab = MG.load_league("mlb", EM.mlb_league)
    finally:
        M.full_cols = real
    assert "mlb" in CAPF, "mlb_depth_eval.full_cols was never called"
    return hold, lab, CAPF["mlb"]


def nhl_load():
    """error_map.nhl_league() unchanged, with the Ctx captured.

    The Ctx carries the game rows (needed for the odds join, which error_map's NHL
    hold does not return) and X_for reproduces the shipped design matrix exactly.
    """
    import nhl_depth_eval as Hd
    real = Hd.Ctx
    box = {}

    class SpyCtx(real):
        def __init__(self):
            super().__init__()
            box["ctx"] = self

    Hd.Ctx = SpyCtx
    try:
        hold, lab = MG.load_league("nhl", EM.nhl_league)
    finally:
        Hd.Ctx = real
    ctx = box["ctx"]
    X = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)[:, 1:]   # drop intercept column
    names = ["elo_logit", "rest_diff", "b2b_home", "b2b_away", "xg_diff"]
    assert X.shape[1] == len(names)
    G = [g for g, keep in zip(ctx.games, ctx.mask) if keep]
    assert len(G) == len(hold["p"])
    CAPF["nhl"] = {"X": X, "names": names}
    return hold, lab, CAPF["nhl"], G, Hd


# ============================================================== NHL odds join ==
def nhl_join(G, y):
    """(index, devigged close) for DEV games in data/odds_nhl.csv. EVALUATION ONLY.

    Identical to phase0/nhl_floor.py's join, including its independent correctness
    check: SBR's own final score must agree with the NHL API result for every joined
    game or the join is wrong and we stop.
    """
    idx_by_key = defaultdict(list)
    for r in csv.DictReader(open(ODDS_NHL, encoding="utf-8")):
        if r["home_close"] and r["away_close"]:
            idx_by_key[(r["date"], r["home"], r["away"])].append(r)
    keep, pm, miss, amb = [], [], 0, 0
    for i, g in enumerate(G):
        m = idx_by_key.get((g["date"], g["home"], g["away"]))
        if not m:
            miss += 1
            continue
        if len(m) != 1:
            amb += 1
            continue
        r = m[0]
        assert int(r["home_win"]) == int(round(float(y[i]))), \
            f"result mismatch on {g['date']} {g['away']}@{g['home']}"
        qh, qa = 1.0 / float(r["home_close"]), 1.0 / float(r["away_close"])
        keep.append(i)
        pm.append(qh / (qh + qa))
    return np.array(keep), np.array(pm), miss, amb


def check_against(ref_ll, ref_n, lg, p_o, p_m, y, tol_ours=3e-4):
    ours, mkt = float(llv(y, p_o).mean()), float(llv(y, p_m).mean())
    assert ref_n == len(y), f"{lg} join size {len(y)} != recorded {ref_n}"
    assert abs(ref_ll[1] - mkt) < 1e-6, f"{lg} market_ll {mkt:.8f} != {ref_ll[1]}"
    assert abs(ref_ll[0] - ours) < tol_ours, f"{lg} our_ll {ours:.8f} != {ref_ll[0]}"
    return {"n": len(y), "our_ll": r6(ours), "market_ll": r6(mkt),
            "recorded_our_ll": ref_ll[0], "recorded_market_ll": ref_ll[1],
            "reproduces_recorded_floor": True}


# ================================================================= projection ==
def ols(X1, z):
    """Least squares with an intercept already in column 0. Returns (beta, se, R2)."""
    beta, *_ = np.linalg.lstsq(X1, z, rcond=None)
    fit = X1 @ beta
    resid = z - fit
    n, k = X1.shape
    ss_res = float(resid @ resid)
    ss_tot = float(((z - z.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(n - k, 1)
    s2 = ss_res / dof
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), r2, resid, fit


def project(lg, Xz, names, p_o, p_m, y, seas):
    """Project logit(p_close) — and, as the symmetric control, logit(p_ours) — onto
    the model's own standardised feature space.

    MEASUREMENT, NOT A MODEL. The left-hand side is the closing line; the coefficients
    are read as "the weight a market-mimicking linear blend would put on this feature",
    purely so they can be compared with ours. Nothing fitted here is ever served.

    Our own R^2 is a machine check, not a result: our forecast IS a logistic linear in
    these columns, so it must come back at essentially 1.0. Anything below ~0.99 means
    the captured matrix is not the matrix the model uses, and the comparison is void.
    """
    n = len(y)
    X1 = np.column_stack([np.ones(n), Xz])
    z_m, z_o = logit(p_m), logit(p_o)

    b_m, se_m, r2_m, res_m, fit_m = ols(X1, z_m)
    b_o, se_o, r2_o, res_o, _ = ols(X1, z_o)
    assert r2_o > 0.985, (f"{lg}: our own forecast is only {r2_o:.4f}-explained by the "
                          "captured feature matrix — the capture is wrong")

    # outcome-optimal weights on the SAME standardised columns: what the results want
    w_y, se_y = MG.fit_glm(X1, y)

    feats = []
    for j, nm in enumerate(names, start=1):
        bo, bm = float(b_o[j]), float(b_m[j])
        feats.append({
            "feature": nm,
            "our_weight": r6(bo), "our_weight_se": r6(float(se_o[j])),
            "market_implied_weight": r6(bm), "market_weight_se": r6(float(se_m[j])),
            "market_minus_ours": r6(bm - bo),
            "market_over_ours": r6(bm / bo) if abs(bo) > 1e-9 else None,
            "diff_t": r6((bm - bo) / math.sqrt(se_m[j] ** 2 + se_o[j] ** 2))
                      if (se_m[j] + se_o[j]) > 0 else None,
            "outcome_fit_weight": r6(float(w_y[j])), "outcome_fit_se": r6(float(se_y[j])),
            "lean": ("market leans HARDER" if bm > bo else "market leans LIGHTER"),
        })

    # ---- residual: the part of the close our feature space cannot represent
    sd_res = float(res_m.std(ddof=1))
    # what that residual is worth in probability terms at the sample's own logits
    dp = np.abs(sig(z_m) - sig(z_m - res_m))
    return {
        "n": n,
        "seasons": [int(seas.min()), int(seas.max())],
        "features": names,
        "r2_market_on_our_features": r6(r2_m),
        "r2_ours_on_our_features": r6(r2_o),
        "r2_ours_reading": ("machine check: must be ~1.0 because our forecast is a "
                            "logistic linear in exactly these columns (fold-to-fold "
                            "weight variation is the only thing that keeps it below 1)."),
        "unexplained_share_of_close": r6(1.0 - r2_m),
        "residual_sd_logit": r6(sd_res),
        "residual_sd_as_prob_shift": r6(float(dp.mean())),
        "residual_p90_as_prob_shift": r6(float(np.percentile(dp, 90))),
        "sd_logit_close": r6(float(z_m.std(ddof=1))),
        "sd_logit_ours": r6(float(z_o.std(ddof=1))),
        "corr_logit_ours_close": r6(float(np.corrcoef(z_o, z_m)[0, 1])),
        "intercept_market": r6(float(b_m[0])), "intercept_ours": r6(float(b_o[0])),
        "per_feature": feats,
        "_resid": res_m,
        "_fit": fit_m,
    }


# ============================================================ residual testing ==
def loso_ll(X1, y, seas):
    """Leave-one-season-out log loss of a logistic on X1 (intercept in column 0)."""
    p = np.empty(len(y))
    for s in sorted(set(seas.tolist())):
        te = seas == s
        w, _ = MG.fit_glm(X1[~te], y[~te])
        p[te] = sig(X1[te] @ w)
    return float(llv(y, p).mean()), p


def nested_residual_gain(p_o, p_m, y, Xz, seas):
    """Fully nested version of the residual test — nothing about the held-out season
    is used to DEFINE the residual, not just to weight it.

    The plain LOSO number below leaks mildly: the projection that defines the residual
    is fitted on every season including the one being scored. Round 1's method standard
    (documents/depth_program.md) is that any quantity chosen with the data must be
    re-chosen without seeing the fold it is scored on, so both are reported. Still a
    diagnostic ceiling — the closing line is on the right-hand side either way.
    """
    n = len(y)
    zo, zm = logit(p_o), logit(p_m)
    X1f = np.column_stack([np.ones(n), Xz])
    p = np.empty(n)
    for s in sorted(set(seas.tolist())):
        te = seas == s
        tr = ~te
        b, *_ = np.linalg.lstsq(X1f[tr], zm[tr], rcond=None)      # projection: train only
        r_tr = zm[tr] - X1f[tr] @ b
        r_te = zm[te] - X1f[te] @ b
        A_tr = np.column_stack([np.ones(int(tr.sum())), zo[tr], r_tr])
        A_te = np.column_stack([np.ones(int(te.sum())), zo[te], r_te])
        w, _ = MG.fit_glm(A_tr, y[tr])
        p[te] = sig(A_te @ w)
    ll = float(llv(y, p).mean())
    d = llv(y, p_o) - llv(y, p)
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, n, size=(B_BOOT, n))].mean(axis=1)
    return {"ll_plus_residual_nested": r6(ll),
            "gain_vs_shipped_nested": r6(float(d.mean())),
            "gain_vs_shipped_nested_ci": [r6(np.percentile(bs, 2.5)),
                                          r6(np.percentile(bs, 97.5))],
            "note": ("projection AND augmentation re-fitted per fold; the residual for a "
                     "held-out season is defined only by other seasons.")}


def residual_test(lg, p_o, p_m, y, resid, seas):
    """Conditional on our own forecast, does the UNEXPLAINED part of the close predict
    outcomes — and by how much?

    DIAGNOSTIC CEILING, NOT A MODEL. Fitting `y ~ logit(p_ours) + residual` uses the
    closing line on the right-hand side; the resulting numbers are an upper bound on
    what the close's orthogonal information would be worth if we could ever obtain it
    from market-blind data. Nothing here is adopted, served or promoted.
    """
    n = len(y)
    zo = logit(p_o)
    X0 = np.column_stack([np.ones(n), zo])
    X1 = np.column_stack([np.ones(n), zo, resid])
    w1, se1 = MG.fit_glm(X1, y)
    ll_our = float(llv(y, p_o).mean())
    ll_mkt = float(llv(y, p_m).mean())
    ll0_is = float(llv(y, sig(X0 @ MG.fit_glm(X0, y)[0])).mean())
    ll1_is = float(llv(y, sig(X1 @ w1)).mean())
    ll0_cv, _ = loso_ll(X0, y, seas)
    ll1_cv, p1 = loso_ll(X1, y, seas)

    d = llv(y, p_o) - llv(y, p1)          # our shipped model vs the residual-augmented
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, n, size=(B_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    gap = ll_our - ll_mkt

    return {
        "n": n,
        "b_residual": r6(float(w1[2])), "b_residual_se": r6(float(se1[2])),
        "b_residual_t": r6(float(w1[2] / se1[2])),
        "b_ours": r6(float(w1[1])), "b_ours_se": r6(float(se1[1])),
        "b_ours_t": r6(float(w1[1] / se1[1])),
        "our_ll": r6(ll_our), "market_ll": r6(ll_mkt), "gap_us_to_market": r6(gap),
        "ll_recal_ours_only_insample": r6(ll0_is),
        "ll_plus_residual_insample": r6(ll1_is),
        "gain_insample": r6(ll0_is - ll1_is),
        "ll_recal_ours_only_loso": r6(ll0_cv),
        "ll_plus_residual_loso": r6(ll1_cv),
        "gain_loso": r6(ll0_cv - ll1_cv),
        "gain_vs_shipped_loso": r6(ll_our - ll1_cv),
        "gain_vs_shipped_loso_ci": [r6(lo), r6(hi)],
        "residual_recovers_share_of_market_gap": r6((ll_our - ll1_cv) / gap) if abs(gap) > 1e-9 else None,
        "beats_market": bool(ll1_cv < ll_mkt),
        "verdict": ("residual CARRIES outcome information our features cannot represent"
                    if lo > 0 else "residual adds nothing beyond our own forecast (n.s.)"),
        "reading": ("b_residual is the outcome weight on the part of the close our feature "
                    "space cannot span, with our own forecast already in the equation. "
                    "t<2 means the close's orthogonal component is not informative — i.e. "
                    "everything the close knows, our features already span. Positive and "
                    "significant means the close holds information outside our span. The "
                    "LOSO gain is the honest size of that information; the in-sample gain "
                    "is its optimistic ceiling."),
        "warning": ("UPPER BOUND / DIAGNOSTIC ONLY. This quantity is only realisable by a "
                    "forecaster who can see the closing line, which no served model in this "
                    "repo may do. It measures how much information is missing, not a gain "
                    "that is available to us."),
    }


def residual_test_full(lg, p_o, p_m, y, resid, seas, Xz):
    out = residual_test(lg, p_o, p_m, y, resid, seas)
    out.update(nested_residual_gain(p_o, p_m, y, Xz, seas))
    gap = out["our_ll"] - out["market_ll"]
    out["nested_recovers_share_of_market_gap"] = (
        r6(out["gain_vs_shipped_nested"] / gap) if abs(gap) > 1e-9 else None)
    out["verdict"] = ("residual CARRIES outcome information our features cannot represent"
                      if out["gain_vs_shipped_nested_ci"][0] > 0
                      else "residual adds nothing beyond our own forecast (n.s.)")
    return out


def symmetry(lg, p_o, p_m, y, resid, seas, Xz, names):
    """The mirror question: does OUR forecast hold information the CLOSE does not?

    Projecting the close onto our features answers "what do we span of theirs". The
    encompassing regression answers the reverse. Together they classify the tie:
      same things   -> small residual, residual n.s., b_ours ~ 0
      different     -> residual informative AND b_ours > 0 with a real t
    """
    n = len(y)
    zo, zm = logit(p_o), logit(p_m)
    enc = MG.encompassing(MG.Frame(lg, p_o, p_m, y, {}))
    # our forecast's own orthogonal component: logit(p_ours) residualised on the close
    A = np.column_stack([np.ones(n), zm])
    bo, *_ = np.linalg.lstsq(A, zo, rcond=None)
    res_o = zo - A @ bo
    X = np.column_stack([np.ones(n), zm, res_o])
    w, se = MG.fit_glm(X, y)
    ll_cv, _ = loso_ll(X, y, seas)
    ll_m_only, _ = loso_ll(A, y, seas)
    return {
        "encompassing_y_on_both": enc,
        "our_orthogonal_component": {
            "sd_logit": r6(float(res_o.std(ddof=1))),
            "b": r6(float(w[2])), "se": r6(float(se[2])), "t": r6(float(w[2] / se[2])),
            "ll_close_only_loso": r6(ll_m_only),
            "ll_close_plus_our_orthogonal_loso": r6(ll_cv),
            "gain_loso": r6(ll_m_only - ll_cv),
            "reading": ("the exact mirror of the residual test: the part of OUR forecast the "
                        "close cannot span, tested for outcome information with the close "
                        "already in the equation."),
        },
        "classification_rule": ("BOTH informative -> 'we know different things'; only the "
                                "close's residual informative -> 'the close knows strictly "
                                "more'; neither -> 'we know the same things'."),
    }


def classify(res_block, sym_block):
    theirs = res_block["b_residual_t"] is not None and abs(res_block["b_residual_t"]) > 2.0 \
        and res_block["gain_vs_shipped_loso_ci"][0] > 0
    ours = abs(sym_block["our_orthogonal_component"]["t"]) > 2.0
    if theirs and ours:
        return "WE KNOW DIFFERENT THINGS — both sides hold information the other lacks"
    if theirs and not ours:
        return "THE CLOSE KNOWS STRICTLY MORE — its orthogonal part predicts, ours does not"
    if ours and not theirs:
        return "WE KNOW STRICTLY MORE — our orthogonal part predicts, the close's does not"
    return "WE KNOW THE SAME THINGS — neither side's orthogonal component predicts"


# ======================================================================= main ==
def standardise(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def main():
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": ("DEV-only projection of the closing line onto each model's own feature "
                 "space, plus the outcome test of the unexplained residual. Third leg of "
                 "the floor programme after phase0/error_map.py and phase0/market_gap.py."),
        "protocol": {
            "read_only": True, "models_touched": 0, "features_created": 0,
            "test_looks_spent": 0, "market_blind_models": True,
            "odds_use": ("DIAGNOSTIC / EVALUATION ONLY. Regressions in this file put the "
                         "closing line on one side of the equals sign in order to MEASURE "
                         "information overlap. They are measuring devices, not predictors: "
                         "no coefficient estimated here is used to fit, tune, select or "
                         "serve any model, and no served model in this repo sees an odds "
                         "value. The residual-test gains are UPPER BOUNDS on missing "
                         "information, not gains available to a market-blind forecaster."),
            "leagues": "MLB + NHL (NFL already carried by phase0/market_gap.py)",
            "loaders": ("error_map.mlb_league / error_map.nhl_league unchanged; slicing "
                        "labels captured by market_gap's spies; feature matrices captured "
                        "from mlb_depth_eval.full_cols and nhl_depth_eval.Ctx; joins "
                        "re-verified against data/error_map.json and data/nhl_floor.json."),
            "standardisation": ("features are z-scored on the joined DEV sample so that our "
                                "weight and the market-implied weight are in identical "
                                "logits-per-SD units and can be compared directly."),
        },
        "leagues": {},
    }

    # ------------------------------------------------------------------ MLB --
    log("MLB: running the shipped DEV harness via error_map ...")
    hold, lab, cap = mlb_load()
    idx, pm = MG.mlb_join(hold)
    seas = hold["seas"][idx]
    assert seas.max() < MLB_TEST_LO, "market join reached a TEST-era MLB season"
    p_o, y = hold["p"][idx], hold["y"][idx]
    ref = json.load(open(EM_REF, encoding="utf-8"))["leagues"]["mlb"]["floor"]
    chk = check_against((ref["our_ll"], ref["market_ll"]), int(ref["n"]),
                        "mlb", p_o, pm, y)
    X = cap["X"][idx]
    Xz, mu, sd = standardise(X)
    log(f"MLB joined n={len(y)} seasons {int(seas.min())}-{int(seas.max())} {chk}")

    proj = project("mlb", Xz, cap["names"], p_o, pm, y, seas)
    resid = proj.pop("_resid")
    proj.pop("_fit")
    rt = residual_test_full("mlb", p_o, pm, y, resid, seas, Xz)
    sym = symmetry("mlb", p_o, pm, y, resid, seas, Xz, cap["names"])
    payload["leagues"]["mlb"] = {
        "join_check": chk,
        "feature_sd_raw": {n_: r6(float(s)) for n_, s in zip(cap["names"], sd)},
        "projection": proj,
        "residual_test": rt,
        "symmetry": sym,
        "classification": classify(rt, sym),
        "coverage_note": ("odds_mlb.csv covers 2010+, DEV is 2002-2015 -> overlap "
                          f"{int(seas.min())}-{int(seas.max())}; doubleheaders and "
                          "duplicate price rows dropped, exactly as error_map does."),
    }
    EM.atomic_json(payload, OUT)

    # ------------------------------------------------------------------ NHL --
    log("NHL: running the shipped DEV harness via error_map ...")
    hold, lab, cap, G, Hd = nhl_load()
    idx, pm, miss, amb = nhl_join(G, hold["y"])
    seas = hold["seas"][idx]
    assert seas.max() <= Hd.DEV_END < Hd.TEST_START, "market join reached a TEST-era season"
    p_o, y = hold["p"][idx], hold["y"][idx]
    nref = json.load(open(NHL_FLOOR_REF, encoding="utf-8"))["floor"]
    chk = check_against((nref["our_ll"], nref["market_ll"]), int(nref["n"]),
                        "nhl", p_o, pm, y)
    chk.update({"unmatched": miss, "ambiguous": amb, "result_mismatches": 0,
                "join_rate": r6(len(idx) / len(G))})
    X = cap["X"][idx]
    Xz, mu, sd = standardise(X)
    log(f"NHL joined n={len(y)} seasons {int(seas.min())}-{int(seas.max())} {chk}")

    proj = project("nhl", Xz, cap["names"], p_o, pm, y, seas)
    resid = proj.pop("_resid")
    proj.pop("_fit")
    rt = residual_test_full("nhl", p_o, pm, y, resid, seas, Xz)
    sym = symmetry("nhl", p_o, pm, y, resid, seas, Xz, cap["names"])

    # ---- NHL BONUS: the full market_gap decomposition, same instrument ------
    log("NHL: market-gap decomposition (market_gap.league_block, reused verbatim) ...")
    labs = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(hold["p"]) else v)
            for k, v in lab.items()}
    labs["season"] = np.array([str(s) for s in seas])
    fr = MG.Frame("nhl", p_o, pm, y, labs)
    dims = ["pred_decile", "month", "season_maturity", "favourite_side", "rest", "b2b",
            "season", "team"]
    mat, b2b = labs["season_maturity"], labs["b2b"]
    dis = np.abs(fr.p_o - fr.p_m)
    cons = [
        ("cold start: team-games 0-4", mat == "team-games 0-4",
         "if the close's edge lives here, the missing information is roster/offseason"),
        ("cold start: team-games 0-14", np.isin(mat, ["team-games 0-4", "team-games 5-14"]),
         "wider cold-start window, more power"),
        ("back-to-back either side", b2b != "b2b: neither",
         "fatigue is schedule information both sides can see; does the close price it "
         "better than our b2b terms?"),
        ("October", labs["month"] == "10", "calendar version of the cold-start test"),
        ("top 5% disagreement", dis >= np.quantile(dis, 0.95),
         "where we and the close differ most"),
    ]
    block = MG.league_block(
        "nhl", fr, dims,
        f"data/odds_nhl.csv (SBR archive via phase0/nhl_odds_load.py) covers "
        f"2010-11..2022-23; DEV is {int(seas.min())}-{int(seas.max())}, join rate "
        f"{len(idx)/len(G):.4f}, 0 result mismatches.", cons)
    block["join_check"] = chk

    payload["leagues"]["nhl"] = {
        "join_check": chk,
        "feature_sd_raw": {n_: r6(float(s)) for n_, s in zip(cap["names"], sd)},
        "projection": proj,
        "residual_test": rt,
        "symmetry": sym,
        "classification": classify(rt, sym),
        "market_gap_decomposition": block,
        "coverage_note": ("first NHL market-gap decomposition — the closing line only "
                          "entered the repo on 2026-07-31, which is why this cell was "
                          "blank in data/market_gap.json."),
    }
    EM.atomic_json(payload, OUT)
    log(f"wrote {OUT}")

    # ------------------------------------------------------------------ print
    for lg in ("mlb", "nhl"):
        d = payload["leagues"][lg]
        p_, rt_, sy_ = d["projection"], d["residual_test"], d["symmetry"]
        print(f"\n=== {lg.upper()}  n={p_['n']}  seasons {p_['seasons']}")
        print(f"  PROJECTION of the close onto our feature space")
        print(f"    R^2 market on our features   {p_['r2_market_on_our_features']:.4f}"
              f"   (unexplained {p_['unexplained_share_of_close']:.4f})")
        print(f"    R^2 ours   on our features   {p_['r2_ours_on_our_features']:.4f}"
              f"   <- machine check, must be ~1")
        print(f"    residual sd {p_['residual_sd_logit']:.4f} logits "
              f"= {p_['residual_sd_as_prob_shift']:.4f} mean |dp| "
              f"(p90 {p_['residual_p90_as_prob_shift']:.4f})")
        print(f"    sd logit close {p_['sd_logit_close']:.3f} vs ours "
              f"{p_['sd_logit_ours']:.3f}; corr {p_['corr_logit_ours_close']:.4f}")
        print("  PER-FEATURE WEIGHTS (logits per SD, same standardisation both sides)")
        print(f"    {'feature':<12} {'ours':>9} {'market':>9} {'mkt-ours':>9} "
              f"{'mkt/ours':>9} {'t(diff)':>8} {'outcome':>9}")
        for f_ in p_["per_feature"]:
            mo = f_["market_over_ours"]
            print(f"    {f_['feature']:<12} {f_['our_weight']:>+9.4f} "
                  f"{f_['market_implied_weight']:>+9.4f} {f_['market_minus_ours']:>+9.4f} "
                  f"{(mo if mo is not None else float('nan')):>9.3f} "
                  f"{f_['diff_t']:>+8.2f} {f_['outcome_fit_weight']:>+9.4f}  {f_['lean']}")
        print("  RESIDUAL TEST  (diagnostic ceiling; the close is on the RHS)")
        print(f"    b_residual {rt_['b_residual']:+.4f} (t {rt_['b_residual_t']:+.2f})   "
              f"b_ours {rt_['b_ours']:+.4f} (t {rt_['b_ours_t']:+.2f})")
        print(f"    ours {rt_['our_ll']:.6f} | +residual LOSO {rt_['ll_plus_residual_loso']:.6f} "
              f"| close {rt_['market_ll']:.6f}")
        print(f"    LOSO   gain vs shipped {rt_['gain_vs_shipped_loso']:+.6f} "
              f"{rt_['gain_vs_shipped_loso_ci']}  recovers "
              f"{rt_['residual_recovers_share_of_market_gap']} of the market gap")
        print(f"    NESTED gain vs shipped {rt_['gain_vs_shipped_nested']:+.6f} "
              f"{rt_['gain_vs_shipped_nested_ci']}  recovers "
              f"{rt_['nested_recovers_share_of_market_gap']} (market gap "
              f"{rt_['gap_us_to_market']:+.6f}) -> {rt_['verdict']}")
        o = sy_["our_orthogonal_component"]
        e = sy_["encompassing_y_on_both"]
        print("  MIRROR  (does OUR forecast hold what the close lacks?)")
        print(f"    our orthogonal sd {o['sd_logit']:.4f}, b {o['b']:+.4f} (t {o['t']:+.2f}), "
              f"LOSO gain over close-only {o['gain_loso']:+.6f}")
        print(f"    encompassing b_ours {e['b_ours']:+.3f} (t {e['b_ours_t']:+.2f})  "
              f"b_mkt {e['b_market']:+.3f} (t {e['b_market_t']:+.2f})")
        print(f"  => {d['classification']}")

    b = payload["leagues"]["nhl"]["market_gap_decomposition"]
    print(f"\n=== NHL MARKET-GAP DECOMPOSITION (first time — odds arrived 2026-07-31)")
    print(f"  n={b['n']} ours {b['our_ll']} close {b['market_ll']} gap {b['gap']:+.5f} "
          f"{b['gap_ci']} total {b['total_gap_nats']:+.1f} nats")
    print(f"  lambda {b['lambda_market_credit']} {b['lambda_ci']}   "
          f"gap-KLnull {b['gap_minus_kl_null']:+.5f}")
    print("  -- slices ranked by total gap contributed")
    print(f"  {'dim':<18} {'slice':<24} {'n':>6} {'gap':>9} {'total':>8} {'share':>6} "
          f"{'nshare':>6} {'lam':>6}")
    for r in b["ranked_by_total_gap"][:12]:
        lm = r["lambda_market_credit"]
        print(f"  {r['dim']:<18} {r['slice']:<24} {r['n']:>6} {r['gap']:>+9.5f} "
              f"{r['total_gap']:>+8.2f} {r['share_of_total_gap']:>6.2f} "
              f"{r['n_share']:>6.2f} {(lm if lm is not None else float('nan')):>6.2f}"
              f"  {r.get('gap_sig','')}")
    print("  -- heterogeneity")
    for dd, h in b["heterogeneity"].items():
        print(f"  {dd:<18} Q={h['Q']:8.2f} df={h['df']:<3} p={h['p']:.4f}  {h['verdict']}")
    print("  -- contrasts")
    for c in b["contrasts"]:
        print(f"  {c['contrast']:<30} n={c['n_in']:>5} in {c['gap_in']:+.5f} out "
              f"{c['gap_out']:+.5f} diff {c['diff']:+.5f} {str(c['diff_ci']):<22} "
              f"{c['verdict']}")
    print("  -- disagreement quantiles")
    print(f"  {'bucket':<16} {'n':>6} {'gap':>9} {'KLnull':>9} {'gap-KL':>9} {'lam':>6} "
          f"{'lamCI':>16}")
    for r in b["disagreement"]["quantiles"]:
        print(f"  {r['slice']:<16} {r['n']:>6} {r['gap']:>+9.5f} "
              f"{r['kl_if_market_truth']:>9.5f} "
              f"{r['gap'] - r['kl_if_market_truth']:>+9.5f} "
              f"{r['lambda_market_credit']:>6.2f} {str(r.get('lambda_ci')):>16}")
    e = b["disagreement"]["encompassing_all"]
    e2 = b["disagreement"]["encompassing_top20pct_disagreement"]
    print(f"  encompassing ALL   b_ours {e['b_ours']:+.3f} (t {e['b_ours_t']:+.2f})  "
          f"b_mkt {e['b_market']:+.3f} (t {e['b_market_t']:+.2f})")
    print(f"  encompassing HIDIS b_ours {e2['b_ours']:+.3f} (t {e2['b_ours_t']:+.2f})  "
          f"b_mkt {e2['b_market']:+.3f} (t {e2['b_market_t']:+.2f})  n={e2['n']}")
    o = b["disagreement"]["opposite_favourite"]
    if "gap" in o:
        print(f"  opposite favourites n={o['n']}: we {o['our_side_win_rate']:.3f} vs close "
              f"{o['market_side_win_rate']:.3f} hit rate, gap {o['gap']:+.5f}")


if __name__ == "__main__":
    main()
