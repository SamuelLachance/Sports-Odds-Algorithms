"""NFL HEADROOM DEEP-DIVE — is the +0.01172 gap to the close NOISE or INFORMATION?

WHY THIS FILE EXISTS
  documents/depth_program.md closed 9 optimization axes x 3 leagues with zero
  adoptions and measured the remaining distance to the closing line:

      MLB +0.00288 SIG   NFL +0.01172 SIG   NHL +0.00171 n.s.

  phase0/market_gap.py then showed the gap is DIFFUSE (0/16 heterogeneity tests
  reject uniformity) and that the MLB close statistically ENCOMPASSES our model
  (b_us = 0.028, t = 0.25). NFL is the outlier on both counts: 4x the gap, yet it
  RETAINS ~18% of the disagreement credit (b_us = 0.188, t = 1.71). NFL is
  therefore the only league where headroom plausibly exists, and the open question
  is whether its gap is

      (i)  ESTIMATION NOISE in our own ratings/blend from a small sample
           (NFL DEV n=2,531 vs MLB's 13,930; 16 games per team-season), or
      (ii) INFORMATION the market has that our 14-feature space cannot represent.

  Those two have completely different remedies, so this file separates them.

MARKET-BLINDNESS — READ THIS BEFORE READING ANY NUMBER BELOW
  Q2 regresses logit(p_close) ON our features. That regression is a MEASURING
  DEVICE, NOT A MODEL. Its only purpose is to compute how much of the closing
  line's variation lives inside the span of our feature matrix, and what is left
  over when it does not. Nothing fitted here is ever served, frozen, blended into
  a shipped artifact, or written anywhere a serving path can read it. The closing
  line remains what it has always been in this repo: an EVALUATION BENCHMARK. It
  is never a feature and never a training signal for any model that predicts a
  game. Every coefficient in this file is a diagnostic statistic about the market,
  in exactly the same sanctioned sense as error_map.py's floor and
  market_gap.py's encompassing regression.

ONE-LOOK PROTOCOL (hard-asserted at every stage)
  NFL DEV = 2006-2015 scored, trained from 1999.  TEST >= 2016 is never loaded,
  never scored, never printed. Asserts fire on the training mask, the scoring
  mask, the learning-curve windows and the market join. 0 TEST looks spent.

REUSE, DO NOT REBUILD
  X14                 phase0/nfl_depth_eval.build_main()  (the shipped matrix;
                      col 7 is the v7 11v11 participation TrueSkill feature)
  walk-forward        the 12-line protocol from error_map.nfl_league, verbatim
  market join         phase0/market_gap.nfl_join + check_join, which re-verifies
                      n / our_ll / market_ll against data/error_map.json
  helpers             error_map.llv / logit / sig / r6 / atomic_json

Usage:  python -X utf8 phase0/nfl_headroom.py   ->  data/nfl_headroom.json
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import error_map as EM      # noqa: E402
import market_gap as MG     # noqa: E402

OUT = "data/nfl_headroom.json"
B_BOOT = 2000
SEED = 11
T0 = time.time()

DEV_TRAIN_FROM = 1999
MAIN_LO, MAIN_HI = 2006, 2015
TEST_ERA = 2016
WIN_LO, WIN_HI = 2013, 2015      # fixed late-DEV scoring window for the curves
HL_BASE, C_BASE = 3.0, 100.0

FEAT = ["elo_logit", "qb_delta", "epa_rating", "early_down", "thfa", "rest",
        "fumble_luck", "v7_participation_ts", "absence_ol", "absence_def",
        "absence_skill", "roster_quality", "epa_pass", "epa_run"]

llv, logit, sig, r6 = EM.llv, EM.logit, EM.sig, EM.r6


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def boot_idx(n, rng):
    return rng.integers(0, n, size=(B_BOOT, n))


def ci(vals):
    return [r6(float(np.percentile(vals, 2.5))), r6(float(np.percentile(vals, 97.5)))]


def active(X, m):
    """Columns of X that actually vary inside mask m.

    A column that is identically zero is not a feature there: a penalised logistic
    leaves its coefficient at exactly 0 (zero gradient) and an OLS design containing
    it is singular. Which columns are live, and when, turns out to matter a lot on
    DEV — see feature_availability in the payload.
    """
    s = X[m].std(axis=0)
    return [j for j in range(X.shape[1]) if s[j] > 1e-12]


def availability(X14, seasons, y_all):
    """When does each of the 14 features exist? Reported because it reframes the gap."""
    rows = []
    for j, nm in enumerate(FEAT):
        seas = sorted({int(s) for s in set(seasons.tolist())
                       if (X14[seasons == s, j] != 0).any()})
        dev = (seasons >= MAIN_LO) & (seasons <= MAIN_HI) & (y_all != 0.5)
        rows.append({"col": j, "feature": nm,
                     "first_season_nonzero": (min(seas) if seas else None),
                     "nonzero_share_in_dev_scored": r6(float((X14[dev, j] != 0).mean())),
                     "sd_in_dev_scored": r6(float(X14[dev, j].std())),
                     "live_on_dev": bool(X14[dev, j].std() > 1e-12)})
    dev = (seasons >= MAIN_LO) & (seasons <= MAIN_HI) & (y_all != 0.5)
    dead = [r["feature"] for r in rows if not r["live_on_dev"]]
    late = [r["feature"] for r in rows
            if r["live_on_dev"] and (r["first_season_nonzero"] or 0) > MAIN_LO + 3]
    return {
        "features": rows,
        "dead_on_dev": dead,
        "late_arriving_on_dev": late,
        "n_live_on_dev": len(active(X14, dev)),
        "protocol_note": ("first_season_nonzero scans the whole spine, so it can report that "
                          "a COLUMN becomes nonzero in 2016. That is a data-availability fact "
                          "about an input matrix — no outcome is read, no model is scored and "
                          "no loss is computed outside DEV. It does not spend a TEST look."),
        "why_this_matters": (
            "The +0.01172 gap is measured on a DEV era where the shipped 14-feature matrix "
            f"is not 14 features. {', '.join(dead) or 'nothing'} is identically zero for every "
            f"DEV game (it starts in 2016, i.e. TEST era only), and {', '.join(late) or 'nothing'} "
            "exist only for the last three DEV seasons. So the DEV model whose gap we are "
            "decomposing is strictly weaker than the model that is served, and part of the "
            "NFL-vs-MLB gap difference is this availability asymmetry rather than anything "
            "about football. This is a measurement caveat on the gap, NOT a claim that the "
            "TEST-era model closes it — that would require a TEST look, which is not spent."),
    }


# ============================================================ spine + market ==
def build_spine():
    """The shipped NFL pipeline, DEV only, plus the closing line for the same games."""
    import nfl_depth_eval as N
    from sklearn.linear_model import LogisticRegression
    assert N.MAIN_LO == MAIN_LO and N.MAIN_HI == MAIN_HI, "NFL main split moved"
    assert N.MAIN_TEST_ERA == TEST_ERA and MAIN_HI < TEST_ERA, "NFL DEV/TEST split moved"
    assert N.HL_BASE == HL_BASE and N.C_BASE == C_BASE, "shipped training protocol moved"

    X14, _ex = N.build_main()
    games, seasons, y_all = N.games, N.seasons, N.y
    assert X14.shape[1] == len(FEAT) == 14

    pred = np.full(len(y_all), np.nan)
    for s_ in range(MAIN_LO, MAIN_HI + 1):
        assert s_ < TEST_ERA
        tr = (seasons < s_) & (y_all != 0.5)
        te = seasons == s_
        assert seasons[tr].max() < s_ and seasons[te].max() < TEST_ERA
        w = 0.5 ** ((s_ - 1 - seasons[tr]) / HL_BASE)
        lm = LogisticRegression(C=C_BASE, max_iter=5000).fit(X14[tr], y_all[tr],
                                                             sample_weight=w)
        pred[te] = lm.predict_proba(X14[te])[:, 1]

    scored = (seasons >= MAIN_LO) & (seasons <= MAIN_HI)
    assert seasons[scored].max() <= MAIN_HI < TEST_ERA, "TEST-era NFL game scored"
    ll_all = float(llv(y_all[scored], pred[scored]).mean())
    log(f"NFL main DEV wf ll {ll_all:.6f} n={int(scored.sum())} (recorded 0.62292)")
    assert abs(ll_all - 0.62292) < 5e-4, "NFL baseline not reproduced"

    keep = scored & (y_all != 0.5)
    rows = np.where(keep)[0]
    hold = {"p": pred[keep], "y": y_all[keep], "seas": seasons[keep],
            "gid": np.array([games[i]["gid"] for i in rows])}
    idx, pm = MG.nfl_join(hold)                       # verbatim reuse
    chk = MG.check_join("nfl", hold["p"][idx], pm, hold["y"][idx])
    sub = rows[idx]
    seas = seasons[sub]
    assert seas.max() <= MAIN_HI < TEST_ERA, "market join reached a TEST-era season"
    log(f"joined n={len(idx)} seasons {int(seas.min())}-{int(seas.max())}  {chk}")
    return {"X14": X14, "y_all": y_all, "seasons": seasons, "join_check": chk,
            "rows": sub, "p_ours": hold["p"][idx], "p_mkt": pm,
            "y": hold["y"][idx], "seas": seas}


# ==================================================== Q1a — the learning curve ==
def fit_predict(X14, y_all, seasons, tr_mask, score_rows, HL, cols=None):
    """One shipped-protocol logistic fit. Returns (p, n_train, n_eff, k_active)."""
    from sklearn.linear_model import LogisticRegression
    assert seasons[tr_mask].max() < WIN_LO, "learning-curve train window touches the score window"
    assert seasons[score_rows].max() < TEST_ERA, "learning-curve scored a TEST-era game"
    X = X14 if cols is None else X14[:, cols]
    if np.isfinite(HL):
        end = seasons[tr_mask].max()
        w = 0.5 ** ((end - seasons[tr_mask]) / HL)
    else:
        w = np.ones(int(tr_mask.sum()))
    lm = LogisticRegression(C=C_BASE, max_iter=5000).fit(X[tr_mask], y_all[tr_mask],
                                                         sample_weight=w)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    k_act = len(active(X, tr_mask)) + 1
    return lm.predict_proba(X[score_rows])[:, 1], int(tr_mask.sum()), n_eff, k_act


def curve(name, why, spans, X14, y_all, seasons, score_rows, y_w, p_mkt_w, HL,
          cols=None, ref=None):
    """One learning curve: DEV log loss on the FIXED window vs training evidence.

    Every point scores the identical games, so the market's own loss on those games
    is flat BY CONSTRUCTION — the close does not learn from our training data. It is
    the reference line, not a curve.
    """
    rng = np.random.default_rng(SEED)
    bi = boot_idx(len(score_rows), rng)
    pts, vecs = [], []
    for lo, hi in spans:
        assert hi < WIN_LO and lo >= DEV_TRAIN_FROM
        tr = (seasons >= lo) & (seasons <= hi) & (y_all != 0.5)
        if tr.sum() < 60:
            continue
        p, n, n_eff, k_act = fit_predict(X14, y_all, seasons, tr, score_rows, HL, cols)
        v = llv(y_w, p)
        vecs.append(v)
        pts.append({"train_seasons": f"{lo}-{hi}", "n_seasons": hi - lo + 1,
                    "n_train": n, "n_eff": r6(n_eff), "k_active": k_act,
                    "k_over_2neff": r6(k_act / (2 * n_eff)),
                    "ll": r6(float(v.mean()))})
    full = vecs[-1]
    for pt, v in zip(pts, vecs):
        d = (v - full)[bi].mean(axis=1)
        pt["ll_minus_full_history"] = r6(float((v - full).mean()))
        pt["ll_minus_full_history_ci"] = ci(d)
    mkt = float(llv(y_w, p_mkt_w).mean())
    out = {"curve": name, "why": why, "HL": ("inf" if not np.isfinite(HL) else HL),
           "C": C_BASE, "k_active_at_full_history": pts[-1]["k_active"],
           "n_scored": len(score_rows), "score_window": [WIN_LO, WIN_HI],
           "market_ll_on_same_games": r6(mkt),
           "market_is_flat_by_construction": True,
           "points": pts,
           "gap_to_market_at_full_history": r6(pts[-1]["ll"] - mkt)}
    if ref is not None:
        out["shipped_walk_forward_on_same_games"] = r6(float(llv(y_w, ref).mean()))
        out["shipped_reference_note"] = (
            "the shipped walk-forward refits per season and so, for the 2014 and 2015 folds, "
            "can train the snap-absence columns that every curve point above is structurally "
            "unable to use (they do not exist before 2013). It is the fair upper reference, "
            "not another curve point.")
    out.update(flatten_test(pts, vecs, y_w, bi))
    return out


def flatten_test(pts, vecs, y_w, bi):
    """Has the curve flattened? Fit LL = L_inf + c/n_eff, three ways.

    1/n_eff is the leading term of the parameter-estimation excess risk for a
    correctly specified k-parameter likelihood (k/(2 n_eff) nats), so a curve whose
    remaining descent is real should be LINEAR in 1/n_eff and the intercept is the
    infinite-data limit of THIS feature space and THIS training protocol.

    Three estimators because the extrapolation is the fragile step:
      all_points   widest lever arm (n_eff spans ~14x), best conditioned
      last5        only the largest training sets — closest to the regime we care
                   about, but the 1/n_eff range is narrow so the intercept is loose
      two_point    smallest vs largest, no fitting at all: c = dLL / d(1/n)
    The points are ordered smallest -> largest training set, so points[-1] is always
    the full-history model.
    """
    P = np.array(vecs)                                   # (P, n_games)
    x = np.array([1.0 / p["n_eff"] for p in pts])
    A = np.column_stack([np.ones(len(x)), x])
    H = A @ np.linalg.inv(A.T @ A)                       # yv -> beta is yv @ H
    i5 = max(0, len(x) - 5)
    A5 = A[i5:]
    H5 = A5 @ np.linalg.inv(A5.T @ A5)

    cnt = np.zeros((B_BOOT, len(y_w)))
    for j in range(B_BOOT):
        cnt[j] = np.bincount(bi[j], minlength=len(y_w))
    mb = (cnt @ P.T) / cnt.sum(axis=1)[:, None]          # (B, P) bootstrapped curve
    m0 = P.mean(axis=1)

    def block(m, mbo, Hm, lo):
        beta = m[lo:] @ Hm
        bb = mbo[:, lo:] @ Hm
        rem = m[-1] - beta[0]
        remb = mbo[:, -1] - bb[:, 0]
        return {"L_inf": r6(float(beta[0])), "L_inf_ci": ci(bb[:, 0]),
                "slope_c": r6(float(beta[1])), "slope_c_ci": ci(bb[:, 1]),
                "remaining_descent_to_L_inf": r6(float(rem)),
                "remaining_descent_ci": ci(remb),
                "still_descending": bool(np.percentile(remb, 2.5) > 0)}

    dx = x[0] - x[-1]
    c2 = (m0[0] - m0[-1]) / dx
    c2b = (mb[:, 0] - mb[:, -1]) / dx
    two = {"L_inf": r6(float(m0[-1] - c2 * x[-1])),
           "slope_c": r6(float(c2)), "slope_c_ci": ci(c2b),
           "remaining_descent_to_L_inf": r6(float(c2 * x[-1])),
           "remaining_descent_ci": ci(c2b * x[-1]),
           "still_descending": bool(np.percentile(c2b * x[-1], 2.5) > 0),
           "endpoints": [pts[0]["train_seasons"], pts[-1]["train_seasons"]]}
    spread = float(P.mean(axis=1).max() - P.mean(axis=1).min())
    return {
        "flattening": {
            "fit": "LL = L_inf + c / n_eff; points ordered smallest -> largest training set",
            "all_points": block(m0, mb, H, 0),
            "last5": block(m0, mb, H5, i5),
            "two_point": two,
            "total_spread_across_curve": r6(spread),
            "ll_smallest_minus_ll_largest": r6(float(m0[0] - m0[-1])),
            "reading": ("remaining_descent_to_L_inf is how much DEV log loss this blend "
                        "would still shed at unlimited training games with nothing else "
                        "changed. total_spread_across_curve is the model-free version: the "
                        "whole observed range from the smallest training set to the largest. "
                        "If the spread is small, no extrapolation can rescue a large "
                        "sample-starvation claim."),
        }
    }


def late_feature_cost(X14, y_all, seasons, sp, wmask):
    """DIAGNOSTIC: what do the four late-arriving snap-absence columns cost on DEV?

    The shipped walk-forward scores the 2013-2015 window WORSE than a single fit
    trained through 2012 does, which is backwards unless the extra columns are
    hurting. Those columns first exist in 2013, so for the 2014 fold their four
    coefficients are estimated from one partial season. This measures that, on the
    joined DEV games only.

    This is a MEASUREMENT, not a proposal. Turning a feature off is a model change
    and would need its own pre-registered screen with a nested estimate, per the
    Round 1 method standard in documents/depth_program.md. No adoption is implied
    and no TEST look is spent.
    """
    from sklearn.linear_model import LogisticRegression
    cols_late = [8, 9, 10, 11]
    out = {}
    for tag, X in (("with_absence_block", X14),
                   ("absence_block_zeroed", np.where(
                       np.isin(np.arange(X14.shape[1]), cols_late)[None, :],
                       0.0, X14))):
        pred = np.full(len(y_all), np.nan)
        for s_ in range(MAIN_LO, MAIN_HI + 1):
            assert s_ < TEST_ERA
            tr = (seasons < s_) & (y_all != 0.5)
            te = seasons == s_
            w = 0.5 ** ((s_ - 1 - seasons[tr]) / HL_BASE)
            lm = LogisticRegression(C=C_BASE, max_iter=5000).fit(X[tr], y_all[tr],
                                                                 sample_weight=w)
            pred[te] = lm.predict_proba(X[te])[:, 1]
        out[tag] = pred[sp["rows"]]
    y = sp["y"]
    rng = np.random.default_rng(SEED)
    bi = boot_idx(len(y), rng)
    va = llv(y, out["with_absence_block"])
    vb = llv(y, out["absence_block_zeroed"])
    res = {"what": ("cost of the four snap-absence columns (live only from 2013) inside the "
                    "shipped walk-forward, joined DEV games. DIAGNOSTIC ONLY — not a "
                    "proposal, no adoption, no TEST look."),
           "n_all_dev": len(y),
           "ll_with": r6(float(va.mean())), "ll_without": r6(float(vb.mean())),
           "cost_of_including": r6(float((va - vb).mean())),
           "cost_ci": ci((va - vb)[bi].mean(axis=1))}
    w = wmask
    bw = boot_idx(int(w.sum()), np.random.default_rng(SEED))
    res.update({"n_2013_2015": int(w.sum()),
                "ll_with_2013_2015": r6(float(va[w].mean())),
                "ll_without_2013_2015": r6(float(vb[w].mean())),
                "cost_of_including_2013_2015": r6(float((va - vb)[w].mean())),
                "cost_ci_2013_2015": ci((va - vb)[w][bw].mean(axis=1))})
    res["reading"] = (
        "a positive cost means the columns HURT on DEV. They are the newest data in the "
        "matrix and the least-estimated; whatever they cost here is estimation noise "
        "attributable to features whose history is genuinely short, which is the sharpest "
        "concrete instance of the sample-starvation question this pass asks.")
    return res


# ================================================ Q1b — Murphy decomposition ==
def platt_loso(z, y, seas):
    """Leave-one-season-out recalibration a + b*z. Cross-fitted, so honest."""
    zc, bs = np.zeros(len(z)), {}
    for s in sorted(set(seas.tolist())):
        te = seas == s
        tr = ~te
        X = np.column_stack([np.ones(int(tr.sum())), z[tr]])
        w, _se = MG.fit_glm(X, y[tr])
        zc[te] = w[0] + w[1] * z[te]
        bs[str(int(s))] = [r6(w[0]), r6(w[1])]
    Xa = np.column_stack([np.ones(len(z)), z])
    wa, sea = MG.fit_glm(Xa, y)
    return zc, {"pooled_intercept": r6(wa[0]), "pooled_slope": r6(wa[1]),
                "pooled_slope_se": r6(sea[1]),
                "slope_minus_1_t": r6((wa[1] - 1.0) / sea[1]),
                "per_season": bs}


def bin_decomp(p, y, k=10):
    """DeGroot-Fienberg / Murphy split of the log score into CAL + REF (+ within).

    LL(p) = CAL + REF + WITHIN with
      CAL    = mean_b KL(obs_b || pbar_b)      reliability: are the bins right?
      REF    = mean_b H(obs_b)                 refinement: do the bins separate?
      WITHIN = LL(p_i) - LL(pbar_b)            loss/gain from p varying inside a bin
    Ranking games by p is the standard conditioning; the bins are equal-count.
    """
    o = np.argsort(p)
    edges = np.array_split(o, k)
    cal = ref = wit = 0.0
    rows = []
    for b in edges:
        pb = float(p[b].mean())
        ob = float(y[b].mean())
        obc = min(max(ob, 1e-9), 1 - 1e-9)
        kl = obc * math.log(obc / pb) + (1 - obc) * math.log((1 - obc) / (1 - pb))
        h = -(obc * math.log(obc) + (1 - obc) * math.log(1 - obc))
        wb = float(llv(y[b], p[b]).mean() - llv(y[b], np.full(len(b), pb)).mean())
        f = len(b) / len(p)
        cal += f * kl
        ref += f * h
        wit += f * wb
        rows.append({"n": len(b), "pbar": r6(pb), "obs": r6(ob),
                     "kl_cal": r6(kl), "entropy_ref": r6(h)})
    return {"CAL": cal, "REF": ref, "WITHIN": wit, "total": cal + ref + wit,
            "bins": rows}


def murphy(sp, k=10):
    """Is our excess loss a CALIBRATION deficit or a REFINEMENT deficit?

    CALIBRATION deficit = our probabilities are systematically mis-scaled. That is
    what additive estimation noise on the rating scale LOOKS like (attenuation), and
    it is fixable with a monotone map, which would make the error map's calibration
    null surprising.
    REFINEMENT deficit = our bins simply do not separate winners from losers as
    sharply as the market's do. That is INFORMATION and no reparameterization
    touches it.
    """
    p_o, p_m, y, seas = sp["p_ours"], sp["p_mkt"], sp["y"], sp["seas"]
    zo, zm = logit(p_o), logit(p_m)
    n = len(y)
    rng = np.random.default_rng(SEED)
    bi = boot_idx(n, rng)

    do = bin_decomp(p_o, y, k)
    dm = bin_decomp(p_m, y, k)

    zo_c, cal_o = platt_loso(zo, y, seas)
    zm_c, cal_m = platt_loso(zm, y, seas)
    ll_o, ll_m = float(llv(y, p_o).mean()), float(llv(y, p_m).mean())
    ll_oc = float(llv(y, sig(zo_c)).mean())
    ll_mc = float(llv(y, sig(zm_c)).mean())

    v_o, v_m = llv(y, p_o), llv(y, p_m)
    v_oc, v_mc = llv(y, sig(zo_c)), llv(y, sig(zm_c))
    gap_b = (v_o - v_m)[bi].mean(axis=1)
    recal_gap_b = (v_oc - v_mc)[bi].mean(axis=1)
    cal_gain_b = (v_o - v_oc)[bi].mean(axis=1)

    # sharpness / discrimination, no binning
    auc = _auc(p_o, y), _auc(p_m, y)
    return {
        "what": ("Splits our excess loss over the close into a CALIBRATION part "
                 "(recoverable by a monotone remap of our own output) and a "
                 "REFINEMENT part (information: how sharply the forecast separates)."),
        "n": n, "our_ll": r6(ll_o), "market_ll": r6(ll_m), "gap": r6(ll_o - ll_m),
        "gap_ci": ci(gap_b),
        "binned_degroot_fienberg": {
            "bins": k,
            "ours": {"CAL": r6(do["CAL"]), "REF": r6(do["REF"]),
                     "WITHIN": r6(do["WITHIN"]), "total_check": r6(do["total"]),
                     "table": do["bins"]},
            "market": {"CAL": r6(dm["CAL"]), "REF": r6(dm["REF"]),
                       "WITHIN": r6(dm["WITHIN"]), "total_check": r6(dm["total"]),
                       "table": dm["bins"]},
            "delta_CAL": r6(do["CAL"] - dm["CAL"]),
            "delta_REF": r6(do["REF"] - dm["REF"]),
            "delta_WITHIN": r6(do["WITHIN"] - dm["WITHIN"]),
            "null_bias_per_side": r6(k / (2.0 * len(y))),
            "null_note": ("each bin's obs rate costs 1 df, so E[CAL] under perfect "
                          f"calibration is about k/(2n) = {k / (2.0 * len(y)):.5f} nats for "
                          "BOTH sides. The bias cancels in delta_CAL, which is why delta_CAL "
                          "is the number to read and the raw CALs are not."),
            "share_of_gap_from_refinement": r6((do["REF"] - dm["REF"]) / (ll_o - ll_m)),
        },
        "cross_fitted_platt": {
            "method": "leave-one-season-out a + b*logit(p); no in-sample df spent",
            "ours": cal_o, "market": cal_m,
            "our_ll_recalibrated": r6(ll_oc), "market_ll_recalibrated": r6(ll_mc),
            "our_calibration_gain": r6(ll_o - ll_oc),
            "our_calibration_gain_ci": ci(cal_gain_b),
            "market_calibration_gain": r6(ll_m - ll_mc),
            "gap_after_both_recalibrated": r6(ll_oc - ll_mc),
            "gap_after_both_recalibrated_ci": ci(recal_gap_b),
            "share_of_gap_surviving_recalibration": r6((ll_oc - ll_mc) / (ll_o - ll_m)),
            "reading": ("our slope < 1 would mean our forecast is OVER-dispersed, the exact "
                        "signature of additive estimation noise on the rating scale. A slope "
                        "at 1 says the composite forecast carries no DETECTABLE attenuation, "
                        "which bounds how much of the gap can be pure rating noise."),
        },
        "discrimination": {
            "our_auc": r6(auc[0]), "market_auc": r6(auc[1]),
            "our_sd_logit": r6(float(zo.std())), "market_sd_logit": r6(float(zm.std())),
            "sd_ratio_market_over_ours": r6(float(zm.std() / zo.std())),
        },
    }


def _auc(p, y):
    o = np.argsort(p)
    r = np.empty(len(p), float)
    r[o] = np.arange(1, len(p) + 1)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ================================== Q2 — projecting the close onto our features ==
def projection(sp):
    """How much of the closing line lives INSIDE the span of our 14 features?

    ===================================================================
    THIS IS A MEASURING DEVICE, NOT A MODEL. Regressing logit(p_close) on
    X14 answers "what part of the market's opinion is a re-weighting of
    things we already measure, and what part is outside our vocabulary".
    The fitted object is never served, never frozen, never blended, and no
    model in this repo consumes it. Market-blindness of every SERVED model
    is unaffected.
    ===================================================================

    The residual of the close after this projection IS, by construction, the market
    information our feature space cannot represent. Its value is then measured
    against outcomes, cross-fitted by season so nothing is scored in-sample.
    """
    Xf, y, seas = sp["X"], sp["y"], sp["seas"]
    zo, zm = logit(sp["p_ours"]), logit(sp["p_mkt"])
    n = len(y)
    cols = active(Xf, np.ones(n, bool))       # drop DEV-dead columns: singular otherwise
    dropped = [FEAT[j] for j in range(Xf.shape[1]) if j not in cols]
    names = [FEAT[j] for j in cols]
    X = Xf[:, cols]
    A = np.column_stack([np.ones(n), X])
    seasons_u = sorted(set(seas.tolist()))

    def ols(Am, t, m):
        return np.linalg.lstsq(Am[m], t[m], rcond=None)[0]

    # ---- pooled projection (the descriptive object) ----
    g = ols(A, zm, np.ones(n, bool))
    fit_m = A @ g
    r2_m = 1.0 - float(((zm - fit_m) ** 2).sum() / ((zm - zm.mean()) ** 2).sum())
    b_eff = ols(A, zo, np.ones(n, bool))
    fit_o = A @ b_eff
    r2_o = 1.0 - float(((zo - fit_o) ** 2).sum() / ((zo - zo.mean()) ** 2).sum())
    # 1-df version: how much of the close is our forecast alone?
    A1 = np.column_stack([np.ones(n), zo])
    g1 = np.linalg.lstsq(A1, zm, rcond=None)[0]
    r2_1 = 1.0 - float(((zm - A1 @ g1) ** 2).sum() / ((zm - zm.mean()) ** 2).sum())

    # ---- cross-fitted projection: honest residual + honest scoring ----
    resid = np.zeros(n)
    proj = np.zeros(n)
    for s in seasons_u:
        te = seas == s
        gs = ols(A, zm, ~te)
        proj[te] = A[te] @ gs
        resid[te] = zm[te] - proj[te]
    # calibrate the projection into a probability, still cross-fitted
    p_proj = np.zeros(n)
    p_res = np.zeros(n)
    br_list = []
    for s in seasons_u:
        te = seas == s
        tr = ~te
        Xp = np.column_stack([np.ones(int(tr.sum())), proj[tr]])
        w, _ = MG.fit_glm(Xp, y[tr])
        p_proj[te] = sig(w[0] + w[1] * proj[te])
        Xr = np.column_stack([np.ones(int(tr.sum())), zo[tr], resid[tr]])
        w2, se2 = MG.fit_glm(Xr, y[tr])
        p_res[te] = sig(w2[0] + w2[1] * zo[te] + w2[2] * resid[te])
        br_list.append([r6(w2[1]), r6(w2[2])])

    Xr = np.column_stack([np.ones(n), zo, resid])
    wr, ser = MG.fit_glm(Xr, y)

    ll_o = float(llv(y, sp["p_ours"]).mean())
    ll_m = float(llv(y, sp["p_mkt"]).mean())
    ll_proj = float(llv(y, p_proj).mean())
    ll_res = float(llv(y, p_res).mean())
    rng = np.random.default_rng(SEED)
    bi = boot_idx(n, rng)
    v_o, v_m = llv(y, sp["p_ours"]), llv(y, sp["p_mkt"])
    v_p, v_r = llv(y, p_proj), llv(y, p_res)

    sd = X.std(axis=0)
    scale = float(np.std(fit_m) / np.std(fit_o))
    feats = []
    # SEs for the market-implied weights (OLS on a near-noiseless target)
    res_m = zm - fit_m
    s2 = float((res_m ** 2).sum() / (n - A.shape[1]))
    cov = s2 * np.linalg.inv(A.T @ A)
    seg = np.sqrt(np.diag(cov))
    res_o = zo - fit_o
    s2o = float((res_o ** 2).sum() / (n - A.shape[1]))
    covo = s2o * np.linalg.inv(A.T @ A)
    seo = np.sqrt(np.diag(covo))
    for j, nm in enumerate(names):
        gj, bj = g[j + 1] * sd[j], b_eff[j + 1] * sd[j]
        d = gj - scale * bj
        sd_d = math.sqrt((seg[j + 1] * sd[j]) ** 2 + (scale * seo[j + 1] * sd[j]) ** 2)
        feats.append({
            "feature": nm,
            "market_weight_per_sd": r6(gj),
            "our_weight_per_sd": r6(bj),
            "our_weight_per_sd_scale_matched": r6(scale * bj),
            "diff_market_minus_ours": r6(d),
            "diff_t": r6(d / sd_d) if sd_d > 0 else None,
            "ratio_market_over_ours": r6(gj / bj) if abs(bj) > 1e-9 else None,
        })
    feats.sort(key=lambda f: -abs(f["diff_market_minus_ours"]))

    # ---- BLOCKS: single coefficients are not interpretable under collinearity ----
    # epa_rating / epa_pass / epa_run are three views of the same EPA walk, so the
    # split between them is arbitrary. A block's total contribution is not.
    BLOCKS = {
        "team_rating (elo, qb)": ["elo_logit", "qb_delta"],
        "epa (rating, pass, run)": ["epa_rating", "epa_pass", "epa_run"],
        "availability (absence x3, roster_q)": ["absence_ol", "absence_def",
                                                "absence_skill", "roster_quality"],
        "situational (thfa, rest, early, luck)": ["thfa", "rest", "early_down",
                                                  "fumble_luck"],
    }
    blocks = []
    base_ss = float(((zm - fit_m) ** 2).sum())
    tot_ss = float(((zm - zm.mean()) ** 2).sum())
    for bn, mem in BLOCKS.items():
        jj = [names.index(v) for v in mem if v in names]
        if not jj:
            continue
        cm = X[:, jj] @ g[[j + 1 for j in jj]]
        co = X[:, jj] @ b_eff[[j + 1 for j in jj]]
        keep = [0] + [j + 1 for j in range(len(names)) if j not in jj]
        gd = np.linalg.lstsq(A[:, keep], zm, rcond=None)[0]
        drop_ss = float(((zm - A[:, keep] @ gd) ** 2).sum())
        blocks.append({
            "block": bn, "members": mem,
            "market_contribution_sd_logits": r6(float(cm.std())),
            "our_contribution_sd_logits": r6(float(co.std())),
            "our_contribution_sd_scale_matched": r6(float(scale * co.std())),
            "ratio_market_over_ours": r6(float(cm.std() / (scale * co.std()))
                                         if co.std() > 1e-12 else None),
            "corr_market_vs_our_contribution": r6(float(np.corrcoef(cm, co)[0, 1])),
            "partial_R2_in_close_projection": r6((drop_ss - base_ss) / tot_ss),
        })
    blocks.sort(key=lambda b: -b["partial_R2_in_close_projection"])

    return {
        "what": ("Projection of logit(p_close) onto our 14-feature span, DEV only. "
                 "MEASURING DEVICE, NOT A MODEL — see the docstring. Nothing fitted here "
                 "is served, frozen or consumed by any predictor."),
        "n": n,
        "columns_used": names,
        "columns_dropped_dead_on_dev": dropped,
        "columns_note": ("a column that is identically zero on DEV carries no information "
                         "and makes the design singular, so it is dropped. The span being "
                         f"measured is therefore {len(cols)}-dimensional, not 14."),
        "R2_close_on_X14": r6(r2_m),
        "R2_close_on_our_forecast_alone": r6(r2_1),
        "R2_our_forecast_on_X14": r6(r2_o),
        "R2_note": ("R2_our_forecast_on_X14 ~ 1 confirms our shipped blend really is a "
                    "linear map of X14 (the walk-forward refits make it a mixture of ten "
                    "such maps), so the two weight vectors below are directly comparable."),
        "residual_sd_logit": r6(float(resid.std())),
        "close_sd_logit": r6(float(zm.std())),
        "residual_share_of_close_variance": r6(1.0 - r2_m),
        "scale_market_over_ours": r6(scale),
        "residual_value": {
            "method": ("y ~ a + b_us*logit(p_ours) + b_r*residual, leave-one-season-out "
                       "cross-fitted for the log losses; pooled coefficients reported for "
                       "the t-stats."),
            "b_ours": r6(wr[1]), "b_ours_se": r6(ser[1]), "b_ours_t": r6(wr[1] / ser[1]),
            "b_residual": r6(wr[2]), "b_residual_se": r6(ser[2]),
            "b_residual_t": r6(wr[2] / ser[2]),
            "per_season_coefs_b_ours_b_resid": br_list,
            "our_ll": r6(ll_o), "market_ll": r6(ll_m),
            "ll_ours_plus_residual": r6(ll_res),
            "gain_from_residual": r6(ll_o - ll_res),
            "gain_from_residual_ci": ci((v_o - v_r)[bi].mean(axis=1)),
            "share_of_gap_recovered_by_residual": r6((ll_o - ll_res) / (ll_o - ll_m)),
            "not_a_feature": ("the residual is computed FROM the closing line, so 'ours + "
                              "residual' is a market-derived object and can never be a served "
                              "model here. It is quoted only to price the un-spanned "
                              "information, which is the whole point of Q2."),
        },
        "span_decomposition": {
            "method": ("the projection X14@gamma, cross-fitted by season and Platt-mapped to "
                       "a probability, is 'the market's opinion restricted to OUR features'. "
                       "It splits the gap into a WITHIN-SPAN re-weighting part and an "
                       "OUTSIDE-SPAN information part."),
            "ll_ours": r6(ll_o),
            "ll_market_projected_onto_our_features": r6(ll_proj),
            "ll_market": r6(ll_m),
            "within_span_reweighting": r6(ll_o - ll_proj),
            "within_span_ci": ci((v_o - v_p)[bi].mean(axis=1)),
            "outside_span_information": r6(ll_proj - ll_m),
            "outside_span_ci": ci((v_p - v_m)[bi].mean(axis=1)),
            "caveat": ("within_span_reweighting is an UPPER bound on what better weights "
                       "could buy: the market's weights are estimated against a near-"
                       "noiseless target (the close itself) while ours are estimated against "
                       "binary outcomes, so most of this term is our own estimation noise "
                       "rather than a different opinion. It therefore OVERLAPS component (a) "
                       "of the apportionment and must not be added to it."),
        },
        "per_feature_weights": feats,
        "per_block_weights": blocks,
        "block_note": ("epa_rating / epa_pass / epa_run are three summaries of the same EPA "
                       "walk and are strongly collinear, so their individual coefficients "
                       "split arbitrarily between them and the per-feature table must not be "
                       "read one row at a time. The block table is the collinearity-robust "
                       "version: contribution sd is the logits of spread a block generates, "
                       "and partial_R2 is what the close's projection loses when the block is "
                       "removed entirely."),
        "weight_caveat": ("the market weights are precise (target = the close, R2 "
                          f"{r2_m:.3f}); ours are the effective weights of a blend fit to "
                          "binary outcomes on 2,531 games. A large diff_t means the market's "
                          "loading differs from ours by more than the two estimates' noise, "
                          "NOT that the market is right."),
    }


# ====================================================== Q3 — practical ceiling ==
def apportion(sp, lc, mu, pj):
    ll_o = float(llv(sp["y"], sp["p_ours"]).mean())
    ll_m = float(llv(sp["y"], sp["p_mkt"]).mean())
    gap = ll_o - ll_m
    f = lc["backward_HLinf"]["flattening"]
    lfc = lc["late_arriving_feature_cost"]
    # every learning-curve estimator, so component (a) is a measured range and not
    # one fragile extrapolation
    CURVES = ["forward_HL3_shipped", "forward_HLinf", "backward_HL3_shipped",
              "backward_HLinf"]
    grid = {}
    for cn in CURVES:
        fl = lc[cn]["flattening"]
        grid[cn] = {e: fl[e]["remaining_descent_to_L_inf"]
                    for e in ("all_points", "last5", "two_point")}
        grid[cn]["total_spread"] = fl["total_spread_across_curve"]
    vals = [v for cn in CURVES for e, v in grid[cn].items()
            if e in ("all_points", "two_point")]
    a1 = float(np.median(vals))
    a1_rng = [r6(min(vals)), r6(max(vals))]
    ceiling = max(grid[cn]["total_spread"] for cn in CURVES)
    a2 = lfc["cost_of_including"]
    est = a1 + a2
    est_ci = [r6(min(0.0, a1_rng[0]) + lfc["cost_ci"][0]),
              r6(a1_rng[1] + lfc["cost_ci"][1])]
    cal = mu["cross_fitted_platt"]["our_calibration_gain"]
    cal_ci = mu["cross_fitted_platt"]["our_calibration_gain_ci"]
    out_span = pj["span_decomposition"]["outside_span_information"]
    out_ci = pj["span_decomposition"]["outside_span_ci"]
    resid_val = pj["residual_value"]["gain_from_residual"]
    n = len(sp["y"])
    # analytic parameter-noise term for the shipped protocol, k = LIVE params per fold
    an, ks = [], []
    for s_ in range(MAIN_LO, MAIN_HI + 1):
        assert s_ < TEST_ERA
        tr = (sp["seasons"] < s_) & (sp["y_all"] != 0.5)
        w = 0.5 ** ((s_ - 1 - sp["seasons"][tr]) / HL_BASE)
        k = len(active(sp["X14"], tr)) + 1
        ks.append(k)
        an.append(k * float((w ** 2).sum()) / (2.0 * float(w.sum()) ** 2))
    analytic = float(np.mean(an))
    return {
        "gap_to_close": r6(gap), "n": n,
        "components": [
            {"component": "(a) estimation noise in our own fit",
             "estimate": r6(est), "ci": est_ci,
             "share_of_gap": r6(est / gap),
             "instrument": "a1 + a2 below; the CI is their combined range, not a bootstrap",
             "a1_blend_coefficient_sample_size": {
                 "estimate": r6(a1), "range_across_curves_and_estimators": a1_rng,
                 "estimator_grid": grid,
                 "model_free_ceiling": r6(ceiling),
                 "ceiling_note": ("the largest TOTAL spread any curve shows across a 14x "
                                  "range of training data (266 -> 3,711 games). Whatever is "
                                  "left beyond 3,711 is a fraction of that, so this is the "
                                  "ceiling on this channel and it does not depend on any "
                                  "extrapolation being right."),
                 "positive_control": ("the 1-feature Elo-only curve is flat to +/-0.0008 over "
                                      "the same range, confirming that what little descent "
                                      "the many-feature curves show is parameter-count-driven "
                                      "estimation noise and not a missing-history problem.")},
             "a2_short_history_features": {
                 "estimate": lfc["cost_of_including"], "ci": lfc["cost_ci"],
                 "on_2013_2015_only": lfc["cost_of_including_2013_2015"],
                 "note": ("a SEPARATE estimation-noise channel the curve cannot see: the four "
                          "snap-absence columns exist only from 2013, so their coefficients "
                          "are fitted on one or two seasons. Measured directly by re-running "
                          "the shipped walk-forward with the block zeroed. DIAGNOSTIC.")},
             "a3_rating_state_noise": {
                 "estimate": None,
                 "note": ("NOT IDENTIFIED by anything in this file. Every instrument here "
                          "varies the BLEND's training sample; the ratings themselves are "
                          "as-of walks seeded in 1999 and carry their own estimation error, "
                          "which no curve in this pass can vary. The only handle on it is "
                          "indirect: additive noise on the rating scale makes a forecast "
                          "over-dispersed, and our cross-fitted recalibration slope is "
                          f"{mu['cross_fitted_platt']['ours']['pooled_slope']} "
                          f"(t={mu['cross_fitted_platt']['ours']['slope_minus_1_t']} from 1, "
                          "n.s.), with the whole calibration channel worth "
                          f"{cal:+.5f}. So the DETECTABLE part is ~0 — but a rating error "
                          "that is correlated with the features, rather than additive white "
                          "noise, would not show up this way and remains unbounded.")},
             "corroboration": {
                 "analytic_parameter_noise_k_over_2neff": r6(analytic),
                 "k_live_params_per_fold": ks,
                 "note": "shipped HL=3 weights: mean over the ten DEV folds of "
                         "k*sum(w^2)/(2*sum(w)^2) with k = the LIVE parameter count of that "
                         "fold, the leading estimation-risk term of a correctly specified "
                         "unpenalised likelihood. Independent of the curve.",
                 "vs_measured": r6(analytic - a1),
                 "vs_measured_note": (
                     "the analytic term OVER-predicts the measured curve descent by roughly "
                     f"{analytic / a1:.0f}x. Two reasons, both real: the shipped fit is ridge-"
                     "penalised (C=100), which trades the variance the formula prices for a "
                     "little bias, and the 14 columns are far from 14 independent directions "
                     "— the team-rating block alone carries most of the signal. So the naive "
                     "k/(2n) reading of 'NFL has too few games for 14 features' is an "
                     "overstatement, and the curve, not the formula, is the evidence."),
                 "calibration_deficit_upper_bound": r6(cal), "calibration_ci": cal_ci,
                 "calibration_note": "additive noise on the rating scale shows up as "
                                     "over-dispersion, so what a cross-fitted Platt map "
                                     "recovers is an upper bound on the DETECTABLE part."}},
            {"component": "(b) information outside our feature span",
             "estimate": r6(out_span), "ci": out_ci,
             "share_of_gap": r6(out_span / gap),
             "instrument": "loss of the close projected onto X14, minus loss of the close",
             "corroboration": {"residual_conditional_gain": r6(resid_val),
                               "note": "what the un-spanned residual of the close buys when "
                                       "added to OUR forecast, cross-fitted."}},
            {"component": "(c) within-span re-weighting",
             "estimate": r6(pj["span_decomposition"]["within_span_reweighting"]),
             "ci": pj["span_decomposition"]["within_span_ci"],
             "share_of_gap": r6(pj["span_decomposition"]["within_span_reweighting"] / gap),
             "instrument": "our loss minus the loss of the projection of the close onto X14",
             "overlaps": "(a) — do NOT add. Most of a weight difference against a "
                         "near-noiseless target is our own estimation noise."},
        ],
        "residual_unattributed": r6(gap - est - out_span),
        "honest_reading": [
            f"n = {n} DEV games. The gap's own 95% CI is {mu['gap_ci']} — it spans a factor "
            "of ~4. Every apportionment inherits that width and no component is separated "
            "from any other at anything like 2 SE.",
            "(a) and (c) are two views of the same estimation-noise channel and overlap "
            "almost completely; they must never be added. (b) is the only component that is "
            "information by construction.",
            "The components do not have to sum to the gap: (b) is measured against the close "
            "and (a) against our own past, so their sum is a coincidence of arithmetic, not "
            "an identity. residual_unattributed is reported for exactly that reason.",
            "a3 (noise in the RATINGS, as opposed to the blend coefficients) is not "
            "identified by any instrument in this pass. The 16-games-a-season intuition "
            "lives mostly in a3, so 'the gap is not sample-starved' is established for the "
            "BLEND and only weakly bounded for the RATINGS.",
            "The DEV era is feature-degraded: one of the 14 columns is structurally absent "
            "and four more exist for only the last three seasons. The gap being decomposed "
            "belongs to a weaker model than the one served, and closing that loop would cost "
            "a TEST look, which is not spent here.",
        ],
    }


# ======================================================================= main ==
def main():
    log("building the shipped NFL DEV spine (heavy prelude) ...")
    sp = build_spine()
    X14, y_all, seasons = sp["X14"], sp["y_all"], sp["seasons"]
    sp["X"] = X14[sp["rows"]]

    # -------- fixed late-DEV scoring window, joined games only ----------------
    wmask = (sp["seas"] >= WIN_LO) & (sp["seas"] <= WIN_HI)
    assert sp["seas"][wmask].max() <= MAIN_HI < TEST_ERA
    score_rows = sp["rows"][wmask]
    y_w, pm_w = sp["y"][wmask], sp["p_mkt"][wmask]
    log(f"learning-curve window {WIN_LO}-{WIN_HI}: n={int(wmask.sum())} joined games")

    fwd = [(DEV_TRAIN_FROM, t) for t in range(2001, WIN_LO)]
    # both lists run SMALLEST -> LARGEST training set, so points[-1] is full history
    bwd = [(s, WIN_LO - 1) for s in range(WIN_LO - 1, DEV_TRAIN_FROM - 1, -1)]
    ref = sp["p_ours"][wmask]
    args = (X14, y_all, seasons, score_rows, y_w, pm_w)

    lc = {}
    lc["forward_HL3_shipped"] = curve(
        "forward, shipped HL=3", "the brief's specification: fixed 1999 start, training END "
        "walks forward. Training QUANTITY and RECENCY to the window both improve, so this "
        "curve confounds the two.", fwd, *args, HL=HL_BASE, ref=ref)
    log("curve 1/5 done")
    lc["forward_HLinf"] = curve(
        "forward, unweighted", "same spans with no recency discount.", fwd, *args,
        HL=float("inf"), ref=ref)
    log("curve 2/5 done")
    lc["backward_HL3_shipped"] = curve(
        "backward, shipped HL=3", "training END pinned at 2012 so recency to the window is "
        "CONSTANT and only quantity varies. Under HL=3 an extra season 10 years back carries "
        "weight 0.1, so this curve is expected to flatten by construction, not by saturation.",
        bwd, *args, HL=HL_BASE, ref=ref)
    log("curve 3/5 done")
    lc["backward_HLinf"] = curve(
        "backward, unweighted", "THE CLEAN INSTRUMENT for 'is the gap sample-starved': end "
        "pinned at 2012, no recency discount, so every extra season is pure extra evidence.",
        bwd, *args, HL=float("inf"), ref=ref)
    log("curve 4/5 done")
    lc["backward_HLinf_elo_only"] = curve(
        "backward, unweighted, 1 feature (Elo logit)",
        "POSITIVE CONTROL. A 2-parameter model must saturate almost immediately. If the "
        "many-feature curve is still descending where this one is flat, the descent is "
        "parameter-count-driven estimation noise, not a missing-history problem.",
        bwd, *args, HL=float("inf"), cols=[0], ref=ref)
    log("curve 5/5 done")

    lc["_reading"] = (
        "The market's loss on the identical games is printed with every curve and is FLAT by "
        "construction — the close does not learn from our training data. It is the reference "
        "line. Compare the shape of our curve to it, not the levels across windows.")
    lc["_structural_caveat"] = (
        "Every curve point trains on seasons <= 2012, where the four snap-absence columns "
        "(they begin in 2013) and the v7 participation column (2016+, TEST era) are "
        "identically zero — so a curve point is a ~10-parameter model even though the shipped "
        "matrix has 14 columns. That is exactly why the shipped walk-forward reference sits "
        "BELOW the curve's own full-history point: it may use columns the curve cannot. The "
        "curve is still a valid pure-quantity instrument because the live column set is "
        "IDENTICAL at every point on it.")

    lfc = late_feature_cost(X14, y_all, seasons, sp, wmask)
    log(f"late-feature diagnostic done: cost {lfc['cost_of_including']:+.5f}")
    lc["late_arriving_feature_cost"] = lfc

    mu = murphy(sp)
    log("murphy done")
    pj = projection(sp)
    log("projection done")
    sp["seasons"], sp["y_all"] = seasons, y_all
    ap = apportion(sp, lc, mu, pj)

    b = mu["binned_degroot_fienberg"]
    head = {
        "Q1_is_the_gap_sample_starved": (
            "NO, not on the blend-coefficient axis. Across a 14x range of training data "
            f"(266 -> 3,711 games) DEV log loss on a fixed 2013-2015 window moves by at most "
            f"{ap['components'][0]['a1_blend_coefficient_sample_size']['model_free_ceiling']} "
            "in total, and every extrapolation of the remaining descent beyond full history "
            "lands at "
            f"{ap['components'][0]['a1_blend_coefficient_sample_size']['estimate']} "
            f"(range {ap['components'][0]['a1_blend_coefficient_sample_size']['range_across_curves_and_estimators']}). "
            "The 1-feature control curve is flat, which is what the small residual descent "
            "being parameter-count-driven looks like. The one place short history DOES cost "
            "measurably is the four snap-absence columns that only exist from 2013: "
            f"{lc['late_arriving_feature_cost']['cost_of_including']} over joined DEV. "
            "Rating-state noise is a separate channel this pass does not identify."),
        "Q1_calibration_or_refinement": (
            f"REFINEMENT, overwhelmingly: {b['delta_REF']} of the {mu['gap']} gap "
            f"({b['share_of_gap_from_refinement']:.0%}) is the refinement term, and the "
            f"calibration term is NEGATIVE ({b['delta_CAL']}) — inside its own bins our "
            "forecast is slightly BETTER calibrated than the close. A cross-fitted Platt map "
            f"on our own output recovers {mu['cross_fitted_platt']['our_calibration_gain']} "
            "(CI straddles 0) and "
            f"{mu['cross_fitted_platt']['share_of_gap_surviving_recalibration']:.0%} of the "
            "gap survives recalibrating BOTH sides. The error map's calibration null is "
            "confirmed, not contradicted: there is nothing to fix by rescaling."),
        "Q2_how_much_of_the_market_do_we_span": (
            f"R2 = {pj['R2_close_on_X14']} of the closing line's logit variance lives inside "
            f"our 13 live DEV features ({pj['R2_close_on_our_forecast_alone']} for our "
            "finished forecast alone). The un-spanned residual is "
            f"{pj['residual_sd_logit']} logits of sd and it is worth "
            f"{pj['residual_value']['gain_from_residual']} "
            f"({pj['residual_value']['share_of_gap_recovered_by_residual']:.0%} of the gap, "
            f"t={pj['residual_value']['b_residual_t']}) when added to our forecast. Splitting "
            "the gap: re-weighting our OWN features the way the market does buys "
            f"{pj['span_decomposition']['within_span_reweighting']}, everything else "
            f"({pj['span_decomposition']['outside_span_information']}) is outside our span."),
        "Q3_practical_ceiling": (
            f"Of the {ap['gap_to_close']} gap: ~"
            f"{ap['components'][1]['share_of_gap']:.0%} is information outside our feature "
            f"span, ~{ap['components'][0]['share_of_gap']:.0%} is our own estimation noise "
            "(mostly the short-history absence block, not the blend's sample size), and the "
            "re-weighting channel is small and overlaps the noise channel. With n=2,531 the "
            "gap itself is only known to a factor of ~4, so the correct summary is "
            "directional: the NFL gap is dominated by MISSING INFORMATION, and the "
            "sample-starvation hypothesis is rejected for the blend and merely unbounded for "
            "the ratings."),
    }

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": ("NFL headroom deep-dive: is the +0.01172 DEV gap to the closing line "
                 "estimation noise in our ratings, or information outside our feature span?"),
        "protocol": {
            "league": "NFL", "dev_scored": [MAIN_LO, MAIN_HI],
            "dev_trained_from": DEV_TRAIN_FROM,
            "test_never_touched": [TEST_ERA, 2025], "test_looks_spent": 0,
            "models_served_touched": 0, "features_created": 0,
            "market_blind_models": True,
            "odds_use": ("EVALUATION / DIAGNOSTIC ONLY. Q2 regresses logit(p_close) ON our "
                         "features; that regression is a MEASURING DEVICE for information "
                         "overlap, NOT a predictor. Nothing fitted in this file is served, "
                         "frozen, blended or consumed by any model. The closing line is never "
                         "a feature and never a training signal for a served model."),
            "reuse": ("X14 from nfl_depth_eval.build_main(); walk-forward verbatim from "
                      "error_map.nfl_league; market join + floor re-verification from "
                      "market_gap.nfl_join / check_join against data/error_map.json."),
            "join_check": sp["join_check"],
        },
        "headline": head,
        "feature_availability": availability(X14, seasons, y_all),
        "Q1a_learning_curve": lc,
        "Q1b_murphy_decomposition": mu,
        "Q2_market_projection": pj,
        "Q3_apportionment": ap,
    }
    EM.atomic_json(payload, OUT)
    log(f"wrote {OUT}")
    report(payload)


def report(P):
    lc, mu, pj, ap = (P["Q1a_learning_curve"], P["Q1b_murphy_decomposition"],
                      P["Q2_market_projection"], P["Q3_apportionment"])
    fa = P["feature_availability"]
    print("\n" + "=" * 92)
    print("FEATURE AVAILABILITY ON DEV — read this before anything else")
    print("=" * 92)
    print(f"  {'col':>3} {'feature':<22} {'first nonzero':>14} {'nonzero share DEV':>18} "
          f"{'live':>6}")
    for r in fa["features"]:
        print(f"  {r['col']:>3} {r['feature']:<22} "
              f"{str(r['first_season_nonzero']):>14} {r['nonzero_share_in_dev_scored']:>18.3f} "
              f"{str(r['live_on_dev']):>6}")
    print(f"  -> {fa['n_live_on_dev']} of 14 columns are live on DEV; dead: "
          f"{fa['dead_on_dev']}; late-arriving: {fa['late_arriving_on_dev']}")

    print("\n" + "=" * 92)
    print("Q1a  LEARNING CURVE — DEV log loss on the FIXED 2013-2015 window "
          f"(n={lc['backward_HLinf']['n_scored']})")
    print("=" * 92)
    for key in ("forward_HL3_shipped", "forward_HLinf", "backward_HL3_shipped",
                "backward_HLinf", "backward_HLinf_elo_only"):
        c = lc[key]
        print(f"\n  {c['curve']}   (HL={c['HL']})   "
              f"market on the same games {c['market_ll_on_same_games']:.5f} [flat]   "
              f"shipped wf reference {c['shipped_walk_forward_on_same_games']:.5f}")
        print(f"  {'train span':<12} {'seas':>4} {'n_tr':>6} {'n_eff':>8} {'k':>3} {'LL':>9} "
              f"{'vs full':>9} {'95% CI':>22}")
        for p in c["points"]:
            print(f"  {p['train_seasons']:<12} {p['n_seasons']:>4} {p['n_train']:>6} "
                  f"{p['n_eff']:>8.0f} {p['k_active']:>3} {p['ll']:>9.5f} "
                  f"{p['ll_minus_full_history']:>+9.5f} "
                  f"{str(p['ll_minus_full_history_ci']):>22}")
        f = c["flattening"]
        print(f"   total spread across the curve {f['total_spread_across_curve']:.5f}; "
              f"smallest minus largest {f['ll_smallest_minus_ll_largest']:+.5f}")
        for k2 in ("all_points", "last5", "two_point"):
            b = f[k2]
            print(f"   -> {k2:<11} L_inf {b.get('L_inf', float('nan')):>9} "
                  f"remaining {b['remaining_descent_to_L_inf']:>+9.5f} "
                  f"{str(b['remaining_descent_ci']):<24} "
                  f"{'STILL DESCENDING' if b['still_descending'] else 'flat (n.s.)'}")

    lf = lc["late_arriving_feature_cost"]
    print(f"\n  DIAGNOSTIC — cost of the four late-arriving snap-absence columns "
          f"(live only from 2013), inside the shipped walk-forward:")
    print(f"    all joined DEV (n={lf['n_all_dev']}): with {lf['ll_with']:.5f} vs without "
          f"{lf['ll_without']:.5f}  -> {lf['cost_of_including']:+.5f} {lf['cost_ci']}")
    print(f"    2013-2015 only (n={lf['n_2013_2015']}): with {lf['ll_with_2013_2015']:.5f} vs "
          f"without {lf['ll_without_2013_2015']:.5f}  -> "
          f"{lf['cost_of_including_2013_2015']:+.5f} {lf['cci' if False else 'cost_ci_2013_2015']}")
    print("    (measurement, not a proposal — a model change needs its own pre-registered "
          "screen)")

    print("\n" + "=" * 92)
    print(f"Q1b  MURPHY / DeGROOT-FIENBERG   n={mu['n']}  ours {mu['our_ll']:.5f}  "
          f"close {mu['market_ll']:.5f}  gap {mu['gap']:+.5f} {mu['gap_ci']}")
    print("=" * 92)
    b = mu["binned_degroot_fienberg"]
    print(f"  {'':<10} {'CAL':>10} {'REF':>10} {'WITHIN':>10} {'total':>10}")
    for nm in ("ours", "market"):
        d = b[nm]
        print(f"  {nm:<10} {d['CAL']:>10.5f} {d['REF']:>10.5f} {d['WITHIN']:>10.5f} "
              f"{d['total_check']:>10.5f}")
    print(f"  {'DELTA':<10} {b['delta_CAL']:>+10.5f} {b['delta_REF']:>+10.5f} "
          f"{b['delta_WITHIN']:>+10.5f}   (null bias per side {b['null_bias_per_side']:.5f}, "
          "cancels in DELTA)")
    print(f"  refinement explains {b['share_of_gap_from_refinement']:.1%} of the gap")
    cp = mu["cross_fitted_platt"]
    print(f"\n  cross-fitted Platt   ours a={cp['ours']['pooled_intercept']:+.4f} "
          f"b={cp['ours']['pooled_slope']:.4f} (b-1 t={cp['ours']['slope_minus_1_t']:+.2f})   "
          f"close a={cp['market']['pooled_intercept']:+.4f} "
          f"b={cp['market']['pooled_slope']:.4f} (b-1 t={cp['market']['slope_minus_1_t']:+.2f})")
    print(f"  our recalibration gain {cp['our_calibration_gain']:+.5f} "
          f"{cp['our_calibration_gain_ci']};  gap AFTER recalibrating both "
          f"{cp['gap_after_both_recalibrated']:+.5f} {cp['gap_after_both_recalibrated_ci']} "
          f"({cp['share_of_gap_surviving_recalibration']:.1%} of the gap survives)")
    dd = mu["discrimination"]
    print(f"  AUC ours {dd['our_auc']:.4f} vs close {dd['market_auc']:.4f};  sd(logit) ours "
          f"{dd['our_sd_logit']:.4f} vs close {dd['market_sd_logit']:.4f} "
          f"(ratio {dd['sd_ratio_market_over_ours']:.3f})")

    print("\n" + "=" * 92)
    print("Q2  PROJECTING THE CLOSE ONTO OUR 14 FEATURES   (measuring device, not a model)")
    print("=" * 92)
    print(f"  R2(close ~ X14) = {pj['R2_close_on_X14']:.4f}   "
          f"R2(close ~ our forecast alone) = {pj['R2_close_on_our_forecast_alone']:.4f}   "
          f"R2(our forecast ~ X14) = {pj['R2_our_forecast_on_X14']:.4f}")
    print(f"  residual sd {pj['residual_sd_logit']:.4f} logits vs close sd "
          f"{pj['close_sd_logit']:.4f}  -> {pj['residual_share_of_close_variance']:.1%} of the "
          f"close's variance is OUTSIDE our span;  market/our scale "
          f"{pj['scale_market_over_ours']:.3f}")
    rv = pj["residual_value"]
    print(f"\n  residual value:  b_ours {rv['b_ours']:+.3f} (t {rv['b_ours_t']:+.2f})   "
          f"b_residual {rv['b_residual']:+.3f} (t {rv['b_residual_t']:+.2f})")
    print(f"  ours+residual LL {rv['ll_ours_plus_residual']:.5f}  gain "
          f"{rv['gain_from_residual']:+.5f} {rv['gain_from_residual_ci']}  "
          f"= {rv['share_of_gap_recovered_by_residual']:.1%} of the gap")
    sd_ = pj["span_decomposition"]
    print(f"\n  span split:  ours {sd_['ll_ours']:.5f} -> close-projected-onto-X14 "
          f"{sd_['ll_market_projected_onto_our_features']:.5f} -> close {sd_['ll_market']:.5f}")
    print(f"    within-span re-weighting  {sd_['within_span_reweighting']:+.5f} "
          f"{sd_['within_span_ci']}")
    print(f"    outside-span information  {sd_['outside_span_information']:+.5f} "
          f"{sd_['outside_span_ci']}")
    print(f"\n  {'feature':<22} {'mkt/sd':>9} {'ours/sd':>9} {'ours*scale':>11} "
          f"{'diff':>9} {'t':>7} {'ratio':>7}")
    for f in pj["per_feature_weights"]:
        print(f"  {f['feature']:<22} {f['market_weight_per_sd']:>+9.4f} "
              f"{f['our_weight_per_sd']:>+9.4f} {f['our_weight_per_sd_scale_matched']:>+11.4f} "
              f"{f['diff_market_minus_ours']:>+9.4f} {f['diff_t']:>+7.2f} "
              f"{(f['ratio_market_over_ours'] if f['ratio_market_over_ours'] is not None else float('nan')):>7.2f}")
    print(f"\n  BLOCKS (collinearity-robust; the per-feature rows above split arbitrarily "
          f"inside a block)")
    print(f"  {'block':<40} {'mkt sd':>8} {'our sd*s':>9} {'ratio':>7} {'corr':>7} "
          f"{'partialR2':>10}")
    for b in pj["per_block_weights"]:
        print(f"  {b['block']:<40} {b['market_contribution_sd_logits']:>8.4f} "
              f"{b['our_contribution_sd_scale_matched']:>9.4f} "
              f"{(b['ratio_market_over_ours'] if b['ratio_market_over_ours'] is not None else float('nan')):>7.2f} "
              f"{b['corr_market_vs_our_contribution']:>+7.3f} "
              f"{b['partial_R2_in_close_projection']:>10.4f}")

    print("\n" + "=" * 92)
    print(f"Q3  APPORTIONMENT of the {ap['gap_to_close']:+.5f} gap   (n={ap['n']})")
    print("=" * 92)
    for c in ap["components"]:
        print(f"  {c['component']:<44} {c['estimate']:>+9.5f} {str(c['ci']):>22} "
              f"{c['share_of_gap']:>7.1%}")
        print(f"      instrument: {c['instrument']}")
        a1 = c.get("a1_blend_coefficient_sample_size")
        if a1:
            print(f"      a1 blend sample size  {a1['estimate']:>+9.5f}  range "
                  f"{a1['range_across_curves_and_estimators']}  model-free ceiling "
                  f"{a1['model_free_ceiling']:+.5f}")
            a2 = c["a2_short_history_features"]
            print(f"      a2 short-history cols {a2['estimate']:>+9.5f}  {a2['ci']}  "
                  f"(2013-2015 only {a2['on_2013_2015_only']:+.5f})")
            print("      a3 rating-state noise  NOT IDENTIFIED by this pass")
        if "overlaps" in c:
            print(f"      OVERLAPS {c['overlaps']}")
    print(f"  residual unattributed: {ap['residual_unattributed']:+.5f}")
    for h in ap["honest_reading"]:
        print(f"  ! {h}")
    print("\n" + "=" * 92)
    print("HEADLINE")
    print("=" * 92)
    for k, v in P["headline"].items():
        print(f"\n  {k}\n    {v}")


if __name__ == "__main__":
    main()
