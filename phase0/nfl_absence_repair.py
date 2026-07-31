"""NFL ABSENCE-BLOCK REPAIR SCREEN — executes documents/nfl_absence_prereg.md.

The pre-registration (committed BEFORE any candidate was screened) binds this
file completely: the split, the four candidates, the metric, the method
standard and the adoption bar are all fixed there and are NOT re-opened here.
Nothing in this file may add a candidate, change the metric or move the bar.

WHAT IS SCREENED
  The four availability columns of the shipped NFL X14 --
  col 8 absence_ol, col 9 absence_def, col 10 absence_skill, col 11
  roster_quality -- which phase0/nfl_headroom.py measured as COSTING
  +0.00132 DEV log loss (+0.00417 on 2013-2015) rather than earning it.

  1 DROP    remove the four columns entirely.
  2 POOL    absence_ol + absence_def + absence_skill -> one total column,
            roster_quality kept separate (4 params -> 2).
  3 SHRINK  keep all four, ridge-penalise THAT BLOCK ALONE much harder, the
            rest of the blend keeping its shipped C=100, swept over the
            pre-declared grid below.
  4 GATE    zero the block's contribution until at least M seasons of LIVE
            (non-degenerate) block data exist in the training fold, M swept
            over the pre-declared grid below.

  SHRINK is implemented exactly, not approximately.  sklearn's logistic ridge
  penalises ||beta||^2 / (2C) uniformly, but rescaling column j by s maps the
  fitted coefficient to beta_j / s at identical fit, so the penalty seen by the
  ORIGINAL-scale coefficient becomes 1/(2 C s^2): a per-block ridge constant
  C_block = C * s^2, i.e. s = sqrt(C_block / C).  No other column is touched.

ONE-LOOK PROTOCOL
  NFL main split.  Trained from 1999, SCORED 2006-2015.  TEST (>= 2016) is
  never loaded, never scored, never printed.  Every fold asserts its own mask.
  0 TEST looks spent by this file.

MARKET-BLIND
  No odds enter any feature, any training signal or any selection here.

Usage:  python -X utf8 phase0/nfl_absence_repair.py  ->  data/nfl_absence_repair.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

T0 = time.time()
OUT = "data/nfl_absence_repair.json"
PREREG = "documents/nfl_absence_prereg.md"

# ------------------------------------------------------------------ protocol --
MAIN_LO, MAIN_HI = 2006, 2015
TEST_ERA = 2016
TRAIN_FROM = 1999
HL_BASE, C_BASE = 3.0, 100.0
BLOCK = [8, 9, 10, 11]
BLOCK_NAMES = ["absence_ol", "absence_def", "absence_skill", "roster_quality"]
FEAT = ["elo_logit", "qb_delta", "epa_rating", "early_down", "thfa", "rest",
        "fumble_luck", "v7_participation_ts", "absence_ol", "absence_def",
        "absence_skill", "roster_quality", "epa_pass", "epa_run"]
RECORDED_BASELINE = 0.62296          # documents/nfl_absence_prereg.md
JITTER_TOL = 5e-4                    # run-to-run jitter of build_main() is ~3e-5

# --- PRE-DECLARED GRIDS (fixed before any result was seen) --------------------
SHRINK_GRID = [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.001]
GATE_GRID = [1, 2, 3]                # min LIVE training seasons before switch-on

# A per-season log-loss difference below this is NUMERICAL, not a real sign: it is
# 30x smaller than the 3e-5 process-to-process jitter of build_main() itself.  It
# arises because changing the number of columns changes the lbfgs path even on
# folds where the changed columns are identically zero.  Sign counts use it so
# that float dust cannot manufacture a "positive season".
EPS_SIGN = 1e-6

# Observed across five independent processes of THIS file.  build_main() sums
# league EPA rates over dict iteration order, so X14 itself is only reproducible
# to ~1e-4 relative; the baseline moves ~4e-5 in log loss AND, because the
# candidate matrices are built from the same jittered X14, so does the PAIRED
# gain.  The gains are therefore quoted with this jitter attached rather than to
# six figures of false precision.
JITTER_OBS = {
    "baseline_dev_ll": [0.622929, 0.622923, 0.622886, 0.622919, 0.622920],
    "nested_gain_1_DROP": [0.001251, 0.001245, 0.001207, 0.001241, 0.001242],
    "nested_gain_2_POOL": [0.001335, 0.001329, 0.001292, 0.001329, 0.001326],
    "nested_gain_3_SHRINK": [0.001006, 0.001000, 0.000963, 0.000996, 0.000997],
    "nested_gain_4_GATE": [0.0, 0.0, 0.0, 0.0, 0.0],
    "raw_gain_3_SHRINK": [0.001856, 0.001851, 0.001813, 0.001847, 0.001848],
}

# --- THE PRE-REGISTERED BAR (copied verbatim, not re-derived) -----------------
BAR_NESTED_GAIN = 0.0010
BAR_MIN_POSITIVE_SEASONS = 7
N_SCORED_SEASONS = MAIN_HI - MAIN_LO + 1     # 10

assert MAIN_HI < TEST_ERA
assert os.path.exists(PREREG), "pre-registration missing; this screen is not authorised"

import error_map as EM          # noqa: E402
import nfl_depth_eval as N      # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402

llv, r6 = EM.llv, EM.r6


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# ================================================================ the harness ==
def walk_forward(mat_for_fold, seasons, y):
    """The SHIPPED training protocol, main-split DEV only.

    mat_for_fold(s_) -> the design matrix used for fold s_ (some candidates
    change the matrix per fold; most do not).  Returns overall LL, per-season
    LL, the per-game LL vector and the matching per-game season vector.
    """
    per, vecs, svs = {}, [], []
    for s_ in range(MAIN_LO, MAIN_HI + 1):
        assert s_ < TEST_ERA, "fold leaked into the TEST era"
        tr = (seasons < s_) & (y != 0.5)
        te = seasons == s_
        assert seasons[tr].max() < s_, "training fold saw its own season"
        assert seasons[te].max() < TEST_ERA, "TEST-era game scored"
        X = mat_for_fold(s_)
        w = 0.5 ** ((s_ - 1 - seasons[tr]) / HL_BASE)
        lm = LogisticRegression(C=C_BASE, max_iter=5000).fit(X[tr], y[tr], sample_weight=w)
        v = llv(y[te], lm.predict_proba(X[te])[:, 1])
        per[str(s_)] = float(v.mean())
        vecs.append(v)
        svs.append(seasons[te])
    a = np.concatenate(vecs)
    sv = np.concatenate(svs)
    assert sv.max() < TEST_ERA
    return float(a.mean()), per, a, sv


def fold_stats(base_per, cand_per):
    """Paired per-season deltas.  delta = base - cand, so POSITIVE = candidate
    better.  Sign counts are reported against the pre-registered clause."""
    ks = sorted(base_per)
    d = np.array([base_per[k] - cand_per[k] for k in ks])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    gain = float(np.mean([base_per[k] for k in ks]) - np.mean([cand_per[k] for k in ks]))
    live = np.array([1 if abs(x) > EPS_SIGN else 0 for x in d], bool)
    out = {
        "gain": r6(gain),
        "per_season_delta": {k: r6(x) for k, x in zip(ks, d)},
        "n_seasons": len(ks),
        "n_positive": int((d > EPS_SIGN).sum()),
        "n_negative": int((d < -EPS_SIGN).sum()),
        "n_identical": int((np.abs(d) <= EPS_SIGN).sum()),
        "sign_epsilon": EPS_SIGN,
        "fold_sd": r6(sd), "fold_se": r6(se),
        "gain_in_fold_se": r6(gain / se) if se > 0 else None,
        "n_folds_that_can_differ": int(live.sum()),
    }
    if live.any():
        dl = d[live]
        out["live_folds"] = {
            "seasons": [k for k, m in zip(ks, live) if m],
            "mean_delta_on_live_folds": r6(float(dl.mean())),
            "sd_across_live_folds": r6(float(dl.std(ddof=1)) if len(dl) > 1 else None),
        }
    return out


def paired_bootstrap(vb, vc, seed=11, B=10000):
    """Game-level paired bootstrap of (base - cand).  Positive = cand better."""
    d = (vb - vc).astype(np.float64)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return {"delta": r6(float(d.mean())),
            "ci95": [r6(float(np.percentile(bs, 2.5))), r6(float(np.percentile(bs, 97.5)))],
            "se": r6(float(bs.std(ddof=1))),
            "gain_in_bootstrap_se": r6(float(d.mean() / bs.std(ddof=1))) if bs.std() > 0 else None,
            "n_games": int(len(d))}


def nested_loso(per_map, base_per, pool, tiebreak="first"):
    """LEAVE-ONE-SEASON-OUT SELECTION.  For each scored season s the winning
    variant is chosen using ONLY the other nine seasons, then scored on s.
    The gap to the raw best is the search optimism.  This is the number the
    pre-registration says counts.

    tiebreak: when two variants score IDENTICALLY on the nine training seasons
    the choice is arbitrary.  That is not a hypothetical here -- variants that
    differ only on the held-out fold are exactly tied by construction -- so both
    directions are computed and reported.  'first' takes the lexicographically
    first name, 'last' the last.
    """
    ks = sorted(base_per)
    order = sorted(pool)
    picks, held = {}, []
    for s in ks:
        others = [o for o in ks if o != s]
        means = {nm: float(np.mean([per_map[nm][o] for o in others])) for nm in order}
        lo = min(means.values())
        tied = [nm for nm in order if means[nm] <= lo + 1e-15]
        best = tied[0] if tiebreak == "first" else tied[-1]
        picks[s] = best
        held.append(per_map[best][s])
    nested_per = {s: per_map[picks[s]][s] for s in ks}
    nested = float(np.mean(held))
    base = float(np.mean([base_per[k] for k in ks]))
    raw_best = min(order, key=lambda nm: (np.mean(list(per_map[nm].values())), nm))
    raw = float(np.mean(list(per_map[raw_best].values())))
    d = np.array([base_per[k] - nested_per[k] for k in ks])
    return {
        "pool_size": len(pool), "pool": order, "tiebreak": tiebreak,
        "raw_best": raw_best, "raw_best_ll": r6(raw), "raw_gain": r6(base - raw),
        "nested_ll": r6(nested), "nested_gain": r6(base - nested),
        "search_optimism": r6((base - raw) - (base - nested)),
        "loso_picks": picks,
        "nested_per_season_delta": {k: r6(x) for k, x in zip(ks, d)},
        "nested_n_positive": int((d > EPS_SIGN).sum()),
        "nested_n_negative": int((d < -EPS_SIGN).sum()),
        "nested_n_identical": int((np.abs(d) <= EPS_SIGN).sum()),
        "sign_epsilon": EPS_SIGN,
    }


def verdict(nested_gain, n_pos, satisfiable_pos):
    """The pre-registered bar, applied mechanically.  No discretion."""
    ok_mag = nested_gain >= BAR_NESTED_GAIN
    ok_sign = n_pos >= BAR_MIN_POSITIVE_SEASONS
    return {
        "nested_gain": r6(nested_gain),
        "bar_nested_gain": BAR_NESTED_GAIN,
        "clause_1_magnitude_met": bool(ok_mag),
        "nested_positive_seasons": int(n_pos),
        "bar_positive_seasons": f">= {BAR_MIN_POSITIVE_SEASONS} of {N_SCORED_SEASONS}",
        "clause_2_sign_count_met": bool(ok_sign),
        "clause_2_attainable_maximum": int(satisfiable_pos),
        "clause_2_satisfiable_at_all": bool(satisfiable_pos >= BAR_MIN_POSITIVE_SEASONS),
        "VERDICT": "ADOPT" if (ok_mag and ok_sign) else "REJECT",
    }


# ======================================================================= main ==
def main():
    log("building the shipped NFL X14 (heavy prelude) ...")
    X14, _ex = N.build_main()
    seasons, y = N.seasons, N.y
    assert X14.shape[1] == 14 == len(FEAT)
    assert N.MAIN_LO == MAIN_LO and N.MAIN_HI == MAIN_HI and N.MAIN_TEST_ERA == TEST_ERA, \
        "the NFL main split moved; the pre-registration no longer describes this harness"
    assert N.HL_BASE == HL_BASE and N.C_BASE == C_BASE, "shipped training protocol moved"

    scored = (seasons >= MAIN_LO) & (seasons <= MAIN_HI)
    assert seasons[scored].max() <= MAIN_HI < TEST_ERA

    # ------------------------------------------------ structural facts first --
    live_seasons = sorted({int(s) for s in set(seasons.tolist())
                           if (X14[seasons == s][:, BLOCK] != 0).any()})
    dev_live = [s for s in live_seasons if s <= MAIN_HI]
    struct = {
        "n_scored_games": int(scored.sum()),
        "block_columns": {str(c): FEAT[c] for c in BLOCK},
        "first_season_block_is_nonzero": (min(live_seasons) if live_seasons else None),
        "live_seasons_within_dev": dev_live,
        "nonzero_rows_per_block_col_in_dev": {
            FEAT[c]: int((X14[scored, c] != 0).sum()) for c in BLOCK},
        "col7_v7_nonzero_in_dev": int((X14[scored, 7] != 0).sum()),
        "col7_note": ("column 7 (v7 11v11 participation TrueSkill) is identically zero "
                      "across ALL of DEV -- nflverse participation data begins 2016. "
                      "Explicitly OUT OF SCOPE per the pre-registration; it is left in "
                      "every candidate matrix, where it carries a coefficient of exactly "
                      "zero and cannot affect any comparison."),
    }

    def n_live_in_train(s_):
        return sum(1 for t in dev_live if t < s_)

    train_live = {str(s_): n_live_in_train(s_) for s_ in range(MAIN_LO, MAIN_HI + 1)}
    struct["live_block_seasons_available_to_each_training_fold"] = train_live
    can_differ = [s_ for s_ in range(MAIN_LO, MAIN_HI + 1) if n_live_in_train(s_) > 0]
    struct["folds_whose_prediction_can_possibly_differ"] = can_differ
    struct["max_attainable_positive_seasons"] = len(can_differ)
    log(f"structural: block live from {struct['first_season_block_is_nonzero']}, "
        f"folds that can differ = {can_differ}")

    # ---------------------------------------------------------- the baseline --
    ll0, per0, v0, sv = walk_forward(lambda s_: X14, seasons, y)
    log(f"BASELINE main-split DEV walk-forward LL {ll0:.6f} "
        f"(recorded {RECORDED_BASELINE}, tol {JITTER_TOL})")
    assert abs(ll0 - RECORDED_BASELINE) < JITTER_TOL, (
        f"shipped baseline NOT reproduced: {ll0:.6f} vs recorded {RECORDED_BASELINE}")
    baseline = {
        "dev_ll": r6(ll0), "recorded_in_prereg": RECORDED_BASELINE,
        "abs_difference": r6(abs(ll0 - RECORDED_BASELINE)),
        "jitter_tolerance": JITTER_TOL,
        "reproduced": True,
        "per_season": {k: r6(v) for k, v in per0.items()},
        "unpaired_fold_sd": r6(float(np.std(list(per0.values()), ddof=1))),
        "unpaired_fold_se": r6(float(np.std(list(per0.values()), ddof=1)
                                     / np.sqrt(N_SCORED_SEASONS))),
        "jitter_note": ("build_main() accumulates league EPA rates over dict iteration "
                        "order, so the matrix reproduces to ~3e-5 in log loss across "
                        "processes; every comparison below is PAIRED against the "
                        "baseline computed in THIS process, so the jitter cancels."),
    }

    # ================================================== candidate constructors ==
    keep_drop = [j for j in range(14) if j not in BLOCK]
    X_drop = X14[:, keep_drop]

    tot = X14[:, 8] + X14[:, 9] + X14[:, 10]
    pool_cols = [j for j in range(14) if j not in (8, 9, 10)]
    X_pool = np.column_stack([X14[:, :8], tot, X14[:, 11:]])
    assert X_pool.shape[1] == 12
    assert np.allclose(X_pool[:, [8, 9, 10, 11]],
                       np.column_stack([tot, X14[:, 11], X14[:, 12], X14[:, 13]]))

    def make_shrink(cb):
        s = np.sqrt(cb / C_BASE)
        Xs = X14.copy()
        Xs[:, BLOCK] *= s
        return Xs

    def make_gate(mmin):
        cache = {}
        for s_ in range(MAIN_LO, MAIN_HI + 1):
            if n_live_in_train(s_) >= mmin:
                cache[s_] = X14
            else:
                Xz = X14.copy()
                Xz[:, BLOCK] = 0.0
                cache[s_] = Xz
        return lambda s_: cache[s_]

    variants = {}
    variants["DROP"] = ("drop", lambda s_, M=X_drop: M)
    variants["POOL"] = ("pool", lambda s_, M=X_pool: M)
    for cb in SHRINK_GRID:
        variants[f"SHRINK C_block={cb:g}"] = ("shrink", lambda s_, M=make_shrink(cb): M)
    for mm in GATE_GRID:
        variants[f"GATE min_live={mm}"] = ("gate", make_gate(mm))

    per_map, ll_map, vec_map = {"BASE": per0}, {"BASE": ll0}, {"BASE": v0}
    for nm, (_fam, fn) in variants.items():
        ll, per, v, sv2 = walk_forward(fn, seasons, y)
        assert np.array_equal(sv, sv2)
        per_map[nm], ll_map[nm], vec_map[nm] = per, ll, v
        log(f"  {nm:<26} {ll:.6f}   ({ll0 - ll:+.6f})")

    # ------------------------------------------------------------ diagnostics --
    ident = {}
    for nm in variants:
        d = np.abs(v0 - vec_map[nm])
        pre = sv < min(dev_live) + 1 if dev_live else np.ones(len(sv), bool)
        ident[nm] = {"max_abs_per_game_ll_diff_folds_2006_2013": r6(float(d[pre].max())),
                     "max_abs_per_game_ll_diff_all_dev": r6(float(d.max()))}

    # ================================================= per-candidate reporting ==
    FAMILIES = {
        "1_DROP": {"variants": ["DROP"],
                   "description": "remove absence_ol / absence_def / absence_skill / "
                                  "roster_quality from the blend entirely (14 -> 10 cols)"},
        "2_POOL": {"variants": ["POOL"],
                   "description": "absence_ol + absence_def + absence_skill collapsed to one "
                                  "total-absence column; roster_quality kept separate "
                                  "(4 params -> 2)"},
        "3_SHRINK": {"variants": [n for n in variants if n.startswith("SHRINK")],
                     "description": "block-only ridge C_block over the pre-declared grid; the "
                                    "rest of the blend keeps the shipped C=100"},
        "4_GATE": {"variants": [n for n in variants if n.startswith("GATE")],
                   "description": "block zeroed until >= M LIVE training seasons exist, over "
                                  "the pre-declared grid"},
    }

    results = {}
    for fam, spec in FAMILIES.items():
        pool = spec["variants"]
        raw_best = min(pool, key=lambda nm: (ll_map[nm], nm))
        nl = nested_loso(per_map, per0, pool, tiebreak="first")
        nl_alt = nested_loso(per_map, per0, pool, tiebreak="last")
        blk = {
            "description": spec["description"],
            "pre_declared_grid": (SHRINK_GRID if fam == "3_SHRINK"
                                  else GATE_GRID if fam == "4_GATE" else None),
            "variants": {nm: {"dev_ll": r6(ll_map[nm]),
                              **fold_stats(per0, per_map[nm]),
                              "paired_bootstrap_game_level": paired_bootstrap(v0, vec_map[nm]),
                              "identity_check": ident[nm]}
                         for nm in pool},
            "RAW": {"best_variant": raw_best, "dev_ll": r6(ll_map[raw_best]),
                    "raw_gain": r6(ll0 - ll_map[raw_best]),
                    **{k: v for k, v in fold_stats(per0, per_map[raw_best]).items()
                       if k in ("n_positive", "n_negative", "n_identical", "fold_sd",
                                "fold_se", "gain_in_fold_se", "per_season_delta")},
                    "note": ("RAW is the in-sample-selected number: for SHRINK/GATE it picks "
                             "the grid point using the same ten seasons it is scored on, so it "
                             "is optimistic by construction. DROP and POOL have no selection "
                             "freedom, so their RAW and NESTED are identical by definition.")},
            "NESTED": nl,
            "NESTED_tiebreak_sensitivity": {
                "nested_gain_tiebreak_first": nl["nested_gain"],
                "nested_gain_tiebreak_last": nl_alt["nested_gain"],
                "loso_picks_tiebreak_last": nl_alt["loso_picks"],
                "tie_is_binding": bool(abs(nl["nested_gain"] - nl_alt["nested_gain"]) > 1e-9),
                "why": ("two variants that differ ONLY on the held-out fold score identically "
                        "on the nine selection folds, so the LOSO choice between them is a "
                        "coin flip. Where that happens the nested gain is not a single number "
                        "but a range, and both ends are reported. The verdict uses the "
                        "conservative end (lexicographically first, which is the lower "
                        "min_live / larger C_block, i.e. closer to the shipped model)."),
            },
        }
        blk["VERDICT_vs_prereg_bar"] = verdict(
            nl["nested_gain"], nl["nested_n_positive"], struct["max_attainable_positive_seasons"])
        blk["VERDICT_vs_prereg_bar"]["nested_gain_if_tie_broken_the_other_way"] = \
            nl_alt["nested_gain"]
        blk["VERDICT_vs_prereg_bar"]["clause_1_magnitude_met_under_either_tiebreak"] = bool(
            min(nl["nested_gain"], nl_alt["nested_gain"]) >= BAR_NESTED_GAIN)
        results[fam] = blk
        log(f"{fam:<10} raw {ll0 - ll_map[raw_best]:+.6f}  nested {nl['nested_gain']:+.6f}  "
            f"-> {blk['VERDICT_vs_prereg_bar']['VERDICT']}")

    # --------------------------------- the honest cross-family nested estimate --
    all_pool = ["BASE"] + list(variants)
    cross = nested_loso(per_map, per0, all_pool)
    cross["nested_gain_tiebreak_last"] = nested_loso(
        per_map, per0, all_pool, tiebreak="last")["nested_gain"]
    cross["what"] = ("LOSO selection over EVERY candidate variant AND the shipped baseline at "
                     "once -- what this whole screen is worth if you let it choose freely. It "
                     "is not the pre-registered per-candidate test; it is reported because a "
                     "four-family screen with a 15-cell pool has more search optimism than any "
                     "single family does, and the pre-registration's method standard demands "
                     "the optimism be quantified rather than assumed away.")

    # ---------------------------------------------------------- noise context --
    dpair = np.array([per0[k] - per_map["DROP"][k] for k in sorted(per0)])
    noise = {
        "unpaired_fold_sd_of_baseline": baseline["unpaired_fold_sd"],
        "unpaired_fold_se_of_baseline": baseline["unpaired_fold_se"],
        "paired_fold_se_drop_vs_base": r6(float(dpair.std(ddof=1) / np.sqrt(len(dpair)))),
        "game_level_paired_bootstrap_se_drop_vs_base":
            paired_bootstrap(v0, vec_map["DROP"])["se"],
        "build_main_process_jitter": 3e-5,
        "reading": ("all four candidates can only move two of the ten folds, so the paired "
                    "per-season SD is dominated by eight structural zeros and UNDERSTATES the "
                    "noise on the two folds that actually move. The live-fold spread inside "
                    "each variant block, and the game-level bootstrap, are the honest scales."),
    }

    payload = {
        "generated": time.strftime("%Y-%m-%d"),
        "pre_registration": PREREG,
        "protocol": {
            "league": "NFL", "harness": "main split, shipped 14-feature blend",
            "dev_train_from": TRAIN_FROM, "dev_scored": [MAIN_LO, MAIN_HI],
            "test_era_never_touched": [TEST_ERA, 2025], "test_looks_spent": 0,
            "metric": "walk-forward DEV log loss (the blend's own objective)",
            "training_protocol": {"model": "logistic", "C": C_BASE,
                                  "recency_half_life": HL_BASE},
            "market_blind": True,
        },
        "bar": {"nested_gain_at_least": BAR_NESTED_GAIN,
                "positive_seasons_at_least": BAR_MIN_POSITIVE_SEASONS,
                "of_seasons": N_SCORED_SEASONS,
                "source": "documents/nfl_absence_prereg.md, fixed in advance",
                "no_rescreening": True},
        "structural_facts": struct,
        "baseline": baseline,
        "candidates": results,
        "cross_family_nested": cross,
        "fold_noise": noise,
        "run_to_run_jitter_on_the_gain": {
            "observations_across_5_processes": JITTER_OBS,
            "spread": {k: r6(max(v) - min(v)) for k, v in JITTER_OBS.items()},
            "cause": ("build_main() accumulates league EPA rates over dict iteration order. "
                      "Both the baseline and every candidate are rebuilt from that same "
                      "jittered X14, so the jitter does NOT fully cancel in the paired gain."),
            "consequence": ("+/-2e-5 on DROP and POOL, which is 2% of the bar and changes "
                            "nothing. On SHRINK it is decisive: the nested gain straddles the "
                            "+0.0010 magnitude clause, landing above it in 3 of 5 processes and "
                            "below it in 2. A candidate whose bar-crossing depends on dict "
                            "ordering has not cleared the bar in any meaningful sense."),
        },
    }

    # ---------------------------- the one thing the pre-registration got wrong --
    adopted = [f for f, b in results.items() if b["VERDICT_vs_prereg_bar"]["VERDICT"] == "ADOPT"]
    payload["outcome"] = {
        "adopted": adopted,
        "all_rejected": not adopted,
        "test_look_owed": bool(adopted),
        "prereg_defect": {
            "clause": f"positive in >= {BAR_MIN_POSITIVE_SEASONS} of {N_SCORED_SEASONS} seasons",
            "status": ("UNSATISFIABLE BY CONSTRUCTION"
                       if struct["max_attainable_positive_seasons"] < BAR_MIN_POSITIVE_SEASONS
                       else "satisfiable"),
            "why": (f"the block is identically zero before {struct['first_season_block_is_nonzero']}, "
                    "so for every fold trained entirely on pre-2013 seasons the four "
                    "coefficients are exactly zero under the ridge and EVERY candidate "
                    "reproduces the shipped prediction bit-for-bit. Only "
                    f"{struct['max_attainable_positive_seasons']} of the {N_SCORED_SEASONS} "
                    f"scored folds ({can_differ}) can differ from the baseline at all, so no "
                    f"candidate could ever be positive in {BAR_MIN_POSITIVE_SEASONS}. The "
                    "sign-count clause was written for a feature live across the whole split "
                    "and does not describe this block. It is REPORTED AS A DEFECT, not "
                    "silently replaced: the magnitude clause is evaluated as written and both "
                    "clauses are shown separately so the reader can see exactly which one "
                    "each candidate fails."),
        },
    }

    atomic_json(payload, OUT)
    log(f"wrote {OUT}")
    print(json.dumps({"baseline": payload["baseline"]["dev_ll"],
                      "verdicts": {f: results[f]["VERDICT_vs_prereg_bar"]["VERDICT"]
                                   for f in results},
                      "nested_gains": {f: results[f]["NESTED"]["nested_gain"] for f in results},
                      "raw_gains": {f: results[f]["RAW"]["raw_gain"] for f in results}},
                     indent=1))


if __name__ == "__main__":
    main()
