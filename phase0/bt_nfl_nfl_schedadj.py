"""Breakthrough program (NFL) - DEV screen nfl_nfl_schedadj. DEV ONLY (2006-2015).

Candidate: schedule-adjusted process block, solved exactly (features from
phase0/bt_nfl_nfl_schedadj_build.py):
  net'/pass'/run'  joint per-gameday opponent-adjusted team EPA channels
                   (REPLACE shipped cols 2 / 12 / 13, raw units),
  QJ               opponent-adjusted QB increment on the shipped QbElo state,
                   past opponents' pass defence taken from the as-of-D joint
                   solve (ADDED as a 15th column, z-scored on the train fold).

Harness: wf()/rep() of phase0/bt_nfl_ideas_quick3.py (exec of their source, file
untouched) = the shipped walk-forward (expanding refit from 1999, recency HL=3,
C=100, ties dropped from fits, every DEV row scored).  Paired per-game bootstrap
(10,000 reps, fixed seed) + season-block bootstrap.

ARMS  A0 shipped 14 | AR one-pass C1c+Q (reference, already viewed, NOT promotable)
      A1 PRIMARY joint block | A2 team channels only | A3 QJ only
BAR   A1 gain >= +0.00150 AND paired CI lower bound > 0, plus guards:
      QJ coef > 0 in >= 8/10 folds; 2002-2005 replication gain >= 0.

MARKET-BLIND: no odds column is read anywhere in this screen or its builder.
No season >= 2016 exists in the cache, is built, fitted or scored.
Output: data/bt_nfl_nfl_schedadj.json
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
NBOOT, SEED = 10000, 20260924
BAR = 0.00150

D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
N = len(G)
SEAS = np.array([g["season"] for g in G])
assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
DEVIDX = np.concatenate([np.where(SEAS == s)[0] for s in range(DEV_LO, DEV_HI + 1)])
assert len(DEVIDX) == 2670 and SEAS[DEVIDX].max() < TEST_ERA
SD_DEV = SEAS[DEVIDX]


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


FJ = np.load("data/bt_nfl_nfl_schedadj_feats.npy")
REF = np.load("data/bt_nfl_nfl_schedadj_ref.npz")
BUILD = json.load(open("data/bt_nfl_nfl_schedadj_build.json", encoding="utf-8"))
assert FJ.shape == (N, 4)
NETJ, PASSJ, RUNJ, QJ = (FJ[:, k].copy() for k in range(4))

# ------------------------------------------ quick3 harness (exec, untouched) ---
_src3 = open("phase0/bt_nfl_ideas_quick3.py", encoding="utf-8").read()
_wf_src = "def wf(" + _src3.split("def wf(", 1)[1].split("b_ll, b_per, b_vec, _ = wf()", 1)[0]
_rep_src = "def rep(" + _src3.split("def rep(", 1)[1].split('print("1. schedule-adjusted process block")', 1)[0]
exec(compile(_wf_src, "quick3<wf>", "exec"))  # noqa: S102
RNG3 = np.random.default_rng(20260926)
b_ll, b_per, b_vec, _ = wf()  # noqa: F821
assert len(b_vec) == 2670
OUT = {"baseline_dev_ll": b_ll, "results": {}}
exec(compile(_rep_src, "quick3<rep>", "exec"))  # noqa: S102


def boot(d, seed=SEED):
    """paired per-game bootstrap of mean(d); d > 0 = candidate better."""
    rng = np.random.default_rng(seed)
    n = len(d)
    ms = np.empty(NBOOT)
    for k in range(0, NBOOT, 1000):
        ms[k:k + 1000] = d[rng.integers(0, n, size=(1000, n))].mean(1)
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return float(lo), float(hi), float((ms <= 0).mean())


def boot_season(d, seas, seed=SEED + 1):
    """season-block bootstrap: resample whole DEV seasons with replacement."""
    rng = np.random.default_rng(seed)
    us = np.unique(seas)
    sums = np.array([d[seas == s].sum() for s in us]); cnts = np.array([(seas == s).sum() for s in us])
    pick = rng.integers(0, len(us), size=(NBOOT, len(us)))
    ms = sums[pick].sum(1) / cnts[pick].sum(1)
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return float(lo), float(hi)


def full_coefs(extra=None, replace=None):
    """same fits as wf(); returns every coefficient by fold (diagnostic only)."""
    X = X14.copy()
    for c_, v in (replace or {}).items():
        X[:, c_] = v
    out, vec = {}, []
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X
        if extra is not None:
            E = extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        vec.append(llv(Y[te], m.predict_proba(Xs[te])[:, 1]))
        out[str(s_)] = m.coef_[0].round(5).tolist()
    return out, float(np.concatenate(vec).mean())


RES = {}


def arm(nm, res, promotable=False):
    ll, per, vec, coefs = res
    assert len(vec) == len(b_vec) == 2670            # identical DEV mask in every arm
    rep(nm, res)  # noqa: F821  (quick3 reporter: 4000-rep CI, per-season, coefs)
    d = b_vec - vec
    lo, hi, p_le0 = boot(d)
    blo, bhi = boot_season(d, SD_DEV)
    sub = SD_DEV >= 2013
    slo, shi, _ = boot(d[sub], SEED + 2)
    RES[nm] = {"ll": ll, "gain": float(d.mean()), "ci95_paired_10k": [lo, hi], "p_gain_le_0": p_le0,
               "ci95_season_block_10k": [blo, bhi],
               "seasons_pos": int(sum(1 for s in per if b_per[s] - per[s] > 0)),
               "per_season_gain": {str(s): round(b_per[s] - per[s], 6) for s in per},
               "added_coefs_by_fold": {str(k): v for k, v in coefs.items()},
               "sub_2013_2015": {"gain": float(d[sub].mean()), "ci95": [slo, shi], "n": int(sub.sum())},
               "promotable": promotable}
    print(f"     10k paired CI [{lo:+.5f},{hi:+.5f}]  season-block CI [{blo:+.5f},{bhi:+.5f}]  "
          f"2013-15 {d[sub].mean():+.5f} [{slo:+.5f},{shi:+.5f}]", flush=True)
    return vec


print(f"A0 baseline DEV LL {b_ll:.6f} n={len(b_vec)} (cache {D['dev_ll']:.6f})", flush=True)

# =========================================================== HARNESS SANITY ===
SAN = {}
SAN["a_baseline"] = {"ll": b_ll, "target": 0.62292, "cache": D["dev_ll"],
                     "pass": bool(abs(b_ll - 0.62292) <= 5e-4 and abs(b_ll - D["dev_ll"]) < 1e-6)}
assert SAN["a_baseline"]["pass"]

print("(b) CONTROL: drop col 1 (qb_delta) - zero column == dropped column under the L2 fit")
cvec = arm("CONTROL drop qb_delta (col 1)", wf(replace={1: np.zeros(N)}))  # noqa: F821
cost = -RES["CONTROL drop qb_delta (col 1)"]["gain"]
cci = RES["CONTROL drop qb_delta (col 1)"]["ci95_paired_10k"]
SAN["b_drop_col1"] = {"cost": cost, "ci95_of_gain": cci, "accept": [0.0045, 0.0075],
                      "refs": {"bt_nfl_anatomy_harness": 0.00603, "quick2_control": 0.006149, "ledger_row3": 0.00689},
                      "pass": bool(0.0045 <= cost <= 0.0075 and cci[1] < 0)}
assert SAN["b_drop_col1"]["pass"], SAN["b_drop_col1"]

FZ, FEQ = REF["FZ"], REF["FEQ"]
idc = {"epa_net_col2": float(np.abs(FZ[:, 0] - X14[:, 2]).max()),
       "pass_col12": float(np.abs(FZ[:, 1] - X14[:, 12]).max()),
       "run_col13": float(np.abs(FZ[:, 2] - X14[:, 13]).max()),
       "QJ_stale_vs_quick2_FE_Q": float(np.abs(FZ[:, 3] - FEQ).max())}
SAN["c_solver_identity"] = {"max_abs_diff_all_1999_2015_rows": idc, "tol": 1e-9,
                            "pass": bool(max(idc.values()) < 1e-9)}
print("(c) solver identity:", idc, flush=True)
assert SAN["c_solver_identity"]["pass"]

print("(d) reference reproduction: one-pass C1c + Q")
AR_REPL = {2: REF["ADJ_ALL"], 12: REF["ADJ_PASS"], 13: REF["ADJ_RUN"]}
arm("AR one-pass C1c+Q (reference, viewed)", wf(FEQ, replace=AR_REPL))  # noqa: F821
g_ar = RES["AR one-pass C1c+Q (reference, viewed)"]["gain"]
SAN["d_reference_repro"] = {"gain": g_ar, "target": 0.00098, "recorded_quick3": 0.000981,
                            "pass": bool(abs(g_ar - 0.00098) <= 5e-5)}
assert SAN["d_reference_repro"]["pass"], g_ar
print(f"harness sanity: all pass ({time.time()-T0:.0f}s)", flush=True)

# ===================================================================== ARMS ===
REPL = {2: NETJ, 12: PASSJ, 13: RUNJ}
print("ARMS")
arm("A1 PRIMARY joint block (replace 2/12/13 + QJ)", wf(QJ, replace=REPL), promotable=True)  # noqa: F821
arm("A2 joint team channels only (replace 2/12/13)", wf(replace=REPL))  # noqa: F821
arm("A3 QJ only (add)", wf(QJ))  # noqa: F821

A1 = RES["A1 PRIMARY joint block (replace 2/12/13 + QJ)"]
qj_coefs = {k: v[0] for k, v in A1["added_coefs_by_fold"].items()}
n_pos_qj = sum(1 for v in qj_coefs.values() if v > 0)

# full coefficient table for diagnostics (same fits as wf)
COEF = {}
for nm, kw in (("A0", {}), ("A1", {"extra": QJ, "replace": REPL}), ("A2", {"replace": REPL})):
    cf, ll_chk = full_coefs(**kw)
    ref_ll = {"A0": b_ll, "A1": A1["ll"], "A2": RES["A2 joint team channels only (replace 2/12/13)"]["ll"]}[nm]
    assert abs(ll_chk - ref_ll) < 1e-9
    COEF[nm] = {k: {"col2_net": v[2], "col12_pass": v[12], "col13_run": v[13], "col1_qb": v[1],
                    **({"QJ": v[14]} if len(v) > 14 else {})} for k, v in cf.items()}

# ===================================================== 2002-2005 replication ===
DEV_LO, DEV_HI = 2002, 2005
r0 = wf()  # noqa: F821
r1 = wf(QJ, replace=REPL)  # noqa: F821
rR = wf(FEQ, replace=AR_REPL)  # noqa: F821
r2 = wf(replace=REPL)  # noqa: F821   (diagnostic arm)
r3 = wf(QJ)  # noqa: F821             (diagnostic arm)
DEV_LO, DEV_HI = 2006, 2015
assert abs(r0[0] - 0.6223914489104433) < 1e-9, r0[0]
SEASO = np.concatenate([SEAS[SEAS == s] for s in range(2002, 2006)])
REPLIC = {}
for nm, r in (("A1 joint block", r1), ("AR one-pass (recorded -0.000718)", rR),
              ("A2 team channels only (diagnostic)", r2), ("A3 QJ only (diagnostic)", r3)):
    d = r0[2] - r[2]
    lo, hi, _ = boot(d, SEED + 3)
    REPLIC[nm] = {"gain": float(d.mean()), "ci95": [lo, hi], "n": int(len(d)),
                  "per_season": {str(s): round(r0[1][s] - r[1][s], 6) for s in r[1]},
                  "added_coef_by_fold": {str(k): (v[0] if v else None) for k, v in r[3].items()}}
    print(f"  2002-05 {nm:<34} gain {d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]", flush=True)
assert abs(REPLIC["AR one-pass (recorded -0.000718)"]["gain"] - (-0.000718)) < 5e-6

# ============================================================ correlations ===
dv = SEAS >= DEV_LO
CORR = {}
for nm, v, col in (("net' vs col2", NETJ, 2), ("pass' vs col12", PASSJ, 12), ("run' vs col13", RUNJ, 13),
                   ("QJ vs col1 qb_delta", QJ, 1)):
    CORR[nm] = {"dev": float(np.corrcoef(v[dv], X14[dv, col])[0, 1]),
                "all_1999_2015": float(np.corrcoef(v, X14[:, col])[0, 1])}
for nm, v, w_ in (("net' vs one-pass ADJ_ALL", NETJ, REF["ADJ_ALL"]), ("pass' vs one-pass ADJ_PASS", PASSJ, REF["ADJ_PASS"]),
                  ("run' vs one-pass ADJ_RUN", RUNJ, REF["ADJ_RUN"]), ("QJ vs one-pass FE_Q", QJ, FEQ)):
    CORR[nm] = {"dev": float(np.corrcoef(v[dv], w_[dv])[0, 1])}
CORR["sd_ratio_new_over_shipped_dev"] = {
    "net": float(NETJ[dv].std() / X14[dv, 2].std()), "pass": float(PASSJ[dv].std() / X14[dv, 12].std()),
    "run": float(RUNJ[dv].std() / X14[dv, 13].std()), "QJ_over_FE_Q": float(QJ[dv].std() / FEQ[dv].std())}
print("correlations:", json.dumps(CORR, indent=0), flush=True)

# ================================================================ grep audit ===
ODDS = re.compile("money" + "line|spread" + "_line|total" + "_line|over" + "_odds|under" + "_odds|spread" + "_odds", re.I)
audit = {}
for p in ("phase0/bt_nfl_nfl_schedadj_build.py", "phase0/bt_nfl_nfl_schedadj.py"):
    audit[p] = [ln.strip() for ln in open(p, encoding="utf-8") if ODDS.search(ln)]
q2pre = open("phase0/bt_nfl_ideas_quick2.py", encoding="utf-8").read().split(
    "# ------------------------------------------------------------ harness")[0]
q1pre = open("phase0/bt_nfl_ideas_quick.py", encoding="utf-8").read().split(
    "# ============================================= C2 playoff leverage")[0]
audit["quick2 builder prefix (exec'd)"] = [ln.strip() for ln in q2pre.splitlines() if ODDS.search(ln)]
audit["quick1 C1 builder prefix (exec'd)"] = [ln.strip() for ln in q1pre.splitlines() if ODDS.search(ln)]
audit["quick3 wf/rep (exec'd)"] = [ln.strip() for ln in (_wf_src + _rep_src).splitlines() if ODDS.search(ln)]
audit_clean = all(len(v) == 0 for v in audit.values())
print("grep audit (odds tokens):", {k: len(v) for k, v in audit.items()}, flush=True)
assert audit_clean

# ================================================================== verdict ===
gain, (lo, hi) = A1["gain"], A1["ci95_paired_10k"]
guards = {"qj_coef_pos_folds": n_pos_qj, "qj_coef_guard_pass": n_pos_qj >= 8,
          "replication_2002_2005_gain": REPLIC["A1 joint block"]["gain"],
          "replication_guard_pass": REPLIC["A1 joint block"]["gain"] >= 0,
          "replication_window_note": "2002-2005 was partly viewed for this channel (one-pass block re-scored "
                                     "there at -0.000718); flagged, not a clean holdout"}
clears = bool(gain >= BAR and lo > 0)
verdict = {"bar": BAR, "A1_gain": gain, "A1_ci95": [lo, hi], "clears_bar": clears,
           "guards": guards, "clears_bar_and_guards": bool(clears and guards["qj_coef_guard_pass"]
                                                           and guards["replication_guard_pass"])}
print("VERDICT:", json.dumps(verdict, indent=1), flush=True)

json.dump({"id": "nfl_nfl_schedadj", "dev_seasons": "2006-2015", "n_dev_games": 2670,
           "baseline_dev_ll": b_ll, "harness_sanity": SAN, "arms": RES, "quick3_rep_4000": OUT["results"],
           "coefs_full_by_fold": COEF, "qj_coef_by_fold_A1": qj_coefs, "replication_2002_2005": REPLIC,
           "correlations": CORR, "grep_audit": audit, "build": {k: BUILD[k] for k in
           ("anchors_shipped_full_1999_2015", "max_season_read", "identity_max_abs_diff", "perturbation")},
           "verdict": verdict, "secs": round(time.time() - T0, 1)},
          open("data/bt_nfl_nfl_schedadj.json", "w"), indent=1)
print(f"wrote data/bt_nfl_nfl_schedadj.json ({time.time()-T0:.0f}s)")
