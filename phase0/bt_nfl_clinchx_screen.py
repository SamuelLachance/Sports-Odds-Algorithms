"""Breakthrough program (NFL) - candidate nfl_clinchx: DEV SCREEN. DEV ONLY (2006-2015 scored).

Feature LKX = locked_away - locked_home from data/bt_nfl_clinchx_flags.npy (built by
phase0/bt_nfl_clinchx_build.py: exact NFL tiebreakers, date-strict standings), added as a
15th column to the shipped 14-feature walk-forward (train seasons < s from 1999, ties
dropped, weight 0.5^((s-1-season)/3), LogisticRegression(C=100, max_iter=5000), new
column z-scored on train rows), scored on all 2670 DEV rows 2006-2015 (ties included).

The walk-forward wf() is bt_nfl_ideas_quick3.py's, loaded by exec of its source slice
(never edited); the proxy flag (AR) is rebuilt by exec of bt_nfl_ideas_quick2.py's
prologue + stakes (S) section with its own RNG 20260925.

Harness sanity (must pass first): (a) shipped 14-col DEV LL 0.62292 +- 5e-4, n=2670;
(b) dropping col 1 (qb_delta) costs 0.0045-0.0075 with CI excluding 0; (c) proxy S3
re-score +0.00206 +- 5e-5.

Arms: A0 baseline; AR proxy (reproduction only); A1 PRIMARY LKX; A2 QB-weighted
(pre-declared secondary). No other arms, no threshold search.
MARKET-BLIND: no odds are used anywhere in this file (the market readout lives in
phase0/bt_nfl_clinchx_eval_market.py). No season >= 2016 exists in any input.
Output: data/bt_nfl_clinchx.json (+ identical copy data/bt_nfl_nfl_clinchx.json)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

T0 = time.time()
sys.path.insert(0, "phase0")
TEST_ERA = 2016
BOOT = 10000
BOOT_SEED = 20261002

# ---------------------------------------------------------------- audits ----
_bsrc = open("phase0/bt_nfl_clinchx_build.py", encoding="utf-8").read()
_bad = ["money" + "line", "spread" + "_line", "total" + "_line", "_" + "odds"]
GREP = {t: (t in _bsrc) for t in _bad}
assert not any(GREP.values()), f"builder source references a market column: {GREP}"
UT = json.load(open("data/bt_nfl_clinchx_unittest.json"))
assert UT["pass"], "tiebreak unit test failed - no scoring allowed"
PT = json.load(open("data/bt_nfl_clinchx_perturb.json"))
assert PT["pass"] and len(PT["groups"]) == 30, "future-perturbation test failed"

# ------------------------------------------- quick2 prologue + S section ----
_q2 = open("phase0/bt_nfl_ideas_quick2.py", encoding="utf-8").read()
_pro = _q2.split("# ------------------------------------------------ per-game EPA aggregates ----")[0]
_sS = _q2.split("# ---- S late-season stakes")[1].split("# ---- P returning production")[0]
NS2: dict = {}
exec(compile(_pro, "quick2<prologue>", "exec"), NS2)  # noqa: S102
NS2["FE"] = {}
exec(compile("# ---- S late-season stakes" + _sS, "quick2<S builder>", "exec"), NS2)  # noqa: S102
for k in ("RAW",):                       # quick2's raw-row dict is not needed here; drop it
    NS2.pop(k, None)
G = NS2["G"]; X14 = NS2["X14"]; SEAS = NS2["SEAS"]; Y = NS2["Y"]; WK = NS2["WK"]; TYP = NS2["TYP"]
N = NS2["N"]; llv = NS2["llv"]; LOCK = NS2["LOCK"]
assert SEAS.max() < TEST_ERA and N == 4515 and X14.shape == (N, 14)
LK = LOCK[:, 1] - LOCK[:, 0]

# --------------------------------------------------------- quick3 wf() ----
_q3 = open("phase0/bt_nfl_ideas_quick3.py", encoding="utf-8").read()
_wf3 = "def wf(" + _q3.split("\ndef wf(")[1].split("\nb_ll, b_per, b_vec, _ = wf()")[0]
_wf2 = "def wf(" + _q2.split("\ndef wf(")[1].split("\nbase_ll, base_per, base_vec = wf()")[0]


def make_wf(src, lo, hi):
    ns = dict(NS2); ns["DEV_LO"], ns["DEV_HI"] = lo, hi
    exec(compile(src, "wf<slice>", "exec"), ns)  # noqa: S102
    return ns["wf"]


assert 2015 < TEST_ERA
wf = make_wf(_wf3, 2006, 2015)            # quick3 wf: returns (ll, per, vec, coefs of cols 14+)
wf_drop = make_wf(_wf2, 2006, 2015)       # quick2 wf: supports drop=
wf_early = make_wf(_wf3, 2002, 2005)      # 2002-2005 replication (window already viewed by the proxy)
DEVIDX = np.concatenate([np.where(SEAS == s)[0] for s in range(2006, 2016)])
EIDX = np.concatenate([np.where(SEAS == s)[0] for s in range(2002, 2006)])
assert len(DEVIDX) == 2670 and SEAS[DEVIDX].max() < TEST_ERA

# ------------------------------------------------------------- features ----
FL = np.load("data/bt_nfl_clinchx_flags.npy")
assert FL.shape == (N, 2)
assert FL[SEAS < 2002].sum() == 0 and FL[TYP != "REG"].sum() == 0 and FL[(WK < 15)].sum() == 0
LKX = FL[:, 1] - FL[:, 0]
# A2: side weight w = max(0, q_starter - q_rep); q_rep = shipped QbElo qb_adj for an id with
# no history and no draft record (beta 1: rep_delta = pedigree 'delta', clipped at +-CLIP)
import nfl_qb_elo as QE  # noqa: E402

_ped = json.load(open("data/nfl_qb_pedigree.json"))["params"]
_m = QE.QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=0.0, rep_delta=_ped["delta"], rep_map={},
              decay=0.8, prior_db=650.0, season_decay=0.4)
Q_REP = float(_m.qb_adj("__no_history_no_draft__"))
_D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
assert [g["gid"] for g in _D["games"]] == [g["gid"] for g in G]
QH = np.array(_D["qh"]); QA = np.array(_D["qa"])
_vals, _cnt = np.unique(np.round(np.concatenate([QH, QA]), 9), return_counts=True)
assert abs(_vals[_cnt.argmax()] - Q_REP) < 1e-8, "q_rep is not the modal (no-history) rating"
WH = np.maximum(0.0, QH - Q_REP); WA = np.maximum(0.0, QA - Q_REP)
A2F = FL[:, 1] * WA - FL[:, 0] * WH


# ------------------------------------------------------------ reporting ----
def boot_ci(d, seed=BOOT_SEED, reps=BOOT):
    rng = np.random.default_rng(seed); n = len(d); out = np.empty(reps)
    for a in range(0, reps, 500):
        k = min(500, reps - a)
        out[a:a + k] = d[rng.integers(0, n, size=(k, n))].mean(1)
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def block_ci(d, seas_vec, seed=BOOT_SEED + 1, reps=BOOT):
    rng = np.random.default_rng(seed); ss = np.unique(seas_vec)
    sums = np.array([d[seas_vec == s].sum() for s in ss]); cnts = np.array([(seas_vec == s).sum() for s in ss])
    pick = rng.integers(0, len(ss), size=(reps, len(ss)))
    b = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def oof_pred(extra, lo, hi):
    """identical protocol to wf(); returns OOF home-win probabilities for seasons lo..hi."""
    from sklearn.linear_model import LogisticRegression
    P = np.full(N, np.nan)
    for s_ in range(lo, hi + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X14
        if extra is not None:
            E = extra.reshape(-1, 1); mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X14, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / 3.0)
        P[te] = LogisticRegression(C=100.0, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w).predict_proba(Xs[te])[:, 1]
    return P


OUT = {"id": "nfl_clinchx", "dev_seasons": "2006-2015", "n_dev": 2670, "audits": {
    "builder_grep_market_tokens": GREP, "unit_test_pass": UT["pass"],
    "unit_test_coin_reached": {s: v["coin_reached"] for s, v in UT["seasons"].items()},
    "unit_test_steps": {s: v["steps"] for s, v in UT["seasons"].items()},
    "unit_test_synthetic": UT["synthetic"],
    "perturbation_pass": PT["pass"], "perturbation_groups": len(PT["groups"]),
    "perturbation_positive_control_moved": PT.get("positive_control_groups_moved"),
    "max_season_in_inputs": int(SEAS.max())}, "q_rep": Q_REP}

print(f"[{time.time()-T0:.0f}s] harness", flush=True)
b_ll, b_per, b_vec, _ = wf()
assert len(b_vec) == 2670
OUT["baseline_dev_ll"] = b_ll
san = {"a_baseline_ll": b_ll, "a_expected": 0.62292, "a_pass": bool(abs(b_ll - 0.62292) < 5e-4)}
assert san["a_pass"], san
d_ll, d_per, d_vec = wf_drop(drop=[1])
cost = d_ll - b_ll; ci_c = boot_ci(d_vec - b_vec)
san.update({"b_drop_qb_cost": cost, "b_ci": ci_c, "b_pass": bool(0.0045 <= cost <= 0.0075 and ci_c[0] > 0)})
print(f"  (a) baseline {b_ll:.6f}   (b) drop qb_delta cost {cost:+.5f} CI[{ci_c[0]:+.5f},{ci_c[1]:+.5f}]", flush=True)
assert san["b_pass"], san
r_ll, r_per, r_vec, r_coef = wf(LK)
gR = b_ll - r_ll
san.update({"c_proxy_gain": gR, "c_expected": 0.00206, "c_pass": bool(abs(gR - 0.00206) < 5e-5)})
print(f"  (c) proxy S3 re-score gain {gR:+.6f}", flush=True)
assert san["c_pass"], san
OUT["harness_sanity"] = san


def arm(nm, feat):
    ll, per, vec, coefs = wf(feat)
    assert len(vec) == len(b_vec) == 2670
    d = b_vec - vec
    g = float(d.mean()); ci = boot_ci(d); bci = block_ci(d, SEAS[DEVIDX])
    sub = np.isin(SEAS[DEVIDX], [2013, 2014, 2015])
    per_g = {str(s): round(b_per[s] - per[s], 6) for s in per}
    cf = {str(k): (v[0] if v else None) for k, v in coefs.items()}
    r = {"ll": ll, "gain": g, "ci": ci, "ci_season_block": bci,
         "seasons_pos": int(sum(1 for s in per if b_per[s] - per[s] > 0)), "per_season_gain": per_g,
         "coef_by_fold": cf, "coef_pos_folds": int(sum(1 for v in cf.values() if v is not None and v > 0)),
         "gain_2013_2015": float(d[sub].mean()), "ci_2013_2015": boot_ci(d[sub]), "n_2013_2015": int(sub.sum())}
    print(f"  {nm:<34} LL {ll:.6f} gain {g:+.6f} CI[{ci[0]:+.6f},{ci[1]:+.6f}] block[{bci[0]:+.6f},{bci[1]:+.6f}] "
          f"seasons+ {r['seasons_pos']}/10 coef+ {r['coef_pos_folds']}/10 13-15 {r['gain_2013_2015']:+.5f}", flush=True)
    return r, vec


print(f"[{time.time()-T0:.0f}s] arms", flush=True)
RES = {}
RES["AR_proxy_repro"], vR = arm("AR proxy LK (repro only)", LK)
RES["A1_LKX_exact"], v1 = arm("A1 LKX exact (PRIMARY)", LKX)
RES["A2_LKX_qbweighted"], v2 = arm("A2 LKX x QB weight (secondary)", A2F)
dd = v1 - v2                              # positive = A2 better than A1
RES["A2_minus_A1"] = {"gain": float(dd.mean()), "ci": boot_ci(dd)}
OUT["arms"] = RES

a1 = RES["A1_LKX_exact"]
bar = bool(a1["gain"] >= 0.0015 and a1["ci"][0] > 0)
guards = bool(a1["coef_pos_folds"] >= 8 and a1["gain_2013_2015"] >= 0)
a2 = RES["A2_LKX_qbweighted"]
a2_prom = bool(RES["A2_minus_A1"]["gain"] > 0.0002 and a2["ci"][0] > 0 and a2["gain"] >= 0.0015
               and a2["coef_pos_folds"] >= 8 and a2["gain_2013_2015"] >= 0)
OUT["verdict"] = {"A1_clears_prereg_bar": bar, "A1_passes_stricter_guards": guards,
                  "A1_promotable": bool(bar and guards), "A2_promotable": a2_prom}

# ------------------------------------------------ 2002-2005 replication ----
print(f"[{time.time()-T0:.0f}s] 2002-2005 replication (window already viewed by the proxy)", flush=True)
e_ll, e_per, e_vec, _ = wf_early()
rep_early = {}
for nm, f in (("AR_proxy", LK), ("A1_LKX", LKX), ("A2_qbw", A2F)):
    ll, per, vec, coefs = wf_early(f)
    d = e_vec - vec
    rep_early[nm] = {"gain": float(d.mean()), "ci": boot_ci(d),
                     "per_season_gain": {str(s): round(e_per[s] - per[s], 6) for s in per},
                     "coef_by_fold": {str(k): (v[0] if v else None) for k, v in coefs.items()}}
    print(f"  {nm:<10} gain {d.mean():+.6f}  per-season {rep_early[nm]['per_season_gain']}", flush=True)
OUT["replication_2002_2005"] = {"note": "ALREADY VIEWED by the proxy (quick2/3); not independent",
                                "baseline_ll": e_ll, "n": int(len(e_vec)), "arms": rep_early}

# ------------------------------------------------ counts, overlap, diffs ----
cnt = {}
for s in range(2002, 2016):
    m = SEAS == s
    cnt[str(s)] = {"ours_home": int(FL[m, 0].sum()), "ours_away": int(FL[m, 1].sum()),
                   "proxy_home": int(LOCK[m, 0].sum()), "proxy_away": int(LOCK[m, 1].sum())}
OUT["locked_side_counts"] = cnt
P0 = oof_pred(None, 2006, 2015); P1 = oof_pred(LKX, 2006, 2015)
P0e = oof_pred(None, 2002, 2005); P1e = oof_pred(LKX, 2002, 2005)
assert np.allclose(llv(Y[DEVIDX], P0[DEVIDX]), b_vec, atol=1e-10) and np.allclose(llv(Y[DEVIDX], P1[DEVIDX]), v1, atol=1e-10)
P0 = np.where(np.isnan(P0), P0e, P0); P1 = np.where(np.isnan(P1), P1e, P1)
GJ = json.load(open("data/bt_nfl_clinchx_groups.json"))["groups"]
SIDE = {(x["i"], x["side"]): x for g in GJ for x in g["sides"]}
both = (FL == 1) & (LOCK == 1); ours_only = (FL == 1) & (LOCK == 0); prox_only = (FL == 0) & (LOCK == 1)
ov = {"both": int(both.sum()), "ours_only": int(ours_only.sum()), "proxy_only": int(prox_only.sum()),
      "dev_2006_2015": {"both": int(both[SEAS >= 2006].sum()), "ours_only": int(ours_only[SEAS >= 2006].sum()),
                        "proxy_only": int(prox_only[SEAS >= 2006].sum())}}
lst = []
for tag, M in (("added", ours_only), ("removed", prox_only)):
    for i, sd in zip(*np.where(M)):
        x = SIDE.get((int(i), int(sd)), {})
        lst.append({"change": tag, "gid": G[i]["gid"], "season": int(SEAS[i]), "week": int(WK[i]),
                    "locked_side": "home" if sd == 0 else "away", "team": G[i]["home" if sd == 0 else "away"],
                    "ppo": x.get("ppo"), "lev": x.get("lev"), "pbye": x.get("pbye"),
                    "p_home_baseline": round(float(P0[i]), 4), "p_home_A1": round(float(P1[i]), 4),
                    "y": float(Y[i])})
ov["changes"] = sorted(lst, key=lambda r: (r["season"], r["week"], r["gid"]))
OUT["overlap_with_proxy"] = ov
lk = (FL.sum(1) > 0) & (SEAS >= 2006)
OUT["locked_games_dev"] = {"n_games": int(lk.sum()),
                           "home_locked": {"n": int(((FL[:, 0] == 1) & (SEAS >= 2006)).sum()),
                                           "p_base": float(P0[(FL[:, 0] == 1) & (SEAS >= 2006)].mean()),
                                           "p_A1": float(P1[(FL[:, 0] == 1) & (SEAS >= 2006)].mean()),
                                           "home_win": float(Y[(FL[:, 0] == 1) & (SEAS >= 2006)].mean())},
                           "away_locked": {"n": int(((FL[:, 1] == 1) & (SEAS >= 2006)).sum()),
                                           "p_base": float(P0[(FL[:, 1] == 1) & (SEAS >= 2006)].mean()),
                                           "p_A1": float(P1[(FL[:, 1] == 1) & (SEAS >= 2006)].mean()),
                                           "home_win": float(Y[(FL[:, 1] == 1) & (SEAS >= 2006)].mean())}}
print("  locked counts:", {s: (v["ours_home"] + v["ours_away"], v["proxy_home"] + v["proxy_away"]) for s, v in cnt.items()})
print("  overlap:", {k: v for k, v in ov.items() if k != "changes"})
print("  locked DEV games:", OUT["locked_games_dev"])
print("  verdict:", OUT["verdict"])
for fn in ("data/bt_nfl_clinchx.json", "data/bt_nfl_nfl_clinchx.json"):
    json.dump(OUT, open(fn, "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] wrote data/bt_nfl_clinchx.json (+ data/bt_nfl_nfl_clinchx.json)", flush=True)
