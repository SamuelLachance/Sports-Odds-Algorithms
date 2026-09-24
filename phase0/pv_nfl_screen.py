"""Player-value program (pv), NFL: the pre-registered GAME-LEVEL DEV SCREEN of the composed player value.

DEV ONLY (seasons 2006-2015 scored; 1999-2005 are training rows only). No odds read. No season >= 2016
row exists in any input; every input is asserted.

Candidate (built by pv_nfl_compose_*; not modified here): data/pv_nfl_compose_game_features.csv
  LV  = lv_home - lv_away   composed lineup value of the players who played, points/game, QB included
                            (snap era 2013-15: snap-table presence; 1999-2012: expected-roster proxy)
  QV  = qv_home - qv_away   the starting QB's composed passing value, points/game
        (passing composite x as-of league dropbacks/game x as-of unit weight w_QB, fit on seasons < s)
The 3 spine games without a pbp feature row (1999-2000, training rows only, never scored) get 0.

HARNESS. bt_nfl_ideas_quick3.wf, extracted VERBATIM from its source with `ast` (its llv from
bt_nfl_ideas_quick2.py the same way) and exec'd over the cached DEV matrix data/bt_nfl_ideas_dev_X.npy.
The scripts themselves are NOT run: quick3's module body reads market odds (its eval-only section 3) and
rewrites data/bt_nfl_ideas_quick3.json. Protocol, unchanged: expanding refit per DEV season, recency
half-life 3 seasons, C=100, ties dropped from fits (scored), `replace` columns entered raw, `extra`
columns z-scored on the train fold.

BASELINES, declared before any arm was scored.
  R0  PRIMARY reference = the SERVED blend: cached X14 with col 1 = the shipped QbElo keyed on the
      STARTER (nfl_qb_elo.apply_starters; served since ade264f2 / ledger row 77; sides from
      data/pv_nfl_passing_qbelo_starter_sides.json, built by the shipped walk truncated at 2016).
      Why not the cached X14: its col 1 is keyed on nflverse's POST-GAME primary passer (the leak the
      clinchx audit flagged, worth ~+0.001 on DEV); a comparison against it would hand the baseline
      post-game information and would bias A3 (which replaces col 1) by that amount.
  C0  the cached X14 as-is (listed QB), used for harness sanity (must reproduce 0.622937) and
      reported as a SECONDARY reference for every arm (sensitivity only; not used for the bar).

ARMS (exactly three; PRIMARY = A1; no arm, scaling or knob was added or changed after scoring):
  A1  PRIMARY  R0 with col 7 (the v7 11v11 participation TrueSkill) REPLACED by LV (raw points)
  A2           R0 + LV as a 15th column (z-scored on the train fold by the harness)
  A3           A1 and col 1 (shipped QbElo) REPLACED by QV (raw points): all player information in
               the model comes from the composed values
  Secondary constructions on C0 (sensitivity): A1c = C0 col 7 <- LV, A2c = C0 + LV;
  A3 replaces col 1, so it is the same model on either reference.
  Known limitation, stated before scoring: col 7 is identically 0 on every 1999-2015 row (v7's
  participation data start in 2016), so on DEV A1 and A2 are the same model up to the L2 penalty's
  scaling. The DEV screen therefore CANNOT price what A1 would lose on TEST by removing v7.

SANITY (same script): H1 C0 reproduces 0.6229368800; H2 R0 reproduces the passing leg's served-blend
reference 0.623924; H3 the known effect, dropping col 1 (QB; done by zeroing it, which H4 checks equals
deletion, 0.629086), must cost clearly (CI < 0) about the recorded -0.005/-0.006; H5 identical game
masks: every arm scores the same 2,670 DEV games in the same order.

STATISTICS. Per-game log-loss differences vs R0 (> 0: arm better). Paired game bootstrap, 10,000
resamples, fixed seed, the SAME resample indices for every arm; season-block bootstrap (resample the 10
DEV seasons with replacement, 10,000, fixed seed) reported beside it.
BAR (pre-registered, NFL; documents/breakthrough_program_prereg_2026_09_24.md): gain >= +0.00150 AND
per-game bootstrap 95% CI lower bound > 0 on the PRIMARY arm vs the primary reference. The season-block
CI is reported and flagged, not used to clear.

LEAK AUDIT (DEV): feature file carries no outcome; as-of unit-weight folds; team-level timing probe
(own LV/QV vs own margin in the previous / this / next game: this >> next would mean same-game
information); QV's starter vs the shipped starter-keyed QB.

POST-HOC AUDIT (added after the first run, disclosed; NOT an arm and NOT used for the bar): the first
run showed A1 and A2 - the same model on DEV - differing by 0.00019 nats. The harness's lbfgs stops at
sklearn's default tol=1e-4 after 30-75 iterations on this badly scaled design (column SDs 0.01-2.5), so
fold predictions carry optimizer noise. The same references and arms are refit with tol=1e-10 and
reported under "converged_sensitivity" to show whether any gain is optimizer noise.
Output: data/pv_nfl_screen.json (+ console log data/pv_nfl_screen.log when run with tee).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
P0 = os.path.join(ROOT, "phase0")
TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
NBOOT = 10_000
SEED = 20260930
BAR = 0.00150


# ------------------------------------------------------------------ the shipped harness, verbatim
def _extract(path, names):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.Assign) and "consts" in names:
            seg = ast.get_source_segment(src, node)
            if seg.startswith("HL, C ="):
                out["consts"] = seg
    return out


q2 = _extract(os.path.join(P0, "bt_nfl_ideas_quick2.py"), {"llv", "consts"})
q3 = _extract(os.path.join(P0, "bt_nfl_ideas_quick3.py"), {"wf"})
assert set(q2) == {"llv", "consts"} and set(q3) == {"wf"}
assert q2["consts"].replace(" ", "") == "HL,C=3.0,100.0", q2["consts"]
HL, C = 3.0, 100.0
HARNESS_SHA = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in (("quick3.wf", q3["wf"]),
                                                                           ("quick2.llv", q2["llv"]))}

D = json.load(open(os.path.join(DATA, "bt_nfl_ideas_dev.json"), encoding="utf-8"))
G = D["games"]
X14 = np.load(os.path.join(DATA, "bt_nfl_ideas_dev_X.npy"))
N = len(G)
assert X14.shape == (N, 14)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
GID = [g["gid"] for g in G]
assert np.all(X14[:, 7] == 0.0), "col 7 (v7) is expected to be identically 0 before 2016"

NS = {"np": np, "LogisticRegression": LogisticRegression, "X14": X14, "SEAS": SEAS, "Y": Y,
      "DEV_LO": DEV_LO, "DEV_HI": DEV_HI, "TEST_ERA": TEST_ERA, "HL": HL, "C": C}
exec(compile(q2["llv"], "bt_nfl_ideas_quick2.py<llv>", "exec"), NS)  # noqa: S102
exec(compile(q3["wf"], "bt_nfl_ideas_quick3.py<wf>", "exec"), NS)  # noqa: S102
wf, llv = NS["wf"], NS["llv"]

# ------------------------------------------------------------------ inputs
SS = json.load(open(os.path.join(DATA, "pv_nfl_passing_qbelo_starter_sides.json"), encoding="utf-8"))
assert all(g in SS for g in GID)
QBS = np.array([SS[g][0] - SS[g][1] for g in GID])

F = pd.read_csv(os.path.join(DATA, "pv_nfl_compose_game_features.csv"))
assert F.season.max() < TEST_ERA
OUTCOME_LIKE = {"margin", "epa_margin", "home_score", "away_score", "result", "y", "win", "total"}
assert not (OUTCOME_LIKE & set(F.columns)), "feature file carries an outcome column"
F = F.set_index("game_id")
rowsF = [F.loc[g] if g in F.index else None for g in GID]


def diff(a, b):
    return np.array([0.0 if r is None else float(r[a] - r[b]) for r in rowsF])


LV = diff("lv_home", "lv_away")
QV = diff("qv_home", "qv_away")
dv = (SEAS >= DEV_LO) & (SEAS <= DEV_HI)
nomiss = [GID[i] for i, r in enumerate(rowsF) if r is None]
assert all(not dv[GID.index(g)] for g in nomiss), "a DEV game has no feature row"
MASK = np.where(dv)[0]                       # the scored games, in wf's order (seasons ascending)
assert np.all(np.diff(SEAS[MASK]) >= 0)
print(f"spine {N} rows (1999-2015); DEV scored {len(MASK)}; rows without a pbp feature row: {len(nomiss)} "
      f"(all pre-2006, training only): {nomiss}")
print("harness source sha256[:16]:", HARNESS_SHA)
corr = {f"c{k}": round(float(np.corrcoef(LV[dv], X14[dv, k])[0, 1]), 3) for k in range(14) if X14[dv, k].std() > 0}
corr["c1_starter"] = round(float(np.corrcoef(LV[dv], QBS[dv])[0, 1]), 3)
corr_qv = {"QV~c1_starter": round(float(np.corrcoef(QV[dv], QBS[dv])[0, 1]), 3),
           "QV~c1_listed": round(float(np.corrcoef(QV[dv], X14[dv, 1])[0, 1]), 3),
           "QV~LV": round(float(np.corrcoef(QV[dv], LV[dv])[0, 1]), 3)}
print("DEV corr LV vs columns:", corr)
print("DEV corr QV:", corr_qv)

# ------------------------------------------------------------------ run every model once
Z1 = np.zeros(N)
RUNS = {
    "C0 cached X14 (listed QB)": dict(),
    "R0 served blend (col1 starter-keyed) [PRIMARY REF]": dict(replace={1: QBS}),
    "H3 drop col 1 (zeroed)": dict(replace={1: Z1}),
    "A1 R0, col7 <- LV [PRIMARY]": dict(replace={1: QBS, 7: LV}),
    "A2 R0 + LV (15th col)": dict(extra=LV, replace={1: QBS}),
    "A3 A1 and col1 <- QV": dict(replace={1: QV, 7: LV}),
    "A1c C0, col7 <- LV": dict(replace={7: LV}),
    "A2c C0 + LV (15th col)": dict(extra=LV),
}
RES = {}
for k, kw in RUNS.items():
    ll, per, vec, coefs = wf(**kw)
    assert len(vec) == len(MASK)
    assert np.allclose(vec.mean(), ll)
    RES[k] = (ll, per, vec, coefs)
    print(f"  ran {k:<52} LL {ll:.6f}", flush=True)

# H5 identical masks: re-derive each arm's per-game vector from an independent fit loop with explicit
# indices and check it is the wf vector (so every arm scores the same games in the same order), and
# collect the coefficients of the replaced / added columns for the audit.


def refit(replace=None, extra=None):
    X = X14.copy()
    if replace:
        for c, v in replace.items():
            X[:, c] = v
    vec, idx, co = [], [], {}
    for s_ in range(DEV_LO, DEV_HI + 1):
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X
        if extra is not None:
            E = extra.reshape(-1, 1); mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=0.5 ** ((s_ - 1 - SEAS[tr]) / HL))
        vec.append(llv(Y[te], m.predict_proba(Xs[te])[:, 1])); idx.append(np.where(te)[0])
        co[s_] = {"c1": round(float(m.coef_[0][1]), 4), "c7": round(float(m.coef_[0][7]), 4)}
        if extra is not None:
            co[s_]["x15"] = round(float(m.coef_[0][14]), 4)
    return np.concatenate(vec), np.concatenate(idx), co


COEFS = {}
for k, kw in RUNS.items():
    v2, ix, co = refit(**kw)
    assert np.array_equal(ix, MASK), f"{k}: scored games differ from the common mask"
    assert np.allclose(v2, RES[k][2], atol=1e-10), f"{k}: independent refit does not reproduce wf"
    COEFS[k] = co

# ------------------------------------------------------------------ sanity
c0_ll, r0_ll, h3_ll = (RES[k][0] for k in ("C0 cached X14 (listed QB)",
                                            "R0 served blend (col1 starter-keyed) [PRIMARY REF]",
                                            "H3 drop col 1 (zeroed)"))
print(f"H1 C0 {c0_ll:.10f} (recorded 0.6229368800)")
assert abs(c0_ll - 0.6229368799678564) < 1e-9
print(f"H2 R0 {r0_ll:.6f} (pv_nfl_passing_game B1 recorded 0.623924)")
assert abs(r0_ll - 0.623924) < 1e-6
print(f"H4 zeroed col 1 {h3_ll:.6f} (deletion recorded 0.629086 in pv_nfl_compose_game G0x)")
assert abs(h3_ll - 0.629086) < 1e-6

rng = np.random.default_rng(SEED)
BIDX = rng.integers(0, len(MASK), size=(NBOOT, len(MASK)), dtype=np.int32)
DEVS = np.arange(DEV_LO, DEV_HI + 1)
SIDX = rng.integers(0, len(DEVS), size=(NBOOT, len(DEVS)))
SEAS_M = SEAS[MASK]


def stats(ref_key, arm_key, res=None):
    res = RES if res is None else res
    rl, rper, rvec, _ = res[ref_key]
    al, aper, avec, _ = res[arm_key]
    d = rvec - avec                                   # > 0: arm better
    bs = np.empty(NBOOT)
    for a in range(0, NBOOT, 1000):
        bs[a:a + 1000] = d[BIDX[a:a + 1000]].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ssum = np.array([d[SEAS_M == s].sum() for s in DEVS]); scnt = np.array([(SEAS_M == s).sum() for s in DEVS])
    sb = ssum[SIDX].sum(1) / scnt[SIDX].sum(1)
    slo, shi = np.percentile(sb, [2.5, 97.5])
    per = {str(s): round(rper[s] - aper[s], 5) for s in DEVS}
    pos = sum(1 for s in DEVS if rper[s] - aper[s] > 0)
    return {"dev_ll": round(al, 6), "ref_dev_ll": round(rl, 6), "gain": round(float(d.mean()), 6),
            "ci95_game": [round(float(lo), 6), round(float(hi), 6)],
            "ci95_season_block": [round(float(slo), 6), round(float(shi), 6)],
            "p_boot_gain_le_0": round(float((bs <= 0).mean()), 4),
            "seasons_better": f"{pos}/10", "per_season_gain": per,
            "clears_bar": bool(d.mean() >= BAR and lo > 0),
            "season_block_lo_gt_0": bool(slo > 0)}


R0K, C0K = "R0 served blend (col1 starter-keyed) [PRIMARY REF]", "C0 cached X14 (listed QB)"
h3 = stats(R0K, "H3 drop col 1 (zeroed)")
h3c = stats(C0K, "H3 drop col 1 (zeroed)")
leak_c0 = stats(R0K, C0K)
print(f"H3 known effect, drop col 1: vs R0 {h3['gain']:+.6f} CI{h3['ci95_game']}  "
      f"vs C0 {h3c['gain']:+.6f} CI{h3c['ci95_game']} (recorded -0.006149 vs C0)")
assert h3["ci95_game"][1] < 0 and -0.0080 < h3["gain"] < -0.0035
assert abs(h3c["gain"] - (-0.006149)) < 2e-6
print(f"     listed-QB (post-game passer) col 1 vs served starter-keyed: {leak_c0['gain']:+.6f} CI{leak_c0['ci95_game']}")

ARMS = {"A1 R0, col7 <- LV [PRIMARY]": R0K, "A2 R0 + LV (15th col)": R0K, "A3 A1 and col1 <- QV": R0K}
SECOND = {"A1c C0, col7 <- LV": C0K, "A2c C0 + LV (15th col)": C0K, "A3 A1 and col1 <- QV": C0K}
OUT_ARMS, OUT_SEC = {}, {}
print("\nARMS vs R0 (served blend; > 0 = arm better)")
for k, ref in ARMS.items():
    st = stats(ref, k); st["coefs_by_fold"] = COEFS[k]; OUT_ARMS[k] = st
    print(f"  {k:<32} LL {st['dev_ll']:.6f} gain {st['gain']:+.6f} CI{st['ci95_game']} "
          f"blockCI{st['ci95_season_block']} seasons+ {st['seasons_better']} bar {st['clears_bar']}")
print("SECONDARY constructions vs C0 (cached listed-QB X14; sensitivity only)")
for k, ref in SECOND.items():
    st = stats(ref, k); OUT_SEC[k + " vs C0"] = st
    print(f"  {k:<32} LL {st['dev_ll']:.6f} gain {st['gain']:+.6f} CI{st['ci95_game']} "
          f"blockCI{st['ci95_season_block']} seasons+ {st['seasons_better']}")
a1a2 = stats("A2 R0 + LV (15th col)", "A1 R0, col7 <- LV [PRIMARY]")
a3a1 = stats("A1 R0, col7 <- LV [PRIMARY]", "A3 A1 and col1 <- QV")
print(f"A1 vs A2 (same model on DEV up to L2 scaling): {a1a2['gain']:+.6f}; "
      f"A3 vs A1 (QV replaces shipped QbElo given LV): {a3a1['gain']:+.6f} CI{a3a1['ci95_game']}")

# ------------------------------------------------------------------ POST-HOC: optimizer convergence
TOL_CONV = 1e-10


def wf_conv(replace=None, extra=None):
    """quick3.wf with the lbfgs run to convergence (tol 1e-10); everything else identical."""
    X = X14.copy()
    if replace:
        for c, v in replace.items():
            X[:, c] = v
    per, vec, iters = {}, [], []
    for s_ in range(DEV_LO, DEV_HI + 1):
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X
        if extra is not None:
            E = extra.reshape(-1, 1); mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        m = LogisticRegression(C=C, max_iter=200_000, tol=TOL_CONV).fit(
            Xs[tr], Y[tr], sample_weight=0.5 ** ((s_ - 1 - SEAS[tr]) / HL))
        assert m.n_iter_[0] < 200_000
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1]); per[s_] = float(v.mean()); vec.append(v)
        iters.append(int(m.n_iter_[0]))
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), iters


CONV_KEYS = [C0K, R0K, "H3 drop col 1 (zeroed)", "A1 R0, col7 <- LV [PRIMARY]", "A2 R0 + LV (15th col)",
             "A3 A1 and col1 <- QV"]
RESC = {k: wf_conv(**RUNS[k]) for k in CONV_KEYS}
CONV = {"tol": TOL_CONV, "harness_default_tol": 1e-4,
        "iters_by_fold": {k: RESC[k][3] for k in CONV_KEYS},
        "dev_ll": {k: round(RESC[k][0], 6) for k in CONV_KEYS},
        "shift_vs_harness_ll": {k: round(RESC[k][0] - RES[k][0], 6) for k in CONV_KEYS}}
print("\nPOST-HOC converged sensitivity (tol 1e-10; not used for the bar)")
for k in CONV_KEYS:
    print(f"  {k:<52} LL {RESC[k][0]:.6f} (harness {RES[k][0]:.6f}, shift {RESC[k][0]-RES[k][0]:+.6f}) "
          f"iters {RESC[k][3]}")
CONV["h3_drop_col1_vs_R0"] = {k_: v_ for k_, v_ in stats(R0K, "H3 drop col 1 (zeroed)", RESC).items()
                              if k_ in ("gain", "ci95_game")}
CONV["listed_vs_starter_col1"] = {k_: v_ for k_, v_ in stats(R0K, C0K, RESC).items() if k_ in ("gain", "ci95_game")}
CONV["arms_vs_R0"] = {}
for k in ("A1 R0, col7 <- LV [PRIMARY]", "A2 R0 + LV (15th col)", "A3 A1 and col1 <- QV"):
    st = stats(R0K, k, RESC); CONV["arms_vs_R0"][k] = st
    print(f"  {k:<32} conv gain {st['gain']:+.6f} CI{st['ci95_game']} blockCI{st['ci95_season_block']} "
          f"seasons+ {st['seasons_better']}")
CONV["A1_vs_A2_converged"] = stats("A2 R0 + LV (15th col)", "A1 R0, col7 <- LV [PRIMARY]", RESC)["gain"]
print(f"  H3 drop col1 vs R0 (conv) {CONV['h3_drop_col1_vs_R0']}; listed vs starter col1 (conv) "
      f"{CONV['listed_vs_starter_col1']}; A1 vs A2 conv {CONV['A1_vs_A2_converged']:+.6f}")

# ------------------------------------------------------------------ leak audit (DEV rows only)
W = json.load(open(os.path.join(DATA, "pv_nfl_compose_weights.json"), encoding="utf-8"))
weights_asof = {"design_train": W["design"]["train"], "folds": sorted(W["weights_by_fold"])[:3] + ["..."],
                "w_QB_by_dev_fold": {s: round(W["weights_by_fold"][s]["QB"], 3) for s in map(str, DEVS)}}
O = pd.read_csv(os.path.join(DATA, "pv_nfl_compose_game_outcomes.csv"))
assert O.season.max() < TEST_ERA
Fr = F.reset_index()
tg = pd.concat([
    pd.DataFrame({"game_id": Fr.game_id, "season": Fr.season, "team": Fr.home, "lv": Fr.lv_home, "qv": Fr.qv_home}),
    pd.DataFrame({"game_id": Fr.game_id, "season": Fr.season, "team": Fr.away, "lv": Fr.lv_away, "qv": Fr.qv_away})])
om = O.set_index("game_id").margin
tg["margin"] = [om[g] if t == F.loc[g, "home"] else -om[g] for g, t in zip(tg.game_id, tg.team)]
tg = tg.sort_values(["team", "season", "game_id"]).reset_index(drop=True)
tg["gix"] = tg.game_id.map({g: i for i, g in enumerate(Fr.game_id)})
tg = tg.sort_values(["team", "season", "gix"]).reset_index(drop=True)
grp = tg.groupby(["team", "season"]).margin
tg["m_prev"] = grp.shift(1); tg["m_next"] = grp.shift(-1)
probe = {}
for col in ("lv", "qv"):
    t = tg[(tg.season >= DEV_LO) & tg.m_prev.notna() & tg.m_next.notna()]
    rng2 = np.random.default_rng(SEED + 1)
    v = t[col].to_numpy(); mp, mt, mn = t.m_prev.to_numpy(), t.margin.to_numpy(), t.m_next.to_numpy()
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    bsd = []
    for _ in range(2000):
        ii = rng2.integers(0, len(v), len(v))
        bsd.append(r(v[ii], mt[ii]) - r(v[ii], mn[ii]))
    probe[col] = {"n_team_games": int(len(t)), "r_prev_game_margin": round(r(v, mp), 4),
                  "r_this_game_margin": round(r(v, mt), 4), "r_next_game_margin": round(r(v, mn), 4),
                  "this_minus_next_ci95": [round(float(x), 4) for x in np.percentile(bsd, [2.5, 97.5])]}
print("timing probe (team-games, DEV):", probe)
st_agree = []
for i in MASK:
    r_ = rowsF[i]; s_ = SS[GID[i]]
    st_agree += [r_["starter_home"] == s_[2], r_["starter_away"] == s_[3]]
starter_check = {"dev_sides": len(st_agree), "agree_with_shipped_starter": int(sum(st_agree)),
                 "disagree": int(len(st_agree) - sum(st_agree)),
                 "note": "compose takes the first dropback passer; the shipped walk requires a QB weekly "
                         "line, so trick/fake-play first passers differ (QV of those sides is a non-QB)"}
print("starter check:", starter_check)

primary = OUT_ARMS["A1 R0, col7 <- LV [PRIMARY]"]
OUT = {
    "screen": "NFL composed player value (pv_nfl_compose) - pre-registered game-level DEV screen",
    "dev_seasons": [DEV_LO, DEV_HI], "n_dev_games": int(len(MASK)),
    "harness": {"wf": "bt_nfl_ideas_quick3.wf extracted verbatim via ast", "sha256_16": HARNESS_SHA,
                "matrix": "data/bt_nfl_ideas_dev_X.npy", "HL": HL, "C": C},
    "bar": "gain >= +0.00150 and per-game bootstrap 95% CI lower bound > 0 (PRIMARY arm vs PRIMARY ref)",
    "bootstrap": {"n": NBOOT, "seed": SEED, "paired_same_indices_all_arms": True,
                  "season_block": "10 DEV seasons resampled with replacement"},
    "primary_ref": {"name": R0K, "dev_ll": round(r0_ll, 6)},
    "secondary_ref": {"name": C0K, "dev_ll": round(c0_ll, 6)},
    "sanity": {"H1_C0_reproduces_0.622937": round(c0_ll, 10), "H2_R0_reproduces_0.623924": round(r0_ll, 6),
               "H3_drop_col1_vs_R0": h3, "H3_drop_col1_vs_C0": h3c, "H4_zeroed_equals_deleted_0.629086": round(h3_ll, 6),
               "H5_identical_masks": "asserted: every run scores the same 2,670 DEV games in the same order",
               "listed_vs_starter_col1": leak_c0},
    "arms": OUT_ARMS, "secondary": OUT_SEC,
    "converged_sensitivity_posthoc": CONV,
    "contrasts": {"A1_vs_A2": a1a2["gain"], "A3_vs_A1": {k: a3a1[k] for k in ("gain", "ci95_game", "seasons_better")}},
    "corr_LV_with_cols_DEV": corr, "corr_QV_DEV": corr_qv,
    "audit": {"weights_asof": weights_asof, "timing_probe": probe, "starter_check": starter_check,
              "rows_without_feature_row": nomiss},
    "verdict": {"primary": "A1", "gain": primary["gain"], "ci95_game": primary["ci95_game"],
                "ci95_season_block": primary["ci95_season_block"], "clears_bar": primary["clears_bar"],
                "any_arm_clears": any(v["clears_bar"] for v in OUT_ARMS.values())},
}
json.dump(OUT, open(os.path.join(DATA, "pv_nfl_screen.json"), "w"), indent=1)
print("\nVERDICT:", OUT["verdict"])
print("wrote data/pv_nfl_screen.json")
