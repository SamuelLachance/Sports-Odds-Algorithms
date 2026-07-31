"""NFL DEPTH LEG — component-level re-tune of the SHIPPED model. DEV SCREENS ONLY.

Two harnesses, two different TEST eras. Every scoring path hard-asserts its own
mask, and the ratings-only harness physically TRUNCATES the engine walk at the
end of 2021 so a TEST-era game is never even computed.

  ratings-only split  warm 2016 | DEV 2017-2021 | TEST 2022-2025  (NEVER touched)
     axis A  v7 engine knobs, jointly (beta, tau, draw band/eps, play weights,
             opponent-quality k)
     axis B  QB Kalman fusion internals (obs noise R, z-scale k, opponent
             adjustment, volume-scaled R)

  main split          DEV 1999-2015 (scored 2006-2015) | TEST 2016-2025 (NEVER)
     axis C  blend model class + training protocol (recency half-life x C,
             HistGradientBoosting, convex stack)
     axis D  functional form + mechanistically-justified interactions on the 14

Method discipline (the point of the campaign):
  * JOINT search, not greedy: axis A = Jacobi coordinate descent (>=2 passes) +
    coarse random search over the joint grid; axis B = a full 4-way grid.
  * Every reported gain carries (a) fold-to-fold noise, measured as the std of
    the per-season deltas, and (b) a nested leave-one-season-out estimate of
    what the SELECTION is actually worth, so search optimism is quantified
    rather than assumed away.

Usage
  python -X utf8 phase0/nfl_depth_eval.py --axis A --workers 4
  python -X utf8 phase0/nfl_depth_eval.py --axis B --workers 4
  python -X utf8 phase0/nfl_depth_eval.py --axis CD
  python -X utf8 phase0/nfl_depth_eval.py --report      # -> data/nfl_depth_dev.json
  (internal) --run-configs <in.json> --out <stem>       # sharded engine worker
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

T0 = time.time()
PART_DIR = os.environ.get(
    "NFL_DEPTH_PARTS", os.path.join(tempfile.gettempdir(), "nfl_depth_parts"))
os.makedirs(PART_DIR, exist_ok=True)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- prelude ----
_SRC = open("phase0/nfl_ratings_only_model.py", encoding="utf-8").read()
_HEAD = _SRC.split("# ================= CORE KNOB SWEEP")[0]
assert "def run_config" in _HEAD and "CORE KNOB SWEEP" not in _HEAD
exec(compile(_HEAD, "nfl_ratings_only_model.py<head>", "exec"), globals())  # noqa: S102
print(f"[{time.time()-T0:.0f}s] depth prelude ready", flush=True)

from sklearn.linear_model import LogisticRegression  # noqa: E402

# ============================ RATINGS-ONLY HARNESS ============================
# G16 is a module global read inside run_config().  Rebinding it to the 2016-2021
# PREFIX means the engine walk stops at the end of DEV: no TEST-era game is ever
# simulated, let alone scored.  Valid only because G16 is season-monotone.
G16_FULL = list(G16)
SEAS_FULL = np.array([g["season"] for _, g in G16_FULL])
Y_FULL = np.array([g["y"] for _, g in G16_FULL])
assert np.all(np.diff(SEAS_FULL) >= 0), "G16 not season-monotone: prefix truncation invalid"
N_DEV = int((SEAS_FULL <= 2021).sum())
assert SEAS_FULL[:N_DEV].max() == 2021 and SEAS_FULL[N_DEV:].min() == 2022
G16 = G16_FULL[:N_DEV]                      # noqa: F811  (deliberate global rebind)
SEAS_D, Y_D = SEAS_FULL[:N_DEV], Y_FULL[:N_DEV]
RO_TEST_ERA = 2022
assert SEAS_D.max() < RO_TEST_ERA, "ratings-only DEV mask leaked into TEST"
SEL_LO, SEL_HI = 2017, 2021
assert SEL_HI < RO_TEST_ERA

BASE_A = dict(tau=0.30, use_lev=True, down_mult=1.25, late_mult=1.5,
              playoff_mult=1.5, dead_mult=0.5, opp_k=0.3, opp_mode="league",
              draw_band=0.40, beta=60.0, lev_exp=0.5, draw_eps=0.05,
              fus_k=4.0, fus_R=36.0, sal_k=1.0, sal_mode="cohort")


def wf_ratings(X):
    """walk-forward yearly refit, DEV seasons only.  Hard-asserts the mask."""
    assert X.shape[0] == N_DEV
    per, vecs = {}, []
    for s_ in range(SEL_LO, SEL_HI + 1):
        assert s_ < RO_TEST_ERA
        tr = (SEAS_D < s_) & (Y_D != 0.5)
        te = SEAS_D == s_
        assert SEAS_D[te].max() < RO_TEST_ERA and SEAS_D[tr].max() < s_
        m = LogisticRegression(C=1e6, max_iter=3000).fit(X[tr], Y_D[tr])
        v = llv(Y_D[te], m.predict_proba(X[te])[:, 1])
        per[str(s_)] = float(v.mean())
        vecs.append(v)
    a = np.concatenate(vecs)
    return float(a.mean()), per, a


def eval_engine(kw):
    """one engine walk (2016-2021 only) -> (overall DEV LL, per-season, per-game)."""
    _, P = run_config(**{**BASE_A, **kw}, return_parts=True)
    X = (P["qb_prec"] + P["off_prec"] + P["def_prec"]).reshape(-1, 1)
    return wf_ratings(X)


# ------------------------------------------------------- shard worker mode ---
def worker(in_path, out_stem):
    spec = json.load(open(in_path, encoding="utf-8"))
    out, vecs = {}, {}
    for name, kw in spec["configs"].items():
        t = time.time()
        ll, per, v = eval_engine(kw)
        out[name] = {"ll": round(ll, 6), "per": {k: round(x, 6) for k, x in per.items()},
                     "kw": kw}
        vecs[name] = v.astype(np.float32)
        print(f"  {name:<44} {ll:.5f}  [{time.time()-t:.0f}s]", flush=True)
    atomic_json(out, out_stem + ".json")
    np.savez_compressed(out_stem + ".npz", **vecs)


# ------------------------------------------------------------ orchestration --
class Store:
    """accumulating result store for one axis (metrics json + per-game npz)."""

    def __init__(self, axis):
        self.axis = axis
        self.jpath = os.path.join(PART_DIR, f"axis{axis}.json")
        self.npath = os.path.join(PART_DIR, f"axis{axis}.npz")
        self.res = json.load(open(self.jpath, encoding="utf-8")) if os.path.exists(self.jpath) else {}
        self.vec = dict(np.load(self.npath)) if os.path.exists(self.npath) else {}

    def have(self, name):
        return name in self.res

    def merge(self, res, vec):
        self.res.update(res)
        self.vec.update(vec)
        atomic_json(self.res, self.jpath)
        np.savez_compressed(self.npath, **self.vec)

    def ll(self, name):
        return self.res[name]["ll"]


def sig(kw):
    """canonical name for a config: only the knobs that differ from BASE_A."""
    d = {k: v for k, v in kw.items() if BASE_A.get(k) != v}
    return "base" if not d else "|".join(f"{k}={v}" for k, v in sorted(d.items()))


def run_stage(store, configs, workers, tag):
    """configs: dict name -> kw.  Skips already-computed names."""
    todo = {n: kw for n, kw in configs.items() if not store.have(n)}
    if not todo:
        print(f"[{tag}] nothing to do", flush=True)
        return
    names = list(todo)
    n_w = max(1, min(workers, len(names)))
    chunks = [names[i::n_w] for i in range(n_w)]
    procs, stems = [], []
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        stem = os.path.join(PART_DIR, f"{store.axis}_{tag}_{i}")
        ip = stem + ".in.json"
        atomic_json({"configs": {n: todo[n] for n in ch}}, ip)
        stems.append(stem)
        procs.append(subprocess.Popen(
            [sys.executable, "-X", "utf8", "phase0/nfl_depth_eval.py",
             "--run-configs", ip, "--out", stem],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    print(f"[{tag}] {len(names)} configs over {len(procs)} workers", flush=True)
    for p in procs:
        _, err = p.communicate()
        if p.returncode != 0:
            raise SystemExit(f"worker failed:\n{err.decode(errors='replace')[-3000:]}")
    res, vec = {}, {}
    for stem in stems:
        res.update(json.load(open(stem + ".json", encoding="utf-8")))
        vec.update(dict(np.load(stem + ".npz")))
    store.merge(res, vec)
    print(f"[{tag}] done  [{time.time()-T0:.0f}s]", flush=True)


# --------------------------------------------------------- honest statistics -
def noise_and_gain(store, base_name, cand_name):
    """delta = base - cand (positive = candidate better).  Per-season folds."""
    pb = store.res[base_name]["per"]
    pc = store.res[cand_name]["per"]
    d = np.array([pb[k] - pc[k] for k in sorted(pb)])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    gain = float(store.ll(base_name) - store.ll(cand_name))
    return {"gain": round(gain, 5), "per_season_delta": [round(x, 5) for x in d],
            "fold_sd": round(sd, 5), "fold_se": round(se, 5),
            "gain_in_se": round(gain / se, 2) if se > 0 else None}


def nested_loso(store, base_name, pool):
    """Leave-one-season-out SELECTION.  For each DEV season s, pick the best
    config using ONLY the other seasons, then score it on s.  The gap between
    the raw best and this is the search optimism."""
    seasons = sorted(store.res[base_name]["per"])
    picks, held = {}, []
    for s in seasons:
        others = [x for x in seasons if x != s]
        best = min(pool, key=lambda n: np.mean([store.res[n]["per"][o] for o in others]))
        picks[s] = best
        held.append(store.res[best]["per"][s])
    nested = float(np.mean(held))
    base = float(np.mean([store.res[base_name]["per"][s] for s in seasons]))
    raw_best = min(pool, key=lambda n: store.ll(n))
    raw = store.ll(raw_best)
    return {"raw_best": raw_best, "raw_best_ll": round(raw, 5),
            "raw_gain": round(base - raw, 5),
            "nested_ll": round(nested, 5), "nested_gain": round(base - nested, 5),
            "optimism": round((base - raw) - (base - nested), 5),
            "loso_picks": picks}


def paired_bootstrap(store, base_name, cand_name, seed=7):
    a, b = store.vec[base_name], store.vec[cand_name]
    d = (a - b).astype(np.float64)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(d.mean()), 5), "ci": [round(float(lo), 5), round(float(hi), 5)],
            "n": int(len(d))}


# ================================== AXIS A ===================================
KNOBS_A = [
    ("beta",         [30.0, 45.0, 60.0, 80.0, 110.0]),
    ("tau",          [0.15, 0.22, 0.30, 0.42, 0.60]),
    ("draw_band",    [0.20, 0.30, 0.40, 0.55, 0.75]),
    ("draw_eps",     [0.02, 0.05, 0.10, 0.20]),
    ("lev_exp",      [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("down_mult",    [1.0, 1.25, 1.5, 1.75]),
    ("late_mult",    [1.0, 1.25, 1.5, 2.0]),
    ("playoff_mult", [1.0, 1.5, 2.0, 2.5]),
    ("dead_mult",    [0.25, 0.5, 0.75, 1.0]),
    ("opp_k",        [0.0, 0.15, 0.3, 0.5, 0.7]),
]


def axis_A(workers):
    st = Store("A")
    run_stage(st, {"base": {}}, 1, "base")
    anchor = dict(BASE_A)

    # --- pass 1 == the 1-D surface scan around the shipped point -------------
    scan = {}
    for k, grid in KNOBS_A:
        for v in grid:
            if v == anchor[k]:
                continue
            scan[sig({**anchor, k: v})] = {k: v}
    run_stage(st, scan, workers, "scanA")

    # --- Jacobi coordinate descent: >=2 passes, joint move each pass ---------
    passes = []
    for p in range(1, 4):
        moves = {}
        for k, grid in KNOBS_A:
            cands = [(st.ll(sig({**anchor, k: v})), v) for v in grid
                     if st.have(sig({**anchor, k: v}))]
            if not cands:
                continue
            ll_b, v_b = min(cands)
            if v_b != anchor[k]:
                moves[k] = (v_b, st.ll(sig(anchor)) - ll_b)
        if not moves:
            passes.append({"pass": p, "moves": {}, "anchor_ll": st.ll(sig(anchor))})
            break
        # greedy cumulative combinations, ordered by 1-D improvement
        order = sorted(moves, key=lambda k: -moves[k][1])
        combos = {}
        cur = dict(anchor)
        for i, k in enumerate(order, 1):
            cur = {**cur, k: moves[k][0]}
            combos[sig(cur)] = {k2: cur[k2] for k2 in cur if BASE_A.get(k2) != cur[k2]}
        run_stage(st, combos, workers, f"cdA{p}")
        best_name = min(list(combos) + [sig(anchor)], key=st.ll)
        new_anchor = {**BASE_A, **st.res[best_name]["kw"]} if best_name != sig(anchor) else anchor
        passes.append({"pass": p, "moves": {k: moves[k][0] for k in moves},
                       "chosen": best_name, "anchor_ll": round(st.ll(best_name), 5)})
        if best_name == sig(anchor):
            break
        anchor = new_anchor
        # next pass needs the 1-D alternatives evaluated AT the new anchor
        nxt = {}
        for k, grid in KNOBS_A:
            for v in grid:
                if v != anchor[k]:
                    nxt[sig({**anchor, k: v})] = {k2: val for k2, val in {**anchor, k: v}.items()
                                                  if BASE_A.get(k2) != val}
        run_stage(st, nxt, workers, f"scanA{p+1}")

    # --- coarse random search over the joint grid ----------------------------
    rng = np.random.default_rng(20260731)
    rnd = {}
    while len(rnd) < 40:
        cfg = {k: float(grid[rng.integers(len(grid))]) for k, grid in KNOBS_A}
        d = {k: v for k, v in cfg.items() if BASE_A.get(k) != v}
        if d:
            rnd[sig({**BASE_A, **d})] = d
    run_stage(st, rnd, workers, "randA")

    atomic_json({"passes": passes, "final_anchor": anchor},
                os.path.join(PART_DIR, "axisA_meta.json"))
    print(f"[axis A] {len(st.res)} configs evaluated", flush=True)


# ================================== AXIS B ===================================
def axis_B(workers):
    st = Store("B")
    run_stage(st, {"base": {}}, 1, "base")
    grid = {}
    for fk in (2.0, 4.0, 6.0, 8.0):
        for fR in (18.0, 36.0, 72.0):
            for fa in (0.0, 0.25, 0.6):
                for fv in (0.0, 1.0):
                    d = {"fus_k": fk, "fus_R": fR, "fus_adj": fa, "fus_vol": fv}
                    d = {k: v for k, v in d.items() if BASE_A.get(k, 0.0) != v}
                    if d:
                        grid[sig({**BASE_A, **d})] = d
    run_stage(st, grid, workers, "gridB")
    best = min(st.res, key=st.ll)
    bkw = st.res[best]["kw"]
    ref = {}
    for fk in (1.0, 3.0, 5.0, 7.0, 10.0, 12.0):
        d = {**bkw, "fus_k": fk}
        ref[sig({**BASE_A, **d})] = d
    for fR in (8.0, 12.0, 25.0, 48.0, 100.0, 150.0):
        d = {**bkw, "fus_R": fR}
        ref[sig({**BASE_A, **d})] = d
    for fa in (-0.5, -0.25, 0.1, 0.4, 0.9):
        d = {**bkw, "fus_adj": fa}
        ref[sig({**BASE_A, **d})] = d
    ref = {n: d for n, d in ref.items() if n not in st.res}
    run_stage(st, ref, workers, "refB")
    print(f"[axis B] {len(st.res)} configs evaluated", flush=True)


def axis_X(workers):
    """Additivity check: does the axis-B fusion tune survive on top of the
    axis-A engine tune, or were both searches fitting the same DEV noise?"""
    sa, sb = Store("A"), Store("B")
    a_best = sa.res[min(sa.res, key=sa.ll)]["kw"]
    b_pool = sorted([n for n in sb.res if n != "base"], key=sb.ll)[:3]
    st = Store("X")
    cfgs = {"A_best": dict(a_best), "base": {}}
    for n in b_pool:
        cfgs["A_best+" + n] = {**a_best, **sb.res[n]["kw"]}
        cfgs["B:" + n] = dict(sb.res[n]["kw"])
    run_stage(st, cfgs, workers, "crossX")
    print(f"[axis X] {len(st.res)} configs evaluated", flush=True)


# ============================== MAIN-SPLIT HARNESS ===========================
MAIN_LO, MAIN_HI = 2006, 2015
MAIN_TEST_ERA = 2016
assert MAIN_HI < MAIN_TEST_ERA


def build_main():
    """the shipped 14-feature matrix + the extra raw pieces axis D needs."""
    import csv
    aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
    with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        ix = {c: i for i, c in enumerate(hdr)}
        for r in rd:
            t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
            a = aggc[r[ix["game_id"]]][t_]
            if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
                a[0] += float(r[ix["epa"]]); a[1] += 1
            else:
                a[2] += float(r[ix["epa"]]); a[3] += 1
    ps = pn = rs = rn = 0.0
    for gid, tm in aggc.items():
        if int(gid[:4]) in DEV_YEARS:
            for t_, (a_, b_, c_, d_) in tm.items():
                ps += a_; pn += b_; rs += c_; rn += d_
    LGP, LGR = ps / pn, rs / rn
    dec_, pn_, sd_ = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
    oP = defaultdict(lambda: [0.0, 0.0]); oR = defaultdict(lambda: [0.0, 0.0])
    dP = defaultdict(lambda: [0.0, 0.0]); dR = defaultdict(lambda: [0.0, 0.0])

    def crate(st_, lg_):
        return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)

    n = len(games)
    f_pass = np.zeros(n); f_run = np.zeros(n)
    f_dpass = np.zeros(n); f_drun = np.zeros(n)
    dpass_h = np.zeros(n); dpass_a = np.zeros(n)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in (list(oP.values()) + list(oR.values())
                        + list(dP.values()) + list(dR.values())):
                st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f_pass[i] = (crate(oP[h], LGP) - crate(dP[h], LGP)) - (crate(oP[a], LGP) - crate(dP[a], LGP))
        f_run[i] = (crate(oR[h], LGR) - crate(dR[h], LGR)) - (crate(oR[a], LGR) - crate(dR[a], LGR))
        dpass_h[i] = crate(dP[h], LGP); dpass_a[i] = crate(dP[a], LGP)
        f_dpass[i] = dpass_h[i] - dpass_a[i]
        f_drun[i] = crate(dR[h], LGR) - crate(dR[a], LGR)
        tm = aggc.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
                o = oP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
                o = oR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
                d2 = dP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
                d2 = dR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN
    X14 = np.column_stack([X_of(F), f_pass, f_run])
    v7 = np.load("data/nfl_v7_feature.npy")
    assert len(v7) == len(games)
    X14[:, 7] = v7

    # per-side QB ratings (same walk as feature 1, both halves retained)
    m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb,
              rep_delta=pedP["delta"], rep_map=rep_map, decay=SP["decay"],
              prior_db=SP["prior_db"], season_decay=SP["season_decay"])
    qh = np.zeros(n); qa = np.zeros(n)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        qh[i] = m.qb_adj(g["home_qb"]); qa[i] = m.qb_adj(g["away_qb"])
        m.predict(g); m.update(g, 0.5, qbw)
    assert np.abs((qh - qa) - X14[:, 1]).max() < 1e-9, "QB side split does not reproduce feature 1"

    raw_rest = np.array([g["hrest"] - g["arest"] for g in games], float)
    ex = {"qh": qh, "qa": qa, "dpass_h": dpass_h, "dpass_a": dpass_a,
          "f_dpass": f_dpass, "f_drun": f_drun, "raw_rest": raw_rest}
    return X14, ex


def _z(a, tr):
    m, s = a[tr].mean(), a[tr].std()
    return (a - m) / (s if s > 1e-12 else 1.0)


HL_BASE, C_BASE = 3.0, 100.0


def wf_main(X14, aug=None, HL=HL_BASE, C=C_BASE, model="logit", hgb=None, mix=0.5,
            seed=7):
    """Shipped training protocol, restricted to the main-split DEV era.

    aug(X14, tr_mask) -> extra columns, built from TRAIN-FOLD statistics only.
    Returns (overall LL, per-season LL, per-game LL, mean train LL)."""
    assert MAIN_HI < MAIN_TEST_ERA
    per, vecs, trlls = {}, [], []
    for s_ in range(MAIN_LO, MAIN_HI + 1):
        assert s_ < MAIN_TEST_ERA
        tr = (seasons < s_) & (y != 0.5)
        te = seasons == s_
        assert seasons[te].max() < MAIN_TEST_ERA and seasons[tr].max() < s_
        X = X14 if aug is None else np.column_stack([X14, aug(X14, tr)])
        w = 0.5 ** ((s_ - 1 - seasons[tr]) / HL) if np.isfinite(HL) else np.ones(int(tr.sum()))
        if model in ("logit", "stack"):
            lm = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr], sample_weight=w)
            p_lo = lm.predict_proba(X[te])[:, 1]
            p_tr = lm.predict_proba(X[tr])[:, 1]
        if model in ("hgb", "stack"):
            from sklearn.ensemble import HistGradientBoostingClassifier
            gm = HistGradientBoostingClassifier(random_state=seed, **(hgb or {}))
            gm.fit(X[tr], y[tr], sample_weight=w)
            p_gb = gm.predict_proba(X[te])[:, 1]
            p_gbtr = gm.predict_proba(X[tr])[:, 1]
        if model == "logit":
            p, ptr = p_lo, p_tr
        elif model == "hgb":
            p, ptr = p_gb, p_gbtr
        else:
            p = (1 - mix) * p_lo + mix * p_gb
            ptr = (1 - mix) * p_tr + mix * p_gbtr
        v = llv(y[te], p)
        per[str(s_)] = float(v.mean())
        vecs.append(v)
        trlls.append(float(llv(y[tr], ptr).mean()))
    a = np.concatenate(vecs)
    return float(a.mean()), per, a, float(np.mean(trlls))


def loso_main(per_map, base_per, pool):
    seasons_ = sorted(base_per)
    held = []
    picks = {}
    for s in seasons_:
        others = [o for o in seasons_ if o != s]
        best = min(pool, key=lambda nm: np.mean([per_map[nm][o] for o in others]))
        picks[s] = best
        held.append(per_map[best][s])
    nested = float(np.mean(held))
    base = float(np.mean([base_per[s] for s in seasons_]))
    raw_best = min(pool, key=lambda nm: np.mean(list(per_map[nm].values())))
    raw = float(np.mean(list(per_map[raw_best].values())))
    return {"raw_best": raw_best, "raw_best_ll": round(raw, 5),
            "raw_gain": round(base - raw, 5), "nested_ll": round(nested, 5),
            "nested_gain": round(base - nested, 5),
            "optimism": round(nested - raw, 5), "loso_picks": picks}


def fold_stats(base_per, cand_per):
    d = np.array([base_per[k] - cand_per[k] for k in sorted(base_per)])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    gain = float(np.mean(list(base_per.values())) - np.mean(list(cand_per.values())))
    return {"gain": round(gain, 5), "fold_sd": round(sd, 5), "fold_se": round(se, 5),
            "gain_in_se": round(gain / se, 2) if se > 0 else None,
            "per_season_delta": [round(x, 5) for x in d]}


# ================================ AXES C + D =================================
def axis_CD():
    X14, ex = build_main()
    out = {}
    dev_mask = (seasons >= MAIN_LO) & (seasons <= MAIN_HI)
    out["_meta"] = {
        "scored_seasons": [MAIN_LO, MAIN_HI], "n_scored": int(dev_mask.sum()),
        "col7_v7_nonzero_in_dev": int((X14[dev_mask, 7] != 0).sum()),
        "cols_8_11_nonzero_in_dev": {
            str(c): int((X14[dev_mask, c] != 0).sum()) for c in (8, 9, 10, 11)},
    }
    print("meta:", out["_meta"], flush=True)

    # ---- C1: recency half-life x ridge C, JOINTLY -------------------------
    per_map, tab = {}, {}
    HLS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, float("inf")]
    CS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0, 1e6]
    for HL in HLS:
        for C in CS:
            nm = f"HL={HL}|C={C:g}"
            ll, per, _, tr = wf_main(X14, HL=HL, C=C)
            per_map[nm] = per
            tab[nm] = {"ll": round(ll, 5), "train_ll": round(tr, 5)}
            print(f"  {nm:<24} {ll:.5f}", flush=True)
    base_nm = f"HL={HL_BASE}|C={C_BASE:g}"
    out["C1_halflife_x_C"] = {
        "baseline": base_nm, "baseline_ll": tab[base_nm]["ll"], "grid": tab,
        "loso": loso_main(per_map, per_map[base_nm], list(per_map)),
        "best_vs_base": fold_stats(per_map[base_nm],
                                   per_map[min(tab, key=lambda n: tab[n]["ll"])]),
        "best": min(tab, key=lambda n: tab[n]["ll"]),
    }

    # ---- C2: model class on the identical features ------------------------
    gb_grid = {
        "hgb_shallow": dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=4,
                            min_samples_leaf=60, l2_regularization=1.0),
        "hgb_tiny":    dict(max_iter=120, learning_rate=0.05, max_leaf_nodes=3,
                            min_samples_leaf=120, l2_regularization=5.0),
        "hgb_default": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                            min_samples_leaf=20, l2_regularization=0.0),
        "hgb_deep":    dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                            min_samples_leaf=40, l2_regularization=1.0),
        "hgb_reg":     dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=8,
                            min_samples_leaf=100, l2_regularization=10.0),
    }
    c2, per2 = {}, {}
    for nm, hp in gb_grid.items():
        ll, per, _, tr = wf_main(X14, model="hgb", hgb=hp)
        c2[nm] = {"ll": round(ll, 5), "train_ll": round(tr, 5),
                  "train_cv_gap": round(ll - tr, 5), "params": hp}
        per2[nm] = per
        print(f"  {nm:<24} {ll:.5f}  (train {tr:.5f}, gap {ll-tr:+.5f})", flush=True)
    best_gb = min(c2, key=lambda n: c2[n]["ll"])
    for mix in (0.15, 0.3, 0.5):
        nm = f"stack_logit+{best_gb}_mix{mix}"
        ll, per, _, tr = wf_main(X14, model="stack", hgb=gb_grid[best_gb], mix=mix)
        c2[nm] = {"ll": round(ll, 5), "train_ll": round(tr, 5), "mix": mix}
        per2[nm] = per
        print(f"  {nm:<32} {ll:.5f}", flush=True)
    lb, lp = tab[base_nm]["ll"], per_map[base_nm]
    out["C2_model_class"] = {
        "logit_baseline_ll": lb, "logit_train_ll": tab[base_nm]["train_ll"],
        "logit_train_cv_gap": round(lb - tab[base_nm]["train_ll"], 5),
        "models": c2,
        "vs_logit": {nm: fold_stats(lp, per2[nm]) for nm in c2},
        "loso": loso_main({**per2, base_nm: lp}, lp, list(per2) + [base_nm]),
    }

    # ---- C3: robustness of the ONE positive survivor ----------------------
    ll_b, per_b, vec_b, _ = wf_main(X14)
    c3 = {"mix_curve": {}, "seed_sensitivity": {}}
    for mix in (0.15, 0.3, 0.5, 0.65, 0.8, 1.0):
        ll, per, _, _ = wf_main(X14, model="stack", hgb=gb_grid[best_gb], mix=mix)
        c3["mix_curve"][str(mix)] = round(ll, 5)
    for sd in (7, 17, 27, 101):
        ll, per, vec, _ = wf_main(X14, model="stack", hgb=gb_grid[best_gb], mix=0.5, seed=sd)
        c3["seed_sensitivity"][str(sd)] = round(ll, 5)
        if sd == 7:
            vec_s, per_s = vec, per
    d_game = (vec_b - vec_s).astype(np.float64)
    rng = np.random.default_rng(7)
    bs = d_game[rng.integers(0, len(d_game), size=(10000, len(d_game)))].mean(axis=1)
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    era_a = [k for k in per_b if int(k) <= 2012]
    era_b = [k for k in per_b if int(k) >= 2013]
    c3.update({
        "stack_mix0.5_ll": round(ll_b - (ll_b - c3["mix_curve"]["0.5"]), 5),
        "paired_bootstrap_game_level": {
            "delta": round(float(d_game.mean()), 5),
            "ci": [round(float(lo_), 5), round(float(hi_), 5)], "n": int(len(d_game))},
        "per_season_delta": {k: round(per_b[k] - per_s[k], 5) for k in sorted(per_b)},
        "gain_2006_2012": round(float(np.mean([per_b[k] - per_s[k] for k in era_a])), 5),
        "gain_2013_2015_absence_cols_live": round(
            float(np.mean([per_b[k] - per_s[k] for k in era_b])), 5),
    })
    out["C3_stack_robustness"] = c3
    print("C3:", json.dumps(c3), flush=True)

    # ---- D: functional form + mechanistic interactions --------------------
    qh, qa = ex["qh"], ex["qa"]
    dph, dpa = ex["dpass_h"], ex["dpass_a"]
    rr = ex["raw_rest"]

    def A_rest_sqrt(X, tr):
        r = X[:, 5]
        return (np.sign(r) * np.sqrt(np.abs(r)))[:, None]

    def A_rest_sq(X, tr):
        r = X[:, 5]
        return (np.sign(r) * r * r)[:, None]

    def A_bye(X, tr):
        return ((rr >= 11).astype(float) - (-rr >= 11).astype(float))[:, None]

    def A_abs_convex(X, tr):
        a = X[:, 8:11]
        return np.sign(a) * a * a

    def A_abs_total(X, tr):
        t = X[:, 8] + X[:, 9] + X[:, 10]
        return np.column_stack([t, np.sign(t) * t * t])

    def A_qb_x_passd(X, tr):
        # home QB's value scales with how bad the AWAY pass D is, and vice versa;
        # dpass_* is EPA/play ALLOWED (higher = worse defence).
        zq_h = _z(qh, tr); zq_a = _z(qa, tr)
        zd_h = _z(dph, tr); zd_a = _z(dpa, tr)
        return (zq_h * zd_a - zq_a * zd_h)[:, None]

    def A_rest_x_travel(X, tr):
        # extra rest should blunt the west-team-at-early-ET body-clock penalty
        return (_z(X[:, 5], tr) * X[:, 3])[:, None]

    def A_abs_x_rq(X, tr):
        # missing snaps cost more when the replacement pool behind them is weak
        t = X[:, 8] + X[:, 9] + X[:, 10]
        return (_z(t, tr) * _z(X[:, 11], tr))[:, None]

    def A_elo_x_epa(X, tr):
        # record-based and process-based edges agreeing should be super-additive
        return (_z(X[:, 0], tr) * _z(X[:, 2], tr))[:, None]

    D = {
        "D1a rest sign*sqrt|rest|": A_rest_sqrt,
        "D1b rest sign*rest^2": A_rest_sq,
        "D1c bye-week indicator (raw rest>=11)": A_bye,
        "D2a absence convex (ol/de/sk squared)": A_abs_convex,
        "D2b absence total + total^2": A_abs_total,
        "D3 QB x opponent pass-defence": A_qb_x_passd,
        "D4 rest x travel(early-ET) ": A_rest_x_travel,
        "D5 absence x roster_quality": A_abs_x_rq,
        "D6 elo_logit x epa_net": A_elo_x_epa,
    }
    dres, perD = {}, {}
    for nm, fn in D.items():
        ll, per, _, tr = wf_main(X14, aug=fn)
        dres[nm] = {"ll": round(ll, 5), **fold_stats(lp, per)}
        perD[nm] = per
        print(f"  {nm:<42} {ll:.5f} ({lb-ll:+.5f})", flush=True)
    # combined: every variant that individually beat the baseline
    winners = [nm for nm in dres if dres[nm]["ll"] < lb]
    if winners:
        def A_comb(X, tr):
            return np.column_stack([D[nm](X, tr) for nm in winners])
        ll, per, _, tr = wf_main(X14, aug=A_comb)
        dres["Dcomb (" + "; ".join(winners) + ")"] = {"ll": round(ll, 5), **fold_stats(lp, per)}
        perD["Dcomb"] = per
        print(f"  COMBINED {len(winners)} winners                    {ll:.5f} ({lb-ll:+.5f})",
              flush=True)
    out["D_form_and_interactions"] = {
        "baseline_ll": lb, "variants": dres,
        "loso": loso_main({**perD, "base14": lp}, lp, list(perD) + ["base14"]),
    }
    atomic_json(out, os.path.join(PART_DIR, "axisCD.json"))
    print(f"[axes C/D] done  [{time.time()-T0:.0f}s]", flush=True)


def subpool_loso(st, axis, pool):
    """Same nested LOSO, but over the pre-specifiable STAGES of the search, so
    the optimism can be attributed to a stage rather than to 'the search'."""
    subs = {}
    if axis == "A":
        one_d = {sig({**BASE_A, k: v}) for k, grid in KNOBS_A for v in grid}
        subs["one_knob_scans_only"] = [n for n in pool if n in one_d or n == "base"]
        subs["joint_configs_only"] = [n for n in pool if n not in one_d or n == "base"]
    elif axis == "B":
        G = {"fus_k": {2.0, 4.0, 6.0, 8.0}, "fus_R": {18.0, 36.0, 72.0},
             "fus_adj": {0.0, 0.25, 0.6}, "fus_vol": {0.0, 1.0}}
        def in_grid(kw):
            return all(kw.get(k, BASE_A.get(k, 0.0)) in v or kw.get(k, BASE_A.get(k, 0.0)) ==
                       BASE_A.get(k, 0.0) for k, v in G.items())
        subs["four_way_grid_only"] = [n for n in pool
                                      if n == "base" or in_grid(st.res[n]["kw"])]
        subs["refinement_only"] = [n for n in pool
                                   if n == "base" or not in_grid(st.res[n]["kw"])]
    out = {}
    for nm, sp in subs.items():
        if len(sp) > 1:
            r = nested_loso(st, "base", sp)
            out[nm] = {"n_pool": len(sp),
                       **{q: r[q] for q in ("raw_best", "raw_gain", "nested_gain", "optimism")}}
    return out


# ================================== REPORT ===================================
def report():
    rep = {
        "generated": time.strftime("%Y-%m-%d"),
        "protocol": {
            "ratings_only": {"warm": 2016, "dev": [SEL_LO, SEL_HI],
                             "test_never_touched": [2022, 2025],
                             "engine_walk_truncated_at": 2021,
                             "n_dev_games_walked": int(N_DEV)},
            "main": {"dev_train_from": 1999, "dev_scored": [MAIN_LO, MAIN_HI],
                     "test_never_touched": [2016, 2025]},
            "market_blind": True,
        },
    }
    for axis in ("A", "B", "X"):
        p = os.path.join(PART_DIR, f"axis{axis}.json")
        if not os.path.exists(p):
            continue
        st = Store(axis)
        pool = list(st.res)
        best = min(pool, key=st.ll)
        block = {
            "baseline_dev_ll": round(st.ll("base"), 5),
            "n_configs": len(pool),
            "best": best, "best_ll": round(st.ll(best), 5),
            "best_kw": st.res[best]["kw"],
            "best_vs_base": noise_and_gain(st, "base", best),
            "nested_loso": nested_loso(st, "base", pool),
            "nested_loso_by_subpool": subpool_loso(st, axis, pool),
            "paired_bootstrap_dev": paired_bootstrap(st, "base", best),
            "top10": [{"cfg": n, "ll": round(st.ll(n), 5),
                       "delta": round(st.ll("base") - st.ll(n), 5)}
                      for n in sorted(pool, key=st.ll)[:10]],
            "all": {n: round(st.ll(n), 5) for n in sorted(pool, key=st.ll)},
        }
        if axis == "A":
            surf = {}
            for k, grid in KNOBS_A:
                row = {}
                for v in grid:
                    nm = sig({**BASE_A, k: v})
                    if st.have(nm):
                        row[str(v)] = round(st.ll(nm) - st.ll("base"), 5)
                surf[k] = row
            block["surface_1d_ll_minus_base__negative_is_better"] = surf
            # a per-knob search is TINY (4-5 cells): its LOSO is an honest
            # estimate of what a ONE-KNOB re-tune is really worth.
            pk = {}
            for k, grid in KNOBS_A:
                pool_k = [sig({**BASE_A, k: v}) for v in grid if st.have(sig({**BASE_A, k: v}))]
                if len(pool_k) > 1:
                    n_ = nested_loso(st, "base", pool_k)
                    pk[k] = {q: n_[q] for q in ("raw_best", "raw_gain", "nested_gain", "optimism")}
            block["per_knob_1d_loso"] = pk
            mp = os.path.join(PART_DIR, "axisA_meta.json")
            if os.path.exists(mp):
                block["coordinate_descent"] = json.load(open(mp, encoding="utf-8"))
        rep["axis_" + axis] = block
    cdp = os.path.join(PART_DIR, "axisCD.json")
    if os.path.exists(cdp):
        cd = json.load(open(cdp, encoding="utf-8"))
        rep["main_split_meta"] = cd.pop("_meta")
        rep.update(cd)
    atomic_json(rep, "data/nfl_depth_dev.json")
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("axis") or k == "protocol"},
                     indent=1)[:4000])
    print("wrote data/nfl_depth_dev.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["A", "B", "CD", "X"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-configs")
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run_configs:
        worker(a.run_configs, a.out)
    elif a.axis == "A":
        axis_A(a.workers)
    elif a.axis == "B":
        axis_B(a.workers)
    elif a.axis == "CD":
        axis_CD()
    elif a.axis == "X":
        axis_X(a.workers)
    elif a.report:
        report()
    else:
        axis_A(a.workers); axis_B(a.workers); axis_X(a.workers); axis_CD(); report()


if __name__ == "__main__":
    main()
