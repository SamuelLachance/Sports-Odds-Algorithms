"""bt_nhl_rapmel STEP 2 -- walk-forward stint-RAPM player values (DEV only).

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md),
candidate nhl_rapmel. Design copied from nhl_rapm2.main (two rows per stint,
offense + defense indicator columns, response = attacking xGF/60 weighted by
duration, context = home / D-composition dummies per side / score state / puck
proxy at CSCALE 50, Ridge alpha 60,000 sparse_cg). nhl_rapm2.main() and
build_stints() are NEVER called; only infer_positions (diagnostic 'posdiag'
only) and constants are imported. Positions come from bt_nhl_rapmel_pos (the
helper oscillates on DEV windows -- see that module and the posdiag record).

Fits (walk-forward):
  pre_<s>  cutoff = day before season s's first game; stints of seasons
           max(2010-11, s-3) .. s-1
  jan_<s>  cutoff = Dec 31 inside season s; the same seasons + season s stints
           dated <= Dec 31. (2012-13 lockout: no game before Jan 19 2013, so the
           Jan-1 set would be identical to pre_20122013 and its cutoff precedes
           it -- it could never be selected by "latest cutoff < t"; skipped.)

Values per fit (spec):
  NET = OFF + (-DEF), each centred within position; rel = T/(T + 83,616);
  v_repl,pos = T-weighted mean pre-EB NET of the fringe pool 400 <= T < 20,000 s;
  mu_p = v_repl,pos if the player's career 5v5 stint seconds before the cutoff
  are < 20,000 else 0;  v_p = rel*NET + (1-rel)*mu_p.

W3: every fit asserts max(training stint date) <= cutoff; the screen asserts
cutoff < first date of use.

Usage:
  python phase0/bt_nhl_rapmel_fit.py fit <name>          one walk-forward fit
  python phase0/bt_nhl_rapmel_fit.py fit <name> --drop YYYY-MM-DD   (W2 refit)
  python phase0/bt_nhl_rapmel_fit.py s4b <odd|even>      held-out half fit
  python phase0/bt_nhl_rapmel_fit.py collect             -> values.csv, fits.json
  python phase0/bt_nhl_rapmel_fit.py list                print fit names
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, "phase0")
from nhl_depth_eval import dev_games  # noqa: E402
from nhl_glicko2_eval import DEV_END  # noqa: E402
from nhl_rapm2 import (COMP_LEVELS, CSCALE, EB_K, LAMBDA, NCTX,  # noqa: E402
                       SCORE_LEVELS, infer_positions)

PFX = "data/bt_nhl_rapmel_"
MAX_GID = 2018000000
SEASONS = [20102011, 20112012, 20122013, 20132014, 20142015, 20152016, 20162017,
           20172018]
EVAL_SEASONS = SEASONS[1:]
FRINGE_LO, FRINGE_HI = 400.0, 20000.0
VET_SECONDS = 20000.0
assert LAMBDA == 60000.0 and CSCALE == 50.0 and EB_K == 83616.0


def season_first_dates():
    first = {}
    for g in dev_games():
        first.setdefault(g["season"], g["date"])
    return first


def fit_specs():
    """name -> (cutoff 'YYYY-MM-DD', window seasons, include-season-s-through-cutoff)."""
    first = season_first_dates()
    specs = {}
    for s in EVAL_SEASONS:
        i = SEASONS.index(s)
        win = SEASONS[max(0, i - 3):i]
        pre_cut = (date.fromisoformat(first[s]) - timedelta(days=1)).isoformat()
        specs[f"pre_{s}"] = (pre_cut, win, None)
        jan_cut = f"{s // 10000}-12-31"
        if jan_cut > pre_cut:
            specs[f"jan_{s}"] = (jan_cut, win, s)
    return specs


def dint(d):
    return int(d.replace("-", ""))


def load_stints(seasons):
    keys = ["hi", "ai", "dur", "xgf", "xga", "sdiff", "prox", "gid", "date"]
    parts = {k: [] for k in keys}
    for s in seasons:
        assert s <= DEV_END
        for k in keys:
            parts[k].append(np.load(f"{PFX}{s}_{k}.npy"))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    assert out["gid"].max() < MAX_GID
    return out


def seed_positions():
    seed = {}
    d = json.load(open("data/nhl_player_names.json", encoding="utf-8"))
    for k, v in d.items():
        if v.get("pos") in ("D", "C", "L", "R"):
            seed[int(k)] = "D" if v["pos"] == "D" else "F"
    return seed


def build_design(st, pid, isD):
    """Exactly nhl_rapm2.main's design on stint dict st with column map pid."""
    n = len(pid)
    Hi = np.vectorize(pid.__getitem__, otypes=[np.int64])(st["hi"]) if len(st["hi"]) else \
        np.zeros((0, 5), np.int64)
    Ai = np.vectorize(pid.__getitem__, otypes=[np.int64])(st["ai"]) if len(st["ai"]) else \
        np.zeros((0, 5), np.int64)
    dur, xgf, xga = st["dur"], st["xgf"], st["xga"]
    sdiff, prox = st["sdiff"], st["prox"]
    S = len(dur)
    nD_h, nD_a = isD[Hi].sum(1), isD[Ai].sum(1)
    R = 2 * S
    off = np.empty((R, 5), dtype=np.int64)
    dfn = np.empty((R, 5), dtype=np.int64)
    off[0::2], dfn[0::2] = Hi, Ai
    off[1::2], dfn[1::2] = Ai, Hi
    y = np.empty(R)
    y[0::2] = xgf * 3600.0 / dur
    y[1::2] = xga * 3600.0 / dur
    w = np.repeat(dur, 2)
    nDo = np.empty(R, dtype=np.int8); nDd = np.empty(R, dtype=np.int8)
    nDo[0::2], nDo[1::2] = nD_h, nD_a
    nDd[0::2], nDd[1::2] = nD_a, nD_h
    sdo = np.empty(R, dtype=np.int8)
    sdo[0::2], sdo[1::2] = sdiff, -sdiff
    prx = np.empty(R, dtype=np.int8)
    prx[0::2], prx[1::2] = prox, -prox
    rows = [np.repeat(np.arange(R), 10)]
    cols = [np.concatenate([off, dfn + n], axis=1).ravel()]
    vals = [np.ones(R * 10)]

    def add_ctx(j, mask):
        w_ = np.where(mask)[0]
        rows.append(w_)
        cols.append(np.full(len(w_), 2 * n + j))
        vals.append(np.full(len(w_), CSCALE))

    add_ctx(0, np.arange(R) % 2 == 0)
    for j, k in enumerate(COMP_LEVELS):
        add_ctx(1 + j, nDo == k)
        add_ctx(1 + len(COMP_LEVELS) + j, nDd == k)
    for j, k in enumerate(SCORE_LEVELS):
        add_ctx(1 + 2 * len(COMP_LEVELS) + j, sdo == k)
    add_ctx(NCTX - 2, prx > 0)
    add_ctx(NCTX - 1, prx < 0)
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(R, 2 * n + NCTX))
    return X, y, w, Hi, Ai


def subset(st, m):
    return {k: v[m] for k, v in st.items()}


def player_index(st):
    pid = {}
    for arr in (st["hi"], st["ai"]):
        for p in np.unique(arr):
            pid.setdefault(int(p), len(pid))
    idx2p = np.empty(len(pid), dtype=np.int64)
    for p, i in pid.items():
        idx2p[i] = p
    return pid, idx2p


_GMETA = None


def game_meta():
    """gid -> (normalised home, normalised away, season); DEV games only."""
    global _GMETA
    if _GMETA is None:
        from nhl_features_eval import TEAM_FIX
        _GMETA = {g["game_id"]: (TEAM_FIX.get(g["home"], g["home"]),
                                 TEAM_FIX.get(g["away"], g["away"]), g["season"])
                  for g in dev_games()}
        assert max(_GMETA) < MAX_GID
    return _GMETA


def positions_for(st, pid, idx2p):
    """F/D labels from the window's own stints (bt_nhl_rapmel_pos; see its
    docstring for why nhl_rapm2.infer_positions cannot be used on DEV windows)."""
    from bt_nhl_rapmel_pos import label_positions
    seed = seed_positions()
    gm = game_meta()
    lk = np.vectorize(pid.__getitem__, otypes=[np.int64])
    Hl, Al = lk(st["hi"]), lk(st["ai"])
    loc_seed = {pid[p]: v for p, v in seed.items() if p in pid}
    isD, diag = label_positions(Hl, Al, st["dur"], st["gid"],
                                {g: m[0] for g, m in gm.items()},
                                {g: m[1] for g, m in gm.items()},
                                {g: m[2] for g, m in gm.items()}, loc_seed)
    assert len(isD) == len(pid)
    return isD, diag


def ridge(X, y, w):
    m = Ridge(alpha=LAMBDA, fit_intercept=True, solver="sparse_cg", max_iter=2000, tol=1e-6)
    m.fit(X, y, sample_weight=w)
    return m


def raw_net(coef, n, isD, present=None):
    """OFF, DEF (sign-flipped) centred within position; pre-EB NET."""
    OFF = coef[:n].copy()
    DEF = coef[n:2 * n].copy()
    base = np.ones(n, bool) if present is None else present
    for msk in (isD, ~isD):
        mm = msk & base
        OFF[msk] -= OFF[mm].mean()
        DEF[msk] -= DEF[mm].mean()
    DEF = -DEF
    return OFF, DEF, OFF + DEF


def do_fit(name, drop=None):
    specs = fit_specs()
    cutoff, win, s_in = specs[name]
    cut_i = dint(cutoff)
    seasons = list(win) + ([s_in] if s_in else [])
    st = load_stints(seasons)
    m = st["date"] <= cut_i
    if drop is not None:
        m &= st["date"] < dint(drop)
    st = subset(st, m)
    tag = name if drop is None else f"{name}_drop{drop}"
    if len(st["dur"]) == 0:
        # W2 only: every window stint is on/after the drop date -> no evidence;
        # every player then falls back to v_repl = 0 (a perturbation like any other)
        assert drop is not None
        pd.DataFrame(columns=["cutoff_date", "pid", "pos", "v", "T", "rel", "net_raw",
                              "off", "def", "careerT"]).to_csv(f"{PFX}fitv_{tag}.csv",
                                                               index=False)
        meta = {"name": name, "cutoff": cutoff, "window_seasons": win,
                "season_through_cutoff": s_in, "drop": drop, "n_stints": 0,
                "max_train_date": 0, "v_repl": {"F": 0.0, "D": 0.0}, "empty_window": True}
        json.dump(meta, open(f"{PFX}fitm_{tag}.json", "w"), indent=1)
        print(json.dumps(meta), flush=True)
        return
    max_train = int(st["date"].max())
    assert max_train <= cut_i, f"W3 violated: train {max_train} > cutoff {cut_i}"
    t0 = time.time()
    pid, idx2p = player_index(st)
    n = len(pid)
    isD, pos_diag = positions_for(st, pid, idx2p)
    X, y, w, Hi, Ai = build_design(st, pid, isD)
    mdl = ridge(X, y, w)
    coef = mdl.coef_
    T = np.zeros(n)
    np.add.at(T, Hi.ravel(), np.repeat(st["dur"], 5))
    np.add.at(T, Ai.ravel(), np.repeat(st["dur"], 5))
    OFF, DEF, NET = raw_net(coef, n, isD)
    rel = T / (T + EB_K)
    # career 5v5 stint seconds before the cutoff: every DEV season up to cutoff
    all_seasons = [s for s in SEASONS if s <= (s_in or win[-1])]
    car = defaultdict(float)
    stc = load_stints(all_seasons)
    mc = stc["date"] <= cut_i
    if drop is not None:
        mc &= stc["date"] < dint(drop)
    assert int(stc["date"][mc].max()) <= cut_i
    for arr in (stc["hi"][mc], stc["ai"][mc]):
        dd = np.repeat(stc["dur"][mc], 5)
        u, inv = np.unique(arr.ravel(), return_inverse=True)
        sums = np.bincount(inv, weights=dd)
        for p, v in zip(u.tolist(), sums.tolist()):
            car[p] += v
    fr = (T >= FRINGE_LO) & (T < FRINGE_HI)
    v_repl = {}
    for pos, msk in (("F", ~isD), ("D", isD)):
        mm = fr & msk
        v_repl[pos] = float(np.average(NET[mm], weights=T[mm]))
    careerT = np.array([car.get(int(p), 0.0) for p in idx2p])
    mu = np.where(careerT < VET_SECONDS,
                  np.where(isD, v_repl["D"], v_repl["F"]), 0.0)
    v = rel * NET + (1 - rel) * mu
    df = pd.DataFrame({"cutoff_date": cutoff, "pid": idx2p,
                       "pos": np.where(isD, "D", "F"), "v": v, "T": T, "rel": rel,
                       "net_raw": NET, "off": OFF, "def": DEF, "careerT": careerT})
    tag = name if drop is None else f"{name}_drop{drop}"
    df.to_csv(f"{PFX}fitv_{tag}.csv", index=False)
    meta = {"name": name, "cutoff": cutoff, "window_seasons": win,
            "season_through_cutoff": s_in, "drop": drop, "n_stints": int(len(st["dur"])),
            "max_train_date": max_train, "n_players": n, "n_D": int(isD.sum()),
            "positions": pos_diag, "v_repl": v_repl,
            "intercept": float(mdl.intercept_),
            "ctx": [float(c) * CSCALE for c in coef[2 * n:]],
            "net_sd_T60000": float(NET[T >= 60000].std()) if (T >= 60000).any() else None,
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(f"{PFX}fitm_{tag}.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


def do_s4b(half):
    """Held-out stint R2 + split-half halves on 2014-15..2016-17 (S4b / S4c)."""
    seasons = [20142015, 20152016, 20162017]
    st = load_stints(seasons)
    pid, idx2p = player_index(st)
    n = len(pid)
    isD, pos_diag = positions_for(st, pid, idx2p)
    X, y, w, Hi, Ai = build_design(st, pid, isD)
    odd = (st["gid"] % 2 == 1)
    tr_st = odd if half == "odd" else ~odd
    tr = np.repeat(tr_st, 2)
    t0 = time.time()
    mdl = ridge(X[tr], y[tr], w[tr])
    te = ~tr
    pred = mdl.predict(X[te])
    base = np.average(y[tr], weights=w[tr])
    sse = float(np.sum(w[te] * (y[te] - pred) ** 2))
    sse0 = float(np.sum(w[te] * (y[te] - base) ** 2))
    T = np.zeros(n)
    np.add.at(T, Hi[tr_st].ravel(), np.repeat(st["dur"][tr_st], 5))
    np.add.at(T, Ai[tr_st].ravel(), np.repeat(st["dur"][tr_st], 5))
    Tall = np.zeros(n)
    np.add.at(Tall, Hi.ravel(), np.repeat(st["dur"], 5))
    np.add.at(Tall, Ai.ravel(), np.repeat(st["dur"], 5))
    OFF, DEF, NET = raw_net(mdl.coef_, n, isD, present=T > 0)
    pd.DataFrame({"pid": idx2p, "pos": np.where(isD, "D", "F"), "net_raw": NET,
                  "T_half": T, "T_all": Tall}).to_csv(f"{PFX}s4b_{half}.csv", index=False)
    out = {"half": half, "sse": sse, "sse0": sse0, "r2_this_half": 1 - sse / sse0,
           "positions": pos_diag,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(f"{PFX}s4b_{half}.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


def posdiag():
    """Record why nhl_rapm2.infer_positions is not used: its 3F+2D side share on
    two DEV windows vs the replacement labeller."""
    out = {}
    for seasons in ([20102011], [20142015, 20152016, 20162017]):
        st = load_stints(seasons)
        pid, idx2p = player_index(st)
        seed = {p: v for p, v in seed_positions().items() if p in pid}
        lab = infer_positions(seed, st["hi"].tolist(), st["ai"].tolist(), list(pid))
        isD_h = np.array([lab.get(int(p), "F") == "D" for p in idx2p])
        isD_n, diag = positions_for(st, pid, idx2p)
        lk = np.vectorize(pid.__getitem__, otypes=[np.int64])
        sides = np.concatenate([lk(st["hi"]), lk(st["ai"])])
        ww = np.concatenate([st["dur"], st["dur"]])
        out[str(seasons)] = {
            "n_players": len(pid), "n_name_map_labels": len(seed),
            "infer_positions_share_sides_3F2D": float(np.average(isD_h[sides].sum(1) == 2,
                                                                 weights=ww)),
            "infer_positions_D_share": float(isD_h.mean()),
            "replacement_share_sides_3F2D": diag["share_sides_3F2D"],
            "replacement_D_share": float(isD_n.mean()),
            "replacement_unsupervised_agreement_with_name_map":
                diag["unsupervised_agreement_with_name_map"]}
        print(seasons, out[str(seasons)], flush=True)
    json.dump(out, open(f"{PFX}posdiag.json", "w"), indent=1)


def collect():
    specs = fit_specs()
    frames, metas = [], {}
    for name in specs:
        frames.append(pd.read_csv(f"{PFX}fitv_{name}.csv"))
        metas[name] = json.load(open(f"{PFX}fitm_{name}.json"))
        assert metas[name]["max_train_date"] <= dint(specs[name][0])
    v = pd.concat(frames, ignore_index=True)
    v[["cutoff_date", "pid", "pos", "v", "T", "rel", "net_raw", "careerT"]].to_csv(
        f"{PFX}values.csv", index=False)
    s4 = {}
    if os.path.exists(f"{PFX}s4b_odd.json") and os.path.exists(f"{PFX}s4b_even.json"):
        a = json.load(open(f"{PFX}s4b_odd.json"))
        b = json.load(open(f"{PFX}s4b_even.json"))
        s4["S4b_heldout_R2"] = 1 - (a["sse"] + b["sse"]) / (a["sse0"] + b["sse0"])
        s4["S4b_halves"] = [a["r2_this_half"], b["r2_this_half"]]
        o = pd.read_csv(f"{PFX}s4b_odd.csv").set_index("pid")
        e = pd.read_csv(f"{PFX}s4b_even.csv").set_index("pid")
        j = o.join(e, lsuffix="_o", rsuffix="_e")
        jj = j[j.T_all_o >= 60000]
        s4["S4c_split_half_r_net_1000min"] = float(np.corrcoef(jj.net_raw_o, jj.net_raw_e)[0, 1])
        s4["S4c_n_players"] = int(len(jj))
        for pos in ("F", "D"):
            q = jj[jj.pos_o == pos]
            s4[f"S4c_r_{pos}"] = float(np.corrcoef(q.net_raw_o, q.net_raw_e)[0, 1])
    json.dump({"fits": metas, "s4": s4}, open(f"{PFX}fits.json", "w"), indent=1)
    print(json.dumps(s4, indent=1))
    print(f"values rows {len(v):,} over {v.cutoff_date.nunique()} cutoffs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "s4b", "collect", "list", "posdiag"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--drop", default=None)
    a = ap.parse_args()
    if a.cmd == "fit":
        do_fit(a.arg, a.drop)
    elif a.cmd == "s4b":
        do_s4b(a.arg)
    elif a.cmd == "collect":
        collect()
    elif a.cmd == "posdiag":
        posdiag()
    else:
        for k, v in fit_specs().items():
            print(k, v[0], v[1], v[2])
