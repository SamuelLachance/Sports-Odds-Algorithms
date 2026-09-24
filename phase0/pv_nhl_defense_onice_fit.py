"""pv_nhl_defense_onice STEP 2 -- walk-forward on-ice defensive RAPM (DEV only).

Component `defense_onice` of the NHL player-value program: the defensive side
that has no individual event. For every walk-forward cutoff this fits ONE
design and solves it for several responses and ridge strengths at once:

  rows      two per 5v5 interval (each side attacking once), weight = seconds x
            recency decay; response = the attacking side's count x 3600 / dur
            (a per-60 rate), so the weighted least squares is the Poisson-
            variance GLS of the counts.
  columns   attacker OFFENCE indicators (5), defender DEFENCE indicators (5),
            the DEFENDING GOALIE (1; he is what separates a skater's goals-
            against from his goalie's save percentage), four "new player" group
            columns (count of attacking / defending F / D whose career 5v5
            seconds before the cutoff are < 20,000 -- the empirical-Bayes prior
            mean of an unproven player), context: home, attacker score state
            (-3..+3), attacker zone start of the interval (faceoff at t0 in the
            O / D / N zone, reference on-the-fly), season intercepts.
  responses xg  (xG against, MoneyPuck), g (goals against), c (all shot
            attempts against).  Plus  gx: goals against with the ridge prior
            mean set to the xg solution ("GA, heavily shrunk TOWARD the xGA
            rating" -- goals may only move a player where they insist).
  penalty   lambda on every player / goalie column (grid), ~0 on the rest.

Player defensive rating (positive = fewer against):
  d_p = -(beta_def_p + gamma_new_def_pos * new_p), centred within position on
  the established regulars of the window; a player unseen in the window gets
  the centred group value of his position ("unseen" rows in the output).

Walk-forward cutoffs (all DEV): pre_<s> = first day of season s (fit on
seasons s-3..s-1), and monthly m<yyyymmdd>_<s> in-season refits (seasons
s-3..s-1 plus season s strictly before the cutoff). Intervals with date >=
cutoff are never in a fit (asserted); ratings from cutoff C are used only for
games dated >= C.

Also fitted, for the reliability study: single-season fits (window = one season,
no decay) and odd/even game-id split-half fits.

Outputs: data/pv_nhl_defense_onice_fit_<name>.npz (ratings for all responses x
lambdas x decays), data/pv_nhl_defense_onice_fits.json (meta).

Usage:
  python phase0/pv_nhl_defense_onice_fit.py wf        all walk-forward fits
  python phase0/pv_nhl_defense_onice_fit.py rel       single-season + split-half
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID, load_rosters  # noqa: E402

SEASONS = [20102011, 20112012, 20122013, 20132014, 20142015, 20152016, 20162017,
           20172018]
EVAL_SEASONS = SEASONS[1:]
PFX = "data/pv_nhl_defense_onice_"
TARGETS = ("xg", "g", "c")
LAMBDAS = [3750.0, 7500.0, 15000.0, 30000.0, 60000.0, 120000.0, 240000.0, 480000.0,
           960000.0, 1920000.0, 3840000.0]
HALFLIVES = [0.0, 730.0]          # days; 0 = no decay
VET_SECONDS = 20000.0             # career 5v5 seconds below which a player is "new"
REG_SECONDS = 20000.0             # window seconds for "established regular" centring
CTX_PEN = 1e-3
N_CTX_BASE = 1 + 1 + 6 + 3        # intercept, home, score x6, zone x3
MONTHS = ["1101", "1201", "0101", "0201", "0301", "0401"]


def dord(d):
    d = int(d)
    return date(d // 10000, d // 100 % 100, d % 100).toordinal()


# ---------------------------------------------------------------- loading ---
_ST = {}


def load_season(s):
    if s not in _ST:
        z = np.load(f"{PFX}st_{s}.npz")
        d = {k: z[k] for k in z.files}
        assert d["gid"].max() < DEV_MAX_GID and s in SEASONS
        d["season"] = np.full(len(d["gid"]), s, np.int64)
        _ST[s] = d
    return _ST[s]


def load_window(seasons, date_lt=None, parity=None):
    parts = [load_season(s) for s in seasons]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    m = np.ones(len(out["gid"]), bool)
    if date_lt is not None:
        m &= out["date"] < date_lt
    if parity is not None:
        m &= (out["gid"] % 2) == parity
    return {k: v[m] for k, v in out.items()}


_POS = None


def roster_pos():
    """(gid, pid) -> 'D'/'F'/'G' from the dressed roster (known pre-game)."""
    global _POS
    if _POS is None:
        ro = load_rosters()
        ro = ro[ro.gtype == 2]
        p = np.where(ro.pos == "D", "D", np.where(ro.pos == "G", "G", "F"))
        _POS = pd.DataFrame({"gid": ro.gid.to_numpy(np.int64), "pid": ro.pid.to_numpy(np.int64),
                             "pos": p, "date": 0})
    return _POS


def window_positions(gids):
    ro = roster_pos()
    r = ro[ro.gid.isin(np.unique(gids))]
    r = r[r.pos != "G"]
    return r.groupby("pid").pos.agg(lambda s: s.value_counts().index[0]).to_dict()


_TOI = None


def toi5_all():
    global _TOI
    if _TOI is None:
        parts = [pd.read_csv(f"{PFX}toi5_{s}.csv") for s in SEASONS]
        t = pd.concat(parts, ignore_index=True)
        g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "date"])
        g = g[g.game_id < DEV_MAX_GID]
        dm = dict(zip(g.game_id, g.date.str.replace("-", "").astype(int)))
        t["date"] = t.gid.map(dm).astype(np.int64)
        assert t.gid.max() < DEV_MAX_GID
        _TOI = t
    return _TOI


def career_seconds(cutoff):
    t = toi5_all()
    t = t[t.date < cutoff]
    return t.groupby("pid").sec5.sum().to_dict()


# ----------------------------------------------------------------- design ---
def _idx(sorted_ids, q):
    i = np.searchsorted(sorted_ids, q)
    assert (sorted_ids[np.clip(i, 0, len(sorted_ids) - 1)] == q).all()
    return i.astype(np.int32)


def build(st, pos_map, career):
    """Unweighted sparse design + responses + per-player bookkeeping."""
    pids = np.unique(np.concatenate([st["hs"].ravel(), st["as_"].ravel()]))
    gids_ = np.unique(np.concatenate([st["hg"], st["ag"]]))
    n, ng = len(pids), len(gids_)
    isD = np.array([pos_map.get(int(p), "F") == "D" for p in pids])
    if career is None:            # reliability fits: no prior-mean groups
        new = np.zeros(n, bool)
    else:
        new = np.array([career.get(int(p), 0.0) < VET_SECONDS for p in pids])
    seasons = sorted(set(st["season"].tolist()))
    ns = len(seasons)
    S = len(st["dur"])
    R = 2 * S
    H = _idx(pids, st["hs"])
    A = _idx(pids, st["as_"])
    HG, AG = _idx(gids_, st["hg"]), _idx(gids_, st["ag"])
    c_grp0 = 2 * n + ng           # new-off-F, new-off-D, new-def-F, new-def-D
    c_ctx0 = c_grp0 + 4
    p = c_ctx0 + N_CTX_BASE + (ns - 1)
    # per row: 11 player/goalie entries, 4 group counts, ctx one-hots
    home_att = np.zeros(R, bool)
    home_att[0::2] = True
    sd = np.empty(R, np.int8)
    sd[0::2], sd[1::2] = st["sd"], -st["sd"]
    zs = st["zs"]
    zo = np.empty(R, np.int8)     # attacker perspective: 1 OZ, 2 DZ, 3 NZ, 0 OTF
    zo[0::2] = zs
    zo[1::2] = np.where(zs == 1, 2, np.where(zs == 2, 1, zs))
    seas = np.repeat(st["season"], 2)
    ctx_cols = [np.ones(R, bool), home_att]
    ctx_cols += [sd == k for k in (-3, -2, -1, 1, 2, 3)]
    ctx_cols += [zo == k for k in (1, 2, 3)]
    ctx_cols += [seas == s_ for s_ in seasons[1:]]
    rows_l, cols_l, vals_l = [], [], []
    for j in range(5):
        for arr_h, arr_a, base in ((H, A, 0), (A, H, n)):
            c = np.empty(R, np.int32)
            c[0::2], c[1::2] = arr_h[:, j] + base, arr_a[:, j] + base
            rows_l.append(np.arange(R, dtype=np.int32))
            cols_l.append(c)
            vals_l.append(np.ones(R, np.float64))
    cg = np.empty(R, np.int32)
    cg[0::2], cg[1::2] = AG + 2 * n, HG + 2 * n
    rows_l.append(np.arange(R, dtype=np.int32))
    cols_l.append(cg)
    vals_l.append(np.ones(R))
    newF, newD = new & ~isD, new & isD
    for j, (attack, msk) in enumerate(((True, newF), (True, newD), (False, newF), (False, newD))):
        ch = msk[H].sum(1)
        ca = msk[A].sum(1)
        cnt = np.empty(R, np.float64)
        if attack:
            cnt[0::2], cnt[1::2] = ch, ca
        else:
            cnt[0::2], cnt[1::2] = ca, ch
        w_ = np.nonzero(cnt)[0].astype(np.int32)
        rows_l.append(w_)
        cols_l.append(np.full(len(w_), c_grp0 + j, np.int32))
        vals_l.append(cnt[w_])
    for j, m in enumerate(ctx_cols):
        w_ = np.nonzero(m)[0].astype(np.int32)
        rows_l.append(w_)
        cols_l.append(np.full(len(w_), c_ctx0 + j, np.int32))
        vals_l.append(np.ones(len(w_)))
    X = sparse.csr_matrix((np.concatenate(vals_l), (np.concatenate(rows_l),
                                                   np.concatenate(cols_l))), shape=(R, p))
    del rows_l, cols_l, vals_l
    Y = {}
    dur = st["dur"].astype(float)
    for t in TARGETS:
        y = np.empty(R)
        y[0::2] = st[f"h_{t}"] * 3600.0 / dur
        y[1::2] = st[f"a_{t}"] * 3600.0 / dur
        Y[t] = y
    meta = dict(pids=pids, gpids=gids_, isD=isD, new=new, n=n, ng=ng, p=p, c_grp0=c_grp0,
                c_ctx0=c_ctx0, seasons=seasons, H=H, A=A)
    return X, Y, meta


def decay(st, cutoff, halflife):
    if halflife and halflife > 0:
        ud, inv = np.unique(st["date"], return_inverse=True)
        age = dord(cutoff) - np.array([dord(d) for d in ud])[inv]
        return np.exp(-np.log(2) * age / halflife)
    return np.ones(len(st["dur"]))


def bookkeeping(st, meta, dec):
    n = meta["n"]
    dur = st["dur"].astype(float)
    Tw, Traw = np.zeros(n), np.zeros(n)
    against = {t: np.zeros(n) for t in TARGETS}
    for side, oppk in ((meta["H"], "a"), (meta["A"], "h")):
        flat = side.ravel()
        Tw += np.bincount(flat, weights=np.repeat(dur * dec, 5), minlength=n)
        Traw += np.bincount(flat, weights=np.repeat(dur, 5), minlength=n)
        for t in TARGETS:
            against[t] += np.bincount(flat, weights=np.repeat(st[f"{oppk}_{t}"] * dec, 5),
                                      minlength=n)
    league = {t: float(np.sum((st[f"h_{t}"] + st[f"a_{t}"]) * dec) * 3600.0
                        / (2 * np.sum(dur * dec))) for t in TARGETS}
    return Tw, Traw, against, league


def solve_all(X, w, Y, meta, lambdas):
    """Ratings for every response x lambda, one factorisation per lambda."""
    n, ng, p = meta["n"], meta["ng"], meta["p"]
    Xw = X.copy()
    Xw.data *= np.repeat(np.sqrt(w), np.diff(X.indptr))
    A = (Xw.T @ Xw).toarray()
    del Xw
    XtWy = {t: X.T @ (w * Y[t]) for t in TARGETS}
    npl = 2 * n + ng
    out = {}
    for lam in lambdas:
        pen = np.full(p, CTX_PEN)
        pen[:npl] = lam
        M = A.copy()
        M[np.diag_indices(p)] += pen
        cf = cho_factor(M, lower=False, check_finite=False)
        for t in TARGETS:
            out[(t, lam)] = cho_solve(cf, XtWy[t], check_finite=False)
    return A, XtWy, out


def solve_gx(A, XtWy, meta, beta_x, lam_g):
    n, ng, p = meta["n"], meta["ng"], meta["p"]
    npl = 2 * n + ng
    pen = np.full(p, CTX_PEN)
    pen[:npl] = lam_g
    mu = np.zeros(p)
    mu[:npl] = beta_x[:npl]
    M = A.copy()
    M[np.diag_indices(p)] += pen
    return np.linalg.solve(M, XtWy["g"] + pen * mu)


def ratings(beta, meta):
    """Defensive rating per player (positive = fewer against), centred within
    position on established regulars; unseen value per position; goalie coef."""
    n, ng = meta["n"], meta["ng"]
    isD, new = meta["isD"], meta["new"]
    bdef = beta[n:2 * n]
    gF, gD = beta[meta["c_grp0"] + 2], beta[meta["c_grp0"] + 3]
    raw = -(bdef + np.where(new, np.where(isD, gD, gF), 0.0))
    unseen = {"F": -gF, "D": -gD}
    reg = (~new) & (meta["Traw"] >= REG_SECONDS)
    d = raw.copy()
    for pos, msk in (("F", ~isD), ("D", isD)):
        mm = msk & reg
        c = float(np.average(raw[mm], weights=meta["Tw"][mm])) if mm.any() else 0.0
        d[msk] -= c
        unseen[pos] -= c
    boff = beta[:n]
    goal = beta[2 * n:2 * n + ng]
    return d, unseen, boff, goal


def raw_onice(meta, t, K):
    """EB-shrunk raw on-ice rate against (no teammate/opponent adjustment)."""
    Tw = meta["Tw"]
    rate = np.where(Tw > 0, meta["against"][t] * 3600.0 / np.maximum(Tw, 1.0), meta["league"][t])
    return -(rate - meta["league"][t]) * Tw / (Tw + K)


RAW_K = [30000.0, 60000.0, 120000.0, 240000.0, 480000.0]


def fit_one(spec):
    """spec: dict(name, seasons, cutoff (yyyymmdd int or None), parity, halflives)."""
    t0 = time.time()
    name = spec["name"]
    st = load_window(spec["seasons"], spec.get("cutoff"), spec.get("parity"))
    cutoff = spec.get("cutoff") or (int(st["date"].max()) + 1)
    assert int(st["date"].max()) < cutoff, "W: training interval on/after cutoff"
    assert st["gid"].max() < DEV_MAX_GID
    pos_map = window_positions(st["gid"])
    career = career_seconds(cutoff) if spec.get("career", True) else None
    res = {}
    X, Y, meta = build(st, pos_map, career)
    for hl in spec["halflives"]:
        dec = decay(st, cutoff, hl)
        Tw, Traw, against, league = bookkeeping(st, meta, dec)
        meta.update(Tw=Tw, Traw=Traw, against=against, league=league)
        w = np.repeat(st["dur"].astype(float) * dec, 2)
        A, XtWy, sol = solve_all(X, w, Y, meta, spec.get("lambdas", LAMBDAS))
        for (t, lam), beta in sol.items():
            d, unseen, boff, goal = ratings(beta, meta)
            res[f"d_{t}_{int(lam)}_{int(hl)}"] = d
            res[f"u_{t}_{int(lam)}_{int(hl)}"] = np.array([unseen["F"], unseen["D"]])
            res[f"g_{t}_{int(lam)}_{int(hl)}"] = goal
        for lx in spec.get("gx_lx", [60000.0]):
            for lg in spec.get("gx_lg", [480000.0, 1920000.0, 7680000.0]):
                beta = solve_gx(A, XtWy, meta, sol[("xg", lx)], lg)
                d, unseen, _, goal = ratings(beta, meta)
                res[f"d_gx{int(lx)}_{int(lg)}_{int(hl)}"] = d
                res[f"u_gx{int(lx)}_{int(lg)}_{int(hl)}"] = np.array([unseen["F"], unseen["D"]])
                res[f"g_gx{int(lx)}_{int(lg)}_{int(hl)}"] = goal
        del A, XtWy, sol
        for t in TARGETS:
            for K in RAW_K:
                res[f"r_{t}_{int(K)}_{int(hl)}"] = raw_onice(meta, t, K)
        res[f"Tw_{int(hl)}"] = meta["Tw"]
        res[f"league_{int(hl)}"] = np.array([meta["league"][t] for t in TARGETS])
    meta0 = meta
    np.savez_compressed(f"{PFX}fit_{name}.npz", pids=meta0["pids"], isD=meta0["isD"],
                        new=meta0["new"], Traw=meta0["Traw"], gpids=meta0["gpids"], **res)
    info = {"name": name, "cutoff": int(cutoff), "seasons": spec["seasons"],
            "parity": spec.get("parity"), "n_intervals": int(len(st["dur"])),
            "max_train_date": int(st["date"].max()), "n_players": int(meta0["n"]),
            "n_goalies": int(meta0["ng"]), "hours": float(st["dur"].sum() / 3600),
            "goals": int(st["h_g"].sum() + st["a_g"].sum()),
            "league": meta0["league"], "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in info.items() if k != "league"}), flush=True)
    return info


def season_first_date(s):
    return int(load_season(s)["date"].min())


def wf_specs():
    specs = []
    for s in EVAL_SEASONS:
        i = SEASONS.index(s)
        prior = SEASONS[max(0, i - 3):i]
        first = season_first_date(s)
        specs.append(dict(name=f"pre_{s}", seasons=prior, cutoff=first, halflives=HALFLIVES,
                          kind="pre", season=s))
        last = int(load_season(s)["date"].max())
        y0 = s // 10000
        for mm in MONTHS:
            yr = y0 if mm[:2] in ("11", "12") else y0 + 1
            c = int(f"{yr}{mm}")
            if first < c <= last:
                specs.append(dict(name=f"m{c}_{s}", seasons=prior + [s], cutoff=c,
                                  halflives=HALFLIVES, kind="month", season=s))
    return specs


def rel_specs():
    """Single-season fits and odd/even split-half fits (reliability study)."""
    specs = []
    for s in SEASONS:
        specs.append(dict(name=f"one_{s}", seasons=[s], cutoff=None, halflives=[0.0],
                          kind="one", season=s, career=False, gx_lg=[1920000.0]))
        for par in (0, 1):
            specs.append(dict(name=f"half{par}_{s}", seasons=[s], cutoff=None, parity=par,
                              halflives=[0.0], kind="half", season=s, career=False,
                              gx_lg=[1920000.0]))
    for par in (0, 1):
        specs.append(dict(name=f"h3{par}_2015_17", seasons=[20142015, 20152016, 20162017],
                          cutoff=None, parity=par, halflives=[0.0], kind="half3",
                          season=20162017, career=False, gx_lg=[1920000.0]))
    return specs


def run(specs, tag, procs=4):
    with Pool(procs) as pool:
        infos = pool.map(fit_one, specs, chunksize=1)
    path = f"{PFX}fits_{tag}.json"
    json.dump({"specs": [{k: v for k, v in s.items()} for s in specs], "fits": infos},
              open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "wf"
    if cmd == "wf":
        run(wf_specs(), "wf")
    elif cmd == "rel":
        run(rel_specs(), "rel")
    elif cmd == "one":
        sp = [s for s in wf_specs() + rel_specs() if s["name"] == sys.argv[2]][0]
        fit_one(sp)
    else:
        raise SystemExit("usage: wf | rel | one <name>")
