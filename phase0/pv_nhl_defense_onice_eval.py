"""pv_nhl_defense_onice STEP 3 -- player-level validation on DEV (no game outcomes).

Reads the walk-forward fits of phase0/pv_nhl_defense_onice_fit.py and asks, on
DEV seasons only:

  predict   PREDICTIVE VALIDITY at the unit level. For every 5v5 interval of an
            evaluation season s (2011-12..2017-18) and each defending side,
            S = sum of the five defenders' walk-forward ratings (fit strictly
            before the interval's date). The FUTURE count against (goals = the
            primary target; xG and shot attempts reported beside it) is modelled
            as Poisson with log mu = log(dur/3600) + context + b*S (context =
            intercept, home, attacker score state, attacker zone start). The gain
            is the deviance drop vs context only (b and context fitted on season s
            itself: one free parameter per arm per season, the same for every arm).
            "face" = the same with b fixed at face value (S taken literally in
            per-60 units): no free parameter at all.
            Hyperparameters (ridge lambda, recency half-life, raw-rate shrink K)
            are chosen by NESTED leave-one-season-out: the pick for season s is the
            best on the other six seasons. Paired bootstrap over games for every
            family-vs-family comparison.
  players   the same future-outcome test at the player-season level: prior rating
            vs the player's future on-ice GA/60 (raw and relative to his team).
  rel       reliability: year-over-year (single-season fits s vs s+1) and split-
            half (odd vs even game ids) correlations, with n.
  face      top / bottom lists (last DEV walk-forward fit) and the goalie column.

Arms ("families"):
  XGA   on-ice xG-against RAPM            (d_xg_<lambda>_<halflife>)
  GA    on-ice goals-against RAPM         (d_g_<lambda>_<halflife>, heavy lambdas)
  GAX   goals-against RAPM shrunk TOWARD the xGA solution (d_gx60000_<lg>_<hl>)
  CA    on-ice shot-attempts-against RAPM (d_c_...)
  RAWGA / RAWXGA  EB-shrunk raw on-ice rates, no teammate/opponent adjustment
  CUR_DEF / CUR_V the CURRENT rating: bt_nhl_rapmel pre-season stint-xG RAPM,
                  defensive half (def) and the net value v that carries it

PROTOCOL: every array is asserted gid < 2018000000; nothing touches a TEST season.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID, load_rosters  # noqa: E402
import pv_nhl_defense_onice_fit as F  # noqa: E402

PFX = F.PFX
OUT = f"{PFX}eval.json"
EVAL = F.EVAL_SEASONS
LAMS = [int(x) for x in F.LAMBDAS]
HLS = [0, 730]
RAWK = [int(x) for x in F.RAW_K]
GXL = [480000, 1920000, 7680000]
TGT_ALL = ("g", "xg", "c")


def families():
    fam = {
        "XGA": [f"d_xg_{l}_{h}" for l in LAMS for h in HLS],
        "GA": [f"d_g_{l}_{h}" for l in LAMS for h in HLS],
        "GAX": [f"d_gx60000_{l}_{h}" for l in GXL for h in HLS],
        "CA": [f"d_c_{l}_{h}" for l in LAMS for h in HLS],
        "RAWGA": [f"r_g_{k}_{h}" for k in RAWK for h in HLS],
        "RAWXGA": [f"r_xg_{k}_{h}" for k in RAWK for h in HLS],
    }
    return fam


# --------------------------------------------------------------- eval data ---
_RO = None


def roster_isD():
    global _RO
    if _RO is None:
        ro = load_rosters()
        ro = ro[(ro.gtype == 2) & (ro.pos != "G")]
        key = ro.gid.to_numpy(np.int64) * 10_000_000 + (ro.pid.to_numpy(np.int64) - 8_000_000)
        o = np.argsort(key)
        _RO = (key[o], (ro.pos.to_numpy() == "D")[o])
    return _RO


def isD_of(gid, pid):
    keys, vals = roster_isD()
    k = gid * 10_000_000 + (pid - 8_000_000)
    i = np.searchsorted(keys, k)
    i = np.clip(i, 0, len(keys) - 1)
    ok = keys[i] == k
    return np.where(ok, vals[i], False)


_GT = None


def game_teams():
    global _GT
    if _GT is None:
        g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "home", "away", "type"])
        g = g[(g.game_id < DEV_MAX_GID) & (g.type == 2)]
        _GT = {int(a): (h, w) for a, h, w in zip(g.game_id, g.home, g.away)}
    return _GT


class Season:
    """Two rows per 5v5 interval of season s (row = one side attacking)."""

    def __init__(self, s):
        st = F.load_season(s)
        assert st["gid"].max() < DEV_MAX_GID
        S = len(st["dur"])
        R = 2 * S
        self.s = s
        self.gid = np.repeat(st["gid"], 2)
        self.date = np.repeat(st["date"], 2)
        self.dur = np.repeat(st["dur"].astype(float), 2)
        dfn = np.empty((R, 5), np.int64)
        dfn[0::2], dfn[1::2] = st["as_"], st["hs"]
        self.dfn = dfn
        self.def_home = np.zeros(R, bool)
        self.def_home[1::2] = True
        home_att = np.zeros(R, bool)
        home_att[0::2] = True
        sd = np.empty(R, np.int64)
        sd[0::2], sd[1::2] = st["sd"], -st["sd"]
        zo = np.empty(R, np.int64)
        zo[0::2] = st["zs"]
        zo[1::2] = np.where(st["zs"] == 1, 2, np.where(st["zs"] == 2, 1, st["zs"]))
        cols = [np.ones(R), home_att.astype(float)]
        cols += [(sd == k).astype(float) for k in (-3, -2, -1, 1, 2, 3)]
        cols += [(zo == k).astype(float) for k in (1, 2, 3)]
        self.Xc = np.column_stack(cols)
        self.y = {}
        for t in TGT_ALL:
            y = np.empty(R)
            y[0::2] = st[f"h_{t}"]
            y[1::2] = st[f"a_{t}"]
            self.y[t] = y
        self.off = np.log(self.dur / 3600.0)
        self.isD = isD_of(np.repeat(self.gid[:, None], 5, 1), dfn)
        # fixed-effect design for the WITHIN spec: defending team-season, attacking
        # team-season and defending goalie-season one-hots (all in-season, common to
        # every arm: the arm must explain which UNITS allow more, not which teams)
        gt = game_teams()
        hteam = np.array([gt[int(g)][0] for g in st["gid"]])
        ateam = np.array([gt[int(g)][1] for g in st["gid"]])
        dteam = np.empty(R, dtype=object)
        dteam[0::2], dteam[1::2] = ateam, hteam
        oteam = np.empty(R, dtype=object)
        oteam[0::2], oteam[1::2] = hteam, ateam
        dg = np.empty(R, np.int64)
        dg[0::2], dg[1::2] = st["ag"], st["hg"]
        ut, dti = np.unique(dteam.astype(str), return_inverse=True)
        _, oti = np.unique(oteam.astype(str), return_inverse=True)
        ug_, dgi = np.unique(dg, return_inverse=True)
        from scipy import sparse as sp_
        nt, ngk = len(ut), len(ug_)
        # drop one level of each block (intercept is in Xc)
        blocks = [sp_.csr_matrix(self.Xc)]
        for idx, n_ in ((dti, nt), (oti, nt), (dgi, ngk)):
            m = idx > 0
            blocks.append(sp_.csr_matrix((np.ones(int(m.sum())), (np.nonzero(m)[0], idx[m] - 1)),
                                         shape=(R, n_ - 1)))
        self.Xfe = sp_.hstack(blocks).tocsr()
        ug, self.ginv = np.unique(self.gid, return_inverse=True)
        self.ugid = ug
        self.R = R


# ---------------------------------------------------------------- Poisson ---
def pois_fit(X, y, off, beta0=None, iters=30):
    if beta0 is None:
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(y.sum() / np.exp(off).sum())
    else:
        beta = beta0.copy()
    for _ in range(iters):
        eta = off + X @ beta
        mu = np.exp(eta)
        z = eta - off + (y - mu) / mu
        H = (X * mu[:, None]).T @ X
        g = (X * mu[:, None]).T @ z
        nb = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), g)
        if np.max(np.abs(nb - beta)) < 1e-9:
            beta = nb
            break
        beta = nb
    return beta, off + X @ beta


def pois_fit_sparse(X, y, off, iters=30):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(iters):
        eta = off + X @ beta
        mu = np.exp(eta)
        z = eta - off + (y - mu) / mu
        Xm = X.multiply(mu[:, None]).tocsr()
        H = (Xm.T @ X).toarray()
        g = Xm.T @ z
        nb = np.linalg.solve(H + 1e-8 * np.eye(len(beta)), g)
        done = np.max(np.abs(nb - beta)) < 1e-8
        beta = nb
        if done:
            break
    return beta, off + X @ beta


def dev_rows(y, eta):
    mu = np.exp(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (t - (y - mu))


# ------------------------------------------------------------ rating maps ---
def fit_names(s, mode):
    specs = [sp for sp in F.wf_specs() if sp["season"] == s]
    if mode == "pre":
        specs = [sp for sp in specs if sp["kind"] == "pre"]
    return sorted([(sp["cutoff"], sp["name"]) for sp in specs])


_FITC = {}


def load_fit(name):
    if name not in _FITC:
        z = np.load(f"{PFX}fit_{name}.npz")
        _FITC[name] = {k: z[k] for k in z.files}
        if len(_FITC) > 12:
            _FITC.pop(next(iter(_FITC)))
    return _FITC[name]


_RAPMEL = {}


def rapmel(s):
    if s not in _RAPMEL:
        f = pd.read_csv(f"data/bt_nhl_rapmel_fitv_pre_{s}.csv")
        meta = json.load(open(f"data/bt_nhl_rapmel_fitm_pre_{s}.json"))
        _RAPMEL[s] = (f, meta)
    return _RAPMEL[s]


def lookup(pids_sorted, vals, q, default):
    i = np.searchsorted(pids_sorted, q)
    i = np.clip(i, 0, len(pids_sorted) - 1)
    ok = pids_sorted[i] == q
    return np.where(ok, vals[i], default), ok


def S_for(sea, key, mode):
    """Sum of the five defenders' ratings per row, walk-forward by date."""
    S = np.zeros(sea.R)
    known = np.zeros((sea.R, 5), bool)
    fits = fit_names(sea.s, mode)
    cut = [c for c, _ in fits]
    which = np.searchsorted(cut, sea.date, side="right") - 1
    assert (which >= 0).all(), "row before its season's first cutoff"
    for k, (c, name) in enumerate(fits):
        m = which == k
        if not m.any():
            continue
        assert sea.date[m].min() >= c
        if key.startswith("CUR_"):
            f, meta = rapmel(sea.s)
            assert meta["max_train_date"] < c
            col = "def" if key == "CUR_DEF" else "v"
            o = np.argsort(f.pid.to_numpy())
            p_s = f.pid.to_numpy()[o]
            v_s = f[col].to_numpy()[o]
            if col == "v":
                uF, uD = meta["v_repl"]["F"], meta["v_repl"]["D"]
            else:
                uF = uD = 0.0
        else:
            z = load_fit(name)
            p_s = z["pids"]
            v_s = z[key]
            ukey = "u" + key[1:]
            if ukey in z:
                uF, uD = z[ukey]
            else:
                uF = uD = 0.0
        q = sea.dfn[m]
        dflt = np.where(sea.isD[m], uD, uF)
        v, ok = lookup(p_s, v_s, q, dflt)
        S[m] = v.sum(1)
        known[m] = ok
    return S, known


# ---------------------------------------------------------------- predict ---
def unit_match(key, t):
    if key.startswith(("d_xg_", "d_g_", "d_gx", "r_g_", "r_xg_")) or key == "CUR_DEF":
        return t in ("g", "xg")
    if key.startswith(("d_c_", "r_c_")):
        return t == "c"
    return False


SEC_LAMS = {15000, 60000, 240000, 960000, 3840000}
SEC_K = {60000, 240000}


def secondary_key(key):
    """Keys also scored on the secondary targets (xG, attempts): a thinned grid."""
    if key.startswith("CUR_") or key.startswith("d_gx"):
        return True
    parts = key.split("_")
    if key.startswith("d_"):
        return int(parts[2]) in SEC_LAMS
    if key.startswith("r_"):
        return int(parts[2]) in SEC_K
    return False


def eval_season(s, mode, keys, targets):
    sea = Season(s)
    res = {}
    base = {}
    basefe = {}
    for t in targets:
        b0, eta0 = pois_fit(sea.Xc, sea.y[t], sea.off)
        d0 = dev_rows(sea.y[t], eta0)
        base[t] = (b0, d0)
        _, etaf = pois_fit_sparse(sea.Xfe, sea.y[t], sea.off)
        df0 = dev_rows(sea.y[t], etaf)
        basefe[t] = (etaf, df0)
        res[("BASE", t)] = dict(dev=float(d0.sum()), n=int(sea.R),
                               tot=float(sea.y[t].sum()),
                               game_dev=np.bincount(sea.ginv, weights=d0),
                               dev_fe=float(df0.sum()),
                               game_dev_fe=np.bincount(sea.ginv, weights=df0))
    for key in keys:
        tlist = [t for t in targets if t == "g" or secondary_key(key)]
        if not tlist:
            continue
        S, known = S_for(sea, key, mode)
        sdS = float(S.std())
        Z = (S - S.mean()) / (sdS if sdS > 0 else 1.0)
        X = np.column_stack([sea.Xc, Z])
        for t in tlist:
            b0, d0 = base[t]
            beta, eta = pois_fit(X, sea.y[t], sea.off, np.append(b0, 0.0))
            d1 = dev_rows(sea.y[t], eta)
            rate = sea.y[t].sum() * 3600.0 / sea.dur.sum()
            b_per_unit = beta[-1] / (sdS if sdS > 0 else 1.0)
            r = dict(dev=float(d1.sum()), gain=float(d0.sum() - d1.sum()),
                     b_cal=float(-b_per_unit * rate), sdS=sdS,
                     known_share=float(known.mean()),
                     game_dev=np.bincount(sea.ginv, weights=d1))
            # WITHIN spec: team-season / opponent-season / goalie-season effects held
            # at the baseline fit; the arm gets a free intercept shift and slope
            etaf, df0 = basefe[t]
            X2 = np.column_stack([np.ones(sea.R), Z])
            _, eta2 = pois_fit(X2, sea.y[t], etaf, np.zeros(2))
            df1 = dev_rows(sea.y[t], eta2)
            r["gain_fe"] = float(df0.sum() - df1.sum())
            r["game_dev_fe"] = np.bincount(sea.ginv, weights=df1)
            if unit_match(key, t) and t == "g":
                # face value: S literally, in per-60 units of the target; no free slope
                off2 = sea.off + np.log(np.clip(1.0 - S / rate, 0.2, 5.0))
                _, eta2 = pois_fit(sea.Xc, sea.y[t], off2, b0)
                r["face_gain"] = float(d0.sum() - dev_rows(sea.y[t], eta2).sum())
            res[(key, t)] = r
    return sea, res


def all_keys(mode):
    fam = families()
    keys = [k for v in fam.values() for k in v]
    if mode == "pre":
        keys += ["CUR_DEF", "CUR_V"]
    return keys


def _eval_worker(s, mode, targets=("g", "xg", "c")):
    import pickle
    t0 = time.time()
    _, res = eval_season(s, mode, all_keys(mode), targets)
    pickle.dump(res, open(f"{PFX}ev_{mode}_{s}.pkl", "wb"))
    print(f"    {mode} season {s} done {time.time()-t0:.0f}s", flush=True)
    return s


def cmd_predict_seasons(mode, seasons, procs):
    from multiprocessing import Pool
    with Pool(procs) as pool:
        pool.starmap(_eval_worker, [(s, mode) for s in seasons])


def nested_pick(G, cfgs, seasons):
    """G[cfg][season] gain; nested LOSO choice per held-out season."""
    pick = {}
    for s in seasons:
        others = [o for o in seasons if o != s]
        pick[s] = max(cfgs, key=lambda c: sum(G[c][o] for o in others))
    return pick


def boot_ci(diff_by_game, B=4000, seed=11):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff_by_game)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    tot = d[idx].sum(1)
    lo, hi = np.percentile(tot, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_predict(mode="pre", targets=("g", "xg", "c")):
    t0 = time.time()
    fam = families()
    keys = [k for v in fam.values() for k in v]
    if mode == "pre":
        keys += ["CUR_DEF", "CUR_V"]
        fam["CUR_DEF"] = ["CUR_DEF"]
        fam["CUR_V"] = ["CUR_V"]
    allres = {}
    gdev = {}
    import pickle
    outs = [pickle.load(open(f"{PFX}ev_{mode}_{s}.pkl", "rb")) for s in EVAL]
    for s, res in zip(EVAL, outs):
        allres[s] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("game_dev")}
                     for k, v in res.items()}
        gdev[s] = {k: {"ctx": v["game_dev"], "fe": v.get("game_dev_fe")}
                   for k, v in res.items()}
    print(f"  {mode}: all seasons evaluated {time.time()-t0:.0f}s", flush=True)
    out = {"mode": mode, "seasons": EVAL, "targets": list(targets),
           "specs": {"ctx": "context only (intercept, home, score state, zone start); "
                            "b and context refit on the evaluation season",
                     "fe": "WITHIN: + defending team-season, attacking team-season and "
                           "defending goalie-season fixed effects (baseline fit held), "
                           "arm gets intercept shift + slope"},
           "families": {}, "families_fe": {}, "h2h": {}, "h2h_fe": {}}
    for spec, gkey, bkey, fkey, hkey in (("ctx", "gain", "dev", "families", "h2h"),
                                         ("fe", "gain_fe", "dev_fe", "families_fe", "h2h_fe")):
        for t in targets:
            tot_goals = sum(allres[s][("BASE", t)]["tot"] for s in EVAL)
            base_dev = sum(allres[s][("BASE", t)][bkey] for s in EVAL)
            fam_game = {}
            for fname, cfgs in fam.items():
                cfgs = [c for c in cfgs if (c, t) in allres[EVAL[0]]]
                G = {c: {s: allres[s][(c, t)][gkey] for s in EVAL} for c in cfgs}
                pick = nested_pick(G, cfgs, EVAL)
                nested = sum(G[pick[s]][s] for s in EVAL)
                best = max(cfgs, key=lambda c: sum(G[c].values()))
                gd = np.concatenate([gdev[s][(pick[s], t)][spec] for s in EVAL])
                gb = np.concatenate([gdev[s][("BASE", t)][spec] for s in EVAL])
                fam_game[fname] = gd
                lo, hi = boot_ci(gb - gd)
                rec = {
                    "nested_gain": round(nested, 2),
                    "nested_gain_ci": [round(lo, 2), round(hi, 2)],
                    "gain_per_1000_events": round(1000 * nested / tot_goals, 3),
                    "rel_deviance_drop": nested / base_dev,
                    "picks": {str(s): pick[s] for s in EVAL},
                    "per_season_nested": {str(s): round(G[pick[s]][s], 2) for s in EVAL},
                    "in_sample_best": best,
                    "in_sample_best_gain": round(sum(G[best].values()), 2),
                    "all_cfg_total_gain": {c: round(sum(G[c].values()), 2) for c in cfgs},
                }
                if spec == "ctx":
                    rec["b_cal_of_picks"] = {str(s): round(allres[s][(pick[s], t)]["b_cal"], 3)
                                             for s in EVAL}
                    rec["known_share"] = round(float(np.mean(
                        [allres[s][(pick[s], t)]["known_share"] for s in EVAL])), 4)
                    if unit_match(cfgs[0], t) and t == "g":
                        rec["face_gain_of_picks"] = round(sum(
                            allres[s][(pick[s], t)]["face_gain"] for s in EVAL), 2)
                        rec["face_best"] = max(
                            ((c, round(sum(allres[s][(c, t)]["face_gain"] for s in EVAL), 2))
                             for c in cfgs), key=lambda x: x[1])
                out[fkey].setdefault(t, {})[fname] = rec
            h2h = {}
            names = list(fam)
            for i, a_ in enumerate(names):
                for b_ in names[i + 1:]:
                    diff = fam_game[b_] - fam_game[a_]      # positive = a better
                    lo, hi = boot_ci(diff)
                    h2h[f"{a_}_vs_{b_}"] = {"a_minus_b_gain": round(float(diff.sum()), 2),
                                           "ci": [round(lo, 2), round(hi, 2)],
                                           "sig": "a" if lo > 0 else ("b" if hi < 0 else "n.s.")}
            out[hkey][t] = h2h
            out.setdefault("totals", {}).setdefault(spec, {})[t] = {
                "events": tot_goals, "base_deviance": base_dev,
                "n_games": int(sum(len(gdev[s][("BASE", t)]["ctx"]) for s in EVAL))}
    json.dump(out, open(f"{PFX}eval_predict_{mode}.json", "w"), indent=1)
    print(f"wrote {PFX}eval_predict_{mode}.json ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------- players ---
def picks_from_predict(mode="pre", target="g"):
    d = json.load(open(f"{PFX}eval_predict_{mode}.json"))
    return {f: {int(k): v for k, v in r["picks"].items()}
            for f, r in d["families"][target].items()}


def player_future(s):
    """Per player: 5v5 seconds and on-ice GA / xGA / CA per 60 in season s, raw and
    relative to the TOI-weighted team-season rate of the rows he played."""
    sea = Season(s)
    gt = game_teams()
    team = np.array([gt[int(g)][0] if h else gt[int(g)][1]
                     for g, h in zip(sea.gid, sea.def_home)])
    ut, tinv = np.unique(team, return_inverse=True)
    flat = sea.dfn.ravel()
    up, pinv = np.unique(flat, return_inverse=True)
    dur5 = np.repeat(sea.dur, 5)
    T = np.bincount(pinv, weights=dur5)
    out = {"pid": up, "T": T}
    for t in TGT_ALL:
        team_rate = (np.bincount(tinv, weights=sea.y[t]) * 3600.0
                     / np.bincount(tinv, weights=sea.dur))
        ag = np.bincount(pinv, weights=np.repeat(sea.y[t], 5))
        expected = np.bincount(pinv, weights=np.repeat(sea.dur * team_rate[tinv], 5)) / T
        out[f"on_{t}"] = ag * 3600.0 / T
        out[f"rel_{t}"] = out[f"on_{t}"] - expected
    out["isD"] = np.bincount(pinv, weights=sea.isD.astype(float).ravel() * dur5) / T > 0.5
    return pd.DataFrame(out)


def prior_rating(s, key):
    fits = fit_names(s, "pre")
    c, name = fits[0]
    if key.startswith("CUR_"):
        f, meta = rapmel(s)
        col = "def" if key == "CUR_DEF" else "v"
        return pd.DataFrame({"pid": f.pid.to_numpy(), "d": f[col].to_numpy(),
                             "Tprior": f["T"].to_numpy()})
    z = load_fit(name)
    return pd.DataFrame({"pid": z["pids"], "d": z[key], "Tprior": z["Traw"]})


def corr_ci(x, y, B=2000, seed=5):
    rng = np.random.default_rng(seed)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return r, float(lo), float(hi)


def cmd_players(min_future=30000.0, min_prior=30000.0):
    t0 = time.time()
    picks = picks_from_predict("pre", "g")
    fams = [f for f in ("XGA", "GA", "GAX", "CA", "RAWGA", "RAWXGA", "CUR_DEF", "CUR_V")
            if f in picks]
    frames = []
    for s in EVAL:
        fut = player_future(s)
        for f in fams:
            pr = prior_rating(s, picks[f][s])
            j = fut.merge(pr, on="pid")
            j = j[(j["T"] >= min_future) & (j.Tprior >= min_prior)].copy()
            j["season"] = s
            j["fam"] = f
            frames.append(j)
        print(f"  players {s} {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    # movers: primary team (most games dressed) in season s differs from season s-1
    ro = load_rosters()
    ro = ro[(ro.gtype == 2) & (ro.pos != "G")]
    prim = (ro.groupby(["pid", "season"]).team.agg(lambda x: x.value_counts().index[0])
            .to_dict())
    prev_season = {s_: F.SEASONS[F.SEASONS.index(s_) - 1] for s_ in EVAL}
    df["mover"] = [(prim.get((p_, s_)) is not None and prim.get((p_, prev_season[s_])) is not None
                    and prim.get((p_, s_)) != prim.get((p_, prev_season[s_])))
                   for p_, s_ in zip(df.pid, df.season)]
    df.to_csv(f"{PFX}eval_players.csv", index=False)
    out = {"min_future_5v5_seconds": min_future, "min_prior_window_seconds": min_prior,
           "sign": "r > 0 means a better prior defensive rating preceded FEWER against",
           "families": {}}
    key = df.groupby(["season", "pid"]).fam.nunique()
    common = key[key == len(fams)].index
    dfc = df.set_index(["season", "pid"]).loc[common].reset_index()
    for f in fams:
        q = dfc[dfc.fam == f]
        res = {"n_player_seasons": int(len(q))}
        for col in ("on_g", "rel_g", "on_xg", "rel_xg", "on_c", "rel_c"):
            r, lo, hi = corr_ci(q.d.to_numpy(), -q[col].to_numpy())
            res[col] = [round(r, 4), round(lo, 4), round(hi, 4)]
        mv = q[q.mover]
        res["movers_n"] = int(len(mv))
        for col in ("on_g", "rel_g", "on_xg", "rel_xg"):
            r, lo, hi = corr_ci(mv.d.to_numpy(), -mv[col].to_numpy())
            res[f"movers_{col}"] = [round(r, 4), round(lo, 4), round(hi, 4)]
        for pos, m in (("F", ~q.isD), ("D", q.isD)):
            res[f"on_g_{pos}"] = round(float(np.corrcoef(q.d[m], -q.on_g[m])[0, 1]), 4)
            res[f"rel_g_{pos}"] = round(float(np.corrcoef(q.d[m], -q.rel_g[m])[0, 1]), 4)
            res[f"n_{pos}"] = int(m.sum())
        out["families"][f] = res
    base = dfc[dfc.fam == "XGA"].sort_values(["season", "pid"])
    rng = np.random.default_rng(3)
    for f in fams:
        if f == "XGA":
            continue
        q = dfc[dfc.fam == f].sort_values(["season", "pid"])
        assert (q.pid.to_numpy() == base.pid.to_numpy()).all()
        for col in ("on_g", "rel_g"):
            x1, x2, yy = base.d.to_numpy(), q.d.to_numpy(), -q[col].to_numpy()
            n = len(yy)
            dd = []
            for _ in range(2000):
                i = rng.integers(0, n, n)
                dd.append(np.corrcoef(x2[i], yy[i])[0, 1] - np.corrcoef(x1[i], yy[i])[0, 1])
            lo, hi = np.percentile(dd, [2.5, 97.5])
            out["families"][f][f"{col}_minus_XGA"] = [
                round(float(np.corrcoef(x2, yy)[0, 1] - np.corrcoef(x1, yy)[0, 1]), 4),
                round(float(lo), 4), round(float(hi), 4)]
    json.dump(out, open(f"{PFX}eval_players.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# -------------------------------------------------------------------- rel ---
def cmd_rel():
    """Year-over-year and split-half reliability of single-season fits."""
    keys = ([f"d_xg_{l}_0" for l in LAMS] + [f"d_g_{l}_0" for l in LAMS]
            + [f"d_c_{l}_0" for l in LAMS] + ["d_gx60000_1920000_0"]
            + [f"r_{t}_30000_0" for t in ("g", "xg", "c")])
    out = {}

    def pairs(a, b, thr_a, thr_b):
        za, zb = load_fit(a), load_fit(b)
        pa = pd.DataFrame({"pid": za["pids"], "Ta": za["Traw"], "isD": za["isD"]})
        pb = pd.DataFrame({"pid": zb["pids"], "Tb": zb["Traw"]})
        for k in keys:
            pa[k] = za[k]
            pb[k] = zb[k]
        j = pa.merge(pb, on="pid", suffixes=("_a", "_b"))
        return j[(j.Ta >= thr_a) & (j.Tb >= thr_b)]

    def summarize(frames, label):
        j = pd.concat(frames, ignore_index=True)
        res = {"n": int(len(j)), "n_F": int((~j.isD).sum()), "n_D": int(j.isD.sum())}
        for k in keys:
            a, b = j[k + "_a"].to_numpy(), j[k + "_b"].to_numpy()
            r = float(np.corrcoef(a, b)[0, 1])
            rF = float(np.corrcoef(a[~j.isD], b[~j.isD])[0, 1])
            rD = float(np.corrcoef(a[j.isD], b[j.isD])[0, 1])
            rs = float(pd.Series(a).rank().corr(pd.Series(b).rank()))
            res[k] = {"r": round(r, 4), "r_F": round(rF, 4), "r_D": round(rD, 4),
                      "spearman": round(rs, 4)}
        out[label] = res

    summarize([pairs(f"one_{a}", f"one_{b}", 30000.0, 30000.0)
               for a, b in zip(F.SEASONS[:-1], F.SEASONS[1:])], "yoy")
    summarize([pairs(f"half0_{s}", f"half1_{s}", 15000.0, 15000.0) for s in F.SEASONS],
              "split_half")
    summarize([pairs("h30_2015_17", "h31_2015_17", 30000.0, 30000.0)], "split_half_3season")
    out["thresholds_5v5_seconds"] = {"yoy": "30000 in both seasons",
                                     "split_half": "15000 in each half",
                                     "split_half_3season": "30000 in each half (2014-17)"}
    json.dump(out, open(f"{PFX}eval_rel.json", "w"), indent=1)
    for lab in ("yoy", "split_half", "split_half_3season"):
        r = out[lab]
        print(lab, "n", r["n"], "F", r["n_F"], "D", r["n_D"])
        for k in keys:
            print(f"   {k:24s} r={r[k]['r']:+.3f}  F {r[k]['r_F']:+.3f}  D {r[k]['r_D']:+.3f}"
                  f"  rho={r[k]['spearman']:+.3f}")


# ------------------------------------------------------------------- face ---
def cmd_face(fit="pre_20172018", min_T=180000.0, top=10):
    picks = picks_from_predict("pre", "g")
    pl = pd.read_csv("data/pv_nhl_players.csv").set_index("pid")

    def nm(p):
        return f"{pl.loc[p, 'first']} {pl.loc[p, 'last']}" if p in pl.index else str(p)

    z = load_fit(fit)
    out = {"fit": fit, "min_window_5v5_seconds": min_T, "lists": {}}
    for fam in ("XGA", "GA", "GAX", "CA"):
        key = picks[fam][20172018]
        d = z[key]
        m = z["Traw"] >= min_T
        lists = {}
        for pos, pm in (("F", ~z["isD"]), ("D", z["isD"])):
            mm = m & pm
            o = np.argsort(-d[mm])
            P, D = z["pids"][mm], d[mm]
            lists[pos] = {"top": [(nm(int(P[i])), round(float(D[i]), 3)) for i in o[:top]],
                          "bottom": [(nm(int(P[i])), round(float(D[i]), 3))
                                     for i in o[::-1][:top]],
                          "n": int(mm.sum())}
        out["lists"][fam] = {"key": key, **lists}
    gk = "g_" + picks["GA"][20172018][2:]
    g = z[gk]
    o = np.argsort(g)
    out["goalie_GA_column"] = {"key": gk,
                               "best": [(nm(int(z["gpids"][i])), round(float(g[i]), 3))
                                        for i in o[:8]],
                               "worst": [(nm(int(z["gpids"][i])), round(float(g[i]), 3))
                                         for i in o[::-1][:8]]}
    json.dump(out, open(f"{PFX}eval_face.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ combo ---
def cmd_combo(mode="pre"):
    """Incremental value on FUTURE GOALS AGAINST: does the GA (or CA / GAX) rating
    add anything once the xGA rating is in? Nested picks from predict <mode>."""
    picks = picks_from_predict(mode, "g")
    sets = {"XGA": ["XGA"], "XGA+GA": ["XGA", "GA"], "XGA+GAX": ["XGA", "GAX"],
            "XGA+CA": ["XGA", "CA"], "XGA+CA+GA": ["XGA", "CA", "GA"], "CA": ["CA"],
            "GA": ["GA"]}
    out = {"mode": mode, "target": "g", "sets": {}}
    games = {k: {"ctx": [], "fe": []} for k in list(sets) + ["BASE"]}
    for s in EVAL:
        sea = Season(s)
        y = sea.y["g"]
        b0, eta0 = pois_fit(sea.Xc, y, sea.off)
        _, etaf = pois_fit_sparse(sea.Xfe, y, sea.off)
        games["BASE"]["ctx"].append(np.bincount(sea.ginv, weights=dev_rows(y, eta0)))
        games["BASE"]["fe"].append(np.bincount(sea.ginv, weights=dev_rows(y, etaf)))
        Z = {}
        for fam in ("XGA", "GA", "GAX", "CA"):
            S, _ = S_for(sea, picks[fam][s], mode)
            Z[fam] = (S - S.mean()) / S.std()
        for name, fams in sets.items():
            Zm = np.column_stack([Z[f] for f in fams])
            _, eta = pois_fit(np.column_stack([sea.Xc, Zm]), y, sea.off,
                              np.append(b0, np.zeros(len(fams))))
            games[name]["ctx"].append(np.bincount(sea.ginv, weights=dev_rows(y, eta)))
            _, eta2 = pois_fit(np.column_stack([np.ones(sea.R), Zm]), y, etaf,
                               np.zeros(1 + len(fams)))
            games[name]["fe"].append(np.bincount(sea.ginv, weights=dev_rows(y, eta2)))
        print(f"  combo {s}", flush=True)
    for spec in ("ctx", "fe"):
        gb = np.concatenate(games["BASE"][spec])
        gx = np.concatenate(games["XGA"][spec])
        for name in sets:
            gd = np.concatenate(games[name][spec])
            lo, hi = boot_ci(gb - gd)
            rec = {"gain_vs_base": round(float((gb - gd).sum()), 2), "ci": [round(lo, 2), round(hi, 2)]}
            if name != "XGA":
                lo2, hi2 = boot_ci(gx - gd)
                rec["gain_vs_XGA_alone"] = round(float((gx - gd).sum()), 2)
                rec["ci_vs_XGA_alone"] = [round(lo2, 2), round(hi2, 2)]
                rec["n_extra_params_per_season"] = len(sets[name]) - 1 if "XGA" in sets[name]                     else 0
            out["sets"].setdefault(spec, {})[name] = rec
    out["note"] = ("each extra rating adds one fitted slope per season (7 in total): a gain "
                   "below ~1 deviance unit per extra parameter (chi2_1 mean) is noise")
    json.dump(out, open(f"{PFX}eval_combo_{mode}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))



if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd == "predict":
        cmd_predict(sys.argv[2] if len(sys.argv) > 2 else "pre")
    elif cmd == "predict_seasons":
        cmd_predict_seasons(sys.argv[2], [int(x) for x in sys.argv[4:]], int(sys.argv[3]))
    elif cmd == "players":
        cmd_players()
    elif cmd == "rel":
        cmd_rel()
    elif cmd == "face":
        cmd_face()
    elif cmd == "combo":
        cmd_combo(sys.argv[2] if len(sys.argv) > 2 else "pre")
    else:
        raise SystemExit("usage: predict [pre|month] | players | rel | face")
