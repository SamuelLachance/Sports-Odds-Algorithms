"""MLB DEPTH campaign — re-examine the COMPONENTS of the shipped MLB model.

DEV SCREEN ONLY. Seasons 2002-2015 are the only seasons ever scored; the game
loader is hard-capped at year_max=2015 so no TEST-era row is even parsed into
this process. Every scoring path asserts it again. The orchestrator spends the
one TEST look, never this file.

Market-blind: the only inputs are Retrosheet game logs and parsed events.

What is being optimised
-----------------------
The shipped "full" tier is a logistic blend

    logit P(home) = b0 + b1*logit(fip_elo_p) + b2*ts_z + b3*bp_z
                       + b4*pw_z + b5*si_z + b6*br_z

so the single objective used by every axis is the pooled out-of-fold log loss of
that tier under 5 season-grouped CV folds inside DEV. Changing a rating engine's
hyperparameters changes one of the six columns; the blend is refit each time, so
every axis is measured where it actually matters -- through the blend.

Axes (subcommands)
  --cache     build data/mlb_depth_cache/*  (games, per-side feature walks, PA arrays)
  --baseline  shipped hyperparameters, the number every axis is compared to
  --axis-a    FIP/xFIP-Elo hyperparameters, JOINTLY (random search + coordinate descent)
  --axis-b    per-PA TrueSkill internals (beta, tau, draw margin, read-time priors)
  --axis-c    blend model class (logistic C, HistGB, CatBoost, 2-model stack)
  --axis-d    functional form (splines / bins on each continuous feature)
  --axis-e    interactions (a short, mechanistically justified list)
  --axis-f    regularisation (blend C, per-feature shrinkage of the rating priors)
  --report    merge every axis result into data/mlb_depth_dev.json

Honesty protocol
  * every search reports BOTH the raw best cell and a NESTED estimate: for each
    outer fold the winning cell is re-chosen using only the other four folds and
    scored on the held-out one. raw - nested = the search optimism.
  * fold-to-fold noise is reported as the std of the five per-fold deltas; every
    gain is quoted in units of that std.
  * negative results are reported with the same precision as positive ones.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import numpy as np  # noqa: E402

CACHE = "data/mlb_depth_cache"
REPORT = "data/mlb_depth_dev.json"

BURN_LO = 2000          # ratings burn in from 2000; nothing before DEV_LO is scored
DEV_LO, DEV_HI = 2002, 2015
N_FOLDS = 5
EPS = 1e-15

# shipped values (mlbwp/train.py PARAMS and mlbwp/artifacts/blend.json ts_params)
SHIP_ELO = dict(k_team=3.6487, hfa=26.5952, team_regress=0.5398,
                beta=37.1196, decay=0.9531, prior_ip=74.3335, season_decay=0.0087)
SHIP_TS = dict(beta_rel=0.5, tau_rel=0.01, eps_rel=0.0, shrink_n=0.0, sig_k=0.0)

BP_PRIOR_OUTS = 200.0
PWR_PRIOR_PA = 150.0
SIERA_PRIOR = 120.0
BR_RV = (0.20, -0.41, 0.19, -0.41, -0.25)
BR_PRIOR_PA = 200.0

FEATS = ("ts", "bp", "pw", "si", "br")   # blend column order after lf


# ===================================================================== utils
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def llvec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logit(X, y, l2=1e-8, iters=40):
    """Unpenalised (tiny ridge for conditioning) Newton-IRLS logistic fit.

    Matches sklearn LogisticRegression(C=1e6) to <1e-6 in log loss but is ~50x
    faster, which is what makes the joint searches affordable. Intercept added."""
    n, k = X.shape
    A = np.empty((n, k + 1))
    A[:, 0] = 1.0
    A[:, 1:] = X
    w = np.zeros(k + 1)
    for _ in range(iters):
        z = A @ w
        p = _sig(z)
        g = A.T @ (p - y) + l2 * w
        s = np.clip(p * (1 - p), 1e-10, None)
        H = (A * s[:, None]).T @ A
        H[np.diag_indices_from(H)] += l2
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def pred_logit(w, X):
    return _sig(w[0] + X @ w[1:])


def folds_for(seasons_arr):
    dev = sorted(set(int(s) for s in seasons_arr))
    assert dev and min(dev) >= DEV_LO and max(dev) <= DEV_HI, dev
    return [set(int(x) for x in g) for g in np.array_split(np.array(dev), N_FOLDS)]


def oof_ll(cols, y, seas, groups, l2=1e-8, standardize=True):
    """Pooled out-of-fold per-game log loss of logistic(cols).

    cols[0] is the FIP-Elo logit and enters raw; the rest are z-scored on the
    TRAINING folds only (no fold ever sees its own moments)."""
    # ONE-LOOK PROTOCOL: nothing outside DEV may ever be scored here.
    assert seas.min() >= DEV_LO and seas.max() <= DEV_HI, \
        f"TEST-era season reached the scorer: {seas.min()}-{seas.max()}"
    X = np.column_stack(cols)
    out = np.empty(len(y))
    for grp in groups:
        te = np.isin(seas, list(grp))
        tr = ~te
        Z = X.copy()
        if standardize and X.shape[1] > 1:
            mu = X[tr, 1:].mean(0)
            sd = X[tr, 1:].std(0)
            sd[sd == 0] = 1.0
            Z[:, 1:] = (X[:, 1:] - mu) / sd
        w = fit_logit(Z[tr], y[tr], l2=l2)
        out[te] = llvec(y[te], pred_logit(w, Z[te]))
    return out


def fold_stats(d, seas, groups):
    """Per-fold mean of a per-game delta vector + the fold-to-fold noise."""
    fd = [float(d[np.isin(seas, list(g))].mean()) for g in groups]
    m = float(np.mean(fd))
    sd = float(np.std(fd, ddof=1))
    return fd, m, sd


def boot_ci(d, seed=7, B=10000):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(name, base_ll, var_ll, seas, groups):
    d = base_ll - var_ll                       # positive = variant better
    fd, m, sd = fold_stats(d, seas, groups)
    lo, hi = boot_ci(d)
    return {
        "name": name,
        "base_ll": round(float(base_ll.mean()), 6),
        "ll": round(float(var_ll.mean()), 6),
        "delta": round(float(d.mean()), 6),
        "fold_deltas": [round(x, 6) for x in fd],
        "fold_sd": round(sd, 6),
        "n_sd": round(m / sd, 2) if sd > 0 else None,
        "ci": [round(lo, 6), round(hi, 6)],
        "sig": "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s."),
    }


def pr(s):
    print(f"  {s['name']:<34} ll {s['ll']:.5f}  d {s['delta']:+.6f} "
          f"[{s['ci'][0]:+.6f},{s['ci'][1]:+.6f}] {s['sig']:<9} "
          f"fold_sd {s['fold_sd']:.6f}  {s['n_sd']} sd", flush=True)


# =============================================================== cache build
def _load_lineups():
    lu = defaultdict(lambda: {0: [], 1: []})
    with open("data/retro_events/lineups.csv", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if r[0] == "game_id":
                continue
            lu[r[0]][int(r[1])].append(sys.intern(r[3]))
    return lu


def _load_starters():
    out = {}
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        if int(path[-8:-4]) > DEV_HI:
            continue
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            out[r[6] + r[0] + r[1]] = (r[103].strip(), r[101].strip())
    return out


def _reliever_aggs(keep):
    per_game = defaultdict(lambda: [0.0, 0])
    lg_num = lg_outs = 0
    with open("data/retro_events/appearances.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["is_starter"] == "1" or r["game_id"] not in keep:
                continue
            outs = int(r["outs"])
            num = 13 * int(r["HR"]) + 3 * (int(r["BB"]) + int(r["HBP"])) - 2 * int(r["SO"])
            a = per_game[(r["game_id"], r["team"])]
            a[0] += num
            a[1] += outs
            lg_num += num
            lg_outs += outs
    return per_game, (lg_num / lg_outs * 3.0 if lg_outs else 1.0)


def _siera_inputs(keep):
    ss, bbl = {}, {}
    with open("data/retro_events/appearances.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["is_starter"] == "1" and r["game_id"] in keep:
                ss[(r["game_id"], r["pitcher"])] = (int(r["SO"]), int(r["BB"]), int(r["bf"]))
    with open("data/retro_events/battedball.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["game_id"] in keep:
                bbl[(r["game_id"], r["pitcher"])] = (int(r["gb"]), int(r["fb"]), int(r["pu"]))
    return ss, bbl


def _load_power(keep):
    by_game = defaultdict(list)
    tp = th = tb = 0
    with open("data/retro_events/power.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["game_id"] not in keep:
                continue
            pa, h, t = int(r["pa"]), int(r["h"]), int(r["tb"])
            by_game[r["game_id"]].append((sys.intern(r["batter"]), pa, h, t))
            tp += pa; th += h; tb += t
    return by_game, (tb - th) / tp


def _load_baserun(keep):
    by_game = defaultdict(list)
    with open("data/retro_events/baserun.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["game_id"] not in keep:
                continue
            by_game[r["game_id"]].append(
                (sys.intern(r["player"]), int(r["pa"]), int(r["sb"]), int(r["cs"]),
                 int(r["xb"]), int(r["oob"]), int(r["gidp"])))
    return by_game


def build_cache():
    from mlbwp.ingest import load_games, league_fip_core
    from mlbwp.siera import SieraRater

    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    games = load_games(year_max=DEV_HI)
    assert max(g["season"] for g in games) <= DEV_HI, "TEST-era game leaked into the cache"
    lg_fip = league_fip_core(games)
    keep = {g["game_id"] for g in games}
    print(f"games {len(games):,} {games[0]['season']}-{games[-1]['season']} "
          f"lg_fip {lg_fip:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    lu = _load_lineups()
    per_game, bp_lg_core = _reliever_aggs(keep)
    power, lg_iso = _load_power(keep)
    br = _load_baserun(keep)
    ss, bbl = _siera_inputs(keep)
    print(f"aux loaded ({time.time()-t0:.0f}s)", flush=True)

    tS = tB = tPA = tG = tF = tP = 0
    for g in games:
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((g["game_id"], pid), (0, 0, 0))
            gb, fb, pu = bbl.get((g["game_id"], pid), (0, 0, 0))
            tS += so; tB += bb; tPA += bf; tG += gb; tF += fb; tP += pu
    siera_lg = dict(so=tS / tPA, bb=tB / tPA, gb=tG / tPA, fb=tF / tPA, pu=tP / tPA)

    # ---- one walk over the games: bullpen / power / baserun / SIERA, per side ----
    sr = SieraRater(decay=SHIP_ELO["decay"], prior=SIERA_PRIOR, lg=siera_lg)
    bp_cum = defaultdict(lambda: [0.0, 0])
    bp_prior = BP_PRIOR_OUTS * bp_lg_core / 3.0
    pw_car = defaultdict(lambda: [0, 0, 0])
    pw_prior = PWR_PRIOR_PA * lg_iso
    br_car = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    rows = []
    # AS-OF raw state, so that every empirical-Bayes prior (axis F) and every
    # read-time transform can be re-evaluated without another walk.
    st_bp, st_pw, st_br, st_si = [], [], [], []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            bp_cum.clear()
        prev = g["season"]
        gid = g["game_id"]

        def bpfip(team):
            c = bp_cum[(g["season"], team)]
            return (c[0] + bp_prior) / (c[1] + BP_PRIOR_OUTS) * 3.0

        def iso(b):
            c = pw_car[b]
            return (c[2] - c[1] + pw_prior) / (c[0] + PWR_PRIOR_PA)

        def brate(b):
            c = br_car[b]
            return sum(BR_RV[k] * c[1 + k] for k in range(5)) / (c[0] + BR_PRIOR_PA)

        def lineup(side, fn):
            bs = lu.get(gid, {}).get(side, [])
            return (sum(fn(b) for b in bs) / len(bs)) if bs else None

        h_pw, a_pw = lineup(1, iso), lineup(0, iso)
        h_br, a_br = lineup(1, brate), lineup(0, brate)
        rows.append((gid, g["date"], g["season"], g["y"],
                     bpfip(g["home"]), bpfip(g["away"]),
                     sr.value(g["home_sp"]), sr.value(g["away_sp"]),
                     h_pw, a_pw, h_br, a_br))
        ch, ca = bp_cum[(g["season"], g["home"])], bp_cum[(g["season"], g["away"])]
        st_bp.append([ch[0], ch[1], ca[0], ca[1]])
        ln = lu.get(gid, {})
        pad3 = [[0, 0, 0]] * 9
        pad6 = [[0] * 6] * 9
        st_pw.append([[list(pw_car[b]) for b in ln.get(s, [])] or pad3 for s in (1, 0)])
        st_br.append([[list(br_car[b]) for b in ln.get(s, [])] or pad6 for s in (1, 0)])
        st_si.append([list(sr.E[g["home_sp"]]), list(sr.E[g["away_sp"]])])

        # ---- state updates strictly AFTER reading ----
        for team in (g["home"], g["away"]):
            num, outs = per_game.get((gid, team), (0.0, 0))
            c = bp_cum[(g["season"], team)]
            c[0] += num
            c[1] += outs
        for b, pa, h, tb in power.get(gid, []):
            c = pw_car[b]; c[0] += pa; c[1] += h; c[2] += tb
        for row in br.get(gid, []):
            c = br_car[row[0]]
            for k in range(6):
                c[k] += row[1 + k]
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)
    print(f"feature walk done ({time.time()-t0:.0f}s)", flush=True)

    ki = [i for i, r in enumerate(rows) if all(x is not None for x in r[8:])]
    keep_rows = [rows[i] for i in ki]
    gid_order = [r[0] for r in keep_rows]
    np.savez_compressed(
        f"{CACHE}/state.npz",
        bp=np.array([st_bp[i] for i in ki], np.float64),
        pw=np.array([st_pw[i] for i in ki], np.int32),
        br=np.array([st_br[i] for i in ki], np.int32),
        si=np.array([st_si[i] for i in ki], np.float64),
        const=np.array([bp_lg_core, lg_iso, siera_lg["so"], siera_lg["bb"],
                        siera_lg["gb"], siera_lg["fb"], siera_lg["pu"]]))
    np.savez_compressed(
        f"{CACHE}/feat.npz",
        game_id=np.array(gid_order),
        season=np.array([r[2] for r in keep_rows], np.int16),
        y=np.array([r[3] for r in keep_rows], np.float64),
        bp_home=np.array([r[4] for r in keep_rows]), bp_away=np.array([r[5] for r in keep_rows]),
        si_home=np.array([r[6] for r in keep_rows]), si_away=np.array([r[7] for r in keep_rows]),
        pw_home=np.array([r[8] for r in keep_rows]), pw_away=np.array([r[9] for r in keep_rows]),
        br_home=np.array([r[10] for r in keep_rows]), br_away=np.array([r[11] for r in keep_rows]),
    )
    with open(f"{CACHE}/games.pkl", "wb") as fh:
        pickle.dump({"games": games, "lg_fip": lg_fip}, fh, protocol=4)

    # ---- PA arrays for the TrueSkill re-walks ----
    starters = _load_starters()
    idx = {}

    def ix(p):
        v = idx.get(p)
        if v is None:
            v = idx[p] = len(idx)
        return v

    pa_g, pa_b, pa_p, pa_o = [], [], [], []
    gpa = defaultdict(list)
    gdate = {}
    with open("data/retro_events/pa.csv", newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if int(r[2]) > DEV_HI:
                continue
            gpa[r[0]].append((r[4], r[5], r[6] == "1"))
            gdate[r[0]] = r[1]
    order = sorted(gpa, key=lambda g: (gdate[g], g))
    starts, lin_h, lin_a, sp_h, sp_a, gids = [], [], [], [], [], []
    for gid in order:
        ln = lu.get(gid)
        sp = starters.get(gid)
        ok = bool(ln and sp and ln[1] and ln[0] and sp[0] and sp[1])
        starts.append(len(pa_b))
        gids.append(gid)
        lin_h.append([ix(b) for b in ln[1]] if ok else [])
        lin_a.append([ix(b) for b in ln[0]] if ok else [])
        sp_h.append(ix(sp[0]) if ok else -1)
        sp_a.append(ix(sp[1]) if ok else -1)
        for b, p, ob in gpa[gid]:
            pa_b.append(ix(b)); pa_p.append(ix(p)); pa_o.append(1 if ob else 0)
    starts.append(len(pa_b))
    flat_h = np.concatenate([np.array(x, np.int32) for x in lin_h] + [np.zeros(0, np.int32)])
    off_h = np.cumsum([0] + [len(x) for x in lin_h]).astype(np.int64)
    flat_a = np.concatenate([np.array(x, np.int32) for x in lin_a] + [np.zeros(0, np.int32)])
    off_a = np.cumsum([0] + [len(x) for x in lin_a]).astype(np.int64)
    np.savez_compressed(
        f"{CACHE}/pa.npz",
        gid=np.array(gids), starts=np.array(starts, np.int64),
        bat=np.array(pa_b, np.int32), pit=np.array(pa_p, np.int32),
        ob=np.array(pa_o, np.int8),
        lin_h=flat_h, off_h=off_h, lin_a=flat_a, off_a=off_a,
        sp_h=np.array(sp_h, np.int32), sp_a=np.array(sp_a, np.int32),
        n_players=np.array([len(idx)]))
    print(f"cached {len(gids):,} PA-games, {len(pa_b):,} plate appearances, "
          f"{len(idx):,} players ({time.time()-t0:.0f}s)", flush=True)


# =============================================================== elo re-walk
def elo_lf(games, lg_fip, params):
    """Walk the FIP-Elo engine and return {game_id: logit(p)} for every game."""
    from mlbwp.rating import FipPitcherElo
    m = FipPitcherElo(lg_fip=lg_fip, **params)
    out = {}
    last = None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        p = min(max(p, 1e-9), 1 - 1e-9)
        out[g["game_id"]] = math.log(p / (1 - p))
        m.update(g)
    return out


# ========================================================= trueskill re-walk
_SQRT2PI = math.sqrt(2 * math.pi)


def ts_walk(P, beta_rel=0.5, tau_rel=0.01, eps_rel=0.0, mu0=25.0, sig0=25.0 / 3.0):
    """Re-walk every plate appearance and snapshot, before each game, the mu/var/n
    of the 20 players whose ratings that game's feature reads.

    beta_rel = beta/sig0 and tau_rel = tau/sig0: the engine is scale-invariant
    (mu0 fixed, everything else scaled by c just rescales the edge, which is
    z-scored away before the blend), so these two ratios plus the draw margin
    eps_rel = eps/c are the whole free parameter space.

    Returns per game: home_off, away_off, home_def, away_def under the plain
    mu read, and the raw (mu, var, n) of every read player so that read-time
    variants (n-shrinkage, sigma discount) cost nothing extra."""
    beta = beta_rel * sig0
    tau = tau_rel * sig0
    exp, sqrt, erf = math.exp, math.sqrt, math.erf
    b2 = 2.0 * beta * beta
    t2 = tau * tau

    n_pl = int(P["n_players"][0])
    mu = [mu0] * n_pl
    var = [sig0 * sig0] * n_pl
    cnt = [0] * n_pl

    bat = P["bat"]; pit = P["pit"]; ob = P["ob"]; starts = P["starts"]
    lin_h = P["lin_h"]; off_h = P["off_h"]; lin_a = P["lin_a"]; off_a = P["off_a"]
    sp_h = P["sp_h"]; sp_a = P["sp_a"]
    ng = len(starts) - 1

    ho = np.full(ng, np.nan); ao = np.full(ng, np.nan)
    hd = np.full(ng, np.nan); ad = np.full(ng, np.nan)
    # variance/count aggregates for read-time variants
    ho_s = np.full(ng, np.nan); ao_s = np.full(ng, np.nan)
    hd_s = np.full(ng, np.nan); ad_s = np.full(ng, np.nan)
    ho_n = np.full(ng, np.nan); ao_n = np.full(ng, np.nan)
    hd_n = np.full(ng, np.nan); ad_n = np.full(ng, np.nan)

    bat_l = bat.tolist(); pit_l = pit.tolist(); ob_l = ob.tolist()
    st_l = starts.tolist()
    lh = lin_h.tolist(); la = lin_a.tolist()
    oh = off_h.tolist(); oa = off_a.tolist()
    sph = sp_h.tolist(); spa = sp_a.tolist()

    for gi in range(ng):
        h = sph[gi]
        if h >= 0:
            L = lh[oh[gi]:oh[gi + 1]]
            A = la[oa[gi]:oa[gi + 1]]
            a = spa[gi]
            nL = len(L); nA = len(A)
            ho[gi] = sum(mu[b] for b in L) / nL
            ao[gi] = sum(mu[b] for b in A) / nA
            hd[gi] = mu[h]
            ad[gi] = mu[a]
            ho_s[gi] = sum(sqrt(var[b]) for b in L) / nL
            ao_s[gi] = sum(sqrt(var[b]) for b in A) / nA
            hd_s[gi] = sqrt(var[h]); ad_s[gi] = sqrt(var[a])
            ho_n[gi] = sum(cnt[b] for b in L) / nL
            ao_n[gi] = sum(cnt[b] for b in A) / nA
            hd_n[gi] = cnt[h]; ad_n[gi] = cnt[a]
        for i in range(st_l[gi], st_l[gi + 1]):
            if ob_l[i]:
                w = bat_l[i]; l = pit_l[i]
            else:
                w = pit_l[i]; l = bat_l[i]
            vw = var[w] + t2
            vl = var[l] + t2
            c2 = b2 + vw + vl
            c = sqrt(c2)
            t = (mu[w] - mu[l]) / c - eps_rel
            d = 0.5 * (1.0 + erf(t * 0.7071067811865476))
            v = (exp(-0.5 * t * t) / _SQRT2PI) / d if d > 1e-12 else -t
            ww = v * (v + t)
            if ww < 0.0:
                ww = 0.0
            elif ww > 1.0:
                ww = 1.0
            mu[w] += vw / c * v
            mu[l] -= vl / c * v
            var[w] = vw * (1.0 - vw / c2 * ww)
            var[l] = vl * (1.0 - vl / c2 * ww)
            cnt[w] += 1
            cnt[l] += 1
    return dict(gid=P["gid"], ho=ho, ao=ao, hd=hd, ad=ad,
                ho_s=ho_s, ao_s=ao_s, hd_s=hd_s, ad_s=ad_s,
                ho_n=ho_n, ao_n=ao_n, hd_n=hd_n, ad_n=ad_n, mu0=mu0)


def ts_edge_from(W, shrink_n=0.0, sig_k=0.0):
    """(home_off-away_def) - (away_off-home_def) with optional read-time priors:
      shrink_n: mu <- mu0 + (mu-mu0)*n/(n+shrink_n)      (debut/low-sample prior)
      sig_k   : mu <- mu - sig_k*sigma                    (uncertainty discount)"""
    mu0 = W["mu0"]

    def rd(m, s, n):
        v = m
        if sig_k:
            v = v - sig_k * s
        if shrink_n:
            v = mu0 + (v - mu0) * (n / (n + shrink_n))
        return v
    ho = rd(W["ho"], W["ho_s"], W["ho_n"])
    ao = rd(W["ao"], W["ao_s"], W["ao_n"])
    hd = rd(W["hd"], W["hd_s"], W["hd_n"])
    ad = rd(W["ad"], W["ad_s"], W["ad_n"])
    return ho, ao, hd, ad, (ho - ad) - (ao - hd)


# ================================================================== dataset
class DS:
    """The DEV design matrix. Everything downstream reads this."""

    def __init__(self, ts_params=None, elo_params=None, quiet=False):
        F = np.load(f"{CACHE}/feat.npz", allow_pickle=True)
        with open(f"{CACHE}/games.pkl", "rb") as fh:
            G = pickle.load(fh)
        self.games = G["games"]
        self.lg_fip = G["lg_fip"]
        self.P = dict(np.load(f"{CACHE}/pa.npz", allow_pickle=True))

        self.ts_params = dict(SHIP_TS if ts_params is None else ts_params)
        self.elo_params = dict(SHIP_ELO if elo_params is None else elo_params)

        # per-side raw features, keyed by game id
        gid = F["game_id"]
        self.raw = {g: i for i, g in enumerate(gid)}
        self.F = F
        self._ts_cache = {}
        self.quiet = quiet
        self._build_index()

    def _build_index(self):
        F = self.F
        season = F["season"].astype(int)
        m = (season >= DEV_LO) & (season <= DEV_HI)
        self.gid = F["game_id"][m]
        self.season = season[m]
        self.y = F["y"][m]
        assert self.season.max() <= DEV_HI and self.season.min() >= DEV_LO, "TEST leak"
        self.side = {k: F[k][m] for k in ("bp_home", "bp_away", "si_home", "si_away",
                                          "pw_home", "pw_away", "br_home", "br_away")}
        self.groups = folds_for(self.season)
        self.pos = {g: i for i, g in enumerate(self.gid)}

    # -- TrueSkill (cached per (beta_rel,tau_rel,eps_rel)) --------------
    def ts_walk_cached(self, beta_rel, tau_rel, eps_rel):
        key = (round(beta_rel, 6), round(tau_rel, 8), round(eps_rel, 6))
        W = self._ts_cache.get(key)
        if W is None:
            path = f"{CACHE}/ts_{key[0]}_{key[1]}_{key[2]}.npz"
            if os.path.isfile(path):
                W = dict(np.load(path, allow_pickle=True))
                W["mu0"] = 25.0
            else:
                t = time.time()
                W = ts_walk(self.P, beta_rel=beta_rel, tau_rel=tau_rel, eps_rel=eps_rel)
                if not self.quiet:
                    print(f"    ts walk {key} {time.time()-t:.0f}s", flush=True)
                np.savez_compressed(path, **{k: v for k, v in W.items() if k != "mu0"})
            self._ts_cache[key] = W
        return W

    def ts_sides(self, tsp=None):
        tsp = tsp or self.ts_params
        W = self.ts_walk_cached(tsp["beta_rel"], tsp["tau_rel"], tsp.get("eps_rel", 0.0))
        ho, ao, hd, ad, edge = ts_edge_from(W, tsp.get("shrink_n", 0.0), tsp.get("sig_k", 0.0))
        idx = {g: i for i, g in enumerate(W["gid"])}
        n = len(self.gid)
        out = np.full((n, 5), np.nan)
        for i, g in enumerate(self.gid):
            j = idx.get(g)
            if j is not None:
                out[i] = (ho[j], ao[j], hd[j], ad[j], edge[j])
        return out

    # -- empirical-Bayes priors re-applied to the AS-OF raw state (axis F) --
    def _state(self):
        if getattr(self, "_S", None) is None:
            S = dict(np.load(f"{CACHE}/state.npz"))
            season = self.F["season"].astype(int)
            m = (season >= DEV_LO) & (season <= DEV_HI)
            self._S = {k: (v[m] if k != "const" else v) for k, v in S.items()}
        return self._S

    def priors_features(self, bp_prior=BP_PRIOR_OUTS, pw_prior=PWR_PRIOR_PA,
                        br_prior=BR_PRIOR_PA, si_prior=SIERA_PRIOR, br_rv=BR_RV):
        """Recompute bp/pw/br/si differentials at arbitrary shrinkage strengths.
        Identical arithmetic to the walk, just re-read from cached as-of counts."""
        from mlbwp.siera import siera_value
        S = self._state()
        c = S["const"]
        bp_lg, lg_iso = c[0], c[1]
        lg = dict(so=c[2], bb=c[3], gb=c[4], fb=c[5], pu=c[6])
        lg_net = lg["gb"] - lg["fb"] - lg["pu"]

        B = S["bp"]
        pr = bp_prior * bp_lg / 3.0
        bp_h = (B[:, 0] + pr) / (B[:, 1] + bp_prior) * 3.0
        bp_a = (B[:, 2] + pr) / (B[:, 3] + bp_prior) * 3.0

        W = S["pw"].astype(float)                      # (n,2,9,3) PA,H,TB
        iso = (W[..., 2] - W[..., 1] + pw_prior * lg_iso) / (W[..., 0] + pw_prior)
        pw_h, pw_a = iso[:, 0].mean(1), iso[:, 1].mean(1)

        R = S["br"].astype(float)                      # (n,2,9,6) PA,SB,CS,XB,OOB,GIDP
        rv = sum(br_rv[k] * R[..., 1 + k] for k in range(5))
        brate = rv / (R[..., 0] + br_prior)
        br_h, br_a = brate[:, 0].mean(1), brate[:, 1].mean(1)

        E = S["si"]                                    # (n,2,6) SO,BB,GB,FB,PU,PA
        pa = E[:, :, 5] + si_prior
        so = (E[:, :, 0] + si_prior * lg["so"]) / pa
        bb = (E[:, :, 1] + si_prior * lg["bb"]) / pa
        net = ((E[:, :, 2] - E[:, :, 3] - E[:, :, 4]) + si_prior * lg_net) / pa
        sv = (6.145 - 16.986 * so + 11.434 * bb - 1.858 * net + 7.653 * so * so
              - 6.664 * net * net + 10.130 * so * net - 5.195 * bb * net)
        return {"bp_h": bp_h, "bp_a": bp_a, "pw_h": pw_h, "pw_a": pw_a,
                "br_h": br_h, "br_a": br_a, "si_h": sv[:, 0], "si_a": sv[:, 1]}

    # -- Elo (cached per param tuple) -----------------------------------
    def elo_lf(self, params=None):
        params = params or self.elo_params
        lf = elo_lf(self.games, self.lg_fip, params)
        return np.array([lf[g] for g in self.gid])

    # -- the full design ------------------------------------------------
    def design(self, elo_params=None, ts_params=None):
        lf = self.elo_lf(elo_params)
        T = self.ts_sides(ts_params)
        s = self.side
        cols = {
            "lf": lf,
            "ts": T[:, 4],
            "bp": s["bp_away"] - s["bp_home"],     # lower FIP better -> + favours home
            "pw": s["pw_home"] - s["pw_away"],
            "si": s["si_home"] - s["si_away"],     # lower SIERA better -> negative coef
            "br": s["br_home"] - s["br_away"],
        }
        ok = np.isfinite(T[:, 4])
        extra = {"ho": T[:, 0], "ao": T[:, 1], "hd": T[:, 2], "ad": T[:, 3],
                 "bp_h": s["bp_home"], "bp_a": s["bp_away"],
                 "si_h": s["si_home"], "si_a": s["si_away"]}
        return cols, extra, ok


def load_ds(quiet=False):
    return DS(quiet=quiet)


def full_cols(cols):
    return [cols["lf"]] + [cols[k] for k in FEATS]


# ================================================================ baseline
def run_baseline(ds=None, save=True):
    ds = ds or load_ds()
    cols, extra, ok = ds.design()
    y, seas, groups = ds.y[ok], ds.season[ok], ds.groups
    C = {k: v[ok] for k, v in cols.items()}
    lfp = _sig(C["lf"])
    res = {"n": int(ok.sum()), "dev": f"{DEV_LO}-{DEV_HI}", "folds": N_FOLDS,
           "fold_seasons": [sorted(g) for g in groups]}
    res["raw_fip_elo_ll"] = round(float(llvec(y, lfp).mean()), 6)
    tiers = {
        "recal": [C["lf"]],
        "recbp": [C["lf"], C["bp"], C["si"]],
        "full": full_cols(C),
    }
    for name, cl in tiers.items():
        d = oof_ll(cl, y, seas, groups)
        fd, m, sd = fold_stats(d, seas, groups)
        res[name] = {"ll": round(float(d.mean()), 6),
                     "fold_ll": [round(x, 6) for x in fd]}
        print(f"  {name:<8} OOF ll {d.mean():.6f}   folds "
              f"{' '.join(f'{x:.5f}' for x in fd)}", flush=True)
    print(f"  raw fip-elo (uncalibrated) ll {res['raw_fip_elo_ll']:.6f}   n={res['n']:,}")
    if save:
        save_axis("baseline", res)
    return ds, res


# ================================================================= reporting
def save_axis(key, payload):
    old = {}
    if os.path.isfile(REPORT):
        old = json.load(open(REPORT))
    old[key] = payload
    old["_meta"] = {"dev": f"{DEV_LO}-{DEV_HI}", "folds": N_FOLDS,
                    "ship_elo": SHIP_ELO, "ship_ts": SHIP_TS,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = REPORT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(old, fh, indent=1)
    os.replace(tmp, REPORT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--baseline"
    if mode == "--cache":
        build_cache()
    elif mode == "--baseline":
        run_baseline()
    else:
        import mlb_depth_axes as AX
        AX.main(mode, sys.argv[2:])
