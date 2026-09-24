"""pv_nhl_duels -- NHL player value, component DISCRETE_DUELS (walk-forward, DEV only).

Player-value program (MLB principle: rate each player on the events he
individually controls, 1v1 and opponent-adjusted, walk-forward, EB-shrunk).
This module measures the three "discrete duel" channels of a skater and converts
each into GOALS PER 60 of his time on ice so they can be summed:

  FO   faceoffs, a true 1v1 (winner vs loser). Online Bayesian Bradley-Terry
       (extended-Kalman / Glicko-style) with home, end-zone-defender and
       man-advantage context terms learned online; new players start from a
       position prior. The goal value of a faceoff WIN is estimated per context
       (end/neutral zone x EV / special teams / empty net) as the goal
       differential of the ensuing sequence (to the next faceoff), with home
       and away team-season fixed effects so team quality cannot pose as
       faceoff value. fo_g60 = (P(win vs avg) - 0.5) * sum_c(beta_c * FO_c/60).
  PEN  penalties drawn minus taken. Each penalty is valued by the man-advantage
       goal differential it creates (window after the call, team-season FE);
       coincidental (offsetting) penalties and misconducts are worth 0.
       pen_g60 = EB-shrunk drawn value/60 - taken value/60, relative to the
       league-average skater.
  TKGV takeaways minus giveaways, ARENA-ADJUSTED (the home rink's scorer
       inflates these 0.4x-2.2x; see data/pv_nhl_qa.json section J). Each event
       is divided by the arena's walk-forward home/road recording ratio and
       valued by the goal differential of the rest of its sequence (by zone,
       team-season FE). Kept SEPARATE from the headline sum until it proves
       reliable and predictive (see pv_nhl_duels_validate.py).

WALK-FORWARD (hard rules)
  * goal values for season s are fit on seasons < s only (expanding window);
    season 2010-11 (warm-up, never scored) uses declared literature priors.
  * every per-player-game value is a PRE-GAME snapshot: player EWMA sums over
    his strictly earlier games; league / position means and arena factors are
    committed only at date changes (strictly earlier dates).
  * faceoff ratings are read at game start, before that game's faceoffs; the
    league mean rating they are centred on is committed per DATE (strictly
    earlier dates; amendment 1 of data/pv_nhl_serve_prereg.json).

PROTOCOL: DEV = gid < 2018000000. Loaders refuse TEST rows. `--allow-test` exists
ONLY for the lead engineer's single TEST look / serving; this agent never runs it.
Market-blind: no odds anywhere.

Outputs (DEV):
  data/pv_nhl_discrete_duels.csv        one row per dressed skater per game
  data/pv_nhl_discrete_duels_values.json goal values per season + tuned hyperparameters
  data/pv_nhl_discrete_duels_fo.parquet  per-faceoff pre-event prob (for validation)

Usage:
  python phase0/pv_nhl_duels.py values      # goal-value regressions only
  python phase0/pv_nhl_duels.py tune        # DEV-A hyperparameter grids
  python phase0/pv_nhl_duels.py build       # full build with tuned params
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID, load_events, load_rosters  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "pv_nhl_discrete_duels")
DEV_SEASONS = [20102011, 20112012, 20122013, 20132014, 20142015, 20152016,
               20162017, 20172018]
TUNE_SEASONS = [20112012, 20122013, 20132014, 20142015]      # DEV-A: tuning
HOLD_SEASONS = [20152016, 20162017, 20172018]                # DEV-B: untouched by tuning

# Declared before any DEV fit: the ONLY values used for the 2010-11 warm-up
# season (never scored). Literature orders of magnitude (EV faceoff win ~0.01
# goals; PP net ~0.17 goals per minor).
PRIOR_VALUES = {
    "fo": {"EV_END": 0.012, "EV_NEU": 0.004, "ST_OZ": 0.02, "ST_DZ": 0.02,
           "ST_NEU": 0.004, "EN_END": 0.03, "EN_NEU": 0.01},
    "pen": {"m2": 0.17, "m4": 0.30, "maj5": 0.45, "ps": 0.30},
    "tkgv": {"tk_O": 0.0, "tk_N": 0.0, "tk_D": 0.0, "gv_O": 0.0, "gv_N": 0.0,
             "gv_D": 0.0},
}
FO_CATS = list(PRIOR_VALUES["fo"])
PEN_CATS = list(PRIOR_VALUES["pen"])
TK_CATS = list(PRIOR_VALUES["tkgv"])
PEN_WINDOW = {"m2": 120, "m4": 240, "maj5": 300, "ps": 60}
PP_MINUTES = {("MIN", 2.0): "m2", ("BEN", 2.0): "m2", ("MIN", 4.0): "m4",
              ("MAJ", 5.0): "maj5", ("MAT", 10.0): "maj5", ("MAT", 5.0): "maj5"}
ARENA_K = 10.0          # arena-factor shrinkage, pseudo home games (declared)
ARENA_CARRY = 0.5       # weight of last season's log-factor in this season's prior


# ======================================================================= data
def load_games(allow_test=False):
    g = pd.read_csv(os.path.join(ROOT, "data", "nhl_games.csv"))
    g = g.rename(columns={"game_id": "gid"})
    if not allow_test:
        g = g[g.gid < DEV_MAX_GID]
    return g[["gid", "date", "season", "type", "home", "away", "home_goals",
              "away_goals"]].reset_index(drop=True)


def load_base(allow_test=False):
    cols = ["season", "gtype", "period", "ptype", "per_sec", "game_sec", "sort",
            "ev", "team", "is_home", "shooter", "assist1", "assist2", "fo_win",
            "fo_lose", "player", "pen_by", "pen_drawn", "pen_code", "pen_min",
            "zone", "a_g", "a_sk", "h_sk", "h_g", "sk_for", "sk_ag"]
    ev = load_events(columns=cols, allow_test=allow_test)
    ev = ev[ev.ptype != "SO"].sort_values(["gid", "game_sec", "sort"],
                                          kind="mergesort").reset_index(drop=True)
    ro = load_rosters(allow_test=allow_test)
    games = load_games(allow_test)
    if not allow_test:
        assert ev.gid.max() < DEV_MAX_GID and ro.gid.max() < DEV_MAX_GID
    return ev, ro, games


def team_season_index(ev, games):
    """home/away team-season integer ids per event (for fixed effects)."""
    g = games.set_index("gid")
    h = ev.gid.map(g.home).astype(str) + "_" + ev.season.astype(str)
    a = ev.gid.map(g.away).astype(str) + "_" + ev.season.astype(str)
    cats = pd.Index(sorted(set(h) | set(a)))
    return cats.get_indexer(h), cats.get_indexer(a), len(cats)


# ============================================================ goal outcomes
def sequence_after(ev):
    """Home and away goals AFTER each event and before the next faceoff."""
    isfo = (ev.ev == "faceoff").values
    newg = np.r_[True, ev.gid.values[1:] != ev.gid.values[:-1]]
    seq = np.cumsum(isfo | newg)
    isg = (ev.ev == "goal").values
    hh = ev.is_home.values == 1
    ch = np.cumsum(isg & hh)
    ca = np.cumsum(isg & ~hh)
    # last index of each sequence
    last = np.r_[np.nonzero(seq[1:] != seq[:-1])[0], len(seq) - 1]
    end_of = last[seq - 1]
    return ch[end_of] - ch, ca[end_of] - ca


def window_goals(ev, t_gid, t_sec, L, inclusive=False):
    """Home/away goals with game_sec in (t, t+L] of the same game ([t, t+L] when
    inclusive -- penalty shots are taken at the second the call is made)."""
    gmask = (ev.ev == "goal").values
    gk = ev.gid.values[gmask].astype(np.int64) * 100000 + ev.game_sec.values[gmask]
    gh = ev.is_home.values[gmask] == 1
    kh, ka = np.sort(gk[gh]), np.sort(gk[~gh])
    lo = t_gid.astype(np.int64) * 100000 + t_sec - (1 if inclusive else 0)
    hi = lo + L
    nh = np.searchsorted(kh, hi, "right") - np.searchsorted(kh, lo, "right")
    na = np.searchsorted(ka, hi, "right") - np.searchsorted(ka, lo, "right")
    return nh, na


def fe_regress(y, Xd, hi, ai, nts):
    """OLS of y on dense regressors Xd + home and away team-season fixed effects.
    Returns the coefficients on Xd (FE absorb team quality)."""
    n, k = Xd.shape
    rows = np.arange(n)
    H = sparse.csr_matrix((np.ones(n), (rows, hi)), shape=(n, nts))
    A = sparse.csr_matrix((np.ones(n), (rows, ai)), shape=(n, nts))
    X = sparse.hstack([sparse.csr_matrix(Xd), H, A]).tocsr()
    sol = lsqr(X, y.astype(float), atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    return sol[:k]


def fo_frame(ev, hi, ai):
    m = (ev.ev == "faceoff").values
    after_h, after_a = sequence_after(ev)
    f = ev.loc[m, ["gid", "season", "is_home", "zone", "h_sk", "a_sk", "h_g", "a_g",
                   "fo_win", "fo_lose", "sk_for", "sk_ag", "game_sec"]].copy()
    f["y"] = (after_h - after_a)[m]
    f["hw"] = (f.is_home == 1).astype(int)
    f["zh"] = np.where(f.hw == 1, f.zone, f.zone.map({"O": "D", "D": "O", "N": "N"}))
    st = np.where((f.h_g == 0) | (f.a_g == 0), "EN",
                  np.where(f.h_sk > f.a_sk, "PP", np.where(f.h_sk < f.a_sk, "PK", "EV")))
    f["st"] = st
    end = f.zh.isin(["O", "D"]).values
    cat = np.where(st == "EV", np.where(end, "EV_END", "EV_NEU"),
          np.where(st == "EN", np.where(end, "EN_END", "EN_NEU"),
          np.where(~end, "ST_NEU",
          np.where(((st == "PP") & (f.zh == "O")) | ((st == "PK") & (f.zh == "D")),
                   "ST_OZ", "ST_DZ"))))
    f["cat"] = cat
    f["hi"], f["ai"] = hi[m], ai[m]
    return f


def fo_values(f, train_seasons):
    d = f[f.season.isin(train_seasons)]
    ctx = (d.zh.astype(str) + "_" + d.st.astype(str)).to_numpy(dtype=object)
    ctx_levels = sorted(set(ctx))
    Xc = (ctx[:, None] == np.array(ctx_levels, dtype=object)[None, :]).astype(float)
    cat = d.cat.to_numpy(dtype=object)
    Xv = ((cat[:, None] == np.array(FO_CATS, dtype=object)[None, :])
          * d.hw.values[:, None]).astype(float)
    b = fe_regress(d.y.values, np.hstack([Xv, Xc]), d.hi.values, d.ai.values,
                   int(max(f.hi.max(), f.ai.max()) + 1))
    return {c: float(b[i]) for i, c in enumerate(FO_CATS)}, int(len(d))


def pen_frame(ev, hi, ai):
    """One row per penalty, with its PP category and whether it is offset."""
    m = (ev.ev == "penalty").values
    p = ev.loc[m, ["gid", "season", "game_sec", "is_home", "pen_by", "pen_drawn",
                   "pen_code", "pen_min"]].copy()
    p["hi"], p["ai"] = hi[m], ai[m]
    key = list(zip(p.pen_code, p.pen_min.astype(float)))
    p["cat"] = pd.Series([("ps" if c == "PS" else PP_MINUTES.get(k, "none"))
                          for c, k in zip(p.pen_code, key)], index=p.index, dtype=object)
    p["mins"] = [{"m2": 2, "m4": 4, "maj5": 5}.get(c, 0) for c in p.cat]
    # offsetting: greedy exact-minute matching inside (gid, game_sec) clusters;
    # then a side keeps value only while its unmatched minutes exceed the other's
    p["live"] = 0
    p = p.reset_index(drop=True)
    for (gid, t), idx in p.groupby(["gid", "game_sec"]).groups.items():
        idx = list(idx)
        if len(idx) == 1:
            if p.at[idx[0], "cat"] != "none":
                p.at[idx[0], "live"] = 1
            continue
        side = {1: [], 0: []}
        for i in idx:
            c = p.at[i, "cat"]
            if c == "ps":
                p.at[i, "live"] = 1
            elif c != "none":
                side[int(p.at[i, "is_home"])].append(i)
        H = sorted(side[1], key=lambda i: p.at[i, "mins"])
        A = sorted(side[0], key=lambda i: p.at[i, "mins"])
        for i in list(H):
            for j in A:
                if p.at[j, "mins"] == p.at[i, "mins"]:
                    H.remove(i)
                    A.remove(j)
                    break
        mh = sum(p.at[i, "mins"] for i in H)
        ma = sum(p.at[i, "mins"] for i in A)
        if H and A:           # unequal leftovers on both sides: net side keeps it
            if mh > ma:
                A = []
            elif ma > mh:
                H = []
            else:
                H, A = [], []
        for i in H + A:
            p.at[i, "live"] = 1
    return p


def pen_values(ev, p, train_seasons):
    out, ns = {}, {}
    nts = int(max(p.hi.max(), p.ai.max()) + 1)
    d0 = p[p.season.isin(train_seasons) & (p.live == 1)]
    # "clean" calls: the only live penalty at that (gid, second)
    cnt = d0.groupby(["gid", "game_sec"]).cat.transform("size")
    d0 = d0[cnt == 1]
    for c in PEN_CATS:
        d = d0[d0.cat == c]
        nh, na = window_goals(ev, d.gid.values, d.game_sec.values, PEN_WINDOW[c],
                              inclusive=(c == "ps"))
        y = (nh - na).astype(float)
        drew = np.where(d.is_home.values == 1, -1.0, 1.0)   # +1: home drew it
        Xd = np.column_stack([drew, np.ones(len(d))])
        b = fe_regress(y, Xd, d.hi.values, d.ai.values, nts)
        out[c] = float(b[0])
        ns[c] = int(len(d))
    return out, ns


def tk_frame(ev, hi, ai):
    m = ev.ev.isin(["giveaway", "takeaway"]).values
    after_h, after_a = sequence_after(ev)
    t = ev.loc[m, ["gid", "season", "is_home", "zone", "player", "ev"]].copy()
    sgn = np.where(t.is_home.values == 1, 1.0, -1.0)
    t["y_act"] = sgn * (after_h - after_a)[m]
    t["cat"] = pd.Series(np.where(t.ev == "takeaway", "tk_", "gv_"), index=t.index,
                         dtype=object) + t.zone.fillna("N").astype(object)
    t["sgn"] = sgn
    t["hi"], t["ai"] = hi[m], ai[m]
    return t


def tk_values(t, train_seasons):
    d = t[t.season.isin(train_seasons)]
    y_home = d.sgn.values * d.y_act.values
    cat = d.cat.to_numpy(dtype=object)
    Xv = ((cat[:, None] == np.array(TK_CATS, dtype=object)[None, :])
          * d.sgn.values[:, None]).astype(float)
    Xd = np.hstack([Xv, np.ones((len(d), 1))])
    b = fe_regress(y_home, Xd, d.hi.values, d.ai.values,
                   int(max(t.hi.max(), t.ai.max()) + 1))
    return {c: float(b[i]) for i, c in enumerate(TK_CATS)}, int(len(d))


def walk_forward_values(ev, games, seasons):
    hi, ai, _ = team_season_index(ev, games)
    f = fo_frame(ev, hi, ai)
    p = pen_frame(ev, hi, ai)
    t = tk_frame(ev, hi, ai)
    vals = {}
    for s in seasons:
        prior = [x for x in seasons if x < s]
        if not prior:
            vals[s] = {"fo": dict(PRIOR_VALUES["fo"]), "pen": dict(PRIOR_VALUES["pen"]),
                       "tkgv": dict(PRIOR_VALUES["tkgv"]), "fit_on": "declared prior"}
            continue
        fv, nf = fo_values(f, prior)
        pv, npn = pen_values(ev, p, prior)
        tv, nt = tk_values(t, prior)
        vals[s] = {"fo": fv, "pen": pv, "tkgv": tv, "n_fo": nf, "n_pen": npn,
                   "n_tkgv": nt, "fit_on": f"{prior[0]}..{prior[-1]}"}
        print(s, "fo", {k: round(v, 4) for k, v in fv.items()}, flush=True)
        print(s, "pen", {k: round(v, 4) for k, v in pv.items()}, flush=True)
        print(s, "tk", {k: round(v, 4) for k, v in tv.items()}, flush=True)
    return vals, f, p, t


# ============================================================ faceoff rating
@njit(cache=True)
def _fo_engine(w, l, ch, cd, cp, gidx, seas, posw, posl, n_players,
               ro_g_ptr, ro_p, sigma0, q_season, mu_c, mu_w, mu_d, lr, gday):
    """Online Bradley-Terry EKF. Faceoffs and roster snapshots in game order.

    ch/cd/cp: context of the WINNER (+1/-1/0): home, defending an end-zone draw,
    man advantage. Returns pre-faceoff win prob of the actual winner, the three
    context coefficients' trajectory end, and per roster-row pre-game (m, v, n).
    gday: per game (engine order = date, gid) its date as yyyymmdd. The league mean
    rating snapshot (snap_bar) is COMMITTED PER DATE (amendment 1): every game of
    date d reads the running mean as of the end of the previous date, so it never
    depends on same-date games with a lower gid. (A player's own m / v at game start
    only moved in his earlier games, all on earlier dates.)
    """
    m = np.zeros(n_players)
    v = np.full(n_players, -1.0)          # -1 = never seen
    last_s = np.full(n_players, -1, np.int64)
    nfo = np.zeros(n_players)
    nf = len(w)
    pw = np.empty(nf)
    th = np.zeros(3)                      # home, def-endzone, man-adv
    mbar = 0.0
    mbar_day = 0.0                        # mbar as of the end of the previous date
    n_ro = len(ro_p)
    snap_m = np.empty(n_ro)
    snap_v = np.empty(n_ro)
    snap_n = np.empty(n_ro)
    snap_bar = np.empty(n_ro)
    ngames = len(ro_g_ptr) - 1
    fi = 0
    for g in range(ngames):
        if g == 0 or gday[g] != gday[g - 1]:
            mbar_day = mbar               # commit at the date change only
        # --- snapshot all dressed skaters of game g BEFORE its faceoffs
        for r in range(ro_g_ptr[g], ro_g_ptr[g + 1]):
            p = ro_p[r]
            if v[p] < 0:
                snap_m[r] = np.nan
                snap_v[r] = np.nan
            else:
                snap_m[r] = m[p]
                snap_v[r] = v[p]
            snap_n[r] = nfo[p]
            snap_bar[r] = mbar_day
        # --- faceoffs of game g
        while fi < nf and gidx[fi] == g:
            a = w[fi]
            b = l[fi]
            s = seas[fi]
            for k in range(2):
                p = a if k == 0 else b
                ps = posw[fi] if k == 0 else posl[fi]
                if v[p] < 0:
                    m[p] = mu_c if ps == 0 else (mu_w if ps == 1 else mu_d)
                    v[p] = sigma0 * sigma0
                    last_s[p] = s
                elif last_s[p] != s:
                    v[p] = min(v[p] + q_season * (s - last_s[p]), sigma0 * sigma0)
                    last_s[p] = s
            z = m[a] - m[b] + th[0] * ch[fi] + th[1] * cd[fi] + th[2] * cp[fi]
            vv = v[a] + v[b]
            zs = z / np.sqrt(1.0 + 0.39269908 * vv)      # pi/8
            pw[fi] = 1.0 / (1.0 + np.exp(-zs))
            p1 = 1.0 / (1.0 + np.exp(-z))
            info = p1 * (1.0 - p1)
            den = 1.0 + vv * info
            r1 = 1.0 - p1
            m[a] += v[a] * r1 / den
            m[b] -= v[b] * r1 / den
            v[a] -= v[a] * v[a] * info / den
            v[b] -= v[b] * v[b] * info / den
            th[0] += lr * r1 * ch[fi]
            th[1] += lr * r1 * cd[fi]
            th[2] += lr * r1 * cp[fi]
            nfo[a] += 1
            nfo[b] += 1
            mbar = 0.99999 * mbar + 0.00001 * 0.5 * (m[a] + m[b])
            fi += 1
    return pw, th, snap_m, snap_v, snap_n, snap_bar


def season_idx(s):
    return int(str(int(s))[:4]) - 2010


def fo_inputs(f, games, ro, pid_index):
    """Arrays for the engine, games in (date, gid) order."""
    gs = games.sort_values(["date", "gid"])
    gorder = gs.gid.values
    gpos = {g: i for i, g in enumerate(gorder)}
    gday = gs.date.astype(str).str.replace("-", "").astype(np.int64).to_numpy()
    assert (np.diff(gday) >= 0).all()
    f = f.assign(g=f.gid.map(gpos)).sort_values(["g", "game_sec"], kind="mergesort")
    posmap = ro.set_index(["gid", "pid"]).pos
    def pc(gids, pids):
        p = posmap.reindex(list(zip(gids, pids))).values
        return np.where(p == "C", 0, np.where(p == "D", 2, 1)).astype(np.int64)
    w = f.fo_win.astype(np.int64).map(pid_index).values.astype(np.int64)
    l = f.fo_lose.astype(np.int64).map(pid_index).values.astype(np.int64)
    zone = f.zone.values
    cd = np.where(zone == "D", 1.0, np.where(zone == "O", -1.0, 0.0))
    ch = np.where(f.is_home.values == 1, 1.0, -1.0)
    diff = f.sk_for.values - f.sk_ag.values
    cp = np.sign(np.nan_to_num(diff)).astype(float)
    seas = f.season.map(season_idx).values.astype(np.int64)
    arr = dict(w=w, l=l, ch=ch, cd=cd, cp=cp, gidx=f.g.values.astype(np.int64),
               seas=seas, posw=pc(f.gid.values, f.fo_win.astype(np.int64).values),
               posl=pc(f.gid.values, f.fo_lose.astype(np.int64).values), gday=gday)
    return arr, f, gpos


def roster_arrays(ro, gpos, pid_index):
    r = ro.assign(g=ro.gid.map(gpos)).sort_values(["g", "pid"], kind="mergesort")
    ptr = np.zeros(len(gpos) + 1, np.int64)
    cnt = np.bincount(r.g.values, minlength=len(gpos))
    ptr[1:] = np.cumsum(cnt)
    return r, ptr, r.pid.map(pid_index).values.astype(np.int64)


def run_fo(arr, ptr, rop, n_players, prm):
    return _fo_engine(arr["w"], arr["l"], arr["ch"], arr["cd"], arr["cp"],
                      arr["gidx"], arr["seas"], arr["posw"], arr["posl"], n_players,
                      ptr, rop, prm["sigma0"], prm["q_season"], prm["mu_c"],
                      prm["mu_w"], prm["mu_d"], prm["lr"], arr["gday"])


def fo_ll(pw, seas_mask):
    return float(-np.log(np.clip(pw[seas_mask], 1e-12, 1)).mean())


# ======================================================= arena adjustment
def arena_factors(ev, games, etype):
    """Walk-forward per-game arena factor for event type `etype`.

    f(arena A, game) = shrunk ratio of per-game counts (both teams) in A's home
    games vs A's road games, using only games on strictly earlier dates of the
    same season, prior = ARENA_CARRY x last season's final log-factor.
    Returns Series gid -> factor.
    """
    cnt = ev[ev.ev == etype].groupby("gid").size()
    g = games.sort_values(["date", "gid"]).copy()
    g["n"] = g.gid.map(cnt).fillna(0.0).values
    fac = {}
    last_logf = {}
    for s, gs in g.groupby("season", sort=True):
        hs, hn, rs, rn = {}, {}, {}, {}
        prior = {t: ARENA_CARRY * last_logf.get(t, 0.0) for t in set(gs.home) | set(gs.away)}
        for d, gd in gs.groupby("date", sort=True):
            for gid, h in zip(gd.gid.values, gd.home.values):
                nh = hn.get(h, 0)
                if nh > 0 and rn.get(h, 0) > 0:
                    raw = np.log((hs[h] / nh + 0.5) / (rs[h] / rn[h] + 0.5))
                    lf = (nh * raw + ARENA_K * prior[h]) / (nh + ARENA_K)
                else:
                    lf = prior[h]
                fac[gid] = float(np.exp(lf))
            for gid, h, a, n in zip(gd.gid.values, gd.home.values, gd.away.values,
                                    gd.n.values):
                hs[h] = hs.get(h, 0.0) + n
                hn[h] = hn.get(h, 0) + 1
                rs[a] = rs.get(a, 0.0) + n
                rn[a] = rn.get(a, 0) + 1
        for t in set(gs.home):
            if hn.get(t, 0) and rn.get(t, 0):
                raw = np.log((hs[t] / hn[t] + 0.5) / (rs[t] / rn[t] + 0.5))
                last_logf[t] = (hn[t] * raw + ARENA_K * prior[t]) / (hn[t] + ARENA_K)
    return pd.Series(fac)


# ===================================================== per player-game table
def player_games(ev, ro, games, vals, fo_sorted):
    """Per dressed skater per game: realised event counts and goal values."""
    sk = ro[ro.pos != "G"][["gid", "season", "side", "team", "pid", "pos", "toi_s"]].copy()
    sk["toi_s"] = sk.toi_s.fillna(0.0)
    key = ["gid", "pid"]
    # faceoffs (winner and loser rows) with the season's context value
    f = fo_sorted
    beta = np.array([vals[s]["fo"][c] for s, c in zip(f.season.values, f.cat.values)])
    fcat = f.cat.to_numpy(dtype=object)
    catcols = {f"fo_n_{c}": (fcat == c).astype(int) for c in FO_CATS}
    fw = pd.DataFrame({"gid": f.gid.values, "pid": f.fo_win.astype(np.int64).values,
                       "fo_n": 1, "fo_w": 1, "fo_val": beta, "fo_real": 0.5 * beta, **catcols})
    fl = pd.DataFrame({"gid": f.gid.values, "pid": f.fo_lose.astype(np.int64).values,
                       "fo_n": 1, "fo_w": 0, "fo_val": beta, "fo_real": -0.5 * beta, **catcols})
    fa = pd.concat([fw, fl]).groupby(key).sum()
    # penalties
    p = vals["_pen_frame"]
    pv = np.array([vals[s]["pen"].get(c, 0.0) if lv == 1 else 0.0
                   for s, c, lv in zip(p.season.values, p.cat.values, p.live.values)])
    pcat = p.cat.to_numpy(dtype=object)
    live = p.live.values == 1
    tcat = {f"pen_take_{c}": ((pcat == c) & live).astype(int) for c in PEN_CATS}
    dcat = {f"pen_draw_{c}": ((pcat == c) & live).astype(int) for c in PEN_CATS}
    pt = pd.DataFrame({"gid": p.gid.values, "pid": p.pen_by.values, "pen_take_all": 1,
                       "pen_take_n": (pv != 0).astype(int), "pen_take_g": pv, **tcat}
                      ).dropna(subset=["pid"])
    pdr = pd.DataFrame({"gid": p.gid.values, "pid": p.pen_drawn.values, "pen_draw_all": 1,
                        "pen_draw_n": (pv != 0).astype(int), "pen_draw_g": pv, **dcat}
                       ).dropna(subset=["pid"])
    pt["pid"] = pt.pid.astype(np.int64)
    pdr["pid"] = pdr.pid.astype(np.int64)
    pta = pt.groupby(key).sum()
    pda = pdr.groupby(key).sum()
    # takeaways / giveaways with arena factor and zone value
    t = vals["_tk_frame"]
    fac_tk = arena_factors(ev, games, "takeaway")
    fac_gv = arena_factors(ev, games, "giveaway")
    isk = t.ev.values == "takeaway"
    fac = np.where(isk, t.gid.map(fac_tk).values, t.gid.map(fac_gv).values)
    tv = np.array([vals[s]["tkgv"][c] for s, c in zip(t.season.values, t.cat.values)])
    tt = pd.DataFrame({"gid": t.gid.values, "pid": t.player.astype(np.int64).values,
                       "tk_n": isk.astype(int), "gv_n": (~isk).astype(int),
                       "tk_adj": np.where(isk, 1.0 / fac, 0.0),
                       "gv_adj": np.where(~isk, 1.0 / fac, 0.0),
                       "tk_g": np.where(isk, tv / fac, 0.0),
                       "gv_g": np.where(~isk, tv / fac, 0.0),
                       **{f"{c}_adj": np.where(t.cat.to_numpy(dtype=object) == c, 1.0 / fac, 0.0)
                          for c in TK_CATS}})
    tka = tt.groupby(key).sum()
    # goals / assists (validation outcome; points)
    gl = ev[ev.ev == "goal"]
    pts = pd.concat([
        pd.DataFrame({"gid": gl.gid.values, "pid": gl.shooter.values, "goals": 1, "assists": 0}),
        pd.DataFrame({"gid": gl.gid.values, "pid": gl.assist1.values, "goals": 0, "assists": 1}),
        pd.DataFrame({"gid": gl.gid.values, "pid": gl.assist2.values, "goals": 0, "assists": 1}),
    ]).dropna(subset=["pid"])
    pts["pid"] = pts.pid.astype(np.int64)
    pta2 = pts.groupby(key).sum()
    out = sk.set_index(key)
    for part in (fa, pta, pda, tka, pta2):
        out = out.join(part, how="left")
    out = out.fillna({c: 0.0 for c in out.columns if c not in ("side", "team", "pos")})
    out = out.reset_index()
    g = games.set_index("gid")
    out["date"] = out.gid.map(g.date)
    return out


# ================================================================ EB-EWMA
@njit(cache=True)
def _ewma_prior(pidx, seas, x, toi, lam, delta):
    """Rows sorted by (player, game order). Returns decayed sums of x and toi
    over the player's STRICTLY EARLIER games (pre-game snapshot)."""
    n = len(pidx)
    S = np.zeros(n)
    T = np.zeros(n)
    cs = 0.0
    ct = 0.0
    for i in range(n):
        if i == 0 or pidx[i] != pidx[i - 1]:
            cs = 0.0
            ct = 0.0
        elif seas[i] != seas[i - 1]:
            cs *= delta
            ct *= delta
        S[i] = cs
        T[i] = ct
        cs = lam * cs + x[i]
        ct = lam * ct + toi[i]
    return S, T


def league_mean_prior(pg, xcol, group):
    """Walk-forward league mean rate (x per hour) by group, committed at date
    changes: for a row dated d, uses all rows with date < d."""
    d = pg.groupby([group, "date"]).agg(x=(xcol, "sum"), t=("toi_h", "sum")).reset_index()
    d = d.sort_values([group, "date"])
    d["cx"] = d.groupby(group).x.cumsum() - d.x
    d["ct"] = d.groupby(group).t.cumsum() - d.t
    # first date of a group: fall back to that date's own mean (2010-11 warm-up only)
    first = d.ct <= 0
    d.loc[first, "cx"] = d.loc[first, "x"]
    d.loc[first, "ct"] = d.loc[first, "t"]
    d["mu"] = d.cx / d.ct
    m = d.set_index([group, "date"]).mu
    return m.reindex(list(zip(pg[group], pg["date"]))).values


def eb_rate(pg, xcol, lam, delta, n0, group="pgrp", mu=None):
    """EB-shrunk walk-forward rate per hour: (S + n0*mu)/(T + n0)."""
    S, T = _ewma_prior(pg.pidx.values, pg.sidx.values, pg[xcol].values.astype(float),
                       pg.toi_h.values, lam, delta)
    if mu is None:
        mu = league_mean_prior(pg, xcol, group)
    return (S + n0 * mu) / (T + n0), mu, S, T


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


# ================================================================== driver
def prepare(allow_test=False, cutoff=None):
    """cutoff='YYYY-MM-DD' keeps only games dated strictly before it (walk-forward
    truncation audit: every value dated before the cutoff must be unchanged)."""
    t0 = time.time()
    ev, ro, games = load_base(allow_test)
    if cutoff is not None:
        games = games[games.date < cutoff].reset_index(drop=True)
        keep = set(games.gid)
        ev = ev[ev.gid.isin(keep)].reset_index(drop=True)
        ro = ro[ro.gid.isin(keep)].reset_index(drop=True)
    seasons = sorted(ev.season.unique().tolist())
    if not allow_test:
        assert max(seasons) <= 20172018
    vals, f, p, t = walk_forward_values(ev, games, seasons)
    print(f"values done {time.time()-t0:.0f}s", flush=True)
    pids = sorted(set(ro.pid.astype(np.int64)))
    pid_index = {p_: i for i, p_ in enumerate(pids)}
    return ev, ro, games, vals, f, p, t, pid_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["values", "tune", "build", "audit"])
    ap.add_argument("--cutoff", default="2015-01-15",
                    help="audit: truncate history at this date and compare")
    ap.add_argument("--allow-test", action="store_true",
                    help="LEAD ENGINEER ONLY: include TEST seasons (serving / TEST look)")
    a = ap.parse_args()
    if a.allow_test:
        print("WARNING: TEST seasons included -- lead-engineer mode", flush=True)
    import pv_nhl_duels_build as B  # noqa: E402  (driver kept in its own module)
    B.run(a.cmd, a.allow_test, cutoff=a.cutoff if a.cmd == "audit" else None)


if __name__ == "__main__":
    main()
