"""Breakthrough program (NFL) - candidate nfl_clinchx: exact 'nothing left to play for' flag. DEV ONLY.

BUILDER. Late-season rest flag computed with the NFL tie-breaking procedure on
date-strict standings:

  targets      REG games of seasons 2002-2015 in weeks L-2..L (L = 17 through
               2020, 18 from 2021 -> DEV weeks 15-17)
  info set     REG games of the same season with gameday STRICTLY before the
               target's gameday (Thu/Sat results count for Sun/Mon games; same-day
               results never count). One Monte Carlo per (season, gameday) group.
  ratings      plain Elo (k=47.43, hfa=52.14, 0 at neutral sites, season regression
               .3648 toward 1500, W/L updates, ties .5) walked from 1999; each team's
               post-game rating after its last game before the group date.
  MC           NS=4000 completions of every same-season REG game not in the info
               set (targets included); simulated games cannot tie; real ties = .5.
               RNG numpy default_rng(20261001), ONE stream, groups chronological;
               per group the draws are U(NS x n_rem) then coin(NS x 32).
  seeding      NFL procedure per conference and sim (see seed_conf):
               division ties: H2H pct, division pct, common-games pct, conference
               pct, SOV, SOS, coin; on any elimination the remaining clubs restart
               at step 1 (two-club procedure when two remain).
               wildcards / division-winner order: best club per division first
               (division procedure, original within-division order kept), then
               H2H (sweep only for 3+; plain H2H pct for two if they met),
               conference pct, common games (min 4), SOV, SOS, coin; after each
               club is placed the rest restart at step 1.
               Point-based steps are unavailable (scores not simulated) -> coin.
               Format: through 2019 4 DW + 2 WC, byes seeds 1-2; from 2020 4 + 3,
               bye seed 1.
  per side     P(PO); leverage = |P(bye|win i) - P(bye|lose i)| from each sim's own
               result of game i, 0 unless both branches have > 30 sims (proxy rule);
               locked = P(PO) >= .999 AND leverage < .02 (thresholds inherited from
               the proxy, not tuned).

MARKET-BLIND: only the columns listed in NEED are taken from data/nfl_games.csv;
no betting column is referenced. Only seasons <= 2015 are loaded (asserted).

Usage (run from repo root):
  python phase0/bt_nfl_clinchx_build.py --unittest   # tiebreak unit test (actual 2002-15 + synthetic 2020 format)
  python phase0/bt_nfl_clinchx_build.py --build      # MC, resumable per season -> flags/groups
  python phase0/bt_nfl_clinchx_build.py --perturb    # future-perturbation leak test (30 groups)
Outputs: data/bt_nfl_clinchx_flags.npy, data/bt_nfl_clinchx_groups.json,
         data/bt_nfl_clinchx_unittest.json, data/bt_nfl_clinchx_perturb.json,
         data/bt_nfl_clinchx_cache/season_<s>.json (per-season cache)
"""
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

T0 = time.time()
DEV_MAX = 2015
TEST_ERA = 2016
SIM_LO = 2002
NS = 4000
SEED = 20261001
K_, HFA_, REG_ = 47.43351932834238, 52.14162888646703, 0.3647684062154459
PPO_LOCK, LEV_LOCK, MIN_BRANCH = 0.999, 0.02, 30
EPS = 1e-9
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
DIV = {"AFC East": "BUF MIA NE NYJ", "AFC North": "BAL CIN CLE PIT",
       "AFC South": "HOU IND JAX TEN", "AFC West": "DEN KC LV LAC",
       "NFC East": "DAL NYG PHI WAS", "NFC North": "CHI DET GB MIN",
       "NFC South": "ATL CAR NO TB", "NFC West": "ARI LA SF SEA"}
TDIV = {t: d for d, ts in DIV.items() for t in ts.split()}
TEAMS = sorted(TDIV); TIX = {t: k for k, t in enumerate(TEAMS)}; NT = len(TEAMS)
CONF = [TDIV[t][:3] for t in TEAMS]; DIVN = [TDIV[t] for t in TEAMS]
DIVS = sorted(DIV)
CACHE_DIR = "data/bt_nfl_clinchx_cache"
NEED = ("game_id", "season", "game_type", "week", "gameday", "gametime", "home_team",
        "away_team", "home_score", "away_score", "location", "div_game")


def last_week(s):
    return 17 if s <= 2020 else 18


def fmt_of(s):
    """(number of wildcards, number of byes) per conference."""
    return (2, 2) if s <= 2019 else (3, 1)


# ------------------------------------------------------------------ data ----
def load_games():
    out = {}
    with open("data/nfl_games.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
        for r in rd:
            sv = r[ix["season"]]
            if not sv or int(sv) > DEV_MAX:
                continue
            row = {c: r[ix[c]] for c in NEED}
            out[row["game_id"]] = row
    assert max(int(v["season"]) for v in out.values()) <= DEV_MAX
    return out


RAWG = load_games()
D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]; N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() <= DEV_MAX
WK = np.array([g["week"] for g in G]); TYP = np.array([g["type"] for g in G])
assert len(RAWG) == N and {g["gid"] for g in G} == set(RAWG)
GD = []
HS0 = np.zeros(N); AS0 = np.zeros(N)
for i, g in enumerate(G):
    r = RAWG[g["gid"]]
    assert FR.get(r["home_team"], r["home_team"]) == g["home"] and FR.get(r["away_team"], r["away_team"]) == g["away"]
    assert int(r["season"]) == g["season"] and int(r["week"]) == g["week"] and r["game_type"] == g["type"]
    assert (r["location"] == "Neutral") == bool(g["neutral"])
    GD.append(r["gameday"])
    HS0[i] = float(r["home_score"]); AS0[i] = float(r["away_score"])
GD = np.array(GD)

# per-team game list in spine order: gameday must be strictly increasing
TEAM_GAMES = defaultdict(list)            # team -> [(gameday, spine idx, side)]
for i, g in enumerate(G):
    TEAM_GAMES[g["home"]].append((GD[i], i, 0)); TEAM_GAMES[g["away"]].append((GD[i], i, 1))
for t, lst in TEAM_GAMES.items():
    days = [x[0] for x in lst]
    assert all(days[k] < days[k + 1] for k in range(len(days) - 1)), f"{t} spine order not chronological"
TEAM_DAYS = {t: [x[0] for x in lst] for t, lst in TEAM_GAMES.items()}


def elo_walk(HS, AS):
    """plain Elo, identical to the ELO_POST walk of bt_nfl_ideas_quick2.py."""
    elo = defaultdict(lambda: 1500.0); post = np.zeros((N, 2)); prev = None
    for i, g in enumerate(G):
        if prev is not None and g["season"] != prev:
            for t in list(elo):
                elo[t] = 1500.0 + (1 - REG_) * (elo[t] - 1500.0)
        prev = g["season"]
        h, a = g["home"], g["away"]
        dr = elo[h] - elo[a] + (0.0 if g["neutral"] else HFA_)
        e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
        elo[h] += K_ * (yy - e); elo[a] -= K_ * (yy - e)
        post[i] = (elo[h], elo[a])
    return post


# --------------------------------------------------------------- engine ----
class SeasonCtx:
    """static schedule structures of one season's REG games (no outcomes)."""

    def __init__(self, s, synthetic=None):
        if synthetic is None:
            assert SIM_LO <= s <= DEV_MAX, s
            idx = [i for i in range(N) if SEAS[i] == s and TYP[i] == "REG"]
            self.idx = np.array(idx)
            self.h = np.array([TIX[G[i]["home"]] for i in idx]); self.a = np.array([TIX[G[i]["away"]] for i in idx])
            self.day = GD[self.idx]; self.neutral = np.array([bool(G[i]["neutral"]) for i in idx])
            dg = np.array([RAWG[G[i]["gid"]]["div_game"] == "1" for i in idx])
        else:                                  # synthetic schedule: list of (h, a) team indices
            self.idx = None
            self.h = np.array([x[0] for x in synthetic]); self.a = np.array([x[1] for x in synthetic])
            self.day = None; self.neutral = np.zeros(len(synthetic), bool); dg = None
        self.s = s; self.ng = ng = len(self.h)
        self.nwc, self.nbye = fmt_of(s)
        self.isdiv = np.array([DIVN[x] == DIVN[y] for x, y in zip(self.h, self.a)])
        self.isconf = np.array([CONF[x] == CONF[y] for x, y in zip(self.h, self.a)])
        if dg is not None:
            assert (dg == self.isdiv).all(), "division map disagrees with div_game"
        self.Hinc = np.zeros((ng, NT)); self.Hinc[np.arange(ng), self.h] = 1.0
        self.Ainc = np.zeros((ng, NT)); self.Ainc[np.arange(ng), self.a] = 1.0
        HG = np.zeros((NT, NT)); np.add.at(HG, (self.h, self.a), 1.0); np.add.at(HG, (self.a, self.h), 1.0)
        self.HG = HG; self.HGl = HG.tolist()
        self.Gt = HG.sum(1)
        self.divG = (self.Hinc * self.isdiv[:, None]).sum(0) + (self.Ainc * self.isdiv[:, None]).sum(0)
        self.confG = (self.Hinc * self.isconf[:, None]).sum(0) + (self.Ainc * self.isconf[:, None]).sum(0)
        self.OPP = [frozenset(np.nonzero(HG[t])[0].tolist()) for t in range(NT)]
        self.HA = np.zeros((ng, NT * NT)); self.HA[np.arange(ng), self.h * NT + self.a] = 1.0
        self.AH = np.zeros((ng, NT * NT)); self.AH[np.arange(ng), self.a * NT + self.h] = 1.0
        self.conf_teams = {cf: [t for t in range(NT) if CONF[t] == cf] for cf in ("AFC", "NFC")}
        self.div_teams = {d: [t for t in range(NT) if DIVN[t] == d] for d in DIVS}
        self.conf_divs = {cf: [d for d in DIVS if d.startswith(cf)] for cf in ("AFC", "NFC")}

    # ---- vectorised stats from full-season home points R (n, ng)
    def stats(self, R):
        W = R @ self.Hinc + (1 - R) @ self.Ainc
        dW = (R * self.isdiv) @ self.Hinc + ((1 - R) * self.isdiv) @ self.Ainc
        cW = (R * self.isconf) @ self.Hinc + ((1 - R) * self.isconf) @ self.Ainc
        return W, dW, cW

    def sim_detail(self, R, W):
        """H2H points matrix, SOV, SOS for the given sims (n, ng)."""
        H2P = (R @ self.HA + (1 - R) @ self.AH).reshape(-1, NT, NT)
        sov = np.einsum("nij,nj->ni", H2P, W) / np.maximum(np.einsum("nij,j->ni", H2P, self.Gt), 1e-12)
        sos = (W @ self.HG.T) / (self.HG @ self.Gt)
        return H2P, sov, sos


class Sim:
    __slots__ = ("pct", "dpct", "cpct", "h2p", "sov", "sos", "coin")


def _best(S, vals):
    m = max(vals)
    return [t for t, v in zip(S, vals) if v >= m - EPS]


class Engine:
    def __init__(self, ctx):
        self.c = ctx

    # -- steps: each returns a list of values aligned with S, or None (not applicable)
    def v_h2h(self, S, X):
        HG = self.c.HGl; out = []
        for i in S:
            g_ = sum(HG[i][j] for j in S if j != i)
            if g_ == 0:
                return None
            out.append(sum(X.h2p[i][j] for j in S if j != i) / g_)
        return out

    def v_common(self, S, X, minimum):
        HG = self.c.HGl; Sset = set(S)
        C = frozenset.intersection(*[self.c.OPP[i] for i in S]) - Sset
        out = []
        for i in S:
            g_ = sum(HG[i][j] for j in C)
            if g_ == 0 or g_ < minimum:
                return None
            out.append(sum(X.h2p[i][j] for j in C) / g_)
        return out

    def div_best(self, S, X, log):
        S = list(S)
        while len(S) > 1:
            n0 = len(S)
            for name, fn in (("d_h2h", lambda S_: self.v_h2h(S_, X)),
                             ("d_div", lambda S_: [X.dpct[t] for t in S_]),
                             ("d_common", lambda S_: self.v_common(S_, X, 0)),
                             ("d_conf", lambda S_: [X.cpct[t] for t in S_]),
                             ("d_sov", lambda S_: [X.sov[t] for t in S_]),
                             ("d_sos", lambda S_: [X.sos[t] for t in S_])):
                vals = fn(S)
                if vals is None:
                    continue
                S2 = _best(S, vals)
                if len(S2) < n0:
                    log[name] += 1; S = S2
                    break
            else:
                log["d_coin"] += 1
                S = [max(S, key=lambda t: X.coin[t])]
        return S[0]

    def div_rank(self, S, X, log):
        S = list(S); order = []
        while S:
            b = self.div_best(S, X, log) if len(S) > 1 else S[0]
            order.append(b); S.remove(b)
        return order

    def sweep(self, S, X):
        HG = self.c.HGl
        for i in S:
            others = [j for j in S if j != i]
            if all(HG[i][j] > 0 for j in others):
                if all(X.h2p[i][j] >= HG[i][j] - EPS for j in others):
                    return [i]
        for i in S:
            others = [j for j in S if j != i]
            if all(HG[i][j] > 0 for j in others) and all(X.h2p[i][j] <= EPS for j in others):
                return others
        return S

    def wc_best(self, S, X, log):
        """S: clubs from different divisions."""
        S = list(S)
        assert len({DIVN[t] for t in S}) == len(S)
        while len(S) > 1:
            n0 = len(S)
            if n0 == 2:
                steps = (("w_h2h", lambda S_: self.v_h2h(S_, X)),)
            else:
                steps = (("w_sweep", None),)
            steps = steps + (("w_conf", lambda S_: [X.cpct[t] for t in S_]),
                             ("w_common4", lambda S_: self.v_common(S_, X, 4)),
                             ("w_sov", lambda S_: [X.sov[t] for t in S_]),
                             ("w_sos", lambda S_: [X.sos[t] for t in S_]))
            for name, fn in steps:
                if fn is None:
                    S2 = self.sweep(S, X)
                else:
                    vals = fn(S)
                    if vals is None:
                        continue
                    S2 = _best(S, vals)
                if len(S2) < n0:
                    log[name] += 1; S = S2
                    break
            else:
                log["w_coin"] += 1
                S = [max(S, key=lambda t: X.coin[t])]
        return S[0]

    def pick_k(self, cands, k, X, log, reduce_div):
        sel = []; rem = list(cands)
        while len(sel) < k and rem:
            m = max(X.pct[t] for t in rem)
            T0 = [t for t in rem if X.pct[t] >= m - EPS]
            slots = k - len(sel)
            if len(T0) <= slots:
                sel.extend(T0); rem = [t for t in rem if t not in T0]
                continue
            bydiv = defaultdict(list)
            for t in T0:
                bydiv[DIVN[t]].append(t)
            ranks = {d: (self.div_rank(ts, X, log) if (reduce_div and len(ts) > 1) else ts)
                     for d, ts in bydiv.items()}
            if not reduce_div:
                assert all(len(v) == 1 for v in ranks.values())
            T = list(T0)
            for _ in range(slots):
                reps = [next(t for t in ranks[d] if t in T) for d in ranks if any(t in T for t in ranks[d])]
                b = reps[0] if len(reps) == 1 else self.wc_best(reps, X, log)
                sel.append(b); T.remove(b)
            rem = [t for t in rem if t not in sel]
        return sel

    def seed_conf(self, cf, X, log):
        """returns (division winners, wildcards, bye clubs) for conference cf."""
        dws = []
        for d in self.c.conf_divs[cf]:
            T = self.c.div_teams[d]
            m = max(X.pct[t] for t in T)
            top = [t for t in T if X.pct[t] >= m - EPS]
            dws.append(top[0] if len(top) == 1 else self.div_best(top, X, log))
        byes = self.pick_k(dws, self.c.nbye, X, log, reduce_div=False)
        pool = [t for t in self.c.conf_teams[cf] if t not in dws]
        wcs = self.pick_k(pool, self.c.nwc, X, log, reduce_div=True)
        return dws, wcs, byes


def make_sim(ctx, pct, dpct, cpct, h2p, sov, sos, coin):
    X = Sim()
    X.pct = pct; X.dpct = dpct; X.cpct = cpct; X.h2p = h2p; X.sov = sov; X.sos = sos; X.coin = coin
    return X


def seed_all(ctx, eng, R, coin, want_detail=False):
    """R (n, ng) full-season home points (0/.5/1). Returns PO, BYE (n, NT) bool, and diagnostics.
    Vectorised where no win-pct tie sits at a position that matters; engine otherwise."""
    n = R.shape[0]
    W, dW, cW = ctx.stats(R)
    pct = W / ctx.Gt; dpct = dW / ctx.divG; cpct = cW / ctx.confG
    PO = np.zeros((n, NT), bool); BYE = np.zeros((n, NT), bool)
    slow_any = np.zeros(n, bool); slow_by_conf = {}
    for cf in ("AFC", "NFC"):
        dws = []; uniq = np.ones(n, bool)
        for d in ctx.conf_divs[cf]:
            tm = np.array(ctx.div_teams[d]); pd_ = pct[:, tm]
            top = pd_.max(1)
            uniq &= (pd_ >= top[:, None] - EPS).sum(1) == 1
            dws.append(tm[pd_.argmax(1)])
        dws = np.stack(dws, 1)                                       # (n, 4)
        dpv = np.take_along_axis(pct, dws, 1)
        dsort = -np.sort(-dpv, 1)
        nb = ctx.nbye
        bye_amb = (np.abs(dsort[:, nb - 1] - dsort[:, nb]) < EPS) if nb < 4 else np.zeros(n, bool)
        ct = np.array(ctx.conf_teams[cf])
        isdw = np.zeros((n, NT), bool); np.put_along_axis(isdw, dws, True, axis=1)
        poolv = np.where(isdw[:, ct], -np.inf, pct[:, ct])
        order = np.argsort(-poolv, axis=1, kind="stable")
        psort = np.take_along_axis(poolv, order, 1)
        nw = ctx.nwc
        wc_amb = np.abs(psort[:, nw - 1] - psort[:, nw]) < EPS
        slow = ~uniq | bye_amb | wc_amb
        fast = ~slow
        # fast path
        f = np.where(fast)[0]
        if len(f):
            po_f = isdw[f].copy()
            wcs = ct[order[f, :nw]]
            np.put_along_axis(po_f, wcs, True, axis=1)
            byeo = np.argsort(-dpv[f], axis=1, kind="stable")[:, :nb]
            bt = np.take_along_axis(dws[f], byeo, 1)
            by_f = np.zeros((len(f), NT), bool); np.put_along_axis(by_f, bt, True, axis=1)
            cmask = np.zeros(NT, bool); cmask[ct] = True
            PO[f] |= po_f & cmask; BYE[f] |= by_f & cmask
        slow_by_conf[cf] = slow
        slow_any |= slow
    log = Counter()
    sl = np.where(slow_any)[0]
    detail = {}
    if len(sl):
        eng_ = eng
        for a0 in range(0, len(sl), 1000):
            ch = sl[a0:a0 + 1000]
            H2P, sov, sos = ctx.sim_detail(R[ch], W[ch])
            for k_, sidx in enumerate(ch):
                X = make_sim(ctx, pct[sidx].tolist(), dpct[sidx].tolist(), cpct[sidx].tolist(),
                             H2P[k_].tolist(), sov[k_].tolist(), sos[k_].tolist(), coin[sidx].tolist())
                for cf in ("AFC", "NFC"):
                    if not slow_by_conf[cf][sidx]:
                        continue
                    dws_, wcs_, byes_ = eng_.seed_conf(cf, X, log)
                    PO[sidx, dws_ + wcs_] = True; BYE[sidx, byes_] = True
                    if want_detail:
                        detail[(int(sidx), cf)] = (dws_, wcs_, byes_)
    diag = {"n_slow": {cf: int(v.sum()) for cf, v in slow_by_conf.items()}, "steps": dict(log)}
    return PO, BYE, diag, detail


# ---------------------------------------------------------------- groups ----
def target_groups():
    """(season, gameday) -> spine indices of target games."""
    grp = defaultdict(list)
    for i in range(N):
        s = int(SEAS[i])
        if s < SIM_LO or TYP[i] != "REG":
            continue
        L = last_week(s)
        if L - 2 <= WK[i] <= L:
            grp[(s, GD[i])].append(i)
    return dict(sorted(grp.items()))


def ratings_asof(s, d, post):
    """each team's post-game Elo after its last game with gameday < d."""
    rat = np.zeros(NT); src = {}
    for t in TEAMS:
        days = TEAM_DAYS[t]
        k = bisect.bisect_left(days, d) - 1          # last index with day < d
        assert k >= 0
        dd, i, side = TEAM_GAMES[t][k]
        assert dd < d and (k + 1 == len(days) or days[k + 1] >= d)
        assert SEAS[i] == s, (t, s, d)
        rat[TIX[t]] = post[i, side]; src[t] = int(i)
    return rat, src


def run_group(ctx, d, targets, HS, AS, post, rng, eng):
    s = ctx.s
    info = ctx.day < d
    rem = ~info
    ii = ctx.idx[info]
    assert (SEAS[ii] == s).all() and (GD[ii] < d).all()
    assert all(GD[i] == d for i in targets) and all(i in set(ctx.idx[rem].tolist()) for i in targets)
    known = np.full(ctx.ng, np.nan)
    known[info] = np.where(HS[ii] > AS[ii], 1.0, np.where(HS[ii] == AS[ii], 0.5, 0.0))
    rat, src = ratings_asof(s, d, post)
    rh = rat[ctx.h[rem]]; ra = rat[ctx.a[rem]]
    dr = rh - ra + np.where(ctx.neutral[rem], 0.0, HFA_)
    p = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    state0 = copy.deepcopy(rng.bit_generator.state)
    U = rng.random((NS, int(rem.sum())))
    coin = rng.random((NS, NT))
    hw = (U < p).astype(float)
    R = np.tile(known, (NS, 1)); R[:, rem] = hw
    PO, BYE, diag, _ = seed_all(ctx, eng, R, coin)
    remidx = ctx.idx[rem].tolist(); col = {i: k for k, i in enumerate(remidx)}
    sides = []
    for i in targets:
        c = col[i]; won_h = hw[:, c] == 1.0
        for k_, t in enumerate((G[i]["home"], G[i]["away"])):
            ti = TIX[t]; w = won_h if k_ == 0 else ~won_h
            nw = int(w.sum())
            ppo = float(PO[:, ti].mean()); pby = float(BYE[:, ti].mean())
            lev = float(abs(BYE[w, ti].mean() - BYE[~w, ti].mean())) if MIN_BRANCH < nw < NS - MIN_BRANCH else 0.0
            sides.append({"i": int(i), "gid": G[i]["gid"], "week": int(WK[i]), "side": k_, "team": t,
                          "ppo": ppo, "pbye": pby, "lev": lev, "n_win": nw,
                          "locked": int(ppo >= PPO_LOCK and lev < LEV_LOCK)})
    return {"season": s, "date": d, "n_info": int(info.sum()), "n_rem": int(rem.sum()),
            "rng_state": state0, "diag": diag, "sides": sides}


# -------------------------------------------------------------- unit test ----
def truth_sets(s):
    rows = [r for r in RAWG.values() if int(r["season"]) == s]
    wc = [r for r in rows if r["game_type"] == "WC"]; dv = [r for r in rows if r["game_type"] == "DIV"]
    f = lambda x: FR.get(x, x)  # noqa: E731
    in_wc = {f(r["home_team"]) for r in wc} | {f(r["away_team"]) for r in wc}
    in_dv = {f(r["home_team"]) for r in dv} | {f(r["away_team"]) for r in dv}
    po = in_wc | in_dv; bye = in_dv - in_wc; dw = bye | {f(r["home_team"]) for r in wc}
    return po, bye, dw


def unittest():
    out = {"seasons": {}, "pass": True}
    for s in range(SIM_LO, DEV_MAX + 1):
        assert s <= DEV_MAX
        ctx = SeasonCtx(s); eng = Engine(ctx)
        ii = ctx.idx
        R = np.where(HS0[ii] > AS0[ii], 1.0, np.where(HS0[ii] == AS0[ii], 0.5, 0.0))[None, :]
        coin = np.random.default_rng(7).random((1, NT))
        W, dW, cW = ctx.stats(R)
        H2P, sov, sos = ctx.sim_detail(R, W)
        X = make_sim(ctx, (W[0] / ctx.Gt).tolist(), (dW[0] / ctx.divG).tolist(), (cW[0] / ctx.confG).tolist(),
                     H2P[0].tolist(), sov[0].tolist(), sos[0].tolist(), coin[0].tolist())
        log = Counter(); po = set(); bye = set(); dw = set()
        for cf in ("AFC", "NFC"):
            d_, w_, b_ = Engine(ctx).seed_conf(cf, X, log)
            po |= {TEAMS[t] for t in d_ + w_}; bye |= {TEAMS[t] for t in b_}; dw |= {TEAMS[t] for t in d_}
        # vectorised/hybrid path must agree with the pure engine
        PO, BYE, diag, _ = seed_all(ctx, eng, R, coin)
        hy_po = {TEAMS[t] for t in np.where(PO[0])[0]}; hy_bye = {TEAMS[t] for t in np.where(BYE[0])[0]}
        tpo, tbye, tdw = truth_sets(s)
        ok = (po == tpo) and (bye == tbye) and (dw == tdw) and hy_po == po and hy_bye == bye
        coin_hit = log.get("d_coin", 0) + log.get("w_coin", 0)
        out["seasons"][str(s)] = {"ok": ok, "steps": dict(log), "coin_reached": coin_hit,
                                  "po_miss": sorted(tpo ^ po), "bye_miss": sorted(tbye ^ bye),
                                  "dw_miss": sorted(tdw ^ dw), "hybrid_agrees": hy_po == po and hy_bye == bye}
        out["pass"] &= ok
        print(f"  {s}: {'OK ' if ok else 'FAIL'} steps={dict(log)} coin={coin_hit} "
              f"po^={sorted(tpo ^ po)} bye^={sorted(tbye ^ bye)} dw^={sorted(tdw ^ dw)}", flush=True)
    # ---- synthetic tests (no data from seasons >= 2016 is ever read)
    syn = {}
    assert fmt_of(2019) == (2, 2) and fmt_of(2020) == (3, 1) and last_week(2020) == 17 and last_week(2021) == 18
    # (1) 2020+ format branch on a synthetic season: the 2012 SCHEDULE (no outcome data
    # beyond 2015 is read) with hand-set, strictly distinct win pcts -> expected field by sorting;
    # and hybrid (vectorised) path vs pure engine on a synthetic full outcome.
    base = SeasonCtx(2012)
    pairs = list(zip(base.h.tolist(), base.a.tolist()))
    res = {}
    for sy in (2019, 2020):
        cx = SeasonCtx(sy, synthetic=pairs); en = Engine(cx)
        pv = [(k * 7 % NT + 1) / 40.0 for k in range(NT)]          # distinct
        assert len(set(pv)) == NT
        z = [0.0] * NT
        X = make_sim(cx, pv, z, z, [[0.0] * NT for _ in range(NT)], z, z, z)
        nwc, nb = fmt_of(sy); got_po, got_bye, exp_po, exp_bye = set(), set(), set(), set()
        for cf in ("AFC", "NFC"):
            d_, w_, b_ = en.seed_conf(cf, X, Counter())
            got_po |= set(d_ + w_); got_bye |= set(b_)
            dws = [max(cx.div_teams[d], key=lambda t: pv[t]) for d in cx.conf_divs[cf]]
            pool = sorted([t for t in cx.conf_teams[cf] if t not in dws], key=lambda t: -pv[t])
            exp_po |= set(dws) | set(pool[:nwc]); exp_bye |= set(sorted(dws, key=lambda t: -pv[t])[:nb])
        # hybrid vs pure engine on 200 random synthetic full outcomes (ties at every position)
        Rr = (np.random.default_rng(11).random((200, cx.ng)) < 0.5).astype(float)
        cn = np.random.default_rng(12).random((200, NT))
        PO, BYE, _, _ = seed_all(cx, en, Rr, cn)
        W, dW, cW = cx.stats(Rr); H2P, sov, sos = cx.sim_detail(Rr, W); agree = True
        for n_ in range(200):
            Xn = make_sim(cx, (W[n_] / cx.Gt).tolist(), (dW[n_] / cx.divG).tolist(), (cW[n_] / cx.confG).tolist(),
                          H2P[n_].tolist(), sov[n_].tolist(), sos[n_].tolist(), cn[n_].tolist())
            po_e, by_e = set(), set()
            for cf in ("AFC", "NFC"):
                d_, w_, b_ = en.seed_conf(cf, Xn, Counter()); po_e |= set(d_ + w_); by_e |= set(b_)
            agree &= po_e == set(np.where(PO[n_])[0].tolist()) and by_e == set(np.where(BYE[n_])[0].tolist())
            agree &= len(po_e) == 2 * (4 + nwc) and len(by_e) == 2 * nb
        res[sy] = {"n_po": len(got_po), "n_bye": len(got_bye), "ok": got_po == exp_po and got_bye == exp_bye,
                   "hybrid_equals_engine_200": bool(agree)}
    syn["format_2019_vs_2020"] = res
    assert res[2019]["n_po"] == 12 and res[2019]["n_bye"] == 4 and res[2020]["n_po"] == 14 and res[2020]["n_bye"] == 2
    # (2) hand-built tie cases on the 2012 schedule (engine semantics)
    cx = SeasonCtx(2012); en = Engine(cx)

    def mk(pct, dpct=None, cpct=None, h2p=None, sov=None, sos=None, coin=None):
        z = [0.0] * NT
        return make_sim(cx, pct, dpct or z, cpct or z, h2p or [[0.0] * NT for _ in range(NT)],
                        sov or z, sos or z, coin or z)
    # two-club division tie decided by H2H
    A, B = TIX["NE"], TIX["MIA"]
    h2p = [[0.0] * NT for _ in range(NT)]; h2p[A][B] = 2.0
    X = mk([0.5] * NT, h2p=h2p); lg = Counter()
    ok1 = en.div_best([A, B], X, lg) == A and lg["d_h2h"] == 1
    # three-club division: H2H eliminates one, the remaining two restart at two-club H2H (1-1) -> division pct
    A, B, Cc = TIX["NE"], TIX["MIA"], TIX["NYJ"]
    h2p = [[0.0] * NT for _ in range(NT)]
    h2p[A][B] = 1; h2p[B][A] = 1; h2p[A][Cc] = 2; h2p[Cc][A] = 0; h2p[B][Cc] = 2; h2p[Cc][B] = 0
    dp = [0.0] * NT; dp[B] = 0.9; dp[A] = 0.5
    X = mk([0.5] * NT, dpct=dp, h2p=h2p); lg = Counter()
    ok2 = en.div_best([A, B, Cc], X, lg) == B and lg["d_h2h"] == 1 and lg["d_div"] == 1
    # wildcard, 3 clubs, all games between them tied (no sweep) -> conference pct removes one,
    # the two left restart at two-club H2H
    HGl = cx.HGl
    A, B, Cc = TIX["DEN"], TIX["PIT"], TIX["HOU"]
    h2p = [[0.5 * HGl[i][j] for j in range(NT)] for i in range(NT)]
    cp = [0.0] * NT; cp[A] = 0.6; cp[B] = 0.7; cp[Cc] = 0.7
    X = mk([0.5] * NT, cpct=cp, h2p=h2p); lg = Counter()
    r3 = en.wc_best([A, B, Cc], X, lg)
    ok3 = r3 in (B, Cc) and lg["w_conf"] == 1 and lg.get("w_sweep", 0) == 0
    # wildcard sweep: a triple of AFC clubs from three divisions that all met; one beat both
    trip = None
    afc = cx.conf_teams["AFC"]
    for x in afc:
        for y in afc:
            for w in afc:
                if len({DIVN[x], DIVN[y], DIVN[w]}) == 3 and x < y < w and HGl[x][y] and HGl[x][w] and HGl[y][w]:
                    trip = trip or (x, y, w)
    ok4 = None
    if trip:
        x, y, w = trip
        h2p = [[0.0] * NT for _ in range(NT)]
        h2p[w][x] = HGl[w][x]; h2p[w][y] = HGl[w][y]; h2p[x][y] = HGl[x][y]
        cp = [0.0] * NT; cp[x] = 0.9                               # conf pct would favour x
        X = mk([0.5] * NT, cpct=cp, h2p=h2p); lg = Counter()
        ok4 = en.wc_best([x, y, w], X, lg) == w and lg["w_sweep"] == 1
    syn["hand_cases"] = {"div2_h2h": ok1, "div3_h2h_then_restart": ok2, "wc3_conf_then_restart": ok3,
                         "wc3_sweep": ok4, "sweep_triple": [TEAMS[t] for t in trip] if trip else None}
    syn["pass"] = bool(res[2019]["ok"] and res[2020]["ok"] and res[2019]["hybrid_equals_engine_200"]
                       and res[2020]["hybrid_equals_engine_200"] and ok1 and ok2 and ok3 and ok4 is not False)
    out["synthetic"] = syn
    out["pass"] = bool(out["pass"] and syn["pass"])
    print("  synthetic:", syn, flush=True)
    json.dump(out, open("data/bt_nfl_clinchx_unittest.json", "w"), indent=1)
    print(f"unit test pass={out['pass']} -> data/bt_nfl_clinchx_unittest.json", flush=True)
    return out


# ------------------------------------------------------------------ build ----
def build():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ut = json.load(open("data/bt_nfl_clinchx_unittest.json"))
    assert ut["pass"], "tiebreak unit test must pass before building"
    post = elo_walk(HS0, AS0)
    groups = target_groups()
    rng = np.random.default_rng(SEED)
    for s in range(SIM_LO, DEV_MAX + 1):
        assert s < TEST_ERA
        fn = f"{CACHE_DIR}/season_{s}.json"
        if os.path.exists(fn):
            c = json.load(open(fn))
            rng.bit_generator.state = c["rng_state_end"]
            print(f"[{time.time()-T0:.0f}s] {s} cached", flush=True)
            continue
        ctx = SeasonCtx(s); eng = Engine(ctx)
        recs = []
        for (s_, d), tg in groups.items():
            if s_ != s:
                continue
            t1 = time.time()
            r = run_group(ctx, d, tg, HS0, AS0, post, rng, eng)
            recs.append(r)
            nl = sum(x["locked"] for x in r["sides"])
            print(f"[{time.time()-T0:.0f}s] {s} {d} rem={r['n_rem']} slow={r['diag']['n_slow']} "
                  f"locked={nl} ({time.time()-t1:.1f}s)", flush=True)
        tmp = fn + ".tmp"
        json.dump({"season": s, "groups": recs, "rng_state_end": rng.bit_generator.state}, open(tmp, "w"))
        os.replace(tmp, fn)
    # assemble
    flags = np.zeros((N, 2)); allg = []
    for s in range(SIM_LO, DEV_MAX + 1):
        c = json.load(open(f"{CACHE_DIR}/season_{s}.json"))
        for r in c["groups"]:
            allg.append(r)
            for x in r["sides"]:
                flags[x["i"], x["side"]] = x["locked"]
    assert flags[SEAS < SIM_LO].sum() == 0 and flags[TYP != "REG"].sum() == 0
    np.save("data/bt_nfl_clinchx_flags.npy", flags)
    json.dump({"note": "per group and side: P(PO), P(bye), leverage, locked; rng_state = state at group start",
               "NS": NS, "seed": SEED, "thresholds": {"ppo": PPO_LOCK, "lev": LEV_LOCK, "min_branch_gt": MIN_BRANCH},
               "groups": allg}, open("data/bt_nfl_clinchx_groups.json", "w"))
    print(f"[{time.time()-T0:.0f}s] flags {flags.shape} locked sides {int(flags.sum())} -> "
          "data/bt_nfl_clinchx_flags.npy, data/bt_nfl_clinchx_groups.json", flush=True)


def perturb(n_groups=30):
    """randomise every score on/after the group date, recompute Elo walk + MC from the saved
    RNG state: flags (and P(PO), leverage) must be bit-identical."""
    GJ = json.load(open("data/bt_nfl_clinchx_groups.json"))["groups"]
    pick = np.random.default_rng(424242).choice(len(GJ), n_groups, replace=False)
    groups = target_groups()
    res = []; allok = True
    for gi in sorted(pick.tolist()):
        g0 = GJ[gi]; s, d = g0["season"], g0["date"]
        prng = np.random.default_rng(1000 + gi)
        fut = GD >= d
        HS = HS0.copy(); AS = AS0.copy()
        HS[fut] = prng.integers(0, 45, fut.sum()); AS[fut] = prng.integers(0, 45, fut.sum())
        post = elo_walk(HS, AS)
        ctx = SeasonCtx(s); eng = Engine(ctx)
        rng = np.random.default_rng(SEED); rng.bit_generator.state = g0["rng_state"]
        r = run_group(ctx, d, groups[(s, d)], HS, AS, post, rng, eng)
        same = all(a["locked"] == b["locked"] and a["ppo"] == b["ppo"] and a["lev"] == b["lev"]
                   and a["pbye"] == b["pbye"] for a, b in zip(r["sides"], g0["sides"])) \
            and len(r["sides"]) == len(g0["sides"])
        changed = int((HS[fut] != HS0[fut]).sum())
        allok &= same
        res.append({"season": s, "date": d, "n_future_scores_changed": changed, "identical": same})
        print(f"  {s} {d} future scores changed {changed}: identical={same}", flush=True)
    # positive control (the test must be able to fail): perturb the scores of the same season's
    # games strictly BEFORE the group date -> standings/ratings change -> P(PO)/leverage should move
    moved = 0
    for gi in sorted(pick.tolist()):
        g0 = GJ[gi]; s, d = g0["season"], g0["date"]
        prng = np.random.default_rng(2000 + gi)
        past = (GD < d) & (SEAS == s) & (TYP == "REG")
        HS = HS0.copy(); AS = AS0.copy()
        HS[past] = prng.integers(0, 45, past.sum()); AS[past] = prng.integers(0, 45, past.sum())
        post = elo_walk(HS, AS)
        ctx = SeasonCtx(s); eng = Engine(ctx)
        rng = np.random.default_rng(SEED); rng.bit_generator.state = g0["rng_state"]
        r = run_group(ctx, d, groups[(s, d)], HS, AS, post, rng, eng)
        moved += int(any(a["ppo"] != b["ppo"] or a["lev"] != b["lev"] for a, b in zip(r["sides"], g0["sides"])))
    print(f"positive control: past-score perturbation moved P(PO)/leverage in {moved}/{len(pick)} groups", flush=True)
    json.dump({"pass": bool(allok), "groups": res, "positive_control_groups_moved": moved,
               "positive_control_n": int(len(pick))}, open("data/bt_nfl_clinchx_perturb.json", "w"), indent=1)
    print(f"perturbation test pass={allok}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unittest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--perturb", action="store_true")
    a = ap.parse_args()
    if a.unittest:
        unittest()
    if a.build:
        build()
    if a.perturb:
        perturb()
