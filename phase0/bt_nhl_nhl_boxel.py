"""Breakthrough pick nhl_boxel -- DEV screen.  DEV ONLY, MARKET-BLIND.

Box-score production lineup value (a-priori Game Score weights, walk-forward
empirical-Bayes, carried across teams and seasons) plus the starting-goalie
level, both INSIDE the W/L Elo -- the public-model lineup sum (GSVA /
Evolving-Hockey style). Protocol: documents/breakthrough_program_prereg_2026_09_24.md

CONSTRUCTION (all constants fixed a priori, never fit)
  GSL  = 0.75 G + 0.65 A + 0.075 SOG + 0.05 BLK - 0.075 PIM   (per skater-game)
  S_p  = sum w GSL,  T_p = sum w TOI_hours over his prior games; w decays 0.99
         per game of that player; S,T x0.8 at every season boundary.
  v_p  = (S_p + mu_p T0) / (T_p + T0),  T0 = 10 h.
  mu_p = established (first game < 2011-07-01): position mean GSL/h over all
         strictly earlier skater-games; entrants (first game >= 2011-07-01) with
         < 82 games: walk-forward mean GSL/h of entrants' first 40 games, by
         position, from COMPLETED seasons (fallback 0.8 x position mean while
         that position's pool has < 2,000 games).
  vbar = TOI-weighted mean v over skaters who dressed in the previous completed
         season, per position, fixed at the season start; in 2010-11 (no
         completed season) a running value recomputed at each date start.
  m_p  = EWMA(0.8) of his own prior TOI minutes, initialised F 11 / D 16.
  L    = sum over tonight's dressed skaters (v_p - vbar_pos) m_p / 60
  Ltil = L / sd_L, sd_L = sd of all strictly earlier team-game L (date-batched).
  E    = (R_h + beta Ltil_h + gamma G_h + 30) - (R_a + beta Ltil_a + gamma G_a)
  R   += 8 (y - p) ; season regress 0.3      (shipped Elo core, not re-tuned)
  G    = starting-goalie level, goals/game (phase0/bt_nhl_boxel_gl.py).
  grid beta {0,15,30,45,60} x gamma {0,60,120,180}; logit of the new Elo
  probability REPLACES blend column 1; rest, b2b_h, b2b_a, xg_diff unchanged;
  blend refit per fold; cell chosen by NESTED inner LOSO (axis_a pair_loss).

ARMS  C0 shipped | C1 GL only | C2 L only | C3 GL + L inside (PRIMARY)
      C4 GL inside + dLtil beside | C5 on-ice process add-on (needs rapmel cache)

MODES
  default   full screen; needs data/bt_nhl_boxel_box.csv (>= 95% of DEV
            game-sides). If the fetch has not been run/approved, only the parts
            that need no boxscore (S1-S3, C1, GL walk-forward proofs) run and
            the JSON says PRIMARY NOT RUN.
  --smoke   plumbing test on a SYNTHETIC boxscore (random production on the
            nhl_dressed.csv lineups): runs the lineup pass and W1-W5 only; no
            log-loss of any L arm is computed. Writes data/bt_nhl_boxel_smoke.json.

Every scored mask passes nhl_depth_eval.assert_dev_only. No TEST-era game
(game_id >= 2018000000, season >= 2018-19) is loaded, rated, joined or scored.
Odds (data/odds_nhl.csv, rows dated < 2018-07-01) are read ONLY for the
evaluation-side model-minus-close diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, DEV_WARM_BEFORE, TEST_START, llv  # noqa: E402
from bt_nhl_anatomy_build import (MAX_GID, load_goalie_games,  # noqa: E402
                                  load_starters, norm)
import bt_nhl_boxel_gl as GL  # noqa: E402

OUT = "data/bt_nhl_nhl_boxel.json"
OUT_SMOKE = "data/bt_nhl_boxel_smoke.json"
BOX = "data/bt_nhl_boxel_box.csv"
assert DEV_END == 20172018 and TEST_START == 20182019 and MAX_GID == 2018000000

K_ELO, HA_ELO, REG_ELO = Hd.SHIPPED_ELO
BETAS = [0, 15, 30, 45, 60]
GAMMAS = [0, 60, 120, 180]
NB = 10000
SEED = 20260924
BAR = 0.00100

# lineup constants (a priori)
WG, WA, WSOG, WBLK, WPIM = 0.75, 0.65, 0.075, 0.05, 0.075
PDECAY, SCARRY, T0 = 0.99, 0.8, 10.0
M_DECAY = 0.8
M0 = {"F": 11.0, "D": 16.0}
ENT_DATE, ENT_MAXN, ENT_FIRST, ENT_MIN, ENT_FALLBACK = "2011-07-01", 82, 40, 2000, 0.8
SD_MIN = 50                 # team-games before sd_L is used (2010-11 warm only)
MIN_SK = 15                 # a box side with fewer skaters is treated as missing
DL_WIN = 10                 # C4: trailing team games for dLtil
LATE = (20152016, 20162017, 20172018)
CUTOFFS = ("2012-02-01", "2015-01-15", "2017-12-01")
W_ONICE = 0.15              # C5: Game Score's on-ice weight, applied to 5v5 xG
RAPMEL = "data/bt_nhl_rapmel_"


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------------------------ inputs ---
def load_box(dev_ids):
    """gid -> side -> {"sk": [(pid, pc, pos, G, A, SOG, BLK, PIM, toi_s)],
                        "gstart": pid|None, "goalies": [pid]}"""
    if not os.path.exists(BOX):
        return None, {"present": False}
    df = pd.read_csv(BOX)
    df = df[df.gid < MAX_GID]
    df = df[df.gid.isin(dev_ids)]
    box = defaultdict(dict)
    for r in df.itertuples(index=False):
        d = box[int(r.gid)].setdefault(r.side, {"sk": [], "gstart": None, "goalies": []})
        if int(r.is_goalie):
            d["goalies"].append(int(r.pid))
            if int(r.starter):
                d["gstart"] = int(r.pid)
        else:
            pc = "D" if str(r.pos) == "D" else "F"
            d["sk"].append((int(r.pid), pc, str(r.pos), int(r.G), int(r.A), int(r.SOG),
                            int(r.BLK), int(r.PIM), float(r.toi_s)))
    nsides = 2 * len(dev_ids)
    ok = sum(1 for gid in box for sd in box[gid].values() if len(sd["sk"]) >= MIN_SK)
    # data QA: a stat field missing from a season's payloads would read as all-zero
    sk = df[df.is_goalie == 0].copy()
    sk["season"] = sk.gid // 1000000
    qa = sk.groupby("season")[["G", "A", "SOG", "BLK", "PIM", "toi_s"]].mean().round(4)
    qa_d = {str(int(k)): v for k, v in qa.to_dict(orient="index").items()}
    zero = [(s_, c) for s_, row in qa_d.items() for c, v in row.items() if not v > 0]
    starters_per_side = df[df.is_goalie == 1].groupby(["gid", "side"]).starter.sum()
    info = {"present": True, "rows": int(len(df)), "games": len(box),
            "valid_sides": ok, "coverage": round(ok / nsides, 5),
            "per_season_skater_means": qa_d, "all_zero_stat_seasons": zero,
            "share_sides_exactly_one_starter": round(float((starters_per_side == 1).mean()), 5)
            if len(starters_per_side) else None}
    assert not zero, f"box QA: stat all-zero in a season {zero}"
    return box, info


def load_dressed_fb(games, goalie_ids):
    """gid -> side -> [pid] from data/nhl_dressed.csv (skaters only)."""
    side_of = {}
    for g in games:
        side_of[(g["game_id"], norm(g["home"]))] = "home"     # games carry raw PHX/ATL
        side_of[(g["game_id"], norm(g["away"]))] = "away"
    ids = {g["game_id"] for g in games}
    d = pd.read_csv("data/nhl_dressed.csv")
    d = d[d.game_id < MAX_GID]
    d = d[d.game_id.isin(ids)]
    d = d[~d.player_id.isin(goalie_ids)]
    out = defaultdict(dict)
    unmatched = 0
    for r in d.itertuples(index=False):
        sd = side_of.get((int(r.game_id), norm(r.team)))
        if sd is None:
            unmatched += 1
            continue
        out[int(r.game_id)].setdefault(sd, []).append(int(r.player_id))
    return out, unmatched


def load_onice(dev_ids):
    """C5 only: (gid, pid) -> on-ice 5v5 xGF - xGA (player's side), from pick
    nhl_rapmel's DEV-only stint cache. Post-game data: used only in updates."""
    seasons = sorted({int(str(g)[:4]) for g in dev_ids})
    out = {}
    for y0 in seasons:
        s_ = y0 * 10000 + y0 + 1
        f = f"{RAPMEL}{s_}_gid.npy"
        if not os.path.exists(f):
            return None
        gid = np.load(f)
        keep = (gid < MAX_GID) & np.isin(gid, list(dev_ids))
        gid = gid[keep]
        net = (np.load(f"{RAPMEL}{s_}_xgf.npy") - np.load(f"{RAPMEL}{s_}_xga.npy"))[keep]
        hi = np.load(f"{RAPMEL}{s_}_hi.npy")[keep]
        ai = np.load(f"{RAPMEL}{s_}_ai.npy")[keep]
        df = pd.DataFrame({"gid": np.r_[np.repeat(gid, 5), np.repeat(gid, 5)],
                           "pid": np.r_[hi.ravel(), ai.ravel()],
                           "v": np.r_[np.repeat(net, 5), np.repeat(-net, 5)]})
        df = df[df.pid > 0].groupby(["gid", "pid"]).v.sum()
        out.update({(int(a), int(b)): float(v) for (a, b), v in df.items()})
    return out


def synth_box(games, dfb, starters, seed=11):
    """SMOKE ONLY: random production on the real dressed lineups."""
    rng = np.random.default_rng(seed)
    box = defaultdict(dict)
    for g in games:
        gid = g["game_id"]
        for side, flag in (("home", 1), ("away", 0)):
            pids = dfb.get(gid, {}).get(side)
            if not pids:
                continue
            sk = []
            for p in pids:
                pc = "D" if (p * 2654435761) % 3 == 0 else "F"
                toi = float(rng.normal(20 if pc == "D" else 14, 3) * 60)
                sk.append((p, pc, pc, int(rng.poisson(0.15)), int(rng.poisson(0.25)),
                           int(rng.poisson(1.8)), int(rng.poisson(0.7)),
                           int(2 * rng.poisson(0.2)), max(toi, 0.0)))
            gs = starters.get(gid, {}).get(flag)
            box[gid][side] = {"sk": sk, "gstart": gs, "goalies": [gs] if gs else []}
    return box


# ------------------------------------------------------------ lineup pass ---
def lineup_pass(games, box, dfb, top_date="2016-01-01", onice=None):
    """One ordered read-then-update pass. Returns per-game L, Ltil, dLtil + diag.

    Same-game inputs read: tonight's dressed skater list (+ roster position).
    Tonight's G/A/SOG/BLK/PIM/TOI are folded into S, T, m only AFTER the read.
    League-level means (position means, sd_L, vbar in 2010-11) are date-batched.
    """
    n = len(games)
    Lh, La = np.zeros(n), np.zeros(n)
    Lth, Lta = np.zeros(n), np.zeros(n)
    dLh, dLa = np.zeros(n), np.zeros(n)
    S, T, N, M, FIRST, POS = {}, {}, {}, {}, {}, {}
    ps = {"F": [0.0, 0.0], "D": [0.0, 0.0]}           # sum GSL, sum hours (all prior)
    pool = {"F": [0.0, 0.0, 0], "D": [0.0, 0.0, 0]}   # entrants' first-40, completed seasons
    ebuf = {"F": [0.0, 0.0, 0], "D": [0.0, 0.0, 0]}   # current season
    pool_max_season = None
    wf = [0, 0.0, 0.0]                                 # welford n, mean, M2 of team-game L
    day_ps = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
    day_L = []
    last_absorbed = ""
    vbar = {"F": 0.0, "D": 0.0}
    vbar_fixed = False
    vbar_hist = {}
    season_toi = defaultdict(float)
    team_hist = defaultdict(lambda: deque(maxlen=DL_WIN))
    fb_sides = 0
    fb_players_nohist = 0
    top10 = None
    cur_date = None
    prev = None
    ts_L = defaultdict(list)                          # (season, team) -> [L]

    def pos_mean(pc):
        s_, h_ = ps[pc]
        return s_ / h_ if h_ > 0 else 0.0

    def ent_mu(pc):
        g_, h_, c_ = pool[pc]
        if c_ >= ENT_MIN and h_ > 0:
            return g_ / h_
        return ENT_FALLBACK * pos_mean(pc)

    def v_of(p, pc, date):
        f = FIRST.get(p, date)
        if f < ENT_DATE:
            mu = pos_mean(pc)
        elif N.get(p, 0) < ENT_MAXN:
            mu = ent_mu(pc)
        else:
            mu = pos_mean(pc)
        return (S.get(p, 0.0) + mu * T0) / (T.get(p, 0.0) + T0)

    def running_vbar(date):
        acc = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
        for p, w in season_toi.items():
            if w <= 0:
                continue
            pc = POS.get(p, "F")
            acc[pc][0] += w * v_of(p, pc, date)
            acc[pc][1] += w
        return {pc: (acc[pc][0] / acc[pc][1] if acc[pc][1] > 0 else pos_mean(pc))
                for pc in acc}

    for i, g in enumerate(games):
        s, date = g["season"], g["date"]
        # ------------- date boundary: absorb the previous date's league data ---
        if date != cur_date:
            if cur_date is not None:
                for pc in ("F", "D"):
                    ps[pc][0] += day_ps[pc][0]
                    ps[pc][1] += day_ps[pc][1]
                for x in day_L:
                    wf[0] += 1
                    dlt = x - wf[1]
                    wf[1] += dlt / wf[0]
                    wf[2] += dlt * (x - wf[1])
                last_absorbed = cur_date
            day_ps = {"F": [0.0, 0.0], "D": [0.0, 0.0]}
            day_L = []
            new_season = prev is not None and s != prev
            if new_season:
                for p in S:
                    S[p] *= SCARRY
                    T[p] *= SCARRY
                for pc in ("F", "D"):
                    for k in range(3):
                        pool[pc][k] += ebuf[pc][k]
                    ebuf[pc] = [0.0, 0.0, 0]
                pool_max_season = prev
                vbar = running_vbar(date)     # = TOI-weighted v over last season's skaters
                vbar_fixed = True
                vbar_hist[str(s)] = {k: r6(v) for k, v in vbar.items()}
                season_toi.clear()
            elif not vbar_fixed:
                vbar = running_vbar(date)     # 2010-11: running value, earlier dates only
            cur_date = date
            if top10 is None and date >= top_date:
                cand = [(v_of(p, POS.get(p, "F"), date), p) for p in season_toi
                        if season_toi[p] > 0 and N.get(p, 0) >= 20]
                cand.sort(reverse=True)
                top10 = [{"pid": p, "pos": POS.get(p), "v_gsl_per_h": r6(v),
                          "games": N.get(p, 0), "T_h": r6(T.get(p, 0.0))}
                         for v, p in cand[:10]]
            prev = s
        # W3: every walk-forward mean is strictly earlier
        assert last_absorbed < date
        assert pool_max_season is None or pool_max_season < s
        sd_L = np.sqrt(wf[2] / (wf[0] - 1)) if wf[0] >= SD_MIN else None
        gid = g["game_id"]
        gb = box.get(gid, {}) if box is not None else {}
        # ---------------------------- read ----------------------------
        for side, Larr, Ltarr, dLarr in (("home", Lh, Lth, dLh), ("away", La, Lta, dLa)):
            sd_rec = gb.get(side)
            if sd_rec is not None and len(sd_rec["sk"]) >= MIN_SK:
                roster = [(r[0], r[1]) for r in sd_rec["sk"]]
            else:
                fb_sides += 1
                roster = []
                for p in dfb.get(gid, {}).get(side, []):
                    if p not in POS:
                        fb_players_nohist += 1
                    roster.append((p, POS.get(p, "F")))
            L = 0.0
            for p, pc in roster:
                L += (v_of(p, pc, date) - vbar[pc]) * M.get(p, M0[pc]) / 60.0
            Larr[i] = L
            Lt = L / sd_L if sd_L else 0.0
            Ltarr[i] = Lt
            th = team_hist[g[side]]
            dLarr[i] = Lt - (sum(th) / len(th)) if th else 0.0
            if DEV_WARM_BEFORE <= s <= DEV_END:
                ts_L[(s, g[side])].append(L)
        # ---------------------------- update (after the read) ----------------
        for side, Ltarr in (("home", Lth), ("away", Lta)):
            day_L.append(Lh[i] if side == "home" else La[i])
            team_hist[g[side]].append(Ltarr[i])
            sd_rec = gb.get(side)
            if sd_rec is None or len(sd_rec["sk"]) < MIN_SK:
                continue
            for (p, pc, _pos, G_, A_, SOG, BLK, PIM, toi) in sd_rec["sk"]:
                gsl = WG * G_ + WA * A_ + WSOG * SOG + WBLK * BLK - WPIM * PIM
                if onice is not None:          # C5 only: post-game on-ice 5v5 net xG
                    gsl += W_ONICE * onice.get((gid, p), 0.0)
                h = toi / 3600.0
                S[p] = PDECAY * S.get(p, 0.0) + gsl
                T[p] = PDECAY * T.get(p, 0.0) + h
                N[p] = N.get(p, 0) + 1
                M[p] = M_DECAY * M.get(p, M0[pc]) + (1 - M_DECAY) * toi / 60.0
                if p not in FIRST:
                    FIRST[p] = date
                POS[p] = pc
                day_ps[pc][0] += gsl
                day_ps[pc][1] += h
                if FIRST[p] >= ENT_DATE and N[p] <= ENT_FIRST:
                    ebuf[pc][0] += gsl
                    ebuf[pc][1] += h
                    ebuf[pc][2] += 1
                season_toi[p] += h
    diag = {"fallback_sides": fb_sides, "fallback_players_without_history": fb_players_nohist,
            "vbar_by_season": vbar_hist, "top10_v_at_2016_01_01": top10,
            "entrant_pool_final": {pc: {"gsl_per_h": r6(pool[pc][0] / pool[pc][1])
                                        if pool[pc][1] else None, "games": pool[pc][2]}
                                   for pc in pool},
            "pos_mean_final": {pc: r6(pos_mean(pc)) for pc in ps},
            "sd_L_final": r6(np.sqrt(wf[2] / (wf[0] - 1))) if wf[0] > 1 else None}
    return {"Lh": Lh, "La": La, "Lth": Lth, "Lta": Lta, "dLh": dLh, "dLa": dLa,
            "ts_L": ts_L, "diag": diag}


# -------------------------------------------------------------------- Elo ---
def elo_cell(games, beta, gamma, Lth, Lta, Gh, Ga):
    """Shipped Elo core with lineup + goalie terms INSIDE the expectation."""
    R = {}
    out = np.empty(len(games))
    prev = None
    k, ha, regress = K_ELO, HA_ELO, REG_ELO
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - regress)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        E = (rh + beta * Lth[i] + gamma * Gh[i] + ha) - (ra + beta * Lta[i] + gamma * Ga[i])
        p = 1.0 / (1.0 + 10 ** (-E / 400.0))
        out[i] = p
        R[g["home"]] += k * (g["y"] - p)
        R[g["away"]] += k * ((1 - g["y"]) - (1 - p))
    return out


def logit_masked(p, mask):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))[mask]


# ------------------------------------------------------------ evaluation ---
class Evaluator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.y = ctx.yv()
        self.f = ctx.folds
        fl = self.f.list
        self.pairs = [(a, b) for i, a in enumerate(fl) for b in fl[i + 1:]]
        self.cells = {}

    def add(self, key, X):
        assert X.shape[0] == self.f.N and not np.isnan(X).any(), f"NaN/shape in {key}"
        f, y = self.f, self.y
        p = f.loso_pred(X, y)
        pl = {}
        w0 = None
        for a, b in self.pairs:
            tr = (f.seasons != a) & (f.seasons != b)
            w = Hd.fit_logit(X[tr], y[tr], Hd.BLEND_C, w0)
            w0 = w
            for s_, o in ((a, b), (b, a)):
                m = f.idx[o]
                pl[(s_, o)] = float(llv(y[m], Hd._sig(X[m] @ w)).mean())
        self.cells[key] = {"p": p, "pl": pl, "ll": f.pooled(y, p)}

    def nested(self, keys):
        f = self.f
        pred = np.empty(f.N)
        pick = {}
        for s_ in f.list:
            inner = [o for o in f.list if o != s_]
            den = sum(f.n[o] for o in inner)

            def score(c):
                return sum(f.n[o] * self.cells[c]["pl"][(s_, o)] for o in inner) / den
            c = min(keys, key=score)
            pick[s_] = c
            pred[f.idx[s_]] = self.cells[c]["p"][f.idx[s_]]
        best = min(keys, key=lambda c: self.cells[c]["ll"])
        return pred, pick, best


def boot(diff, seed=SEED, nb=NB):
    rng = np.random.default_rng(seed)
    n = len(diff)
    out = np.empty(nb)
    for j in range(0, nb, 500):
        m = min(500, nb - j)
        out[j:j + m] = diff[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def arm_report(ev, p_base, p_var, extra=None):
    y, f = ev.y, ev.f
    st = Hd.fold_stats(y, f, p_base, p_var)
    diff = llv(y, p_base) - llv(y, p_var)
    r = {"dev_ll": r6(f.pooled(y, p_var)), "gain": r6(diff.mean()),
         "boot_ci_10k": boot(diff), "fold_sd": st["fold_sd"], "fold_se": st["fold_se"],
         "sigma": st["sigma"], "n_pos_folds": st["n_pos_folds"], "n_folds": st["n_folds"],
         "per_season_gain": {str(s_): r6(diff[f.idx[s_]].mean()) for s_ in f.list}}
    late = np.isin(f.seasons, LATE)
    Hd.assert_dev_only(f.seasons[late])
    r["late_dev_2015_18"] = {"gain": r6(diff[late].mean()), "boot_ci_10k": boot(diff[late]),
                             "n": int(late.sum())}
    if extra:
        r.update(extra)
    return r


# ------------------------------------------------------------ diagnostics ---
def join_close(G):
    idx = defaultdict(list)
    for r in csv.DictReader(open("data/odds_nhl.csv", encoding="utf-8")):
        if r["date"] >= "2018-07-01":          # DEV era ends 2018-06; TEST never read
            continue
        idx[(r["date"], r["home"], r["away"])].append(r)
    pc = np.full(len(G), np.nan)
    for i, g in enumerate(G):
        m = idx.get((g["date"], g["home"], g["away"]))
        if not m or len(m) != 1:
            continue
        r = m[0]
        if int(r["home_win"]) != int(g["y"]):
            continue
        try:
            qh, qa = 1 / float(r["home_close"]), 1 / float(r["away_close"])
        except (ValueError, ZeroDivisionError):
            continue
        pc[i] = qh / (qh + qa)
    return pc


def close_diag(ctx, preds):
    """EVALUATION ONLY: model-minus-close LL on DEV slices (positive = model loses)."""
    G = [g for g, k in zip(ctx.games, ctx.mask) if k]
    y = ctx.yv()
    seas = ctx.seasons[ctx.mask]
    Hd.assert_dev_only(seas)
    pc = join_close(G)
    feats = pd.read_csv("data/bt_nhl_anatomy_feats.csv")
    feats = feats[feats.game_id < MAX_GID].set_index("game_id")
    gids = [g["game_id"] for g in G]
    fe = feats.reindex(gids)
    slices = {
        "overall": np.ones(len(G), bool),
        "non_no1_goalie": ((fe.g_notmodal_home == 1) | (fe.g_notmodal_away == 1)).to_numpy(),
        "top4_absent": ((fe.abs_top4_home >= 1) | (fe.abs_top4_away >= 1)).to_numpy(),
        "team_games_0_5": ((fe.gp_home <= 5) | (fe.gp_away <= 5)).to_numpy(),
    }
    ok = ~np.isnan(pc)
    lc = llv(y, np.where(ok, pc, 0.5))
    out = {"n_joined": int(ok.sum())}
    for nm, sl in slices.items():
        m = ok & sl
        Hd.assert_dev_only(seas[m])
        row = {"n": int(m.sum()), "close_ll": r6(lc[m].mean())}
        for arm, p in preds.items():
            row[f"{arm}_minus_close"] = r6((llv(y, p) - lc)[m].mean())
        out[nm] = row
    return out


def face_validity(ctx, ts_L):
    raw = {int(r["game_id"]): r for r in csv.DictReader(open("data/nhl_games.csv",
                                                             encoding="utf-8"))
           if int(r["game_id"]) < MAX_GID}
    gf = defaultdict(list)
    for g in ctx.games:
        s_ = g["season"]
        if not (DEV_WARM_BEFORE <= s_ <= DEV_END):
            continue
        r = raw[g["game_id"]]
        gf[(s_, g["home"])].append(float(r["home_goals"]))
        gf[(s_, g["away"])].append(float(r["away_goals"]))
    keys = [k for k in ts_L if k in gf]
    Hd.assert_dev_only([k[0] for k in keys])
    a = np.array([np.mean(ts_L[k]) for k in keys])
    b = np.array([np.mean(gf[k]) for k in keys])
    return {"n_team_seasons": len(keys), "corr_meanL_vs_GF_per_game": r6(np.corrcoef(a, b)[0, 1])}


# ------------------------------------------------------- walk-forward (W2) ---
def perturb_inputs(games, box, dfb, starters, ggames, cutoff, seed, full=True):
    """Scramble everything dated on/after `cutoff`.

    full=True : results, GA/xGA, box stats, dressed lists and starters.
    full=False: results, GA/xGA and box production/TOI only (dressed lists and
                starters kept) -> W5: only those two are same-game inputs.
    """
    rng = np.random.default_rng(seed)
    post = [i for i, g in enumerate(games) if g["date"] >= cutoff]
    pg = [dict(g) for g in games]
    ys = rng.permutation([games[i]["y"] for i in post])
    ys = 1 - ys if len(set(ys)) > 1 else ys          # guarantee a change
    for j, i in enumerate(post):
        pg[i]["y"] = int(ys[j])
    post_ids = [games[i]["game_id"] for i in post]
    # box
    nb = {gid: {sd: {"sk": list(v["sk"]), "gstart": v["gstart"], "goalies": list(v["goalies"])}
                for sd, v in box[gid].items()} for gid in box} if box is not None else {}
    keys = [(gid, sd) for gid in post_ids for sd in nb.get(gid, {})]
    if full and keys:
        vals = [nb[k[0]][k[1]] for k in keys]
        order = rng.permutation(len(vals))
        for k, j in zip(keys, order):
            nb[k[0]][k[1]] = vals[j]
    rows = [(k, j) for k in keys for j in range(len(nb[k[0]][k[1]]["sk"]))]
    stats = [nb[k[0]][k[1]]["sk"][j][3:] for k, j in rows]
    order = rng.permutation(len(stats))
    for (k, j), o in zip(rows, order):
        r = nb[k[0]][k[1]]["sk"][j]
        st = list(stats[o])
        st[-1] = st[-1] * 1.37 + 11.0                  # TOI changes for sure
        nb[k[0]][k[1]]["sk"][j] = r[:3] + tuple(st)
    # dressed fallback
    nd = {gid: {sd: list(v) for sd, v in dfb[gid].items()} for gid in dfb}
    if full:
        dk = [(gid, sd) for gid in post_ids for sd in nd.get(gid, {})]
        dv = [nd[k[0]][k[1]] for k in dk]
        for k, j in zip(dk, rng.permutation(len(dv))):
            nd[k[0]][k[1]] = dv[j]
    # starters
    ns = {gid: dict(v) for gid, v in starters.items()}
    if full:
        sk_ = [gid for gid in post_ids if gid in ns]
        sv = [ns[gid] for gid in sk_]
        for gid, j in zip(sk_, rng.permutation(len(sv))):
            ns[gid] = {1: sv[j].get(0), 0: sv[j].get(1)}
    # goalie GA/xGA
    ng = {gid: list(v) for gid, v in ggames.items()}
    gr = [(gid, j) for gid in post_ids if gid in ng for j in range(len(ng[gid]))]
    gv = [ng[gid][j][2:] for gid, j in gr]
    for (gid, j), o in zip(gr, rng.permutation(len(gv))):
        gk, team = ng[gid][j][:2]
        xga, ga = gv[o]
        ng[gid][j] = (gk, team, xga * 1.21 + 0.3, ga + 1.0)
    return pg, nb, nd, ns, ng


GMAR = "phase0/bt_nhl_gmar_gl.py"


def goalie_terms(games, starters, ggames):
    """Starting-goalie level G (goals/game) per game.

    Spec: import phase0/bt_nhl_gmar_gl.py (pick nhl_gmar's shared module) so that
    C1 equals gmar's A1 per game. Only if that module is absent does boxel fall
    back to its own stand-in build (phase0/bt_nhl_boxel_gl.py)."""
    if os.path.exists(GMAR):
        import bt_nhl_gmar_gl as GM
        r = GM.gl_elo(games, starters, ggames, 0.0, mode="new")
        sha = hashlib.sha1(open(GMAR, "rb").read()).hexdigest()[:16]
        return r["G_h"], r["G_a"], {
            "source": "bt_nhl_gmar_gl.gl_elo(mode='new') [pick nhl_gmar shared module]",
            "module_sha1_16": sha,
            "prev_starter_fallback_sides": int(r["fb_h"].sum() + r["fb_a"].sum())}
    Gh, Ga, info = GL.goalie_levels(games, starters, ggames)
    info["source"] = "bt_nhl_boxel_gl.goalie_levels [stand-in: gmar module absent]"
    return Gh, Ga, info


def features_all(games, box, dfb, starters, ggames, with_box=True):
    Gh, Ga, _ = goalie_terms(games, starters, ggames)
    if with_box:
        lp = lineup_pass(games, box, dfb)
        Lth, Lta = lp["Lth"], lp["Lta"]
        F = {"Lh": lp["Lh"], "La": lp["La"], "Lth": Lth, "Lta": Lta,
             "dLh": lp["dLh"], "dLa": lp["dLa"]}
    else:
        Lth = Lta = np.zeros(len(games))
        F = {}
    F["Gh"], F["Ga"] = Gh, Ga
    F["elo_30_120"] = elo_cell(games, 30, 120, Lth, Lta, Gh, Ga)
    F["elo_0_120"] = elo_cell(games, 0, 120, Lth, Lta, Gh, Ga)
    return F


def w2_w5(games, box, dfb, starters, ggames, with_box):
    base = features_all(games, box, dfb, starters, ggames, with_box)
    dates = np.array([g["date"] for g in games])
    res = {}
    for ci, cut in enumerate(CUTOFFS):
        pre = dates < cut
        upto = dates <= cut
        # W2: everything on/after the cutoff scrambled -> pre-cutoff bit-identical
        pert = features_all(*perturb_inputs(games, box, dfb, starters, ggames, cut,
                                            seed=100 + ci, full=True), with_box=with_box)
        ok2 = all(np.array_equal(base[k][pre], pert[k][pre]) for k in base)
        changed = {k: bool(not np.array_equal(base[k][~pre], pert[k][~pre])) for k in base}
        # W5: only production/results/GA scrambled -> identical THROUGH the cutoff date
        pert5 = features_all(*perturb_inputs(games, box, dfb, starters, ggames, cut,
                                             seed=200 + ci, full=False), with_box=with_box)
        ok5 = all(np.array_equal(base[k][upto], pert5[k][upto]) for k in base)
        res[cut] = {"W2_pre_cutoff_bit_identical": bool(ok2),
                    "W2_post_cutoff_changed": changed,
                    "W5_through_cutoff_date_identical": bool(ok5),
                    "n_pre": int(pre.sum()), "n_on_cutoff_date": int((dates == cut).sum())}
        assert ok2, f"W2 FAILED at {cut}"
        assert ok5, f"W5 FAILED at {cut}"
    return res


# ------------------------------------------------------------------ sanity ---
def s4a_s4b(games, box, dfb, starters):
    eq = tot = 0
    st_eq = st_tot = 0
    for g in games:
        gid = g["game_id"]
        for side, flag in (("home", 1), ("away", 0)):
            b = box.get(gid, {}).get(side)
            if b is None or len(b["sk"]) < MIN_SK:
                continue
            d = dfb.get(gid, {}).get(side)
            if d is not None:
                tot += 1
                eq += int(set(r[0] for r in b["sk"]) == set(d))
            fs = starters.get(gid, {}).get(flag)
            if fs is not None and b["gstart"] is not None:
                st_tot += 1
                st_eq += int(fs == b["gstart"])
    return ({"match_share": r6(eq / tot) if tot else None, "n_sides": tot,
             "pass": bool(tot and eq / tot >= 0.99)},
            {"match_share": r6(st_eq / st_tot) if st_tot else None, "n_sides": st_tot,
             "pass": bool(st_tot and st_eq / st_tot >= 0.995)})


def s4c_check(games, starters, ggames, Gh, Ga):
    """C1 cells must equal gmar's own gl_elo probabilities per game (all gammas)."""
    if not os.path.exists(GMAR):
        return {"status": "n/a: phase0/bt_nhl_gmar_gl.py absent; C1 uses boxel's stand-in"}
    import bt_nhl_gmar_gl as GM
    z = np.zeros(len(games))
    d = 0.0
    for gm in GAMMAS:
        pg = GM.gl_elo(games, starters, ggames, float(gm), mode="new")["p"]
        d = max(d, float(np.max(np.abs(pg - elo_cell(games, 0, gm, z, z, Gh, Ga)))))
    out = {"status": "compared per game vs bt_nhl_gmar_gl.gl_elo, gamma in GAMMAS",
           "max_abs_diff_p": d, "pass": d <= 1e-12}
    return out


def gmar_a1_ll():
    try:
        d = json.load(open("data/bt_nhl_gmar.json", encoding="utf-8"))
        for k, v in d.get("arms", {}).items():
            if k.startswith("A1"):
                return k, v.get("dev_ll"), v.get("gain")
    except Exception:  # noqa: BLE001
        pass
    return None, None, None


# -------------------------------------------------------------------- main ---
def redact(o):
    """smoke-full: keep structure + booleans + strings, hide every number."""
    if isinstance(o, dict):
        return {k: redact(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [redact(v) for v in o]
    if isinstance(o, (bool, np.bool_)) or isinstance(o, str) or o is None:
        return o if not isinstance(o, np.bool_) else bool(o)
    return "#" if np.isfinite(float(o)) else "NONFINITE"


def mask_hash(m):
    return hashlib.sha1(np.asarray(m, bool).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-full", action="store_true",
                    help="run the FULL pipeline on a synthetic boxscore; every number "
                         "is redacted before writing data/bt_nhl_boxel_smoke_full.json")
    args = ap.parse_args()
    t0 = time.time()
    ctx = Hd.Ctx()
    games = ctx.games
    assert max(g["season"] for g in games) <= DEV_END
    ids = {g["game_id"] for g in games}
    assert max(ids) < MAX_GID
    y = ctx.yv()
    folds = ctx.folds
    Hd.assert_dev_only(ctx.seasons[ctx.mask])
    MH = mask_hash(ctx.mask)

    starters = load_starters(ids)
    ggames, goalie_ids = load_goalie_games(ids)
    dfb, dfb_unmatched = load_dressed_fb(games, goalie_ids)
    print(f"[{time.time()-t0:.0f}s] inputs: starters {len(starters)}, goalie-games "
          f"{len(ggames)}, dressed {len(dfb)} (unmatched rows {dfb_unmatched})", flush=True)

    # ------------------------------------------------------------ smoke ---
    if args.smoke:
        box = synth_box(games, dfb, starters)
        lp = lineup_pass(games, box, dfb)
        m = ctx.mask
        s4a, s4b = s4a_s4b(games, box, dfb, starters)
        W = w2_w5(games, box, dfb, starters, ggames, with_box=True)
        nan_ok = not any(np.isnan(lp[k][m]).any() for k in ("Lth", "Lta", "dLh", "dLa"))
        out = {"mode": "SMOKE (synthetic production on real lineups; NO log-loss computed)",
               "W2_W5": W, "W4_no_nan_masked": nan_ok, "mask_hash": MH,
               "S4a_synthetic": s4a, "S4b_synthetic": s4b,
               "lineup_diag": lp["diag"], "sd_Ltil_masked": r6(np.std(lp["Lth"][m])),
               "seconds": round(time.time() - t0, 1)}
        json.dump(out, open(OUT_SMOKE, "w", encoding="utf-8"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "lineup_diag"}, indent=1))
        print(f"wrote {OUT_SMOKE}")
        return

    rep = {"pick": "nhl_boxel", "protocol": {
        "prereg": "documents/breakthrough_program_prereg_2026_09_24.md",
        "dev_seasons_scored": [DEV_WARM_BEFORE, DEV_END], "test_seasons_touched": 0,
        "eval": "leave-one-season-out over 2011-12..2017-18, nested cell selection",
        "market_blind": True, "bar": f"nested C3 gain >= {BAR} and 10k-boot CI lo > 0"}}

    # ------------------------------------------------------ S1 / S2 -----
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    ll0 = folds.pooled(y, p0)
    print(f"S1 shipped DEV LOSO {ll0:.6f} n={folds.N}", flush=True)
    s1 = {"ll": r6(ll0), "n": folds.N, "pass": bool(folds.N == 7929 and abs(ll0 - 0.672398) < 2e-6)}
    p_nx = folds.loso_pred(X0[:, :-1], y)
    cost = folds.pooled(y, p_nx) - ll0
    s2 = {"drop_xg_cost": r6(cost), "expected": 0.00303, "pass": bool(abs(cost - 0.00303) <= 0.0002)}
    print(f"S2 dropping xg_diff costs {cost:+.5f}", flush=True)
    assert s1["pass"], "S1 shipped harness not reproduced"
    assert s2["pass"], "S2 known effect not reproduced"

    # ------------------------------------------------------ inputs ------
    Gh, Ga, ginfo = goalie_terms(games, starters, ggames)
    print(f"goalie term: {ginfo['source']}", flush=True)
    if args.smoke_full:
        box = synth_box(games, dfb, starters)
        binfo = {"present": "SYNTHETIC (smoke-full)"}
        have_box = True
    else:
        box, binfo = load_box(ids)
        have_box = bool(box is not None and binfo.get("coverage", 0) >= 0.95)
    rep["box_info"] = binfo
    rep["goalie_info"] = ginfo
    if have_box:
        lp = lineup_pass(games, box, dfb)
        Lth, Lta, dLh, dLa = lp["Lth"], lp["Lta"], lp["dLh"], lp["dLa"]
    else:
        lp = None
        Lth = Lta = dLh = dLa = np.zeros(len(games))

    # ------------------------------------------------------ S3 ----------
    ref = Hd.run_elo_arr(games, K_ELO, HA_ELO, REG_ELO)
    e00 = elo_cell(games, 0, 0, Lth, Lta, Gh, Ga)
    s3d = float(np.max(np.abs(ref - e00)))
    s3 = {"max_abs_diff": s3d, "pass": s3d <= 1e-12}
    assert s3["pass"], "S3 (0,0) cell != run_elo_arr"
    print(f"S3 (0,0) vs run_elo_arr max|diff| {s3d:.2e}", flush=True)

    base = ctx.base_cols()
    xgc = ctx.xg(*Hd.SHIPPED_XG)
    m = ctx.mask
    ev = Evaluator(ctx)

    def X_cell(b, gm, beside=False):
        el = logit_masked(elo_cell(games, b, gm, Lth, Lta, Gh, Ga), m)
        cols = [el] + base + [xgc]
        if beside:
            cols = cols + [(dLh - dLa)[m]]
        return Hd.design(cols)

    # the (0,0) cell must be the shipped design exactly
    X00 = X_cell(0, 0)
    assert np.array_equal(X00, X0), "(0,0) cell design != shipped design"

    arms = {}
    # ------------------------------------------------------ C1 GL only --
    for gm in GAMMAS:
        ev.add(("C", 0, gm), X_cell(0, gm))
    c1_keys = [("C", 0, gm) for gm in GAMMAS]
    pC1, pick1, best1 = ev.nested(c1_keys)
    arms["C1_GL_only"] = arm_report(ev, p0, pC1, {
        "nested_pick": {str(s_): {"gamma": c[2]} for s_, c in pick1.items()},
        "optimistic_best": {"gamma": best1[2], "ll": r6(ev.cells[best1]["ll"]),
                            "gain": r6(ll0 - ev.cells[best1]["ll"])},
        "cell_ll": {f"gamma={c[2]}": r6(ev.cells[c]["ll"]) for c in c1_keys}})
    if not args.smoke_full:
        print(f"C1 GL-only nested gain {arms['C1_GL_only']['gain']:+.5f} "
              f"CI {arms['C1_GL_only']['boot_ci_10k']}", flush=True)
    assert abs(ev.cells[("C", 0, 0)]["ll"] - ll0) < 1e-12

    s4c = s4c_check(games, starters, ggames, Gh, Ga)
    k_a1, ll_a1, gain_a1 = gmar_a1_ll()
    s4c["gmar_json_arm"] = k_a1
    s4c["gmar_A1_dev_ll"] = ll_a1
    s4c["boxel_C1_dev_ll"] = arms["C1_GL_only"]["dev_ll"]
    if ll_a1 is not None:
        s4c["A1_vs_C1_ll_absdiff"] = r6(abs(ll_a1 - arms["C1_GL_only"]["dev_ll"]))
    if not args.smoke_full:
        print(f"S4c {s4c}", flush=True)
    if os.path.exists(GMAR):
        # DISCLOSED: before the gmar module existed this screen's first run used a
        # stand-in GL build (phase0/bt_nhl_boxel_gl.py); its C1 is reported here,
        # never used by C3 and never gating.
        Gh_b, Ga_b, _ = GL.goalie_levels(games, starters, ggames)
        for gm in GAMMAS[1:]:
            el = logit_masked(elo_cell(games, 0, gm, Lth * 0, Lta * 0, Gh_b, Ga_b), m)
            ev.add(("S", 0, gm), Hd.design([el] + base + [xgc]))
        sk = [("C", 0, 0)] + [("S", 0, gm) for gm in GAMMAS[1:]]
        pS, pickS, bestS = ev.nested(sk)
        arms["C1b_standin_boxel_GL_disclosed"] = arm_report(ev, p0, pS, {
            "note": "boxel's stand-in goalie build, run before bt_nhl_gmar_gl.py existed; "
                    "diagnostic only",
            "nested_pick": {str(s_): {"gamma": c[2]} for s_, c in pickS.items()}})
    sanity = {"S1": s1, "S2": s2, "S3": s3, "S4c": s4c}
    walk = {"W1": "one ordered pass per module (lineup_pass, gmar gl_elo, elo_cell); "
                  "state read into tonight's features BEFORE tonight's box stats/result/"
                  "GA are folded in; league-level means date-batched",
            "W3": "asserted in-loop: last absorbed date < tonight for every league mean "
                  "(pos means, sd_L, running vbar, league goalie rate, xbar); entrant "
                  "pools and fixed vbar come from seasons < current season",
            "W4_mask_hash": MH}

    if not have_box:
        W = w2_w5(games, None, dfb, starters, ggames, with_box=False)
        walk["W2_W5_goalie_and_elo_only"] = W
        walk["W4"] = {"mask_hash_all_arms": MH,
                      "no_nan_masked": bool(not (np.isnan(Gh[m]).any() or np.isnan(Ga[m]).any())),
                      "goalie_info": ginfo}
        rep["close_diag_eval_only"] = close_diag(ctx, {"C0": p0, "C1": pC1})
        rep.update({
            "status": "PRIMARY NOT RUN -- data/bt_nhl_boxel_box.csv absent or < 95% "
                      "coverage. The boxscore fetch (phase0/bt_nhl_boxel_fetch.py, ~9,371 "
                      "public api-web.nhle.com JSON documents) requires the user's OK and "
                      "was NOT run. Only S1-S3, C1 (GL only) and the goalie/Elo "
                      "walk-forward proofs ran.",
            "sanity": sanity, "walk_forward": walk, "arms": arms,
            "bar": {"eligible": False, "reason": "primary C3 not run"},
            "seconds": round(time.time() - t0, 1)})
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"wrote {OUT} (PRIMARY NOT RUN: no boxscore data)")
        return

    # ------------------------------------------------------ full screen --
    for b in BETAS:
        for gm in GAMMAS:
            if ("C", b, gm) not in ev.cells:
                ev.add(("C", b, gm), X_cell(b, gm))
    for gm in GAMMAS:
        ev.add(("B", 0, gm), X_cell(0, gm, beside=True))
    print(f"[{time.time()-t0:.0f}s] {len(ev.cells)} cells evaluated", flush=True)

    c2_keys = [("C", b, 0) for b in BETAS]
    c3_keys = [("C", b, gm) for b in BETAS for gm in GAMMAS]
    c4_keys = [("B", 0, gm) for gm in GAMMAS]
    pC2, pick2, best2 = ev.nested(c2_keys)
    pC3, pick3, best3 = ev.nested(c3_keys)
    pC4, pick4, best4 = ev.nested(c4_keys)

    def picks(pk):
        return {str(s_): {"beta": c[1], "gamma": c[2]} for s_, c in pk.items()}

    def bestd(c):
        return {"beta": c[1], "gamma": c[2], "ll": r6(ev.cells[c]["ll"]),
                "gain": r6(ll0 - ev.cells[c]["ll"])}

    arms["C0_shipped"] = {"dev_ll": r6(ll0), "n": folds.N}
    arms["C2_L_only"] = arm_report(ev, p0, pC2, {"nested_pick": picks(pick2),
                                                 "optimistic_best": bestd(best2)})
    arms["C3_GL_plus_L_inside_PRIMARY"] = arm_report(ev, p0, pC3, {
        "nested_pick": picks(pick3), "optimistic_best": bestd(best3),
        "cell_ll": {f"b{c[1]}_g{c[2]}": r6(ev.cells[c]["ll"]) for c in c3_keys}})
    arms["C4_GL_inside_dLtil_beside"] = arm_report(ev, p0, pC4, {
        "nested_pick": picks(pick4), "optimistic_best": bestd(best4)})
    # inside vs beside, paired on identical games
    arms["C3_vs_C4_inside_minus_beside"] = {"gain": r6((llv(y, pC4) - llv(y, pC3)).mean()),
                                           "boot_ci_10k": boot(llv(y, pC4) - llv(y, pC3))}
    arms["C3_vs_C1_lineup_increment"] = {"gain": r6((llv(y, pC1) - llv(y, pC3)).mean()),
                                        "boot_ci_10k": boot(llv(y, pC1) - llv(y, pC3))}
    onice = load_onice(ids)
    if onice is None:
        arms["C5_onice_process"] = {"status": "not run: pick nhl_rapmel stint cache absent"}
    else:
        lp5 = lineup_pass(games, box, dfb, onice=onice)
        L5h, L5a = lp5["Lth"], lp5["Lta"]
        c5_keys = []
        for b in BETAS:
            for gm in GAMMAS:
                if b == 0:
                    c5_keys.append(("C", 0, gm))        # beta=0 cells are identical
                    continue
                el = logit_masked(elo_cell(games, b, gm, L5h, L5a, Gh, Ga), m)
                ev.add(("P", b, gm), Hd.design([el] + base + [xgc]))
                c5_keys.append(("P", b, gm))
        pC5, pick5, best5 = ev.nested(c5_keys)
        arms["C5_onice_process"] = arm_report(ev, p0, pC5, {
            "note": "optional diagnostic, never gating: GSL += 0.15 x on-ice 5v5 "
                    "(xGF - xGA) per game from nhl_rapmel's DEV stint cache",
            "nested_pick": picks(pick5), "optimistic_best": bestd(best5),
            "players_games_with_onice": len(onice)})
    for k, v in arms.items():
        if isinstance(v, dict) and "gain" in v and "sigma" in v:
            if args.smoke_full:
                continue
            print(f"  {k:<32} gain {v['gain']:+.6f} CI {v['boot_ci_10k']} "
                  f"({v['sigma']:+.2f} SE, {v['n_pos_folds']}/{v['n_folds']})", flush=True)

    # ------------------------------------------------------ sanity S4 ---
    s4a, s4b = s4a_s4b(games, box, dfb, starters)
    sanity["S4a_dressed_vs_nhl_dressed"] = s4a
    sanity["S4b_box_starter_vs_first_shot"] = s4b
    fv = face_validity(ctx, lp["ts_L"])
    fv["pass"] = bool(fv["corr_meanL_vs_GF_per_game"] > 0)
    fv["top10_v_at_2016_01_01"] = lp["diag"]["top10_v_at_2016_01_01"]
    try:
        names = json.load(open("data/nhl_player_names.json", encoding="utf-8"))
        for r in fv["top10_v_at_2016_01_01"] or []:
            r["name"] = names.get(str(r["pid"]), {}).get("name")
    except Exception:  # noqa: BLE001
        pass
    sanity["S4d_face_validity"] = fv

    # ------------------------------------------------------ W2/W4/W5 -----
    W = w2_w5(games, box, dfb, starters, ggames, with_box=True)
    walk["W2_W5"] = W
    nan_ok = not any(np.isnan(a[m]).any() for a in (Lth, Lta, dLh, dLa, Gh, Ga))
    walk["W4"] = {"mask_hash_all_arms": MH, "no_nan_masked": nan_ok,
                  "fallback_sides_nhl_dressed": lp["diag"]["fallback_sides"],
                  "fallback_players_without_history":
                      lp["diag"]["fallback_players_without_history"],
                  "goalie_info": ginfo}
    walk["W5"] = ("same-game inputs = tonight's dressed skater list (with roster position) "
                  "and tonight's first-shot starter; W5 test scrambles results, GA/xGA and "
                  "box production/TOI from the cutoff date on with lineups kept and asserts "
                  "features THROUGH the cutoff date are bit-identical")
    assert nan_ok

    # ------------------------------------------------------ close diag ---
    rep["close_diag_eval_only"] = close_diag(ctx, {"C0": p0, "C1": pC1, "C3": pC3})

    # ------------------------------------------------------ comparisons --
    cmp_ = {"rule": "within 0.0002 nats -> simpler wins: gmar > boxel > rapmel"}
    c3ll = arms["C3_GL_plus_L_inside_PRIMARY"]["dev_ll"]
    for pick_, f in (("nhl_gmar", "data/bt_nhl_gmar.json"),
                     ("nhl_rapmel", "data/bt_nhl_rapmel.json")):
        if not os.path.exists(f):
            cmp_[pick_] = "n/a (no result file)"
            continue
        try:
            v = json.load(open(f, encoding="utf-8")).get("verdict", {})
            sib = v.get("candidate_dev_ll")
            cmp_[pick_] = {"primary_dev_ll": sib, "boxel_C3_dev_ll": c3ll,
                           "boxel_minus_sibling": r6(c3ll - sib) if sib is not None else None}
        except Exception as ex:  # noqa: BLE001
            cmp_[pick_] = f"present, unreadable: {ex!r}"
    rep["sibling_comparison"] = cmp_

    c3 = arms["C3_GL_plus_L_inside_PRIMARY"]
    all_ok = (s1["pass"] and s2["pass"] and s3["pass"] and s4a["pass"] and s4b["pass"]
              and fv["pass"] and nan_ok)
    rep["bar"] = {"nested_gain": c3["gain"], "ci": c3["boot_ci_10k"],
                  "sanity_and_walk_forward_pass": bool(all_ok),
                  "clears": bool(c3["gain"] >= BAR and c3["boot_ci_10k"][0] > 0 and all_ok)}
    rep.update({"status": "COMPLETE", "sanity": sanity, "walk_forward": walk, "arms": arms,
                "lineup_diag": {k: v for k, v in lp["diag"].items()
                                if k != "top10_v_at_2016_01_01"},
                "seconds": round(time.time() - t0, 1)})
    if args.smoke_full:
        red = redact(rep)
        red["mode"] = ("SMOKE-FULL: full pipeline on SYNTHETIC production; every "
                       "number redacted (plumbing test only, carries no result)")
        json.dump(red, open("data/bt_nhl_boxel_smoke_full.json", "w", encoding="utf-8"),
                  indent=1)
        print("wrote data/bt_nhl_boxel_smoke_full.json (redacted)", flush=True)
        return
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}  bar clears: {rep['bar']['clears']}", flush=True)


if __name__ == "__main__":
    main()
