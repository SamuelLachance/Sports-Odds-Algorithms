"""bt_nhl_rapmel -- lineup power P and the ratings that carry it INSIDE (DEV only).

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md),
candidate nhl_rapmel. Feature module; the screen is phase0/bt_nhl_nhl_rapmel.py.

Lineup power (STEP 3 of the spec)
  m5_p   EWMA of the player's own prior 5v5 minutes per game (any team, across
         seasons): m <- 0.8 m + 0.2 x, initialised at his first game with shift
         data (x = 5v5 seconds / 60 from data/bt_nhl_rapmel_toi5.csv; 0 if he was
         dressed but logged no 5v5 time).
  prior  a player with no history gets the walk-forward mean 5v5 minutes of
         debutants' first 5 games by position (debutant = first appearance in the
         data on/after 2011-07-01; 2010-11 players are left-censored veterans),
         committed only at date changes (strictly earlier dates); F 10.0 / D 14.0
         until that position has 200 debutant games.
  v_p    value from the latest values file whose cutoff < game date
         (data/bt_nhl_rapmel_values.csv); a player absent from that fit gets the
         fit's v_repl,pos; position = the fit's label, else the name-map label,
         else F. Before the first cutoff (2010-11, never scored) v = 0 for all.
  P(T,t) = sum over tonight's dressed skaters of v_p * m5_p / 60  (team 5v5 net
         xG per game implied by tonight's lineup, goal units).
  dP     P_t minus the mean P of the team's previous 10 dressed lineups, all
         re-valued with the values and m5 current at t (window crosses seasons).

Inside the ratings (STEP 4)
  W/L Elo: bt_nhl_gmar_gl.gl_elo with d = -P and delta = gamma_P, i.e.
           E = (R_h + gamma*G_h + gamma_P*P_h + 30) - (R_a + gamma*G_a + gamma_P*P_a).
  xG-Elo : xg_beta() mirrors nhl_depth_eval.run_xg_arr line by line with
           eh = X_h + beta*P_h, ea = X_a + beta*P_a.

W1: every read of a per-player m5 state asserts its last write came from a
strictly earlier game (runtime guard); updates happen only after all reads.
W5: tonight's dressed MEMBERSHIP is the only same-game input here; tonight's
5v5 TOI enters only the post-game update.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import nhl_depth_eval as Hd
from nhl_features_eval import TEAM_FIX

PFX = "data/bt_nhl_rapmel_"
MAX_GID = 2018000000
M_DECAY = 0.8
DEB_FROM = "2011-07-01"
DEB_N = 5
DEB_MIN_GAMES = 200
DEB_FALLBACK = {"F": 10.0, "D": 14.0}
DP_WIN = 10


def norm(t):
    return TEAM_FIX.get(t, t)


def dint(d):
    return int(d.replace("-", ""))


# ------------------------------------------------------------------ loading --
def load_toi5(dev_ids):
    t = pd.read_csv(f"{PFX}toi5.csv")
    t = t[(t.gid < MAX_GID) & t.gid.isin(dev_ids)]
    out = defaultdict(dict)
    for g, s, p, v in zip(t.gid.to_numpy(), t.side.to_numpy(), t.pid.to_numpy(),
                          t.sec5.to_numpy()):
        out[int(g)][(int(s), int(p))] = float(v)
    return out


def seed_positions():
    seed = {}
    for k, v in json.load(open("data/nhl_player_names.json", encoding="utf-8")).items():
        if v.get("pos") in ("D", "C", "L", "R"):
            seed[int(k)] = "D" if v["pos"] == "D" else "F"
    return seed


class ValueSet:
    """One fit: cutoff, {pid: v}, {pid: pos}, v_repl, max training date."""

    def __init__(self, name, cutoff, df, meta):
        self.name = name
        self.cutoff = cutoff
        self.v = dict(zip(df.pid.astype(np.int64).tolist(), df.v.astype(float).tolist()))
        self.pos = dict(zip(df.pid.astype(np.int64).tolist(), df.pos.tolist()))
        self.v_repl = dict(meta["v_repl"])
        self.max_train = int(meta["max_train_date"])
        assert self.max_train <= dint(cutoff), f"W3: {name} trained past its cutoff"


def load_values(pre_only=False, drop=None):
    """Sorted list of ValueSet. drop='YYYY-MM-DD' swaps in the W2 refits (stints
    on/after drop removed) for every fit whose cutoff >= drop."""
    fits = json.load(open(f"{PFX}fits.json"))["fits"]
    allv = pd.read_csv(f"{PFX}values.csv")
    out = []
    for name, meta in fits.items():
        if pre_only and not name.startswith("pre_"):
            continue
        cut = meta["cutoff"]
        if drop is not None and cut >= drop:
            df = pd.read_csv(f"{PFX}fitv_{name}_drop{drop}.csv")
            meta = json.load(open(f"{PFX}fitm_{name}_drop{drop}.json"))
            assert meta["drop"] == drop
            if meta.get("n_stints", 0) > 0:
                assert meta["max_train_date"] < dint(drop)
        else:
            df = allv[allv.cutoff_date == cut]
        out.append(ValueSet(name, cut, df, meta))
    out.sort(key=lambda vs: vs.cutoff)
    return out


# ------------------------------------------------------------ lineup power ---
def build_P(games, dressed, toi5, values, seed, guard=True):
    """Per-game P_h, P_a, dP_h, dP_a and bookkeeping (full length over games)."""
    n = len(games)
    P = {"home": np.zeros(n), "away": np.zeros(n)}
    dP = {"home": np.zeros(n), "away": np.zeros(n)}
    fb = {"home": np.zeros(n, bool), "away": np.zeros(n, bool)}
    n_unval = np.zeros(n)
    n_prior = np.zeros(n)
    vidx = np.full(n, -1)
    m5, m5_w = {}, {}
    napp5 = defaultdict(int)
    first_date = {}
    deb_sum = {"F": 0.0, "D": 0.0}
    deb_cnt = {"F": 0, "D": 0}
    pend_sum = {"F": 0.0, "D": 0.0}
    pend_cnt = {"F": 0, "D": 0}
    cur_date, last_committed = None, ""
    hist = defaultdict(lambda: deque(maxlen=DP_WIN))
    last_lineup = {}
    first_use = {}
    k = -1
    for i, g in enumerate(games):
        gid = g["game_id"]
        dstr = g["date"]
        if dstr != cur_date:
            if cur_date is not None:
                assert cur_date < dstr, "games not in date order"
                for q in ("F", "D"):
                    deb_sum[q] += pend_sum[q]
                    deb_cnt[q] += pend_cnt[q]
                    pend_sum[q], pend_cnt[q] = 0.0, 0
                last_committed = cur_date
            cur_date = dstr
        assert last_committed < dstr                   # W3: debutant prior strictly earlier
        while k + 1 < len(values) and values[k + 1].cutoff < dstr:
            k += 1
        vs = values[k] if k >= 0 else None
        if vs is not None:
            assert vs.cutoff < dstr and vs.max_train <= dint(vs.cutoff)   # W3
            first_use.setdefault(vs.name, dstr)
        vidx[i] = k

        def pos_of(p):
            if vs is not None and p in vs.pos:
                return vs.pos[p]
            return seed.get(p, "F")

        def deb(q):
            return deb_sum[q] / deb_cnt[q] if deb_cnt[q] >= DEB_MIN_GAMES else DEB_FALLBACK[q]

        def val(p):
            if vs is None:
                return 0.0
            v = vs.v.get(p)
            return v if v is not None else vs.v_repl[pos_of(p)]

        def mins(p):
            if p in m5:
                if guard:
                    assert m5_w[p] < i, "W1 violated (m5)"
                return m5[p]
            return deb(pos_of(p))

        gdr = dressed.get(gid, {})
        actual = {}
        for side in ("home", "away"):
            key = norm(g[side])
            dr = gdr.get(key)
            actual[side] = (key, dr)
            lineup = list(dr) if dr else last_lineup.get(key)
            if not dr:
                fb[side][i] = True
            if not lineup:
                continue
            tot = 0.0
            for p in lineup:
                tot += val(p) * mins(p) / 60.0
                if vs is not None and p not in vs.v:
                    n_unval[i] += 1
                if p not in m5:
                    n_prior[i] += 1
            P[side][i] = tot
            past = hist[key]
            if past:
                ps = [sum(val(p) * mins(p) / 60.0 for p in lu) for lu in past]
                dP[side][i] = tot - float(np.mean(ps))
        # ---------------- updates AFTER every read ----------------
        t5 = toi5.get(gid)
        for side, flag in (("home", 1), ("away", 0)):
            key, dr = actual[side]
            if not dr:
                continue
            for p in dr:
                first_date.setdefault(p, dstr)
                if t5 is not None:
                    x = t5.get((flag, p), 0.0) / 60.0
                    napp5[p] += 1
                    if first_date[p] >= DEB_FROM and napp5[p] <= DEB_N:
                        q = pos_of(p)
                        pend_sum[q] += x
                        pend_cnt[q] += 1
                    m5[p] = x if p not in m5 else M_DECAY * m5[p] + (1 - M_DECAY) * x
                    m5_w[p] = i
            hist[key].append(list(dr))
            last_lineup[key] = list(dr)
    return {"P_h": P["home"], "P_a": P["away"], "dP_h": dP["home"], "dP_a": dP["away"],
            "fb_h": fb["home"], "fb_a": fb["away"], "n_unvalued": n_unval,
            "n_prior_minutes": n_prior, "value_idx": vidx, "first_use": first_use,
            "deb_prior_final": {q: (deb_sum[q] / deb_cnt[q] if deb_cnt[q] else None)
                                for q in ("F", "D")},
            "deb_games_final": dict(deb_cnt)}


# ------------------------------------------------------------ rating runs ---
def xg_beta(games, P_h, P_a, beta, k=6.0, ha=0.15, regress=0.30):
    """nhl_depth_eval.run_xg_arr, line by line, with beta*P inside the rating read."""
    if Hd._TXG is None:
        Hd._TXG = Hd.load_team_xg_dev(games)
    TXG = Hd._TXG
    xr = defaultdict(float)
    out = np.full(len(games), np.nan)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in xr:
                xr[t] *= (1 - regress)
        prev = g["season"]
        rh, ra = xr[g["home"]], xr[g["away"]]
        eh = rh + beta * P_h[i]
        ea = ra + beta * P_a[i]
        out[i] = (eh + ha) - ea
        tx = TXG.get(g["game_id"])
        if tx and g["home"] in tx and g["away"] in tx:
            hxgf, _ = tx[g["home"]]
            axgf, _ = tx[g["away"]]
            pred = eh - ea + ha
            err = (hxgf - axgf) - pred
            xr[g["home"]] += k / 100.0 * err
            xr[g["away"]] -= k / 100.0 * err
        else:
            out[i] = np.nan
    return out


def elo_cell(GL, games, starters, gg, gamma_g, gamma_p, P_h, P_a):
    """W/L Elo with goalie level (gamma_g) and lineup power (gamma_p) inside."""
    return GL.gl_elo(games, starters, gg, gamma_g, mode="new",
                     d_home=-np.asarray(P_h), d_away=-np.asarray(P_a), delta=gamma_p)
