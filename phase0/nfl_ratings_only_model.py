"""Ratings-only game model + context/recency tuning for the 11v11 TrueSkill engine.

Samuel's program: (1) weight each snap by its CONTEXT (win-prob leverage, money
downs, late-and-close), (2) value RECENT snaps much more (bigger weekly tau =
faster Bayesian forgetting), (3) judge the engine by a NEW model that predicts
games from player ratings ALONE — one feature: the pre-game share-weighted
TrueSkill aggregate (home actives minus away actives), logistic w/ intercept
(intercept = home field), walk-forward yearly refits.

Own protocol (participation exists 2016+ only; the GLASSBOX main split cannot
apply): warm 2016, DEV 2017-2021 (all knob selection here), TEST 2022-2025
scored ONCE for the chosen config. Benchmarks on the same TEST games: plain
team Elo (the 1999+ walk) and the shipped 14-feature GLASSBOX model.

Leak discipline (same audited pattern as nfl_ts_model_test): per game the
feature is computed from PRE-game TS state and PRE-game snap shares, then the
game's plays update TS, then the share walk updates.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from math import erf, exp, pi, sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102  (games/snaps/pfr2gsis/pe/llv)
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

MU0, SIG0 = 25.0, 25.0 / 3.0
SIG0_2 = SIG0 * SIG0
BETA = SIG0
SIG2_FLOOR = 0.5 ** 2
V_CLAMP = 3.0

def pdf(x): return exp(-0.5 * x * x) / sqrt(2 * pi)
def cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
def v_win(t):
    c = cdf(t)
    return -t if c < 1e-12 else pdf(t) / c

# ---- plays + full context ----
ctx = {}
with open("data/nfl_play_ctx2.csv") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        try:
            ctx[(r["game_id"], int(float(r["play_id"])))] = (
                float(r["wp"]) if r["wp"] else 0.5,
                int(float(r["down"])) if r["down"] else 1,
                int(float(r["qtr"])) if r["qtr"] else 1,
                abs(float(r["score_differential"])) if r["score_differential"] else 0.0,
                float(r["half_seconds_remaining"]) if r["half_seconds_remaining"] else 900.0,
            )
        except ValueError:
            continue
plays_by_gid = defaultdict(list)
with open("data/nfl_rapm_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); next(rd)
    for gid, pid_, off, dfn, epa in rd:
        O = off.split(";"); D = dfn.split(";")
        if 6 <= len(O) <= 12 and 6 <= len(D) <= 12:
            plays_by_gid[gid].append((int(float(pid_)), tuple(O), tuple(D), float(epa)))
for v in plays_by_gid.values():
    v.sort()
print(f"[{time.time()-T0:.0f}s] {sum(len(v) for v in plays_by_gid.values()):,} plays, "
      f"{len(ctx):,} ctx rows", flush=True)

prot_of = {}
try:
    with open("data/nfl_play_prot.csv") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            prot_of[(r["game_id"], int(float(r["play_id"])))] = (
                r["passer_player_id"], r["rusher_player_id"], r["receiver_player_id"])
    print(f"[{time.time()-T0:.0f}s] protagonists for {len(prot_of):,} plays", flush=True)
except FileNotFoundError:
    print("no protagonist file; micro-matches unavailable", flush=True)
GPOS = {}
_PM = {"QB":"QB","RB":"RB","FB":"RB","HB":"RB","WR":"WR","TE":"TE",
       "T":"OL","G":"OL","C":"OL","OT":"OL","OG":"OL","OL":"OL","LT":"OL","RT":"OL","LG":"OL","RG":"OL",
       "DE":"DL","DT":"DL","NT":"DL","DL":"DL","EDGE":"DL",
       "LB":"LB","ILB":"LB","OLB":"LB","MLB":"LB",
       "CB":"DB","S":"DB","SS":"DB","FS":"DB","SAF":"DB","DB":"DB"}
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if r.get("gsis_id"):
        GPOS[r["gsis_id"]] = _PM.get(r.get("position") or "")

# salary priors: active contract's APY cap % per player-season, z-scored within
# position group per season. year_signed <= season (FA/rookie deals land pre-season;
# midseason extensions are a negligible look-ahead, noted).
Z_SAL = {}
try:
    import pandas as _pd
    _c = _pd.read_parquet("data/nfl_contracts.parquet")
    _c = _c.dropna(subset=["gsis_id"])
    _c = _c[(_c.year_signed > 0) & _c.apy_cap_pct.notna()]
    _rows = []
    for _r in _c.itertuples():
        y0 = int(_r.year_signed)
        span = max(int(_r.years) if _pd.notna(_r.years) and _r.years > 0 else 1, 1)
        for s_ in range(max(2016, y0), min(2026, y0 + span)):
            _rows.append((_r.gsis_id, s_, y0, float(_r.apy_cap_pct)))
    _df = _pd.DataFrame(_rows, columns=["gsis", "season", "y0", "pct"])
    _df = _df.sort_values("y0").groupby(["gsis", "season"]).tail(1)   # latest signed wins
    _df["grp"] = _df.gsis.map(GPOS)
    _df = _df.dropna(subset=["grp"])
    for (_s, _g), _sub in _df.groupby(["season", "grp"]):
        m_, sd_ = _sub.pct.mean(), max(_sub.pct.std(), 1e-6)
        for _t in _sub.itertuples():
            Z_SAL[(_t.gsis, _s)] = max(-2.5, min(2.5, (_t.pct - m_) / sd_))
    # Samuel's cohort variant: z vs the SIGNING-YEAR position market (frozen)
    Z_SAL2 = {}
    _stats = {}
    for (_y, _g), _sub in _df.groupby(["y0", "grp"]):
        _stats[(_y, _g)] = (_sub.pct.mean(), max(_sub.pct.std(), 1e-6))
    for _t in _df.itertuples():
        m_, sd_ = _stats[(_t.y0, _t.grp)]
        Z_SAL2[(_t.gsis, _t.season)] = max(-2.5, min(2.5, (_t.pct - m_) / sd_))
    print(f"[{time.time()-T0:.0f}s] salary priors: {len(Z_SAL):,} player-seasons "
          f"(+ cohort variant)", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"salary priors unavailable ({type(e).__name__})", flush=True)

FTN = {}
try:
    import pandas as _pd2
    for _y in (2022, 2023, 2024, 2025):
        _f = _pd2.read_parquet(f"data/ftn_{_y}.parquet",
                               columns=["nflverse_game_id", "nflverse_play_id",
                                        "is_throw_away", "is_drop", "is_interception_worthy"])
        for _r in _f.itertuples():
            FTN[(_r.nflverse_game_id, int(_r.nflverse_play_id))] = (
                bool(_r.is_throw_away), bool(_r.is_drop), bool(_r.is_interception_worthy))
    print(f"[{time.time()-T0:.0f}s] FTN flags: {len(FTN):,} charted plays", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"FTN unavailable ({type(e).__name__})", flush=True)

INJ = set()
for _y in range(2016, 2026):
    try:
        for _r in csv.DictReader(open(f"data/inj_{_y}.csv", encoding="utf-8")):
            if (_r.get("report_status") or "") in ("Questionable", "Doubtful") and _r.get("gsis_id"):
                try:
                    INJ.add((_r["gsis_id"], int(_r["season"]), int(_r["week"])))
                except ValueError:
                    continue
    except FileNotFoundError:
        continue
print(f"[{time.time()-T0:.0f}s] injury tags (Q/D): {len(INJ):,} player-weeks", flush=True)

QBOBS = {}
_rates = [ln[0] / ln[1] for (q_, s_, w_), ln in qbw.items()
          if 2016 <= s_ <= 2021 and ln[1] >= 10]
_SDQB = float(np.std(_rates)) if _rates else 0.15
_dbs = [ln[1] for (q_, s_, w_), ln in qbw.items()
        if 2016 <= s_ <= 2021 and ln[1] >= 10]
MED_DB = float(np.median(_dbs)) if _dbs else 30.0
for (q_, s_, w_), ln in qbw.items():
    if s_ >= 2016 and ln[1] >= 10:
        z_ = max(-2.5, min(2.5, (ln[0] / ln[1] - lg_qb) / _SDQB))
        QBOBS[(q_, str(s_), f"{w_:02d}")] = (z_, float(ln[1]))
print(f"[{time.time()-T0:.0f}s] QB weekly obs: {len(QBOBS):,} lines (sd {_SDQB:.3f}, "
      f"median db {MED_DB:.1f})", flush=True)

BYEAR = {}
for _r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    bd = _r.get("birth_date") or ""
    if _r.get("gsis_id") and len(bd) >= 4:
        try:
            BYEAR[_r["gsis_id"]] = int(bd[:4])
        except ValueError:
            pass
print(f"[{time.time()-T0:.0f}s] birth years: {len(BYEAR):,}", flush=True)

SKOBS, DOBS = {}, {}
_effs, _woprs = [], []
for (pid, s_, w_), o in OFFW.items():
    if 2016 <= s_ <= 2021 and bucket(o["pos"]) == "SK":
        opp = o["car"] + o["tgt"]
        if opp >= 3:
            _effs.append((o["repa"] + o["cepa"]) / opp)
        _woprs.append(o["wopr"])
_me, _se = (np.mean(_effs), max(np.std(_effs), 1e-6)) if _effs else (0.0, 1.0)
_mw, _sw = (np.mean(_woprs), max(np.std(_woprs), 1e-6)) if _woprs else (0.0, 1.0)
for (pid, s_, w_), o in OFFW.items():
    if s_ >= 2016 and bucket(o["pos"]) == "SK":
        opp = o["car"] + o["tgt"]
        z_w = (o["wopr"] - _mw) / _sw
        if opp >= 3:
            z_e = ((o["repa"] + o["cepa"]) / opp - _me) / _se
            z = 0.6 * z_e + 0.4 * z_w
        else:
            z = 0.4 * z_w - 0.3          # barely used = weak negative usage signal
        SKOBS[(pid, str(s_), f"{w_:02d}")] = max(-2.5, min(2.5, z))
_dr, _db_, _dt = [], [], []
for (pid, s_, w_), d in DEFW.items():
    if 2016 <= s_ <= 2021:
        _dr.append(d["sack"] + 0.5 * d["tfl"] + 0.5 * d["hit"])
        _db_.append(3.0 * d["int"] + d["pd"])
        _dt.append(d["solo"])
_sr = (np.mean(_dr), max(np.std(_dr), 1e-6)); _sb = (np.mean(_db_), max(np.std(_db_), 1e-6))
_st = (np.mean(_dt), max(np.std(_dt), 1e-6))
for (pid, s_, w_), d in DEFW.items():
    if s_ >= 2016:
        z = (0.45 * (d["sack"] + 0.5 * d["tfl"] + 0.5 * d["hit"] - _sr[0]) / _sr[1]
             + 0.35 * (3.0 * d["int"] + d["pd"] - _sb[0]) / _sb[1]
             + 0.20 * (d["solo"] - _st[0]) / _st[1])
        DOBS[(pid, str(s_), f"{w_:02d}")] = max(-2.5, min(2.5, z))
print(f"[{time.time()-T0:.0f}s] skill obs: {sum(1 for k in SKOBS if int(k[1])>=2016):,} | "
      f"def obs: {len(DOBS):,}", flush=True)

G16 = [(i, g) for i, g in enumerate(games) if g["season"] >= 2016]
seas16 = np.array([g["season"] for _, g in G16])
y16 = np.array([g["y"] for _, g in G16])

_MODE_DIAG = {}   # mode_vd -> offset diagnostics, snapshot at end of 2021 (pre-shrink)
_B2_PAIRS = []    # (raw QB weekly z, oppdef mean mu) pairs from the last fus_adj>0 walk
_TTT_DIAG = {}    # (ttt_depth, ttt_passes) -> re-walk runtime + mean |delta mu| per boundary
_TC_DIAG = {}     # tc_w -> count of team-change sigma widenings applied
_TNU_DIAG = {}    # t_nu -> share of duels with damp < 0.5

def run_config(tau, use_lev, down_mult, late_mult, season_shrink=1.0 / 3.0,
               widen2=1.5 ** 2, playoff_mult=1.0, dead_mult=1.0, opp_k=0.0,
               opp_mode="league", flat_scale=1.0, role_r=0.0, micro="off",
               sal_k=0.0, grade="off", mov_scale=0.5, mov_cap=2.0,
               draw_band=0.0, neutral_skip=0.0, return_parts=False,
               beta=BETA, floor2=SIG2_FLOOR, lev_exp=1.0, draw_eps=0.05,
               hfa_t=0.0, f_ta=False, f_drop=False, f_iw=False,
               inj_sym=1.0, inj_neg=1.0, agg_k=1.0, share_p=1.0,
               two_track=False, fus_k=0.0, fus_R=36.0, sal_mode="active",
               opp_anchor="alltime", rel_clamp=False, bnd_mode="mu0",
               dead_scale=False, share_bnd=False, win_snaps=0,
               age_k=0.0, age_y=0.0, age_w=0.0, skf_k=0.0, skf_R=60.0,
               dff_k=0.0, dff_R=60.0, run_w=1.0, draw_band_run=None,
               ent_d=0.0, ent_qb=False, mode_vd=0.0, mode_tau2=0.0025,
               fus_vol=0.0, fus_adj=0.0, ttt_depth=0, ttt_passes=0,
               cont=False, tc_w=0.0, t_nu=0.0):
    """One engine walk -> per-game ts_diff feature (pre-game, leak-free).

    Game importance: playoff snaps x playoff_mult; 'useless' late-season games
    (BOTH teams mathematically short of 7 wins, pre-game records) x dead_mult.
    Opponent quality: each side's update also scales with the OPPOSING unit's
    mean rating, factor 1 + opp_k*(opp_mean_mu - 25), clamped [0.3, 2].
    """
    tau2 = tau * tau
    ZS = Z_SAL2 if sal_mode == "cohort" else Z_SAL
    mu, s2, lastwk = {}, {}, {}
    from collections import deque as _dq
    ev_q = defaultdict(_dq)      # player -> deque of (delta, snaps) events
    win_tot = defaultdict(int)   # snaps currently inside the window
    exp_sum = defaultdict(float) # sum of deltas EXPIRED out of the window
    gd = defaultdict(float)      # current-game delta accumulator
    gs_n = defaultdict(int)      # current-game snap count
    lastg = {}
    def push_game(p):
        if gs_n[p] or gd[p]:
            ev_q[p].append((gd[p], gs_n[p]))
            win_tot[p] += gs_n[p]
            gd[p] = 0.0; gs_n[p] = 0
            while win_tot[p] > win_snaps and ev_q[p]:
                d0, n0 = ev_q[p].popleft()
                exp_sum[p] += d0; win_tot[p] -= n0
    def push_event(p, delta):
        ev_q[p].append((delta, 0))
    muR, s2R = {}, {}                    # run track (pass track = mu/s2)
    offP, offR = {}, {}                  # TS2 per-mode offsets [mu, s2], lazy-created
    oppdef = {}                          # (qb, season_str, week_str) -> [sum mean-def-mu, n]
    if fus_adj:
        _B2_PAIRS.clear()
    # ---- TS3 sliver state (all default-off; default path stays bit-identical) ----
    ttt_cache = defaultdict(list) if ttt_depth else None   # season -> [(gid, gmult)]
    ttt_sec = 0.0; ttt_dmu = []
    if ttt_depth:
        assert (not two_track and not mode_vd and micro == "off" and not win_snaps
                and not role_r and not hfa_t and inj_sym == 1.0 and inj_neg == 1.0
                and not f_ta and not f_drop and not f_iw and not neutral_skip
                and grade == "off" and run_w == 1.0 and draw_band_run is None
                and not ent_d and opp_mode == "league" and opp_anchor == "alltime"
                and not rel_clamp), "ttt re-walk supports the V7 knob subset only"
    cop = {}                       # S2: unordered pfr-pid pair -> co-snap mass
    c_m, c_v = 0.0, 1.0            # S2: leak-free running mean/var of the raw index
    last_team = {}                 # S3: gsis -> last seen team
    tc_n = 0                       # S3: widening events applied
    tn_ct = [0, 0]                 # S4: [duels with damp<0.5, duels total]

    def rewalk_game(gid_r, seas_r, gmult_r):
        """S1: one extra forward pass over an already-walked game's plays (warm
        start from current states). Base-track duels only; fusion, weekly tau
        widening, share2 walk and standings walk are all SKIPPED — pure
        re-absorption of strictly-past evidence, so walk-forward legality holds."""
        lg = (sum(mu.values()) / len(mu)) if (opp_k and mu) else MU0
        for pid_, O, D, epa in plays_by_gid.get(gid_r, ()):
            for p in O:
                if p not in mu:
                    mu[p] = MU0 + sal_k * ZS.get((p, seas_r), 0.0); s2[p] = SIG0_2
            for p in D:
                if p not in mu:
                    mu[p] = MU0 + sal_k * ZS.get((p, seas_r), 0.0); s2[p] = SIG0_2
            wp, down, qtr, adiff, _hs = ctx.get((gid_r, pid_), (0.5, 1, 1, 0.0, 900.0))
            wgt = ((4.0 * wp * (1.0 - wp)) ** lev_exp) if use_lev else \
                (0.25 if (wp < 0.05 or wp > 0.95) else 1.0)
            if down >= 3:
                wgt *= down_mult
            if qtr >= 4 and adiff <= 8:
                wgt *= late_mult
            wgt *= gmult_r * flat_scale
            A, B = O, D
            sA = sum(mu[p] for p in A); sB = sum(mu[p] for p in B)
            c2 = (len(A) + len(B)) * beta * beta + sum(s2[p] for p in A) + sum(s2[p] for p in B)
            c = sqrt(c2)
            t = (sA - sB) / c
            if opp_k:
                w_A = wgt * min(2.0, max(0.3, 1.0 + opp_k * (sB / len(B) - lg)))
                w_B = wgt * min(2.0, max(0.3, 1.0 + opp_k * (sA / len(A) - lg)))
            else:
                w_A = w_B = wgt
            if t_nu:
                _dmp = 1.0 / (1.0 + (t * t) / t_nu)
                w_A *= _dmp; w_B *= _dmp
            if draw_band > 0 and abs(epa) < draw_band:
                e = draw_eps
                den = cdf(e - t) - cdf(-e - t)
                if den > 1e-9:
                    vd = (pdf(-e - t) - pdf(e - t)) / den
                    wd = vd * vd + ((e - t) * pdf(e - t) + (e + t) * pdf(-e - t)) / den
                    wd = max(0.0, min(1.0, wd))
                    for p in A:
                        mu[p] += w_A * (s2[p] / c) * vd
                        s2[p] = max(s2[p] * (1.0 - w_A * (s2[p] / c2) * wd), floor2)
                    for p in B:
                        mu[p] -= w_B * (s2[p] / c) * vd
                        s2[p] = max(s2[p] * (1.0 - w_B * (s2[p] / c2) * wd), floor2)
                continue
            if epa > 0:
                win, lose, tt, w_w, w_l = A, B, t, w_A, w_B
            else:
                win, lose, tt, w_w, w_l = B, A, -t, w_B, w_A
            v = v_win(tt); w = v * (v + tt)
            for p in win:
                mu[p] += w_w * (s2[p] / c) * v
                s2[p] = max(s2[p] * (1.0 - w_w * (s2[p] / c2) * w), floor2)
            for p in lose:
                mu[p] -= w_l * (s2[p] / c) * v
                s2[p] = max(s2[p] * (1.0 - w_l * (s2[p] / c2) * w), floor2)
    share2 = {}
    pos2 = {}
    PG = {"QB":"QB","RB":"RB","FB":"RB","HB":"RB","WR":"WR","TE":"TE",
          "T":"OL","G":"OL","C":"OL","OT":"OL","OG":"OL","OL":"OL","LT":"OL","RT":"OL","LG":"OL","RG":"OL",
          "DE":"DL","DT":"DL","NT":"DL","DL":"DL","EDGE":"DL",
          "LB":"LB","ILB":"LB","OLB":"LB","MLB":"LB",
          "CB":"DB","S":"DB","SS":"DB","FS":"DB","SAF":"DB","DB":"DB"}
    f = np.zeros(len(G16))
    KEYS = ["qb_raw", "off_raw", "def_raw", "qb_prec", "off_prec", "def_prec",
            "qb_pn", "off_pn", "def_pn", "p_prec", "r_prec", "var"]
    if cont:
        KEYS = KEYS + ["cont"]
    parts = {k: np.zeros(len(G16)) for k in KEYS} if return_parts else None
    gm = {"QB": MU0, "RB": MU0, "WR": MU0, "TE": MU0, "OL": MU0, "DL": MU0, "LB": MU0, "DB": MU0}
    gv = {k: 4.0 for k in gm}                      # running group mean/var (EWMA, leak-free)
    prev = None
    wins = defaultdict(float); played = defaultdict(int)
    act_pool, seen_season = set(), set()
    for j, (i, g) in enumerate(G16):
        if prev is not None and g["season"] != prev:
            if ttt_depth and ttt_passes:   # S1: re-absorb past seasons BEFORE shrink/widen
                _t0r = time.time()
                _mu_pre = dict(mu)
                for _ in range(ttt_passes):
                    for s_r in range(g["season"] - ttt_depth, g["season"]):
                        for gid_r, gm_r in ttt_cache.get(s_r, ()):
                            rewalk_game(gid_r, s_r, gm_r)
                if _mu_pre:
                    ttt_dmu.append(float(np.mean([abs(mu[p] - _mu_pre[p]) for p in _mu_pre])))
                ttt_sec += time.time() - _t0r
            if bnd_mode == "group":
                for p in mu:
                    base = gm.get(GPOS.get(p) or "OL", MU0)
                    tgt = base + sal_k * ZS.get((p, g["season"]), 0.0)
                    _new = tgt + (mu[p] - tgt) * (1.0 - season_shrink)
                    if win_snaps:
                        push_event(p, _new - mu[p])
                    mu[p] = _new
                    s2[p] = min(s2[p] + widen2, SIG0_2)
            else:
                for p in mu:
                    tgt = MU0 + sal_k * ZS.get((p, g["season"]), 0.0)
                    _new = tgt + (mu[p] - tgt) * (1.0 - season_shrink)
                    if win_snaps:
                        push_event(p, _new - mu[p])
                    mu[p] = _new
                    s2[p] = min(s2[p] + widen2, SIG0_2)
            if age_k or age_y or age_w:
                for p in mu:
                    by = BYEAR.get(p)
                    if by is None:
                        continue
                    age = g["season"] - by
                    if age_k and age > 29:
                        mu[p] -= age_k * min(age - 29, 6)
                    if age_y and age < 24:
                        mu[p] += age_y * min(24 - age, 4)
                    if age_w and age > 30:
                        s2[p] = min(s2[p] + age_w * (age - 30), SIG0_2)
            for p in muR:
                muR[p] = MU0 + (muR[p] - MU0) * (1.0 - season_shrink)
                s2R[p] = min(s2R[p] + widen2, SIG0_2)
            if mode_vd:
                if prev == 2021:          # snapshot end-of-DEV offset state (pre-shrink)
                    _n_off = len(set(offP) | set(offR))
                    _allo = ([abs(st[0]) for st in offP.values()]
                             + [abs(st[0]) for st in offR.values()])
                    _MODE_DIAG[mode_vd] = {
                        "players": len(mu), "with_offset": _n_off,
                        "share_with_offset": round(_n_off / max(len(mu), 1), 4),
                        "mean_abs_offP": round(float(np.mean([abs(st[0]) for st in offP.values()])), 4) if offP else 0.0,
                        "mean_abs_offR": round(float(np.mean([abs(st[0]) for st in offR.values()])), 4) if offR else 0.0,
                        "mean_abs_off": round(float(np.mean(_allo)), 4) if _allo else 0.0}
                for _off in (offP, offR):  # offsets shrink hard toward 0
                    for _ost in _off.values():
                        _ost[0] *= 0.5
                        _ost[1] = min(_ost[1] + mode_vd / 4.0, mode_vd)
            if share_bnd:
                for st_ in share2.values():
                    st_[0] *= 0.7 ** 4; st_[1] *= 0.7 ** 4
            act_pool = set(seen_season); seen_season = set()
            wins.clear(); played.clear()
        prev = g["season"]
        # game-importance multiplier from PRE-game standings
        sched_len = 16 if g["season"] < 2021 else 17
        if g["type"] != "REG":
            gmult = playoff_mult
        else:
            thr = 7.0 * sched_len / 16.0 if dead_scale else 7.0
            dead_h = wins[g["home"]] + (sched_len - played[g["home"]]) < thr
            dead_a = wins[g["away"]] + (sched_len - played[g["away"]]) < thr
            gmult = dead_mult if (dead_h and dead_a) else 1.0
        if ttt_cache is not None:          # S1: remember this game for future re-walks
            ttt_cache[g["season"]].append((g["gid"], gmult))
        if opp_k and opp_mode == "league" and mu:
            if opp_anchor == "active" and act_pool:
                _vals = [mu[p] for p in act_pool if p in mu]
                lg_mu = sum(_vals) / len(_vals) if _vals else MU0
            else:
                lg_mu = sum(mu.values()) / len(mu)
        else:
            lg_mu = MU0
        lg_muR = (sum(muR.values()) / len(muR)) if (opp_k and opp_mode == "league" and muR) else MU0
        if tc_w:               # S3: widen sigma once when a listed player switched teams
            for team in (g["home"], g["away"]):
                tbl = snaps.get((g["gid"], team))
                if not tbl:
                    continue
                for pid in tbl:
                    g_ = pfr2gsis.get(pid)
                    if not g_:
                        continue
                    lt = last_team.get(g_)
                    if lt is not None and lt != team and g_ in s2:
                        s2[g_] = min(s2[g_] + tc_w * tc_w, SIG0_2)
                        tc_n += 1
                    last_team[g_] = team
        # 1. feature from pre-game state (+ component aggregates when requested)
        seen_mus = []
        for side, team in ((1, g["home"]), (-1, g["away"])):
            tbl = snaps.get((g["gid"], team))
            if tbl is None:
                continue
            tot = 0.0
            for pid in tbl:
                st = share2.get(pid)
                if not st or st[1] <= 0:
                    continue
                g_ = pfr2gsis.get(pid)
                m_ = mu.get(g_) if g_ else None
                if m_ is None:
                    continue
                if mode_vd:               # readout: base + pass-share-weighted offsets
                    _op = offP.get(g_); _orr = offR.get(g_)
                    m_ = m_ + 0.58 * (_op[0] if _op else 0.0) + 0.42 * (_orr[0] if _orr else 0.0)
                w_sh = (st[0] / st[1]) ** share_p
                if win_snaps:
                    m_ = m_ - exp_sum.get(g_, 0.0)
                if rel_clamp:
                    _base = gm.get(GPOS.get(g_) or "OL", MU0)
                    vraw = max(-V_CLAMP, min(V_CLAMP, m_ - _base))
                else:
                    vraw = max(-V_CLAMP, min(V_CLAMP, m_ - MU0))
                tot += w_sh * vraw
                if return_parts:
                    grp = GPOS.get(g_) or "OL"
                    cat = "qb" if grp == "QB" else ("def" if grp in ("DL", "LB", "DB") else "off")
                    sd_g = sqrt(max(gv.get(grp, 4.0), 1e-6))
                    vpn = max(-2.5, min(2.5, (m_ - gm.get(grp, MU0)) / sd_g))
                    prec = agg_k / (agg_k + s2.get(g_, SIG0_2))
                    parts[cat + "_raw"][j] += side * w_sh * vraw
                    parts[cat + "_prec"][j] += side * w_sh * vraw * prec
                    parts["var"][j] += (w_sh * prec) ** 2 * s2.get(g_, SIG0_2)
                    parts[cat + "_pn"][j] += side * w_sh * vpn
                    if two_track:
                        mR = muR.get(g_)
                        if mR is not None:
                            vR = max(-V_CLAMP, min(V_CLAMP, mR - MU0))
                            parts["r_prec"][j] += side * w_sh * vR * (agg_k / (agg_k + s2R.get(g_, SIG0_2)))
                        parts["p_prec"][j] += side * w_sh * vraw * prec
                    seen_mus.append((grp, m_))
            f[j] += side * tot
        if return_parts:                            # update group stats AFTER the feature
            for grp, m_ in seen_mus:
                d_ = m_ - gm[grp]
                gm[grp] += 0.002 * d_
                gv[grp] = (1 - 0.002) * gv[grp] + 0.002 * d_ * d_
        if cont and return_parts:      # S2: pre-game co-snap continuity (leak-free z)
            _idxs = []
            for side, team in ((1, g["home"]), (-1, g["away"])):
                tbl = snaps.get((g["gid"], team))
                idx = 0.0
                if tbl:
                    _off_l = [pid for pid, (pos, op, dp) in tbl.items() if pos not in DEFPOS]
                    _off_l.sort(key=lambda pid: -(share2[pid][0] / share2[pid][1]
                                if pid in share2 and share2[pid][1] > 0 else 0.0))
                    _off_l = _off_l[:11]               # top-11 by as-of share
                    if len(_off_l) >= 2:
                        _vals = []
                        for _a in range(len(_off_l)):
                            for _b in range(_a + 1, len(_off_l)):
                                _pr = ((_off_l[_a], _off_l[_b]) if _off_l[_a] < _off_l[_b]
                                       else (_off_l[_b], _off_l[_a]))
                                _vals.append(np.log1p(cop.get(_pr, 0.0)))
                        idx = float(np.mean(_vals))
                parts["cont"][j] += side * (idx - c_m) / sqrt(max(c_v, 1e-6))
                _idxs.append(idx)
            for idx in _idxs:                          # EWMA stats update AFTER
                _dc = idx - c_m
                c_m += 0.002 * _dc
                c_v = (1 - 0.002) * c_v + 0.002 * _dc * _dc
        # role multipliers: depth-chart position by AS-OF usage within team-position
        # (starter = hardest job -> amplified; deep backup -> discounted)
        role_m = {}
        if role_r:
            for team in (g["home"], g["away"]):
                tbl = snaps.get((g["gid"], team))
                if not tbl:
                    continue
                grp = defaultdict(list)
                for pid, (pos, op, dp) in tbl.items():
                    pgk = PG.get(pos)
                    st = share2.get(pid)
                    if pgk and st and st[1] > 0:
                        grp[pgk].append((pid, st[0] / st[1]))
                for pgk, lst in grp.items():
                    mx = max(s for _, s in lst)
                    if mx <= 0:
                        continue
                    for pid, s in lst:
                        g_ = pfr2gsis.get(pid)
                        if g_:
                            role_m[g_] = max(1.0 - role_r,
                                             min(1.0 + role_r, 1.0 + role_r * (2.0 * s / mx - 1.0)))
        # 2. TS updates from this game's plays, context-weighted
        gid = g["gid"]
        wk = (gid[:4], gid[5:7])
        inj_tag = set()
        if inj_sym != 1.0 or inj_neg != 1.0:
            wk_i = int(gid[5:7])
            s_i = g["season"]
            for team in (g["home"], g["away"]):
                tbl = snaps.get((gid, team))
                if tbl:
                    for pid in tbl:
                        g_ = pfr2gsis.get(pid)
                        if g_ and (g_, s_i, wk_i) in INJ:
                            inj_tag.add(g_)
        h_set = set()
        if hfa_t and not g["neutral"]:
            tbl_h = snaps.get((gid, g["home"]))
            if tbl_h:
                h_set = {pfr2gsis.get(pid) for pid in tbl_h}
                h_set.discard(None)
        seas_now = g["season"]
        for pid_, O, D, epa in plays_by_gid.get(gid, ()):
            if two_track or run_w != 1.0 or draw_band_run is not None or mode_vd:
                _pr = prot_of.get((gid, pid_))
                trk_run = bool(_pr and not _pr[0] and _pr[1])
            else:
                _pr = None
                trk_run = False
            if mode_vd and _pr:           # mode: passer -> offP, run -> offR, else base-only
                play_off = offP if _pr[0] else (offR if trk_run else None)
            else:
                play_off = None
            muD = muR if (two_track and trk_run) else mu
            s2D = s2R if (two_track and trk_run) else s2
            if opp_anchor == "active":
                seen_season.update(O); seen_season.update(D)
                act_pool.update(O); act_pool.update(D)
            lw_key = "R" if (two_track and trk_run) else "P"
            if win_snaps:
                for p in O + D:
                    if lastg.get(p) != gid:
                        push_game(p)
                        lastg[p] = gid
            for p in O:
                if p not in muD:
                    _e = ent_d if (ent_d and (not ent_qb or GPOS.get(p) == "QB")) else 0.0
                    muD[p] = MU0 - _e + sal_k * ZS.get((p, seas_now), 0.0); s2D[p] = SIG0_2
                elif lastwk.get((p, lw_key)) != wk:
                    _gp = GPOS.get(p)
                    _z = None; _ndb = 0.0; _fk = 0.0; _fR = 60.0
                    _old = lastwk.get((p, lw_key))
                    if _old:
                        if fus_k and _gp == "QB":
                            _qo = QBOBS.get((p, _old[0], _old[1]))
                            if _qo is not None:
                                _z, _ndb = _qo
                            _fk = fus_k; _fR = fus_R
                        elif skf_k and _gp in ("RB", "WR", "TE"):
                            _z = SKOBS.get((p, _old[0], _old[1])); _fk = skf_k; _fR = skf_R
                        elif dff_k and _gp in ("DL", "LB", "DB"):
                            _z = DOBS.get((p, _old[0], _old[1])); _fk = dff_k; _fR = dff_R
                    if _z is not None:
                        if fus_adj and _gp == "QB":   # B2: opponent-adjusted observation
                            _od = oppdef.get((p, _old[0], _old[1]))
                            if _od is not None and _od[1] > 0:
                                _odm = _od[0] / _od[1]
                                if seas_now <= 2021:
                                    _B2_PAIRS.append((_z, _odm))
                                _z = _z + fus_adj * (_odm - lg_mu)
                        _obs = MU0 + _fk * _z
                        _Rf = (fus_vol * _fR * (MED_DB / _ndb)) if (fus_vol and _ndb) else _fR
                        _K = s2D[p] / (s2D[p] + _Rf)
                        if win_snaps:
                            push_event(p, _K * (_obs - muD[p]))
                        muD[p] += _K * (_obs - muD[p])
                        s2D[p] = max(s2D[p] * (1.0 - _K), floor2)
                    s2D[p] = min(s2D[p] + tau2, SIG0_2)
                    if mode_vd:                        # offsets widen with their own tau
                        for _off in (offP, offR):
                            _ost = _off.get(p)
                            if _ost is not None:
                                _ost[1] = min(_ost[1] + mode_tau2, mode_vd)
                lastwk[(p, lw_key)] = wk
            for p in D:
                if p not in muD:
                    _e = ent_d if (ent_d and (not ent_qb or GPOS.get(p) == "QB")) else 0.0
                    muD[p] = MU0 - _e + sal_k * ZS.get((p, seas_now), 0.0); s2D[p] = SIG0_2
                elif lastwk.get((p, lw_key)) != wk:
                    s2D[p] = min(s2D[p] + tau2, SIG0_2)
                    if mode_vd:
                        for _off in (offP, offR):
                            _ost = _off.get(p)
                            if _ost is not None:
                                _ost[1] = min(_ost[1] + mode_tau2, mode_vd)
                lastwk[(p, lw_key)] = wk
            if fus_adj:                    # B2: record mean def mu faced by each QB
                _dmn = sum(muD[p] for p in D) / len(D)
                for p in O:
                    if GPOS.get(p) == "QB":
                        _qst = oppdef.setdefault((p, wk[0], wk[1]), [0.0, 0])
                        _qst[0] += _dmn; _qst[1] += 1
            wp, down, qtr, adiff, hsec = ctx.get((gid, pid_), (0.5, 1, 1, 0.0, 900.0))
            wgt = ((4.0 * wp * (1.0 - wp)) ** lev_exp) if use_lev else \
                (0.25 if (wp < 0.05 or wp > 0.95) else 1.0)
            if down >= 3:
                wgt *= down_mult
            if qtr >= 4 and adiff <= 8:
                wgt *= late_mult
            wgt *= gmult * flat_scale
            if run_w != 1.0 and trk_run:
                wgt *= run_w              # facet importance: run evidence discounted

            adv = 0.0
            if hfa_t and h_set:
                n_home = sum(1 for p in O[:5] if p in h_set)
                adv = hfa_t if n_home >= 3 else -hfa_t
            ta, dr, iw = FTN.get((gid, pid_), (False, False, False))
            if (inj_sym != 1.0 or inj_neg != 1.0) and pid_ == plays_by_gid[gid][0][0]:
                pass  # tag set built once per game below (cheap enough per play)

            def duel(A, B, base_wgt, won):
                if not A or not B:
                    return
                if f_iw and iw:
                    won = False                              # process truth: defense won it
                if neutral_skip and abs(epa) < neutral_skip:
                    return                                   # C: small plays don't count
                if grade == "mov":
                    base_wgt *= min(mov_cap, max(0.25, abs(epa) / mov_scale))
                elif grade == "movsqrt":
                    base_wgt *= min(mov_cap, max(0.5, sqrt(abs(epa) / mov_scale)))
                _bnd = draw_band_run if (trk_run and draw_band_run is not None) else draw_band
                is_draw = _bnd > 0 and abs(epa) < _bnd
                if f_ta and ta:
                    is_draw = True                           # throw-away: decision, not defeat
                if f_iw and iw:
                    is_draw = False
                if play_off is not None:
                    for p in A:
                        if p not in play_off:
                            play_off[p] = [0.0, mode_vd]
                    for p in B:
                        if p not in play_off:
                            play_off[p] = [0.0, mode_vd]
                    sA = sum(muD[p] + play_off[p][0] for p in A)
                    sB = sum(muD[p] + play_off[p][0] for p in B)
                    c2 = ((len(A) + len(B)) * beta * beta
                          + sum(s2D[p] + play_off[p][1] for p in A)
                          + sum(s2D[p] + play_off[p][1] for p in B))
                else:
                    sA = sum(muD[p] for p in A); sB = sum(muD[p] for p in B)
                    c2 = (len(A) + len(B)) * beta * beta + sum(s2D[p] for p in A) + sum(s2D[p] for p in B)
                c = sqrt(c2)
                t = (sA - sB) / c + adv       # venue shifts the EXPECTED outcome
                if opp_k:
                    if opp_mode == "self":
                        rel = sB / len(B) - sA / len(A)
                        w_A = base_wgt * min(2.0, max(0.3, 1.0 + opp_k * rel))
                        w_B = base_wgt * min(2.0, max(0.3, 1.0 - opp_k * rel))
                    else:
                        anchor = MU0 if opp_mode == "abs" else (lg_muR if (two_track and trk_run) else lg_mu)
                        w_A = base_wgt * min(2.0, max(0.3, 1.0 + opp_k * (sB / len(B) - anchor)))
                        w_B = base_wgt * min(2.0, max(0.3, 1.0 + opp_k * (sA / len(A) - anchor)))
                else:
                    w_A = w_B = base_wgt
                if t_nu:                     # S4: Student-t IRLS damping of surprises
                    _dmp = 1.0 / (1.0 + (t * t) / t_nu)
                    w_A *= _dmp; w_B *= _dmp
                    tn_ct[1] += 1
                    if _dmp < 0.5:
                        tn_ct[0] += 1
                def imul(p, delta_pos):
                    if p not in inj_tag:
                        return 1.0
                    return inj_sym if delta_pos else inj_sym * inj_neg
                if is_draw:                                  # B: TrueSkill draw update
                    e = draw_eps
                    den = cdf(e - t) - cdf(-e - t)
                    if den > 1e-9:
                        vd = (pdf(-e - t) - pdf(e - t)) / den
                        wd = vd * vd + ((e - t) * pdf(e - t) + (e + t) * pdf(-e - t)) / den
                        wd = max(0.0, min(1.0, wd))
                        for p in A:
                            im = imul(p, vd >= 0)
                            _d = im * w_A * (s2D[p] / c) * vd
                            if win_snaps:
                                gd[p] += _d; gs_n[p] += 1
                            muD[p] += _d
                            s2D[p] = max(s2D[p] * (1.0 - im * w_A * (s2D[p] / c2) * wd), floor2)
                            if play_off is not None:
                                _ost = play_off[p]
                                _ost[0] += im * w_A * (_ost[1] / c) * vd
                                _ost[1] *= (1.0 - im * w_A * (_ost[1] / c2) * wd)
                        for p in B:
                            im = imul(p, vd <= 0)
                            _d = -im * w_B * (s2D[p] / c) * vd
                            if win_snaps:
                                gd[p] += _d; gs_n[p] += 1
                            muD[p] += _d
                            s2D[p] = max(s2D[p] * (1.0 - im * w_B * (s2D[p] / c2) * wd), floor2)
                            if play_off is not None:
                                _ost = play_off[p]
                                _ost[0] -= im * w_B * (_ost[1] / c) * vd
                                _ost[1] *= (1.0 - im * w_B * (_ost[1] / c2) * wd)
                    return
                if won:
                    win, lose, tt, w_win_, w_lose_ = A, B, t, w_A, w_B
                else:
                    win, lose, tt, w_win_, w_lose_ = B, A, -t, w_B, w_A
                if f_drop and dr and not won:
                    w_win_ = w_win_ * 0.3                    # drop: defense credit unearned
                v = v_win(tt); w = v * (v + tt)
                for p in win:
                    rm = (role_m.get(p, 1.0) if role_r else 1.0) * imul(p, True)
                    _d = rm * w_win_ * (s2D[p] / c) * v
                    if win_snaps:
                        gd[p] += _d; gs_n[p] += 1
                    muD[p] += _d
                    s2D[p] = max(s2D[p] * (1.0 - rm * w_win_ * (s2D[p] / c2) * w), floor2)
                    if play_off is not None:
                        _ost = play_off[p]
                        _ost[0] += rm * w_win_ * (_ost[1] / c) * v
                        _ost[1] *= (1.0 - rm * w_win_ * (_ost[1] / c2) * w)
                for p in lose:
                    rm = (role_m.get(p, 1.0) if role_r else 1.0) * imul(p, False)
                    _d = -rm * w_lose_ * (s2D[p] / c) * v
                    if win_snaps:
                        gd[p] += _d; gs_n[p] += 1
                    muD[p] += _d
                    s2D[p] = max(s2D[p] * (1.0 - rm * w_lose_ * (s2D[p] / c2) * w), floor2)
                    if play_off is not None:
                        _ost = play_off[p]
                        _ost[0] -= rm * w_lose_ * (_ost[1] / c) * v
                        _ost[1] *= (1.0 - rm * w_lose_ * (_ost[1] / c2) * w)

            micro_done = False
            if micro != "off":
                pr = prot_of.get((gid, pid_))
                if pr:
                    passer, rusher, receiver = pr
                    Oset = set(O)
                    ol = [p for p in O if GPOS.get(p) == "OL"]
                    box = [p for p in D if GPOS.get(p) in ("DL", "LB")]
                    dbs = [p for p in D if GPOS.get(p) == "DB"]
                    if passer and passer in Oset:
                        duel([passer] + ol, box, wgt, epa > 0)          # protection/QB match
                        if receiver and receiver in Oset:
                            duel([receiver], dbs, wgt, epa > 0)          # coverage match
                        micro_done = True
                    elif rusher and rusher in Oset:
                        duel([rusher] + ol, box, wgt, epa > 0)           # run-box match
                        micro_done = True
            if micro != "replace" or not micro_done:
                duel(O, D, wgt, epa > 0)                                 # full 11v11
        # 3. snap-share walk
        for team in (g["home"], g["away"]):
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEFPOS else op
                st = share2.setdefault(pid, [0.0, 0.0])
                st[0] = snapP["decay"] * st[0] + pct
                st[1] = snapP["decay"] * st[1] + 1.0
        if cont:                       # S2: post-game co-appearance walk (offense actives)
            for team in (g["home"], g["away"]):
                tbl = snaps.get((g["gid"], team))
                if not tbl:
                    continue
                _act = [(pid, op) for pid, (pos, op, dp) in tbl.items() if op > 0]
                for _a in range(len(_act)):
                    pa_, oa_ = _act[_a]
                    for _b in range(_a + 1, len(_act)):
                        pb_, ob_ = _act[_b]
                        _pr = (pa_, pb_) if pa_ < pb_ else (pb_, pa_)
                        cop[_pr] = cop.get(_pr, 0.0) + (oa_ if oa_ < ob_ else ob_)
        # 4. standings walk (for next games' dead-game detection)
        if g["type"] == "REG":
            wins[g["home"]] += g["y"]; wins[g["away"]] += 1.0 - g["y"]
            played[g["home"]] += 1; played[g["away"]] += 1
    if ttt_depth:
        _TTT_DIAG[(ttt_depth, ttt_passes)] = {
            "rewalk_sec": round(ttt_sec, 1), "boundaries": len(ttt_dmu),
            "mean_abs_dmu": round(float(np.mean(ttt_dmu)), 5) if ttt_dmu else 0.0,
            "dmu_per_boundary": [round(x, 5) for x in ttt_dmu]}
    if tc_w:
        _TC_DIAG[tc_w] = tc_n
    if t_nu:
        _TNU_DIAG[t_nu] = {"duels": tn_ct[1],
                           "share_damp_lt_05": round(tn_ct[0] / max(tn_ct[1], 1), 4)}
    return (f, parts) if return_parts else f

def wf_ll(f, lo, hi):
    """walk-forward yearly refit of logistic([ts_diff]) on 2016+ history."""
    X = f.reshape(-1, 1)
    out = []
    for s_ in range(lo, hi + 1):
        tr = (seas16 < s_) & (y16 != 0.5)
        te = seas16 == s_
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y16[tr])
        p = m.predict_proba(X[te])[:, 1]
        out.append(llv(y16[te], p))
    return float(np.concatenate(out).mean()), np.concatenate(out)

# ================= CORE KNOB SWEEP (objective: sigma-discounted sum, DEV) =================
V4 = dict(tau=0.30, use_lev=True, down_mult=1.25, late_mult=1.5,
          playoff_mult=1.5, dead_mult=0.5, opp_k=0.3, opp_mode="league",
          draw_band=0.55)

def wf_X(X, lo, hi):
    lls, accs = [], []
    for s_ in range(lo, hi + 1):
        tr = (seas16 < s_) & (y16 != 0.5)
        te = seas16 == s_
        m = LogisticRegression(C=1e6, max_iter=3000).fit(X[tr], y16[tr])
        p = m.predict_proba(X[te])[:, 1]
        lls.append(llv(y16[te], p))
        msk = y16[te] != 0.5
        accs.append(((p > 0.5) == (y16[te] > 0.5))[msk])
    return float(np.concatenate(lls).mean()), float(np.concatenate(accs).mean())

SEL_LO, SEL_HI = 2017, 2021          # injuries cover all of DEV
def obj(**kw):
    cfg = {**V4, **kw}
    _, P = run_config(**cfg, return_parts=True)
    X = (P["qb_prec"] + P["off_prec"] + P["def_prec"]).reshape(-1, 1)
    return wf_X(X, SEL_LO, SEL_HI)[0], P

results = {}
def trial(name, **kw):
    ll, _ = obj(**kw)
    results[name] = (ll, kw)
    print(f"  {name:<36} {ll:.5f}  [{time.time()-T0:.0f}s]", flush=True)

V5 = dict(beta=60.0, lev_exp=0.5, draw_band=0.40)
V6 = {**V5, "fus_k": 4.0, "fus_R": 36.0}
V7 = {**V6, "sal_k": 1.0, "sal_mode": "cohort"}

print("\nTTT ONE TEST LOOK (2022-2025): v7 vs v7+ttt(depth2, pass1)", flush=True)

def wf_X_detail(X, lo, hi):
    lls, preds, idxs = [], [], []
    for s_ in range(lo, hi + 1):
        tr = (seas16 < s_) & (y16 != 0.5)
        te = seas16 == s_
        m = LogisticRegression(C=1e6, max_iter=3000).fit(X[tr], y16[tr])
        pp = m.predict_proba(X[te])[:, 1]
        lls.append(llv(y16[te], pp)); preds.append(pp); idxs.append(np.where(te)[0])
    return np.concatenate(lls), np.concatenate(preds), np.concatenate(idxs)

_, P0 = obj(**V7)
X0 = (P0["qb_prec"] + P0["off_prec"] + P0["def_prec"]).reshape(-1, 1)
ll0v, p0v, ix = wf_X_detail(X0, 2022, 2025)
ll0 = float(ll0v.mean())
print(f"  v7 TEST LL {ll0:.5f} (repro vs 0.62973)", flush=True)
assert abs(ll0 - 0.62973) <= 0.0007, f"v7 TEST repro FAILED: {ll0:.5f}"

_, P1 = obj(**V7, ttt_depth=2, ttt_passes=1)
X1 = (P1["qb_prec"] + P1["off_prec"] + P1["def_prec"]).reshape(-1, 1)
ll1v, p1v, _ = wf_X_detail(X1, 2022, 2025)
ll1 = float(ll1v.mean())

d = ll0v - ll1v
rng = np.random.default_rng(7)
bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
lo_, hi_ = np.percentile(bs, [2.5, 97.5])
yt = y16[ix]
msk = yt != 0.5
acc0 = float((((p0v > 0.5) == (yt > 0.5))[msk]).mean())
acc1 = float((((p1v > 0.5) == (yt > 0.5))[msk]).mean())
a0 = (p0v > 0.5)[msk]; a1 = (p1v > 0.5)[msk]; yy = (yt > 0.5)[msk]
b_ = int(((a1 == yy) & (a0 != yy)).sum()); c_ = int(((a0 == yy) & (a1 != yy)).sum())
try:
    from scipy.stats import binomtest
    pmc = float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0
except ImportError:
    pmc = -1.0
print(f"  v7+ttt TEST LL {ll1:.5f}  delta {ll0-ll1:+.5f}  CI [{lo_:+.5f},{hi_:+.5f}]", flush=True)
print(f"  acc {acc0:.4f} -> {acc1:.4f}   McNemar only-ttt-right {b_} / only-v7-right {c_}  p={pmc:.4f}", flush=True)
verdict = "ADOPT (point estimate better)" if ll1 < ll0 else "REJECT (TEST refused)"
print(f"  -> {verdict}", flush=True)
json.dump({"v7_test": round(ll0, 5), "ttt_test": round(ll1, 5),
           "delta": round(ll0 - ll1, 5), "ci": [round(float(lo_), 5), round(float(hi_), 5)],
           "acc": [round(acc0, 4), round(acc1, 4)],
           "mcnemar": {"b_ttt": b_, "c_v7": c_, "p": round(pmc, 4)},
           "config": {"ttt_depth": 2, "ttt_passes": 1},
           "dev": {"base": 0.62952, "ttt_d2": 0.62648}},
          open("data/nfl_ttt_test.json", "w"), indent=1)
np.save("data/nfl_v8_feature.npy", X1[:, 0])
print("wrote data/nfl_ttt_test.json + data/nfl_v8_feature.npy (G16-aligned ttt feature)", flush=True)
raise SystemExit(0)
cfg = {**V4, **V7}
_, P = run_config(**cfg, return_parts=True)
ts = P["qb_prec"] + P["off_prec"] + P["def_prec"]
VG = P["var"]
print(f"  aggregate sd: median {np.sqrt(np.median(VG)):.3f}, p90 {np.sqrt(np.percentile(VG,90)):.3f}", flush=True)

def wf_mc(lo, hi, n_draws=0):
    """walk-forward; n_draws>0 -> average sigmoid over N(b*ts, b^2*V) draws."""
    rng = np.random.default_rng(11)
    lls, accs = [], []
    for s_ in range(lo, hi + 1):
        tr = (seas16 < s_) & (y16 != 0.5)
        te = seas16 == s_
        m = LogisticRegression(C=1e6, max_iter=3000).fit(ts[tr].reshape(-1, 1), y16[tr])
        a_, b_ = float(m.intercept_[0]), float(m.coef_[0][0])
        z = a_ + b_ * ts[te]
        if n_draws:
            zs = z[:, None] + abs(b_) * np.sqrt(VG[te])[:, None] * rng.standard_normal((int(te.sum()), n_draws))
            p = (1.0 / (1.0 + np.exp(-zs))).mean(axis=1)
        else:
            p = 1.0 / (1.0 + np.exp(-z))
        lls.append(llv(y16[te], p))
        msk = y16[te] != 0.5
        accs.append(((p > 0.5) == (y16[te] > 0.5))[msk])
    return float(np.concatenate(lls).mean()), float(np.concatenate(accs).mean())

print("\nMC posterior-predictive (DEV 2017-21, gate -0.0020):", flush=True)
ll0, _ = wf_mc(2017, 2021, 0)
print(f"  point estimate (current)   {ll0:.5f}", flush=True)
for nd in (200, 500):
    llm, _ = wf_mc(2017, 2021, nd)
    print(f"  MC-PP {nd} draws            {llm:.5f}  ({ll0-llm:+.5f})", flush=True)
results["mcpp"] = (llm, {})
base_ll = results.get("baseline v7 (QB fusion only)", (9e9,))[0]
print(f"\nknobs beating baseline ({base_ll:.5f}) by >= 0.0010:")
for k, (ll, kw) in sorted(results.items(), key=lambda x: x[1][0]):
    if ll < base_ll - 0.0010 and kw:
        print(f"  {k:<36} {ll:.5f}  ({ll-base_ll:+.5f})")
json.dump({k: round(v[0], 5) for k, v in results.items()},
          open("data/nfl_knob_sweep.json", "w"), indent=1)
print("wrote data/nfl_knob_sweep.json + data/nfl_v5_feature.npy")
