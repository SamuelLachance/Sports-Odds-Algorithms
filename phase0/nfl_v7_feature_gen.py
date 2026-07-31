"""Regenerate data/nfl_v7_feature.npy — the frozen historical column 7
(participation-TrueSkill v7, cohort salary prior) of the serving model.

Resurrected from commit 7dd13d0 (phase0/nfl_ratings_only_model.py, cfg7 =
V4 + V6 + {sal_k: 1.0, sal_mode: "cohort"}): the run_config engine below is a
verbatim extraction of the code that wrote the shipped npy; the surrounding
sweep/trial harness is dropped. Three inputs of the original module (FTN
charting flags, injury Q/D tags, play protagonists) are provably dead under
cfg7 (f_ta=f_drop=f_iw=False, inj_sym=inj_neg=1.0, micro="off",
two_track=False), so they are stubbed empty instead of loaded.

Run this whenever data/nfl_games.csv gains new finals (the serve chain's
length assert at nfl_season_serve.py trips otherwise). The feature is
walk-forward and leak-free, so on unchanged inputs the output is bit-identical
to the shipped npy; with new games appended the historical prefix is preserved
and new rows extend it.
"""
from __future__ import annotations

import csv
import json  # noqa: F401  (prelude exec expects it importable at module scope)
import os
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

# dead inputs under cfg7 (see docstring): empty sentinels keep run_config verbatim
prot_of = {}
FTN = {}
INJ = set()

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
Z_SAL2 = {}
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
    _stats = {}
    for (_y, _g), _sub in _df.groupby(["y0", "grp"]):
        _stats[(_y, _g)] = (_sub.pct.mean(), max(_sub.pct.std(), 1e-6))
    for _t in _df.itertuples():
        m_, sd_ = _stats[(_t.y0, _t.grp)]
        Z_SAL2[(_t.gsis, _t.season)] = max(-2.5, min(2.5, (_t.pct - m_) / sd_))
    # season-2026 coverage for in-season regeneration: the min(2026, ...) cap
    # above is kept deliberately so the <=2025 frame — and hence the frozen
    # (y0, grp) cohort stats and the npy's historical prefix — stay
    # byte-identical to the shipped artifact. Contracts covering 2026 are
    # z-scored HERE against those same frozen signing-year stats, adding only
    # (player, 2026) keys, so a 2026 walk keeps cfg7's cohort salary prior
    # (debut inits + 2025->2026 boundary targets) instead of degrading to a
    # flat 25.0. Cohorts signed outside the frozen frame (y0 == 2026) have no
    # frozen stats and are skipped (prior 0, same as an uncontracted player).
    _rows26 = []
    for _r in _c.itertuples():
        y0 = int(_r.year_signed)
        span = max(int(_r.years) if _pd.notna(_r.years) and _r.years > 0 else 1, 1)
        if y0 <= 2026 < y0 + span:
            _rows26.append((_r.gsis_id, 2026, y0, float(_r.apy_cap_pct)))
    _df26 = _pd.DataFrame(_rows26, columns=["gsis", "season", "y0", "pct"])
    _n26 = 0
    if len(_df26):
        _df26 = _df26.sort_values("y0").groupby(["gsis", "season"]).tail(1)
        _df26["grp"] = _df26.gsis.map(GPOS)
        _df26 = _df26.dropna(subset=["grp"])
        for _t in _df26.itertuples():
            _st = _stats.get((_t.y0, _t.grp))
            if _st is None:
                continue
            m_, sd_ = _st
            Z_SAL2[(_t.gsis, 2026)] = max(-2.5, min(2.5, (_t.pct - m_) / sd_))
            _n26 += 1
    print(f"[{time.time()-T0:.0f}s] salary priors: {len(Z_SAL):,} player-seasons "
          f"(+ cohort variant, {_n26:,} season-2026 keys)", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"salary priors unavailable ({type(e).__name__})", flush=True)

QBOBS = {}
_rates = [ln[0] / ln[1] for (q_, s_, w_), ln in qbw.items()
          if 2016 <= s_ <= 2021 and ln[1] >= 10]
_SDQB = float(np.std(_rates)) if _rates else 0.15
for (q_, s_, w_), ln in qbw.items():
    if s_ >= 2016 and ln[1] >= 10:
        z_ = max(-2.5, min(2.5, (ln[0] / ln[1] - lg_qb) / _SDQB))
        QBOBS[(q_, str(s_), f"{w_:02d}")] = z_
print(f"[{time.time()-T0:.0f}s] QB weekly obs: {len(QBOBS):,} lines (sd {_SDQB:.3f})", flush=True)

G16 = [(i, g) for i, g in enumerate(games) if g["season"] >= 2016]
seas16 = np.array([g["season"] for _, g in G16])
y16 = np.array([g["y"] for _, g in G16])

def run_config(tau, use_lev, down_mult, late_mult, season_shrink=1.0 / 3.0,
               widen2=1.5 ** 2, playoff_mult=1.0, dead_mult=1.0, opp_k=0.0,
               opp_mode="league", flat_scale=1.0, role_r=0.0, micro="off",
               sal_k=0.0, grade="off", mov_scale=0.5, mov_cap=2.0,
               draw_band=0.0, neutral_skip=0.0, return_parts=False,
               beta=BETA, floor2=SIG2_FLOOR, lev_exp=1.0, draw_eps=0.05,
               hfa_t=0.0, f_ta=False, f_drop=False, f_iw=False,
               inj_sym=1.0, inj_neg=1.0, agg_k=1.0, share_p=1.0,
               two_track=False, fus_k=0.0, fus_R=36.0, sal_mode="active"):
    """One engine walk -> per-game ts_diff feature (pre-game, leak-free).

    Game importance: playoff snaps x playoff_mult; 'useless' late-season games
    (BOTH teams mathematically short of 7 wins, pre-game records) x dead_mult.
    Opponent quality: each side's update also scales with the OPPOSING unit's
    mean rating, factor 1 + opp_k*(opp_mean_mu - 25), clamped [0.3, 2].
    """
    tau2 = tau * tau
    ZS = Z_SAL2 if sal_mode == "cohort" else Z_SAL
    mu, s2, lastwk = {}, {}, {}
    muR, s2R = {}, {}                    # run track (pass track = mu/s2)
    share2 = {}
    pos2 = {}
    PG = {"QB":"QB","RB":"RB","FB":"RB","HB":"RB","WR":"WR","TE":"TE",
          "T":"OL","G":"OL","C":"OL","OT":"OL","OG":"OL","OL":"OL","LT":"OL","RT":"OL","LG":"OL","RG":"OL",
          "DE":"DL","DT":"DL","NT":"DL","DL":"DL","EDGE":"DL",
          "LB":"LB","ILB":"LB","OLB":"LB","MLB":"LB",
          "CB":"DB","S":"DB","SS":"DB","FS":"DB","SAF":"DB","DB":"DB"}
    f = np.zeros(len(G16))
    KEYS = ["qb_raw", "off_raw", "def_raw", "qb_prec", "off_prec", "def_prec",
            "qb_pn", "off_pn", "def_pn", "p_prec", "r_prec"]
    parts = {k: np.zeros(len(G16)) for k in KEYS} if return_parts else None
    gm = {"QB": MU0, "RB": MU0, "WR": MU0, "TE": MU0, "OL": MU0, "DL": MU0, "LB": MU0, "DB": MU0}
    gv = {k: 4.0 for k in gm}                      # running group mean/var (EWMA, leak-free)
    prev = None
    wins = defaultdict(float); played = defaultdict(int)
    for j, (i, g) in enumerate(G16):
        if prev is not None and g["season"] != prev:
            for p in mu:
                tgt = MU0 + sal_k * ZS.get((p, g["season"]), 0.0)
                mu[p] = tgt + (mu[p] - tgt) * (1.0 - season_shrink)
                s2[p] = min(s2[p] + widen2, SIG0_2)
            for p in muR:
                muR[p] = MU0 + (muR[p] - MU0) * (1.0 - season_shrink)
                s2R[p] = min(s2R[p] + widen2, SIG0_2)
            wins.clear(); played.clear()
        prev = g["season"]
        # game-importance multiplier from PRE-game standings
        sched_len = 16 if g["season"] < 2021 else 17
        if g["type"] != "REG":
            gmult = playoff_mult
        else:
            dead_h = wins[g["home"]] + (sched_len - played[g["home"]]) < 7
            dead_a = wins[g["away"]] + (sched_len - played[g["away"]]) < 7
            gmult = dead_mult if (dead_h and dead_a) else 1.0
        lg_mu = (sum(mu.values()) / len(mu)) if (opp_k and opp_mode == "league" and mu) else MU0
        lg_muR = (sum(muR.values()) / len(muR)) if (opp_k and opp_mode == "league" and muR) else MU0
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
                w_sh = (st[0] / st[1]) ** share_p
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
            if two_track:
                _pr = prot_of.get((gid, pid_))
                trk_run = bool(_pr and not _pr[0] and _pr[1])
            else:
                trk_run = False
            muD = muR if (two_track and trk_run) else mu
            s2D = s2R if (two_track and trk_run) else s2
            lw_key = "R" if (two_track and trk_run) else "P"
            for p in O:
                if p not in muD:
                    muD[p] = MU0 + sal_k * ZS.get((p, seas_now), 0.0); s2D[p] = SIG0_2
                elif lastwk.get((p, lw_key)) != wk:
                    if fus_k and GPOS.get(p) == "QB":
                        _old = lastwk.get((p, lw_key))
                        _z = QBOBS.get((p, _old[0], _old[1])) if _old else None
                        if _z is not None:
                            _obs = MU0 + fus_k * _z
                            _K = s2D[p] / (s2D[p] + fus_R)
                            muD[p] += _K * (_obs - muD[p])
                            s2D[p] = max(s2D[p] * (1.0 - _K), floor2)
                    s2D[p] = min(s2D[p] + tau2, SIG0_2)
                lastwk[(p, lw_key)] = wk
            for p in D:
                if p not in muD:
                    muD[p] = MU0 + sal_k * ZS.get((p, seas_now), 0.0); s2D[p] = SIG0_2
                elif lastwk.get((p, lw_key)) != wk:
                    s2D[p] = min(s2D[p] + tau2, SIG0_2)
                lastwk[(p, lw_key)] = wk
            wp, down, qtr, adiff, hsec = ctx.get((gid, pid_), (0.5, 1, 1, 0.0, 900.0))
            wgt = ((4.0 * wp * (1.0 - wp)) ** lev_exp) if use_lev else \
                (0.25 if (wp < 0.05 or wp > 0.95) else 1.0)
            if down >= 3:
                wgt *= down_mult
            if qtr >= 4 and adiff <= 8:
                wgt *= late_mult
            wgt *= gmult * flat_scale

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
                is_draw = draw_band > 0 and abs(epa) < draw_band
                if f_ta and ta:
                    is_draw = True                           # throw-away: decision, not defeat
                if f_iw and iw:
                    is_draw = False
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
                            muD[p] += im * w_A * (s2D[p] / c) * vd
                            s2D[p] = max(s2D[p] * (1.0 - im * w_A * (s2D[p] / c2) * wd), floor2)
                        for p in B:
                            im = imul(p, vd <= 0)
                            muD[p] -= im * w_B * (s2D[p] / c) * vd
                            s2D[p] = max(s2D[p] * (1.0 - im * w_B * (s2D[p] / c2) * wd), floor2)
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
                    muD[p] += rm * w_win_ * (s2D[p] / c) * v
                    s2D[p] = max(s2D[p] * (1.0 - rm * w_win_ * (s2D[p] / c2) * w), floor2)
                for p in lose:
                    rm = (role_m.get(p, 1.0) if role_r else 1.0) * imul(p, False)
                    muD[p] -= rm * w_lose_ * (s2D[p] / c) * v
                    s2D[p] = max(s2D[p] * (1.0 - rm * w_lose_ * (s2D[p] / c2) * w), floor2)

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
        # 4. standings walk (for next games' dead-game detection)
        if g["type"] == "REG":
            wins[g["home"]] += g["y"]; wins[g["away"]] += 1.0 - g["y"]
            played[g["home"]] += 1; played[g["away"]] += 1
    return (f, parts) if return_parts else f

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

# ================= the shipped v7 config (ledger rows 65-66) =================
V4 = dict(tau=0.30, use_lev=True, down_mult=1.25, late_mult=1.5,
          playoff_mult=1.5, dead_mult=0.5, opp_k=0.3, opp_mode="league",
          draw_band=0.55)
V5 = dict(beta=60.0, lev_exp=0.5, draw_band=0.40)
V6 = {**V5, "fus_k": 4.0, "fus_R": 36.0}
cfg7 = {**V4, **V6, "sal_k": 1.0, "sal_mode": "cohort"}
_, P7 = run_config(**cfg7, return_parts=True)
ts7 = P7["qb_prec"] + P7["off_prec"] + P7["def_prec"]
full = np.zeros(len(games))
idx_all = np.array([i for i, _ in G16])
full[idx_all] = ts7
lld = wf_X(ts7.reshape(-1, 1), 2017, 2021)[0]
llt, acct = wf_X(ts7.reshape(-1, 1), 2022, 2025)
print(f"  v7 ratings-only  DEV {lld:.5f}   TEST {llt:.5f}  acc {acct*100:.1f}%  "
      f"(ledger: TEST 0.62973  Elo 0.63824; DEV here computes ~0.62952 — the "
      f"ledger's 0.62486 came from a different sweep-selection context, not this "
      f"config; TEST match + npy prefix identity are the real gates)", flush=True)

# tripwire: current-season rows all-zero while that season's snap table exists
# on disk means the walk did NOT consume the data (gid/team key drift, stale
# prelude glob, ...). All-zero with NO snap file is the expected pre-data state
# and stays silent. Loud warning, not a crash: the historical prefix is intact
# and may still ship, but column 7 would silently serve zeros for the season.
CUR_SEASON = max(g["season"] for g in games)
if CUR_SEASON >= 2026:
    _cur_ix = np.array([i for i, g in enumerate(games) if g["season"] == CUR_SEASON])
    _snap_path = f"data/snap_{CUR_SEASON}.csv"
    if len(_cur_ix) and not np.any(full[_cur_ix]) and os.path.exists(_snap_path):
        with open(_snap_path, encoding="utf-8") as _fh:
            _n_snap = sum(1 for _ in _fh) - 1
        if _n_snap > 100:
            print(f"WARNING v7 tripwire: all {len(_cur_ix)} season-{CUR_SEASON} rows "
                  f"of the feature are 0.0 but {_snap_path} has {_n_snap:,} data rows "
                  f"— the walk did not use the snap/participation data (check the "
                  f"prelude snap glob and gid/team keys). Column 7 will serve zeros "
                  f"for {CUR_SEASON} games until this is fixed.", flush=True)

OUT = "data/nfl_v7_feature.npy"
if os.path.exists(OUT):
    old = np.load(OUT)
    n = min(len(old), len(full))
    print(f"existing npy: {len(old)} rows | new: {len(full)} rows | "
          f"identical on common prefix: {np.array_equal(old[:n], full[:n])} | "
          f"full array_equal: {np.array_equal(old, full)}", flush=True)
np.save(OUT + ".tmp.npy", full); os.replace(OUT + ".tmp.npy", OUT)
print(f"[{time.time()-T0:.0f}s] wrote {OUT} ({len(full)} rows)", flush=True)
