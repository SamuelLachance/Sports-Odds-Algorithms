"""Stint-based xG/60 RAPM for NHL skaters — a trustworthy player-value display.

The per-shot RAPM (nhl_rapm.py) failed because raw xGoal is always positive, so
every offensive indicator absorbed a baseline and the ridge flattened. The
industry-standard fix (Evolving-Hockey style):

  1. Reconstruct STINTS — intervals of constant on-ice personnel (between any line
     change), keeping only 5v5 to control for strength state; each has a duration.
  2. Per stint, net xG = (home xGF - away xGA within the interval); response is
     the per-60 RATE = net_xg * 3600 / duration, weighted by duration (WLS).
  3. Ridge on offense/defense skater indicators + deployment context. Each
     player's coefficient is their teammate/opponent-adjusted net xG per 60.

Fit on a trailing window for stable current ratings. Display only (the predictive
team-aggregate test came back null, ledger row 8).

2026-07-31 rebuild — the shipped version had four defects that between them made
every defenceman look replacement-level and floated fourth-liners above stars:

  a. STRENGTH LEAK. The stint was 5v5 by the shift chart but the shots inside it
     were taken at every strength: 13.2% of the attributed xG came from shots
     whose own home_sk/away_sk said otherwise (2.4% of shots, but special-teams
     shots carry ~6x the xG). The shot rows always carried the strength state;
     the code just never read it. This was the single largest distortion — held-
     out stint R2 doubles (+0.0068 -> +0.0136) once the response is strictly 5v5.
  b. UNIDENTIFIED POSITION LEVEL. 98.6% of stint sides are exactly 3F+2D, so
     "all forwards +t / all defencemen -1.5t" moves no fitted value: the F-vs-D
     level is a (near-)null direction of the design. At lambda=150 against ~67M
     seconds of sample weight it was effectively unpenalised, so it was estimated
     off the 0.84% of TOI with a non-standard 4F/1D or 2F/3D side — where xGF/60
     runs 8.34 and 2.20 against a league 3.09. That handed forwards +0.78 and
     defencemen -1.48 in raw offence coefficients: a 2.26 xG/60 pure artefact,
     with 1.3% distribution overlap. Fixed by (i) explicit side-composition
     dummies that absorb the real effect and (ii) centring WITHIN position, the
     only defensible resolution of a direction the data cannot identify.
  c. NO DEPLOYMENT CONTEXT. Added home-ice, score state, and a puck-location
     proxy. True faceoff zone starts are NOT derivable from this repo's data
     (nhl_shots.csv has no faceoff rows and no coordinates), so the proxy is
     "which team shot in the 12s before the stint began" — a stint that starts
     right after the attacking team shot runs +1.78 xG/60, one after the
     defending team shot -0.78. Quality of competition was checked and is NOT a
     missing control: opponent quality faced has sd 0.010 against a player sd of
     0.19, and corr(net, QoC) = -0.08. The shared regression already handles it.
  d. NO SAMPLE-SIZE SHRINKAGE. lambda=150 against millions of seconds of weight
     is no regularisation at all; split-half reliability at the display floor is
     only 0.42. lambda is now set by held-out stint MSE and the display value is
     multiplied by an empirical-Bayes reliability rho = TOI/(TOI+EB_K), which
     shrinks toward the (zero) position mean.
"""
from __future__ import annotations

import bisect
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, "phase0")

LAMBDA = 60000.0       # chosen by held-out stint MSE (see _tuning note below)
MIN_TOI = 400.0        # min 5v5 seconds to be rated
DISPLAY_TOI = 60000.0  # 1000 min — mirrors the nhl_serve display floor; the
                       # 0-100 scale is standardised over this pool
EB_K = 83616.0         # empirical-Bayes half-reliability point, in 5v5 seconds
CSCALE = 50.0          # context columns are scaled so their penalty is alpha/2500
PROX_WINDOW = 12       # seconds before the stint for the puck-location proxy

# _tuning (2026-07-31, holdout = fit on odd game ids, score the even ones, and
# vice versa; weighted MSE on stint xG/60, R2 vs an intercept-only baseline):
#   lambda      150     +0.01160        controls          held-out R2
#   lambda    20000     +0.01336        none               +0.01062
#   lambda    60000     +0.01363  <-    +composition       +0.01112
#   lambda   100000     +0.01361        +home              +0.01118
#   lambda   150000     +0.01352        +score state       +0.01139
#   lambda   400000     +0.01314        +puck proxy        +0.01363  <-
# Split-half correlation is NOT usable to tune lambda here: heavier shrinkage
# scales every coefficient by a TOI-dependent factor that is itself stable
# across halves, so r climbs monotonically (0.386 -> 0.767) even where held-out
# MSE is getting worse.

COMP_LEVELS = (0, 1, 3, 4, 5)      # side D-counts; 2 is the reference
SCORE_LEVELS = (-3, -2, -1, 1, 2, 3)
CTX = (["home_att"]
       + [f"attD{k}" for k in COMP_LEVELS] + [f"dfnD{k}" for k in COMP_LEVELS]
       + [f"score{k:+d}" for k in SCORE_LEVELS] + ["prox_att", "prox_dfn"])
NCTX = len(CTX)


def _fit_seasons(n: int = 4) -> set[int]:
    """Trailing window for current ratings: the latest n distinct seasons in the
    spine (auto-rolls each season; was hardcoded {20222023..20252026})."""
    with open("data/nhl_games.csv", encoding="utf-8") as fh:
        seasons = sorted({int(r["season"]) for r in csv.DictReader(fh)})
    return set(seasons[-n:])


FIT_SEASONS = _fit_seasons()


def stints_for_game(gshifts, goalies):
    """Yield (period, t0, t1, on) constant-personnel intervals.
    gshifts rows: (player, team, period, start_s, end_s). Team codes come from meta."""
    by_period = defaultdict(list)
    for (p, t, pe, s0, s1) in gshifts:
        by_period[pe].append((p, t, s0, s1))
    for pe, rows in by_period.items():
        bps = sorted({s for (_, _, s0, s1) in rows for s in (s0, s1)})
        for i in range(len(bps) - 1):
            t0, t1 = bps[i], bps[i + 1]
            if t1 <= t0:
                continue
            mid = (t0 + t1) / 2.0
            on = defaultdict(list)
            for (p, t, s0, s1) in rows:
                if s0 <= mid < s1 and p not in goalies:
                    on[t].append(p)
            yield pe, t0, t1, on


def load_shifts(keep):
    """game_id -> list of (player, team, period, start_s, end_s), window only.

    Filtered while reading: the full file is 16 seasons and ~4x the rows we fit
    on, and materialising all of it as tuples costs GB for nothing."""
    sh = defaultdict(list)
    with open("data/nhl_shifts.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if not r or not r[0].isdigit():
                continue
            g = int(r[0])
            if g in keep:
                sh[g].append((int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])))
    return sh


def infer_positions(seed, h5, a5, players, iters=4):
    """F/D label for every skater in the window.

    data/nhl_player_names.json only covers current rosters (~700 of the 1,300
    skaters in a 4-season window), but the label is needed for every column: the
    F-vs-D level is unidentified and has to be pinned by construction. A 5-man
    side is 3F+2D, so a forward's four on-ice team-mates hold two defencemen and
    a defenceman's hold one — score each player by the share of his
    position-labelled team-mates who are D (F ~ 0.50, D ~ 0.25) and iterate,
    feeding assignments back in as labels. Scored against the known labels this
    rule is exact (706/706 in the 2022-26 window)."""
    lab = dict(seed)
    for _ in range(iters):
        num = defaultdict(float)
        den = defaultdict(float)
        for side in (h5, a5):
            for row in side:
                lp = [lab.get(p) for p in row]
                for j, p in enumerate(row):
                    oth = [lp[k] for k in range(5) if k != j and lp[k]]
                    if not oth:
                        continue
                    num[p] += sum(1 for o in oth if o == "D")
                    den[p] += len(oth)
        new = dict(seed)
        for p in players:
            if p not in seed and den[p] >= 20:
                new[p] = "D" if num[p] / den[p] < 0.375 else "F"
        lab = new
    return lab


def build_stints():
    """Scan the window once and return per-stint arrays."""
    meta = {}
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        meta[int(r["game_id"])] = (r["date"], r["home"], r["away"], int(r["season"]))
    keep = {g for g, m in meta.items() if m[3] in FIT_SEASONS}

    goalies = set()
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        try:
            goalies.add(int(r["goalie_id"]))
        except (KeyError, ValueError):
            pass

    # Per game/period, parallel arrays sorted by second. xg5 is zeroed for any
    # shot that was not actually 5v5 (the strength leak, defect (a)); shot TIMES
    # and goals stay all-strengths, because the puck-location proxy and the score
    # state describe the state of the game, not the 5v5 response.
    shots = defaultdict(lambda: defaultdict(lambda: ([], [], [], [])))
    goals = defaultdict(list)
    with open("data/nhl_shots.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            gid = int(r[0])
            if gid not in keep:
                continue
            sec, xg, is5 = int(r[2]), float(r[5]), r[7] == "5" and r[8] == "5"
            sec_l, team_l, xg5_l, xga_l = shots[gid][int(r[1])]
            sec_l.append(sec)
            team_l.append(r[3])
            xg5_l.append(xg if is5 else 0.0)
            xga_l.append(xg)
            if r[6] == "1":
                goals[gid].append((int(r[1]), sec, r[3]))

    shifts = load_shifts(keep)
    H, A, DUR, XGF, XGA, SDF, PRX = [], [], [], [], [], [], []
    ngames = 0
    leak_all = leak_5 = 0.0        # xG landing inside kept stints, all vs 5v5 only
    for gid, per_map in shots.items():
        gshifts = shifts.get(gid)
        if not gshifts:
            continue
        home, away = meta[gid][1], meta[gid][2]
        prep = {}
        for pe, (sec_l, team_l, xg5_l, xga_l) in per_map.items():
            order = sorted(range(len(sec_l)), key=lambda i: sec_l[i])
            secs = [sec_l[i] for i in order]
            hom = [team_l[i] == home for i in order]
            xg5 = [xg5_l[i] for i in order]
            prep[pe] = (secs, hom,
                        np.cumsum([0.0] + [x if h else 0.0 for x, h in zip(xg5, hom)]),
                        np.cumsum([0.0] + [x if not h else 0.0 for x, h in zip(xg5, hom)]),
                        np.cumsum([0.0] + [xga_l[i] for i in order]))
        gg = sorted(goals.get(gid, ()))
        ngames += 1
        for pe, t0, t1, on in stints_for_game(gshifts, goalies):
            hh, aa = on.get(home, []), on.get(away, [])
            if len(hh) != 5 or len(aa) != 5:
                continue
            dur = t1 - t0
            if dur < 3:
                continue
            secs, hom, ch, ca, call = prep.get(
                pe, ([], [], np.zeros(1), np.zeros(1), np.zeros(1)))
            lo = bisect.bisect_left(secs, t0)
            hi = bisect.bisect_left(secs, t1)
            leak_all += float(call[hi] - call[lo])
            leak_5 += float(ch[hi] - ch[lo]) + float(ca[hi] - ca[lo])
            prox = 0
            for k in range(bisect.bisect_left(secs, t0 - PROX_WINDOW), lo):
                prox = 1 if hom[k] else -1        # most recent shot in the window
            gh = ga = 0
            for (gp, gs, gt) in gg:
                if (gp, gs) >= (pe, t0):
                    break
                if gt == home:
                    gh += 1
                else:
                    ga += 1
            H.append(hh); A.append(aa); DUR.append(dur)
            XGF.append(float(ch[hi] - ch[lo])); XGA.append(float(ca[hi] - ca[lo]))
            PRX.append(prox); SDF.append(max(-3, min(3, gh - ga)))
    print(f"{ngames:,} games -> {len(H):,} 5v5 stints; xG falling inside them "
          f"{leak_all:,.0f}, of which strictly 5v5 {leak_5:,.0f} "
          f"({(1-leak_5/max(leak_all,1e-9))*100:.2f}% dropped as a strength leak)",
          flush=True)
    return H, A, DUR, XGF, XGA, np.asarray(SDF, dtype=np.int8), PRX


def main():
    print(f"[nhl_rapm2] FIT_SEASONS (derived): {sorted(FIT_SEASONS)}", flush=True)
    H, A, DUR, XGF, XGA, sdiff, PRX = build_stints()

    pid = {}
    for row in H:
        for p in row:
            pid.setdefault(p, len(pid))
    for row in A:
        for p in row:
            pid.setdefault(p, len(pid))
    n = len(pid)
    idx2p = np.empty(n, dtype=np.int64)
    for p, i in pid.items():
        idx2p[i] = p

    seed = {}
    try:
        for k, v in json.load(open("data/nhl_player_names.json", encoding="utf-8")).items():
            if v.get("pos") in ("D", "C", "L", "R"):
                seed[int(k)] = "D" if v["pos"] == "D" else "F"
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    lab = infer_positions({p: v for p, v in seed.items() if p in pid}, H, A, list(pid))
    isD = np.array([lab.get(int(p), "F") == "D" for p in idx2p])
    print(f"{n:,} skaters; {isD.sum():,} D / {(~isD).sum():,} F "
          f"({sum(1 for p in idx2p if int(p) in seed):,} from the roster map)", flush=True)

    Hi = np.array([[pid[p] for p in r] for r in H], dtype=np.int32)
    Ai = np.array([[pid[p] for p in r] for r in A], dtype=np.int32)
    dur = np.asarray(DUR, dtype=float)
    xgf = np.asarray(XGF, dtype=float)
    xga = np.asarray(XGA, dtype=float)
    prox = np.asarray(PRX, dtype=np.int8)
    S = len(dur)
    nD_h, nD_a = isD[Hi].sum(1), isD[Ai].sum(1)

    # two-way RAPM: each stint -> two rows (each team attacking). A player has an
    # OFFENSE column (cols 0..n) and a DEFENSE column (cols n..2n). Response = the
    # attacking team's xGF per 60. off coef = danger created; def coef = danger
    # allowed (lower/negative = better suppressor).
    R = 2 * S
    off = np.empty((R, 5), dtype=np.int32)
    dfn = np.empty((R, 5), dtype=np.int32)
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

    add_ctx(0, np.arange(R) % 2 == 0)                         # attacking team is home
    for j, k in enumerate(COMP_LEVELS):
        add_ctx(1 + j, nDo == k)
        add_ctx(1 + len(COMP_LEVELS) + j, nDd == k)
    for j, k in enumerate(SCORE_LEVELS):
        add_ctx(1 + 2 * len(COMP_LEVELS) + j, sdo == k)
    add_ctx(NCTX - 2, prx > 0)
    add_ctx(NCTX - 1, prx < 0)

    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(R, 2 * n + NCTX))
    print(f"{R:,} stint-sides, {n:,} skaters, {NCTX} context columns; "
          f"mean 5v5 xGF/60 {np.average(y, weights=w):.3f}", flush=True)
    m = Ridge(alpha=LAMBDA, fit_intercept=True, solver="sparse_cg", max_iter=2000, tol=1e-6)
    m.fit(X, y, sample_weight=w)
    coef = m.coef_

    pcount = np.zeros(n)
    np.add.at(pcount, Hi.ravel(), np.repeat(dur, 5))
    np.add.at(pcount, Ai.ravel(), np.repeat(dur, 5))

    # Centre WITHIN position. With the composition dummies in the design, "all F
    # +t / all D -1.5t" is an exact null direction: the data cannot identify the
    # F-vs-D level, so it must be pinned, not estimated (defect (b)).
    OFF = coef[:n].copy()
    DEF = coef[n:2 * n].copy()
    for msk in (isD, ~isD):
        OFF[msk] -= OFF[msk].mean()
        DEF[msk] -= DEF[msk].mean()
    DEF = -DEF                                   # higher = better suppression
    rel = pcount / (pcount + EB_K)               # empirical-Bayes reliability
    OFF, DEF = OFF * rel, DEF * rel              # shrink toward the position mean
    NET = OFF + DEF

    pool = NET[pcount >= DISPLAY_TOI]
    if len(pool) < 50:
        pool = NET[pcount >= MIN_TOI]
    mu, sd = float(pool.mean()), float(pool.std())
    out = {}
    for i in range(n):
        if pcount[i] < MIN_TOI:
            continue
        out[str(int(idx2p[i]))] = {
            "off": round(float(OFF[i]), 3), "def": round(float(DEF[i]), 3),
            "net": round(float(NET[i]), 3), "toi_min": round(pcount[i] / 60.0, 1),
            "pos": "D" if isD[i] else "F", "rel": round(float(rel[i]), 3),
            "rating": round(float(np.clip(50 + 15 * (NET[i] - mu) / (sd + 1e-9), 1, 99)), 1),
        }
    for j, cname in enumerate(CTX):
        print(f"  ctx {cname:<10} {coef[2*n+j]*CSCALE:+7.3f} xG/60")
    dm = np.median(NET[(pcount >= DISPLAY_TOI) & isD]) if (pcount >= DISPLAY_TOI).any() else 0.0
    fm = np.median(NET[(pcount >= DISPLAY_TOI) & ~isD]) if (pcount >= DISPLAY_TOI).any() else 0.0
    print(f"display pool medians: F {fm:+.3f}  D {dm:+.3f}  gap {fm-dm:+.3f}", flush=True)

    # temp + os.replace: a kill mid-dump (e.g. nhl_weekly timeout) must not
    # leave truncated JSON that bricks every later nhl_serve run
    tmp = "data/nhl_rapm2_ratings.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, "data/nhl_rapm2_ratings.json")
    print(f"{len(out):,} rated -> data/nhl_rapm2_ratings.json")


if __name__ == "__main__":
    main()
