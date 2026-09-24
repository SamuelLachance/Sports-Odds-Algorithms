"""bt_nhl_rapmel -- F/D position labels for a stint-RAPM fit window (DEV).

WHY THIS EXISTS (pre-result, plumbing fix): the spec asks for
nhl_rapm2.infer_positions seeded from data/nhl_player_names.json. That helper
was validated on the 2022-26 window where the name map labels ~700 of ~1,300
skaters. On DEV windows the map (a 2025-26 roster file) labels only 40 of 891
skaters in 2010-11 and 246 of 1,217 in 2014-17, and the helper then
OSCILLATES with period 2 (D share 0.44-0.54, only 9-15% of 5v5 sides come out
3F+2D). Running it as-is would put roughly half the league in the wrong
position group, corrupting the within-position centring, the composition
dummies and the replacement prior. It is therefore replaced by a labeller that
enforces the same structural fact the helper relies on (a 5v5 side is 3F+2D):

  1. per team-season (sides of one team, normalised code, one season of the
     window): with A = sqrt(w)[M | -2] (M = side x player incidence), the exact
     null vector is (0.4*1, 1) and x_true - 0.4 is a near-null direction. Take
     the k smallest non-trivial eigenvectors of A'A (k in 1,2,3,4,6,8), run an
     alternating projection from a 5v5-TOI/game threshold start (13.5/14.5/15.5
     min), each step labelling as D the top-z players holding 40% of on-ice
     time; keep the candidate with the highest share of 3F+2D sides;
  2. polish that team-season by greedy single flips and teammate swaps on
     sum w (nD - 2)^2;
  3. TOI-weighted vote across the window's team-seasons;
  4. name-map seeds override (the spec's seeding) and a global greedy polish
     with seeds fixed.

Labels use only the fit window's own stints (static labels, never outcomes).
Validation (reported by the fit script): share of 5v5 sides that are exactly
3F+2D, and unsupervised agreement with the name-map labels.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

INIT_THR = (13.5, 14.5, 15.5)
KS = (1, 2, 3, 4, 6, 8)


def _side_lists(sides, n):
    S = len(sides)
    inc_s = np.repeat(np.arange(S), 5)
    inc_p = sides.ravel()
    o = np.argsort(inc_p, kind="stable")
    inc_s = inc_s[o]
    st_ = np.searchsorted(inc_p[o], np.arange(n + 1))
    return [inc_s[st_[p]:st_[p + 1]] for p in range(n)]


def polish(sides, ww, n, x, fixed, max_rounds=60):
    """Greedy single flips + teammate swaps minimising sum w (nD-2)^2."""
    x = x.copy()
    c = x[sides].sum(1)
    SL = _side_lists(sides, n)
    for _ in range(max_rounds):
        nf = 0
        for p in range(n):
            if fixed[p]:
                continue
            ss = SL[p]
            if not len(ss):
                continue
            cc = c[ss]
            w2 = ww[ss]
            d = np.sum(w2 * (2 * cc - 3)) if x[p] == 0 else np.sum(w2 * (5 - 2 * cc))
            if d < -1e-9:
                dl = 1 if x[p] == 0 else -1
                x[p] += dl
                c[ss] += dl
                nf += 1
        ns = 0
        for p in range(n):
            if fixed[p]:
                continue
            ss = SL[p]
            if not len(ss):
                continue
            cand = np.bincount(sides[ss].ravel(), weights=np.repeat(ww[ss], 5), minlength=n)
            cand[p] = 0
            opp = np.where((x != x[p]) & ~fixed & (cand > 0))[0]
            if not len(opp):
                continue
            for q in opp[np.argsort(-cand[opp], kind="stable")[:4]]:
                sq = SL[q]
                both = np.intersect1d(ss, sq, assume_unique=True)
                op = np.setdiff1d(ss, both, assume_unique=True)
                oq = np.setdiff1d(sq, both, assume_unique=True)
                dp = 1 if x[p] == 0 else -1
                dq = -dp
                cp, cq = c[op], c[oq]
                d = (np.sum(ww[op] * ((cp + dp - 2) ** 2 - (cp - 2) ** 2))
                     + np.sum(ww[oq] * ((cq + dq - 2) ** 2 - (cq - 2) ** 2)))
                if d < -1e-9:
                    x[p] += dp
                    x[q] += dq
                    c[op] += dp
                    c[oq] += dq
                    ns += 1
                    break
        if nf == 0 and ns == 0:
            break
    return x


def _rate(x, sl, w):
    return float(np.average(x[sl].sum(1) == 2, weights=w))


def _team_season(sl_loc, w, m, tpg_loc):
    S = len(sl_loc)
    rows = np.repeat(np.arange(S), 6)
    cols = np.column_stack([sl_loc, np.full(S, m)]).ravel()
    vals = np.column_stack([np.ones((S, 5)), np.full(S, -2.0)]).ravel()
    A = sparse.csr_matrix((vals * np.repeat(np.sqrt(w), 6), (rows, cols)), shape=(S, m + 1))
    _ev, V = np.linalg.eigh((A.T @ A).toarray())
    T = np.bincount(sl_loc.ravel(), weights=np.repeat(w, 5), minlength=m)
    best = None
    for k in KS:
        k = min(k, m)
        B = V[:m, 1:k + 1] - np.outer(np.full(m, 0.4), V[m, 1:k + 1])
        Q, _ = np.linalg.qr(B)
        for thr in INIT_THR:
            x = (tpg_loc > thr).astype(float)
            for _ in range(100):
                z = Q @ (Q.T @ (x - 0.4))
                o = np.argsort(-z, kind="stable")
                cs = np.cumsum(T[o]) / T.sum()
                xn = np.zeros(m)
                xn[o[:np.searchsorted(cs, 0.4) + 1]] = 1
                if np.array_equal(xn, x):
                    break
                x = xn
            xi = x.astype(np.int64)
            r = _rate(xi, sl_loc, w)
            if best is None or r > best[0]:
                best = (r, xi)
    x = polish(sl_loc, w, m, best[1], np.zeros(m, bool))
    return x, T, best[0], _rate(x, sl_loc, w)


def label_positions(H, A, dur, gid, team_home, team_away, season_of, seed, fix_seed=True):
    """H, A: (S,5) LOCAL player indices; dur (S,); gid (S,);
    team_home/team_away/season_of: dicts gid -> value (normalised team codes);
    seed: {local index: 'D'|'F'}.  Returns (isD bool array, diagnostics)."""
    n = int(max(H.max(), A.max())) + 1
    sides = np.concatenate([H, A])
    ww = np.concatenate([dur, dur]).astype(float)
    gid2 = np.concatenate([gid, gid])
    team = np.array([team_home[g] for g in gid] + [team_away[g] for g in gid])
    sea = np.array([season_of[g] for g in gid2])
    toi = np.bincount(sides.ravel(), weights=np.repeat(ww, 5), minlength=n)
    pg = np.unique(np.column_stack([np.repeat(gid2, 5), sides.ravel()]), axis=0)
    ng = np.bincount(pg[:, 1], minlength=n)
    tpg = toi / np.maximum(ng, 1) / 60.0
    vote = np.zeros(n)
    ts_rates = []
    for tm in np.unique(team):
        for s in np.unique(sea):
            msk = (team == tm) & (sea == s)
            if not msk.any():
                continue
            sl = sides[msk]
            w = ww[msk]
            loc_p = np.unique(sl)
            sl_loc = np.searchsorted(loc_p, sl)
            x, T, r0, r1 = _team_season(sl_loc, w, len(loc_p), tpg[loc_p])
            ts_rates.append(r1)
            np.add.at(vote, loc_p, T * (2 * x - 1))
    x = (vote > 0).astype(np.int64)
    fixed = np.zeros(n, bool)
    unsup = polish(sides, ww, n, x, fixed)
    agree = [int(unsup[p] == (1 if v == "D" else 0)) for p, v in seed.items()]
    if fix_seed:
        for p, v in seed.items():
            x[p] = 1 if v == "D" else 0
            fixed[p] = True
        x = polish(sides, ww, n, x, fixed)
    else:
        x = unsup
    diag = {"share_sides_3F2D": _rate(x, sides, ww),
            "share_sides_3F2D_unsupervised": _rate(unsup, sides, ww),
            "unsupervised_agreement_with_name_map": (float(np.mean(agree)) if agree else None),
            "n_name_map_labels": len(seed),
            "worst_team_season_3F2D": float(min(ts_rates)),
            "D_share_players": float(x.mean())}
    return x.astype(bool), diag
