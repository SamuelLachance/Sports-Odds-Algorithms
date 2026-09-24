"""Breakthrough program (NFL, ideas leg) - seed-level lock with NFL tiebreakers. DEV ONLY.

Candidate 'seedlock_exact': the proxy lock (P(PO)>=.999 & bye leverage<.02, random
tiebreaks) misses locks decided by tiebreakers (e.g. 2010 PHI, 2006 NO in the DEV
top-disagreement list). This builder simulates the remaining regular season from
as-of plain Elo and SEEDS each conference with an approximation of the NFL
procedure:
   division ties : head-to-head pct among tied -> division pct -> conference pct -> coin
   wildcard/seed : one team per division enters a multi-team tie (best by the
                   division procedure); then head-to-head pct among tied ->
                   conference pct -> coin
   format        : 2002-2019 4 div winners + 2 WC, seeds 1-2 bye (this DEV builder)
                   (>=2020: 3 WC, seed 1 bye; final REG week 18 from 2021 - the
                   serving/TEST implementation must switch on season)
A side is LOCKED in week W (last 3 REG weeks) if it makes the playoffs with the
SAME seed in >= 99.9% of simulations regardless of this game's result (both the
win-branch and the loss-branch modal seed agree and each branch is >= 99.9%
concentrated). Feature = locked_away - locked_home. As-of: standings and Elo use
only REG games of weeks < W of the same season. No odds read.
Output: data/bt_nfl_ideas_seedlock.npy (feature aligned to the cache spine) and
data/bt_nfl_ideas_seedlock.json (indication).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260929)
D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]; X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
WK = np.array([g["week"] for g in G]); TYP = np.array([g["type"] for g in G])
Y = np.array([g["y"] for g in G], float)
RAW = {r["game_id"]: r for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8"))
       if r["season"] and int(r["season"]) < TEST_ERA}
HS = np.array([float(RAW[g["gid"]]["home_score"]) for g in G]); AS = np.array([float(RAW[g["gid"]]["away_score"]) for g in G])
DIV = {"AFC East": "BUF MIA NE NYJ", "AFC North": "BAL CIN CLE PIT", "AFC South": "HOU IND JAX TEN",
       "AFC West": "DEN KC LV LAC", "NFC East": "DAL NYG PHI WAS", "NFC North": "CHI DET GB MIN",
       "NFC South": "ATL CAR NO TB", "NFC West": "ARI LA SF SEA"}
TDIV = {t: d for d, ts in DIV.items() for t in ts.split()}
TEAMS = sorted(TDIV); TIX = {t: k for k, t in enumerate(TEAMS)}; NT = len(TEAMS)
CONF = np.array([TDIV[t][:3] for t in TEAMS]); DIVA = np.array([TDIV[t] for t in TEAMS])
K_, HFA_, REG_ = 47.43351932834238, 52.14162888646703, 0.3647684062154459


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


elo = defaultdict(lambda: 1500.0); ELO_POST = np.zeros((N, 2)); prev = None
for i, g in enumerate(G):
    if prev is not None and g["season"] != prev:
        for t in list(elo):
            elo[t] = 1500.0 + (1 - REG_) * (elo[t] - 1500.0)
    prev = g["season"]
    h, a = g["home"], g["away"]
    e = 1.0 / (1.0 + 10 ** (-(elo[h] - elo[a] + (0.0 if g["neutral"] else HFA_)) / 400.0))
    yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
    elo[h] += K_ * (yy - e); elo[a] -= K_ * (yy - e)
    ELO_POST[i] = (elo[h], elo[a])


def seed_one(W, H2H, DV, CF, noise):
    """W: wins (NT,), H2H (NT,NT) points of i vs j and games (NT,NT) as tuple, DV/CF pct.
    Returns seed array (NT,) with 0 = out, 1..6."""
    hp, hg = H2H
    key_base = W + noise

    def h2h_pct(i, grp):
        g_ = sum(hg[i, j] for j in grp if j != i)
        return (sum(hp[i, j] for j in grp if j != i) / g_) if g_ > 0 else 0.5

    def div_order(grp):
        # sort tied teams by (wins, h2h among exactly-tied, div pct, conf pct, noise)
        out, grp = [], list(grp)
        while grp:
            best = max(W[j] for j in grp)
            tied = [j for j in grp if W[j] == best]
            if len(tied) > 1:
                tied.sort(key=lambda j: (h2h_pct(j, tied), DV[j], CF[j], noise[j]), reverse=True)
            out.append(tied[0]); grp.remove(tied[0])
        return out

    def conf_order(grp):
        out, grp = [], list(grp)
        while grp:
            best = max(W[j] for j in grp)
            tied = [j for j in grp if W[j] == best]
            if len(tied) > 1:
                # one team per division enters the tie
                reps = {}
                for j in div_order(tied):
                    reps.setdefault(DIVA[j], j)
                tied = list(reps.values())
                tied.sort(key=lambda j: (h2h_pct(j, tied), CF[j], noise[j]), reverse=True)
            out.append(tied[0]); grp.remove(tied[0])
        return out

    seed = np.zeros(NT, int)
    for cf in ("AFC", "NFC"):
        winners = []
        for dv in sorted(set(DIVA)):
            if not dv.startswith(cf):
                continue
            winners.append(div_order([j for j in range(NT) if DIVA[j] == dv])[0])
        ordered = conf_order(winners)
        for k, j in enumerate(ordered):
            seed[j] = k + 1
        rest = [j for j in range(NT) if CONF[j] == cf and j not in winners]
        for k, j in enumerate(conf_order(rest)[:2]):
            seed[j] = 5 + k
    return seed


NS = 1500
LOCK = np.zeros((N, 2))
for s in range(2002, DEV_HI + 1):
    reg = [i for i in range(N) if SEAS[i] == s and TYP[i] == "REG"]
    last_wk = max(WK[i] for i in reg)
    for W_ in range(last_wk - 2, last_wk + 1):
        wk_games = [i for i in reg if WK[i] == W_]
        done = [i for i in reg if WK[i] < W_]; rem = [i for i in reg if WK[i] >= W_]
        w0 = np.zeros(NT); hp0 = np.zeros((NT, NT)); hg0 = np.zeros((NT, NT))
        dv0 = np.zeros((NT, 2)); cf0 = np.zeros((NT, 2)); rat = {}
        def add(hh, aa, yy, w, hp, hg, dvr, cfr):
            w[hh] += yy; w[aa] += 1 - yy
            hp[hh, aa] += yy; hp[aa, hh] += 1 - yy; hg[hh, aa] += 1; hg[aa, hh] += 1
            if DIVA[hh] == DIVA[aa]:
                dvr[hh] += (yy, 1); dvr[aa] += (1 - yy, 1)
            if CONF[hh] == CONF[aa]:
                cfr[hh] += (yy, 1); cfr[aa] += (1 - yy, 1)
        for i in done:
            hh, aa = TIX[G[i]["home"]], TIX[G[i]["away"]]
            yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
            add(hh, aa, yy, w0, hp0, hg0, dv0, cf0)
            rat[G[i]["home"]] = ELO_POST[i, 0]; rat[G[i]["away"]] = ELO_POST[i, 1]
        pe = np.array([1.0 / (1.0 + 10 ** (-(rat.get(G[i]["home"], 1500.0) - rat.get(G[i]["away"], 1500.0)
                                            + (0.0 if G[i]["neutral"] else HFA_)) / 400.0)) for i in rem])
        HW = RNG.random((NS, len(rem))) < pe
        seeds = np.zeros((NS, NT), int)
        for k in range(NS):
            w = w0.copy(); hp = hp0.copy(); hg = hg0.copy(); dvr = dv0.copy(); cfr = cf0.copy()
            for c, i in enumerate(rem):
                add(TIX[G[i]["home"]], TIX[G[i]["away"]], 1.0 if HW[k, c] else 0.0, w, hp, hg, dvr, cfr)
            DV = np.where(dvr[:, 1] > 0, dvr[:, 0] / np.maximum(dvr[:, 1], 1), 0.5)
            CF = np.where(cfr[:, 1] > 0, cfr[:, 0] / np.maximum(cfr[:, 1], 1), 0.5)
            seeds[k] = seed_one(w, (hp, hg), DV, CF, RNG.random(NT))
        col = {i: c for c, i in enumerate(rem)}
        for i in wk_games:
            c = col[i]
            for side, t in ((0, G[i]["home"]), (1, G[i]["away"])):
                ti = TIX[t]
                won = HW[:, c] if side == 0 else ~HW[:, c]
                if won.sum() < 20 or (~won).sum() < 20:
                    continue
                ok = True; modal = None
                for br in (won, ~won):
                    sv = seeds[br, ti]
                    vals, cnts = np.unique(sv, return_counts=True)
                    m_ = vals[np.argmax(cnts)]
                    if m_ == 0 or cnts.max() / br.sum() < 0.999 or (modal is not None and m_ != modal):
                        ok = False
                    modal = m_
                LOCK[i, side] = 1.0 if ok else 0.0
    print(f"  {s}: locked sides {int(LOCK[SEAS == s].sum())}", flush=True)

LK = LOCK[:, 1] - LOCK[:, 0]
np.save("data/bt_nfl_ideas_seedlock.npy", LK)


def wf(extra=None, lo=DEV_LO, hi=DEV_HI):
    per, vec = {}, []
    for s_ in range(lo, hi + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X14
        if extra is not None:
            E = extra.reshape(-1, 1); mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X14, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1]); per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec)


OUT = {}
for nm, lo, hi in (("DEV 2006-2015", DEV_LO, DEV_HI), ("pre-DEV 2002-2005 (already viewed by the proxy)", 2002, 2005)):
    b = wf(lo=lo, hi=hi); r = wf(LK, lo=lo, hi=hi)
    d = b[2] - r[2]
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in r[1] if b[1][s] - r[1][s] > 0)
    OUT[nm] = {"gain": round(b[0] - r[0], 6), "ci": [round(lo_, 6), round(hi_, 6)], "seasons_pos": pos,
               "n_locked_games": int(((LK != 0) & (SEAS >= lo) & (SEAS <= hi)).sum())}
    print(f"  seedlock_exact {nm}: gain {b[0]-r[0]:+.5f} CI[{lo_:+.5f},{hi_:+.5f}] seasons+ {pos} "
          f"locked games {OUT[nm]['n_locked_games']}")
json.dump(OUT, open("data/bt_nfl_ideas_seedlock.json", "w"), indent=1)
