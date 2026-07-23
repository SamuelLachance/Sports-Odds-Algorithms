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

G16 = [(i, g) for i, g in enumerate(games) if g["season"] >= 2016]
seas16 = np.array([g["season"] for _, g in G16])
y16 = np.array([g["y"] for _, g in G16])

def run_config(tau, use_lev, down_mult, late_mult, season_shrink=1.0 / 3.0,
               widen2=1.5 ** 2):
    """One engine walk -> per-game ts_diff feature (pre-game, leak-free)."""
    tau2 = tau * tau
    mu, s2, lastwk = {}, {}, {}
    share2 = {}
    f = np.zeros(len(G16))
    prev = None
    for j, (i, g) in enumerate(G16):
        if prev is not None and g["season"] != prev:
            for p in mu:
                mu[p] = MU0 + (mu[p] - MU0) * (1.0 - season_shrink)
                s2[p] = min(s2[p] + widen2, SIG0_2)
        prev = g["season"]
        # 1. feature from pre-game state
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
                if m_ is not None:
                    tot += (st[0] / st[1]) * max(-V_CLAMP, min(V_CLAMP, m_ - MU0))
            f[j] += side * tot
        # 2. TS updates from this game's plays, context-weighted
        gid = g["gid"]
        wk = (gid[:4], gid[5:7])
        for pid_, O, D, epa in plays_by_gid.get(gid, ()):
            for p in O:
                if p not in mu:
                    mu[p] = MU0; s2[p] = SIG0_2
                elif lastwk.get(p) != wk:
                    s2[p] = min(s2[p] + tau2, SIG0_2)
                lastwk[p] = wk
            for p in D:
                if p not in mu:
                    mu[p] = MU0; s2[p] = SIG0_2
                elif lastwk.get(p) != wk:
                    s2[p] = min(s2[p] + tau2, SIG0_2)
                lastwk[p] = wk
            wp, down, qtr, adiff, hsec = ctx.get((gid, pid_), (0.5, 1, 1, 0.0, 900.0))
            wgt = (4.0 * wp * (1.0 - wp)) if use_lev else \
                (0.25 if (wp < 0.05 or wp > 0.95) else 1.0)
            if down >= 3:
                wgt *= down_mult
            if qtr >= 4 and adiff <= 8:
                wgt *= late_mult
            c2 = (len(O) + len(D)) * BETA * BETA + sum(s2[p] for p in O) + sum(s2[p] for p in D)
            c = sqrt(c2)
            t = (sum(mu[p] for p in O) - sum(mu[p] for p in D)) / c
            if epa > 0:
                win, lose, tt = O, D, t
            else:
                win, lose, tt = D, O, -t
            v = v_win(tt); w = v * (v + tt)
            for p in win:
                mu[p] += wgt * (s2[p] / c) * v
                s2[p] = max(s2[p] * (1.0 - wgt * (s2[p] / c2) * w), SIG2_FLOOR)
            for p in lose:
                mu[p] -= wgt * (s2[p] / c) * v
                s2[p] = max(s2[p] * (1.0 - wgt * (s2[p] / c2) * w), SIG2_FLOOR)
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
    return f

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

# ================= DEV sweep (2017-2021) =================
print("\nDEV 2017-2021 — engine knob sweep (ratings-only model LL):", flush=True)
results = {}
def trial(name, **kw):
    f = run_config(**kw)
    ll, _ = wf_ll(f, 2017, 2021)
    results[name] = (ll, kw)
    print(f"  {name:<44} {ll:.5f}  [{time.time()-T0:.0f}s]", flush=True)

trial("baseline (old engine: tau .10, cutoff)", tau=0.10, use_lev=False, down_mult=1.0, late_mult=1.0)
trial("leverage only", tau=0.10, use_lev=True, down_mult=1.0, late_mult=1.0)
trial("full context (lev + 3rd/4th 1.25 + late 1.5)", tau=0.10, use_lev=True, down_mult=1.25, late_mult=1.5)
trial("recency tau .30 + full context", tau=0.30, use_lev=True, down_mult=1.25, late_mult=1.5)
trial("recency tau .50 + full context", tau=0.50, use_lev=True, down_mult=1.25, late_mult=1.5)
trial("recency tau .30 + full ctx + shrink 1/2", tau=0.30, use_lev=True, down_mult=1.25, late_mult=1.5, season_shrink=0.5)
trial("recency tau .30, context off", tau=0.30, use_lev=False, down_mult=1.0, late_mult=1.0)

best_name = min(results, key=lambda k: results[k][0])
best_ll, best_kw = results[best_name]
print(f"\nDEV winner: {best_name}  ({best_ll:.5f})")

# ================= ONE TEST look (2022-2025) =================
f_best = run_config(**best_kw)
ll_test, lv_test = wf_ll(f_best, 2022, 2025)
te_mask = (seas16 >= 2022) & (seas16 <= 2025)
# benchmarks on the SAME games
idx_all = np.array([i for i, _ in G16])
pe16 = pe[idx_all]                       # plain-Elo probs from the 1999+ walk
ll_elo = float(llv(y16[te_mask], pe16[te_mask]).mean())
print(f"\nTEST 2022-2025 (n={int(te_mask.sum())}, scored once):")
print(f"  ratings-only model (1 feature)  LL {ll_test:.5f}")
print(f"  plain team Elo                  LL {ll_elo:.5f}")
print(f"  home prior                      LL {float(llv(y16[te_mask], np.full(int(te_mask.sum()), y16[seas16<2022].mean())).mean()):.5f}")
acc = float(((f_best[te_mask] > 0) == (y16[te_mask] > 0.5))[y16[te_mask] != 0.5].mean())
print(f"  ratings-only accuracy {acc*100:.1f}%")
json.dump({"dev": {k: round(v[0], 5) for k, v in results.items()},
           "winner": best_name, "winner_kw": {k: v for k, v in best_kw.items()},
           "test_ll": round(ll_test, 5), "elo_ll": round(ll_elo, 5),
           "test_n": int(te_mask.sum()), "acc": round(acc, 4)},
          open("data/nfl_ratings_only.json", "w"), indent=1, default=float)
print("\nwrote data/nfl_ratings_only.json")
