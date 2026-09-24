"""Breakthrough program (NFL, ideas leg) - round 5: two last structural probes. DEV ONLY.

  E  season-phase re-weighting: at the season boundary every carried state is
     transformed uniformly (Elo regress, EPA decay, QB decay), so in weeks 1-4 the
     blend cannot know that team Elo is stale while the QB state (which follows
     the player) is not. Adds fade x {elo_logit, qb_delta, epa_net} with
     fade = 1, .75, .5, .25 over each side's first four REG games (averaged).
  W  smooth leverage weighting of the team EPA channel (WEPA's bell-curve WP
     weighting; the v7 engine's sqrt-leverage idea transplanted to the team
     channel). Row 41 rejected HARD garbage-time filters; a smooth, floored
     weight was never tried: w = 0.5 + 0.5*sqrt(4 wp (1-wp)) with wp the
     market-free nflfastR WP (data/nfl_plays_wp.csv). REPLACES col 2.
Single pre-declared constructions; indications only. No odds read.
Output: data/bt_nfl_ideas_quick5.json
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260928)
D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]; X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
TYP = np.array([g["type"] for g in G]); Y = np.array([g["y"] for g in G], float)
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
DEC, PN, SDEC = 0.8813581205848026, 754.0438610349224, 0.7130881070009863


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---- E: phase fade
gpl = defaultdict(int); FADE = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25}
PH = np.zeros(N)
for i, g in enumerate(G):
    s = g["season"]
    if g["type"] == "REG":
        PH[i] = 0.5 * (FADE.get(gpl[(g["home"], s)] + 1, 0.0) + FADE.get(gpl[(g["away"], s)] + 1, 0.0))
    gpl[(g["home"], s)] += 1; gpl[(g["away"], s)] += 1
E_int = [PH * X14[:, 0], PH * X14[:, 1], PH * X14[:, 2]]

# ---- W: leverage-weighted epa_net
agg = defaultdict(lambda: [0.0, 0.0]); agg_raw = defaultdict(lambda: [0.0, 0.0])
with open("data/nfl_plays_wp.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: k for k, c in enumerate(hdr)}
    for r in rd:
        if int(r[ix["game_id"]][:4]) >= TEST_ERA:
            continue
        wp = min(max(float(r[ix["wp"]]), 0.0), 1.0)
        w = 0.5 + 0.5 * math.sqrt(4 * wp * (1 - wp))
        k = (r[ix["game_id"]], FR.get(r[ix["posteam"]], r[ix["posteam"]]))
        e = float(r[ix["epa"]])
        a = agg[k]; a[0] += w * e; a[1] += w
        b = agg_raw[k]; b[0] += e; b[1] += 1.0


def walk(src):
    tot = [v for k, v in src.items() if 1999 <= int(k[0][:4]) <= 2015]
    lgv = sum(v[0] for v in tot) / sum(v[1] for v in tot)
    off = defaultdict(lambda: [0.0, 0.0]); dfn = defaultdict(lambda: [0.0, 0.0])
    rate = lambda st: (st[0] + PN * lgv) / (st[1] + PN)  # noqa: E731
    out = np.zeros(N); prev = None
    for i, g in enumerate(G):
        if prev is not None and g["season"] != prev:
            for dct in (off, dfn):
                for st in dct.values():
                    st[0] *= (1 - SDEC); st[1] *= (1 - SDEC)
        prev = g["season"]
        h, a = g["home"], g["away"]
        out[i] = (rate(off[h]) - rate(dfn[h])) - (rate(off[a]) - rate(dfn[a]))
        for t_off, t_def in ((h, a), (a, h)):
            e = src.get((g["gid"], t_off))
            if not e or e[1] == 0:
                continue
            o = off[t_off]; o[0] = DEC * o[0] + e[0]; o[1] = DEC * o[1] + e[1]
            d_ = dfn[t_def]; d_[0] = DEC * d_[0] + e[0]; d_[1] = DEC * d_[1] + e[1]
    return out


raw_net = walk(agg_raw); lev_net = walk(agg)
dv = SEAS >= DEV_LO
rc = float(np.corrcoef(raw_net[dv], X14[dv, 2])[0, 1])
print(f"plays_wp raw walk vs shipped col 2 corr {rc:.5f}")


def wf(extra=None, replace=None):
    X = X14.copy()
    if replace:
        for c_, v in replace.items():
            X[:, c_] = v
    per, vec = {}, []
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_ and SEAS[te].max() < TEST_ERA
        Xs = X
        if extra is not None:
            Em = np.column_stack(extra); mu, sd = Em[tr].mean(0), Em[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (Em - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1]); per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec)


b_ll, b_per, b_vec = wf()
assert abs(b_ll - 0.62292) < 5e-4
OUT = {"baseline": b_ll, "raw_walk_corr": rc, "results": {}}
for nm, res in (("E phase fade x (elo, qb, epa)", wf(E_int)),
                ("W leverage-weighted epa_net (replace col 2)", wf(replace={2: lev_net}))):
    ll, per, vec = res
    d = b_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    OUT["results"][nm] = {"gain": round(b_ll - ll, 6), "ci": [round(lo, 6), round(hi, 6)], "seasons_pos": pos}
    print(f"  {nm:<46} gain {b_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10")
json.dump(OUT, open("data/bt_nfl_ideas_quick5.json", "w"), indent=1)
