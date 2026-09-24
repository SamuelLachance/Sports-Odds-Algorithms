"""ONE TEST look — NFL movcore (breakthrough program, 2026-09-24).

Candidate earned the look by clearing the pre-registered DEV bar
(documents/breakthrough_program_prereg_2026_09_24.md): margin-of-victory Elo
REPLACES the shipped win/loss Elo column (col 0) of the 14-feature blend.
DEV (2006-2015, phase0/bt_nfl_nfl_movcore.py): +0.00236 nats, CI [+0.00087,
+0.00385], season-block CI [+0.00137, +0.00338], 9/10 seasons; adversarial audit
sound, no leak.

Ledger caveat carried from the screen: row 10 (SRS margin beside the blend) was
the strongest DEV signal ever and LOST on TEST. This look is taken anyway because
the protocol says a candidate that clears the bar gets exactly one - and it gets
a ledger row whatever it shows.

Construction is fixed by the DEV screen; nothing here is tuned:
  - shipped X14 built by nfl_season_serve's own prelude (identical code path);
  - MOV walk = the screen's walk() verbatim, shipped core k/hfa/regress, c frozen
    from 1999-2001 margins;
  - protocol = the shipped walk-forward: for s in 2016..2025, train seasons < s
    (ties dropped), weight 0.5^((s-1-season)/3), LogisticRegression(C=100).
Arms: A0 shipped, A1 col 0 := MOV logit. Paired bootstrap (10k) + season block.
"""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()

# ---- the shipped X14, from the serve's own prelude (stops before its repro check)
_src = open("phase0/nfl_season_serve.py", encoding="utf-8").read()
_pre = _src.split("# ---------------- protocol check")[0]
assert "X14[:, 7] = v6_hist" in _pre
exec(compile(_pre, "nfl_season_serve<prelude>", "exec"))  # noqa: S102
print(f"[{time.time()-T0:.0f}s] shipped X14 {X14.shape}", flush=True)  # noqa: F821

G = games                                   # noqa: F821  (from the prelude)
SEAS = np.array([g["season"] for g in G])
Y = np.array([g["y"] for g in G], float)

# ---- scores for every spine game
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
SC = {}
for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
    if r["home_score"] != "" and r["away_score"] != "":
        SC[r["game_id"]] = (int(r["home_score"]), int(r["away_score"]))
HS = np.array([SC[g["gid"]][0] for g in G], float)
AS = np.array([SC[g["gid"]][1] for g in G], float)
assert np.array_equal(Y, np.where(HS > AS, 1.0, np.where(HS < AS, 0.0, 0.5)))

BASE = json.load(open("data/nfl_elo_base.json", encoding="utf-8"))
K0, HFA0, REG0 = BASE["params"]["k"], BASE["params"]["hfa"], BASE["params"]["regress"]
assert (K0, HFA0, REG0) == (47.43351932834238, 52.14162888646703, 0.3647684062154459)
PD = np.abs(HS - AS)
C_NORM = float(np.mean(np.log(PD[(SEAS <= 2001) & (PD > 0)] + 1.0)))
NEU = [bool(g.get("neutral")) for g in G]


def walk(k, hfa, reg, c, mode):
    """The DEV screen's walk(), verbatim in logic (bt_nfl_nfl_movcore.walk)."""
    R = {}
    out = np.empty(len(G))
    prev = None
    ln2c = math.log(2.0) / c
    for i, g in enumerate(G):
        s = int(SEAS[i])
        if prev is not None and s != prev:
            for t in R:
                R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - reg)
        prev = s
        h, a = g["home"], g["away"]
        rh = R.setdefault(h, 1500.0)
        ra = R.setdefault(a, 1500.0)
        hh = 0.0 if NEU[i] else hfa
        p = 1.0 / (1.0 + 10 ** (-((rh + hh) - ra) / 400.0))
        out[i] = p
        x, z = HS[i], AS[i]
        y = 1.0 if x > z else (0.0 if x < z else 0.5)
        if mode == "wl":
            mult = 1.0
        else:
            pd = abs(x - z)
            if pd > 0:
                dW = ((rh + hh) - ra) if x > z else (ra - (rh + hh))
                den = 0.001 * dW + 2.2
                assert den > 0
                mult = math.log(pd + 1.0) * 2.2 / den / c
            else:
                mult = ln2c
        d = k * mult * (y - p)
        R[h] += d
        R[a] -= d
    return out


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


WL = logit(walk(K0, HFA0, REG0, C_NORM, "wl"))
# identity: the W/L walk must BE the shipped column 0, or alignment is broken
d_id = float(np.max(np.abs(WL - X14[:, 0])))  # noqa: F821
print(f"identity |WL - X14[:,0]| max = {d_id:.2e}", flush=True)
assert d_id < 1e-6, "W/L walk does not reproduce shipped col 0 - misaligned spine"
MOV = logit(walk(K0, HFA0, REG0, C_NORM, "mov"))

HL, C = 3.0, 100.0
TEST = list(range(2016, 2026))


def wf(X):
    per, vec, sea = {}, [], []
    for s_ in TEST:
        tr = (SEAS < s_) & (Y != 0.5)
        te = SEAS == s_
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(X[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v); sea.append(np.full(len(v), s_))
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), np.concatenate(sea)


X0 = X14.copy()  # noqa: F821
X1 = X14.copy(); X1[:, 0] = MOV  # noqa: F821
b_ll, b_per, b_vec, sea = wf(X0)
c_ll, c_per, c_vec, _ = wf(X1)
print(f"A0 shipped TEST LL {b_ll:.5f}  (ledger 0.61947 / serve repro ~0.6192)", flush=True)
assert abs(b_ll - 0.61947) < 0.0005, "baseline does not reproduce the shipped TEST number"

d = b_vec - c_vec
rng = np.random.default_rng(20260924)
bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
blocks = [d[sea == s] for s in TEST]
bb = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(TEST), len(TEST))]).mean()
               for _ in range(10000)])
blo, bhi = (float(x) for x in np.percentile(bb, [2.5, 97.5]))
pos = sum(1 for s in TEST if b_per[s] > c_per[s])
print(f"A1 MOV core TEST LL {c_ll:.5f}  delta {b_ll - c_ll:+.5f} CI [{lo:+.5f},{hi:+.5f}] "
      f"block [{blo:+.5f},{bhi:+.5f}]  seasons better {pos}/{len(TEST)}", flush=True)
for s in TEST:
    print(f"   {s}: {b_per[s]:.5f} -> {c_per[s]:.5f}  ({b_per[s] - c_per[s]:+.5f})")
sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
json.dump({"n": int(len(d)), "a0": round(b_ll, 5), "a1": round(c_ll, 5),
           "delta": round(b_ll - c_ll, 5), "ci": [round(lo, 5), round(hi, 5)],
           "block_ci": [round(blo, 5), round(bhi, 5)], "seasons_better": pos, "verdict": sig,
           "per_season": {s: [round(b_per[s], 5), round(c_per[s], 5)] for s in TEST}},
          open("data/nfl_movcore_test.json", "w"), indent=1)
print(f"verdict: {sig}  -> data/nfl_movcore_test.json")
