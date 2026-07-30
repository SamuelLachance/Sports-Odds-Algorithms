"""2026-relevant MVP-impact list: recency-weighted WOWY, active players only.

Row weight = 0.8^(2025 - season): a 2013 absence carries ~7% of a 2025 one.
Ridge shrinkage unchanged in absolute terms -> old-evidence players shrink harder
toward their position baseline (desired). Display: 2025-active only, n_abs >= 4,
with recent-absence counts for transparency.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

wowy_src = open("phase0/nfl_wowy_eval.py", encoding="utf-8").read()
exec(wowy_src.split("# ---- THE VALUE TABLE")[0])  # full collection walk  # noqa: S102

DECAY_S = 0.8
def weighted_fit():
    rows = train_rows
    wts = np.array([DECAY_S ** (2025 - s) for (s, _, _) in rows])
    pa = defaultdict(float)
    for (s, _, terms) in rows:
        for (pid, b, w) in terms:
            pa[pid] += DECAY_S ** (2025 - s)          # decayed absence count
    players = sorted({pid for (_, _, terms) in rows for (pid, b, w) in terms
                      if pa[pid] >= 2.0})
    pix = {p: len(BUCKETS) + j for j, p in enumerate(players)}
    bix = {b: j for j, b in enumerate(BUCKETS)}
    ncol = len(BUCKETS) + len(players)
    A = np.zeros((len(rows) + len(players), ncol))
    Y = np.zeros(len(rows) + len(players))
    for n, (s, yv, terms) in enumerate(rows):
        sw = np.sqrt(wts[n])
        Y[n] = yv * sw
        for (pid, b, w) in terms:
            A[n, bix[b]] += w * sw
            if pid in pix:
                A[n, pix[pid]] += w * sw
    for j, p in enumerate(players):
        A[len(rows) + j, pix[p]] = 3.0
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return ({b: float(sol[bix[b]]) for b in BUCKETS},
            {p: float(sol[pix[p]]) for p in players}, pa)

cp, cd, pa = weighted_fit()
print("RECENCY-WEIGHTED position values (pts/game per starter absent):")
for b in BUCKETS:
    print(f"  {b:<4} {cp.get(b, 0)*62:+.2f}")

active25 = {pid for (gid, team) in snaps if gid[:4] == "2025"
            for pid in snaps[(gid, team)]}
recent_abs = defaultdict(int)
for (s, _, terms) in train_rows:
    if s >= 2022:
        for (pid, b, w) in terms:
            recent_abs[pid] += 1

def show(bucket_filter, title, k=10):
    cands = []
    for pid, dv in cd.items():
        if pid not in active25:
            continue
        b = B8.get(pos_of.get(pid, ""), None)
        if b is None or not bucket_filter(b):
            continue
        if pa.get(pid, 0) < 3.0:
            continue
        cands.append((cp.get(b, 0) + dv, pid, b))
    cands.sort(reverse=True)
    print(f"\n{title}")
    for i, (tot, pid, b) in enumerate(cands[:k], 1):
        nm = names.get(pfr2gsis.get(pid, ""), pid)
        print(f"  {i:>2} {nm:<24} {b:<3} {tot*62:+.2f} pts/game  "
              f"(recent abs {recent_abs.get(pid, 0)}, decayed n {pa.get(pid, 0):.1f})")

show(lambda b: b == "QB", "2026 MVP-IMPACT — QBs (recency-weighted, 2025-active):")
show(lambda b: b != "QB", "2026 MVP-IMPACT — non-QBs:", 10)
json.dump({"pos_values_pts": {b: round(cp.get(b, 0) * 62, 2) for b in BUCKETS}},
          open("data/nfl_wowy_2026.json", "w"), indent=1)
