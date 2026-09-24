"""Breakthrough program (NFL, ideas leg) - placebo for the seed-lock lead. DEV ONLY.

Permutes the lock feature among games of the SAME season-week (keeps its
frequency/timing, destroys the team assignment). A real effect must vanish.
Also checks the sign-flip control (feature negated must hurt). No odds read.
Output: data/bt_nfl_ideas_lock_placebo.json
"""
import json
import numpy as np

src = open("phase0/bt_nfl_ideas_quick3.py", encoding="utf-8").read()
exec(compile(src.split('print("1. schedule-adjusted process block")')[0], "q3head", "exec"))  # noqa: S102
LK = LOCK[:, 1] - LOCK[:, 0]
real = wf(LK)
rng = np.random.default_rng(99)
keys = {}
for i, g in enumerate(G):
    keys.setdefault((g["season"], g["week"]), []).append(i)
gains = []
for rep_ in range(12):
    P = LK.copy()
    for k, ix in keys.items():
        ix = np.array(ix)
        P[ix] = LK[rng.permutation(ix)] * rng.choice([-1, 1], size=len(ix))
    ll = wf(P)[0]
    gains.append(b_ll - ll)
flip = wf(-LK)
out = {"real_gain": b_ll - real[0], "placebo_gains": [round(x, 6) for x in gains],
       "placebo_mean": float(np.mean(gains)), "placebo_max": float(np.max(gains)),
       "note": "sign-flip is NOT a control for a z-scored linear column (the fit can flip the coef); reported for completeness",
       "flip_gain": b_ll - flip[0]}
print(json.dumps(out, indent=1))
json.dump(out, open("data/bt_nfl_ideas_lock_placebo.json", "w"), indent=1)
