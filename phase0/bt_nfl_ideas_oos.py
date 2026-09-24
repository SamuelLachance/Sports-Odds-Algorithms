"""Breakthrough program (NFL, ideas leg) - out-of-search window re-score. PRE-DEV ONLY.

The depth-program standing bar (documents/depth_program.md, Round 4): any lead
must be re-scored on a temporal window that was never part of the search. The
DEV search scored 2006-2015; this re-scores the SAME pre-declared constructions
on 2002-2005 (walk-forward, trained on 1999..s-1 only, shipped HL/C). Seasons
>= 2016 are never touched. No odds read. Output: data/bt_nfl_ideas_oos.json
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

src = open("phase0/bt_nfl_ideas_quick3.py", encoding="utf-8").read()
exec(compile(src.split('print("1. schedule-adjusted process block")')[0], "q3head", "exec"))  # noqa: S102
LO, HI = 2002, 2005
assert HI < DEV_LO < TEST_ERA


def wf_oos(extra=None, replace=None):
    X = X14.copy()
    if replace:
        for c_, v in replace.items():
            X[:, c_] = v
    per, vec = {}, []
    for s_ in range(LO, HI + 1):
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[te].max() < DEV_LO
        Xs = X
        if extra is not None:
            Em = np.column_stack(extra) if isinstance(extra, (list, tuple)) else extra.reshape(-1, 1)
            mu, sd = Em[tr].mean(0), Em[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (Em - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1]); per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec)


LK = LOCK[:, 1] - LOCK[:, 0]
b = wf_oos()
OUTO = {"window": [LO, HI], "baseline": b[0], "results": {}}
print(f"OOS 2002-2005 baseline {b[0]:.5f} n={len(b[2])}; locked games in window:",
      {s: int(((SEAS == s) & (LK != 0)).sum()) for s in range(LO, HI + 1)})
rng = np.random.default_rng(11)
for nm, kw in (("S3 locked", dict(extra=LK)),
               ("C1c team opp-adj replace", dict(replace={2: ADJ_ALL, 12: ADJ_PASS, 13: ADJ_RUN})),
               ("Q opp-adj QB add", dict(extra=FE["Q"])),
               ("C1c + Q", dict(extra=FE["Q"], replace={2: ADJ_ALL, 12: ADJ_PASS, 13: ADJ_RUN})),
               ("C1c + Q + S3", dict(extra=[FE["Q"], LK], replace={2: ADJ_ALL, 12: ADJ_PASS, 13: ADJ_RUN}))):
    ll, per, vec = wf_oos(**kw)
    d = b[2] - vec
    bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    OUTO["results"][nm] = {"gain": round(b[0] - ll, 6), "ci": [round(lo, 6), round(hi, 6)],
                           "per_season": {str(s): round(b[1][s] - per[s], 5) for s in per}}
    print(f"  {nm:<28} gain {b[0]-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] per-season {OUTO['results'][nm]['per_season']}")
json.dump(OUTO, open("data/bt_nfl_ideas_oos.json", "w"), indent=1)
