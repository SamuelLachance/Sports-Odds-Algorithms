"""Breakthrough program (NFL, ideas leg) - round 6: a slow second timescale. DEV ONLY.

Top-disagreement reading (DEV, eval-only) showed the close keeping faith in
multi-season elite teams (e.g. SEA 2013-15) after short slumps our fast states
punish. Probe: ADD a slow net-EPA rating (per-game decay .97, season carry-over
decay .30, same prior/anchor) beside the shipped fast one; and a slow Elo
(k=15, regress .20). Single pre-declared constructions. Anatomy EVAL-ONLY.
Output: data/bt_nfl_ideas_quick6.json
"""
import csv, json
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression

src = open("phase0/bt_nfl_ideas_quick5.py", encoding="utf-8").read()
exec(compile(src.split("raw_net = walk(agg_raw)")[0], "q5head", "exec"))  # noqa: S102
exec(compile("def wf(" + src.split("def wf(")[1].split("b_ll, b_per, b_vec = wf()")[0], "q5wf", "exec"))  # noqa: S102


def walk_slow(src_, dec, sdec):
    tot = [v for k, v in src_.items() if 1999 <= int(k[0][:4]) <= 2015]
    lgv = sum(v[0] for v in tot) / sum(v[1] for v in tot)
    off = defaultdict(lambda: [0.0, 0.0]); dfn = defaultdict(lambda: [0.0, 0.0])
    rate = lambda st: (st[0] + PN * lgv) / (st[1] + PN)  # noqa: E731
    out = np.zeros(N); prev = None
    for i, g in enumerate(G):
        if prev is not None and g["season"] != prev:
            for dct in (off, dfn):
                for st in dct.values():
                    st[0] *= (1 - sdec); st[1] *= (1 - sdec)
        prev = g["season"]
        h, a = g["home"], g["away"]
        out[i] = (rate(off[h]) - rate(dfn[h])) - (rate(off[a]) - rate(dfn[a]))
        for t_off, t_def in ((h, a), (a, h)):
            e = src_.get((g["gid"], t_off))
            if not e or e[1] == 0:
                continue
            o = off[t_off]; o[0] = dec * o[0] + e[0]; o[1] = dec * o[1] + e[1]
            d_ = dfn[t_def]; d_[0] = dec * d_[0] + e[0]; d_[1] = dec * d_[1] + e[1]
    return out


slow_epa = walk_slow(agg_raw, 0.97, 0.30)
fast = walk_slow(agg_raw, DEC, SDEC)
print("corr fast vs shipped col2", round(float(np.corrcoef(fast[SEAS >= 2006], X14[SEAS >= 2006, 2])[0, 1]), 5),
      " corr slow vs fast", round(float(np.corrcoef(slow_epa[SEAS >= 2006], fast[SEAS >= 2006])[0, 1]), 3))
RAW = {r["game_id"]: r for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8"))
       if r["season"] and int(r["season"]) < TEST_ERA}
HS = np.array([float(RAW[g["gid"]]["home_score"]) for g in G]); AS = np.array([float(RAW[g["gid"]]["away_score"]) for g in G])
elo = defaultdict(lambda: 1500.0); slow_elo = np.zeros(N); prev = None
for i, g in enumerate(G):
    if prev is not None and g["season"] != prev:
        for t in list(elo):
            elo[t] = 1500.0 + 0.8 * (elo[t] - 1500.0)
    prev = g["season"]
    h, a = g["home"], g["away"]
    slow_elo[i] = elo[h] - elo[a]
    dr = slow_elo[i] + (0.0 if g["neutral"] else 52.14)
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    yy = 1.0 if HS[i] > AS[i] else 0.5 if HS[i] == AS[i] else 0.0
    elo[h] += 15.0 * (yy - e); elo[a] -= 15.0 * (yy - e)

b_ll, b_per, b_vec = wf()
assert abs(b_ll - 0.62292) < 5e-4
OUT = {"baseline": b_ll, "results": {}}
for nm, res in (("slow EPA net add", wf([slow_epa])), ("slow Elo add", wf([slow_elo])),
                ("slow EPA + slow Elo add", wf([slow_epa, slow_elo]))):
    ll, per, vec = res
    d = b_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    OUT["results"][nm] = {"gain": round(b_ll - ll, 6), "ci": [round(lo, 6), round(hi, 6)], "seasons_pos": pos}
    print(f"  {nm:<28} gain {b_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10")
json.dump(OUT, open("data/bt_nfl_ideas_quick6.json", "w"), indent=1)
