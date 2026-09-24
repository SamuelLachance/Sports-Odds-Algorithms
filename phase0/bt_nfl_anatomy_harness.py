"""Breakthrough program 2026-09-24 -- NFL anatomy harness check (DEV only).

Re-runs the shipped walk-forward protocol (train seasons < s, HL 3, C=100) on the
cached DEV feature matrix (data/bt_nfl_anatomy_dev.csv, seasons <= 2015 only) and
drops one feature block at a time, so the anatomy harness can be checked against
known effects (QB feature is the biggest single win in the ledger). DEV 2006-2015,
ties dropped, market-present games only (the anatomy's n=2531 mask) and all games.
Output data/bt_nfl_anatomy_harness.json.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

rows = list(csv.DictReader(open("data/bt_nfl_anatomy_dev.csv", encoding="utf-8")))
assert max(int(r["season"]) for r in rows) <= 2015
COLS = ["lgt", "qd", "epa", "early", "thfa", "rest", "luck", "v7", "ol", "de", "sk", "rq", "pass", "run"]
X = np.array([[float(r[c]) for c in COLS] for r in rows])
y = np.array([float(r["y"]) for r in rows]); s = np.array([int(r["season"]) for r in rows])
gid = [r["gid"] for r in rows]
ml = {}
for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
    if r["home_score"] == "" or int(r["season"]) > 2015:
        continue
    ml[r["game_id"]] = bool(r["home_moneyline"] and r["away_moneyline"])
mk = np.array([ml.get(g, False) for g in gid])


def llv(yy, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def wf(cols):
    idx = [COLS.index(c) for c in cols]
    p = np.full(len(y), np.nan)
    for s_ in range(2006, 2016):
        tr = (s < s_) & (y != 0.5); te = s == s_
        w = 0.5 ** ((s_ - 1 - s[tr]) / 3.0)
        m = LogisticRegression(C=100.0, max_iter=5000).fit(X[tr][:, idx], y[tr], sample_weight=w)
        p[te] = m.predict_proba(X[te][:, idx])[:, 1]
    return p


full = wf(COLS)
scored = (s >= 2006) & (y != 0.5)
mask = scored & mk
out = {"full": {"ll_all_nonties": round(float(llv(y[scored], full[scored]).mean()), 5),
                "ll_market_mask": round(float(llv(y[mask], full[mask]).mean()), 5),
                "n_all": int(scored.sum()), "n_mask": int(mask.sum())}}
rng = np.random.default_rng(7)
for name, drop in (("drop qd", ["qd"]), ("drop elo lgt", ["lgt"]), ("drop epa+pass+run", ["epa", "pass", "run"]),
                   ("drop absence+rq", ["ol", "de", "sk", "rq"]), ("elo only", [c for c in COLS if c != "lgt"])):
    keep = [c for c in COLS if c not in drop]
    p = wf(keep)
    dd = llv(y[scored], p[scored]) - llv(y[scored], full[scored])
    bs = dd[rng.integers(0, len(dd), size=(10000, len(dd)))].mean(1)
    out[name] = {"cost_nats": round(float(dd.mean()), 5),
                 "ci": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]}
    print(name, out[name])
json.dump(out, open("data/bt_nfl_anatomy_harness.json", "w"), indent=1)
print(out["full"])
