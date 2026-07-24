"""Does Neil Paine's (ex-538) MLB Elo add anything to our model?

Joins his mlb-elo-latest.csv (1871-2025, elo_prob per game) to model_probs
(2001-2025). Walk-forward stack by season: logistic [logit(ours), logit(paine)]
trained on prior seasons, scored per season, pooled with bootstrap CI.
Third-party MODEL output — market-free, but external dependency; diagnostic.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

P2R = {"CHC": "CHN", "CHW": "CHA", "KCR": "KCA", "LAD": "LAN", "NYY": "NYA",
       "NYM": "NYN", "SDP": "SDN", "SFG": "SFN", "STL": "SLN", "TBD": "TBA",
       "WSN": "WAS", "FLA": "MIA"}

SP = (r"C:/Users/Admin/AppData/Local/Temp/claude/"
      r"C--Users-Admin-Projects-Sports-Odds-Algorithms/"
      r"c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")

paine = {}
pk = Counter()
for r in csv.DictReader(open(SP + "/paine_elo_latest.csv", encoding="utf-8",
                             errors="replace")):
    if r["is_home"] != "1" or r["season"] < "2000":
        continue
    h = P2R.get(r["team1"], r["team1"])
    a = P2R.get(r["team2"], r["team2"])
    k = (r["date"], h, a)
    pk[k] += 1
    paine[k] = float(r["elo_prob1"])

mp = [r for r in csv.DictReader(open("data/model_probs.csv")) if r["season"] >= "2000"]
mk = Counter((r["date"], r["home"], r["away"]) for r in mp)
J = []
for r in mp:
    k = (r["date"], r["home"], r["away"])
    if mk[k] > 1 or pk.get(k, 0) != 1:
        continue
    y = float(r["y"])
    if y not in (0.0, 1.0):
        continue
    J.append((int(r["season"]), y, float(r["full_p"]), paine[k]))
print(f"joined {len(J)}/{len(mp)} games "
      f"({sum(1 for k, c in mk.items() if c > 1)} DH keys dropped)")
print("by era:", {e: sum(1 for s, *_ in J if e[0] <= s <= e[1])
                  for e in ((2000, 2015), (2016, 2020), (2021, 2025))})

eps = 1e-12
def llv(y, p):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

seas = np.array([s for s, *_ in J])
yv = np.array([y for _, y, *_ in J])
p_us = np.array([p for _, _, p, _ in J])
p_pe = np.array([p for _, _, _, p in J])
print(f"\nfull-join LL: ours {llv(yv, p_us).mean():.5f} | paine {llv(yv, p_pe).mean():.5f} "
      f"| corr(logits) {np.corrcoef(logit(p_us), logit(p_pe))[0, 1]:.4f}")

res = {}
for lo, hi, tag in ((2004, 2025, "pooled 2004-2025"), (2016, 2025, "2016-2025"),
                    (2021, 2025, "2021-2025")):
    lls_b, lls_s = [], []
    coefs = []
    for s_ in range(lo, hi + 1):
        tr = seas < s_
        te = seas == s_
        if tr.sum() < 500 or te.sum() < 100:
            continue
        X = np.column_stack([logit(p_us), logit(p_pe)])
        m = LogisticRegression(C=1e6, max_iter=3000).fit(X[tr], yv[tr])
        ps = m.predict_proba(X[te])[:, 1]
        lls_b.append(llv(yv[te], p_us[te]))
        lls_s.append(llv(yv[te], ps))
        coefs.append(m.coef_[0].tolist())
    b = np.concatenate(lls_b); s2 = np.concatenate(lls_s)
    d = b - s2
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    sig = "SIG" if lo_ > 0 else ("SIG WORSE" if hi_ < 0 else "n.s.")
    cf = np.mean(coefs, axis=0)
    print(f"{tag:<18} n={len(d):<6} base {b.mean():.5f} -> stack {s2.mean():.5f}  "
          f"delta {d.mean():+.5f} CI [{lo_:+.5f},{hi_:+.5f}] {sig}  "
          f"coefs [ours {cf[0]:.2f}, paine {cf[1]:.2f}]")
    res[tag] = {"n": int(len(d)), "ll_base": round(float(b.mean()), 5),
                "ll_stack": round(float(s2.mean()), 5),
                "delta": round(float(d.mean()), 5),
                "ci": [round(float(lo_), 5), round(float(hi_), 5)], "sig": sig,
                "mean_coefs": [round(float(c), 3) for c in cf]}

json.dump(res, open("data/paine_mlb_test.json", "w"), indent=1)
print("\nwrote data/paine_mlb_test.json")
