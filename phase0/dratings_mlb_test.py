"""Does DRatings (third-party model, Wayback+live reconstruction) add to our MLB model?

Join data/dratings_mlb.csv (2,337 predictions 2020-2026) to model_probs by
date + team nicknames. Walk-forward stack by season (score 2021-2025):
logistic [logit(ours), logit(dratings)] trained on prior overlap seasons.
External model feed — diagnostic; cannot ship regardless (identity).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

NICK2RETRO = {
    "Diamondbacks": "ARI", "D-backs": "ARI", "Braves": "ATL", "Orioles": "BAL",
    "Red Sox": "BOS", "Cubs": "CHN", "White Sox": "CHA", "Reds": "CIN",
    "Guardians": "CLE", "Indians": "CLE", "Rockies": "COL", "Tigers": "DET",
    "Astros": "HOU", "Royals": "KCA", "Angels": "ANA", "Dodgers": "LAN",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYN",
    "Yankees": "NYA", "Athletics": None, "A's": None, "Phillies": "PHI",
    "Pirates": "PIT", "Padres": "SDN", "Giants": "SFN", "Mariners": "SEA",
    "Cardinals": "SLN", "Rays": "TBA", "Rangers": "TEX", "Blue Jays": "TOR",
    "Nationals": "WAS",
}

def retro_of(nick, season):
    if nick in ("Athletics", "A's"):
        return "ATH" if season >= 2025 else "OAK"
    return NICK2RETRO.get(nick)

dr = defaultdict(list)
unmapped = Counter()
for r in csv.DictReader(open("data/dratings_mlb.csv", encoding="utf-8")):
    season = int(r["date"][:4])
    h = retro_of(r["home_nick"], season); a = retro_of(r["away_nick"], season)
    if not h or not a:
        unmapped[r["home_nick"] if not h else r["away_nick"]] += 1
        continue
    dr[(r["date"], h, a)].append(float(r["p_home"]))
print(f"dratings rows mapped; unmapped: {dict(unmapped)}")

mp = [r for r in csv.DictReader(open("data/model_probs.csv")) if r["season"] >= "2020"]
mk = Counter((r["date"], r["home"], r["away"]) for r in mp)
J = []
for r in mp:
    k = (r["date"], r["home"], r["away"])
    if mk[k] > 1 or len(dr.get(k, [])) != 1:
        continue
    y = float(r["y"])
    if y not in (0.0, 1.0):
        continue
    J.append((int(r["season"]), y, float(r["full_p"]), dr[k][0]))
print(f"joined {len(J)} games")
print("by season:", dict(sorted(Counter(s for s, *_ in J).items())))

eps = 1e-12
def llv(y, p):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

seas = np.array([s for s, *_ in J]); yv = np.array([y for _, y, *_ in J])
p_us = np.array([p for _, _, p, _ in J]); p_dr = np.array([p for _, _, _, p in J])
print(f"\nfull-join LL: ours {llv(yv, p_us).mean():.5f} | dratings {llv(yv, p_dr).mean():.5f} "
      f"| corr(logits) {np.corrcoef(logit(p_us), logit(p_dr))[0, 1]:.4f}")

lls_b, lls_s, coefs = [], [], []
for s_ in range(2021, 2026):
    tr = seas < s_
    te = seas == s_
    if tr.sum() < 150 or te.sum() < 80:
        continue
    X = np.column_stack([logit(p_us), logit(p_dr)])
    m = LogisticRegression(C=1e6, max_iter=3000).fit(X[tr], yv[tr])
    ps = m.predict_proba(X[te])[:, 1]
    lls_b.append(llv(yv[te], p_us[te])); lls_s.append(llv(yv[te], ps))
    coefs.append(m.coef_[0].tolist())
b = np.concatenate(lls_b); s2 = np.concatenate(lls_s)
d = b - s2
rng = np.random.default_rng(7)
bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
cf = np.mean(coefs, axis=0)
print(f"\nstack 2021-25: n={len(d)}  base {b.mean():.5f} -> {s2.mean():.5f}  "
      f"delta {d.mean():+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  "
      f"coefs [ours {cf[0]:.2f}, drat {cf[1]:.2f}]")
json.dump({"n": int(len(d)), "ll_base": round(float(b.mean()), 5),
           "ll_stack": round(float(s2.mean()), 5), "delta": round(float(d.mean()), 5),
           "ci": [round(float(lo), 5), round(float(hi), 5)], "sig": sig,
           "ll_dratings_alone": round(float(llv(yv, p_dr).mean()), 5),
           "ll_ours_alone": round(float(llv(yv, p_us).mean()), 5),
           "corr_logits": round(float(np.corrcoef(logit(p_us), logit(p_dr))[0, 1]), 4),
           "mean_coefs": [round(float(c), 3) for c in cf]},
          open("data/dratings_mlb_test.json", "w"), indent=1)
print("wrote data/dratings_mlb_test.json")
