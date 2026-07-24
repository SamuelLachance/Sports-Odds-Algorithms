"""Does Covers contest-pick consensus add to our MLB model?

Three variants (overall bettors / expert team-leaders / top-10% contestants)
+ the skill differential (top10pct share minus overall share — Levitt says
bettor skill doesn't exist; this measures it directly on 4 seasons).
Walk-forward stacks: 2021 warm, score 2022-2025. Crowd-pick data —
evaluation-layer only (constitution); diagnostic.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

C2R = {"Ath": None, "Atl": "ATL", "Az": "ARI", "Bal": "BAL", "Bos": "BOS",
       "Chc": "CHN", "Chw": "CHA", "Cin": "CIN", "Cle": "CLE", "Col": "COL",
       "Det": "DET", "Hou": "HOU", "Kc": "KCA", "Laa": "ANA", "Lad": "LAN",
       "Mia": "MIA", "Mil": "MIL", "Min": "MIN", "Nym": "NYN", "Nyy": "NYA",
       "Phi": "PHI", "Pit": "PIT", "Sd": "SDN", "Sea": "SEA", "Sf": "SFN",
       "Stl": "SLN", "Tb": "TBA", "Tex": "TEX", "Tor": "TOR", "Was": "WAS"}

def retro_of(code, season):
    if code == "Ath":
        return "ATH" if season >= 2025 else "OAK"
    return C2R.get(code)          # 'Al'/'Nl' = all-star junk -> None

cv = defaultdict(dict)
for r in csv.DictReader(open("data/covers_mlb.csv", encoding="utf-8")):
    season = int(r["date"][:4])
    h = retro_of(r["home"], season); a = retro_of(r["away"], season)
    if not h or not a or not r["home_pct"]:
        continue
    tot = (int(r["away_picks"] or 0) + int(r["home_picks"] or 0))
    cv[(r["date"], h, a)][r["variant"]] = (int(r["home_pct"]) / 100.0, tot)

mp = [r for r in csv.DictReader(open("data/model_probs.csv")) if r["season"] >= "2021"]
mk = Counter((r["date"], r["home"], r["away"]) for r in mp)
J = []
for r in mp:
    k = (r["date"], r["home"], r["away"])
    if mk[k] > 1 or k not in cv:
        continue
    y = float(r["y"])
    if y not in (0.0, 1.0):
        continue
    v = cv[k]
    J.append({"season": int(r["season"]), "y": y, "p": float(r["full_p"]),
              "ov": v.get("overall"), "ex": v.get("expert"), "tp": v.get("top10pct")})
print(f"joined {len(J)} games; per season:",
      dict(sorted(Counter(r['season'] for r in J).items())))

eps = 1e-12
def llv(y, p):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def stack(name, extra_of, min_picks=0):
    lls_b, lls_s, coefs = [], [], []
    for s_ in range(2022, 2026):
        tr = [r for r in J if r["season"] < s_ and extra_of(r) is not None]
        te = [r for r in J if r["season"] == s_ and extra_of(r) is not None]
        if len(tr) < 200 or len(te) < 100:
            continue
        Xtr = np.array([[logit(r["p"]), extra_of(r)] for r in tr])
        Xte = np.array([[logit(r["p"]), extra_of(r)] for r in te])
        ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
        m = LogisticRegression(C=1e6, max_iter=3000).fit(Xtr, ytr)
        ps = m.predict_proba(Xte)[:, 1]
        lls_b.append(llv(yte, np.array([r["p"] for r in te])))
        lls_s.append(llv(yte, ps))
        coefs.append(float(m.coef_[0][1]))
    if not lls_b:
        print(f"  {name:<26} insufficient data")
        return None
    b = np.concatenate(lls_b); s2 = np.concatenate(lls_s)
    d = b - s2
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
    print(f"  {name:<26} n={len(d):<5} base {b.mean():.5f} -> {s2.mean():.5f}  "
          f"delta {d.mean():+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  coef {np.mean(coefs):+.3f}")
    return {"n": int(len(d)), "delta": round(float(d.mean()), 5),
            "ci": [round(float(lo), 5), round(float(hi), 5)], "sig": sig,
            "mean_coef": round(float(np.mean(coefs)), 3)}

print("\nCovers consensus stacks (walk-forward, 2021 warm, score 2022-25):")
res = {}
res["overall"] = stack("a) overall pick share", lambda r: (r["ov"][0] - 0.5) if r["ov"] else None)
res["expert"] = stack("b) expert pick share", lambda r: (r["ex"][0] - 0.5) if r["ex"] else None)
res["top10pct"] = stack("c) top-10% pick share", lambda r: (r["tp"][0] - 0.5) if r["tp"] else None)
res["skill_diff"] = stack("d) top10 minus overall", lambda r: (r["tp"][0] - r["ov"][0]) if (r["tp"] and r["ov"]) else None)
json.dump(res, open("data/covers_mlb_test.json", "w"), indent=1)
print("\nwrote data/covers_mlb_test.json")
