"""Does OddsShark daily-API info add predictive value to the MLB model?

Joins data/oddsshark_mlb.csv to data/model_probs.csv (2023-2025, full_p =
shipped model prob) and tests, walk-forward by season (2023 warm; score
2024, 2025):
  a) de-vigged closing ML     (market diagnostic — quantifies gap, not shippable)
  b) fan-vote share           (crowd sentiment; the novel non-price field)
  c) roof status              (only if it VARIES within stadium, else static park id)
  d) total line               (market run-environment diagnostic)
All stacks: logistic on [logit(full_p), extra] trained on prior seasons.
Constitution: nothing here can ship as a model input; research diagnostic.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

NAME2RETRO = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHN", "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYN",
    "New York Yankees": "NYA", "Oakland Athletics": "OAK", "Athletics": "ATH",
    "Sacramento Athletics": "ATH", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN", "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN", "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
}

def imp(ml):
    if ml in (None, ""):
        return None
    v = float(ml)
    return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))

# ---- load OddsShark rows, keep completed games, resolve teams ----
os_rows = list(csv.DictReader(open("data/oddsshark_mlb.csv", encoding="utf-8")))
osd = defaultdict(list)
unmapped = Counter()
for r in os_rows:
    h = NAME2RETRO.get(r["home_name"]); a = NAME2RETRO.get(r["away_name"])
    if not h or not a:
        unmapped[r["home_name"] if not h else r["away_name"]] += 1
        continue
    osd[(r["date"], h, a)].append(r)
print(f"oddsshark: {len(os_rows)} rows, unmapped names: {dict(unmapped)}")

# ---- load model probs 2023-2025, join (drop ambiguous doubleheader keys) ----
mp = [r for r in csv.DictReader(open("data/model_probs.csv"))
      if "2023" <= r["season"] <= "2025"]
mp_keys = Counter((r["date"], r["home"], r["away"]) for r in mp)
J = []
for r in mp:
    k = (r["date"], r["home"], r["away"])
    if mp_keys[k] > 1 or len(osd.get(k, [])) != 1:
        continue
    o = osd[k][0]
    ph, pa = imp(o["ml_h"]), imp(o["ml_a"])
    J.append({
        "season": int(r["season"]), "y": float(r["y"]), "p_model": float(r["full_p"]),
        "p_ml": ph / (ph + pa) if ph and pa else None,
        "vsh": (float(o["votes_h"]) / (float(o["votes_h"]) + float(o["votes_a"])))
               if o["votes_h"] not in ("", None) and o["votes_a"] not in ("", None)
               and float(o["votes_h"] or 0) + float(o["votes_a"] or 0) >= 20 else None,
        "roof": o["roof"], "stadium": o["stadium"],
        "total": float(o["total"]) if o["total"] not in ("", None) else None,
    })
print(f"joined {len(J)}/{len(mp)} games "
      f"(dropped {sum(1 for k, c in mp_keys.items() if c > 1)} DH keys)")
for s in (2023, 2024, 2025):
    n = sum(1 for r in J if r["season"] == s)
    nv = sum(1 for r in J if r["season"] == s and r["vsh"] is not None)
    nm = sum(1 for r in J if r["season"] == s and r["p_ml"] is not None)
    print(f"  {s}: n={n}  ml={nm}  votes={nv}")

# ---- roof variance check ----
rv = defaultdict(Counter)
for r in J:
    if r["roof"]:
        rv[r["stadium"]][r["roof"]] += 1
dyn = {s: dict(c) for s, c in rv.items() if len(c) > 1}
print(f"\nroof: {len(rv)} stadiums, {len(dyn)} with WITHIN-stadium variation")
for s, c in list(dyn.items())[:8]:
    print("  ", s, c)

eps = 1e-12
def llv(y, p):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def stack_test(name, extra_of, subset=None, base_of=None):
    """walk-forward by season: train logistic [logit(base), extra] on prior
    seasons (2023 warm), score 2024+2025; baseline = base prob on same games.
    base defaults to the shipped model prob; pass base_of for e.g. the close."""
    if base_of is None:
        base_of = lambda r: r["p_model"]  # noqa: E731
    out = {}
    lls_b, lls_s = [], []
    for s in (2024, 2025):
        tr = [r for r in J if r["season"] < s and (subset is None or subset(r))
              and extra_of(r) is not None and base_of(r) is not None
              and r["y"] in (0.0, 1.0)]
        te = [r for r in J if r["season"] == s and (subset is None or subset(r))
              and extra_of(r) is not None and base_of(r) is not None
              and r["y"] in (0.0, 1.0)]
        if len(tr) < 200 or len(te) < 100:
            continue
        Xtr = np.array([[logit(base_of(r)), extra_of(r)] for r in tr])
        Xte = np.array([[logit(base_of(r)), extra_of(r)] for r in te])
        ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
        m = LogisticRegression(C=1e6, max_iter=3000).fit(Xtr, ytr)
        p_s = m.predict_proba(Xte)[:, 1]
        p_b = np.array([base_of(r) for r in te])
        lls_b.append(llv(yte, p_b)); lls_s.append(llv(yte, p_s))
        out[s] = {"n": len(te), "ll_base": round(float(llv(yte, p_b).mean()), 5),
                  "ll_stack": round(float(llv(yte, p_s).mean()), 5),
                  "coef_extra": round(float(m.coef_[0][1]), 4)}
    if not lls_b:
        print(f"  {name:<28} insufficient data")
        return {"pooled": None}
    b = np.concatenate(lls_b); s_ = np.concatenate(lls_s)
    d = b - s_
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pooled = {"n": int(len(d)), "ll_base": round(float(b.mean()), 5),
              "ll_stack": round(float(s_.mean()), 5),
              "delta": round(float(d.mean()), 5),
              "ci": [round(float(lo), 5), round(float(hi), 5)]}
    sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
    print(f"  {name:<28} n={pooled['n']:<5} base {pooled['ll_base']:.5f} -> "
          f"stack {pooled['ll_stack']:.5f}  delta {pooled['delta']:+.5f} "
          f"CI [{lo:+.5f},{hi:+.5f}]  {sig}")
    return {"pooled": pooled, "by_season": out}

print("\nstacking tests (walk-forward, 2023 warm, score 2024-2025):")
res = {}
res["close_ml"] = stack_test("a) devig closing ML", lambda r: None if r["p_ml"] is None else logit(r["p_ml"]))
res["votes"] = stack_test("b) fan-vote share", lambda r: None if r["vsh"] is None else r["vsh"] - 0.5)
if dyn:
    res["roof"] = stack_test("c) roof closed/dome",
                             lambda r: 1.0 if r["roof"] in ("Closed", "Dome", "Retractable - Closed") else (0.0 if r["roof"] else None),
                             subset=lambda r: r["stadium"] in dyn)
else:
    print("  c) roof: STATIC per stadium -> park identity, contained; skipped")
    res["roof"] = {"pooled": None, "note": "static attribute, no within-stadium variation"}
res["total"] = stack_test("d) total line", lambda r: r["total"])
res["votes_plus_ml"] = stack_test("e) votes ON TOP of close",
                                  lambda r: None if r["vsh"] is None else r["vsh"] - 0.5,
                                  base_of=lambda r: r["p_ml"])

json.dump(res, open("data/oddsshark_mlb_test.json", "w"), indent=1)
print("\nwrote data/oddsshark_mlb_test.json")
