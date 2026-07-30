"""Does 'The Menu' sheet's labels (sharp / RLM / book-needs / squares) carry
predictive information for MLB games — beyond the close, and beyond our model?

Samuel's hypothesis: betting-return losses don't preclude predictive value in
the LABELS. Test (Flepp-style efficiency regression + out-of-time stack):

  For each category: flag_g = +1 if the labeled team is home, -1 if away, 0 none.
  (a) y ~ [logit(p_close), flag]  over Mar-2025..Jul-2026 (OddsShark devig close)
  (b) y ~ [logit(p_model), flag]  over Mar..Sep-2025 (model_probs overlap)
  Report the flag coefficient with a bootstrap CI (does the CI exclude 0?) and
  the out-of-time LL delta (chronological 60/40 split).

Moneyline/side picks only (UNDER/OVER rows are totals — skipped). Crowd/market
label data: evaluation-layer diagnostic only (constitution).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

NAME2RETRO = {
    "TORONTO": "TOR", "TORONTO BLUE JAYS": "TOR", "BLUE JAYS": "TOR",
    "BOSTON": "BOS", "RED SOX": "BOS", "BOSTON RED SOX": "BOS",
    "NY YANKEES": "NYA", "NEW YANKEES": "NYA", "YANKEES": "NYA", "NEW YORK YANKEES": "NYA",
    "NY METS": "NYN", "METS": "NYN", "NEW YORK METS": "NYN",
    "BALTIMORE": "BAL", "ORIOLES": "BAL", "TAMPA BAY": "TBA", "TAMPA": "TBA", "RAYS": "TBA",
    "CHICAGO CUBS": "CHN", "CUBS": "CHN", "CHI CUBS": "CHN",
    "CHICAGO WHITE SOX": "CHA", "WHITE SOX": "CHA", "CHI WHITE SOX": "CHA", "CHI. WHITE SOX": "CHA",
    "CLEVELAND": "CLE", "GUARDIANS": "CLE", "DETROIT": "DET", "TIGERS": "DET",
    "KANSAS CITY": "KCA", "ROYALS": "KCA", "MINNESOTA": "MIN", "TWINS": "MIN",
    "HOUSTON": "HOU", "ASTROS": "HOU", "LA ANGELS": "ANA", "ANGELS": "ANA", "LOS ANGELES ANGELS": "ANA",
    "ATHLETICS": "ATH", "OAKLAND": "ATH", "AS": "ATH", "A'S": "ATH",
    "SEATTLE": "SEA", "MARINERS": "SEA", "TEXAS": "TEX", "TEXAS RANGERS": "TEX", "RANGERS": "TEX",
    "ATLANTA": "ATL", "BRAVES": "ATL", "MIAMI": "MIA", "MARLINS": "MIA",
    "PHILADELPHIA": "PHI", "PHILLIES": "PHI", "WASHINGTON": "WAS", "NATIONALS": "WAS",
    "CINCINNATI": "CIN", "REDS": "CIN", "MILWAUKEE": "MIL", "BREWERS": "MIL",
    "PITTSBURGH": "PIT", "PIRATES": "PIT", "ST LOUIS": "SLN", "ST. LOUIS": "SLN", "CARDINALS": "SLN",
    "ARIZONA": "ARI", "DIAMONDBACKS": "ARI", "ARIZONA DIAMONDBACKS": "ARI", "DBACKS": "ARI",
    "COLORADO": "COL", "ROCKIES": "COL", "LA DODGERS": "LAN", "DODGERS": "LAN", "LOS ANGELES DODGERS": "LAN",
    "SAN DIEGO": "SDN", "PADRES": "SDN", "SAN FRANCISCO": "SFN", "GIANTS": "SFN", "SF GIANTS": "SFN",
    "CHICAGO": None, "NEW YORK": None, "LOS ANGELES": None,   # ambiguous alone
}
CATS = ("sharp", "rlm", "book_needs", "squares", "model_best", "mega_sharps")


def team_of(text):
    t = text.upper().strip()
    if re.search(r"\b(UNDER|OVER|O\d|U\d)\b", t):
        return None                                     # totals pick
    t = re.sub(r"[+\-]\d+(\.\d+)?", " ", t)             # strip odds/lines
    t = re.sub(r"\b\d+K\b", " ", t)                     # whale amounts
    t = re.sub(r"\b\d{1,2}/\d{1,2}\b", " ", t)          # date tags
    t = re.sub(r"[^A-Z.' ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    if t in NAME2RETRO:
        return NAME2RETRO[t]
    for name, code in sorted(NAME2RETRO.items(), key=lambda x: -len(x[0])):
        if code and (t.startswith(name) or name in t):
            return code
    return None


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def llv(y, p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    # ---- picks -> per (date, team, category) ----
    picks = defaultdict(set)
    n_ml, n_tot, n_unmap = 0, 0, Counter()
    for r in csv.DictReader(open("data/menu_picks.csv", encoding="utf-8")):
        if r["league"] != "MLB" or r["category"] not in CATS:
            continue
        code = team_of(r["text"])
        if code is None:
            if re.search(r"\b(UNDER|OVER)\b", r["text"].upper()):
                n_tot += 1
            else:
                n_unmap[r["text"][:30]] += 1
            continue
        n_ml += 1
        picks[(r["date"], r["category"])].add(code)
    print(f"MLB side-picks mapped: {n_ml:,} | totals skipped: {n_tot:,} | "
          f"unmapped: {sum(n_unmap.values())}")
    if n_unmap:
        print("  top unmapped:", n_unmap.most_common(5))

    # ---- games: OddsShark close (2025-03-01+) + model probs ----
    def imp(ml):
        if ml in (None, ""):
            return None
        v = float(ml)
        return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))

    NAME2R = {  # oddsshark full names -> retro (reuse from oddsshark test)
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS", "Chicago Cubs": "CHN", "Chicago White Sox": "CHA",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
        "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCA",
        "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN", "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYN",
        "New York Yankees": "NYA", "Athletics Athletics": "ATH", "Athletics": "ATH",
        "Oakland Athletics": "ATH", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SDN", "San Francisco Giants": "SFN", "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "SLN", "Tampa Bay Rays": "TBA", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS"}
    games = []
    seen = Counter()
    for r in csv.DictReader(open("data/oddsshark_mlb.csv", encoding="utf-8")):
        if r["date"] < "2025-03-01" or r["status"] not in ("Complete", "FINAL", "Final"):
            continue
        h, a = NAME2R.get(r["home_name"]), NAME2R.get(r["away_name"])
        ph, pa = imp(r["ml_h"]), imp(r["ml_a"])
        if not h or not a or not ph or not pa or r["hs"] in ("", None):
            continue
        k = (r["date"], h, a)
        seen[k] += 1
        games.append({"date": r["date"], "home": h, "away": a,
                      "y": 1.0 if int(r["hs"]) > int(r["as"]) else 0.0,
                      "p_close": ph / (ph + pa)})
    games = [g for g in games if seen[(g["date"], g["home"], g["away"])] == 1]
    games.sort(key=lambda g: g["date"])
    print(f"joined close games (Mar-2025+, DH dropped): {len(games):,}")

    mp = {}
    mpk = Counter()
    for r in csv.DictReader(open("data/model_probs.csv")):
        if r["season"] == "2025":
            k = (r["date"], r["home"], r["away"])
            mpk[k] += 1
            mp[k] = float(r["full_p"])

    # ---- flags ----
    for g in games:
        for cat in CATS:
            s = picks.get((g["date"], cat), ())
            g[cat] = (1.0 if g["home"] in s else 0.0) - (1.0 if g["away"] in s else 0.0)
        k = (g["date"], g["home"], g["away"])
        g["p_model"] = mp.get(k) if mpk.get(k, 0) == 1 else None

    y = np.array([g["y"] for g in games])
    pc = np.array([g["p_close"] for g in games])
    res = {}
    print(f"\n{'category':<12} {'n_flag':>6} | coef vs CLOSE [CI]            | "
          f"OOT dLL | coef vs MODEL [CI]", flush=True)
    for cat in CATS:
        f = np.array([g[cat] for g in games])
        nf = int((f != 0).sum())
        if nf < 60:
            print(f"{cat:<12} {nf:>6} | too few", flush=True)
            continue
        # (a) efficiency regression vs close, all games; bootstrap coef CI
        X = np.column_stack([logit(pc), f])
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        coef = float(m.coef_[0][1])
        rng = np.random.default_rng(7)
        bs = []
        idx = np.arange(len(y))
        for _ in range(600):
            s_ = rng.choice(idx, size=len(idx), replace=True)
            try:
                bs.append(float(LogisticRegression(C=1e6, max_iter=500)
                                .fit(X[s_], y[s_]).coef_[0][1]))
            except Exception:  # noqa: BLE001
                pass
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # out-of-time: fit on first 60%, score last 40% (flagged games only)
        cut = int(len(games) * 0.6)
        m2 = LogisticRegression(C=1e6, max_iter=2000).fit(X[:cut], y[:cut])
        te = np.arange(cut, len(games))
        te_f = te[f[te] != 0]
        d_ll = 0.0
        if len(te_f) > 30:
            p_base = pc[te_f]
            p_stack = m2.predict_proba(X[te_f])[:, 1]
            d_ll = float(llv(y[te_f], p_base).mean() - llv(y[te_f], p_stack).mean())
        # (b) vs model on 2025 overlap
        mm = np.array([g["p_model"] is not None and g[cat] != 0 for g in games])
        cm, cl, ch = 0.0, 0.0, 0.0
        if mm.sum() >= 60:
            sub = [g for g, keep in zip(games, mm) if keep]
            ym = np.array([g["y"] for g in sub])
            Xm = np.column_stack([logit(np.array([g["p_model"] for g in sub])),
                                  np.array([g[cat] for g in sub])])
            m3 = LogisticRegression(C=1e6, max_iter=2000).fit(Xm, ym)
            cm = float(m3.coef_[0][1])
            bs2 = []
            for _ in range(600):
                s_ = rng.choice(len(ym), size=len(ym), replace=True)
                try:
                    bs2.append(float(LogisticRegression(C=1e6, max_iter=500)
                                     .fit(Xm[s_], ym[s_]).coef_[0][1]))
                except Exception:  # noqa: BLE001
                    pass
            cl, ch = np.percentile(bs2, [2.5, 97.5])
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        sigm = "SIG" if (cl > 0 or ch < 0) and mm.sum() >= 60 else "n.s."
        print(f"{cat:<12} {nf:>6} | {coef:+.3f} [{lo:+.3f},{hi:+.3f}] {sig:<5}| "
              f"{d_ll:+.4f} | {cm:+.3f} [{cl:+.3f},{ch:+.3f}] {sigm} (n={int(mm.sum())})",
              flush=True)
        res[cat] = {"n_flag": nf, "coef_close": round(coef, 4),
                    "ci_close": [round(float(lo), 4), round(float(hi), 4)],
                    "oot_dll": round(d_ll, 5), "coef_model": round(cm, 4),
                    "ci_model": [round(float(cl), 4), round(float(ch), 4)],
                    "n_model": int(mm.sum())}
    json.dump(res, open("data/menu_mlb_test.json", "w"), indent=1)
    print("\nwrote data/menu_mlb_test.json")


if __name__ == "__main__":
    main()
