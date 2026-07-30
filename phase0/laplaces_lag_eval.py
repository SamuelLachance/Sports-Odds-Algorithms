"""Adversarial re-test of the laplaces42 feature set with a ONE-GAME LAG.

The raw blended stats are season-to-date computed with Fangraphs enddate={game date},
which INCLUDES the game being predicted -> the team's to-date wOBA/ERA leak that game's
runs scored/allowed. Fix: for each team, use its blended row from the PREVIOUS game
(strictly earlier, same season) as the feature. If the +0.008 improvement collapses to
the ~null my own leak-free wOBA test found, the improvement was current-game leakage.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Projects-Sports-Odds-Algorithms/c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad"
FEATS = ["avg", "obp", "slg", "woba", "wrc_plus", "war", "k_pct", "bb_pct",
         "k_per_9", "bb_per_9", "hr_per_9", "era", "fip", "owar"]
TID = {108: "ANA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHN", 113: "CIN",
       114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCA", 119: "LAN",
       120: "WAS", 121: "NYN", 133: "OAK", 134: "PIT", 135: "SDN", 136: "SEA",
       137: "SFN", 138: "SLN", 139: "TBA", 140: "TEX", 141: "TOR", 142: "MIN",
       143: "PHI", 144: "ATL", 145: "CHA", 146: "MIA", 147: "NYA", 158: "MIL"}

bl = pd.read_csv(os.path.join(SCRATCH, "blended.csv"))
bl["year"] = pd.to_datetime(bl["game_date"]).dt.year
bl["retro"] = [("FLO" if (t == 146 and y < 2012) else ("MIA" if t == 146 else TID.get(t)))
               for t, y in zip(bl.offensive_team_id, bl.year)]
bl = bl.dropna(subset=["retro"]).sort_values(["retro", "year", "game_date", "game_id"])

# LAG: previous game's features within the same team-season
lag = bl.groupby(["retro", "year"])[FEATS].shift(1)
bl_lag = bl[["game_date", "retro"]].copy()
for f in FEATS:
    bl_lag[f] = lag[f]
bl_lag = bl_lag.dropna(subset=FEATS)

look = defaultdict(list)
for r in bl_lag.itertuples(index=False):
    look[(r.game_date, r.retro)].append(np.array([getattr(r, f) for f in FEATS], float))

mp = pd.read_csv(os.path.join(HERE, "data", "model_probs.csv"))
mp = mp[(mp.season >= 2010) & mp.recbp_p.notna() & mp.full_p.notna()]
seen = defaultdict(int)
S, PREC, PFUL, Y, X = [], [], [], [], []
for r in mp.itertuples(index=False):
    hl, al = look.get((r.date, r.home)), look.get((r.date, r.away))
    if not hl or not al:
        continue
    ih = min(seen[(r.date, r.home)], len(hl) - 1); ia = min(seen[(r.date, r.away)], len(al) - 1)
    seen[(r.date, r.home)] += 1; seen[(r.date, r.away)] += 1
    X.append(hl[ih] - al[ia]); S.append(r.season)
    PREC.append(r.recbp_p); PFUL.append(r.full_p); Y.append(r.y)

S = np.array(S); prec = np.array(PREC); pful = np.array(PFUL); y = np.array(Y); X = np.array(X)
print(f"matched {len(S)} games (one-game-lagged, leak-free) x {len(FEATS)} features")

eps = 1e-9
def lg(p): p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def LL(p): p = np.clip(p, eps, 1 - eps); return -(y2 * np.log(p) + (1 - y2) * np.log(1 - p))
dev = S <= 2015
test = (S >= 2016) & (S != 2020)
mu, sd = X[dev].mean(0), X[dev].std(0)
Xz = (X - mu) / sd
rng = np.random.default_rng(7)

def evaluate(base_p, name, C=1.0):
    global y2
    Xb = lg(base_p).reshape(-1, 1)
    Xa = np.column_stack([lg(base_p), Xz])
    base = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[dev], y[dev])
    aug = LogisticRegression(C=C, max_iter=5000).fit(Xa[dev], y[dev])
    pb = base.predict_proba(Xb[test])[:, 1]; pa = aug.predict_proba(Xa[test])[:, 1]
    y2 = y[test]
    llb, lla = LL(pb).mean(), LL(pa).mean()
    delta = LL(pb) - LL(pa); n = len(delta)
    bs = delta[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
    print(f"\n{name} base + 14 lagged features (n_test={n})")
    print(f"  base-only LL   {llb:.5f}")
    print(f"  base + theirs  {lla:.5f}   (delta {llb-lla:+.5f})")
    print(f"  95% CI on delta [{lo:+.5f}, {hi:+.5f}]  -> {sig}")

evaluate(prec, "recbp")
evaluate(pful, "full")
