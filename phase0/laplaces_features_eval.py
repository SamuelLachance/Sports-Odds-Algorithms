"""Test the laplaces42 predictor's ENTIRE feature set as an increment over our model.

blended_stats.csv = their per-game, leak-free (as-of, projection-blended) team stat
lines: offense (avg/obp/slg/woba/wrc+/war/k%/bb%) + pitching (k9/bb9/hr9/era/fip/owar).
Verified as-of: within-season values evolve from a game-1 projection toward season-end.

For each game we form home-minus-away diffs of all 14 features and ask whether a
regularized logistic stack on top of our base model prob improves TEST log loss.
Their data starts 2010, so DEV = 2010-2015, TEST = 2016-2024 ex-2020 (locked).
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

def retro(tid, yr):
    if tid == 146:
        return "FLO" if yr < 2012 else "MIA"
    return TID.get(tid)

# (date, retro_team) -> list of feature vectors (list handles doubleheaders)
look = defaultdict(list)
for r in bl.itertuples(index=False):
    rt = retro(r.offensive_team_id, r.year)
    if rt is None:
        continue
    look[(r.game_date, rt)].append(np.array([getattr(r, f) for f in FEATS], float))

# join to model_probs
mp = pd.read_csv(os.path.join(HERE, "data", "model_probs.csv"))
mp = mp[(mp.season >= 2010) & mp.recbp_p.notna() & mp.full_p.notna()]
seen = defaultdict(int)
S, PREC, PFUL, Y, X = [], [], [], [], []
matched = 0
for r in mp.itertuples(index=False):
    hk, ak = (r.date, r.home), (r.date, r.away)
    hl, al = look.get(hk), look.get(ak)
    if not hl or not al:
        continue
    ih = min(seen[hk], len(hl) - 1); ia = min(seen[ak], len(al) - 1)
    seen[hk] += 1; seen[ak] += 1
    matched += 1
    X.append(hl[ih] - al[ia])          # home-minus-away feature diffs
    S.append(r.season); PREC.append(r.recbp_p); PFUL.append(r.full_p); Y.append(r.y)

S = np.array(S); prec = np.array(PREC); pful = np.array(PFUL); y = np.array(Y); X = np.array(X)
print(f"matched {matched} games ({matched/len(mp):.0%} of 2010+ model rows) x {len(FEATS)} features")

eps = 1e-9
def lg(p): p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def LL(p): p = np.clip(p, eps, 1 - eps); return -(y2 * np.log(p) + (1 - y2) * np.log(1 - p))
dev = S <= 2015
test = (S >= 2016) & (S != 2020)
mu, sd = X[dev].mean(0), X[dev].std(0)
Xz = (X - mu) / sd
rng = np.random.default_rng(7)

def evaluate(base_p, name, C):
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
    top = sorted(zip(FEATS, aug.coef_[0][1:]), key=lambda t: -abs(t[1]))[:4]
    print(f"\n{name} base + all 14 features (C={C}, n_test={n})")
    print(f"  base-only LL   {llb:.5f}")
    print(f"  base + theirs  {lla:.5f}   (delta {llb-lla:+.5f})")
    print(f"  95% CI on delta [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
    print(f"  biggest coefs: " + ", ".join(f"{f}{c:+.3f}" for f, c in top))

for base_p, name in [(prec, "recbp"), (pful, "full")]:
    for C in (1.0, 0.1):
        evaluate(base_p, name, C)
