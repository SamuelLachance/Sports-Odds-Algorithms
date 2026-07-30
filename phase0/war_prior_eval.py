"""Clean (leak-free) WAR read: use PRIOR-season team WAR as the feature.

Prior-season WAR is entirely in the past -> zero leak. It still carries WAR's unique
ingredients (defense, park, positional, replacement level) that our component model
lacks, so it is a fair test of whether an independent team-quality measure helps -- just
lagged one year (a legitimate preseason prior). Contrast with war_feature_eval.py, whose
season-TOTAL WAR is leaky (contains future games).
"""
import csv, os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Projects-Sports-Odds-Algorithms/c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad"
BR2RETRO = {
    "ANA": "ANA", "LAA": "ANA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CHW": "CHA", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "FLA": "FLO", "MIA": "MIA", "HOU": "HOU", "KCR": "KCA", "LAD": "LAN", "MIL": "MIL",
    "MIN": "MIN", "MON": "MON", "WSN": "WAS", "NYM": "NYN", "NYY": "NYA", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDN", "SFG": "SFN", "SEA": "SEA", "STL": "SLN",
    "TBD": "TBA", "TBR": "TBA", "TEX": "TEX", "TOR": "TOR",
}
RELOC_PRIOR = {"WAS": "MON", "MIA": "FLO"}   # map a team back to its prior-year code

team_war = defaultdict(float)
for fn in ["war_bat.txt", "war_pitch.txt"]:
    for r in csv.DictReader(open(os.path.join(SCRATCH, fn), encoding="latin-1")):
        yr = r["year_ID"]
        if not yr.isdigit():
            continue
        rt = BR2RETRO.get(r["team_ID"])
        if rt is None:
            continue
        try:
            team_war[(int(yr), rt)] += float(r["WAR"])
        except (ValueError, KeyError):
            continue

def prior(season, team):
    w = team_war.get((season - 1, team))
    if w is None and team in RELOC_PRIOR:
        w = team_war.get((season - 1, RELOC_PRIOR[team]))
    return w

rows = []; matched = 0; missing = 0
for r in csv.DictReader(open(os.path.join(HERE, "data", "model_probs.csv"))):
    if r["recbp_p"] == "" or r["full_p"] == "":
        continue
    s = int(r["season"])
    wh = prior(s, r["home"]); wa = prior(s, r["away"])
    if wh is None or wa is None:
        missing += 1
        wh = wh if wh is not None else 0.0   # neutral prior for expansion/relocation gaps
        wa = wa if wa is not None else 0.0
    matched += 1
    rows.append((s, float(r["recbp_p"]), float(r["full_p"]), wh - wa, int(r["y"])))

S = np.array([x[0] for x in rows])
prec = np.array([x[1] for x in rows]); pful = np.array([x[2] for x in rows])
war = np.array([x[3] for x in rows]); y = np.array([x[4] for x in rows])
print(f"matched {matched} games ({missing} used a neutral prior for reloc/expansion)")
print(f"prior-year WAR diff: mean {war.mean():+.2f}  sd {war.std():.2f}")

eps = 1e-9
def lg(p): p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def LL(p): p = np.clip(p, eps, 1 - eps); return -(y2 * np.log(p) + (1 - y2) * np.log(1 - p))
dev = S <= 2015
test = (S >= 2016) & (S != 2020)
war_z = (war - war[dev].mean()) / war[dev].std()
rng = np.random.default_rng(7)

def evaluate(base_p, name):
    global y2
    Xb = lg(base_p).reshape(-1, 1)
    Xa = np.column_stack([lg(base_p), war_z])
    base = LogisticRegression(C=1e6).fit(Xb[dev], y[dev])
    aug = LogisticRegression(C=1e6).fit(Xa[dev], y[dev])
    pb = base.predict_proba(Xb[test])[:, 1]; pa = aug.predict_proba(Xa[test])[:, 1]
    y2 = y[test]
    llb, lla = LL(pb).mean(), LL(pa).mean()
    delta = LL(pb) - LL(pa); n = len(delta)
    bs = delta[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
    print(f"\n{name} base (n_test={n})")
    print(f"  base-only LL              {llb:.5f}")
    print(f"  base + prior-WAR feature  {lla:.5f}   (delta {llb-lla:+.5f})")
    print(f"  prior-WAR coef in stack   {aug.coef_[0][1]:+.4f}")
    print(f"  bootstrap 95% CI on delta [{lo:+.5f}, {hi:+.5f}]  -> {sig}")

evaluate(prec, "recbp (no-lineup)")
evaluate(pful, "full (with-lineup)")
