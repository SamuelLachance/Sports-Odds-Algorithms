"""Test team WAR (Baseball-Reference) as an incremental feature over the frozen model.

LEAKY UPPER BOUND: uses full-SEASON team WAR (bat+pitch), which includes games AFTER
each contest -> strictly more informative than the season-to-date WAR the laplaces42
repo uses. If even this can't beat our model, WAR is conclusively dead (same logic that
killed Statcast xwOBA). WAR source = BR public bulk files war_daily_{bat,pitch}.txt.

Increment protocol: fit y ~ base_logit + war_diff on DEV (2000-2015), evaluate on TEST
(2016-2024 ex-2020), paired bootstrap on the per-game log-loss delta vs base-only.
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

# ---- team-season WAR (bat + pitch), summed over stints ----
team_war = defaultdict(float)
for fn, col in [("war_bat.txt", "WAR"), ("war_pitch.txt", "WAR")]:
    for r in csv.DictReader(open(os.path.join(SCRATCH, fn), encoding="latin-1")):
        yr = r["year_ID"]
        if not yr.isdigit():
            continue
        rt = BR2RETRO.get(r["team_ID"])
        if rt is None:
            continue
        try:
            team_war[(int(yr), rt)] += float(r[col])
        except (ValueError, KeyError):
            continue

# ---- join to model_probs ----
rows = []; matched = 0
for r in csv.DictReader(open(os.path.join(HERE, "data", "model_probs.csv"))):
    if r["recbp_p"] == "" or r["full_p"] == "":
        continue
    s = int(r["season"])
    wh = team_war.get((s, r["home"])); wa = team_war.get((s, r["away"]))
    if wh is None or wa is None:
        continue
    matched += 1
    rows.append((s, float(r["recbp_p"]), float(r["full_p"]), wh - wa, int(r["y"])))

S = np.array([x[0] for x in rows])
prec = np.array([x[1] for x in rows]); pful = np.array([x[2] for x in rows])
war = np.array([x[3] for x in rows]); y = np.array([x[4] for x in rows])
print(f"matched {matched} games with team WAR")
print(f"WAR diff (home-away, season total): mean {war.mean():+.2f}  sd {war.std():.2f}")

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
    pb = base.predict_proba(Xb[test])[:, 1]
    pa = aug.predict_proba(Xa[test])[:, 1]
    y2 = y[test]
    llb, lla = LL(pb).mean(), LL(pa).mean()
    delta = LL(pb) - LL(pa)
    n = len(delta)
    bs = delta[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
    print(f"\n{name} base (n_test={n})")
    print(f"  base-only LL         {llb:.5f}")
    print(f"  base + WAR feature   {lla:.5f}   (delta {llb-lla:+.5f})")
    print(f"  WAR coef in stack    {aug.coef_[0][1]:+.4f}")
    print(f"  bootstrap 95% CI on delta LL [{lo:+.5f}, {hi:+.5f}]  -> {sig}")

evaluate(prec, "recbp (no-lineup)")
evaluate(pful, "full (with-lineup)")
