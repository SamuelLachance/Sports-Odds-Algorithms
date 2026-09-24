"""Breakthrough program - NHL idea sizing: SHORT vs LONG absences. DEV ONLY.

Row 10's absent-regular feature decays a player's dressed share on every missed
game (SH_DECAY 0.90), so a star on long-term injured reserve stops counting as
"absent" after ~7 games - while the team Elo has not yet absorbed his loss.
This diagnostic asks whether the close keeps pricing absences past that point.
Odds used ONLY for evaluation (close-minus-model residual loadings), DEV only.
Writes data/bt_nhl_ideas_longabs.json.
"""
import csv, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "phase0")
from nhl_features_eval import build_features
from nhl_glicko2_eval import llv, load_games, run_elo
from nhl_xg_screen import loso

DEV = [20112012, 20122013, 20132014, 20142015, 20152016, 20162017, 20172018]
games = [g for g in load_games() if g["season"] <= 20172018]
assert max(g["season"] for g in games) <= 20172018
keep = {g["game_id"] for g in games}
n = len(games)
dressed = defaultdict(dict)
for r in csv.DictReader(open("data/nhl_dressed.csv", encoding="utf-8")):
    gid = int(r["game_id"])
    if gid in keep:
        dressed[gid].setdefault(r["team"], {})[r["player_id"]] = float(r["toi_s"])
gk = set()
for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
    gk.add(r["goalie_id"])

# per player state: team, team-game index at last dressing, toi EWMA, n dressed on this team
last_team, last_idx, toi_e, ndr = {}, {}, {}, defaultdict(int)
tcount = defaultdict(int)          # team -> team games (all-time)
short = np.full((n, 2), np.nan); long_ = np.full((n, 2), np.nan)
for i, g in enumerate(games):
    for j, side in enumerate(("home", "away")):
        t = g[side]
        d = dressed.get(g["game_id"], {}).get(t)
        if not d:
            continue
        s = l = 0.0
        for p, tm in last_team.items():
            if tm != t or p in d or p in gk:
                continue
            if ndr[p] < 20 or toi_e[p] < 720:          # established regular, >=12 min
                continue
            gap = tcount[t] - last_idx[p]               # team games missed so far (before tonight)
            if gap < 0 or gap > 60:
                continue
            if gap < 7:
                s += toi_e[p] / 60
            else:
                l += toi_e[p] / 60
        short[i, j], long_[i, j] = s, l
    for side in ("home", "away"):
        t = g[side]
        d = dressed.get(g["game_id"], {}).get(t)
        if not d:
            continue
        for p, v in d.items():
            if last_team.get(p) != t:
                ndr[p] = 0; toi_e[p] = v
            ndr[p] += 1
            toi_e[p] = 0.8 * toi_e[p] + 0.2 * v
            last_team[p] = t
            last_idx[p] = tcount[t] + 1
        tcount[t] += 1

model = json.load(open("data/nhl_model.json"))
pe = np.clip(np.array([o[1] for o in run_elo(games, **model["elo_cfg"])]), 1e-9, 1 - 1e-9)
el = np.log(pe / (1 - pe)); F = build_features(games)
seas = np.array([g["season"] for g in games]); y = np.array([g["y"] for g in games])
odds = {}
for r in csv.DictReader(open("data/odds_nhl.csv")):
    try: ho, ao = float(r["home_close"]), float(r["away_close"])
    except ValueError: continue
    if ho > 1 and ao > 1: odds[(r["date"], r["home"], r["away"])] = (1 / ho) / (1 / ho + 1 / ao)
pc = np.array([odds.get((g["date"], g["home"], g["away"]), np.nan) for g in games])
dev = np.isin(seas, DEV) & ~np.isnan(F["xg_diff"])
X = np.column_stack([el, F["rest_diff"], F["b2b_home"], F["b2b_away"], F["xg_diff"]])
pm = np.full(n, np.nan); pm[dev] = loso(X[dev], y[dev], seas[dev], DEV)
m = dev & ~np.isnan(pc) & ~np.isnan(short).any(1) & ~np.isnan(long_).any(1)
assert set(np.unique(seas[m])) <= set(DEV)
lm = np.log(pm / (1 - pm)); lc = np.log(pc / (1 - pc)); r = lc - lm
sd = short[:, 1] - short[:, 0]; ld = long_[:, 1] - long_[:, 0]      # away minus home (+ = home stronger)
Z = np.column_stack([np.ones(m.sum()), sd[m], ld[m], lm[m]])
beta, *_ = np.linalg.lstsq(Z, r[m], rcond=None)
res = r[m] - Z @ beta
se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * res.var())
out = {"n": int(m.sum()),
       "short_abs_min_diff": [float(beta[1]), float(beta[1] / se[1]), float(np.std(sd[m]))],
       "long_abs_min_diff": [float(beta[2]), float(beta[2] / se[2]), float(np.std(ld[m]))],
       "share_games_with_long_abs": float(np.mean((long_[m] > 0).any(1)))}
print(json.dumps(out, indent=1))
# OOF market-taught shift value of each
def oof(cols):
    Zf = np.column_stack([np.ones(n)] + cols)
    shv = np.full(n, np.nan)
    for s in DEV:
        tr = m & (seas != s); te = m & (seas == s)
        b, *_ = np.linalg.lstsq(Zf[tr], r[tr], rcond=None); shv[te] = Zf[te, 1:] @ b[1:]
    pz = 1 / (1 + np.exp(-(lm + shv)))
    return float(llv(y[m], pm[m]).mean() - llv(y[m], pz[m]).mean())
out["oof_gain_short"] = oof([sd]); out["oof_gain_long"] = oof([ld]); out["oof_gain_both"] = oof([sd, ld])
print({k: round(v, 5) for k, v in out.items() if k.startswith("oof")})
json.dump(out, open("data/bt_nhl_ideas_longabs.json", "w"), indent=1)
