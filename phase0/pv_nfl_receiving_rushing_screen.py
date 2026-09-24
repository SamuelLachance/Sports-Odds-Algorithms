"""Player-value program (pv), NFL receiving_rushing, STEP 4: one DEV game-level screen.

Asks whether the player-level receiving / rushing ratings of STEP 2 add to the
SHIPPED 14-feature blend on DEV (2006-2015), using the shipped walk-forward
harness exactly as phase0/bt_nfl_ideas_quick2.py defines it (expanding refit per
season, recency half-life 3 seasons, C=100, new columns z-scored on the train
fold). DEV ONLY - no season >= 2016 row exists in the spine.

Feature (strictly pre-game, no same-game information):
  projected skill lineup of a team = WR/TE/RB who appeared for it in the team's
  PREVIOUS game (first game of a season: the team's last game of the prior
  season); each weighted by his pre-game usage share.
    rec  = sum pre_tsh * 34 * (mu_xypt + mu_yptoe)   receiving yards/game over avg
    rush = sum pre_csh * 26 * mu_ryds                  rushing yards/game over avg
  (player states are the STEP 2 pre-game values of that player's most recent
  appearance before this game, i.e. strictly before kickoff).
Pre-declared variants (no knob search):
  S1 add [rec_diff, rush_diff]; S2 add [rec_diff + rush_diff]; S3 add [rec_diff]
  S4 (POST-HOC, added after S1-S3 were seen) the same two columns in EPA units:
     rec_e = sum pre_tsh * 34 * (mu_xepat + mu_epat), rush_e = sum pre_csh * 26 * mu_epar
Harness sanity: baseline must reproduce 0.62294 and dropping the QB column must
cost about what bt_nfl_ideas_quick2 recorded (-0.00615).

Output: data/pv_nfl_receiving_rushing_screen.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260927)

D = json.load(open(os.path.join(DATA, "bt_nfl_ideas_dev.json"), encoding="utf-8"))
G = D["games"]
X14 = np.load(os.path.join(DATA, "bt_nfl_ideas_dev_X.npy"))
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
N = len(G)


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------------------------ feature
PG = pd.read_parquet(os.path.join(DATA, "pv_nfl_receiving_rushing_player_games.parquet"))
PG = PG[PG.season < TEST_ERA]
assert PG.season.max() < TEST_ERA
PLY = pd.read_csv(os.path.join(DATA, "nfl_players.csv"), usecols=["gsis_id", "position_group"], low_memory=False)
POSG = dict(zip(PLY.gsis_id, PLY.position_group))
PG = PG[PG.player_id.map(POSG).isin({"WR", "TE", "RB"})].copy()
PG["vrec"] = PG.pre_tsh * 34.0 * (PG.mu_xypt + PG.mu_yptoe)
PG["vrush"] = PG.pre_csh * 26.0 * PG.mu_ryds
PG["vrec_e"] = PG.pre_tsh * 34.0 * (PG.mu_xepat + PG.mu_epat)   # S4 (post-hoc): EPA units
PG["vrush_e"] = PG.pre_csh * 26.0 * PG.mu_epar
EVH = pd.read_parquet(os.path.join(DATA, "pv_nfl_events.parquet"), columns=["game_id", "season", "home_team"],
                      filters=[("season", "<", TEST_ERA)]).drop_duplicates("game_id")
assert EVH.season.max() < TEST_ERA
HOME = dict(zip(EVH.game_id, EVH.home_team))

# walk games in order; per team remember the lineup of its previous game; per player his latest
# pre-game state (updated only AFTER the current game's feature is computed)
PG = PG.sort_values(["gidx", "team"])
last_state = {}          # pid -> (vrec, vrush) from his most recent pre-game snapshot
prev_lineup = {}         # team -> list of pids in the team's previous game
feat = {}                # (game_id, side) -> (rec, rush, n_players)
for g, d in PG.groupby("gidx", sort=True):
    gid = d.game_id.iat[0]
    teams = d.team.unique()
    # 1) feature from the PREVIOUS lineup and states known before this game
    for t in teams:
        side = "H" if t == HOME.get(gid) else "A"
        lu = prev_lineup.get(t, [])
        feat[(gid, side)] = tuple(sum(last_state[p][k] for p in lu if p in last_state) for k in range(4))
    # 2) now reveal this game: its players' pre-game states become the latest known states
    #    (a pre-game state is pre-game for THIS game, so it may be used by LATER games only)
    for r in d.itertuples(index=False):
        last_state[r.player_id] = (r.vrec, r.vrush, r.vrec_e, r.vrush_e)
    for t in teams:
        prev_lineup[t] = d.player_id[d.team == t].tolist()

rec_d = np.zeros(N); rush_d = np.zeros(N); rece_d = np.zeros(N); rushe_d = np.zeros(N); miss = 0
for i, g in enumerate(G):
    h = feat.get((g["gid"], "H")); a = feat.get((g["gid"], "A"))
    if h is None or a is None:
        miss += 1
        continue
    rec_d[i] = h[0] - a[0]; rush_d[i] = h[1] - a[1]; rece_d[i] = h[2] - a[2]; rushe_d[i] = h[3] - a[3]
dv = SEAS >= DEV_LO
print(f"feature coverage: {N - miss}/{N} spine games (missing = 3 pbp-less 1999-2000 games or first games)")
print(f"DEV feature sd: rec {rec_d[dv].std():.2f} yds/g, rush {rush_d[dv].std():.2f} yds/g; "
      f"corr with shipped cols (DEV): " + ", ".join(f"c{k} {np.corrcoef(rec_d[dv], X14[dv, k])[0, 1]:+.2f}"
                                                  for k in range(14)))


# ------------------------------------------------------------------ harness (verbatim logic of quick2.wf)
def wf(extra=None, drop=None):
    X = X14.copy()
    if drop:
        X = np.delete(X, drop, axis=1)
    per, vec = {}, []
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X
        if extra is not None:
            E = np.column_stack(extra) if isinstance(extra, (list, tuple)) else extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), m.coef_[0]


b_ll, b_per, b_vec, _ = wf()
print(f"baseline DEV wf LL {b_ll:.5f} (recorded 0.62294)")
assert abs(b_ll - 0.62294) < 5e-4
OUT = {"baseline_dev_ll": b_ll, "results": {}}


def rep(nm, res):
    ll, per, vec, coef = res
    d = b_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    print(f"  {nm:<44} {ll:.5f} gain {b_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10 "
          f"last-fold added coefs {np.round(coef[14:], 4).tolist() if len(coef) > 14 else '-'}", flush=True)
    OUT["results"][nm] = {"ll": round(ll, 6), "gain": round(b_ll - ll, 6), "ci": [round(lo, 6), round(hi, 6)],
                          "seasons_pos": pos, "per_season_gain": {str(s): round(b_per[s] - per[s], 5) for s in per}}


rep("CONTROL drop qb col 1 (quick2: -0.00615)", wf(drop=[1]))
rep("S1 add rec_diff, rush_diff", wf([rec_d, rush_d]))
rep("S2 add rec_diff + rush_diff (one col)", wf(rec_d + rush_d))
rep("S3 add rec_diff", wf(rec_d))
# S4 was added AFTER S1-S3 were seen (the player-level work showed EPA units, v_e, are the better value
# currency for receivers) -> reported as a post-hoc variant, not a pre-declared one
rep("S4 (post-hoc) add rec_e_diff, rush_e_diff (EPA units)", wf([rece_d, rushe_d]))
OUT["bar"] = "NFL pre-registered DEV bar: gain >= +0.00150 with bootstrap 95% CI lower bound > 0"
json.dump(OUT, open(os.path.join(DATA, "pv_nfl_receiving_rushing_screen.json"), "w"), indent=1)
print("wrote data/pv_nfl_receiving_rushing_screen.json")
