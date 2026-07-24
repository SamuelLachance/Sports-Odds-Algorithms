"""Goalie-adjusted rating — the hockey analog of MLB's FIP-adjusted pitcher Elo.

The team Glicko-2 rating absorbs the team's USUAL goaltending. But the goalie who
actually starts tonight varies (starter vs backup, injuries, trades), and the
goalie is hockey's single most outcome-determining player. So we build an as-of
goalie quality rating from GSAx (goals saved above expected, from MoneyPuck xG)
and test whether the STARTING-goalie quality differential adds predictive signal
on top of the team Glicko-2 logit.

  goalie value per game   = gsax (xga - actual ga), the fielding-independent
                            save performance; positive = stole shots
  goalie rating (as-of)   = EWMA of per-shot gsax, empirical-Bayes shrunk toward
                            0 (league mean) by a prior of PRIOR_SH shots
  feature                 = home_starter_rating - away_starter_rating
  starter identification  = goalie with the most shots faced on that team/game

Increment test: logistic[glicko_logit, goalie_diff] vs logistic[glicko_logit],
DEV-tuned split, one TEST look. Gate mirrors the MLB feature protocol (-0.0007).
Prior: goaltending is one of the three places NHL gains actually live.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
from glicko2 import Glicko2  # noqa: E402
from nhl_glicko2_eval import (DEV_START, DEV_END, DEV_WARM_BEFORE, TEST_START,  # noqa: E402
                              TEST_END, load_games, llv, run_glicko, run_elo)

# MoneyPuck -> spine abbrev normalization (relocations / code drift)
TEAM_FIX = {"ARI": "ARI", "PHX": "ARI", "ATL": "WPG", "L.A": "LAK", "N.J": "NJD",
            "S.J": "SJS", "T.B": "TBL", "UTA": "UTA"}
PRIOR_SH = 400.0            # EB prior weight in shots
DECAY = 0.997              # per-game EWMA decay of goalie evidence
SEASON_REGRESS_G = 0.20    # goalies regress lightly across seasons


def norm(t):
    return TEAM_FIX.get(t, t)


def load_starters():
    """nhl_game_id -> {team: (goalie_id, gsax, shots)} for the max-shots goalie."""
    per = defaultdict(lambda: defaultdict(lambda: [None, 0.0, 0]))
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        gid = int(r["nhl_game_id"])
        team = norm(r["team"])
        shots = int(round(float(r["xga"]) + 0 if False else 0)) or 0
        # shots faced not stored directly; use ga+ (xga is expected). Use a
        # shots proxy: count via the team-xg table is cleaner. Here rank by |gsax|
        # is wrong; rank by shots faced. Recover shots from goalie rows is absent,
        # so treat each goalie row's shot volume via xga magnitude is unreliable.
        pass
    return per


def load_goalie_games():
    """Return list of (nhl_game_id, team, goalie_id, gsax, shots) using shots
    faced reconstructed from the team-xg table (opponent shots split by goalie
    share of xga is unavailable, so starter = the goalie with the larger xga)."""
    # goalie rows: gid, team, goalie_id, xga, ga, gsax
    rows = defaultdict(list)
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        rows[int(r["nhl_game_id"])].append(
            (norm(r["team"]), int(r["goalie_id"]), float(r["xga"]), float(r["gsax"])))
    return rows


def main():
    rep = json.load(open("data/nhl_glicko2_report.json"))
    cfg = rep["elo_cfg"]                       # Elo is the stronger, adopted core
    print(f"using tuned Elo core: {cfg}", flush=True)

    games = load_games()
    g_out = run_elo(games, **cfg)             # (season, p_home, y) aligned to games
    goalie_rows = load_goalie_games()

    # as-of goalie ratings (EWMA of gsax per starter, EB-shrunk); starter per
    # team = the goalie with the larger xga faced that game.
    grate = {}                                 # goalie_id -> (num, den) EWMA of gsax, shots-weighted by xga
    gseen = {}
    feats = np.full(len(games), np.nan)
    prev_season = None
    for i, g in enumerate(games):
        gid = g["game_id"]
        if prev_season is not None and g["season"] != prev_season:
            for k in grate:
                num, den = grate[k]
                grate[k] = (num * (1 - SEASON_REGRESS_G), den * (1 - SEASON_REGRESS_G))
        prev_season = g["season"]
        gr = goalie_rows.get(gid, [])
        # pick starter per side (max xga faced = most work)
        by_team = defaultdict(list)
        for team, gk, xga, gsax in gr:
            by_team[team].append((xga, gk, gsax))
        h_start = max(by_team.get(g["home"], []), default=None)
        a_start = max(by_team.get(g["away"], []), default=None)

        def rate_of(entry):
            if not entry:
                return 0.0
            _, gk, _ = entry
            num, den = grate.get(gk, (0.0, 0.0))
            return num / (den + PRIOR_SH)      # EB-shrunk gsax-per-shot-ish
        if h_start and a_start:
            feats[i] = rate_of(h_start) - rate_of(a_start)
        # update AFTER reading (leak-safe)
        for team, gk, xga, gsax in gr:
            num, den = grate.get(gk, (0.0, 0.0))
            grate[gk] = (num * DECAY + gsax, den * DECAY + max(xga, 0.1))

    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    glogit = np.log(np.clip(np.array([o[1] for o in g_out]), 1e-9, 1 - 1e-9)
                    / np.clip(1 - np.array([o[1] for o in g_out]), 1e-9, 1 - 1e-9))
    have = ~np.isnan(feats)
    print(f"games with both starters identified: {have.sum():,}/{len(games):,} "
          f"({100*have.mean():.1f}%)", flush=True)

    from sklearn.linear_model import LogisticRegression

    def evalb(mask_lo, mask_hi, cols, warm=None):
        m = have & (seas >= mask_lo) & (seas <= mask_hi)
        if warm:
            m &= seas >= warm
        return m

    # DEV fit of coefficients, TEST score (walk-forward would be ideal; here we
    # fit blend on DEV, score TEST — the glicko logit is already leak-safe as-of)
    dev = evalb(DEV_START, DEV_END, None, warm=DEV_WARM_BEFORE)
    test = evalb(TEST_START, TEST_END, None)

    def fit_score(cols):
        X = np.column_stack(cols)
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X[dev], y[dev])
        p = m.predict_proba(X[test])[:, 1]
        return float(llv(y[test], p).mean()), m

    base_ll, _ = fit_score([glogit])
    add_ll, madd = fit_score([glogit, feats])
    d = llv(y[test], LogisticRegression(C=1e6, max_iter=2000).fit(
        np.column_stack([glogit])[dev], y[dev]).predict_proba(np.column_stack([glogit])[test])[:, 1]) \
        - llv(y[test], madd.predict_proba(np.column_stack([glogit, feats])[test])[:, 1])
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n=== TEST (n={test.sum()}) — goalie GSAx increment ===")
    print(f"  glicko logit only     {base_ll:.5f}")
    print(f"  + goalie GSAx diff     {add_ll:.5f}  (delta {base_ll-add_ll:+.5f}  "
          f"CI [{lo:+.5f},{hi:+.5f}])  coef {madd.coef_[0][1]:+.3f}")
    print(f"  -> {'ADOPT' if add_ll < base_ll - 0.0007 else 'below gate'}")
    json.dump({"base_ll": round(base_ll, 5), "goalie_ll": round(add_ll, 5),
               "delta": round(base_ll - add_ll, 5), "ci": [round(float(lo), 5), round(float(hi), 5)],
               "coef": round(float(madd.coef_[0][1]), 4), "n_test": int(test.sum()),
               "coverage": round(float(have.mean()), 4)},
              open("data/nhl_goalie_report.json", "w"), indent=1)
    print("wrote data/nhl_goalie_report.json")


if __name__ == "__main__":
    main()
