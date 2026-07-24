"""Freeze the NHL model: Elo core + rest/B2B/xG blend, plus serving state.

After the one TEST look validated the features (data/nhl_features_report.json),
we refit the blend logistic on ALL available history and snapshot the final
rating state so the live server can predict today's slate by replaying this
season from the checkpoint. Market-blind: no odds anywhere.

Writes data/nhl_model.json:
  elo_cfg, xg_cfg          tuned hyperparameters
  blend                    logistic coefficients [elo_logit, rest, b2b_h, b2b_a, xg]
  elo_ratings             team -> final Elo rating
  xg_ratings              team -> final xG-Elo rating (goal units)
  last_game               team -> last game date (for rest at serve time)
  as_of                   last game date in the training data
  test_ll                 shipped TEST log-loss (frozen headline)
"""
from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import DEV_WARM_BEFORE, llv, load_games, run_elo  # noqa: E402
from nhl_features_eval import build_features  # noqa: E402


def main():
    rep = json.load(open("data/nhl_glicko2_report.json"))
    feat_rep = json.load(open("data/nhl_features_report.json"))
    elo_cfg = rep["elo_cfg"]
    games = load_games()
    e_out = run_elo(games, **elo_cfg)
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    F = build_features(games)

    xgm = ~np.isnan(F["xg_diff"])
    fit = xgm & (seas >= DEV_WARM_BEFORE)        # all warm, xG-covered history
    X = np.column_stack([elogit, F["rest_diff"], F["b2b_home"], F["b2b_away"],
                         np.nan_to_num(F["xg_diff"])])
    clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
    coefs = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])
    print(f"blend refit on {int(fit.sum()):,} games")
    print(f"  intercept {intercept:+.4f}  coefs {[round(c,4) for c in coefs]}")

    # ---- snapshot final rating state for serving ----
    from collections import defaultdict
    from nhl_features_eval import XG_HA, XG_K, XG_REGRESS, TEAM_FIX, load_team_xg
    R = {}
    xg_rating = defaultdict(float)
    last_game = {}
    txg = load_team_xg()
    prev = None
    for g in games:
        d = date.fromisoformat(g["date"])
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - elo_cfg["regress"])
            for t in xg_rating:
                xg_rating[t] *= (1 - XG_REGRESS)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0); ra = R.setdefault(g["away"], 1500.0)
        pe = 1.0 / (1.0 + 10 ** (-((rh + elo_cfg["ha"]) - ra) / 400.0))
        R[g["home"]] += elo_cfg["k"] * (g["y"] - pe)
        R[g["away"]] += elo_cfg["k"] * ((1 - g["y"]) - (1 - pe))
        last_game[g["home"]] = g["date"]; last_game[g["away"]] = g["date"]
        tx = txg.get(g["game_id"])
        if tx and g["home"] in tx and g["away"] in tx:
            hxgf, _ = tx[g["home"]]; axgf, _ = tx[g["away"]]
            pred = xg_rating[g["home"]] - xg_rating[g["away"]] + XG_HA
            err = (hxgf - axgf) - pred
            xg_rating[g["home"]] += XG_K / 100.0 * err
            xg_rating[g["away"]] -= XG_K / 100.0 * err

    model = {
        "elo_cfg": elo_cfg, "glicko_cfg": rep["glicko_cfg"],
        "xg_cfg": {"k": XG_K, "ha": XG_HA, "regress": XG_REGRESS},
        "blend": {"intercept": round(intercept, 5),
                  "coefs": {"elo_logit": round(coefs[0], 5), "rest": round(coefs[1], 5),
                            "b2b_home": round(coefs[2], 5), "b2b_away": round(coefs[3], 5),
                            "xg": round(coefs[4], 5)}},
        "elo_ratings": {t: round(r, 2) for t, r in sorted(R.items(), key=lambda x: -x[1])},
        "xg_ratings": {t: round(v, 4) for t, v in sorted(xg_rating.items(), key=lambda x: -x[1])},
        "last_game": last_game,
        "as_of": games[-1]["date"],
        "baseline_elo_test": rep["elo_test"],
        "baseline_glicko_test": rep["glicko_test"],
        "test_ll": feat_rep["all"]["with"],
        "test_delta_vs_elo": feat_rep["all"]["delta"],
        "test_ci": feat_rep["all"]["ci"],
        "home_win_rate": rep["home_win_rate"],
    }
    json.dump(model, open("data/nhl_model.json", "w"), indent=1)
    print(f"\nwrote data/nhl_model.json (as_of {model['as_of']}, TEST {model['test_ll']})")
    top = list(model["elo_ratings"].items())[:5]
    print("top Elo:", top)
    topx = list(model["xg_ratings"].items())[:5]
    print("top xG:", topx)


if __name__ == "__main__":
    main()
