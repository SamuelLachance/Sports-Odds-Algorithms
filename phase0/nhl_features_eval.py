"""NHL Phase-2 feature bake-off over the tuned Elo core (locked split).

The prior said NHL gains live in goaltending, rest, and lineups. We test, as
increments over the Elo team-rating logit, on DEV-fit / TEST-scored:

  rest_diff   home rest-days minus away rest-days (capped)
  b2b_home    home team on the back half of a back-to-back (played yesterday)
  b2b_away    away team on a back-to-back
  xg_diff     as-of team xG rating differential — an Elo-style rating updated by
              EXPECTED goal margin (Poisson-ish), NOT realized goals; the FIP-vs-
              ERA idea for hockey. Prior: expected shot diff predicts (corr .20),
              realized does not (corr -.02) — so update on xG, never on goals.
  goalie_diff starting-goalie GSAx rating differential (from nhl_goalie_eval)

Each feature is added over the Elo logit; combined model at the end. Coefs fit
on DEV (2011-12..2017-18), scored once on TEST (2018-19..2025-26).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import date

import numpy as np

# sklearn is imported lazily, inside the two eval functions that fit models.
# `build_features` — the only thing the SERVE path calls — needs numpy alone, and
# a module-level sklearn import made the whole NHL serve unimportable without it.
# That is why phase0/nhl_serve.py has been skipping on every CI run.

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import (DEV_END, DEV_START, DEV_WARM_BEFORE, TEST_END,  # noqa: E402
                              TEST_START, llv, load_games, run_elo)

XG_K = 6.0            # xG-Elo update rate
XG_HA = 0.15          # home expected-goal edge (goals)
XG_REGRESS = 0.30
TEAM_FIX = {"PHX": "ARI", "ATL": "WPG", "L.A": "LAK", "N.J": "NJD",
            "S.J": "SJS", "T.B": "TBL"}


def load_team_xg():
    """nhl_game_id -> {team: (xgf, xga)}."""
    out = defaultdict(dict)
    for r in csv.DictReader(open("data/nhl_team_xg.csv", encoding="utf-8")):
        t = TEAM_FIX.get(r["team"], r["team"])
        out[int(r["nhl_game_id"])][t] = (float(r["xgf"]), float(r["xga"]))
    return out


def build_features(games):
    n = len(games)
    rest_diff = np.zeros(n)
    b2b_home = np.zeros(n)
    b2b_away = np.zeros(n)
    xg_diff = np.full(n, np.nan)

    last_game = {}                       # team -> date of previous game
    xg_rating = defaultdict(lambda: 0.0)  # team xG-Elo (in goal units, ~0 center)
    txg = load_team_xg()
    prev_season = None

    for i, g in enumerate(games):
        d = date.fromisoformat(g["date"])
        if prev_season is not None and g["season"] != prev_season:
            for t in xg_rating:
                xg_rating[t] *= (1 - XG_REGRESS)
        prev_season = g["season"]
        # rest / b2b
        for team, arr, is_home in ((g["home"], b2b_home, True), (g["away"], b2b_away, False)):
            lg = last_game.get(team)
            rest = (d - lg).days if lg else 3
            rest = min(rest, 5)
            if is_home:
                h_rest = rest
            else:
                a_rest = rest
            if lg and (d - lg).days <= 1:
                arr[i] = 1.0
        rest_diff[i] = h_rest - a_rest
        # xG rating (pre-game, leak-safe)
        xg_diff[i] = (xg_rating[g["home"]] + XG_HA) - xg_rating[g["away"]]
        # update states AFTER reading
        last_game[g["home"]] = d
        last_game[g["away"]] = d
        tx = txg.get(g["game_id"])
        if tx and g["home"] in tx and g["away"] in tx:
            hxgf, hxga = tx[g["home"]]
            axgf, axga = tx[g["away"]]
            # expected margin from THIS game's xG (home perspective)
            exp_margin = (hxgf - hxga) - 0.0    # home net xG
            pred = xg_rating[g["home"]] - xg_rating[g["away"]] + XG_HA
            err = (hxgf - axgf) - pred          # home xG-diff vs predicted
            xg_rating[g["home"]] += XG_K / 100.0 * err
            xg_rating[g["away"]] -= XG_K / 100.0 * err
        else:
            xg_diff[i] = np.nan                  # no xG this game -> feature missing
    return dict(rest_diff=rest_diff, b2b_home=b2b_home, b2b_away=b2b_away, xg_diff=xg_diff)


def main():
    rep = json.load(open("data/nhl_glicko2_report.json"))
    games = load_games()
    e_out = run_elo(games, **rep["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])

    F = build_features(games)
    dev = (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    test = (seas >= TEST_START) & (seas <= TEST_END)

    from sklearn.linear_model import LogisticRegression   # eval-only dependency

    def run(cols, name, mask_extra=None):
        m_dev, m_test = dev.copy(), test.copy()
        if mask_extra is not None:
            m_dev &= mask_extra; m_test &= mask_extra
        X = np.column_stack([elogit] + cols)
        clf = LogisticRegression(C=1e6, max_iter=2000).fit(X[m_dev], y[m_dev])
        pb = clf.predict_proba(X[m_test])[:, 1]
        base = LogisticRegression(C=1e6, max_iter=2000).fit(
            elogit[m_dev].reshape(-1, 1), y[m_dev])
        p0 = base.predict_proba(elogit[m_test].reshape(-1, 1))[:, 1]
        b_ll = float(llv(y[m_test], p0).mean())
        f_ll = float(llv(y[m_test], pb).mean())
        d = llv(y[m_test], p0) - llv(y[m_test], pb)
        rng = np.random.default_rng(7)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        coefs = [round(float(c), 3) for c in clf.coef_[0][1:]]
        print(f"  {name:<28} n={int(m_test.sum()):<5} base {b_ll:.5f} -> {f_ll:.5f}  "
              f"delta {b_ll-f_ll:+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  coefs {coefs}", flush=True)
        return {"name": name, "n": int(m_test.sum()), "base": round(b_ll, 5),
                "with": round(f_ll, 5), "delta": round(b_ll - f_ll, 5),
                "ci": [round(float(lo), 5), round(float(hi), 5)], "sig": sig, "coefs": coefs}

    print("\nNHL Phase-2 feature increments over Elo core (TEST 2018-2026):", flush=True)
    res = {}
    res["rest"] = run([F["rest_diff"]], "rest_diff")
    res["b2b"] = run([F["b2b_home"], F["b2b_away"]], "b2b (home,away)")
    res["rest_b2b"] = run([F["rest_diff"], F["b2b_home"], F["b2b_away"]], "rest + b2b")
    xgm = ~np.isnan(F["xg_diff"])
    res["xg"] = run([np.nan_to_num(F["xg_diff"])], "xg_diff", mask_extra=xgm)
    res["all"] = run([F["rest_diff"], F["b2b_home"], F["b2b_away"],
                      np.nan_to_num(F["xg_diff"])], "rest+b2b+xg", mask_extra=xgm)
    json.dump(res, open("data/nhl_features_report.json", "w"), indent=1)
    print("\nwrote data/nhl_features_report.json")


if __name__ == "__main__":
    main()
