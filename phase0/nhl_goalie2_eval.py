"""Starting-goalie DELTA feature — the live-goalie adjustment the model lacks.

The first goalie test (nhl_goalie_eval.py) added the starter's absolute GSAx
differential over Elo and came back n.s. — because the team rating already
prices a team's USUAL goalie, so absolute goalie quality is redundant with team
strength. The signal the team rating CANNOT know pre-game is *who is in net
tonight vs who usually is*: when the backup starts (rest, injury, load mgmt) the
team is weaker that night.

So the feature is the DELTA, not the level:

  goalie_delta = (home_starter_rtg - home_team_baseline)
               - (away_starter_rtg - away_team_baseline)

goalie rtg   = as-of EWMA of GSAx-per-expected-goal (xga - ga), EB-shrunk
team_baseline= as-of EWMA over recent games of that team's STARTER rating
               (tracks "the goaltending this team has been getting lately")

Starter identification: the goalie who faced the most shots that game (a pre-game
proxy). All ratings read PRE-game (leak-safe). Tested as an increment over the
shipped Elo+rest+b2b+xG blend on the locked split. If it pays, it ships and the
serve layer uses the ANNOUNCED probable goalie.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import (DEV_END, DEV_WARM_BEFORE, TEST_END, TEST_START,  # noqa: E402
                              llv, load_games, run_elo)
from nhl_features_eval import build_features  # noqa: E402

G_DECAY = 0.96         # per-game EWMA decay of goalie GSAx evidence
G_PRIOR = 15.0         # EB prior weight in GAMES (rating = avg GSAx per game)
G_SEASON = 0.25        # goalie rating season regression
TEAM_DECAY = 0.80      # team-baseline EWMA over recent starters
TEAM_FIX = {"PHX": "ARI", "ATL": "WPG", "L.A": "LAK", "N.J": "NJD",
            "S.J": "SJS", "T.B": "TBL"}


def norm(t):
    return TEAM_FIX.get(t, t)


def main():
    model = json.load(open("data/nhl_model.json"))
    games = load_games()
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    b = model["blend"]; c = b["coefs"]
    base_z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
              + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
              + c["xg"] * np.nan_to_num(F["xg_diff"]))

    # goalie-game rows: gid -> list of (team, goalie_id, xga, gsax)
    grows = defaultdict(list)
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        grows[int(r["nhl_game_id"])].append(
            (norm(r["team"]), int(r["goalie_id"]), float(r["xga"]), float(r["gsax"])))

    # as-of walk: goalie ratings (num/den EWMA of gsax over xga) + team baselines
    grate = {}                       # gk -> (num, den)
    team_base = {}                   # team -> EWMA of starter ratings
    delta = np.full(len(games), np.nan)
    absdiff = np.full(len(games), np.nan)
    prev_season = None
    for i, g in enumerate(games):
        if prev_season is not None and g["season"] != prev_season:
            for k in grate:
                n_, d_ = grate[k]
                grate[k] = (n_ * (1 - G_SEASON), d_ * (1 - G_SEASON))
        prev_season = g["season"]
        gr = grows.get(g["game_id"], [])
        by_team = defaultdict(list)
        for team, gk, xga, gsax in gr:
            by_team[team].append((xga, gk, gsax))

        def starter_rtg(team):
            lst = by_team.get(team)
            if not lst:
                return None, None
            xga, gk, gsax = max(lst)          # most work = starter
            n_, d_ = grate.get(gk, (0.0, 0.0))
            return n_ / (d_ + G_PRIOR), gk     # avg GSAx per game (goals units), EB-shrunk

        hr, hk = starter_rtg(g["home"])
        ar, ak = starter_rtg(g["away"])
        if hr is not None and ar is not None:
            hb = team_base.get(g["home"], 0.0)
            ab = team_base.get(g["away"], 0.0)
            delta[i] = (hr - hb) - (ar - ab)
            absdiff[i] = hr - ar
        # update AFTER reading (leak-safe): goalie evidence + team baseline
        for team, gk, xga, gsax in gr:
            n_, d_ = grate.get(gk, (0.0, 0.0))
            grate[gk] = (n_ * G_DECAY + gsax, d_ * G_DECAY + 1.0)   # per-game count
        for team, r_ in ((g["home"], hr), (g["away"], ar)):
            if r_ is not None:
                team_base[team] = TEAM_DECAY * team_base.get(team, r_) + (1 - TEAM_DECAY) * r_

    have = ~np.isnan(delta)
    dev = have & (seas >= DEV_WARM_BEFORE) & (seas <= DEV_END)
    test = have & (seas >= TEST_START) & (seas <= TEST_END)
    print(f"coverage {have.mean():.1%}  TEST n={int(test.sum())}", flush=True)
    print(f"delta feature: std {np.nanstd(delta):.4f}, "
          f"share |delta|>0.05: {np.mean(np.abs(delta[have])>0.05):.1%} "
          f"(backup-start nights)", flush=True)

    def run(name, feat):
        Xb = base_z.reshape(-1, 1)
        X = np.column_stack([base_z, np.nan_to_num(feat)])
        mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[dev], y[dev])
        ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[dev], y[dev])
        p0 = mb.predict_proba(Xb[test])[:, 1]; p1 = ms.predict_proba(X[test])[:, 1]
        b_ll = float(llv(y[test], p0).mean()); s_ll = float(llv(y[test], p1).mean())
        d = llv(y[test], p0) - llv(y[test], p1)
        rng = np.random.default_rng(7)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<26} base {b_ll:.5f} -> {s_ll:.5f}  delta {b_ll-s_ll:+.5f} "
              f"CI [{lo:+.5f},{hi:+.5f}] {sig}  coef {ms.coef_[0][1]:+.3f}", flush=True)
        return {"base": round(b_ll, 5), "with": round(s_ll, 5), "delta": round(b_ll - s_ll, 5),
                "ci": [round(float(lo), 5), round(float(hi), 5)], "sig": sig,
                "coef": round(float(ms.coef_[0][1]), 4)}

    print("\nstarting-goalie increment over shipped model (TEST):", flush=True)
    res = {"n_test": int(test.sum()), "coverage": round(float(have.mean()), 4)}
    res["delta"] = run("goalie DELTA (vs baseline)", delta)
    res["absdiff"] = run("goalie ABS diff (control)", absdiff)
    json.dump(res, open("data/nhl_goalie2_report.json", "w"), indent=1)
    print("\nwrote data/nhl_goalie2_report.json")


if __name__ == "__main__":
    main()
