"""Serve the NHL model into site/data/nhl.json.

Replays the frozen Elo + xG + rest/B2B model through all history (leak-safe:
each game predicted from pre-game state), then emits the current-season
schedule with per-game model probabilities and results, plus team ratings and
standings. Market-blind — no odds in the payload (the edge layer adds those
separately, mirroring nfl.json).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import llv, load_games, run_elo  # noqa: E402
from nhl_features_eval import build_features  # noqa: E402

CUR_SEASON = 20252026
EPS = 1e-9


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    model = json.load(open("data/nhl_model.json"))
    b = model["blend"]
    c = b["coefs"]
    games = load_games()
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), EPS, 1 - EPS)
    elogit = np.log(p / (1 - p))
    F = build_features(games)

    z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
         + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
         + c["xg"] * np.nan_to_num(F["xg_diff"]))
    phome = sig(z)

    # ---- current-season schedule with predictions + results ----
    sched = []
    for i, g in enumerate(games):
        if g["season"] != CUR_SEASON:
            continue
        done = True  # spine only has completed games
        sched.append({
            "id": g["game_id"], "d": g["date"], "home": g["home"], "away": g["away"],
            "hp": round(float(phome[i]), 4),
            "hs": g["home_goals"] if "home_goals" in g else None,
            "as": g["away_goals"] if "away_goals" in g else None,
            "y": g["y"], "ot": g["so"] or None,
            "playoff": 1 if str(g["game_id"])[4:6] == "03" else 0,
        })

    # spine dicts lack goals; reload raw for scores
    import csv
    raw = {int(r["game_id"]): r for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8"))}
    for s in sched:
        r = raw.get(s["id"])
        if r:
            s["hs"] = int(r["home_goals"]); s["as"] = int(r["away_goals"])
            s["last"] = r["last_period"]

    # ---- team ratings + standings (current season record) ----
    rec = defaultdict(lambda: {"w": 0, "l": 0, "otl": 0, "gf": 0, "ga": 0})
    for s in sched:
        if s["hs"] is None:
            continue
        h, a = s["home"], s["away"]
        rec[h]["gf"] += s["hs"]; rec[h]["ga"] += s["as"]
        rec[a]["gf"] += s["as"]; rec[a]["ga"] += s["hs"]
        ot = (r := raw.get(s["id"])) and r["last_period"] in ("OT", "SO")
        if s["y"] == 1:
            rec[h]["w"] += 1
            rec[a]["otl" if ot else "l"] += 1
        else:
            rec[a]["w"] += 1
            rec[h]["otl" if ot else "l"] += 1

    elo_r = model["elo_ratings"]; xg_r = model["xg_ratings"]
    # current teams = those that played this season (drops defunct ATL/PHX/ARI codes)
    current = {t for t in rec}
    cur_elo = {t: elo_r[t] for t in elo_r if t in current}
    ranks = {t: i + 1 for i, t in enumerate(sorted(cur_elo, key=lambda x: -cur_elo[x]))}
    teams = {}
    for t in cur_elo:
        r = rec.get(t, {})
        teams[t] = {
            "elo": elo_r[t], "xg": xg_r.get(t, 0.0), "rank": ranks[t],
            "w": r.get("w", 0), "l": r.get("l", 0), "otl": r.get("otl", 0),
            "gf": r.get("gf", 0), "ga": r.get("ga", 0),
            "pts": r.get("w", 0) * 2 + r.get("otl", 0),
        }

    # ---- realized current-season accuracy / LL ----
    done = [s for s in sched if s["hs"] is not None and not s["playoff"]]
    y = np.array([s["y"] for s in done])
    pp = np.array([s["hp"] for s in done])
    cur_ll = float(llv(y, np.clip(pp, EPS, 1 - EPS)).mean()) if len(y) else None
    cur_acc = float(((pp > 0.5) == (y > 0.5)).mean()) if len(y) else None

    payload = {
        "status": "offseason" if True else "season",
        "as_of": model["as_of"],
        "cur_season": CUR_SEASON,
        "schedule": sched,
        "teams": teams,
        "model_card": {
            "test_ll": model["test_ll"],
            "test_delta_vs_elo": model["test_delta_vs_elo"],
            "baseline_elo_test": model["baseline_elo_test"],
            "home_win_rate": model["home_win_rate"],
            "cur_season_ll": round(cur_ll, 5) if cur_ll else None,
            "cur_season_acc": round(cur_acc, 4) if cur_acc else None,
            "features": ["Elo (k8,ha30)", "rest", "back-to-back", "xG team rating"],
            "n_games_train": 17709,
        },
    }
    json.dump(payload, open("site/data/nhl.json", "w"))
    print(f"wrote site/data/nhl.json: {len(sched)} games, {len(teams)} teams")
    print(f"current-season model LL {cur_ll:.5f}  acc {cur_acc:.4f} (n={len(done)})")
    print("top teams:", [(t, teams[t]["pts"], teams[t]["elo"])
                         for t in list(teams)[:5]])


if __name__ == "__main__":
    main()
