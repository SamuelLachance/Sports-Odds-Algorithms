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

EPS = 1e-9


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def final_states(games, model):
    """Walk Elo + xG-Elo + last-game-date to the end of the spine (mirrors the
    freeze walk) so upcoming games are predicted from CURRENT states."""
    from collections import defaultdict
    from nhl_features_eval import XG_HA, XG_K, XG_REGRESS, load_team_xg
    cfg = model["elo_cfg"]
    R, xg, last = {}, defaultdict(float), {}
    txg = load_team_xg()
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - cfg["regress"])
            for t in xg:
                xg[t] *= (1 - XG_REGRESS)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0); ra = R.setdefault(g["away"], 1500.0)
        pe = 1.0 / (1.0 + 10 ** (-((rh + cfg["ha"]) - ra) / 400.0))
        R[g["home"]] += cfg["k"] * (g["y"] - pe)
        R[g["away"]] += cfg["k"] * ((1 - g["y"]) - (1 - pe))
        last[g["home"]] = g["date"]; last[g["away"]] = g["date"]
        tx = txg.get(g["game_id"])
        if tx and g["home"] in tx and g["away"] in tx:
            hxgf, _ = tx[g["home"]]; axgf, _ = tx[g["away"]]
            err = (hxgf - axgf) - (xg[g["home"]] - xg[g["away"]] + XG_HA)
            xg[g["home"]] += XG_K / 100.0 * err
            xg[g["away"]] -= XG_K / 100.0 * err
    return R, xg, last


def main():
    model = json.load(open("data/nhl_model.json"))
    b = model["blend"]
    c = b["coefs"]
    games = load_games()
    CUR_SEASON = max(g["season"] for g in games)      # auto-rolls to new seasons
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

    # ---- upcoming slate: predict from CURRENT states (in-season only) ----
    import os
    from datetime import date as _date
    n_upcoming = 0
    if os.path.exists("data/nhl_upcoming.json"):
        upcoming = json.load(open("data/nhl_upcoming.json"))
        if upcoming:
            R, xg_st, last = final_states(games, model)
            from nhl_features_eval import XG_HA
            for u in upcoming:
                h, a = u["home"], u["away"]
                if h not in R or a not in R:
                    continue                      # refuse unresolved teams
                elp = 1.0 / (1.0 + 10 ** (-((R[h] + model["elo_cfg"]["ha"]) - R[a]) / 400.0))
                elogit_u = float(np.log(max(elp, EPS) / max(1 - elp, EPS)))
                d = _date.fromisoformat(u["d"])
                def rst(team):
                    lg = last.get(team)
                    return min((d - _date.fromisoformat(lg)).days, 5) if lg else 3
                hb2b = 1.0 if (last.get(h) and (d - _date.fromisoformat(last[h])).days <= 1) else 0.0
                ab2b = 1.0 if (last.get(a) and (d - _date.fromisoformat(last[a])).days <= 1) else 0.0
                xgd = (xg_st.get(h, 0.0) + XG_HA) - xg_st.get(a, 0.0)
                z = (b["intercept"] + c["elo_logit"] * elogit_u + c["rest"] * (rst(h) - rst(a))
                     + c["b2b_home"] * hb2b + c["b2b_away"] * ab2b + c["xg"] * xgd)
                sched.append({"id": u["id"], "d": u["d"], "home": h, "away": a,
                              "hp": round(float(sig(z)), 4), "hs": None, "as": None,
                              "y": None, "ot": None, "playoff": u.get("playoff", 0)})
                n_upcoming += 1
            sched.sort(key=lambda s: (s["d"], s["id"]))

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

    # ---- player ratings (RAPM xG/60, teammate-adjusted; display only) ----
    players = {}
    try:
        rap = json.load(open("data/nhl_rapm2_ratings.json"))
        nm = json.load(open("data/nhl_player_names.json"))
        for pid_, v in rap.items():
            info = nm.get(pid_)
            if not info or v["toi_min"] < 1000:   # established players only (display)
                continue
            players[pid_] = {"name": info["name"], "pos": info["pos"], "team": info["team"],
                             "off": v["off"], "def": v["def"], "net": v["net"],
                             "rating": v["rating"], "toi": v["toi_min"]}
        for t, tm in teams.items():
            roster = [(pid, p) for pid, p in players.items() if p["team"] == t]
            # ids only — the SPA renders from players[pid] (no duplicate dicts)
            tm["top"] = [pid for pid, p in
                         sorted(roster, key=lambda x: -x[1]["net"])[:5]]
    except (FileNotFoundError, json.JSONDecodeError):
        pass    # missing or corrupt ratings/names file: serve without players

    # carry the edge layer's fields forward: odds.yml owns them, but a re-serve
    # must not blank the badges for up to 20 min until its next cycle
    value_updated = None
    try:
        old = json.load(open("site/data/nhl.json", encoding="utf-8"))
        old_val = {g["id"]: g["value"] for g in old.get("schedule", []) if "value" in g}
        for s in sched:
            if s["id"] in old_val:
                s["value"] = old_val[s["id"]]
        value_updated = old.get("value_updated")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    payload = {
        "status": "season" if n_upcoming else "offseason",
        "as_of": model["as_of"],
        "cur_season": CUR_SEASON,
        "schedule": sched,
        "teams": teams,
        "players": players,
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
    if value_updated:
        payload["value_updated"] = value_updated
    json.dump(payload, open("site/data/nhl.json", "w"))
    print(f"wrote site/data/nhl.json: {len(sched)} games, {len(teams)} teams")
    print(f"current-season model LL {cur_ll:.5f}  acc {cur_acc:.4f} (n={len(done)})")
    print("top teams:", [(t, teams[t]["pts"], teams[t]["elo"])
                         for t in list(teams)[:5]])


if __name__ == "__main__":
    main()
