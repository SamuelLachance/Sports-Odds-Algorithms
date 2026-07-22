"""Produce today's predictions JSON for the site.

Loads the frozen model, brings team ratings current from this season's finals,
fetches today's slate + probable pitchers, and writes site/data/predictions.json
with a per-game factor decomposition and an honest "what we don't know" block.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from mlbwp.live import schedule, season_finals
from mlbwp.serve import Predictor

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "site" / "data" / "predictions.json"
FINALS_CACHE = PROJECT / "data" / "season_2026_finals.json"


def unknowns(g: dict, pred: dict) -> list[str]:
    out = []
    if not pred.get("home_pitcher_matched") or not pred.get("away_pitcher_matched"):
        who = []
        if not pred.get("home_pitcher_matched"):
            who.append(g.get("home_sp") or "home TBD")
        if not pred.get("away_pitcher_matched"):
            who.append(g.get("away_sp") or "away TBD")
        out.append(f"No rating for {', '.join(who)} — treated as a league-average starter.")
    for side in ("home", "away"):
        ip = pred.get(f"{side}_pitcher_ip", 0)
        if pred.get(f"{side}_pitcher_matched") and ip < 150:
            out.append(f"{g[side+'_sp']} has a short track record ({ip:.0f} IP); "
                       f"his rating leans on the league prior.")
    out.append("Bullpen availability and today's lineup are not model inputs.")
    return out


def main(day: str | None = None):
    day = day or date.today().isoformat()
    pred = Predictor()

    finals = json.loads(FINALS_CACHE.read_text()) if FINALS_CACHE.is_file() else \
        season_finals(pred.serve_season, end_date=(date.fromisoformat(day) - timedelta(days=1)).isoformat())
    n = pred.bring_current(finals)

    games = [g for g in schedule(day) if g["state"] != "Postponed"]
    cards = []
    for g in games:
        r = pred.predict(g["home"], g["away"], g.get("home_sp") or "", g.get("away_sp") or "")
        if "error" in r:
            continue
        cards.append({
            "game_pk": g["game_pk"], "state": g["state"], "start_utc": g["start_utc"],
            "home": g["home"], "away": g["away"],
            "home_name": g["home_name"], "away_name": g["away_name"],
            "home_abbr": g["home_abbr"], "away_abbr": g["away_abbr"],
            "home_sp": g.get("home_sp") or "TBD", "away_sp": g.get("away_sp") or "TBD",
            **r,
            "unknowns": unknowns(g, r),
        })
    cards.sort(key=lambda c: -abs(c["home_win_prob"] - 0.5))

    payload = {
        "date": day,
        "model": pred.model, "version": pred.version,
        "ratings_trained_through": pred.trained_through,
        "team_form_current_through": pred.current_through,
        "season_games_applied": n,
        "n_games": len(cards),
        "games": cards,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {len(cards)} games for {day} "
          f"(team form through {pred.current_through}, {n} season games applied)")
    for c in cards[:6]:
        print(f"  {c['away']}({c['away_sp'][:14]:14s}) @ {c['home']}({c['home_sp'][:14]:14s}) "
              f"-> home {c['home_win_prob']*100:4.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
