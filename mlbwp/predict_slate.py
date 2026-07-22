"""Build the predictions board data: today through a horizon, all games.

One date-range request to StatsAPI covers the whole window. Team ratings are
brought current from this season's finals; each game is predicted with the
starting pitchers when they are known (<=2 days out) and team-only otherwise,
flagged so the board can show which forecasts include the pitcher edge.

Output: site/data/board.json — leagues (MLB active, others pending) each with a
flat, date-tagged game list the front end filters by date range.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from mlbwp.live import TEAM_ID_TO_RETRO, _get, BASE, schedule, season_finals
from mlbwp.serve import Predictor

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "site" / "data" / "board.json"
FINALS_CACHE = PROJECT / "data" / "season_2026_finals.json"

PENDING_LEAGUES = [
    {"code": "nhl", "name": "NHL"}, {"code": "nba", "name": "NBA"},
    {"code": "nfl", "name": "NFL"}, {"code": "soccer", "name": "Soccer"},
]


def horizon_games(start: date, days: int) -> list[dict]:
    end = start + timedelta(days=days)
    url = (f"{BASE}/schedule?sportId=1&startDate={start}&endDate={end}"
           f"&hydrate=probablePitcher,team")
    d = _get(url)
    out = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") != "R":
                continue
            a, h = g["teams"]["away"], g["teams"]["home"]
            ht, at = h["team"]["id"], a["team"]["id"]
            if ht not in TEAM_ID_TO_RETRO or at not in TEAM_ID_TO_RETRO:
                continue
            out.append({
                "game_pk": g["gamePk"],
                "date": g["gameDate"][:10], "start_utc": g["gameDate"],
                "state": g["status"]["abstractGameState"],
                "home": TEAM_ID_TO_RETRO[ht], "away": TEAM_ID_TO_RETRO[at],
                "home_abbr": h["team"].get("abbreviation", ""),
                "away_abbr": a["team"].get("abbreviation", ""),
                "home_name": h["team"]["name"], "away_name": a["team"]["name"],
                "home_sp": (h.get("probablePitcher") or {}).get("fullName"),
                "away_sp": (a.get("probablePitcher") or {}).get("fullName"),
            })
    return out


def main(days: int = 30, today: str | None = None):
    day0 = date.fromisoformat(today) if today else date.today()
    pred = Predictor()

    finals = json.loads(FINALS_CACHE.read_text()) if FINALS_CACHE.is_file() else \
        season_finals(pred.serve_season, end_date=(day0 - timedelta(days=1)).isoformat())
    applied = pred.bring_current(finals)

    games = horizon_games(day0, days)
    cards = []
    for g in games:
        # Keep Live and Final games: the board is a live product, so today's
        # in-progress and completed games stay on it and the client overlays the
        # score + pick result. The projection stays a pre-game number — team
        # ratings are current only through yesterday, so today's results never
        # leak into the number shown for today's games.
        pitcher_known = bool(g["home_sp"]) and bool(g["away_sp"])
        r = pred.predict(g["home"], g["away"], g.get("home_sp") or "", g.get("away_sp") or "")
        if "error" in r:
            continue
        hp = r["home_win_prob"]
        pick_home = hp >= 0.5
        cards.append({
            "game_pk": g["game_pk"],
            "date": g["date"], "start_utc": g["start_utc"], "state": g["state"],
            "home": g["home"], "away": g["away"],
            "away_abbr": g["away_abbr"], "home_abbr": g["home_abbr"],
            "away_name": g["away_name"], "home_name": g["home_name"],
            "away_sp": g["away_sp"] or "TBD", "home_sp": g["home_sp"] or "TBD",
            "pitcher_known": pitcher_known,
            "home_win_prob": hp,
            "pick": g["home_abbr"] if pick_home else g["away_abbr"],
            "pick_prob": round(max(hp, 1 - hp), 4),
            "edge": r["contributions_pp"],
            "home_pitcher_matched": r["home_pitcher_matched"],
            "away_pitcher_matched": r["away_pitcher_matched"],
        })
    cards.sort(key=lambda c: (c["start_utc"]))

    metrics = json.loads((PROJECT / "mlbwp" / "artifacts" / "metrics.json").read_text())
    payload = {
        "generated": day0.isoformat(),
        "model": pred.model, "version": pred.version,
        "current_through": pred.current_through,
        "season_games_applied": applied,
        "accuracy": {
            "log_loss": metrics["model_log_loss"],
            "elo_log_loss": metrics["plain_elo_log_loss"],
            "coinflip": metrics["constant_log_loss"],
        },
        "leagues": [
            {"code": "mlb", "name": "MLB", "active": True,
             "n_games": len(cards), "games": cards},
            *[{**lg, "active": False} for lg in PENDING_LEAGUES],
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    by_day = {}
    for c in cards:
        by_day[c["date"]] = by_day.get(c["date"], 0) + 1
    print(f"wrote {OUT}: {len(cards)} MLB games over {len(by_day)} days "
          f"(team form through {pred.current_through})")
    for d in sorted(by_day)[:5]:
        print(f"  {d}: {by_day[d]} games")
    return 0


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sys.exit(main(days=d, today=sys.argv[2] if len(sys.argv) > 2 else None))
