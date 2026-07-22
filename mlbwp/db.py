"""Build the MLB database the site reads: teams, pitchers, standings.

Deliberately narrow. ESPN shows everything; GLASSBOX shows only what moves a
betting decision, with the model's own ratings woven in:

  teams    — standings (record, run differential, L10, home/road, streak) joined
             to our Elo rating and rank. The gap between our rating rank and the
             standings rank is a regression signal; pythagorean win% vs actual is
             a luck signal.
  pitchers — season line (ERA, WHIP, K/9, BB/9, HR/9) for every probable starter
             on the board, joined to our FIP adjustment and its rank.

Sources: MLB Stats API /standings (one call) and /people/{id}/stats (per pitcher).
Market-free — no odds are fetched or stored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mlbwp.live import BASE, TEAM_ID_TO_RETRO, _get
from mlbwp.serve import Predictor, norm_name

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "site" / "data" / "db.json"
BOARD = PROJECT / "site" / "data" / "board.json"

DIV_NAME = {200: "AL West", 201: "AL East", 202: "AL Central",
            203: "NL West", 204: "NL East", 205: "NL Central"}


def _split(records, kind):
    for r in records.get("splitRecords", []):
        if r["type"] == kind:
            return f"{r['wins']}-{r['losses']}"
    return "-"


def _l10(records):
    for r in records.get("splitRecords", []):
        if r["type"] == "lastTen":
            return r["wins"], r["losses"]
    return 0, 0


def pythag(rs, ra):
    if rs <= 0 and ra <= 0:
        return 0.5
    e = 1.83
    return rs ** e / (rs ** e + ra ** e)


def build_teams(pred: Predictor, season: int) -> dict:
    s = _get(f"{BASE}/standings?leagueId=103,104&season={season}"
             f"&standingsTypes=regularSeason&hydrate=team")
    teams = {}
    for grp in s["records"]:
        div = grp.get("division", {}).get("id")
        for tr in grp["teamRecords"]:
            tid = tr["team"]["id"]
            code = TEAM_ID_TO_RETRO.get(tid)
            if not code:
                continue
            rs, ra = tr["runsScored"], tr["runsAllowed"]
            l10w, l10l = _l10(tr.get("records", {}))
            gp = tr["gamesPlayed"]
            py = pythag(rs, ra)
            teams[code] = {
                "code": code, "id": tid, "name": tr["team"]["name"],
                "abbr": tr["team"].get("abbreviation", code),
                "div": DIV_NAME.get(div, ""), "div_rank": int(tr["divisionRank"]),
                "w": tr["wins"], "l": tr["losses"],
                "pct": float(tr["winningPercentage"]), "gb": tr["gamesBack"],
                "rs": rs, "ra": ra, "run_diff": tr["runDifferential"],
                "rs_g": round(rs / gp, 2) if gp else 0, "ra_g": round(ra / gp, 2) if gp else 0,
                "streak": tr.get("streak", {}).get("streakCode", "-"),
                "l10": f"{l10w}-{l10l}",
                "home": _split(tr.get("records", {}), "home"),
                "away": _split(tr.get("records", {}), "away"),
                "pythag": round(py, 3),
                "luck": round(tr["wins"] / gp - py, 3) if gp else 0.0,
                "elo": round(pred.R.get(code, 1500.0), 1),
            }
    # our Elo rank (1 = best)
    order = sorted(teams.values(), key=lambda t: -t["elo"])
    for i, t in enumerate(order, 1):
        teams[t["code"]]["elo_rank"] = i
    # actual rank by win pct, for the regression/overperformance signal
    for i, t in enumerate(sorted(teams.values(), key=lambda x: -x["pct"]), 1):
        teams[t["code"]]["record_rank"] = i
    return teams


def build_pitchers(pred: Predictor, season: int) -> dict:
    # id lookup for every player this season (one call)
    players = {p["fullName"]: p["id"]
               for p in _get(f"{BASE}/sports/1/players?season={season}")["people"]}

    # which pitchers appear on the board
    board = json.loads(BOARD.read_text(encoding="utf-8"))
    names = set()
    for g in board["leagues"][0]["games"]:
        for sp in (g["home_sp"], g["away_sp"]):
            if sp and sp != "TBD":
                names.add(sp)

    out = {}
    for name in sorted(names):
        adj, rec = pred.pitcher_adj(name)
        pid = players.get(name)
        line = {}
        if pid:
            try:
                st = _get(f"{BASE}/people/{pid}/stats?stats=season&group=pitching&season={season}")
                sp = st["stats"][0]["splits"]
                if sp:
                    s0 = sp[0]["stat"]
                    line = {
                        "gs": s0.get("gamesStarted"), "ip": s0.get("inningsPitched"),
                        "era": s0.get("era"), "whip": s0.get("whip"),
                        "k9": s0.get("strikeoutsPer9Inn"), "bb9": s0.get("walksPer9Inn"),
                        "hr9": s0.get("homeRunsPer9"), "kbb": s0.get("strikeoutWalkRatio"),
                        "w": s0.get("wins"), "l": s0.get("losses"),
                        "k": s0.get("strikeOuts"), "bb": s0.get("baseOnBalls"),
                    }
                time.sleep(0.12)
            except Exception:  # noqa: BLE001
                pass
        out[norm_name(name)] = {
            "name": name, "mlbam": pid, "adj": adj,
            "matched": rec is not None, "career_ip": rec["ip"] if rec else 0.0,
            **line,
        }
    # our FIP-adjustment rank among matched starters
    ranked = sorted([p for p in out.values() if p["matched"]], key=lambda p: -p["adj"])
    for i, p in enumerate(ranked, 1):
        out[norm_name(p["name"])]["adj_rank"] = i
    out["__n_ranked"] = len(ranked)
    return out


def main(season: int = 2026):
    pred = Predictor()
    finals = json.loads((PROJECT / "data" / "season_2026_finals.json").read_text()) \
        if (PROJECT / "data" / "season_2026_finals.json").is_file() else []
    pred.bring_current(finals)

    teams = build_teams(pred, season)
    pitchers = build_pitchers(pred, season)
    payload = {"season": season, "teams": teams, "pitchers": pitchers,
               "team_order": [t["code"] for t in sorted(teams.values(), key=lambda x: -x["elo"])]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {len(teams)} teams, {pitchers['__n_ranked']} ranked pitchers "
          f"({len(pitchers)-1} on the board)")
    top = sorted(teams.values(), key=lambda t: -t["elo"])[:4]
    print("  our top teams:", ", ".join(f"{t['abbr']} {t['elo']:.0f}(#{t['record_rank']} rec)" for t in top))


if __name__ == "__main__":
    main()
