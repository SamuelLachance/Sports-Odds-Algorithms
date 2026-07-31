"""Build the MLB database the site reads: teams, players, standings, ratings.

Only what matters for a betting/projection product, with the model's own ratings
woven in:

  teams    — standings (record, run differential, L10, home/road, streak) joined
             to our Elo rating, plus a 0-100 GlassBox team rating (roster average
             of the players' TrueSkill ratings). Pythagorean win% and luck too.
  players  — every rostered player: season line, career totals, and a 0-100
             GlassBox rating from the per-PA TrueSkill mu, normalised within role
             (batter / pitcher) so 50 is league-average and the two roles are
             comparable on one scale.

Sources: MLB Stats API (standings, rosters, bulk season stats, per-player career)
plus the frozen TrueSkill ratings + mlbam->retro crosswalk. Market-free.
"""

from __future__ import annotations

def _write_atomic(path, text):
    """tmp + os.replace: an interrupted run must never truncate a live file."""
    import os
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, str(path))


import json
import math
import time
from pathlib import Path

from mlbwp.live import BASE, TEAM_ID_TO_RETRO, _get
from mlbwp.serve import Predictor

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "site" / "data" / "db.json"
BOARD = PROJECT / "site" / "data" / "board.json"
ART = PROJECT / "mlbwp" / "artifacts"

DIV_NAME = {200: "AL West", 201: "AL East", 202: "AL Central",
            203: "NL West", 204: "NL East", 205: "NL Central"}
POS_GROUP = {"P": "P", "C": "C", "1B": "IF", "2B": "IF", "3B": "IF", "SS": "IF",
             "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH", "TWP": "P"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


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
    return rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)


# ----- standings -> teams -------------------------------------------------
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
                "elo": round(pred.fe.R.get(code, 1500.0), 1),
                "roster": [],
            }
    for i, t in enumerate(sorted(teams.values(), key=lambda x: -x["elo"]), 1):
        teams[t["code"]]["elo_rank"] = i
    for i, t in enumerate(sorted(teams.values(), key=lambda x: -x["pct"]), 1):
        teams[t["code"]]["record_rank"] = i
    return teams


# ----- rosters + player stats + TrueSkill 0-100 ---------------------------
def bulk_stats(season: int, group: str) -> dict:
    d = _get(f"{BASE}/stats?stats=season&group={group}&season={season}"
             f"&sportId=1&playerPool=all&limit=3000")
    out = {}
    for sp in d["stats"][0]["splits"]:
        out[sp["player"]["id"]] = sp["stat"]
    return out


def build_players(teams: dict, season: int, pred=None) -> dict:
    id2team = {t["id"]: code for code, t in teams.items()}
    xwalk = json.loads((ART / "mlbam_to_retro.json").read_text()) if (ART / "mlbam_to_retro.json").is_file() else {}
    prev = json.loads(OUT.read_text()).get("players", {}) if OUT.is_file() else {}
    # current TrueSkill mu straight from the (replayed) engine, so player cards
    # match the ratings the model predicted with.
    def ts_mu(retro):
        return pred.tse.mu.get(retro) if (pred and retro) else None

    hit = bulk_stats(season, "hitting")
    pit = bulk_stats(season, "pitching")

    players = {}
    for tid, code in id2team.items():
        try:
            roster = _get(f"{BASE}/teams/{tid}/roster?rosterType=active&season={season}")["roster"]
        except Exception:  # noqa: BLE001
            continue
        for e in roster:
            pid = e["person"]["id"]
            pos = e["position"]["abbreviation"]
            retro = xwalk.get(str(pid))
            mu = ts_mu(retro)
            is_p = pos in ("P", "TWP")
            rec = {
                "id": pid, "name": e["person"]["fullName"], "team": code,
                "pos": pos, "grp": POS_GROUP.get(pos, "IF"),
                "num": e.get("jerseyNumber", ""), "ts_mu": mu,
                "role": "pitcher" if is_p else "batter",
            }
            hs = hit.get(pid)
            if hs:
                rec["bat"] = {"avg": hs.get("avg"), "obp": hs.get("obp"), "slg": hs.get("slg"),
                              "ops": hs.get("ops"), "hr": hs.get("homeRuns"), "rbi": hs.get("rbi"),
                              "pa": hs.get("plateAppearances"), "sb": hs.get("stolenBases"),
                              "bb": hs.get("baseOnBalls"), "so": hs.get("strikeOuts"),
                              "h": hs.get("hits"), "r": hs.get("runs")}
            ps = pit.get(pid)
            if ps:
                rec["pit"] = {"w": ps.get("wins"), "l": ps.get("losses"), "era": ps.get("era"),
                              "whip": ps.get("whip"), "ip": ps.get("inningsPitched"),
                              "so": ps.get("strikeOuts"), "bb": ps.get("baseOnBalls"),
                              "k9": ps.get("strikeoutsPer9Inn"), "bb9": ps.get("walksPer9Inn"),
                              "hr9": ps.get("homeRunsPer9"), "gs": ps.get("gamesStarted"),
                              "sv": ps.get("saves"), "era_": ps.get("era")}
            # career: reuse prior db.json, else fetch once
            pk = str(pid)
            if pk in prev and prev[pk].get("career"):
                rec["career"] = prev[pk]["career"]
            else:
                rec["career"] = fetch_career(pid, is_p)
                time.sleep(0.05)
            players[pk] = rec

    # TrueSkill -> 0-100 within role (50 = league average roster player)
    for role in ("batter", "pitcher"):
        mus = [p["ts_mu"] for p in players.values() if p["role"] == role and p["ts_mu"] is not None]
        if len(mus) < 5:
            continue
        m = sum(mus) / len(mus)
        sd = (sum((x - m) ** 2 for x in mus) / len(mus)) ** 0.5 or 1.0
        for p in players.values():
            if p["role"] == role and p["ts_mu"] is not None:
                p["ts100"] = round(100 * _phi((p["ts_mu"] - m) / sd), 1)

    # attach roster + team GlassBox rating (0-100)
    by_team = {}
    for p in players.values():
        by_team.setdefault(p["team"], []).append(p)
    for code, ps in by_team.items():
        teams[code]["roster"] = sorted(
            [p["id"] for p in ps],
            key=lambda i: -(next((q for q in ps if q["id"] == i), {}).get("ts100") or 0))
        rated = [p["ts100"] for p in ps if p.get("ts100") is not None]
        off = [p["ts100"] for p in ps if p["role"] == "batter" and p.get("ts100") is not None]
        pitch = [p["ts100"] for p in ps if p["role"] == "pitcher" and p.get("ts100") is not None]
        teams[code]["ts100"] = round(sum(rated) / len(rated), 1) if rated else None
        teams[code]["ts_off"] = round(sum(off) / len(off), 1) if off else None
        teams[code]["ts_pit"] = round(sum(pitch) / len(pitch), 1) if pitch else None
    for i, code in enumerate(sorted(teams, key=lambda c: -(teams[c].get("ts100") or 0)), 1):
        teams[code]["ts_rank"] = i
    return players


def fetch_career(pid: int, is_p: bool) -> dict:
    grp = "pitching" if is_p else "hitting"
    try:
        d = _get(f"{BASE}/people/{pid}/stats?stats=career&group={grp}")
        s = d["stats"][0]["splits"]
        if not s:
            return {}
        c = s[-1]["stat"]
        if is_p:
            return {"w": c.get("wins"), "l": c.get("losses"), "era": c.get("era"),
                    "so": c.get("strikeOuts"), "ip": c.get("inningsPitched"),
                    "whip": c.get("whip"), "gs": c.get("gamesStarted"), "sv": c.get("saves")}
        return {"avg": c.get("avg"), "hr": c.get("homeRuns"), "rbi": c.get("rbi"),
                "h": c.get("hits"), "ops": c.get("ops"), "sb": c.get("stolenBases"),
                "g": c.get("gamesPlayed")}
    except Exception:  # noqa: BLE001
        return {}


def main(season: int = 2026):
    from mlbwp.predict_slate import ensure_lines_cache, build_replay
    pred = Predictor()
    finals = json.loads((PROJECT / "data" / "season_2026_finals.json").read_text()) \
        if (PROJECT / "data" / "season_2026_finals.json").is_file() else []
    # apply the model to this season, same as the board build, so the standings
    # Elo and GlassBox ratings match the numbers the predictions were made with
    cache = ensure_lines_cache(finals)
    games, pas, bullpen, power, baserun = build_replay(finals, cache, pred.mlbam_to_retro, pred.lg_hrfb)
    pred.replay_season(games, pas, bullpen=bullpen, power=power, baserun=baserun)

    teams = build_teams(pred, season)
    players = build_players(teams, season, pred)
    payload = {"season": season, "teams": teams, "players": players,
               "team_order": [t["code"] for t in sorted(teams.values(), key=lambda x: -x["elo"])]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(OUT, json.dumps(payload, indent=1))
    rated = sum(1 for p in players.values() if p.get("ts100") is not None)
    print(f"wrote {OUT}: {len(teams)} teams, {len(players)} players ({rated} TrueSkill-rated)")
    top = sorted(teams.values(), key=lambda t: -(t.get("ts100") or 0))[:4]
    print("  top GlassBox teams:", ", ".join(f"{t['abbr']} {t.get('ts100')}" for t in top))


if __name__ == "__main__":
    main()
