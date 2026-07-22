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
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from mlbwp.live import TEAM_ID_TO_RETRO, _get, BASE, et_date, game_data, schedule, season_finals
from mlbwp.projection import Projector
from mlbwp.serve import Predictor

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "site" / "data" / "board.json"
FINALS_CACHE = PROJECT / "data" / "season_2026_finals.json"
LINES_CACHE = PROJECT / "data" / "season_2026_lines.json"


def ensure_lines_cache(finals: list[dict], cache_path: Path = LINES_CACHE) -> dict:
    """Incrementally fetch each completed game's box lines + plate appearances
    (one feed/live fetch per game), keyed by game_pk. Only games not already
    cached are fetched, so the first build backfills the season and every later
    build fetches only newly-final games. This is the season's raw data the model
    is applied to — never re-fetched once cached.
    """
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    fetched = 0
    for g in finals:
        pk = str(g["game_pk"])
        if pk in cache:
            continue
        try:
            gd = game_data(g["game_pk"])
            cache[pk] = {"date": g["date"], "home": gd["home"], "away": gd["away"],
                         "pa": gd["pa"], "baserun": gd["baserun"]}
            fetched += 1
            time.sleep(0.15)
            if fetched % 200 == 0:      # checkpoint a long backfill so it can't lose it all
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:  # noqa: BLE001 — a bad game feed must not abort the build
            continue
    if fetched:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    print(f"[lines] cache {len(cache)} games (+{fetched} fetched)", flush=True)
    return cache


def build_replay(finals: list[dict], cache: dict, x: dict, lg_hrfb=None):
    """Turn this season's cached data into the inputs the engines replay, using the
    mlbam->retro crosswalk x. Starter lines are transformed to xFIP (actual HR ->
    expected HR from fly balls) with the frozen league HR/FB, matching training.
    Returns:
      games   [ {home,away,home_sp,away_sp(retro),y,home_line,away_line,date} ]  for FipPitcherElo.update
      pas     [ (batter_retro, pitcher_retro, on_base) ]  date-ordered  for TrueSkill.update
      bullpen {team: (fip_num, outs)}      season-to-date reliever aggregate
      power   {retro: [PA, H, TB]}         season-to-date batting counts
      baserun {retro: [PA,SB,CS,XB,OOB,GIDP]}  season-to-date baserunning counts
    """
    from mlbwp.ingest import to_xfip
    games, pas = [], []
    bullpen = defaultdict(lambda: [0.0, 0])
    power = defaultdict(lambda: [0, 0, 0])
    baserun = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    have = [g for g in finals if str(g["game_pk"]) in cache]
    for g in sorted(have, key=lambda g: (g["date"], g["game_pk"])):
        rec = cache[str(g["game_pk"])]
        for mlbam, c in (rec.get("baserun") or {}).items():
            r = x.get(str(mlbam))
            if r:
                for k in range(6):
                    baserun[r][k] += c[k]
        sides = {}
        for side, team in (("home", g["home"]), ("away", g["away"])):
            s = rec[side]
            bullpen[team][0] += s["rel"][0]
            bullpen[team][1] += s["rel"][1]
            for mlbam, line in (s.get("bat") or {}).items():
                r = x.get(str(mlbam))
                if r:
                    power[r][0] += line[0]; power[r][1] += line[1]; power[r][2] += line[2]
            sp_retro = x.get(str(s.get("sp_id")))
            sp = s.get("sp") or [0, 0, 0, 0, 0]
            line = sline = None
            if sp_retro and sp[0] > 0:
                line = to_xfip(sp, s.get("sp_fb", 0), lg_hrfb) if lg_hrfb else sp
                # SIERA line: SO, BB, GB, FB, PU, PA(battersFaced)
                sline = [sp[4], sp[2], s.get("sp_gb", 0), s.get("sp_fb", 0),
                         s.get("sp_pu", 0), s.get("sp_bf", 0)]
            sides[side] = (sp_retro or "", line, sline)
        games.append({
            "home": g["home"], "away": g["away"],
            "home_sp": sides["home"][0], "away_sp": sides["away"][0],
            "home_line": sides["home"][1], "away_line": sides["away"][1],
            "home_sline": sides["home"][2], "away_sline": sides["away"][2],
            "y": g["home_win"], "date": g["date"],
        })
        for b, p, ob in rec.get("pa", []):
            br, pr = x.get(str(b)), x.get(str(p))
            if br and pr:
                pas.append((br, pr, ob))
    return (games, pas,
            {t: (v[0], v[1]) for t, v in bullpen.items()},
            {r: v for r, v in power.items()},
            {r: v for r, v in baserun.items()})

PENDING_LEAGUES = [
    {"code": "nhl", "name": "NHL"}, {"code": "nba", "name": "NBA"},
    {"code": "nfl", "name": "NFL"}, {"code": "soccer", "name": "Soccer"},
]


def horizon_games(start: date, days: int) -> list[dict]:
    end = start + timedelta(days=days)
    url = (f"{BASE}/schedule?sportId=1&startDate={start}&endDate={end}"
           f"&hydrate=probablePitcher,team,lineups")
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
            lu = g.get("lineups") or {}
            out.append({
                "game_pk": g["gamePk"],
                "date": et_date(g["gameDate"]), "start_utc": g["gameDate"],
                "state": g["status"]["abstractGameState"],
                "home": TEAM_ID_TO_RETRO[ht], "away": TEAM_ID_TO_RETRO[at],
                "home_abbr": h["team"].get("abbreviation", ""),
                "away_abbr": a["team"].get("abbreviation", ""),
                "home_name": h["team"]["name"], "away_name": a["team"]["name"],
                "home_sp": (h.get("probablePitcher") or {}).get("fullName"),
                "away_sp": (a.get("probablePitcher") or {}).get("fullName"),
                "home_lineup": [p["id"] for p in lu.get("homePlayers", [])],
                "away_lineup": [p["id"] for p in lu.get("awayPlayers", [])],
            })
    return out


def main(days: int = 30, today: str | None = None):
    day0 = date.fromisoformat(today) if today else date.today()
    pred = Predictor()

    finals = json.loads(FINALS_CACHE.read_text()) if FINALS_CACHE.is_file() else \
        season_finals(pred.serve_season, end_date=(day0 - timedelta(days=1)).isoformat())
    # Apply the frozen model to this season: fetch the raw data, then replay it
    # through the same engines the model was trained with. No retraining — the
    # coefficients are fixed; only the rating STATE is walked to today.
    cache = ensure_lines_cache(finals)
    games, pas, bullpen, power, baserun = build_replay(finals, cache, pred.mlbam_to_retro, pred.lg_hrfb)
    applied = pred.replay_season(games, pas, bullpen=bullpen, power=power, baserun=baserun)
    bp_loaded = sum(1 for v in bullpen.values() if v[1] > 0)
    sp_loaded = len(games)
    bat_loaded = sum(1 for v in power.values() if v[0] > 0)

    games = horizon_games(day0, days)
    projector = Projector(day0)
    print(f"[proj] injury-aware projector: rosters for {len(projector.active)} teams", flush=True)
    cards = []
    for g in games:
        # Keep Live and Final games: the board is a live product, so today's
        # in-progress and completed games stay on it and the client overlays the
        # score + pick result. The projection stays a pre-game number — team
        # ratings are current only through yesterday, so today's results never
        # leak into the number shown for today's games.
        # Starter: the official probable only (MLB sets it ~2 days ahead). Guard
        # against a stub/blank name so an unknown starter can never read as "known".
        def _sp(x):
            x = (x or "").strip()
            return x if x and x.upper() != "TBD" else None
        h_sp, a_sp = _sp(g["home_sp"]), _sp(g["away_sp"])
        pitcher_known = bool(h_sp) and bool(a_sp)

        # Lineup: the full tier needs both starters known (never a full pick next to a
        # TBD starter). Prefer the OFFICIAL lineup; else an injury-aware PROJECTION
        # (most-frequent recent starters still on the active roster -- IL'd players
        # excluded). Both upgrade to the official lineup automatically once it posts.
        oh, oa = g.get("home_lineup") or [], g.get("away_lineup") or []
        home_lineup = away_lineup = None
        lineup_source = None
        if pitcher_known:
            if len(oh) >= 5 and len(oa) >= 5:
                home_lineup, away_lineup, lineup_source = oh, oa, "official"
            else:
                pb, pa_ = projector.batters(g["home"]), projector.batters(g["away"])
                if pb and pa_:
                    home_lineup, away_lineup, lineup_source = pb, pa_, "projected"
        r = pred.predict(g["home"], g["away"], h_sp or "", a_sp or "",
                         home_lineup=home_lineup, away_lineup=away_lineup)
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
            "away_sp": a_sp or "TBD", "home_sp": h_sp or "TBD",
            "pitcher_known": pitcher_known,
            "lineup_source": lineup_source,      # "official" | "projected" | None
            "home_win_prob": hp,
            "pick": g["home_abbr"] if pick_home else g["away_abbr"],
            "pick_prob": round(max(hp, 1 - hp), 4),
            "edge": r["contributions_pp"],
            "tier": r["tier"],
            "recal_prob": r["recal_prob"],
            "ts_edge": r["ts_edge"],
            "bullpen_edge": r["bullpen_edge"],
            "home_bp_fip": r["home_bp_fip"],
            "away_bp_fip": r["away_bp_fip"],
            "power_edge": r["power_edge"],
            "baserun_edge": r["baserun_edge"],
            "home_pitcher_matched": r["home_pitcher_matched"],
            "away_pitcher_matched": r["away_pitcher_matched"],
        })
    cards.sort(key=lambda c: (c["start_utc"]))

    metrics = json.loads((PROJECT / "mlbwp" / "artifacts" / "metrics.json").read_text())
    # Headline accuracy is the full blended model (0.67509) when a blend is shipped,
    # else the raw FIP-Elo. lineup_ll is the recalibration fallback.
    blend = pred.blend
    payload = {
        "generated": day0.isoformat(),
        "model": pred.model, "version": pred.version,
        "current_through": pred.current_through,
        "bullpen_through": pred.bullpen_through,
        "season_games_applied": applied,
        "bullpen_teams_loaded": bp_loaded,
        "pitchers_updated": sp_loaded,
        "batters_updated": bat_loaded,
        "accuracy": {
            "log_loss": blend["holdout"]["full_ll"] if blend else metrics["model_log_loss"],
            "recal_log_loss": (blend["holdout"].get("recbp_ll") or blend["holdout"]["recal_ll"])
            if blend else metrics["model_log_loss"],
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
