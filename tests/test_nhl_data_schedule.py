"""The NHL season schedule and the refresh step that feeds it.

  * the serve predicts every unplayed game of the season, not a 10-day window
    (phase0/nhl_site_schedule.py);
  * nhl_update keeps games in progress and uses the US-Eastern date
    (a UTC date dropped tonight's games from 00:00-04:00 UTC);
  * a serve during a game never overwrites its pre-game ledger entry
    (phase0/nhl_serve.py hold_started).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_site_fetch as FX  # noqa: E402
import nhl_update as U  # noqa: E402
from nhl_site_schedule import is_near, spine_rest, unplayed_rows  # noqa: E402

S = 20262027


def _sg(gid, d, h, a, typ=2, state="FUT", sched="OK", t=None):
    return {"id": gid, "d": d, "home": h, "away": a, "type": typ, "state": state,
            "sched": sched, "t": t}


def test_unplayed_rows_merge_slate_and_season():
    season = [_sg(2026020001, "2026-10-01", "TOR", "MTL", t="2026-10-01T23:00:00Z"),
              _sg(2026020002, "2026-10-02", "BOS", "NYR"),
              _sg(2026020003, "2026-10-03", "EDM", "CGY", sched="CNCL"),   # cancelled
              _sg(2026020004, "2026-10-04", "SEA", "VAN", sched="PPD"),    # postponed
              _sg(2026010001, "2026-09-20", "TOR", "OTT", typ=1),          # preseason
              _sg(2025020999, "2026-04-10", "TOR", "OTT")]                 # other season
    slate = [{"id": 2026020002, "d": "2026-10-05", "home": "BOS", "away": "NYR",
              "t": "2026-10-05T23:00:00Z", "state": "PRE"}]
    rows = unplayed_rows({2026020001}, slate, season, S)    # 001 already final
    ids = [r["id"] for r in rows]
    assert ids == [2026020004, 2026020002]
    moved = rows[1]
    assert moved["d"] == "2026-10-05" and moved["src"] == "slate"   # slate wins
    assert rows[0]["ppd"] == 1


def test_near_window():
    assert is_near("2026-10-04", date(2026, 9, 24))
    assert not is_near("2026-10-05", date(2026, 9, 24))


def test_spine_rest_matches_the_model_feature_rule():
    np = pytest.importorskip("numpy")
    from nhl_features_eval import build_features
    from nhl_glicko2_eval import load_games
    games = load_games()[-3000:]
    rests = spine_rest(games)
    F = build_features(games)
    rd = np.array([r[0] - r[1] for r in rests], dtype=float)
    assert np.array_equal(rd, F["rest_diff"])
    assert np.array_equal(np.array([r[2] for r in rests], dtype=float), F["b2b_home"])
    assert np.array_equal(np.array([r[3] for r in rests], dtype=float), F["b2b_away"])


def _week(games):
    return {"gameWeek": [{"date": d, "games": gs} for d, gs in games]}


def _api_game(gid, state, hs=None, as_=None):
    return {"id": gid, "gameType": 2, "gameState": state, "season": S,
            "homeTeam": {"abbrev": "TOR", "score": hs}, "awayTeam": {"abbrev": "MTL", "score": as_},
            "startTimeUTC": "2026-10-01T23:00:00Z",
            "gameOutcome": {"lastPeriodType": "OT"} if state == "FINAL" else None}


def test_collect_week_keeps_live_games_and_finals():
    data = _week([("2026-09-30", [_api_game(1, "FINAL", 3, 2), _api_game(2, "LIVE", 1, 1)]),
                  ("2026-10-01", [_api_game(3, "FUT"), _api_game(4, "PRE")]),
                  ("2026-09-29", [_api_game(5, "FUT")])])        # stale unplayed, dropped
    seen = set()
    new, up = U.collect_week(data, seen, "2026-10-01")
    assert [r["game_id"] for r in new] == [1] and new[0]["last_period"] == "OT"
    assert 1 in seen
    assert {u["id"]: u["state"] for u in up} == {2: "LIVE", 3: "FUT", 4: "PRE"}
    new2, _ = U.collect_week(data, seen, "2026-10-01")
    assert new2 == []                                       # dedup by id


def test_today_is_the_eastern_date():
    d = U.et_today()
    assert isinstance(d, date)


def test_parse_club_schedule_and_merge():
    d = {"games": [
        {"id": 11, "gameType": 1, "gameDate": "2026-09-20"},
        {"id": 12, "gameType": 2, "gameDate": "2026-10-01", "startTimeUTC": "T",
         "gameState": "FINAL", "gameScheduleState": "OK",
         "homeTeam": {"abbrev": "TOR", "score": 4}, "awayTeam": {"abbrev": "MTL", "score": 1},
         "gameOutcome": {"lastPeriodType": "REG"}},
        {"id": 13, "gameType": 2, "gameDate": "2026-10-03", "gameState": "FUT",
         "gameScheduleState": "PPD", "homeTeam": {"abbrev": "OTT"}, "awayTeam": {"abbrev": "TOR"}}]}
    rows = FX.parse_club_schedule(d)
    assert [r["id"] for r in rows] == [12, 13]
    assert rows[0]["hs"] == 4 and rows[0]["last"] == "REG" and rows[1]["sched"] == "PPD"
    old = [{"id": 99, "d": "2026-11-01"}, {"id": 13, "d": "2026-10-02"}]
    part = FX.merge_schedule(old, rows, complete=False)
    assert {g["id"] for g in part} == {12, 13, 99}
    assert next(g for g in part if g["id"] == 13)["d"] == "2026-10-03"   # new wins
    assert {g["id"] for g in FX.merge_schedule(old, rows, complete=True)} == {12, 13}


def test_parse_standings_block():
    d = {"standings": [{"teamAbbrev": {"default": "COL"}, "seasonId": 20252026,
                        "teamName": {"default": "Colorado Avalanche"},
                        "placeName": {"default": "Colorado"},
                        "teamCommonName": {"default": "Avalanche"},
                        "divisionName": "Central", "conferenceName": "Western",
                        "clinchIndicator": "p", "points": 121, "wins": 55}]}
    season, teams = FX.parse_standings(d)
    assert season == 20252026
    assert teams["COL"]["clinch"] == "p" and teams["COL"]["div"] == "Central"
    assert teams["COL"]["name"] == "Colorado Avalanche" and teams["COL"]["pts"] == 121


def test_stats_fall_back_to_club_stats_when_the_rest_api_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(FX, "stats_path", lambda s: str(tmp_path / f"st_{s}.json"))

    def fake(url):
        if "stats/rest" in url:
            raise OSError("rest down")
        team = url.split("/club-stats/")[1].split("/")[0]
        if team == "TOR":
            return {"skaters": [{"playerId": 1, "gamesPlayed": 10, "goals": 5, "assists": 5,
                                 "points": 10, "plusMinus": 2, "shots": 30,
                                 "penaltyMinutes": 4, "avgTimeOnIcePerGame": 1000.0}],
                    "goalies": [{"playerId": 9, "gamesPlayed": 5, "gamesStarted": 5, "wins": 3,
                                 "losses": 2, "overtimeLosses": 0, "shutouts": 1,
                                 "shotsAgainst": 150, "goalsAgainst": 12, "timeOnIce": 18000}]}
        if team == "FLA":                                   # traded skater: summed
            return {"skaters": [{"playerId": 1, "gamesPlayed": 10, "goals": 1, "assists": 1,
                                 "points": 2, "plusMinus": 0, "shots": 10,
                                 "penaltyMinutes": 0, "avgTimeOnIcePerGame": 800.0}],
                    "goalies": []}
        return {"skaters": [], "goalies": []}
    monkeypatch.setattr(FX, "TEAMS", ("TOR", "FLA", "MTL"))
    d = FX.fetch_season_stats_club(20252026, teams=("TOR", "FLA", "MTL"), get_fn=fake, sleep=0)
    s = d["skaters"]["1"]
    assert (s["gp"], s["g"], s["p"], s["toi_pg"], s["ppp"]) == (20, 6, 12, 900, None)
    assert s["tm"] == "TOR, FLA"
    g = d["goalies"]["9"]
    assert g["svp"] == round(1 - 12 / 150, 4) and g["gaa"] == 2.4
    msg = FX.fetch_stats(20252026, 3, force=True, get_fn=fake)
    assert "(club)" in msg


def test_rest_lines_keep_power_play_points():
    s = FX._skater_line({"gamesPlayed": 82, "goals": 48, "assists": 90, "points": 138,
                         "plusMinus": 17, "ppPoints": 54, "shots": 306,
                         "timeOnIcePerGame": 1379.4, "teamAbbrevs": "EDM"})
    assert s["ppp"] == 54 and s["toi_pg"] == 1379 and s["tm"] == "EDM"
    g = FX._goalie_line({"gamesPlayed": 51, "savePct": 0.911612, "goalsAgainstAverage": 2.4973,
                         "teamAbbrevs": "NYR,BOS"})
    assert g["svp"] == 0.9116 and g["gaa"] == 2.5 and g["tm"] == "NYR, BOS"


def test_a_started_game_keeps_its_pre_game_ledger_entry():
    pytest.importorskip("numpy")
    from nhl_serve import hold_started, started
    now = "2026-10-01T23:30:00Z"
    ledger = {"1": {"hp": 0.61, "ct": {"elo": 11.0}, "t": "2026-10-01T20:00:00Z", "tier": "EARLY"}}
    live = {"id": 1, "hp": 0.55, "hs": None, "start_utc": "2026-10-01T23:00:00Z"}
    later = {"id": 2, "hp": 0.52, "hs": None, "start_utc": "2026-10-02T23:00:00Z"}
    fresh = {"id": 3, "hp": 0.50, "hs": None, "state": "LIVE"}          # no entry yet
    assert started(live, now) and not started(later, now) and started(fresh, now)
    rows, held = hold_started([live, later, fresh], ledger, now)
    assert held == 1 and rows == [later, fresh]
    assert live["hp"] == 0.61 and live["frozen_at"] == "2026-10-01T20:00:00Z"
