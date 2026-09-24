"""NHL standings, alignment and team identity (phase0/nhl_site_teams.py).

The NHL pages showed one league-wide list, sorted by raw points, with no
division, conference, games played, tiebreaker or form, and in preseason a
table of 32 teams at 0-0-0 ranked 1-32. These pin the NHL's own rules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_site_teams as ST  # noqa: E402


def _g(i, d, h, a, hs, as_, last="REG"):
    return {"id": i, "d": d, "home": h, "away": a, "hs": hs, "as": as_, "last": last}


def test_alignment_is_32_teams_in_4_divisions_of_8():
    assert len(ST.TEAMS) == 32 == len(set(ST.TEAMS))
    for div, (conf, teams) in ST.ALIGN.items():
        assert len(teams) == 8
        assert div in ST.CONFERENCES[conf]
    assert "UTA" in ST.ALIGN["Central"][1]
    assert set(ST.NAMES) == set(ST.TEAMS)


def test_team_meta_static_fallback_and_api_override():
    m = ST.team_meta()
    assert m["TOR"] == {"name": "Toronto Maple Leafs", "city": "Toronto",
                        "nick": "Maple Leafs", "div": "Atlantic", "conf": "Eastern"}
    assert m["NYR"]["name"] == "New York Rangers"
    api = {"UTA": {"name": "Utah Mammoth", "city": "Utah", "nick": "Mammoth",
                   "div": "Central", "conf": "Western"},
           "ARI": {"name": "Arizona Coyotes", "div": "Central"},     # defunct: ignored
           "TOR": {"name": "Toronto Maple Leafs", "div": None}}      # partial: keeps div
    m2 = ST.team_meta(api)
    assert "ARI" not in m2 and len(m2) == 32
    assert m2["TOR"]["div"] == "Atlantic"
    assert m2["UTA"]["nick"] == "Mammoth"


def test_standings_counts_ot_so_regulation_wins_and_splits():
    t = ST.standings([
        _g(1, "2026-10-01", "TOR", "MTL", 3, 2, "REG"),   # TOR RW, MTL L
        _g(2, "2026-10-02", "MTL", "TOR", 4, 3, "OT"),    # MTL ROW, TOR OTL
        _g(3, "2026-10-03", "TOR", "MTL", 2, 1, "SO"),    # TOR W (not ROW), MTL OTL
    ], ["TOR", "MTL", "BOS"])
    tor, mtl, bos = t["TOR"], t["MTL"], t["BOS"]
    assert (tor["w"], tor["l"], tor["otl"], tor["pts"]) == (2, 0, 1, 5)
    assert (tor["rw"], tor["row"]) == (1, 1)
    assert (mtl["w"], mtl["l"], mtl["otl"], mtl["pts"]) == (1, 1, 1, 3)
    assert (mtl["rw"], mtl["row"]) == (0, 1)
    assert (tor["gf"], tor["ga"], tor["gd"]) == (8, 7, 1)   # SO winner credited a goal
    assert tor["home"] == "2-0-0" and tor["away"] == "0-0-1"
    assert tor["l10"] == "2-0-1" and tor["streak"] == "W1"
    assert mtl["streak"] == "OT1"
    assert tor["pts_pct"] == round(5 / 6, 3)
    assert bos["gp"] == 0 and bos["pts_pct"] is None and bos["streak"] is None


def test_l10_and_streak_use_the_most_recent_games():
    games = [_g(i, f"2026-10-{i:02d}", "TOR", "MTL", 1, 3) for i in range(1, 6)]
    games += [_g(10 + i, f"2026-11-{i:02d}", "TOR", "MTL", 4, 1) for i in range(1, 9)]
    t = ST.standings(list(reversed(games)), ["TOR", "MTL"])   # order-insensitive input
    assert t["TOR"]["l10"] == "8-2-0"
    assert t["TOR"]["streak"] == "W8"
    assert t["MTL"]["streak"] == "L8"


def test_tiebreak_order_points_then_fewer_games_then_rw():
    base = {"w": 0, "l": 0, "otl": 0, "row": 0, "gd": 0, "gf": 0}
    a = {**base, "pts": 10, "gp": 8, "rw": 3}
    b = {**base, "pts": 10, "gp": 7, "rw": 1}     # games in hand wins the tie
    c = {**base, "pts": 10, "gp": 8, "rw": 4}     # more RW beats a
    order = sorted({"A": a, "B": b, "C": c}.items(), key=lambda kv: ST.sort_key(kv[1]))
    assert [k for k, _ in order] == ["B", "C", "A"]


def test_rank_standings_marks_division_top3_and_two_wild_cards():
    meta = ST.team_meta()
    games, i = [], 0
    # give every team a distinct point total: team k wins k games vs a dummy loss
    for k, t in enumerate(ST.TEAMS):
        for _ in range(k % 16):
            i += 1
            games.append(_g(i, "2026-10-01", t, "ZZZ", 3, 1))
    table = ST.rank_standings(ST.standings(games, ST.TEAMS), meta)
    for conf in ST.CONFERENCES:
        members = [t for t in ST.TEAMS if meta[t]["conf"] == conf]
        assert sum(1 for t in members if table[t]["po"] == "div") == 6
        assert sum(1 for t in members if table[t]["po"] == "wc") == 2
        assert sorted(table[t]["conf_rank"] for t in members) == list(range(1, 17))
        wcs = sorted((table[t]["wc_rank"], t) for t in members if table[t]["wc_rank"])
        assert [table[t]["po"] for _, t in wcs[:2]] == ["wc", "wc"]
        assert all(table[t]["po"] is None for _, t in wcs[2:])
    for div in ST.DIVISIONS:
        members = [t for t in ST.TEAMS if meta[t]["div"] == div]
        assert sorted(table[t]["div_rank"] for t in members) == list(range(1, 9))
    assert sorted(r["league_rank"] for r in table.values()) == list(range(1, 33))


def test_preseason_table_has_no_ranks_or_playoff_flags():
    """0-0-0 everywhere: an order would only restate the team codes."""
    table = ST.rank_standings(ST.standings([], ST.TEAMS), ST.team_meta())
    for r in table.values():
        assert r["league_rank"] is None and r["div_rank"] is None and r["po"] is None


def test_spine_standings_reproduce_the_official_final_table():
    """Display parity with the NHL: the previous season's served final table
    (computed from the results spine) equals the league's own - record, goals,
    RW/ROW, every rank and who made the playoffs. Skips without the caches."""
    pay = ROOT / "site" / "data" / "nhl.json"
    cache = ROOT / "data" / "nhl_site_standings.json"
    if not pay.is_file() or not cache.is_file():
        pytest.skip("payload or standings cache not built")
    n = json.loads(pay.read_text(encoding="utf-8"))
    prev = (n.get("prev_season") or {}).get("season")
    api = (json.loads(cache.read_text(encoding="utf-8")).get("seasons", {})
           .get(str(prev)) or {}).get("teams")
    if not api:
        pytest.skip("no official table cached for the previous season")
    for t, tm in n["teams"].items():
        p, a = tm["prev"], api[t]
        for k in ("gp", "w", "l", "otl", "pts", "gf", "ga", "rw", "row"):
            assert p[k] == a[k], (t, k)
        assert (p["league_rank"], p["conf_rank"], p["div_rank"]) == (
            a["league_seq"], a["conf_seq"], a["div_seq"]), t
        assert (p["po"] is not None) == (a["clinch"] in ("x", "y", "z", "p")), t
