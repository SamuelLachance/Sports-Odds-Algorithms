"""Every rostered NHL skater and goalie on the site (phase0/nhl_site_players.py).

Before: 563 RAPM skaters only - zero goalies, 773 rostered players with no page,
no production, and departed players still on their old clubs."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_site_players as SP  # noqa: E402


def _row(season, pid, xga, ga, gid="2025020001", team="TOR"):
    return (season, pid, team, gid, xga, ga, xga - ga)


def test_tau_uses_dev_seasons_only():
    """A wild TEST-season goalie must not move the talent-spread estimate."""
    rows = [_row(2012, f"g{i}", 50.0, 50.0 - (i % 7 - 3) * 2, gid=f"2012020{i:03d}")
            for i in range(60)]
    t1 = SP.estimate_tau(rows)
    rows2 = rows + [_row(2024, "wild", 100.0, 20.0, gid="2024020001")]
    assert SP.estimate_tau(rows2) == t1
    assert SP.estimate_tau([]) == 0.043          # documented DEV fallback


def test_eb_shrinks_small_samples_toward_average():
    league = [_row(2025, f"z{i}", 150.0, 150.0, gid=f"2025020{i:03d}") for i in range(20)]
    hot_small = [_row(2025, "small", 10.0, 5.0, gid=f"20250209{i:02d}") for i in range(3)]
    hot_big = [_row(2025, "big", 10.0, 7.5, gid=f"20250208{i:02d}") for i in range(40)]
    q, meta = SP.goalie_quality(league + hot_small + hot_big, 2026, tau=0.043)
    # raw rates: small +0.50 of xGA, big +0.25; shrunk: the big sample wins
    assert q["big"]["theta"] > q["small"]["theta"] > 0
    assert 0 < q["small"]["rel"] < q["big"]["rel"] < 1
    assert meta["window"] == "2025-26 to 2025-26"
    SP.rate_goalies(q, list(q))
    assert q["big"]["rating"] > q["small"]["rating"] > 50


def test_recency_weighting_prefers_recent_seasons():
    league = [_row(s, f"z{s}{i}", 150.0, 150.0, gid=f"{s}0201{i:02d}")
              for s in (2022, 2025) for i in range(20)]
    old_good = [_row(2022, "a", 10.0, 8.0, gid=f"20220200{i:02d}") for i in range(30)]
    new_bad = [_row(2025, "a", 10.0, 11.0, gid=f"20250200{i:02d}") for i in range(30)]
    q, _ = SP.goalie_quality(league + old_good + new_bad, 2026, tau=0.043)
    # equal volume, opposite signs: the recent (bad) season dominates
    assert q["a"]["theta"] < 0


def _names():
    return {
        "1": {"name": "Star Center", "pos": "C", "team": "EDM", "on_roster": True, "born": "1997-01-13"},
        "2": {"name": "Rookie Wing", "pos": "L", "team": "EDM", "on_roster": True},
        "3": {"name": "Thin D", "pos": "D", "team": "EDM", "on_roster": True},
        "4": {"name": "Starter G", "pos": "G", "team": "EDM", "on_roster": True},
        "5": {"name": "Backup G", "pos": "G", "team": "EDM", "on_roster": True},
        "6": {"name": "Departed", "pos": "R", "team": "CHI", "on_roster": False},
        "7": {"name": "Legacy", "pos": "R", "team": "ARI"},
    }


def _build():
    rapm = {"1": {"off": 0.3, "def": 0.1, "net": 0.4, "toi_min": 5000.0, "rel": 0.8, "rating": 88.0},
            "3": {"off": 0.0, "def": 0.01, "net": 0.01, "toi_min": 400.0, "rel": 0.3, "rating": 51.0},
            "6": {"off": 0.2, "def": 0.0, "net": 0.2, "toi_min": 3000.0, "rel": 0.7, "rating": 70.0}}
    stats = {"cur": {"skaters": {"1": {"gp": 3, "g": 1, "a": 2, "p": 3}},
                     "goalies": {"4": {"gp": 3, "w": 2, "toi": 10800}}},
             "prev": {"skaters": {"1": {"gp": 82, "g": 48, "a": 90, "p": 138, "toi": 1379}},
                      "goalies": {"4": {"gp": 60, "w": 35, "toi": 200000}}}}
    career = {"skaters": {"1": {"gp": 794, "p": 1220}}, "goalies": {}}
    gq = {"4": {"theta": 0.05, "rel": 0.4, "gsax60": 0.14, "gp": 150, "xga": 400.0,
                "ga": 380.0, "gsax": 20.0, "rating": 80.0},
          "5": {"theta": 0.0, "rel": 0.02, "gsax60": 0.0, "gp": 4, "xga": 10.0,
                "ga": 10.0, "gsax": 0.0, "rating": 50.0}}
    gseason = {2025: {"4": {"gp": 60, "xga": 160.0, "ga": 150.0, "gsax": 10.0}}}
    return SP.build_players(_names(), rapm, stats, career, gq, gseason, 2026,
                            date(2026, 9, 24), "2022-23 to 2025-26", "2022-23 to 2025-26")


def test_only_current_roster_players_appear():
    ps = _build()
    assert set(ps) == {"1", "2", "3", "4", "5"}      # departed + legacy excluded


def test_every_player_carries_the_contract_keys():
    for p in _build().values():
        assert {"name", "pos", "team", "off", "def", "net", "rating", "toi",
                "grp", "stats", "nr", "rel"} <= set(p)


def test_ratings_and_their_reasons():
    ps = _build()
    assert ps["1"]["rating"] == 88.0 and ps["1"]["nr"] is None and ps["1"]["age"] == 29
    assert ps["2"]["rating"] is None and "no 5v5 minutes" in ps["2"]["nr"]
    assert ps["3"]["rating"] is None and "under 1,000 5v5 minutes" in ps["3"]["nr"]
    assert ps["3"]["net"] == 0.01 and ps["3"]["toi"] == 400.0   # value kept, rating not
    assert ps["4"]["rating"] == 80.0 and ps["4"]["gsax60"] == 0.14 and ps["4"]["grp"] == "G"
    assert ps["4"]["off"] is None and ps["4"]["g_win"]["gp"] == 150
    assert ps["5"]["rating"] is None and "fewer than 10" in ps["5"]["nr"]


def test_stat_lines_and_the_line_to_lead_with():
    ps = _build()
    s = ps["1"]["stats"]
    assert s["cur"]["gp"] == 3 and s["prev"]["p"] == 138 and s["career"]["p"] == 1220
    assert s["main"] == "prev"                          # < 10 GP this season
    g = ps["4"]["stats"]["prev"]
    assert g["gsax"] == 10.0 and g["xga"] == 160.0      # MoneyPuck GSAx joined in
    assert ps["2"]["stats"]["main"] is None
    assert SP.main_line({"cur": {"gp": 12}, "prev": {"gp": 80}}) == "cur"


def test_ranks_among_displayed_ratings():
    ps = _build()
    assert ps["1"]["rk"] == 1 and ps["1"]["rk_pos"] == 1
    assert "rk" not in ps["2"]
    assert ps["4"]["rk"] == 1


def test_team_blocks_lineup_rating_and_roster_order():
    ps = _build()
    tb = SP.team_blocks(ps, ["EDM", "TOR"])["EDM"]
    # every lineup slot counts; unrated slots with no RAPM row take the prior 50
    assert tb["gb_f"] == 69.0 and tb["n_f"] == 1 and tb["gb_d"] == 50.0
    assert tb["glassbox"] == 62.7 and tb["gb_rank"] == 1
    assert tb["goalies"][0] == "4" and tb["gb_g"] == 80.0      # most-used goalie
    assert tb["roster"][:1] == ["1"] and set(tb["roster"]) == set(ps)
    assert tb["use"] == "cur"                            # the team has played
    assert tb["top"] == ["1"]
    empty = SP.team_blocks(ps, ["EDM", "TOR"])["TOR"]
    assert empty["glassbox"] is None and empty["roster"] == [] and empty["gb_rank"] is None


def test_rapm_window_is_recorded_against_the_ratings_hash(tmp_path):
    r = tmp_path / "r.json"
    m = tmp_path / "m.json"
    shifts = tmp_path / "shifts.csv"                  # local refit machine only
    spine = tmp_path / "games.csv"
    spine.write_text("game_id,season,type\n2025020001,20252026,2\n")
    shifts.write_text("game_id,player_id\n2025020001,1\n")
    r.write_text("{}")
    kw = dict(meta_path=str(m), shifts_path=str(shifts), spine_path=str(spine),
              warn=lambda *_: None)
    info = SP.rapm_meta([20212022, 20222023, 20232024, 20242025, 20252026], str(r), **kw)
    assert info["window"] == "2022-23 to 2025-26" and info["built"] is not None
    assert not info["inferred"] and m.is_file()
    # same ratings file, spine rolled a season: the recorded window stands
    info2 = SP.rapm_meta([20222023, 20232024, 20242025, 20252026, 20262027], str(r), **kw)
    assert info2 == info
    r.write_text('{"x": 1}')                           # local refit -> re-derived
    info3 = SP.rapm_meta([20232024, 20242025, 20252026, 20262027], str(r), **kw)
    assert info3["window"] == "2023-24 to 2026-27"


def test_real_goalie_data_gives_a_sane_ladder():
    rows = SP.load_goalie_games()
    if len(rows) < 1000:
        pytest.skip("goalie xG file not present")
    tau = SP.estimate_tau(rows)
    assert 0.02 < tau < 0.08
    q, meta = SP.goalie_quality(rows, 2026, tau)
    sd = SP.rate_goalies(q, list(q))
    assert sd > 0
    qual = [v for v in q.values() if v["gp"] >= SP.MIN_GP]
    assert len(qual) > 50
    # shrinkage: no qualified goalie's estimate exceeds his raw rate
    for v in qual:
        raw = v["gsax"] / v["xga"]
        assert abs(v["theta"]) <= abs(raw) + 0.05
