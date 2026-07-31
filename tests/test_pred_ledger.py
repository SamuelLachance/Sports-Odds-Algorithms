"""MLB prediction-ledger rules: pre-game-only writes, immutable graded rows,
information-tier stamping (documents/pick_policy.md)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from mlbwp import pred_ledger


def _board(games):
    return {"leagues": [{"code": "mlb", "active": True, "games": games}]}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


NOW = datetime(2026, 7, 30, 18, 0, tzinfo=timezone.utc)


def test_pregame_recorded_final_ignored(tmp_path):
    board = _board([
        {"game_pk": 1, "date": "2026-07-30", "home_abbr": "TB", "away_abbr": "TEX",
         "home_win_prob": 0.61, "recal_prob": 0.59, "pitcher_known": True,
         "state": "Preview", "start_utc": "2026-07-30T23:10:00Z"},
        {"game_pk": 2, "date": "2026-07-30", "home_abbr": "NYY", "away_abbr": "BOS",
         "home_win_prob": 0.55, "state": "Final",                 # already played
         "start_utc": "2026-07-30T16:10:00Z"},
        {"game_pk": 3, "date": "2026-07-30", "home_abbr": "LAD", "away_abbr": "SD",
         "home_win_prob": 0.58, "state": "Live",                  # started
         "start_utc": "2026-07-30T17:00:00Z"},
    ])
    bp = _write(tmp_path, "board.json", board)
    fp = _write(tmp_path, "finals.json", [])
    lp = str(tmp_path / "ledger.json")
    st = pred_ledger.update(bp, fp, lp, now=NOW)
    led = json.load(open(lp))
    assert st["new"] == 1 and set(led) == {"1"}   # only the strictly-pre-game one
    assert led["1"]["p"] == 0.61 and led["1"]["rp"] == 0.59


def test_latest_pregame_wins_then_graded_immutable(tmp_path):
    game = {"game_pk": 1, "date": "2026-07-30", "home_abbr": "TB", "away_abbr": "TEX",
            "home_win_prob": 0.61, "pitcher_known": True,
            "state": "Preview", "start_utc": "2026-07-30T23:10:00Z"}
    bp = _write(tmp_path, "board.json", _board([game]))
    fp = _write(tmp_path, "finals.json", [])
    lp = str(tmp_path / "ledger.json")
    pred_ledger.update(bp, fp, lp, now=NOW)
    game["home_win_prob"] = 0.64                   # pitcher announced, prob moved
    _write(tmp_path, "board.json", _board([game]))
    st = pred_ledger.update(bp, fp, lp, now=NOW)
    assert st["updated"] == 1
    assert json.load(open(lp))["1"]["p"] == 0.64   # latest pre-game prediction

    # game goes final -> graded; later board updates must NOT touch it
    _write(tmp_path, "finals.json",
           [{"game_pk": 1, "home_win": 1, "home_score": 5, "away_score": 2}])
    st = pred_ledger.update(bp, fp, lp, now=NOW)
    assert st["graded"] == 1
    game["home_win_prob"] = 0.99                   # post-hoc prob (leak) attempt
    _write(tmp_path, "board.json", _board([game]))
    pred_ledger.update(bp, fp, lp, now=NOW)
    led = json.load(open(lp))
    assert led["1"]["p"] == 0.64 and led["1"]["y"] == 1


def test_info_tier_mapping():
    """The mapping documented in pred_ledger's docstring, pinned.

    A rotation-PROJECTED starter still counts as PROJECTED (the model has a
    named arm); starters-known-but-no-lineup is PROJECTED, not CONFIRMED.
    """
    T = pred_ledger.info_tier
    assert T({"pitcher_known": True, "lineup_source": "official"}) == "CONFIRMED"
    assert T({"pitcher_known": True, "lineup_source": "projected"}) == "PROJECTED"
    assert T({"pitcher_known": True, "lineup_source": None}) == "PROJECTED"
    assert T({"pitcher_known": True, "sp_projected": True,
              "lineup_source": "projected"}) == "PROJECTED"
    assert T({"pitcher_known": False, "lineup_source": None}) == "EARLY"
    assert T({}) == "EARLY"
    # pre-stamp entries fall back to sp_known rather than being back-filled
    assert pred_ledger.entry_tier({"sp_known": True}) == "PROJECTED"
    assert pred_ledger.entry_tier({"sp_known": False}) == "EARLY"
    assert pred_ledger.entry_tier({"tier": "CONFIRMED", "sp_known": False}) == "CONFIRMED"
    assert pred_ledger.entry_tier({"tier": "GARBAGE"}) == "EARLY"


def test_tier_is_restamped_on_every_pregame_update(tmp_path):
    """A game first seen EARLY must end up stamped with the tier of the forecast
    that will actually be GRADED — the last pre-game one."""
    game = {"game_pk": 1, "date": "2026-07-30", "home_abbr": "TB", "away_abbr": "TEX",
            "home_win_prob": 0.52, "state": "Preview",
            "start_utc": "2026-07-30T23:10:00Z",
            "pitcher_known": False, "lineup_source": None}
    bp = _write(tmp_path, "board.json", _board([game]))
    fp = _write(tmp_path, "finals.json", [])
    lp = str(tmp_path / "ledger.json")
    pred_ledger.update(bp, fp, lp, now=NOW)
    assert json.load(open(lp))["1"]["tier"] == "EARLY"

    game.update(pitcher_known=True, lineup_source="projected", home_win_prob=0.58)
    _write(tmp_path, "board.json", _board([game]))
    pred_ledger.update(bp, fp, lp, now=NOW)
    assert json.load(open(lp))["1"]["tier"] == "PROJECTED"

    # lineups post; the probability happens NOT to move -> the tier must still
    # be re-stamped (an update counted only on p-change would miss this).
    game.update(lineup_source="official")
    _write(tmp_path, "board.json", _board([game]))
    st = pred_ledger.update(bp, fp, lp, now=NOW)
    assert st["updated"] == 1
    assert json.load(open(lp))["1"]["tier"] == "CONFIRMED"

    # graded -> immutable, tier included
    _write(tmp_path, "finals.json",
           [{"game_pk": 1, "home_win": 1, "home_score": 3, "away_score": 1}])
    pred_ledger.update(bp, fp, lp, now=NOW)
    game.update(pitcher_known=False, lineup_source=None, home_win_prob=0.99)
    _write(tmp_path, "board.json", _board([game]))
    pred_ledger.update(bp, fp, lp, now=NOW)
    ent = json.load(open(lp))["1"]
    assert ent["tier"] == "CONFIRMED" and ent["p"] == 0.58 and ent["y"] == 1


def test_post_start_stamps_the_label_only_when_the_forecast_is_the_same(tmp_path):
    """A pre-stamp entry whose game has started is still ungraded. Its NUMBER is
    locked forever, but if the card's probability is identical the stored
    forecast IS that card's forecast, so the card's information state labels it
    exactly. If the probability moved at all, the card knows something the
    stored forecast did not and the entry keeps its fallback tier."""
    game = {"game_pk": 1, "date": "2026-07-30", "home_abbr": "TB", "away_abbr": "TEX",
            "home_win_prob": 0.6, "state": "Preview",
            "start_utc": "2026-07-30T23:10:00Z",
            "pitcher_known": True, "lineup_source": "official"}
    bp = _write(tmp_path, "board.json", _board([game]))
    fp = _write(tmp_path, "finals.json", [])
    lp = str(tmp_path / "ledger.json")
    pred_ledger.update(bp, fp, lp, now=NOW)
    led = json.load(open(lp))
    del led["1"]["tier"]                       # simulate a pre-stamp entry
    json.dump(led, open(lp, "w"))

    started = datetime(2026, 7, 31, 0, 0, tzinfo=timezone.utc)   # first pitch passed
    st = pred_ledger.update(bp, fp, lp, now=started)
    ent = json.load(open(lp))["1"]
    assert st["updated"] == 1 and st["new"] == 0
    assert ent["tier"] == "CONFIRMED" and ent["p"] == 0.6        # label only

    # a DIFFERENT probability means the card is a different, better-informed
    # forecast -> no stamp, and still no post-start write of p
    led = json.load(open(lp)); del led["1"]["tier"]; json.dump(led, open(lp, "w"))
    game["home_win_prob"] = 0.64
    _write(tmp_path, "board.json", _board([game]))
    pred_ledger.update(bp, fp, lp, now=started)
    ent = json.load(open(lp))["1"]
    assert "tier" not in ent and ent["p"] == 0.6
    assert pred_ledger.entry_tier(ent) == "PROJECTED"            # fallback, unchanged

    # the stamp still happens on the pass that grades the game (label first,
    # result second), and after that the row is frozen against any card
    game["home_win_prob"] = 0.6
    _write(tmp_path, "board.json", _board([game]))
    _write(tmp_path, "finals.json",
           [{"game_pk": 1, "home_win": 1, "home_score": 3, "away_score": 1}])
    pred_ledger.update(bp, fp, lp, now=started)
    ent = json.load(open(lp))["1"]
    assert ent["tier"] == "CONFIRMED" and ent["y"] == 1 and ent["p"] == 0.6

    game.update(pitcher_known=False, lineup_source=None, home_win_prob=0.6)
    _write(tmp_path, "board.json", _board([game]))
    st = pred_ledger.update(bp, fp, lp, now=started)
    ent = json.load(open(lp))["1"]
    assert st["updated"] == 0 and ent["tier"] == "CONFIRMED" and ent["p"] == 0.6


def test_sp_proj_is_recorded(tmp_path):
    """A rotation-PROJECTED starter is PROJECTED, but the row must say so: the
    board badges it and the record page cannot without the flag."""
    game = {"game_pk": 1, "date": "2026-07-30", "home_abbr": "TB", "away_abbr": "TEX",
            "home_win_prob": 0.58, "state": "Preview",
            "start_utc": "2026-07-30T23:10:00Z",
            "pitcher_known": True, "sp_projected": True, "lineup_source": "projected"}
    bp = _write(tmp_path, "board.json", _board([game]))
    fp = _write(tmp_path, "finals.json", [])
    lp = str(tmp_path / "ledger.json")
    pred_ledger.update(bp, fp, lp, now=NOW)
    ent = json.load(open(lp))["1"]
    assert ent["tier"] == "PROJECTED" and ent["sp_proj"] is True


def test_realized_math(tmp_path):
    lp = str(tmp_path / "ledger.json")
    json.dump({"1": {"p": 0.7, "y": 1}, "2": {"p": 0.6, "y": 0},
               "3": {"p": 0.5, "d": "x"}},          # ungraded: excluded
              open(lp, "w"))
    r = pred_ledger.realized(lp)
    assert r["n"] == 2 and r["acc"] == 0.5
    import math
    exp = (-(math.log(0.7)) - math.log(0.4)) / 2
    assert abs(r["log_loss"] - exp) < 1e-4      # stored value is 5-dp rounded
