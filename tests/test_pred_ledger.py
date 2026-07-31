"""MLB prediction-ledger rules: pre-game-only writes, immutable graded rows."""
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
