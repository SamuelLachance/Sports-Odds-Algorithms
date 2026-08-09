"""Live edge ledger rules: first-record-wins, pre-game-only touches, CLV math,
immutable graded rows, report arithmetic, missing-payload no-op.

Mirrors tests/test_pred_ledger.py: tmp_path fixtures only, no network, no clock
dependence (every call passes an explicit `now`).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from market import edge_ledger as EL

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
START = "2026-07-31T23:10:00Z"          # first pitch, 11 hours after NOW


# ------------------------------------------------------------------ fixtures

def _board(games, code="mlb"):
    return {"leagues": [{"code": code, "active": True, "games": games}]}


def _mlb_game(pk=1, p=0.60, side="home", cur=2.10, state="Preview", start=START,
              value=True, home="CHN", away="NYA", d="2026-07-31", ev_cur=0.1,
              available=True, mkt=None, **kw):
    g = {"game_pk": pk, "date": d, "start_utc": start, "state": state,
         "home": home, "away": away, "home_abbr": "CHC", "away_abbr": "NYY",
         "home_win_prob": p}
    if value:
        g["value"] = {"side": side, "team": "CHC" if side == "home" else "NYY",
                      "ev_open": 0.24, "ev_cur": ev_cur, "open_dec": 2.2,
                      "cur_dec": cur, "available": available, "books": 8}
    if mkt is not None:                       # (home_dec, away_dec) consensus quote
        g["mkt"] = {"home_dec": mkt[0], "away_dec": mkt[1], "books": 8, "ts": "t"}
    g.update(kw)
    return g


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _paths(tmp_path):
    return tmp_path / "board.json", tmp_path / "ledger.json"


def _rows(lp):
    return json.loads(lp.read_text(encoding="utf-8"))["rows"]


# ------------------------------------------------------------------ recording

def test_first_record_wins_under_a_changing_price(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.10)]))
    st = EL.update("mlb", bp, lp, now=NOW)
    assert st["recorded"] == 1
    key = "mlb|1"                                # game_pk IS the key (no date in it)
    row = _rows(lp)[key]
    assert row["dec_at_record"] == 2.10
    assert row["p_model"] == 0.6 and row["p_home"] == 0.6
    assert row["imp_at_record"] == pytest.approx(1 / 2.10, abs=1e-6)
    assert row["ev_at_record"] == pytest.approx(0.6 * 2.10 - 1, abs=1e-6)
    assert row["y"] is None and row["dec_close"] is None and row["n_obs"] == 1

    # price improves 20 minutes later: the RECORD must not move, only the close
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.60)]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(minutes=20))
    assert st["recorded"] == 0 and st["touched"] == 1
    row = _rows(lp)[key]
    assert row["dec_at_record"] == 2.10          # immutable entry price
    assert row["last_pregame_dec"] == 2.60 and row["n_obs"] == 2


def test_away_side_record_uses_complement_prob(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(p=0.40, side="away", cur=3.0)]))
    EL.update("mlb", bp, lp, now=NOW)
    row = next(iter(_rows(lp).values()))
    assert row["side"] == "away" and row["p_home"] == 0.4
    assert row["p_model"] == pytest.approx(0.6)
    assert row["ev_at_record"] == pytest.approx(0.6 * 3.0 - 1, abs=1e-6)


def test_no_badge_and_started_games_are_never_recorded(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([
        _mlb_game(pk=1, value=False),                                   # no edge
        _mlb_game(pk=2, state="Live"),                                  # under way
        _mlb_game(pk=3, state="Final"),                                 # played
        _mlb_game(pk=4, start="2026-07-31T11:00:00Z"),                  # start passed
    ]))
    st = EL.update("mlb", bp, lp, now=NOW)
    assert st["recorded"] == 0
    assert _rows(lp) == {}


def test_mlb_doubleheader_gets_two_independent_rows(tmp_path):
    """Same date+home+away; game_pk is the discriminator, so both edges survive."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([
        _mlb_game(pk=8241, cur=2.10, start="2026-07-31T17:10:00Z"),
        _mlb_game(pk=8242, cur=1.80, start="2026-07-31T23:10:00Z"),
    ]))
    st = EL.update("mlb", bp, lp, now=NOW)
    assert st["recorded"] == 2
    rows = _rows(lp)
    assert set(rows) == {"mlb|8241", "mlb|8242"}
    assert rows["mlb|8241"]["dec_at_record"] == 2.10
    assert rows["mlb|8242"]["dec_at_record"] == 1.80


def test_a_badge_the_site_marks_unavailable_is_not_recorded(tmp_path):
    """edges.py attaches the badge on ev_OPEN but sets available = (ev_cur > 0). A
    badge whose current price is already past our value is advertised as not takeable;
    recording it would put a wager nobody could place into ROI, z and the CLV sample."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([
        _mlb_game(pk=1, cur=1.55, ev_cur=-0.07, available=False),
        _mlb_game(pk=2, home="BOS", away="TOR", cur=1.60, ev_cur=-0.04,
                  available=True),                    # ev_cur alone disqualifies it
        _mlb_game(pk=3, home="SEA", away="TEX", cur=2.30, ev_cur=0.38),
    ]))
    st = EL.update("mlb", bp, lp, now=NOW)
    assert st["recorded"] == 1
    row = _rows(lp)["mlb|3"]
    assert row["dec_at_record"] == 2.30 and row["ev_cur"] == 0.38
    assert row["ev_at_record"] > 0


def test_an_edge_that_becomes_available_later_is_recorded_then(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json",
           _board([_mlb_game(cur=1.55, ev_cur=-0.07, available=False)]))
    assert EL.update("mlb", bp, lp, now=NOW)["recorded"] == 0
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.40, ev_cur=0.44)]))
    assert EL.update("mlb", bp, lp, now=NOW + timedelta(minutes=20))["recorded"] == 1
    assert next(iter(_rows(lp).values()))["dec_at_record"] == 2.40


# ------------------------------------------------------------------ touching

def test_started_or_final_game_never_mutates_a_recorded_row(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.10)]))
    EL.update("mlb", bp, lp, now=NOW)

    # game is now LIVE and the feed shows an in-game price: must not be absorbed
    _write(tmp_path, "board.json", _board([_mlb_game(cur=9.99, state="Live")]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=12))
    assert st["touched"] == 0 and st["closed"] == 1
    row = next(iter(_rows(lp).values()))
    assert row["last_pregame_dec"] == 2.10 and row["dec_close"] == 2.10
    assert row["n_obs"] == 1


def test_side_flip_without_a_market_quote_skips_the_touch(tmp_path):
    """Badge-only payload (no `mkt`): if the badge flips to the other side there is
    no price for the side we hold -> skip rather than log the wrong number."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(side="home", cur=2.10)]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([_mlb_game(side="away", cur=1.55)]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(minutes=20))
    assert st["touched"] == 0
    assert next(iter(_rows(lp).values()))["last_pregame_dec"] == 2.10


# ------------------------------------------------------------- badge-independent close

def test_close_keeps_rolling_after_the_badge_disappears(tmp_path):
    """THE censoring bug: the badge is re-gated every cycle on the CURRENT model
    probability, so adverse news that moves p removes it at exactly the moment the
    market lengthens our price. Rolling the close off the badge would freeze the
    pre-news price and report positive CLV on a bet that lost CLV. The close must
    roll off the unconditional `mkt` quote instead."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.50, mkt=(2.50, 1.60))]))
    EL.update("mlb", bp, lp, now=NOW)

    # model prob drops -> ev_open falls under the threshold -> NO badge at all,
    # while the market drifts our side out to 2.90 (we lost CLV).
    _write(tmp_path, "board.json",
           _board([_mlb_game(p=0.53, value=False, mkt=(2.90, 1.42))]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=7))
    assert st["touched"] == 1 and st["recorded"] == 0
    row = next(iter(_rows(lp).values()))
    assert row["last_pregame_dec"] == 2.90 and not row["clv_censored"]

    _write(tmp_path, "board.json",
           _board([_mlb_game(p=0.53, value=False, state="Live", mkt=(2.90, 1.42))]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=12))
    row = next(iter(_rows(lp).values()))
    assert row["dec_close"] == 2.90
    assert row["clv_pts"] == pytest.approx(1 / 2.90 - 1 / 2.50, abs=1e-6)
    assert row["clv_pts"] < 0                     # truthfully negative, not +0.008


def test_side_flip_still_rolls_our_side_from_the_market_quote(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json",
           _board([_mlb_game(side="home", cur=2.10, mkt=(2.10, 1.80))]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json",
           _board([_mlb_game(side="away", cur=1.55, mkt=(2.35, 1.55))]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(minutes=20))
    assert st["touched"] == 1
    assert next(iter(_rows(lp).values()))["last_pregame_dec"] == 2.35   # OUR side


def test_a_pulled_market_censors_the_row_instead_of_faking_a_close(tmp_path):
    """Postponement / books pulling the event: the feed is live for other games but
    carries no price for ours. That hole ends the price path early and is not
    missing-at-random, so the row is graded for the record but kept out of the gate."""
    bp, lp = _paths(tmp_path)
    other = _mlb_game(pk=2, home="BOS", away="TOR", value=False, mkt=(1.90, 2.00))
    _write(tmp_path, "board.json",
           _board([_mlb_game(cur=2.50, mkt=(2.50, 1.60)), other]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json",
           _board([_mlb_game(cur=2.45, mkt=(2.45, 1.63)), other]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=3))       # n_obs 2, real CLV
    _write(tmp_path, "board.json", _board([_mlb_game(value=False), other]))  # no mkt
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=5))
    assert st["touched"] == 0
    row = _rows(lp)["mlb|1"]
    assert row["n_obs"] == 2 and row["n_missed"] == 1 and row["clv_censored"] is True

    _write(tmp_path, "board.json", _board([_mlb_game(value=False, state="Live"), other]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=13))
    m = EL.report(lp)["leagues"]["mlb"]
    assert _rows(lp)["mlb|1"]["clv_pts"] is not None      # graded for the record
    assert m["n_clv_graded"] == 0 and m["n_clv_censored"] == 1   # but not counted


def test_an_odds_api_outage_censors_nothing(tmp_path):
    """No `mkt` anywhere = the fetch failed, not 'our market was pulled'. A transient
    outage must not censor every open row in the book."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.50, mkt=(2.50, 1.60))]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([_mlb_game(value=False)]))   # nothing quoted
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=5))
    row = next(iter(_rows(lp).values()))
    assert row["n_missed"] == 0 and row["clv_censored"] is False


def test_a_rescheduled_game_keeps_one_row_at_the_original_price(tmp_path):
    """The stable id is the whole key, so a postponed game does not open a second
    row for the same bet (both would count toward the >=100 gate)."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json",
           _board([_mlb_game(cur=2.20, mkt=(2.20, 1.72))]))
    EL.update("mlb", bp, lp, now=NOW)
    moved = _mlb_game(d="2026-08-03", start="2026-08-03T23:10:00Z",
                      cur=2.05, mkt=(2.05, 1.85))
    _write(tmp_path, "board.json", _board([moved]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(days=1))
    assert (st["recorded"], st["touched"]) == (0, 1)
    rows = _rows(lp)
    assert set(rows) == {"mlb|1"}
    row = rows["mlb|1"]
    assert row["dec_at_record"] == 2.20 and row["rescheduled"] is True
    assert row["d"] == "2026-08-03" and row["start_utc"] == "2026-08-03T23:10:00Z"
    assert row["dec_close"] is None            # NOT force-graded off the old date


# ------------------------------------------------------------------ grading

def _record_then_close(tmp_path, cur_open, cur_close, side="home", p=0.60,
                       home_score=None, away_score=None):
    tmp_path.mkdir(parents=True, exist_ok=True)     # allows a nested second fixture
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(p=p, side=side, cur=cur_open)]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([_mlb_game(p=p, side=side, cur=cur_close)]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=10))          # still pre-game
    fin = tmp_path / "finals.json"
    if home_score is None:
        fin.write_text("[]", encoding="utf-8")
    else:
        fin.write_text(json.dumps([{"game_pk": 1, "home_win": float(home_score > away_score),
                                    "home_score": home_score, "away_score": away_score}]),
                       encoding="utf-8")
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=13), results_path=fin)  # started
    return lp, next(iter(_rows(lp).values()))


def test_clv_positive_when_the_line_moves_toward_us(tmp_path):
    lp, row = _record_then_close(tmp_path, 2.50, 2.00)      # price SHORTENED
    assert row["dec_close"] == 2.00
    assert row["clv_pts"] == pytest.approx(1 / 2.00 - 1 / 2.50, abs=1e-6)
    assert row["clv_pts"] > 0                               # moved toward us


def test_clv_negative_when_the_line_moves_away(tmp_path):
    lp, row = _record_then_close(tmp_path, 2.00, 2.50)      # price DRIFTED out
    assert row["clv_pts"] == pytest.approx(1 / 2.50 - 1 / 2.00, abs=1e-6)
    assert row["clv_pts"] < 0


def test_home_side_win_and_loss_roi(tmp_path):
    _, win = _record_then_close(tmp_path, 2.50, 2.00, side="home",
                                home_score=6, away_score=1)
    assert win["y"] == 1 and win["roi"] == pytest.approx(1.50)     # dec_at_record-1
    _, loss = _record_then_close(tmp_path / "b", 2.50, 2.00, side="home",
                                 home_score=1, away_score=6)
    assert loss["y"] == 0 and loss["roi"] == -1.0


def test_away_side_y_is_our_side_not_the_home_side(tmp_path):
    _, r = _record_then_close(tmp_path, 3.00, 2.80, side="away", p=0.40,
                              home_score=1, away_score=6)          # AWAY won
    assert r["side"] == "away" and r["y"] == 1
    assert r["roi"] == pytest.approx(2.00)
    _, r2 = _record_then_close(tmp_path / "b", 3.00, 2.80, side="away", p=0.40,
                               home_score=6, away_score=1)         # HOME won
    assert r2["y"] == 0 and r2["roi"] == -1.0


def test_mlb_grades_after_the_board_drops_the_game(tmp_path):
    """The MLB board is forward-only: a played game vanishes. The stored start_utc
    plus the finals archive must still close and settle the row."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.50)]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.00)]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=10))
    _write(tmp_path, "board.json", _board([]))                 # dropped off the board
    fin = _write(tmp_path, "finals.json",
                 [{"game_pk": 1, "home_win": 1.0, "home_score": 5, "away_score": 2}])
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=14), results_path=fin)
    assert st["closed"] == 1 and st["settled"] == 1
    row = next(iter(_rows(lp).values()))
    assert row["dec_close"] == 2.00 and row["y"] == 1


def test_graded_row_is_immutable_against_a_post_hoc_write(tmp_path):
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.50)]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.00)]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=10))
    fin = _write(tmp_path, "finals.json",
                 [{"game_pk": 1, "home_win": 1.0, "home_score": 5, "away_score": 2}])
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=14), results_path=fin)
    before = dict(next(iter(_rows(lp).values())))

    # a post-hoc payload claiming the game is pre-game again, at a fantasy price
    # and a fantasy probability, must change nothing at all
    _write(tmp_path, "board.json",
           _board([_mlb_game(p=0.99, cur=15.0, state="Preview",
                             start="2026-08-05T23:10:00Z")]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=15), results_path=fin)
    assert (st["recorded"], st["touched"], st["closed"], st["settled"]) == (0, 0, 0, 0)
    assert next(iter(_rows(lp).values())) == before


def test_close_is_frozen_before_the_result_is_known(tmp_path):
    """Two-stage grading: CLV freezes the moment the game starts, y arrives later."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.50)]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.00)]))
    EL.update("mlb", bp, lp, now=NOW + timedelta(hours=10))
    fin = _write(tmp_path, "finals.json", [])
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.00, state="Live")]))
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=12), results_path=fin)
    assert st["closed"] == 1 and st["settled"] == 0
    assert next(iter(_rows(lp).values()))["y"] is None

    fin = _write(tmp_path, "finals.json",
                 [{"game_pk": 1, "home_win": 0.0, "home_score": 2, "away_score": 5}])
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=16), results_path=fin)
    assert st["closed"] == 0 and st["settled"] == 1         # close not re-frozen
    row = next(iter(_rows(lp).values()))
    assert row["dec_close"] == 2.00 and row["y"] == 0 and row["roi"] == -1.0


# ------------------------------------------------------------------ NFL / NHL

def test_nfl_records_and_grades_from_hs_as(tmp_path):
    lp = tmp_path / "ledger.json"
    game = {"w": 1, "d": "2026-09-13", "t": "13:00", "home": "SEA", "away": "NE",
            "ph": 0.611, "hs": None, "as": None,
            "value": {"side": "away", "team": "NE", "ev_open": 0.25, "ev_cur": 0.05,
                      "open_dec": 3.3, "cur_dec": 3.2, "available": True, "books": 9}}
    p = _write(tmp_path, "nfl.json", {"schedule": [game]})
    n0 = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)      # 8am ET, pre-kickoff
    assert EL.update("nfl", p, lp, now=n0)["recorded"] == 1
    key = "nfl|2026-09-13|NE@SEA"
    row = _rows(lp)[key]
    assert row["start_utc"] == "2026-09-13T17:00:00Z"           # 13:00 ET -> UTC
    assert row["p_model"] == pytest.approx(1 - 0.611)

    game["value"]["cur_dec"] = 3.0                              # line moved toward us
    _write(tmp_path, "nfl.json", {"schedule": [game]})
    assert EL.update("nfl", p, lp, now=n0 + timedelta(hours=4))["touched"] == 1

    game.pop("value"); game["hs"], game["as"] = 17, 24          # away won
    _write(tmp_path, "nfl.json", {"schedule": [game]})
    st = EL.update("nfl", p, lp, now=n0 + timedelta(hours=10))
    assert st["closed"] == 1 and st["settled"] == 1
    row = _rows(lp)[key]
    assert row["dec_close"] == 3.0 and row["clv_pts"] > 0
    assert row["y"] == 1 and row["roi"] == pytest.approx(2.2)


def test_nhl_uses_payload_id_and_hs_grading(tmp_path):
    lp = tmp_path / "ledger.json"
    game = {"id": 2026020123, "d": "2026-11-02", "home": "COL", "away": "DAL",
            "hp": 0.55, "hs": None, "as": None,
            "value": {"side": "home", "team": "COL", "ev_open": 0.21, "ev_cur": 0.02,
                      "open_dec": 2.3, "cur_dec": 2.2, "available": True, "books": 7}}
    p = _write(tmp_path, "nhl.json", {"schedule": [game]})
    assert EL.update("nhl", p, lp, now=NOW)["recorded"] == 1
    assert set(_rows(lp)) == {"nhl|2026020123"}
    game.pop("value"); game["hs"], game["as"] = 4, 1
    _write(tmp_path, "nhl.json", {"schedule": [game]})
    st = EL.update("nhl", p, lp, now=NOW + timedelta(days=1))
    assert st["settled"] == 1
    row = next(iter(_rows(lp).values()))
    assert row["y"] == 1 and row["roi"] == pytest.approx(1.2)


def test_blank_payload_does_not_prematurely_freeze_a_dateless_row(tmp_path):
    """NHL rows carry no start time. A payload that briefly serves an empty schedule
    must not close a row on game day; only a long-stale date force-grades it."""
    lp = tmp_path / "ledger.json"
    game = {"id": 7, "d": "2026-11-02", "home": "COL", "away": "DAL", "hp": 0.55,
            "hs": None, "as": None,
            "value": {"side": "home", "team": "COL", "ev_open": 0.21, "ev_cur": 0.02,
                      "open_dec": 2.3, "cur_dec": 2.2, "available": True, "books": 7}}
    p = _write(tmp_path, "nhl.json", {"schedule": [game]})
    gameday = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)
    EL.update("nhl", p, lp, now=gameday)
    _write(tmp_path, "nhl.json", {"schedule": []})              # blank serve
    assert EL.update("nhl", p, lp, now=gameday)["closed"] == 0
    assert next(iter(_rows(lp).values()))["dec_close"] is None
    st = EL.update("nhl", p, lp, now=gameday + timedelta(days=4))
    assert st["closed"] == 1 and st["settled"] == 0             # closed, result unknown


# ------------------------------------------------------------------ no-ops

def test_missing_payload_is_a_clean_noop(tmp_path):
    lp = tmp_path / "ledger.json"
    st = EL.update("mlb", tmp_path / "nope.json", lp, now=NOW)
    assert st == {"league": "mlb", "recorded": 0, "touched": 0, "closed": 0,
                  "settled": 0, "payload": False}
    assert _rows(lp) == {}                       # ledger still written, still valid


def test_empty_and_corrupt_payloads_are_noops(tmp_path):
    lp = tmp_path / "ledger.json"
    p = _write(tmp_path, "nfl.json", {"schedule": []})
    assert EL.update("nfl", p, lp, now=NOW)["recorded"] == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert EL.update("nhl", bad, lp, now=NOW)["payload"] is False


def test_offseason_payload_with_no_edges_records_nothing(tmp_path):
    """Every game played, no badges anywhere (today's NFL/NHL state)."""
    lp = tmp_path / "ledger.json"
    p = _write(tmp_path, "nhl.json", {"schedule": [
        {"id": 1, "d": "2026-04-16", "home": "WPG", "away": "SJS",
         "hp": 0.69, "hs": 1, "as": 6}]})
    st = EL.update("nhl", p, lp, now=NOW)
    assert (st["recorded"], st["touched"], st["closed"], st["settled"]) == (0, 0, 0, 0)
    assert _rows(lp) == {}


def test_a_corrupt_ledger_is_quarantined_not_silently_wiped(tmp_path):
    """A ledger committed every 20 minutes and rebased will eventually be unparseable.
    Returning an empty ledger and letting the same cycle os.replace it over the file
    destroys the only copy of the CLV history — silently, and the workflow commits it."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.10)]))
    EL.update("mlb", bp, lp, now=NOW)
    assert len(_rows(lp)) == 1
    lp.write_text("<<<<<<< HEAD\n{}\n=======", encoding="utf-8")   # merge conflict
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(minutes=20))
    assert st["recorded"] == 1                       # continues on a fresh ledger
    saved = sorted(tmp_path.glob("ledger.json.corrupt-*"))
    assert len(saved) == 1                           # ...with the old one preserved
    assert "<<<<<<<" in saved[0].read_text(encoding="utf-8")


def test_an_unreadable_ledger_raises_instead_of_returning_empty(tmp_path):
    """Distinguish 'file absent' (a fresh ledger) from 'file unreadable' (an AV or
    editor lock, a permission blip): the second must never lead to a save() over it."""
    lp = tmp_path / "ledger.json"
    lp.write_text(json.dumps({"v": 1, "rows": {"a": {}}}), encoding="utf-8")
    real = EL.Path.read_text

    def boom(self, *a, **kw):
        if self.name == "ledger.json":
            raise PermissionError(13, "locked")
        return real(self, *a, **kw)

    EL.Path.read_text = boom
    try:
        with pytest.raises(RuntimeError):
            EL.load(lp)
    finally:
        EL.Path.read_text = real
    assert json.loads(lp.read_text(encoding="utf-8"))["rows"]     # untouched


def test_mlb_settles_from_the_tracked_pred_ledger(tmp_path):
    """data/season_2026_finals.json is gitignored and is never built by the odds
    workflow, so MLB rows would CLV-grade and then sit at y=null forever, with ROI and
    z printed as a flat 0.0 that reads like a measurement. The tracked pred ledger is
    the settlement source that exists in that checkout."""
    bp, lp = _paths(tmp_path)
    _write(tmp_path, "board.json", _board([_mlb_game(cur=2.50, mkt=(2.50, 1.60))]))
    EL.update("mlb", bp, lp, now=NOW)
    _write(tmp_path, "board.json", _board([]))            # board is forward-only
    pl = _write(tmp_path, "pred.json",
                {"1": {"d": "2026-07-31", "p": 0.60, "y": 1, "hs": 5, "as": 2}})
    st = EL.update("mlb", bp, lp, now=NOW + timedelta(hours=14),
                   results_path=tmp_path / "gitignored_finals.json",   # absent, as in CI
                   pred_ledger_path=pl)
    assert st["closed"] == 1 and st["settled"] == 1
    row = next(iter(_rows(lp).values()))
    assert row["y"] == 1 and row["roi"] == pytest.approx(1.50)


def test_update_all_shares_one_ledger_and_writes_once(tmp_path):
    lp = tmp_path / "ledger.json"
    bp = _write(tmp_path, "board.json", _board([_mlb_game(cur=2.1)]))
    np_ = _write(tmp_path, "nfl.json", {"schedule": []})
    hp = _write(tmp_path, "nhl.json", {"schedule": []})
    fin = _write(tmp_path, "finals.json", [])
    out = EL.update_all(lp, now=NOW, payloads={"mlb": bp, "nfl": np_, "nhl": hp},
                        results_path=fin)
    assert [s["league"] for s in out] == ["mlb", "nfl", "nhl"]
    assert sum(s["recorded"] for s in out) == 1
    obj = json.loads(lp.read_text(encoding="utf-8"))
    assert obj["v"] == 1 and obj["updated"] and len(obj["rows"]) == 1


# ------------------------------------------------------------------ report math

def _row(league, p, dec, clv, y, n_obs=2):
    return {"league": league, "p_model": p, "dec_at_record": dec,
            "imp_at_record": 1 / dec, "clv_pts": clv, "n_obs": n_obs,
            "y": y, "roi": (dec - 1) if y == 1 else -1.0}


def test_report_math_on_a_hand_computed_fixture(tmp_path):
    lp = tmp_path / "ledger.json"
    rows = {
        "a": _row("mlb", 0.60, 2.00, +0.05, 1),
        "b": _row("mlb", 0.50, 2.50, -0.03, 0),
        "c": _row("mlb", 0.40, 3.00, +0.01, 1),
        "d": _row("nfl", 0.55, 2.20, None, None, n_obs=1),      # recorded only
    }
    lp.write_text(json.dumps({"v": 1, "updated": "x", "rows": rows}), encoding="utf-8")
    rep = EL.report(lp)
    m = rep["leagues"]["mlb"]
    assert m["n_recorded"] == 3 and m["n_clv_graded"] == 3 and m["n_settled"] == 3
    assert m["avg_clv_pts"] == pytest.approx((0.05 - 0.03 + 0.01) / 3, abs=1e-6)
    assert m["pct_positive_clv"] == pytest.approx(2 / 3, abs=1e-4)
    # flat stake: +1.00, -1.00, +2.00 over 3 bets
    assert m["roi"] == pytest.approx((1.00 - 1.00 + 2.00) / 3, abs=1e-5)
    exp = 0.60 + 0.50 + 0.40
    var = 0.6 * 0.4 + 0.5 * 0.5 + 0.4 * 0.6
    assert m["expected_wins"] == pytest.approx(exp, abs=1e-2)
    assert m["realized_wins"] == 2
    assert m["z"] == pytest.approx((2 - exp) / var ** 0.5, abs=1e-3)
    assert m["gate"] == "PENDING" and m["gate_needs"] == EL.GATE_MIN_GRADED - 3

    n = rep["leagues"]["nfl"]
    assert n["n_recorded"] == 1 and n["n_clv_graded"] == 0 and n["n_settled"] == 0
    assert n["avg_clv_pts"] == 0.0 and n["roi"] == 0.0 and n["gate"] == "PENDING"
    assert rep["leagues"]["nhl"]["n_recorded"] == 0
    assert rep["overall"]["n_recorded"] == 4 and rep["overall"]["n_settled"] == 3
    assert EL.format_report(rep)                      # renders without raising


def test_units_accounting_in_report(tmp_path):
    """Flat 1u per settled bet: units_net = sum(dec-1 | -1) and roi is that same
    quantity per unit staked, so the two can never disagree."""
    lp = tmp_path / "ledger.json"
    rows = {
        "a": _row("mlb", 0.60, 2.50, +0.01, 1),   # +1.50u
        "b": _row("mlb", 0.55, 2.00, -0.01, 0),   # -1.00u
        "c": _row("nfl", 0.50, 1.80, +0.02, 1),   # +0.80u
        "d": _row("nfl", 0.55, 2.20, None, None, n_obs=1),   # unsettled: no stake
    }
    lp.write_text(json.dumps({"v": 1, "rows": rows}), encoding="utf-8")
    rep = EL.report(lp)
    m = rep["leagues"]["mlb"]
    assert m["units_staked"] == 2.0
    assert m["units_net"] == pytest.approx(0.50)
    assert m["roi"] == pytest.approx(m["units_net"] / m["units_staked"], abs=1e-5)
    ov = rep["overall"]
    assert ov["units_staked"] == 3.0 and ov["units_net"] == pytest.approx(1.30)
    assert "u " in EL.format_report(rep) or "units" in EL.format_report(rep)


def test_single_observation_rows_are_excluded_from_clv(tmp_path):
    """A row recorded once and never seen again pre-game has a structural clv of 0;
    counting it would dilute the gate toward a false zero."""
    lp = tmp_path / "ledger.json"
    rows = {"a": _row("mlb", 0.6, 2.0, 0.0, 1, n_obs=1),
            "b": _row("mlb", 0.6, 2.0, 0.04, 1, n_obs=3)}
    lp.write_text(json.dumps({"v": 1, "rows": rows}), encoding="utf-8")
    m = EL.report(lp)["leagues"]["mlb"]
    assert m["n_recorded"] == 2 and m["n_clv_graded"] == 1
    assert m["avg_clv_pts"] == pytest.approx(0.04)


def test_gate_status_transitions(tmp_path):
    lp = tmp_path / "ledger.json"
    for clv, want in ((+0.01, "PASS"), (-0.01, "FAIL")):
        rows = {str(i): _row("mlb", 0.6, 2.0, clv, i % 2)
                for i in range(EL.GATE_MIN_GRADED)}
        lp.write_text(json.dumps({"v": 1, "rows": rows}), encoding="utf-8")
        assert EL.report(lp)["leagues"]["mlb"]["gate"] == want


def test_gate_rejects_a_mean_that_does_not_clear_its_standard_error(tmp_path):
    """avg CLV > 0 alone is a bare sign test on a quantity whose per-bet noise dwarfs
    any plausible edge: 100 rows alternating +0.0300 / -0.0299 average +0.00005, i.e.
    0.017 standard errors from zero, and would promote a coin flip to live money."""
    lp = tmp_path / "ledger.json"
    rows = {str(i): _row("mlb", 0.6, 2.0, 0.0300 if i % 2 else -0.0299, i % 2)
            for i in range(EL.GATE_MIN_GRADED)}
    lp.write_text(json.dumps({"v": 1, "rows": rows}), encoding="utf-8")
    m = EL.report(lp)["leagues"]["mlb"]
    assert m["n_clv_graded"] == EL.GATE_MIN_GRADED and m["avg_clv_pts"] > 0
    assert abs(m["t_clv"]) < 0.1 and m["gate"] == "FAIL"


def test_censored_rows_are_excluded_from_the_gate_but_still_reported(tmp_path):
    lp = tmp_path / "ledger.json"
    good = _row("mlb", 0.6, 2.0, 0.04, 1)
    bad = {**_row("mlb", 0.6, 2.0, 0.40, 1), "clv_censored": True}
    lp.write_text(json.dumps({"v": 1, "rows": {"a": good, "b": bad}}), encoding="utf-8")
    m = EL.report(lp)["leagues"]["mlb"]
    assert m["n_clv_graded"] == 1 and m["n_clv_censored"] == 1
    assert m["avg_clv_pts"] == pytest.approx(0.04)      # the censored +0.40 is out


def test_rows_closed_long_ago_with_no_result_are_surfaced(tmp_path):
    """Postponed/void games CLV-grade and then never settle; nothing else in the
    report would show that they are quietly filling the 100-row gate."""
    lp = tmp_path / "ledger.json"
    r = {**_row("mlb", 0.6, 2.0, 0.02, None), "dec_close": 1.95,
         "ts_close": "2026-07-01T00:00:00Z"}
    r["y"] = None
    lp.write_text(json.dumps({"v": 1, "rows": {"a": r}}), encoding="utf-8")
    m = EL.report(lp, now=NOW)["leagues"]["mlb"]
    assert m["n_settled"] == 0 and m["n_stale_unsettled"] == 1


def test_z_formula_matches_ev_gate_audit(tmp_path):
    """The live z must be the SAME number the historical audit reports, so live and
    backtest telemetry are directly comparable."""
    np = pytest.importorskip("numpy")                 # noqa: F841
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(EL.PROJECT) / "phase0"))
    from ev_gate_audit import exp_vs_real, roi        # noqa: PLC0415
    bets = [{"p": 0.6, "win": True, "odds": 2.0}, {"p": 0.5, "win": False, "odds": 2.5},
            {"p": 0.4, "win": True, "odds": 3.0}]
    assert EL._exp_vs_real(bets)[3] == pytest.approx(exp_vs_real(bets)[3], abs=1e-9)
    assert EL._roi(bets) == pytest.approx(roi(bets), abs=1e-9)


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    lp = tmp_path / "ledger.json"
    EL.save({"v": 1, "rows": {}}, lp, NOW)
    assert lp.is_file() and not (tmp_path / "ledger.json.tmp").exists()
    assert json.loads(lp.read_text(encoding="utf-8"))["updated"] == "2026-07-31T12:00:00Z"


def test_close_lag_is_measured_not_assumed():
    """Every CLV figure must carry how STALE its 'close' actually was.

    Our closing price is only as fresh as the last cron that saw the game. The
    odds job is scheduled every 20 minutes, but GitHub runs schedules
    best-effort and the observed cadence has been ~1h with multi-hour gaps
    (measured 2026-08-01). A stale last observation biases CLV toward zero
    because the final market move is missed — so the promotion gate must be
    read alongside this metric, never on its own.
    """
    from market.edge_ledger import _close_lag_mins, _lag_block

    rows = [
        {"start_utc": "2026-08-01T23:10:00Z", "ts_last_pregame": "2026-08-01T23:00:00Z"},
        {"start_utc": "2026-08-01T23:10:00Z", "ts_last_pregame": "2026-08-01T21:10:00Z"},
        {"start_utc": "2026-08-01T23:10:00Z", "ts_last_pregame": "2026-08-01T22:10:00Z"},
        {"start_utc": None, "ts_last_pregame": "2026-08-01T22:10:00Z"},   # unusable
        {"start_utc": "2026-08-01T23:10:00Z", "ts_last_pregame": None},   # unusable
    ]
    assert _close_lag_mins(rows) == [10.0, 120.0, 60.0]
    b = _lag_block(_close_lag_mins(rows))
    assert b["close_lag_n"] == 3
    assert b["close_lag_median_min"] == 60.0
    assert b["close_lag_p90_min"] == 120.0
    # empty is reported as unknown, never as zero (which would read as "fresh")
    empty = _lag_block([])
    assert empty["close_lag_n"] == 0 and empty["close_lag_median_min"] is None
