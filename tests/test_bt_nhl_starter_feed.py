"""The prototype CONFIRMED-goalie feed (phase0/bt_nhl_starter_feed.py).

The nhl.com Playing Roster report is the only official source observed to name
tonight's starting goalie before puck drop (bold = starting lineup), and only
minutes before it. These tests pin the three properties a serving pipeline
cannot get wrong:

  * side orientation: the report prints visitor then home; swapping them would
    put the away starter in the home net, the worst silent failure available;
  * no bold goalie -> NOT confirmed (the report exists ~an hour before puck drop
    listing BOTH dressed goalies; treating the first listed as the starter would
    be a guess wearing the CONFIRMED label);
  * a read at/after the scheduled start, or with the game LIVE, is never a
    pre-game confirmation (played games keep their pre-game number).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import bt_nhl_starter_feed as feed  # noqa: E402


def _row(num, pos, name, bold=False):
    cls = "bold" if bold else ""
    return (f'<tr> <td align="center" width="15%" class="{cls}">{num}</td> '
            f'<td align="center" width="15%" class="{cls}">{pos}</td> '
            f'<td align="left" width="70%" class="{cls}">{name}</td> </tr>')


def _table(rows):
    head = ('<table><tr> <td width="15%" class="heading + bborder" align="center">#</td> '
            '<td width="15%" class="heading + bborder" align="center">Pos</td> '
            '<td width="70%" class="heading + bborder" align="left">Name</td> </tr>')
    return head + "".join(_row(*r) for r in rows) + "</table>"


def report(away, home, away_scr=(), home_scr=(), start="Start&nbsp;:&nbsp;EST",
           status="Period 1 (20:00 Remaining)", gen="2024-12-03-18.50.39"):
    parts = [
        "<html><style>.bold{font-weight:bold;}</style><body><table>",
        '<tr><td align="center" style="font-size: 10px;font-weight:bold">Tuesday, December 3, 2024</td></tr>',
        f'<tr><td align="center" style="font-size: 10px;font-weight:bold">{start}</td></tr>',
        f'<tr><td align="center" style="font-size: 10px;font-weight:bold">{status}</td></tr>',
        '<tr><td class="teamHeading + border ">DETROIT RED WINGS</td>'
        '<td class="teamHeading + border">BOSTON BRUINS</td></tr>',
        "<tr><td>", _table(away), "</td><td>", _table(home), "</td></tr>",
    ]
    if away_scr or home_scr:
        parts += ['<tr><td colspan="2" align="center" class="header">Scratches</td></tr>',
                  "<tr><td>", _table(away_scr), "</td><td>", _table(home_scr), "</td></tr>"]
    parts += ['<tr><td colspan="2" align="center" class="header">Head Coaches</td></tr>',
              "</table>",
              f"<table><tr><td>&#169; Copyright 2024, National Hockey League &nbsp;{gen}</td></tr>"
              "</table></body></html>"]
    return "".join(parts).encode("utf-8")


AWAY = [("2", "D", "BEN CHIAROT", True), ("35", "G", "VILLE HUSSO", True),
        ("39", "G", "CAM TALBOT")]
HOME = [("1", "G", "JEREMY SWAYMAN"), ("70", "G", "JOONAS KORPISALO", True),
        ("88", "R", "DAVID PASTRNAK")]


def test_parse_orients_visitor_then_home_and_reads_bold():
    p = feed.parse_playing_roster(report(AWAY, HOME, [("24", "L", "X Y")], [("4", "D", "Z W")]))
    assert p["n_tables"] == 4 and not p["started"] and not p["final"]
    assert p["gen_local"] == "2024-12-03-18.50.39"
    s = feed.starters_from_report(p)
    assert s["away"]["starter"] == ["35", "VILLE HUSSO"]
    assert s["home"]["starter"] == ["70", "JOONAS KORPISALO"]
    assert s["away"]["dressed_goalies"] == [["35", "VILLE HUSSO"], ["39", "CAM TALBOT"]]
    assert p["away"]["scratches"][0]["name"] == "X Y"
    assert p["home"]["scratches"][0]["name"] == "Z W"


def test_no_bold_goalie_is_not_confirmed():
    plain = [(n, pos, name) for n, pos, name, *_ in AWAY]
    p = feed.parse_playing_roster(report(plain, HOME))
    s = feed.starters_from_report(p)
    assert s["away"]["starter"] is None and s["away"]["confirmed"] is False
    assert s["away"]["n_bold_goalies"] == 0
    assert s["home"]["confirmed"] is True


def test_two_bold_goalies_is_ambiguous_not_confirmed():
    both = [("35", "G", "VILLE HUSSO", True), ("39", "G", "CAM TALBOT", True)]
    s = feed.starters_from_report(feed.parse_playing_roster(report(both, HOME)))
    assert s["away"]["n_bold_goalies"] == 2 and s["away"]["confirmed"] is False


def test_started_report_is_flagged():
    p = feed.parse_playing_roster(report(AWAY, HOME, start="Start&nbsp;7:08&nbsp;EST",
                                         status="Period 1 (12:00 Remaining)"))
    assert p["started"] is True


def test_player_id_by_team_and_sweater_then_name():
    spots = [{"teamId": 17, "playerId": 111, "sweaterNumber": 35, "positionCode": "G",
              "firstName": {"default": "Ville"}, "lastName": {"default": "Husso"}},
             {"teamId": 6, "playerId": 222, "sweaterNumber": 71, "positionCode": "G",
              "firstName": {"default": "Joonas"}, "lastName": {"default": "Korpisalo"}}]
    s = feed.starters_from_report(feed.parse_playing_roster(report(AWAY, HOME)),
                                  roster_spots=spots, team_ids={"away": 17, "home": 6})
    assert s["away"]["starter_id"] == 111           # sweater match
    assert s["home"]["starter_id"] == 222           # sweater mismatch -> name match


def _fake_get(state, start_utc, ro_bytes):
    pbp = {"gameState": state, "startTimeUTC": start_utc,
           "homeTeam": {"id": 6}, "awayTeam": {"id": 17}, "rosterSpots": []}

    def get(url, timeout=20):
        if "play-by-play" in url:
            return 200, json.dumps(pbp).encode()
        return (200, ro_bytes) if ro_bytes is not None else (404, None)
    return get


def test_fetch_confirmed_only_before_the_scheduled_start(monkeypatch):
    start = "2024-12-04T00:00:00Z"
    monkeypatch.setattr(feed, "_get", _fake_get("PRE", start, report(AWAY, HOME)))
    before = dt.datetime(2024, 12, 3, 23, 51, tzinfo=dt.timezone.utc)
    r = feed.fetch_confirmed(2024020396, now_utc=before)
    assert r["pre_game"] and r["away"]["confirmed"] and r["home"]["confirmed"]
    assert r["min_to_start"] == 9.0

    after = dt.datetime(2024, 12, 4, 0, 1, tzinfo=dt.timezone.utc)
    r = feed.fetch_confirmed(2024020396, now_utc=after)
    assert not r["pre_game"] and not r["away"]["confirmed"] and not r["home"]["confirmed"]


def test_fetch_confirmed_live_state_is_never_pre_game(monkeypatch):
    monkeypatch.setattr(feed, "_get", _fake_get("LIVE", "2024-12-04T00:00:00Z",
                                                report(AWAY, HOME)))
    r = feed.fetch_confirmed(2024020396,
                             now_utc=dt.datetime(2024, 12, 3, 23, 55, tzinfo=dt.timezone.utc))
    assert not r["pre_game"] and not r["home"]["confirmed"]


def test_fetch_confirmed_report_not_posted(monkeypatch):
    monkeypatch.setattr(feed, "_get", _fake_get("FUT", "2024-12-04T00:00:00Z", None))
    r = feed.fetch_confirmed(2024020396,
                             now_utc=dt.datetime(2024, 12, 3, 20, 0, tzinfo=dt.timezone.utc))
    assert r["ro_status"] == 404 and r["away"] is None and r["pre_game"]
