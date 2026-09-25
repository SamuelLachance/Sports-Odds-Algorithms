"""nhl-shift-repair: the HTML time-on-ice report parser (phase0/nhl_shifts_html.py).

Pins the failure classes that would silently corrupt per-60 rates if the HTML
fallback drifted from the shift API's rows: wrong period for overtime, shootout
rows counted as shifts, a live (non-final) report accepted, a sweater number
mapped to the wrong player, goal rows missing or duplicated, summary drift.
The fixtures reproduce the real report markup (TH/TV{gg}.HTM, 2010-2026 format).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nhl_shifts_html as H  # noqa: E402


def _mmss(s):
    return f"{s // 60}:{s % 60:02d}"


def _dur(s):
    return f"{s // 60:02d}:{s % 60:02d}"


def make_report(side="Home", status="Final", team="SEATTLE KRAKEN", players=(),
                bad_summary=False, plen=1200):
    """players: [(num, 'LAST, FIRST', [(per_label, start_s, end_s, event)])]."""
    out = [f"<html><head><title>Time On Ice Report {side} Team</title></head>",
           "<style>.playerHeading{font-weight:bold;}</style>",
           "<body><script language=\"javascript\">var x = '<td>junk</td>';</script>",
           "<table><tr><td align=\"center\" style=\"font-size: 10px\">Game 1000</td></tr>",
           f"<tr><td align=\"center\" style=\"font-size: 10px\">{status}</td></tr></table>",
           "<table><tr><td class=\"teamHeading + border\" align=\"center\">"
           f"{team}</td></tr></table><table>"]
    for num, name, shifts in players:
        out.append("<tr><td align=\"center\" valign=\"top\" class=\"playerHeading + border\" "
                   f"colspan=\"8\">{num} {name}</td></tr>")
        out.append("<tr><td class=\"heading + lborder + bborder\">Shift #</td><td>Per</td>"
                   "<td>Start of Shift<br>Elapsed / Game</td><td>End of Shift<br>Elapsed / Game"
                   "</td><td>Duration</td><td class=\"heading\">Event<br><span class=\"note\">"
                   "G=Goal<br>P=Penalty</span></td></tr>")
        per_n, per_t = {}, {}
        for i, (per, st, en, ev) in enumerate(shifts, 1):
            cls = "oddColor" if i % 2 else "\tevenColor"
            out.append(f"<tr class=\"{cls}\"><td align=\"center\" class=\"lborder + bborder\">{i}"
                       f"</td><td align=\"center\">{per}</td>"
                       f"<td align=\"center\">{_mmss(st)} / {_mmss(plen - st)}</td>"
                       f"<td align=\"center\">{_mmss(en)} / {_mmss(plen - en)}</td>"
                       f"<td align=\"center\">{_dur(en - st)}</td>"
                       f"<td align=\"center\">{ev or '&nbsp;'}</td></tr>")
            per_n[per] = per_n.get(per, 0) + 1
            per_t[per] = per_t.get(per, 0) + (en - st)
        out.append("<tr><td><table><tr><td class=\"heading\">Per</td><td>SHF</td><td>AVG</td>"
                   "<td>TOI</td><td>EV TOT</td><td>PP TOT</td><td>SH TOT</td></tr>")
        for per in per_n:
            t = per_t[per] + (7 if bad_summary else 0)
            out.append(f"<tr class=\"oddColor\"><td>{per}</td><td>{per_n[per]}</td><td>00:40</td>"
                       f"<td>{_dur(t)}</td><td>{_dur(t)}</td><td>00:00</td><td>00:00</td></tr>")
        tt = sum(per_t.values())
        out.append(f"<tr><td class=\"bold\">TOT</td><td>{sum(per_n.values())}</td><td>00:40</td>"
                   f"<td>{_dur(tt)}</td><td>{_dur(tt)}</td><td>00:00</td><td>00:00</td></tr>")
        out.append("</table></td></tr>")
    out.append("</table></body></html>")
    return "\n".join(out)


HOME_PLAYERS = [
    (6, "LARSSON, ADAM", [("1", 11, 42, ""), ("2", 94, 124, "G"), ("OT", 0, 45, "")]),
    (35, "GRUBAUER, PHILIPP", [("1", 0, 1200, ""), ("2", 0, 1200, ""), ("3", 0, 1200, ""),
                               ("OT", 0, 190, "")]),
]
AWAY_PLAYERS = [
    (7, "BRADY TKACHUK, B", [("1", 0, 40, ""), ("3", 100, 160, "P")]),
    (18, "STÜTZLE, TIM", [("1", 40, 95, "")]),
]


def make_pbp(extra_goals=()):
    goals = [
        {"typeDescKey": "goal", "timeInPeriod": "01:34",
         "periodDescriptor": {"number": 2, "periodType": "REG"},
         "details": {"scoringPlayerId": 1006, "eventOwnerTeamId": 55}},
        {"typeDescKey": "goal", "timeInPeriod": "00:00",
         "periodDescriptor": {"number": 5, "periodType": "SO"},
         "details": {"scoringPlayerId": 2018, "eventOwnerTeamId": 9}},
    ] + list(extra_goals)
    return {
        "homeTeam": {"id": 55, "abbrev": "SEA"},
        "awayTeam": {"id": 9, "abbrev": "OTT"},
        "rosterSpots": [
            {"teamId": 55, "playerId": 1006, "sweaterNumber": 6, "lastName": {"default": "Larsson"}},
            {"teamId": 55, "playerId": 1035, "sweaterNumber": 35, "lastName": {"default": "Grubauer"}},
            {"teamId": 55, "playerId": 1031, "sweaterNumber": 31, "lastName": {"default": "Daccord"}},
            {"teamId": 9, "playerId": 2007, "sweaterNumber": 7, "lastName": {"default": "Tkachuk"}},
            {"teamId": 9, "playerId": 2018, "sweaterNumber": 18, "lastName": {"default": "Stützle"}},
        ],
        "plays": goals + [{"typeDescKey": "shot-on-goal"}],
    }


def test_parse_report_reads_every_shift_and_the_summary():
    rep = H.parse_report(make_report(players=HOME_PLAYERS))
    assert rep["final"] and rep["status"] == "Final"
    assert rep["team_name"] == "SEATTLE KRAKEN"
    assert "Home" in rep["title"]
    assert [p["num"] for p in rep["players"]] == [6, 35]
    lars = rep["players"][0]
    assert lars["name"] == "LARSSON, ADAM"
    # regular-season "OT" is period 4; start/end are the ELAPSED halves
    assert lars["shifts"] == [(1, 11, 42, 31, ""), (2, 94, 124, 30, "G"), (4, 0, 45, 45, "")]
    assert lars["summary"] == {1: (1, 31), 2: (1, 30), 4: (1, 45)}
    assert lars["tot"] == (3, 106)
    assert H.report_qa(rep) == {}


def test_parse_report_bytes_and_playoff_numeric_overtimes():
    pl = [(9, "STEPHENSON, CHANDLER", [("4", 0, 60, ""), ("5", 30, 90, ""), ("6", 5, 70, "G")])]
    rep = H.parse_report(make_report(players=pl).encode("utf-8"))
    assert [s[0] for s in rep["players"][0]["shifts"]] == [4, 5, 6]


def test_parse_report_flags_summary_drift_and_skips_shootout():
    pl = [(6, "LARSSON, ADAM", [("1", 11, 42, ""), ("SO", 0, 0, "")])]
    rep = H.parse_report(make_report(players=pl, bad_summary=True))
    assert rep["players"][0]["shifts"] == [(1, 11, 42, 31, "")]   # SO is never a shift
    assert rep["players"][0]["bad_rows"] == []
    q = H.report_qa(rep)
    assert q.get("summary_toi_mismatch", 0) >= 1


def test_bilingual_latin1_report_header_is_read():
    """Older reports from Canadian arenas are bilingual ("Match/Game 0570") and
    Latin-1 encoded; the status cell must still be found or the fallback would
    wait forever on a finished game."""
    txt = make_report(players=[(2, "COWEN, JARED", [("1", 155, 187, "")])])
    txt = txt.replace("Game 1000", "Match/Game 0570").replace(
        "Shift #", "Présence #/Shift #")
    rep = H.parse_report(txt.encode("cp1252"))
    assert rep["status"] == "Final" and rep["final"]
    assert rep["players"][0]["shifts"] == [(1, 155, 187, 32, "")]
    assert H.decode_report("Présence".encode("cp1252")) == "Présence"
    assert H.decode_report("Stützle".encode("utf-8")) == "Stützle"


def test_live_report_is_not_final():
    rep = H.parse_report(make_report(status="End of 2nd Period", players=HOME_PLAYERS))
    assert not rep["final"]


def test_period_label_and_mmss():
    assert H.period_label("OT") == 4 and H.period_label(" 3 ") == 3
    assert H.period_label("SO") is None and H.period_label("") is None
    assert H.mmss("12:34") == 754 and H.mmss("0:00") == 0 and H.mmss("x") is None


def test_report_url_and_season():
    assert H.season_of(2025021000) == 20252026
    assert H.report_url(2025021000, "TH") == \
        "https://www.nhl.com/scores/htmlreports/20252026/TH021000.HTM"
    assert H.report_url(2010030244, "TV").endswith("/20102011/TV030244.HTM")


def test_build_game_maps_ids_teams_and_recreates_goal_rows():
    rows, qa = H.build_game(2025021000, make_report(players=HOME_PLAYERS),
                            make_report(side="Away", team="OTTAWA SENATORS",
                                        players=AWAY_PLAYERS), make_pbp())
    assert qa["ok"], qa
    assert all(len(r) == len(H.SCHEMA) for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    assert set(by) == {1006, 1035, 2007, 2018}
    assert {r[2] for r in by[1006]} == {"SEA"} and {r[2] for r in by[2018]} == {"OTT"}
    # the API's zero-length goal row: scorer, scoring team, period, goal time;
    # a shootout goal is stored by the API as period 5 at 00:00 and re-created so;
    # goal rows sort first within the player
    assert by[1006][0] == [2025021000, 1006, "SEA", 2, 94, 94]
    assert by[2018][0] == [2025021000, 2018, "OTT", 5, 0, 0]
    assert qa["goal_rows"] == 2
    assert sum(1 for r in rows if r[4] == r[5]) == 2
    assert by[1035][-1] == [2025021000, 1035, "SEA", 4, 0, 190]
    # rows sorted by player id (the API order)
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    assert qa["TH_players"] == 2 and qa["TV_players"] == 2


def test_build_game_rejects_unmapped_player_and_swapped_reports():
    pl = AWAY_PLAYERS + [(99, "NOBODY, JOHN", [("1", 0, 30, "")])]
    _rows, qa = H.build_game(1, make_report(players=HOME_PLAYERS),
                             make_report(side="Away", players=pl), make_pbp())
    assert not qa["ok"] and any("unmapped" in r for r in qa["reasons"])
    # home report passed as away -> title check fails
    _rows, qa = H.build_game(1, make_report(players=HOME_PLAYERS),
                             make_report(side="Home", players=AWAY_PLAYERS), make_pbp())
    assert not qa["ok"] and any("title" in r for r in qa["reasons"])
    _rows, qa = H.build_game(1, make_report(status="3rd Period", players=HOME_PLAYERS),
                             make_report(side="Away", players=AWAY_PLAYERS), make_pbp())
    assert not qa["ok"] and any("not_final" in r for r in qa["reasons"])


def test_build_game_maps_by_unique_last_name_when_number_is_wrong():
    pl = [(77, "STÜTZLE, TIM", [("1", 40, 95, "")])]     # number not on the roster
    rows, qa = H.build_game(1, make_report(players=HOME_PLAYERS),
                            make_report(side="Away", players=pl), make_pbp())
    assert qa["ok"] and qa.get("TV_mapped_by_name") == 1
    assert [1, 2018, "OTT", 1, 40, 95] in rows


def test_compare_game_counts_agreement():
    api = [[1, 10, "SEA", 1, 0, 40], [1, 10, "SEA", 1, 40, 40], [1, 11, "SEA", 1, 0, 50],
           [1, 11, "SEA", 1, 0, 50]]                      # an exact duplicate row
    same = [[1, 10, "SEA", 1, 0, 40], [1, 10, "SEA", 1, 40, 40], [1, 11, "SEA", 1, 0, 50]]
    c = H.compare_game(api, same)
    assert c["player_games"] == 2 and c["both_exact"] == 2 and c["api_dups"] == 1
    assert c["shift_rows_identical_dedup"] and not c["rows_identical"]
    off = [[1, 10, "SEA", 1, 0, 41], [1, 12, "SEA", 1, 0, 9]]
    c = H.compare_game(api, off)
    assert c["toi_exact"] == 0 and c["only_api"] == 1 and c["only_html"] == 1
    assert sorted(c["abs_dtoi"]) == [1, 9, 50]
    s = H.summarize([c])
    assert s["games"] == 1 and s["abs_dtoi_s"]["max"] == 50


def test_throttle_spaces_requests_at_most_two_per_second():
    t = {"now": 0.0}
    slept = []

    def clock():
        return t["now"]

    def sleep(dt):
        slept.append(dt)
        t["now"] += dt

    thr = H.Throttle(0.5, clock=clock, sleep=sleep)
    stamps = []
    for _ in range(5):
        thr.wait()
        stamps.append(t["now"])
        t["now"] += 0.1          # the request itself
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.5 - 1e-9
    assert H.MIN_INTERVAL >= 0.5


# ------------------------------------------------------------------- real data (local)

@pytest.mark.skipif(not os.path.exists(H.BACKFILL_CSV), reason="backfill not built locally")
def test_backfill_file_schema_and_scope():
    with open(H.BACKFILL_CSV, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        assert next(rd) == H.SCHEMA
        gids = set()
        for r in rd:
            gids.add(int(r[0]))
            assert int(r[3]) >= 1 and 0 <= int(r[4]) <= 1200 and 0 <= int(r[5]) <= 1200
    assert gids <= set(H.gap_games())
    with open(H.BACKFILL_GAMES, encoding="utf-8") as fh:
        led = list(csv.DictReader(fh))
    ok = {int(r["gid"]) for r in led if r["status"] == "ok"}
    assert ok == gids
    assert {int(r["gid"]) for r in led} == set(H.gap_games())


@pytest.mark.skipif(not os.path.exists(H.VALIDATION_JSON), reason="validation not run locally")
def test_recorded_validation_agreement():
    v = json.load(open(H.VALIDATION_JSON, encoding="utf-8"))
    s = v["sample_validation"]
    assert s["sample"]["dev"] >= 50 and s["sample"]["recent"] >= 50
    assert s["all"]["share_player_games_toi_exact"] >= 0.99
    assert s["all"]["share_player_games_shifts_exact"] >= 0.99
