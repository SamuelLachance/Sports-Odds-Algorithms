"""nhl-shift-repair: the puller's HTML fallback and the backfill merge path.

Offline: every network call is monkeypatched. Pins that
  * an API-empty future game is filled from the reports in ONE block, logged as
    HTML provenance, and a not-yet-posted / live report writes nothing (retried
    next run, never a partial or empty-marker game);
  * games below the fallback season floor (the historical gaps, which go through
    the separate backfill + merge) are never filled by the puller;
  * --no-html-fallback is the old API-only behaviour;
  * merge is a dry run by default, inserts in spine order, skips games already
    present, keeps every existing row, and logs what it merged.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nhl_pull_shifts as P  # noqa: E402
import nhl_shifts_html as H  # noqa: E402

HDR = ["game_id", "player_id", "team", "period", "start_s", "end_s"]


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _spine(tmp, gids):
    _write(tmp / "data" / "nhl_games.csv", ["game_id", "date", "season"],
           [[g, "2026-10-10", H.season_of(g)] for g in gids])


def _setup_puller(tmp_path, monkeypatch, api, html):
    """api: gid -> list of API shift dicts; html: gid -> (rows, status)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    monkeypatch.setattr(P, "get", lambda url: {"data": api[int(url.rsplit("=", 1)[1])]})
    calls = []

    def fake_fallback(gid, throttle):
        calls.append(gid)
        rows, status = html[gid]
        qa = {"reasons": ["x"]} if status == "qa_fail" else None
        return rows, status, qa

    monkeypatch.setattr(P.html_fb, "fallback_rows", fake_fallback)
    return calls


API_SHIFT = {"playerId": 8470000, "teamAbbrev": "SEA", "period": 1,
             "startTime": "00:11", "endTime": "00:42"}


def test_puller_fills_api_empty_future_game_from_html(tmp_path, monkeypatch):
    g_api, g_html, g_wait, g_live, g_old = 2026020001, 2026020002, 2026020003, 2026020004, 2025021000
    _spine(tmp_path, [g_old, g_api, g_html, g_wait, g_live])
    api = {g_api: [API_SHIFT], g_html: [], g_wait: [], g_live: [], g_old: []}
    html = {g_html: ([[g_html, 8470001, "OTT", 1, 0, 40], [g_html, 8470001, "OTT", 2, 5, 5]], "ok"),
            g_wait: ([], "not_posted"), g_live: ([], "not_final"),
            g_old: ([[g_old, 1, "X", 1, 0, 1]], "ok")}
    calls = _setup_puller(tmp_path, monkeypatch, api, html)
    P.main([])
    rows = _read(tmp_path / "data" / "nhl_shifts.csv")
    assert rows[0] == HDR
    body = [[int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])] for r in rows[1:]]
    assert body == [[g_api, 8470000, "SEA", 1, 11, 42]] + html[g_html][0]
    # the historical gap (season below the floor) is never sent to the fallback
    assert calls == [g_html, g_wait, g_live]
    filled = _read(tmp_path / "data" / "nhl_shifts_html_filled.csv")
    assert filled[0] == ["gid", "rows", "source", "filled_utc"]
    assert [r[:3] for r in filled[1:]] == [[str(g_html), "2", "html_toi_report"]]
    pend = json.load(open(tmp_path / "data" / "nhl_shifts_html_pending.json"))["pending"]
    assert pend == {str(g_wait): "not_posted", str(g_live): "not_final"}

    # second run: the filled game is done; the waiting ones are retried and now post
    html[g_wait] = ([[g_wait, 8470002, "SEA", 3, 100, 160]], "ok")
    calls.clear()
    P.main([])
    assert calls == [g_wait, g_live]
    rows = _read(tmp_path / "data" / "nhl_shifts.csv")
    assert rows.count(HDR) == 1
    assert [str(g_wait), "8470002", "SEA", "3", "100", "160"] in rows
    pend = json.load(open(tmp_path / "data" / "nhl_shifts_html_pending.json"))["pending"]
    assert pend == {str(g_live): "not_final"}


def test_puller_api_only_mode(tmp_path, monkeypatch):
    g = 2026020001
    _spine(tmp_path, [g])
    calls = _setup_puller(tmp_path, monkeypatch, {g: []},
                          {g: ([[g, 5, "SEA", 1, 0, 30]], "ok")})
    P.main(["--no-html-fallback"])
    assert calls == []
    assert _read(tmp_path / "data" / "nhl_shifts.csv") == [HDR]
    assert not (tmp_path / "data" / "nhl_shifts_html_pending.json").exists()
    assert not (tmp_path / "data" / "nhl_shifts_html_filled.csv").exists()


def test_puller_min_season_flag_opens_older_seasons(tmp_path, monkeypatch):
    g_old = 2025021000
    _spine(tmp_path, [g_old])
    calls = _setup_puller(tmp_path, monkeypatch, {g_old: []},
                          {g_old: ([[g_old, 5, "SEA", 1, 0, 30]], "ok")})
    P.main(["--html-min-season", "20252026"])
    assert calls == [g_old]
    assert _read(tmp_path / "data" / "nhl_shifts.csv") == \
        [HDR, [str(g_old), "5", "SEA", "1", "0", "30"]]


def test_puller_qa_failure_writes_nothing(tmp_path, monkeypatch):
    g = 2026020010
    _spine(tmp_path, [g])
    _setup_puller(tmp_path, monkeypatch, {g: []}, {g: ([], "qa_fail")})
    P.main([])
    assert _read(tmp_path / "data" / "nhl_shifts.csv") == [HDR]
    pend = json.load(open(tmp_path / "data" / "nhl_shifts_html_pending.json"))["pending"]
    assert pend[str(g)].startswith("qa_fail")


def test_fallback_rows_statuses(monkeypatch):
    from test_nhl_shift_repair_parser import AWAY_PLAYERS, HOME_PLAYERS, make_pbp, make_report

    reports = {}

    def fake_get_report(gid, kind, throttle, use_cache=True, write_cache=True):
        v = reports[kind]
        return (200, v) if isinstance(v, str) and v.startswith("<html") else (v, None)

    monkeypatch.setattr(H, "get_report", fake_get_report)
    monkeypatch.setattr(H, "load_pbp", lambda gid, throttle=None, fetch=True: make_pbp())
    thr = H.Throttle(0)
    reports.update(TH=404, TV=404)
    assert H.fallback_rows(2026020001, thr)[1] == "not_posted"
    reports.update(TH=make_report(players=HOME_PLAYERS), TV="http503")
    assert H.fallback_rows(2026020001, thr)[1].startswith("error")
    reports.update(TV=make_report(side="Away", status="End of 3rd Period", players=AWAY_PLAYERS))
    assert H.fallback_rows(2026020001, thr)[1] == "not_final"
    reports.update(TV=make_report(side="Away", players=AWAY_PLAYERS))
    rows, status, qa = H.fallback_rows(2026020001, thr)
    assert status == "ok" and qa["ok"] and rows and all(r[0] == 2026020001 for r in rows)
    monkeypatch.setattr(H, "load_pbp", lambda gid, throttle=None, fetch=True: None)
    assert H.fallback_rows(2026020001, thr)[1] == "no_pbp"

    def boom(*a, **k):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(H, "get_report", boom)
    assert H.fallback_rows(2026020001, thr)[1].startswith("error:RuntimeError")


def test_merge_dry_run_then_apply_in_spine_order(tmp_path):
    spine = [101, 102, 103, 104, 105]
    idx = {g: i for i, g in enumerate(spine)}
    shifts = tmp_path / "shifts.csv"
    _write(shifts, HDR, [[101, 1, "A", 1, 0, 10], [101, 2, "A", 1, 10, 20],
                         [103, 3, "B", 1, 0, 30], [105, 4, "C", 2, 5, 9]])
    bf = tmp_path / "bf.csv"
    _write(bf, HDR, [[102, 7, "A", 1, 0, 40], [102, 7, "A", 1, 12, 12],
                     [104, 8, "B", 3, 0, 50], [103, 9, "B", 1, 0, 1]])   # 103 already present
    before = shifts.read_bytes()
    rep = H.merge(str(shifts), str(bf), idx, apply=False)
    assert rep["applied"] is False and shifts.read_bytes() == before
    assert rep["games_to_insert"] == 2 and rep["rows_to_insert"] == 3
    assert rep["skip_already_present"] == 1 and rep["skip_gids"] == [103]
    log = tmp_path / "merged.csv"
    rep = H.merge(str(shifts), str(bf), idx, apply=True, merged_log=str(log))
    assert rep["applied"] and rep["shifts_rows_after"] == 4 + 3
    got = _read(shifts)
    assert got[0] == HDR
    assert [r[0] for r in got[1:]] == ["101", "101", "102", "102", "103", "104", "105"]
    assert ["103", "9", "B", "1", "0", "1"] not in got
    assert [r[0] for r in _read(log)[1:]] == ["102", "104"]
    # line endings preserved byte-for-byte (the real file is csv.writer CRLF)
    raw = shifts.read_bytes()
    assert raw.count(b"\r\n") == 8 and raw.count(b"\n") == 8
    assert before in raw.replace(b"102,7,A,1,0,40\r\n102,7,A,1,12,12\r\n", b"") \
        .replace(b"104,8,B,3,0,50\r\n", b"")
    # idempotent: a second apply finds everything present
    rep = H.merge(str(shifts), str(bf), idx, apply=True, merged_log=str(log))
    assert rep["games_to_insert"] == 0 and rep["applied"] is False
    assert len(_read(shifts)) == 8


def test_merge_refuses_a_foreign_schema(tmp_path):
    shifts = tmp_path / "shifts.csv"
    _write(shifts, HDR, [[101, 1, "A", 1, 0, 10]])
    bf = tmp_path / "bf.csv"
    _write(bf, ["gid", "pid"], [[102, 7]])
    try:
        H.merge(str(shifts), str(bf), {101: 0, 102: 1}, apply=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("merge accepted a backfill with the wrong header")
    assert len(_read(shifts)) == 2
