"""Structural contracts between the serve payloads and the SPA.

The single-page app reads specific fields from site/data/*.json; a serve-script
refactor that drops or renames one silently blanks a league on the live site
(fetches are fault-tolerant by design). These tests pin the exact field sets the
JS consumes — build fails loudly instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / "site" / "data"


def _load(name):
    p = SITE / name
    if not p.is_file():
        pytest.skip(f"{name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def test_board_contract():
    b = _load("board.json")
    assert "leagues" in b and isinstance(b["leagues"], list) and b["leagues"]
    assert {"code", "name"} <= set(b["leagues"][0])
    acc = b.get("accuracy")
    assert acc and {"log_loss", "coinflip"} <= set(acc)      # updAcc MLB branch


def test_board_record_contract():
    """#/record's MLB section reads board.record.mlb — the ONLY place a played
    game's pre-game probability survives (the board itself is forward-only)."""
    b = _load("board.json")
    rec = b.get("record")
    assert rec and "mlb" in rec, "board.json lost the record block"
    m = rec["mlb"]
    assert {"rows", "rollup", "since", "n_pending"} <= set(m)      # recSource/recRows
    assert isinstance(m["rows"], list) and len(m["rows"]) <= 400   # payload stays lean
    for r in m["rows"]:
        assert {"d", "away", "home", "p", "pick", "y", "hs", "as"} <= set(r)
        assert 0.0 < r["p"] < 1.0 and r["y"] in (0, 1)
        assert r["pick"] == (r["home"] if r["p"] >= 0.5 else r["away"])
    if m["rows"]:
        assert m["rows"] == sorted(m["rows"], key=lambda r: r["d"], reverse=True)
        assert m["since"] and len(m["since"]) == 10
        ru = m["rollup"]
        assert ru and {"n", "log_loss", "acc", "brier"} <= set(ru)
        assert ru["n"] >= len(m["rows"])          # rollup covers every graded pick
        assert 0.0 <= ru["acc"] <= 1.0 and ru["log_loss"] > 0


def test_record_page_reads_played_games_from_nfl_and_nhl():
    """NFL/NHL track-record sections are computed client-side from the payload
    schedules: a played game must keep its frozen pre-game prob AND its score."""
    for name, prob in (("nfl.json", "ph"), ("nhl.json", "hp")):
        played = [g for g in _load(name)["schedule"]
                  if g.get("hs") is not None and g.get("as") is not None]
        for g in played:
            assert g.get(prob) is not None, f"{name}: played game without {prob}"
            assert isinstance(g["hs"], int) and isinstance(g["as"], int)


def test_record_block_grades_only_finished_games(tmp_path):
    """record_block() ships graded rows only, newest first, capped — and its
    rollup is scored over every graded row, not just the shipped ones."""
    import math

    from mlbwp.predict_slate import record_block
    ledger = {
        "1": {"d": "2026-07-30", "home": "LAD", "away": "SEA", "p": 0.6,
              "rec": "2026-07-31T00:42Z", "sp_known": True, "y": 1, "hs": 6, "as": 2},
        "2": {"d": "2026-07-31", "home": "ATH", "away": "BOS", "p": 0.4,
              "rec": "2026-08-01T00:42Z", "sp_known": False, "y": 0, "hs": 4, "as": 5},
        "3": {"d": "2026-08-01", "home": "SD", "away": "SF", "p": 0.55,
              "rec": "2026-08-01T10:00Z"},                       # pending: excluded
    }
    p = tmp_path / "led.json"
    p.write_text(json.dumps(ledger), encoding="utf-8")
    blk = record_block(p, cap=1)
    assert len(blk["rows"]) == 1 and blk["rows"][0]["d"] == "2026-07-31"   # newest first
    assert blk["rows"][0]["pick"] == "BOS"                # p < 0.5 -> the away side
    assert blk["n_pending"] == 1 and blk["since"] == "2026-07-31"
    ru = blk["rollup"]
    assert ru["n"] == 2 and ru["acc"] == 1.0              # both picks landed
    assert ru["log_loss"] == pytest.approx(-math.log(0.6), abs=1e-5)
    assert record_block(tmp_path / "missing.json") == {
        "rows": [], "rollup": None, "since": None, "n_pending": 0}


def test_nfl_contract():
    n = _load("nfl.json")
    mc = n["model_card"]
    assert {"test_log_loss", "close_log_loss"} <= set(mc)    # updAcc NFL branch
    sched = n["schedule"]
    assert sched, "empty NFL schedule"
    g = sched[0]
    assert {"w", "d", "home", "away", "ph"} <= set(g)        # nflPage/nflProb
    assert n["teams"], "empty NFL teams"
    t = next(iter(n["teams"].values()))
    assert {"elo", "rank"} <= set(t)


def test_nhl_contract():
    n = _load("nhl.json")
    mc = n["model_card"]
    assert {"test_ll", "baseline_elo_test"} <= set(mc)       # updAcc NHL branch
    sched = n["schedule"]
    assert sched, "empty NHL schedule"
    g = sched[0]
    assert {"id", "d", "home", "away", "hp"} <= set(g)       # nhlCard/nhlGamePage
    assert 0.0 < g["hp"] < 1.0
    assert len(n["teams"]) == 32
    t = next(iter(n["teams"].values()))
    assert {"elo", "xg", "pts", "rank", "w", "l", "otl"} <= set(t)  # nhlTeams/standings
    assert n["players"], "empty NHL players"
    p = next(iter(n["players"].values()))
    assert {"name", "pos", "team", "off", "def", "net", "rating", "toi"} <= set(p)


def test_nhl_probs_sane():
    """Model probabilities must be probabilities, and not degenerate."""
    n = _load("nhl.json")
    hps = [g["hp"] for g in n["schedule"]]
    assert all(0.0 < h < 1.0 for h in hps)
    assert max(hps) - min(hps) > 0.15, "prediction spread collapsed"


def test_record_route_is_wired_into_the_spa():
    """A payload block nobody routes to is invisible: pin the nav entry, the
    #/record route and the view function together."""
    from mlbwp_site.build_site import JS, SHELL
    assert 'href="#/record" data-v="record"' in SHELL             # nav entry
    assert 'if(v==="record"){setNav("record");return recordPage();}' in JS
    for fn in ("function recordPage(", "function recSection(", "function recRows(",
               "function recScore(", "function recPeriods("):
        assert fn in JS, fn
    html = (Path(__file__).resolve().parents[1] / "site" / "index.html")
    if html.is_file():
        body = html.read_text(encoding="utf-8")
        assert "function recordPage(" in body and "#/record" in body


def test_nhl_edges_noop_without_payload(tmp_path):
    """The edge layer must degrade to a clean no-op when inputs are missing —
    never raise inside the CI odds refresh."""
    from market import nhl_edges
    assert nhl_edges.attach_and_save(payload_path=tmp_path / "missing.json",
                                     opening_path=tmp_path / "open.json") == 0


def test_nhl_edges_noop_when_all_played(tmp_path):
    """All games completed (offseason) -> 0 edges, no odds API call needed."""
    from market import nhl_edges
    payload = {"schedule": [{"d": "2026-04-01", "home": "COL", "away": "DAL",
                             "hp": 0.6, "hs": 3, "as": 2}]}
    p = tmp_path / "nhl.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert nhl_edges.attach_and_save(payload_path=p,
                                     opening_path=tmp_path / "open.json") == 0
    out = json.loads(p.read_text(encoding="utf-8"))
    assert "value_updated" in out                            # touched, cleanly
