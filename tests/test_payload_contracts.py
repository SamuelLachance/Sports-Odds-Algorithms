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
