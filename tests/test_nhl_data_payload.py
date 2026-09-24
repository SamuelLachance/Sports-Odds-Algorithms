"""Invariants on the SERVED NHL payload (site/data/nhl.json).

The NHL pages were far below MLB: zero goalies, departed players on old clubs,
a 10-day schedule, no divisions, 0-0-0 preseason standings. These check the
built payload itself; each degrades to a skip when it is not built."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "site" / "data" / "nhl.json"
LEDGER = ROOT / "data" / "nhl_hp_ledger.json"
NAMES = ROOT / "data" / "nhl_player_names.json"


@pytest.fixture(scope="module")
def n():
    if not PAYLOAD.is_file():
        pytest.skip("nhl.json not built")
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


def test_played_games_keep_their_ledgered_pre_game_probability(n):
    """Honesty rule 3: a played game's hp is the number published before puck
    drop - payload == ledger for every played game that has an entry."""
    if not LEDGER.is_file():
        pytest.skip("no ledger")
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    for g in n["schedule"]:
        if g.get("hs") is None:
            continue
        ent = led.get(str(g["id"]))
        if ent is None:
            assert g.get("replay") is True, g["id"]
        else:
            assert g["hp"] == ent["hp"], g["id"]


def test_unplayed_rows_are_in_the_ledger_with_the_served_number(n):
    if not LEDGER.is_file():
        pytest.skip("no ledger")
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    for g in n["schedule"]:
        if g.get("hs") is None:
            assert g["tier"] == "EARLY"
            if g.get("near"):                    # the near slate is what gets frozen
                ent = led[str(g["id"])]
                assert ent["hp"] == g["hp"] or g.get("frozen_at") == ent["t"]


def test_the_whole_season_is_served(n):
    reg = [g for g in n["schedule"] if not g.get("playoff")]
    if n.get("phase") in ("preseason", "regular"):
        assert len(reg) >= 1300, "only a near-term window is served"
        assert n["n_games"] == len(reg)
        assert any(g.get("near") == 0 for g in reg)
        for g in reg:
            assert {"hrest", "arest", "hb2b", "ab2b"} <= set(g)
    assert sum(1 for g in n["schedule"] if g.get("start_utc")) > 0.9 * len(n["schedule"])


def test_32_teams_with_identity_and_division(n):
    assert len(n["teams"]) == 32
    divs = {}
    for code, t in n["teams"].items():
        assert t["name"] and t["div"] in n["divisions"] and t["conf"] in n["conferences"]
        divs.setdefault(t["div"], []).append(code)
        assert t["div"] in n["conferences"][t["conf"]]
    assert sorted(len(v) for v in divs.values()) == [8, 8, 8, 8]


def test_preseason_has_last_season_and_no_fake_ranks(n):
    if n.get("phase") != "preseason":
        pytest.skip("not preseason")
    for t in n["teams"].values():
        assert t["gp"] == 0 and t["league_rank"] is None and t["po"] is None
        assert t["prev"] and t["prev"]["gp"] > 0 and t["prev"]["pts"] > 0
    ps = n["prev_season"]
    assert ps["kind"] == "replay" and "not a live record" in ps["note"]
    assert ps["n"] == len([r for r in ps["rows"] if r[5] is not None])


def test_players_are_exactly_the_current_rosters(n):
    if not NAMES.is_file():
        pytest.skip("no roster map")
    names = json.loads(NAMES.read_text(encoding="utf-8"))
    on = {k for k, v in names.items() if v.get("on_roster")}
    if not on:
        pytest.skip("legacy map without roster flags")
    assert set(n["players"]) == on
    for pid, p in n["players"].items():
        assert p["team"] == names[pid]["team"]
        assert pid in n["teams"][p["team"]]["roster"]


def test_goalies_are_served_with_a_rating_or_a_reason(n):
    gs = [p for p in n["players"].values() if p["pos"] == "G"]
    assert len(gs) >= 64, "every club carries at least two goalies"
    for p in gs:
        assert (p["rating"] is None) != (p["nr"] is None)
        if p["rating"] is not None:
            assert 1 <= p["rating"] <= 99 and p["gsax60"] is not None
    assert all(n["teams"][t]["goalies"] for t in n["teams"])


def test_unrated_skaters_say_why(n):
    for p in n["players"].values():
        if p["pos"] != "G" and p["rating"] is None:
            assert p["nr"], p["name"]
        if p["rating"] is not None and p["pos"] != "G":
            assert p["toi"] >= n["rapm"]["min_toi"]


def test_projection_covers_every_team(n):
    if not n.get("proj"):
        pytest.skip("no remaining games")
    assert set(n["proj"]) == set(n["teams"])
    assert abs(sum(v["po"] for v in n["proj"].values()) - 1600) < 1
    assert n["proj_info"]["tier"] == "EARLY"


def _keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def test_no_market_fields_in_model_blocks(n):
    """Rule 1: odds live only in the edge layer's value/mkt fields - never in
    the team, player, projection or last-season blocks this package builds."""
    keys = _keys({k: n[k] for k in ("teams", "players", "proj", "prev_season")}, set())
    market = {"odds", "dec", "mkt", "value", "book", "books", "ev", "vig", "line", "price"}
    bad = {k for k in keys if market & set(k.lower().split("_"))}
    assert not bad, bad
    for g in n["schedule"]:                       # only the edge layer's two keys
        assert not (market - {"value", "mkt"}) & set(g), g["id"]


def test_freshness_stamps(n):
    assert n["generated"] and n["served_at"]
    assert n["season_label"] and n["first_game"]
    assert "rosters" in n["sources"]
