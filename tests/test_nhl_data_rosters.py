"""Current NHL rosters are the source of truth for who plays where
(phase0/nhl_names.py). The old map merged forever, so a player who left a club
stayed listed on it; these pin the new flip-to-departed rule and its safety
rails (a failed request never removes anyone; entries are never deleted)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_names as N  # noqa: E402

NOW = "2026-09-24T12:00:00Z"


def _p(pid, first, last, pos="C", num=None):
    return {"id": pid, "firstName": {"default": first}, "lastName": {"default": last},
            "positionCode": pos, "sweaterNumber": num, "birthDate": "1997-01-13",
            "shootsCatches": "L", "heightInInches": 73, "weightInPounds": 190,
            "birthCountry": "CAN", "headshot": f"https://x/{pid}.png"}


def test_listed_player_gets_club_bio_and_on_roster():
    out = N.merge_rosters({}, {"EDM": [_p(97, "Connor", "McDavid", num=97)]}, NOW)
    e = out["97"]
    assert e["team"] == "EDM" and e["on_roster"] is True and e["num"] == 97
    assert e["name"] == "Connor McDavid" and e["born"] == "1997-01-13"
    assert e["shoots"] == "L" and e["img"] == "https://x/97.png" and e["seen"] == NOW


def test_player_missing_from_his_answering_club_is_flipped_to_departed():
    old = {"88": {"name": "Old Winger", "pos": "R", "team": "CHI", "on_roster": True}}
    out = N.merge_rosters(old, {"CHI": [_p(1, "A", "B")]}, NOW)
    assert out["88"]["on_roster"] is False
    assert out["88"]["team"] == "CHI"          # entry kept for research readers
    assert out["1"]["on_roster"] is True


def test_club_whose_request_failed_is_left_untouched():
    old = {"88": {"name": "X", "pos": "R", "team": "CHI", "on_roster": True}}
    out = N.merge_rosters(old, {"TOR": [_p(1, "A", "B")]}, NOW)     # CHI did not answer
    assert out["88"]["on_roster"] is True


def test_traded_player_moves_to_his_new_club():
    old = {"5": {"name": "Moved", "pos": "D", "team": "FLA", "on_roster": True}}
    out = N.merge_rosters(old, {"FLA": [], "TOR": [_p(5, "Mo", "Ved", "D")]}, NOW)
    assert out["5"]["team"] == "TOR" and out["5"]["on_roster"] is True


def test_legacy_entries_without_the_flag_are_never_treated_as_rostered_by_default():
    old = {"7": {"name": "Legacy", "pos": "C", "team": "ARI"}}      # defunct club
    out = N.merge_rosters(old, {"UTA": [_p(1, "A", "B")]}, NOW)
    assert "on_roster" not in out["7"] or out["7"]["on_roster"] is not True
    assert len(out) == 2                                              # never deleted


def test_refresh_writes_nothing_when_every_request_fails(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"1": {"name": "A", "team": "TOR", "on_roster": True}}))

    def boom(url):
        raise OSError("offline")
    ok, n = N.refresh(0, teams=("TOR", "MTL"), get_fn=boom, path=str(path), sleep=0)
    assert (ok, n) == (0, 2)
    assert json.loads(path.read_text())["1"]["on_roster"] is True


def test_refresh_with_a_stubbed_api_and_the_freshness_skip(tmp_path):
    path = tmp_path / "names.json"

    def fake(url):
        team = url.split("/roster/")[1].split("/")[0]
        return {"forwards": [_p(100 + len(team), "F", team)], "defensemen": [],
                "goalies": [_p(900 + len(team), "G", team, "G")]}
    ok, n = N.refresh(0, teams=("TOR",), get_fn=fake, path=str(path), sleep=0)
    assert (ok, n) == (1, 1)
    data = json.loads(path.read_text())
    assert {v["pos"] for v in data.values()} == {"C", "G"}
    assert N.fetched_at(data) is not None
    # fresh map + max age -> skipped, not refetched
    assert N.refresh(6, teams=("TOR",), get_fn=fake, path=str(path), sleep=0) == (0, 0)


def test_the_committed_map_flags_current_rosters():
    p = ROOT / "data" / "nhl_player_names.json"
    if not p.is_file():
        return
    names = json.loads(p.read_text(encoding="utf-8"))
    on = [v for v in names.values() if v.get("on_roster")]
    if not on:
        return                                    # pre-refresh legacy map
    assert len({v["team"] for v in on}) == 32
    assert any(v["pos"] == "G" for v in on)
