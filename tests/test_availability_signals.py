"""Regression tests for ESPN availability / injury $ref resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.availability_signals import (  # noqa: E402
    AvailabilitySnapshot,
    _count_team_injuries,
    _injury_status_text,
    _status_means_out,
    availability_home_prob_shift,
)


def test_injury_status_text_reads_type_dict_not_str_repr() -> None:
    """type is a dict on detail payloads; str(dict) must not be the status source."""
    text = _injury_status_text(
        {
            "type": {
                "id": "1",
                "name": "INJURY_STATUS_OUT",
                "description": "out",
                "abbreviation": "O",
            }
        }
    )
    assert text == "out"
    assert _status_means_out(text)


def test_status_means_out_rejects_negations_and_incidental_substrings() -> None:
    """Bare ``in`` matching treated 'not injured' / 'timeout' as OUT."""
    assert _status_means_out("out") is True
    assert _status_means_out("o") is True
    assert _status_means_out("injury_status_out") is True
    assert _status_means_out("Injured Reserve") is True
    assert _status_means_out("suspended") is True
    assert _status_means_out("not injured") is False
    assert _status_means_out("timeout") is False
    assert _status_means_out("day-to-day") is False
    assert _status_means_out("questionable") is False


def test_count_team_injuries_resolves_ref_stubs_for_out() -> None:
    """ESPN list endpoints return {$ref} stubs with no inline status — must fetch."""
    list_payload = {
        "count": 2,
        "items": [
            {"$ref": "https://example.test/injuries/1"},
            {"$ref": "https://example.test/injuries/2"},
        ],
    }
    details = {
        "https://example.test/injuries/1": {"status": "Out", "type": {"description": "out"}},
        "https://example.test/injuries/2": {
            "status": "Day-To-Day",
            "type": {"description": "day-to-day"},
        },
    }

    def fake_fetch(url: str):
        if url in details:
            return details[url]
        if "/teams/13/injuries" in url:
            return list_payload
        return None

    with patch("web.availability_signals.get_league_profile", return_value={"sport_path": "basketball/nba"}):
        with patch("web.availability_signals._fetch_json", side_effect=fake_fetch):
            total, out = _count_team_injuries("nba", "13")

    assert total == 2
    assert out == 1  # only the Out player; Day-To-Day is not out


def test_count_team_injuries_fetches_when_type_is_ref_stub() -> None:
    """type: {$ref} without status/description must not skip the detail fetch."""
    list_payload = {
        "count": 1,
        "items": [
            {
                "$ref": "https://example.test/injuries/out-1",
                "type": {"$ref": "https://example.test/types/out"},
            }
        ],
    }
    details = {
        "https://example.test/injuries/out-1": {
            "status": "Out",
            "type": {"description": "out", "abbreviation": "O"},
        },
    }

    def fake_fetch(url: str):
        if url in details:
            return details[url]
        if "/teams/7/injuries" in url:
            return list_payload
        return None

    with patch("web.availability_signals.get_league_profile", return_value={"sport_path": "basketball/nba"}):
        with patch("web.availability_signals._fetch_json", side_effect=fake_fetch):
            total, out = _count_team_injuries("nba", "7")

    assert total == 1
    assert out == 1


def test_count_without_ref_resolution_would_miss_out() -> None:
    """Document the pre-fix failure mode: stub-only items never look like Out."""
    stubs = [{"$ref": "https://example.test/injuries/1"}]
    out = 0
    for item in stubs:
        status = str(item.get("status") or item.get("type") or "").lower()
        if "out" in status or "injured" in status or "suspended" in status:
            out += 1
    assert out == 0


def test_availability_shift_weights_out_heavier_than_generic() -> None:
    light = availability_home_prob_shift(
        AvailabilitySnapshot(home_injuries=2, away_injuries=0, home_out=0, away_out=0)
    )
    heavy = availability_home_prob_shift(
        AvailabilitySnapshot(home_injuries=2, away_injuries=0, home_out=2, away_out=0)
    )
    assert heavy < light  # more home OUT → lower home win%
    assert heavy < 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
