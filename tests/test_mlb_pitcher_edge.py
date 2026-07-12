"""Tests for MLB pitcher edge helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_pitcher_edge import _pitcher_skill  # noqa: E402


def test_pitcher_skill_favors_lower_era() -> None:
    good = _pitcher_skill({"era": 3.0, "whip": 1.1, "innings": 50})
    bad = _pitcher_skill({"era": 5.0, "whip": 1.45, "innings": 50})
    assert good > bad


@patch("web.mlb_pitcher_edge.fetch_probable_pitchers", return_value={})
def test_pitcher_matchup_margin_none_without_data(mock_fetch) -> None:
    from web.mlb_pitcher_edge import pitcher_matchup_margin

    assert pitcher_matchup_margin("nyy", "bos", game_date="2025-06-01") is None


def test_pitching_snapshot_reads_schedule_hydrate_direct_stats() -> None:
    """Schedule hydrate puts season ERA/WHIP on the block, not under splits."""
    from web.mlb_pitcher_edge import _pitching_snapshot

    person = {
        "id": 1,
        "fullName": "Test Arm",
        "stats": [
            {
                "type": {"displayName": "gameLog"},
                "group": {"displayName": "pitching"},
                "stats": {"era": "0.00", "inningsPitched": "5.0", "whip": "1.00"},
            },
            {
                "type": {"displayName": "statsSingleSeason"},
                "group": {"displayName": "pitching"},
                "exemptions": [],
                "stats": {
                    "era": "3.20",
                    "inningsPitched": "80.0",
                    "whip": "1.05",
                },
            },
        ],
    }
    snap = _pitching_snapshot(person)
    assert snap is not None
    assert snap["era"] == 3.2
    assert snap["whip"] == 1.05
    assert snap["innings"] == 80.0


def test_pitching_snapshot_still_reads_people_api_splits() -> None:
    from web.mlb_pitcher_edge import _pitching_snapshot

    person = {
        "stats": [
            {
                "type": {"displayName": "season"},
                "group": {"displayName": "pitching"},
                "splits": [
                    {
                        "stat": {
                            "era": "2.90",
                            "inningsPitched": "100.0",
                            "whip": "0.98",
                        }
                    }
                ],
            }
        ]
    }
    snap = _pitching_snapshot(person)
    assert snap is not None
    assert snap["era"] == 2.9


def test_fetch_probable_pitchers_uses_direct_season_stats(monkeypatch) -> None:
    from web import mlb_pitcher_edge as pe

    pe.fetch_probable_pitchers.cache_clear()
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "teams": {
                            "home": {
                                "team": {"id": 147},
                                "probablePitcher": {
                                    "id": 1,
                                    "stats": [
                                        {
                                            "type": {"displayName": "statsSingleSeason"},
                                            "group": {"displayName": "pitching"},
                                            "stats": {
                                                "era": "2.50",
                                                "inningsPitched": "60.0",
                                                "whip": "0.95",
                                            },
                                        }
                                    ],
                                },
                            },
                            "away": {
                                "team": {"id": 111},
                                "probablePitcher": {
                                    "id": 2,
                                    "stats": [
                                        {
                                            "type": {"displayName": "statsSingleSeason"},
                                            "group": {"displayName": "pitching"},
                                            "stats": {
                                                "era": "5.00",
                                                "inningsPitched": "55.0",
                                                "whip": "1.40",
                                            },
                                        }
                                    ],
                                },
                            },
                        }
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(pe, "_fetch_json", lambda *_a, **_k: payload)
    lookup = pe.fetch_probable_pitchers("2025-06-15")
    assert (147, 111) in lookup
    margin = pe.pitcher_matchup_margin("nyy", "bos", game_date="2025-06-15")
    assert margin is not None
    assert margin > 0
    pe.fetch_probable_pitchers.cache_clear()
