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
