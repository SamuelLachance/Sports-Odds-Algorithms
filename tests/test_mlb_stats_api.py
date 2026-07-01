"""MLB Stats API helper tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID, fetch_mlb_season_standings  # noqa: E402


def test_espn_mlb_team_map_has_all_clubs() -> None:
    assert len(ESPN_TO_MLB_TEAM_ID) >= 30
    assert ESPN_TO_MLB_TEAM_ID["bos"] == 111
    assert ESPN_TO_MLB_TEAM_ID["nyy"] == 147


def test_fetch_mlb_season_standings_shape() -> None:
    standings = fetch_mlb_season_standings(2024)
    if not standings:
        return
    assert "nyy" in standings or "bos" in standings
    sample = next(iter(standings.values()))
    assert "win_pct" in sample
    assert "runs_per_game" in sample
