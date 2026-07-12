"""Team profiles must not count draws/ties as losses."""

from __future__ import annotations

from unittest.mock import patch

from web.team_service import get_team_profile


def test_get_team_profile_counts_draws_as_ties_not_losses() -> None:
    team_row = {
        "abbr": "ars",
        "slug": "arsenal",
        "label": "Arsenal",
    }
    live_entry = {
        "game_scores": [[2, 1], [1, 1], [0, 1]],
        "other_team": ["che", "tot", "liv"],
        "dates": ["8-10-2024", "8-17-2024", "8-24-2024"],
        "home_away": ["home", "away", "home"],
        "seasons_used": [2025],
    }

    with (
        patch("web.team_service.get_teams", return_value=[team_row]),
        patch("web.team_service.fetch_espn_team_ids", return_value={"ars": "359"}),
        patch(
            "web.team_service.resolve_team",
            return_value={"abbr": "ARS", "label": "Arsenal"},
        ),
        patch("web.team_service.load_live_team_data", return_value=[live_entry]),
        patch("web.team_service.current_season_year", return_value=2025),
    ):
        profile = get_team_profile("epl", "ars")

    stats = profile["season_stats"]
    assert stats["wins"] == 1
    assert stats["ties"] == 1
    assert stats["losses"] == 1
    assert stats["games_played"] == 3
    assert stats["win_pct"] == 33.3

    results = [g["result"] for g in profile["recent_games"]]
    assert results == ["L", "T", "W"]  # newest first
    assert all(g["result"] != "L" or g["score"] != [1, 1] for g in profile["recent_games"])
    tie_row = next(g for g in profile["recent_games"] if g["score"] == [1, 1])
    assert tie_row["result"] == "T"
