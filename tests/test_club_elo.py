"""ClubElo helper tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.club_elo import (  # noqa: E402
    elo_binary_home_win_pct,
    resolve_club_elo_slug,
    supports_club_elo,
)


def test_supports_club_elo_for_top_leagues() -> None:
    assert supports_club_elo("epl")
    assert supports_club_elo("mls")
    assert not supports_club_elo("worldcup")


def test_resolve_club_elo_slug() -> None:
    assert resolve_club_elo_slug("epl", "liv") == "Liverpool"
    assert resolve_club_elo_slug("epl", "zzz") is None


def test_laliga_club_elo_abbrs_match_football_data_keys() -> None:
    """atm = Atlético Madrid, ath = Athletic Bilbao — not swapped/missing."""
    assert resolve_club_elo_slug("laliga", "atm") == "Atletico"
    assert resolve_club_elo_slug("laliga", "ath") == "Athletic"
    assert resolve_club_elo_slug("laliga", "atl") == "Atletico"
    assert resolve_club_elo_slug("laliga", "rso") == "Sociedad"


def test_ligue1_club_elo_aliases_cover_football_data_abbrs() -> None:
    assert resolve_club_elo_slug("ligue1", "olm") == "Marseille"
    assert resolve_club_elo_slug("ligue1", "lyon") == "Lyon"
    assert resolve_club_elo_slug("ligue1", "rcl") == "Lens"
    assert resolve_club_elo_slug("ligue1", "nice") == "Nice"


def test_elo_binary_home_win_pct_favors_stronger_home() -> None:
    strong = elo_binary_home_win_pct(1900.0, 1600.0)
    weak = elo_binary_home_win_pct(1600.0, 1900.0)
    assert strong > 50.0
    assert weak < 50.0
    assert strong > weak


@patch("web.club_elo.fetch_team_elo", return_value=1800.0)
def test_get_team_elo_rating_uses_mapping(mock_fetch) -> None:
    from web.club_elo import get_team_elo_rating

    rating = get_team_elo_rating("epl", "ars")
    assert rating == 1800.0
    mock_fetch.assert_called_once_with("Arsenal")
