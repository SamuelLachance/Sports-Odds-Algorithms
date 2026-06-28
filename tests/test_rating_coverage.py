"""Player rating coverage audit tests."""

from unittest.mock import patch

from web.sports_db.external_ratings import clear_rating_cache
from web.sports_db.rating_coverage import audit_league_ratings


def test_nba_sample_team_has_full_coverage() -> None:
    clear_rating_cache()
    with patch("web.sports_db.rating_scrape.scrape_league_team_players", return_value=[]):
        report = audit_league_ratings("nba", team_abbrs=["bos"], max_teams=1)
    assert report.players_checked > 0
    assert report.uncovered == 0
    assert report.direct_matches == report.players_checked


def test_prior_counts_as_uncovered() -> None:
    clear_rating_cache()
    with patch("web.sports_db.external_ratings.match_external_rating", return_value=None), patch(
        "web.sports_db.external_ratings.derived_team_rating", return_value=None
    ), patch(
        "web.sports_db.external_ratings.conservative_prior_rating",
        return_value=type(
            "R",
            (),
            {
                "rating": 50.0,
                "rating_source": "prior",
                "rating_year": 2026,
                "matched": False,
            },
        )(),
    ):
        report = audit_league_ratings("nba", team_abbrs=["bos"], max_teams=1)
    assert report.uncovered > 0
