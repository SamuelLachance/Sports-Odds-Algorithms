"""External game-rating lookup and name matching tests."""

from web.sports_db.external_ratings import (
    RATING_PRIOR_SOURCE,
    clear_rating_cache,
    lookup_player_rating,
    match_external_rating,
    name_signature,
    normalize_player_name,
    rating_source_label,
    team_csv_ovr,
)


def test_normalize_player_name_strips_punctuation() -> None:
    assert normalize_player_name("D'Angelo Russell") == normalize_player_name("Dangelo Russell")


def test_name_signature_matches_initial_form() -> None:
    assert name_signature("Natasha Howard") == "howard|n"
    assert name_signature("N. Howard") == "howard|n"


def test_nba_2k_lookup_by_name_and_team() -> None:
    clear_rating_cache()
    result = lookup_player_rating("nba", "Jayson Tatum", "bos")
    assert result.matched is True
    assert result.rating == 96.0
    assert result.rating_source == "2k"
    assert result.rating_year == 2026


def test_wnba_dream_roster_lookup() -> None:
    clear_rating_cache()
    result = lookup_player_rating("wnba", "Natasha Howard", "atl")
    assert result.matched is True
    assert result.rating == 84.0
    assert result.rating_source == "2k"


def test_initial_form_fuzzy_match() -> None:
    clear_rating_cache()
    result = lookup_player_rating("wnba", "N. Howard", "atl")
    assert result.matched is True
    assert result.rating == 84.0


def test_unknown_player_on_team_with_csv_gets_derived() -> None:
    clear_rating_cache()
    result = lookup_player_rating(
        "nba",
        "Two-Way Callup",
        "bos",
        position="G",
        roster_meta={"experience": 0},
    )
    assert result.matched is False
    assert result.rating_source == "derived"
    assert 70 <= result.rating <= 90


def test_madden_nfl_lookup() -> None:
    clear_rating_cache()
    result = lookup_player_rating("nfl", "Patrick Mahomes", "kc")
    assert result.matched is True
    assert result.rating == 99.0
    assert result.rating_source == "madden"


def test_rating_source_label() -> None:
    assert rating_source_label("2k", 2026) == "2K '26"
    assert rating_source_label("prior", 2026) == "Missing '26"
    assert rating_source_label("derived", 2026) == "Team est. '26"


def test_match_external_rating_returns_none_when_missing() -> None:
    clear_rating_cache()
    assert match_external_rating("nba", "Nobody", "bos") is None


def test_team_csv_ovr_for_nba_bos() -> None:
    clear_rating_cache()
    entry = team_csv_ovr("nba", "bos")
    assert entry is not None
    assert entry["rating"] >= 80
    assert entry["method"] == "csv_top8_ovr"
    assert entry["players_matched"] >= 5


def test_team_csv_ovr_sparse_team_returns_none() -> None:
    clear_rating_cache()
    assert team_csv_ovr("nba", "zzz") is None
