"""External game-rating lookup and name matching tests."""

from web.sports_db.external_ratings import (
    clear_rating_cache,
    lookup_player_rating,
    name_signature,
    normalize_player_name,
    rating_source_label,
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


def test_unknown_player_gets_prior_not_99() -> None:
    clear_rating_cache()
    result = lookup_player_rating(
        "nba",
        "Unknown Player XYZ",
        "bos",
        position="G",
        roster_meta={"experience": 1},
    )
    assert result.matched is False
    assert result.rating_source == "prior"
    assert 42 <= result.rating <= 58
    assert result.rating < 85


def test_madden_nfl_lookup() -> None:
    clear_rating_cache()
    result = lookup_player_rating("nfl", "Patrick Mahomes", "kc")
    assert result.matched is True
    assert result.rating == 99.0
    assert result.rating_source == "madden"


def test_rating_source_label() -> None:
    assert rating_source_label("2k", 2026) == "2K '26"
    assert rating_source_label("prior", 2026) == "Estimated '26"


def test_league_without_file_uses_prior() -> None:
    clear_rating_cache()
    result = lookup_player_rating("dwl", "Some Player", "abc")
    assert result.matched is False
    assert result.rating_source == "prior"
