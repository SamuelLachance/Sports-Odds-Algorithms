"""Tests for international soccer closing odds loader."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.international_soccer_odds import (  # noqa: E402
    country_to_espn_abbr,
    international_odds_lookup,
    load_international_fd_games,
    load_international_odds_index,
)


def test_country_to_espn_abbr_maps_major_nations() -> None:
    assert country_to_espn_abbr("Brazil") == "bra"
    assert country_to_espn_abbr("United States") == "usa"
    assert country_to_espn_abbr("Bosnia & Herzegovina") == "bih"


def test_mauritius_not_aliased_to_mauritania() -> None:
    """Mauritius is FIFA/ESPN `mri`; Mauritania is `mtn` — never conflate."""
    from web.international_soccer_odds import _espn_country_to_abbr

    _espn_country_to_abbr.cache_clear()
    assert country_to_espn_abbr("Mauritania") == "mtn"
    assert country_to_espn_abbr("Mauritius") == "mri"
    assert country_to_espn_abbr("Mauritius") != country_to_espn_abbr("Mauritania")


def test_country_to_espn_abbr_rejects_ambiguous_korea_stem() -> None:
    """Bare 'Korea' must not prefix-match Korea DPR / Republic."""
    from web.international_soccer_odds import _espn_country_to_abbr

    _espn_country_to_abbr.cache_clear()
    assert country_to_espn_abbr("Korea") is None
    assert country_to_espn_abbr("South Korea") == "kor"
    assert country_to_espn_abbr("North Korea") == "prk"
    assert country_to_espn_abbr("Guinea") == "gui"
    assert country_to_espn_abbr("Guinea-Bissau") == "gnb"
    assert country_to_espn_abbr("Guinea") != country_to_espn_abbr("Guinea-Bissau")


def test_international_fd_games_include_scores_and_odds() -> None:
    games = load_international_fd_games()
    assert len(games) >= 400
    sample = games[0]
    assert sample["home_goals"] >= 0
    assert sample["away_goals"] >= 0
    assert sample["home_odds"] is not None


def test_international_fd_does_not_treat_max_as_close() -> None:
    """H_Max/D_Max/A_Max are best prices, not average closes — refuse the row."""
    from unittest.mock import patch

    csv_text = (
        "Date,Home,Away,HG,AG,H_Avg,D_Avg,A_Avg,H_Max,D_Max,A_Max\n"
        "15/06/2024,Brazil,Argentina,1,0,,,,2.10,3.40,3.50\n"
        "16/06/2024,France,Germany,2,1,1.90,3.40,4.00,1.95,3.50,4.20\n"
    )
    load_international_fd_games.cache_clear()
    with patch(
        "web.international_soccer_odds._fetch_source_file",
        return_value=csv_text,
    ):
        with patch(
            "web.international_soccer_odds.SOURCE_FILES",
            ("WorldCup2024.csv",),
        ):
            games = load_international_fd_games()
    assert len(games) == 1
    assert games[0]["home_name"] == "France"
    assert games[0]["home_odds"] is not None


def test_international_odds_lookup_known_sample() -> None:
    index = load_international_odds_index()
    assert index
    game_date, home, away = next(iter(index))
    hit = international_odds_lookup(game_date, home, away)
    assert hit is not None
    assert hit["home_odds"] is not None
    assert hit["draw_odds"] is not None
    assert hit["away_odds"] is not None
