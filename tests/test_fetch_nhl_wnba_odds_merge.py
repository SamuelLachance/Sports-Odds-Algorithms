"""NHL/WNBA ESPN odds merge must keep prior closes when fresh rows are empty stubs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_nhl_odds import _merge_rows as nhl_merge_rows  # noqa: E402
from scripts.fetch_wnba_odds import _merge_rows as wnba_merge_rows  # noqa: E402


def _nhl_row(**overrides: object) -> dict:
    base: dict = {
        "date": "2024-07-04",
        "home_key": "bos",
        "away_key": "nyi",
        "home_close_ml": "-150",
        "away_close_ml": "130",
        "home_open_ml": "",
        "away_open_ml": "",
        "home_close_spread": "-1.5",
        "away_close_spread": "1.5",
        "home_spread_odds": "-110",
        "away_spread_odds": "-110",
        "close_total": "",
        "open_total": "",
        "n_books": "2",
        "source": "sbr",
    }
    base.update(overrides)
    return base


def _wnba_row(**overrides: object) -> dict:
    base: dict = {
        "date": "2024-07-04",
        "home_key": "nya",
        "away_key": "con",
        "home_close_ml": "-150",
        "away_close_ml": "130",
        "home_close_spread": "-3.5",
        "away_close_spread": "3.5",
        "home_spread_odds": "-110",
        "away_spread_odds": "-110",
        "source": "sbr",
    }
    base.update(overrides)
    return base


def test_nhl_merge_keeps_prior_closes_when_fresh_empty() -> None:
    existing = [_nhl_row()]
    fresh = [
        _nhl_row(
            home_close_ml=None,
            away_close_ml=None,
            home_close_spread=None,
            away_close_spread=None,
            home_spread_odds=None,
            away_spread_odds=None,
            n_books=0,
            source="espn-core",
        )
    ]
    merged = nhl_merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"
    assert merged[0]["source"] == "sbr"


def test_wnba_merge_keeps_prior_closes_when_fresh_empty() -> None:
    existing = [_wnba_row()]
    fresh = [
        _wnba_row(
            home_close_ml=None,
            away_close_ml=None,
            home_close_spread=None,
            away_close_spread=None,
            home_spread_odds=None,
            away_spread_odds=None,
            source="espn-core",
        )
    ]
    merged = wnba_merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"
    assert merged[0]["source"] == "sbr"


def test_nhl_merge_fresh_with_odds_still_replaces() -> None:
    existing = [_nhl_row(home_close_ml="-140", source="sbr")]
    fresh = [_nhl_row(home_close_ml="-155", n_books="1", source="espn")]
    merged = nhl_merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-155"
    assert merged[0]["source"] == "espn"


def test_nhl_merge_spread_only_n_books_keeps_prior_mls() -> None:
    """n_books>0 without MLs must not wipe prior SBR moneylines."""
    existing = [
        _nhl_row(
            home_close_ml="-150",
            away_close_ml="130",
            home_open_ml="-130",
            away_open_ml="110",
            n_books="2",
            source="sbr",
        )
    ]
    fresh = [
        _nhl_row(
            home_close_ml=None,
            away_close_ml=None,
            home_open_ml=None,
            away_open_ml=None,
            home_close_spread="-1.5",
            away_close_spread="1.5",
            home_spread_odds="-105",
            away_spread_odds="-115",
            n_books=3,
            source="espn-core",
        )
    ]
    merged = nhl_merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"
    assert merged[0]["away_close_ml"] == "130"
    assert merged[0]["home_open_ml"] == "-130"
    assert merged[0]["home_spread_odds"] == "-105"
    assert merged[0]["n_books"] == "3"


def test_nhl_merge_ml_refresh_keeps_prior_spreads_and_opens() -> None:
    """Partial ESPN ML refresh must not blank prior spreads/opens."""
    existing = [
        _nhl_row(
            home_close_ml="-150",
            home_open_ml="-130",
            away_open_ml="110",
            home_close_spread="-1.5",
            away_close_spread="1.5",
            source="sbr",
        )
    ]
    fresh = [
        _nhl_row(
            home_close_ml="-155",
            away_close_ml="135",
            home_open_ml=None,
            away_open_ml=None,
            home_close_spread=None,
            away_close_spread=None,
            home_spread_odds=None,
            away_spread_odds=None,
            n_books=1,
            source="espn",
        )
    ]
    merged = nhl_merge_rows(existing, fresh)
    assert merged[0]["home_close_ml"] == "-155"
    assert merged[0]["home_close_spread"] == "-1.5"
    assert merged[0]["away_close_spread"] == "1.5"
    assert merged[0]["home_open_ml"] == "-130"
    assert merged[0]["home_spread_odds"] == "-110"


def test_wnba_merge_ml_refresh_keeps_prior_spreads() -> None:
    existing = [_wnba_row(home_close_ml="-150", home_close_spread="-3.5")]
    fresh = [
        _wnba_row(
            home_close_ml="-160",
            away_close_ml="140",
            home_close_spread=None,
            away_close_spread=None,
            home_spread_odds=None,
            away_spread_odds=None,
            source="espn",
        )
    ]
    merged = wnba_merge_rows(existing, fresh)
    assert merged[0]["home_close_ml"] == "-160"
    assert merged[0]["home_close_spread"] == "-3.5"
    assert merged[0]["home_spread_odds"] == "-110"
