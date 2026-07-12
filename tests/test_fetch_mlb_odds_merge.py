"""MLB ESPN odds merge must not last-wins collapse doubleheaders."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_mlb_odds import _merge_rows  # noqa: E402


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "date": "2024-07-04",
        "home_key": "nyy",
        "away_key": "bos",
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
        "n_books": "1",
        "source": "espn",
    }
    base.update(overrides)
    return base


def test_merge_keeps_conflicting_doubleheader_fresh_rows() -> None:
    g1 = _row(home_close_ml="-150", away_close_ml="130")
    g2 = _row(home_close_ml="-120", away_close_ml="100", home_spread_odds="-115")
    merged = _merge_rows([], [g1, g2])
    assert len(merged) == 2


def test_merge_fresh_single_replaces_existing() -> None:
    existing = [_row(home_close_ml="-140", source="sbr")]
    fresh = [_row(home_close_ml="-155", source="espn")]
    merged = _merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-155"
    assert merged[0]["source"] == "espn"


def test_merge_keeps_existing_when_no_fresh_for_key() -> None:
    existing = [_row(date="2024-07-03")]
    fresh = [_row(date="2024-07-04")]
    merged = _merge_rows(existing, fresh)
    assert len(merged) == 2


def test_merge_keeps_prior_closes_when_fresh_empty_stub() -> None:
    """Empty ESPN stubs (n_books=0) must not wipe good SBR/ESPN closes."""
    existing = [_row(home_close_ml="-150", away_close_ml="130", n_books="2", source="sbr")]
    fresh = [
        _row(
            home_close_ml="",
            away_close_ml="",
            home_close_spread="",
            away_close_spread="",
            home_spread_odds="",
            away_spread_odds="",
            n_books="0",
            source="espn-core",
        )
    ]
    merged = _merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"
    assert merged[0]["n_books"] == "2"
    assert merged[0]["source"] == "sbr"


def test_merge_drops_empty_stub_alongside_real_dh_sibling() -> None:
    """A blank stub must not sit next to a real DH row and force ambiguity."""
    g1 = _row(home_close_ml="-150", away_close_ml="130", n_books="1")
    stub = _row(
        home_close_ml="",
        away_close_ml="",
        home_close_spread="",
        away_close_spread="",
        home_spread_odds="",
        away_spread_odds="",
        n_books="0",
        source="espn-core",
    )
    merged = _merge_rows([], [g1, stub])
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"


def test_merge_partial_fresh_keeps_existing_dh_sibling() -> None:
    """A one-game ESPN refresh must not delete the other DH close."""
    g1 = _row(home_close_ml="-150", away_close_ml="130", source="sbr")
    g2 = _row(home_close_ml="-120", away_close_ml="100", home_spread_odds="-115", source="sbr")
    fresh_g1 = _row(home_close_ml="-155", away_close_ml="135", source="espn")
    merged = _merge_rows([g1, g2], [fresh_g1])
    mls = sorted(row["home_close_ml"] for row in merged)
    assert mls == ["-120", "-155"]
    assert len(merged) == 2


def test_merge_spread_only_n_books_keeps_prior_mls() -> None:
    """Run-line-only ESPN rows (n_books>0, empty MLs) must not wipe prior MLs."""
    existing = [_row(home_close_ml="-150", away_close_ml="130", n_books="2", source="sbr")]
    fresh = [
        _row(
            home_close_ml="",
            away_close_ml="",
            home_close_spread="-1.5",
            away_close_spread="1.5",
            home_spread_odds="-110",
            away_spread_odds="-110",
            n_books="3",
            source="espn-core",
        )
    ]
    merged = _merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"
    assert merged[0]["source"] == "sbr"


def test_merge_spread_only_stub_does_not_replace_wrong_dh_leg() -> None:
    """Empty home ML must not coerce to 0 and match the nearer DH favorite."""
    from scripts.fetch_mlb_odds import _home_ml_distance

    g1 = _row(home_close_ml="-150", away_close_ml="130", source="sbr")
    g2 = _row(home_close_ml="-120", away_close_ml="100", home_spread_odds="-115", source="sbr")
    stub = _row(
        home_close_ml="",
        away_close_ml="",
        home_close_spread="-1.5",
        away_close_spread="1.5",
        n_books="2",
        source="espn",
    )
    assert _home_ml_distance(g1, stub) == float("inf")
    assert _home_ml_distance(g2, stub) == float("inf")
    merged = _merge_rows([g1, g2], [stub])
    mls = sorted(row["home_close_ml"] for row in merged)
    assert mls == ["-120", "-150"]
    assert all(row["source"] == "sbr" for row in merged)


def test_merge_ml_refresh_keeps_prior_spreads() -> None:
    existing = [
        _row(
            home_close_ml="-150",
            home_open_ml="-140",
            home_close_spread="-1.5",
            away_close_spread="1.5",
            source="sbr",
        )
    ]
    fresh = [
        _row(
            home_close_ml="-155",
            away_close_ml="135",
            home_open_ml="",
            away_open_ml="",
            home_close_spread="",
            away_close_spread="",
            home_spread_odds="",
            away_spread_odds="",
            source="espn",
        )
    ]
    merged = _merge_rows(existing, fresh)
    assert merged[0]["home_close_ml"] == "-155"
    assert merged[0]["home_close_spread"] == "-1.5"
    assert merged[0]["home_open_ml"] == "-140"
    assert merged[0]["home_spread_odds"] == "-110"
