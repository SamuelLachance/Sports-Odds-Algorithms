"""Closing-odds database lookup and normalization."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web import closing_odds_db  # noqa: E402


def test_normalize_team_key_maps_espn_abbreviations() -> None:
    assert closing_odds_db.normalize_team_key("nfl", "KC") == "kc"
    assert closing_odds_db.normalize_team_key("nba", "Celtics") == "bos"


def test_nhl_odds_key_alias_resolves_st_louis() -> None:
    assert closing_odds_db.closing_odds_lookup is not None
    from web.closing_odds_db import _canonical_odds_team_key

    assert _canonical_odds_team_key("nhl", "st.louis") == "stl"
    assert _canonical_odds_team_key("nhl", "winnipegjets") == "wpg"


def test_closing_odds_lookup_same_day_swap_and_fuzzy_date(tmp_path: Path, monkeypatch) -> None:
    odds_dir = tmp_path / "closing-odds"
    odds_dir.mkdir()
    csv_path = odds_dir / "nba.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "home_key",
                "away_key",
                "home_close_ml",
                "away_close_ml",
                "home_close_spread",
                "away_close_spread",
                "home_spread_odds",
                "away_spread_odds",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date": "2024-01-15",
                "home_key": "bos",
                "away_key": "ny",
                "home_close_ml": -150,
                "away_close_ml": 130,
                "home_close_spread": -3.5,
                "away_close_spread": 3.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
                "source": "test",
            }
        )

    monkeypatch.setattr(closing_odds_db, "ODDS_DIR", odds_dir)
    closing_odds_db.clear_closing_odds_cache()

    row = closing_odds_db.closing_odds_lookup("nba", "2024-01-15", "bos", "ny")
    assert row is not None
    assert row["home_close_ml"] == -150
    assert row["home_close_spread"] == -3.5
    assert row["home_spread_odds"] == -110

    # Same-day home/away flip is tolerated (source feed orientation).
    swapped = closing_odds_db.closing_odds_lookup("nba", "2024-01-15", "ny", "bos")
    assert swapped is not None
    assert swapped["home_close_ml"] == 130
    assert swapped["away_close_ml"] == -150
    assert swapped["home_close_spread"] == 3.5

    # Fuzzy ±1 day keeps the same home/away keys.
    fuzzy = closing_odds_db.closing_odds_lookup("nba", "2024-01-16", "bos", "ny")
    assert fuzzy is not None
    assert fuzzy["home_close_ml"] == -150

    missing = closing_odds_db.closing_odds_lookup("nba", "2024-01-20", "bos", "ny")
    assert missing is None


def test_closing_odds_lookup_does_not_swap_across_fuzzy_dates(
    tmp_path: Path, monkeypatch
) -> None:
    """Adjacent-day + flipped venue must not attach the other game's line."""
    odds_dir = tmp_path / "closing-odds"
    odds_dir.mkdir()
    (odds_dir / "nba.csv").write_text(
        "date,home_key,away_key,home_close_ml,away_close_ml,"
        "home_close_spread,away_close_spread,home_spread_odds,away_spread_odds,source\n"
        "2024-01-16,ny,bos,-200,170,-6.5,6.5,-110,-110,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(closing_odds_db, "ODDS_DIR", odds_dir)
    closing_odds_db.clear_closing_odds_cache()

    # Looking up bos@ny on 1/15 must not flip the 1/16 ny@bos row.
    row = closing_odds_db.closing_odds_lookup("nba", "2024-01-15", "bos", "ny")
    assert row is None


def test_closing_odds_parse_int_rejects_invalid_magnitude() -> None:
    assert closing_odds_db._parse_int(0) == 100
    assert closing_odds_db._parse_int(50) is None
    assert closing_odds_db._parse_int(-75) is None
    assert closing_odds_db._parse_int(-110) == -110
    assert closing_odds_db._parse_int(150) == 150


def test_closing_odds_even_zero_maps_to_plus_100(tmp_path: Path, monkeypatch) -> None:
    odds_dir = tmp_path / "closing-odds"
    odds_dir.mkdir()
    (odds_dir / "nba.csv").write_text(
        "date,home_key,away_key,home_close_ml,away_close_ml,"
        "home_close_spread,away_close_spread,home_spread_odds,away_spread_odds,source\n"
        "2024-02-01,bos,ny,0,-110,-3.5,3.5,0,-105,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(closing_odds_db, "ODDS_DIR", odds_dir)
    closing_odds_db.clear_closing_odds_cache()

    row = closing_odds_db.closing_odds_lookup("nba", "2024-02-01", "bos", "ny")
    assert row is not None
    assert row["home_close_ml"] == 100
    assert row["home_spread_odds"] == 100
    assert row["away_spread_odds"] == -105


def test_closing_odds_coverage_counts_rows(tmp_path: Path, monkeypatch) -> None:
    odds_dir = tmp_path / "closing-odds"
    odds_dir.mkdir()
    (odds_dir / "nhl.csv").write_text(
        "date,home_key,away_key,home_close_ml,away_close_ml,"
        "home_close_spread,away_close_spread,home_spread_odds,away_spread_odds,source\n"
        "2023-10-10,bos,tor,-120,100,-1.5,1.5,-110,-110,sbr-online\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(closing_odds_db, "ODDS_DIR", odds_dir)
    closing_odds_db.clear_closing_odds_cache()

    coverage = closing_odds_db.closing_odds_coverage("nhl")
    assert coverage["rows"] == 1
    assert "nhl.csv" in coverage["source"]


def test_flip_two_way_odds_swaps_sparse_spreads() -> None:
    """Home/away flip must swap sparse spread lines with their juice."""
    flipped = closing_odds_db._flip_two_way_odds(
        {
            "home_close_ml": -150,
            "away_close_ml": 130,
            "home_close_spread": None,
            "away_close_spread": 3.5,
            "home_spread_odds": None,
            "away_spread_odds": -110,
        }
    )
    assert flipped["home_close_ml"] == 130
    assert flipped["away_close_ml"] == -150
    assert flipped["home_close_spread"] == 3.5
    assert flipped["away_close_spread"] is None
    assert flipped["home_spread_odds"] == -110
    assert flipped["away_spread_odds"] is None
