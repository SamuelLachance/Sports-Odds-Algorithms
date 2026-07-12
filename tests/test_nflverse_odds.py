"""nflverse closing-odds import normalization."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nflverse_odds import _parse_int  # noqa: E402


def test_nflverse_parse_int_matches_closing_odds_rules() -> None:
    assert _parse_int(0) == 100
    assert _parse_int("0") == 100
    assert _parse_int(-50) is None
    assert _parse_int(75) is None
    assert _parse_int("1.91") is None
    assert _parse_int(-110) == -110
    assert _parse_int(150) == 150
    assert _parse_int(None) is None
    assert _parse_int("") is None
    # inf used to raise OverflowError and abort the CSV import mid-file.
    assert _parse_int("inf") is None
    assert _parse_int("-inf") is None
    assert _parse_int("nan") is None


def test_nflverse_rows_flip_spread_line_to_betting_convention(tmp_path: Path) -> None:
    """nflverse +3.0 (home favored) must become home_close_spread -3.0."""
    from web.nflverse_odds import rows_from_nflverse_csv

    csv_path = tmp_path / "games.csv"
    csv_path.write_text(
        "game_type,gameday,home_team,away_team,spread_line,"
        "home_moneyline,away_moneyline,home_spread_odds,away_spread_odds\n"
        "REG,2024-09-08,KC,BAL,3.0,-150,130,-110,-110\n"
        "REG,2024-09-08,ATL,PIT,-4.0,160,-180,-105,-115\n",
        encoding="utf-8",
    )
    rows = rows_from_nflverse_csv(csv_path)
    assert len(rows) == 2
    assert float(rows[0]["home_close_spread"]) == -3.0
    assert float(rows[0]["away_close_spread"]) == 3.0
    # Underdog home: nflverse -4 → betting +4
    assert float(rows[1]["home_close_spread"]) == 4.0
    assert float(rows[1]["away_close_spread"]) == -4.0


def test_nflverse_drops_juice_sized_spread_line(tmp_path: Path) -> None:
    """American juice magnitudes must not become NFL closing spreads."""
    from web.nflverse_odds import rows_from_nflverse_csv

    csv_path = tmp_path / "games.csv"
    csv_path.write_text(
        "game_type,gameday,home_team,away_team,spread_line,"
        "home_moneyline,away_moneyline,home_spread_odds,away_spread_odds\n"
        "REG,2024-09-08,KC,BAL,-110,-150,130,-110,-110\n"
        "REG,2024-09-15,BUF,MIA,3.5,-165,145,-105,-115\n",
        encoding="utf-8",
    )
    rows = rows_from_nflverse_csv(csv_path)
    assert len(rows) == 2
    assert rows[0]["home_close_spread"] == ""
    assert rows[0]["away_close_spread"] == ""
    assert rows[0]["home_close_ml"] == -150
    assert float(rows[1]["home_close_spread"]) == -3.5
    assert float(rows[1]["away_close_spread"]) == 3.5


def test_nflverse_drops_non_finite_spread_line(tmp_path: Path) -> None:
    """NaN spread_line must not poison the closing-odds cache CSV."""
    from web.nflverse_odds import rows_from_nflverse_csv

    csv_path = tmp_path / "games.csv"
    csv_path.write_text(
        "game_type,gameday,home_team,away_team,spread_line,"
        "home_moneyline,away_moneyline,home_spread_odds,away_spread_odds\n"
        "REG,2024-09-08,KC,BAL,nan,-150,130,-110,-110\n",
        encoding="utf-8",
    )
    row = rows_from_nflverse_csv(csv_path)[0]
    assert row["home_close_spread"] == ""
    assert row["away_close_spread"] == ""
    assert row["home_close_ml"] == -150


def test_nflverse_keeps_playoff_game_types(tmp_path: Path) -> None:
    """WC/DIV/CON/SB must import; nflverse no longer uses a single POST label."""
    from web.nflverse_odds import rows_from_nflverse_csv

    csv_path = tmp_path / "games.csv"
    csv_path.write_text(
        "game_type,gameday,home_team,away_team,spread_line,"
        "home_moneyline,away_moneyline,home_spread_odds,away_spread_odds\n"
        "WC,2025-01-11,BUF,DEN,8.5,-450,360,-110,-110\n"
        "DIV,2025-01-19,KC,HOU,9.5,-500,390,-110,-110\n"
        "CON,2025-01-26,PHI,WAS,6.0,-250,210,-110,-110\n"
        "SB,2025-02-09,PHI,KC,-1.5,110,-130,-105,-115\n"
        "PRE,2024-08-10,KC,CHI,3.0,-150,130,-110,-110\n",
        encoding="utf-8",
    )
    rows = rows_from_nflverse_csv(csv_path)
    assert len(rows) == 4
    dates = {r["date"] for r in rows}
    assert "2025-02-09" in dates
    assert all(r["date"] != "2024-08-10" for r in rows)
