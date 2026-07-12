"""Archive CSV loader: newest-window cap and oriented home/away names."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import web.archive_csv_games as archive  # noqa: E402


def test_archive_cap_keeps_newest_games(tmp_path: Path, monkeypatch) -> None:
    """Ascending date sort + [:N] used to keep the oldest window."""
    root = tmp_path / "nba" / "team_data" / "2010"
    root.mkdir(parents=True)
    (root / "boston-celtics.csv").write_text(
        "01-01-2010,LAL,Home,100,90\n"
        "01-02-2010,LAL,Home,101,90\n"
        "01-03-2010,LAL,Home,102,90\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(archive, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(archive, "ARCHIVE_MAX_GAMES", 2)
    archive.load_archive_csv_games.cache_clear()

    with patch.object(archive, "_slug_to_abbr", return_value="bos"):
        with patch.object(archive, "_normalize_abbr", return_value="lal"):
            games = archive.load_archive_csv_games("nba", "2099-01-01")

    assert len(games) == 2
    # Newest two: Jan 2 and Jan 3 (not Jan 1 and Jan 2).
    assert games[0][4] == 101
    assert games[1][4] == 102


def test_archive_away_row_orients_names_with_keys(tmp_path: Path, monkeypatch) -> None:
    """Away-team CSV rows must not put the file team into home_name."""
    root = tmp_path / "nba" / "team_data" / "2010"
    root.mkdir(parents=True)
    (root / "boston-celtics.csv").write_text(
        "01-05-2010,LAL,Away,95,110\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(archive, "PROJECT_ROOT", tmp_path)
    archive.load_archive_csv_games.cache_clear()

    with patch.object(archive, "_slug_to_abbr", return_value="bos"):
        with patch.object(archive, "_normalize_abbr", return_value="lal"):
            games = archive.load_archive_csv_games("nba", "2099-01-01")

    assert len(games) == 1
    home_key, away_key, home_name, away_name, home_score, away_score = games[0]
    assert home_key == "lal"
    assert away_key == "bos"
    assert home_score == 110
    assert away_score == 95
    # Names follow oriented keys (home = LAL opponent, away = file team).
    assert home_name == "LAL"
    assert "Boston" in away_name or "Celtics" in away_name
