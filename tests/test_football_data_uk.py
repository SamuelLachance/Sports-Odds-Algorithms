"""football-data.co.uk closing-odds column preference."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.football_data_uk import (  # noqa: E402
    _parse_american_odds,
    load_football_data_uk_games,
)


def test_parse_american_odds_from_decimal() -> None:
    assert _parse_american_odds("2.00") == 100
    assert _parse_american_odds("1.50") == -200
    assert _parse_american_odds("") is None


def test_parse_american_odds_rejects_nan_and_inf() -> None:
    """Corrupt FD cells must fail closed — never raise into the season loader."""
    assert _parse_american_odds("nan") is None
    assert _parse_american_odds("NaN") is None
    assert _parse_american_odds("inf") is None
    assert _parse_american_odds("-inf") is None


def test_load_football_data_uk_skips_nan_close_for_avg_fallback() -> None:
    """Truthy NaN in PSCH must not crash or block AvgCH fallback."""
    csv_text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,"
        "PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
        "E0,16/08/25,Arsenal,Everton,2,0,"
        "nan,inf,-,1.95,3.60,4.00\n"
    )
    load_football_data_uk_games.cache_clear()
    with patch("web.football_data_uk._season_tags", return_value=["2526"]):
        with patch("web.football_data_uk._fetch_csv", return_value=csv_text):
            rows = load_football_data_uk_games("epl")
    assert len(rows) == 1
    assert rows[0]["home_odds"] == _parse_american_odds("1.95")
    assert rows[0]["draw_odds"] == _parse_american_odds("3.60")
    assert rows[0]["away_odds"] == _parse_american_odds("4.00")


def test_team_abbr_fails_closed_on_ambiguous_manchester_prefix() -> None:
    """Truncated 'Manchester'/'Man' must not map to City via first prefix hit."""
    from web.football_data_uk import _team_abbr

    assert _team_abbr("epl", "Manchester United") == "man"
    assert _team_abbr("epl", "Manchester City") == "mnc"
    assert _team_abbr("epl", "Manchester") is None
    assert _team_abbr("epl", "Man") is None
    assert _team_abbr("epl", "Arsenal") == "ars"


def test_load_football_data_uk_prefers_pinnacle_close() -> None:
    """PSCH (close) must win over PSH (open) when both are present."""
    csv_text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,"
        "PSH,PSD,PSA,PSCH,PSCD,PSCA\n"
        "E0,16/08/25,Arsenal,Everton,2,0,"
        "2.10,3.50,3.50,1.95,3.60,4.00\n"
    )
    load_football_data_uk_games.cache_clear()
    with patch("web.football_data_uk._season_tags", return_value=["2526"]):
        with patch("web.football_data_uk._fetch_csv", return_value=csv_text):
            rows = load_football_data_uk_games("epl")
    assert len(rows) == 1
    # 1.95 decimal → American about -105; 2.10 → +110.
    assert rows[0]["home_odds"] == _parse_american_odds("1.95")
    assert rows[0]["home_odds"] != _parse_american_odds("2.10")
    assert rows[0]["draw_odds"] == _parse_american_odds("3.60")
    assert rows[0]["away_odds"] == _parse_american_odds("4.00")


def test_load_football_data_uk_skips_unparseable_close_for_avg_fallback() -> None:
    """Truthy placeholder in PSCH must not block AvgCH fallback."""
    csv_text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,"
        "PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
        "E0,16/08/25,Arsenal,Everton,2,0,"
        "-,-,-,1.95,3.60,4.00\n"
    )
    load_football_data_uk_games.cache_clear()
    with patch("web.football_data_uk._season_tags", return_value=["2526"]):
        with patch("web.football_data_uk._fetch_csv", return_value=csv_text):
            rows = load_football_data_uk_games("epl")
    assert len(rows) == 1
    assert rows[0]["home_odds"] == _parse_american_odds("1.95")
    assert rows[0]["draw_odds"] == _parse_american_odds("3.60")
    assert rows[0]["away_odds"] == _parse_american_odds("4.00")


def test_load_football_data_uk_does_not_label_opens_as_closes() -> None:
    """Open-only rows must leave odds None — never store PSH as a close."""
    csv_text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,"
        "PSH,PSD,PSA,B365H,B365D,B365A\n"
        "E0,16/08/25,Arsenal,Everton,2,0,"
        "2.10,3.50,3.50,2.05,3.40,3.60\n"
    )
    load_football_data_uk_games.cache_clear()
    with patch("web.football_data_uk._season_tags", return_value=["2526"]):
        with patch("web.football_data_uk._fetch_csv", return_value=csv_text):
            rows = load_football_data_uk_games("epl")
    assert len(rows) == 1
    assert rows[0]["home_goals"] == 2
    assert rows[0]["home_odds"] is None
    assert rows[0]["draw_odds"] is None
    assert rows[0]["away_odds"] is None


def test_team_abbr_maps_fd_spelling_variants() -> None:
    """Unmapped FD labels silently drop whole games from closing odds."""
    from web.football_data_uk import _team_abbr

    assert _team_abbr("laliga", "Espanol") == "esp"
    assert _team_abbr("laliga", "Espanyol") == "esp"
    assert _team_abbr("bundesliga", "St Pauli") == "stp"
    assert _team_abbr("bundesliga", "Holstein Kiel") == "ksv"


def test_load_football_data_uk_keeps_espanol_and_st_pauli_rows() -> None:
    csv_text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,PSCH,PSCD,PSCA\n"
        "SP1,16/08/24,Espanol,Barcelona,1,2,4.50,3.60,1.75\n"
        "D1,24/08/24,St Pauli,Holstein Kiel,0,0,2.20,3.30,3.40\n"
    )
    load_football_data_uk_games.cache_clear()
    with patch("web.football_data_uk._season_tags", return_value=["2425"]):
        with patch("web.football_data_uk._fetch_csv", return_value=csv_text):
            laliga = load_football_data_uk_games("laliga")
            bundes = load_football_data_uk_games("bundesliga")
    assert len(laliga) == 1
    assert laliga[0]["home_key"] == "esp"
    assert laliga[0]["away_key"] == "bar"
    assert len(bundes) == 1
    assert bundes[0]["home_key"] == "stp"
    assert bundes[0]["away_key"] == "ksv"
