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
