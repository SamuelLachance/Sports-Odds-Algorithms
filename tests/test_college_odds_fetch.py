"""CFB/CBB odds fetch must keep blowout spreads and not invent juice."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_odds_espn import MAX_NBA_SPREAD, _consensus  # noqa: E402


def _blowout_item(spread: float = -45.5) -> dict:
    return {
        "provider": {"name": "consensus"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": spread},
                "moneyLine": {"american": -5000},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": -spread},
                "moneyLine": {"american": 1800},
                "spread": {"american": -110},
            },
        },
        "spread": spread,
    }


def test_nba_consensus_default_drops_cfb_sized_blowout() -> None:
    consensus = _consensus([_blowout_item(-45.5)])
    assert consensus["home_close_spread"] is None
    assert consensus["home_close_ml"] == -5000.0


def test_cfb_consensus_keeps_blowout_spread() -> None:
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    consensus = _consensus([_blowout_item(-45.5)], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] == -45.5
    assert consensus["away_close_spread"] == 45.5
    assert MAX_CFB_SPREAD > MAX_NBA_SPREAD


def test_cbb_consensus_keeps_blowout_spread() -> None:
    from scripts.fetch_cbb_odds import MAX_CBB_SPREAD

    consensus = _consensus([_blowout_item(-45.5)], max_handicap_abs=MAX_CBB_SPREAD)
    assert consensus["home_close_spread"] == -45.5
    assert MAX_CBB_SPREAD > MAX_NBA_SPREAD


def test_cfb_odds_row_does_not_invent_spread_juice() -> None:
    """Missing juice must stay None — never fake -110."""
    from scripts import fetch_cfb_odds as mod

    event = {
        "date": "2024-09-01",
        "home_key": "ala",
        "away_key": "uga",
        "event": "1",
        "comp": "1",
        "home_final": 28,
        "away_final": 21,
    }
    payload = {
        "items": [
            {
                "provider": {"name": "book"},
                "homeTeamOdds": {
                    "favorite": True,
                    "close": {
                        "pointSpread": {"american": -7.5},
                        "moneyLine": {"american": -300},
                        # spread juice omitted
                    },
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "close": {
                        "pointSpread": {"american": 7.5},
                        "moneyLine": {"american": 250},
                    },
                },
                "spread": -7.5,
            }
        ]
    }
    with patch.object(mod, "_throttled_get", return_value=payload):
        row = mod._odds_row(event)
    assert row["home_close_spread"] == -7.5
    assert row["home_spread_odds"] is None
    assert row["away_spread_odds"] is None
    assert row["home_close_ml"] == -300.0
