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


def test_cfb_provider_rejects_juice_sized_point_spread() -> None:
    """CFB max_abs=120 must not keep −110 juice dumped into pointSpread."""
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    item = {
        "provider": {"name": "book"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": -110},  # juice dump
                "moneyLine": {"american": -300},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": 110},
                "moneyLine": {"american": 250},
                "spread": {"american": -110},
            },
        },
        "spread": -7.5,
    }
    consensus = _consensus([item], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] == -7.5
    assert consensus["away_close_spread"] == 7.5
    assert consensus["home_close_ml"] == -300.0


def test_cfb_provider_rejects_juice_point_spread_without_flat_spread() -> None:
    """Missing flat `spread` must still drop |pointSpread| >= 100 juice dumps."""
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    item = {
        "provider": {"name": "book"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": -110},
                "moneyLine": {"american": -200},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": 110},
                "moneyLine": {"american": 170},
                "spread": {"american": -110},
            },
        },
        # no top-level "spread"
    }
    consensus = _consensus([item], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] is None
    assert consensus["away_close_spread"] is None
    assert consensus["home_close_ml"] == -200.0


def test_cbb_odds_row_excludes_live_books_from_consensus() -> None:
    """Live/in-game books must not pollute CBB closing consensus."""
    from scripts import fetch_cbb_odds as mod

    event = {
        "date": "2024-01-15",
        "home_key": "duke",
        "away_key": "unc",
        "event": "1",
        "comp": "1",
        "home_final": 80,
        "away_final": 72,
    }
    payload = {
        "items": [
            {
                "provider": {"name": "DraftKings"},
                "homeTeamOdds": {
                    "favorite": True,
                    "close": {
                        "pointSpread": {"american": -5.5},
                        "moneyLine": {"american": -220},
                        "spread": {"american": -110},
                    },
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "close": {
                        "pointSpread": {"american": 5.5},
                        "moneyLine": {"american": 180},
                        "spread": {"american": -110},
                    },
                },
                "spread": -5.5,
            },
            {
                "provider": {"name": "DraftKings Live"},
                "homeTeamOdds": {
                    "favorite": True,
                    "close": {
                        "pointSpread": {"american": -12.5},
                        "moneyLine": {"american": -500},
                        "spread": {"american": -115},
                    },
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "close": {
                        "pointSpread": {"american": 12.5},
                        "moneyLine": {"american": 375},
                        "spread": {"american": -105},
                    },
                },
                "spread": -12.5,
            },
        ]
    }
    with patch.object(mod, "_throttled_get", return_value=payload):
        row = mod._odds_row(event)
    assert row["home_close_spread"] == -5.5
    assert row["n_books"] == 1
