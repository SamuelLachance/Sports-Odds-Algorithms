"""Unit tests for live multi-book odds enrichment."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.live_odds_enrichment import (  # noqa: E402
    best_american_odds,
    best_available_for_pick,
    enrich_market_dict,
    line_shopping_edge_from_market,
    line_shopping_fields_for_pick,
    multi_book_enabled,
    shopping_edge_pp,
    summarize_book_items,
)


def _book_item(
    *,
    home_ml: int,
    away_ml: int,
    home_spread_odds: int = -110,
    away_spread_odds: int = -110,
    home_spread: float = -3.5,
    name: str = "DraftKings",
) -> dict:
    return {
        "provider": {"name": name},
        "spread": home_spread,
        "homeTeamOdds": {
            "favorite": True,
            "moneyLine": home_ml,
            "close": {
                "moneyLine": {"american": home_ml},
                "pointSpread": {"american": home_spread},
                "spread": {"american": home_spread_odds},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "moneyLine": away_ml,
            "close": {
                "moneyLine": {"american": away_ml},
                "pointSpread": {"american": -home_spread},
                "spread": {"american": away_spread_odds},
            },
        },
    }


def test_best_american_odds_prefers_higher_payout() -> None:
    assert best_american_odds([-110, -105, -115]) == -105
    assert best_american_odds([140, 150, 130]) == 150


def test_shopping_edge_pp_positive_when_best_better() -> None:
    edge = shopping_edge_pp(-110, -105)
    assert edge is not None and edge > 0


def test_summarize_book_items_consensus_and_best() -> None:
    items = [
        _book_item(home_ml=-150, away_ml=130, name="BookA"),
        _book_item(home_ml=-140, away_ml=120, name="BookB"),
        _book_item(home_ml=-160, away_ml=140, name="BookC"),
    ]
    summary = summarize_book_items(items)
    assert summary["n_books"] == 3
    assert summary["best_home_ml"] == -140  # best (lowest implied) among favorites
    assert summary["best_away_ml"] == 140
    assert summary["consensus_home_ml"] is not None
    assert summary["consensus_away_ml"] is not None


def test_line_shopping_edge_from_market() -> None:
    enrichment = {"best_home_ml": -105, "best_away_ml": 120}
    edge = line_shopping_edge_from_market(
        enrichment,
        espn_home_ml=-110,
        espn_away_ml=100,
    )
    assert edge is not None and edge > 0


def test_enrich_market_dict_soft_fail() -> None:
    market = {"home_moneyline": -110, "away_moneyline": -110, "provider": "ESPN BET"}
    with patch(
        "web.live_odds_enrichment.fetch_multi_book_odds",
        side_effect=OSError("network down"),
    ):
        out = enrich_market_dict(market, "nba", "401")
    assert out == market


def test_enrich_market_dict_merges_fields() -> None:
    market = {"home_moneyline": -110, "away_moneyline": 100, "provider": "ESPN BET"}
    fake = {
        "n_books": 4,
        "best_home_ml": -105,
        "best_away_ml": 110,
        "best_home_spread": -105,
        "best_away_spread": -110,
        "consensus_home_ml": -108,
        "consensus_away_ml": 105,
    }
    with patch("web.live_odds_enrichment.fetch_multi_book_odds", return_value=fake):
        with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "1"}):
            out = enrich_market_dict(market, "nba", "401")
    assert out["n_books"] == 4
    assert out["best_home_ml"] == -105
    assert out["line_shopping_edge_pp"] is not None


def test_line_shopping_fields_for_pick() -> None:
    market = {
        "n_books": 5,
        "home_moneyline": -110,
        "away_moneyline": 100,
        "best_home_ml": -105,
        "best_away_ml": 110,
        "line_shopping_edge_pp": 1.2,
        "consensus_home_ml": -108,
        "consensus_away_ml": 105,
    }
    fields = line_shopping_fields_for_pick(market, side="home", bet_type="moneyline")
    assert fields["best_available_odds"] == -105
    assert fields["n_books"] == 5
    assert best_available_for_pick(market, side="away") == 110


def test_multi_book_disabled_via_env() -> None:
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "0"}):
        assert multi_book_enabled("nba") is False
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "1"}):
        assert multi_book_enabled("nba") is True
        assert multi_book_enabled("epl") is False
