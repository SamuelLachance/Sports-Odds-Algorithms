"""Unit tests for live multi-book odds enrichment."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.live_odds_enrichment import (  # noqa: E402
    apply_enrichment_to_market,
    best_american_odds,
    best_available_for_pick,
    enrich_market_dict,
    enrichment_budget_exhausted,
    line_shopping_edge_from_market,
    line_shopping_fields_for_pick,
    multi_book_enabled,
    reset_enrichment_budget,
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
    open_home_ml: int | None = None,
    open_away_ml: int | None = None,
) -> dict:
    item = {
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
    # Mirror ESPN's real shape: open.moneyLine.american is a signed string.
    if open_home_ml is not None:
        item["homeTeamOdds"]["open"] = {
            "moneyLine": {"american": f"{open_home_ml:+d}"}
        }
    if open_away_ml is not None:
        item["awayTeamOdds"]["open"] = {
            "moneyLine": {"american": f"{open_away_ml:+d}"}
        }
    return item


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
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "0"}, clear=False):
        assert multi_book_enabled("nba") is False
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "1"}, clear=False):
        assert multi_book_enabled("nba") is True
        assert multi_book_enabled("epl") is False


def test_multi_book_skipped_during_fast_daily_build() -> None:
    env = {"FAST_DAILY_BUILD": "1"}
    # Clear LIVE_MULTI_BOOK so the fast-build default applies.
    with patch.dict("os.environ", env, clear=False):
        os.environ.pop("LIVE_MULTI_BOOK", None)
        assert multi_book_enabled("nba") is False
    with patch.dict(
        "os.environ",
        {"FAST_DAILY_BUILD": "1", "LIVE_MULTI_BOOK": "1"},
        clear=False,
    ):
        assert multi_book_enabled("nba") is True


def test_summarize_extracts_open_moneylines_and_providers() -> None:
    items = [
        _book_item(
            home_ml=-150, away_ml=130, name="BookA", open_home_ml=-130, open_away_ml=110
        ),
        _book_item(
            home_ml=-140, away_ml=120, name="BookB", open_home_ml=-125, open_away_ml=105
        ),
        _book_item(
            home_ml=-160, away_ml=140, name="BookC", open_home_ml=-135, open_away_ml=115
        ),
    ]
    summary = summarize_book_items(items)
    assert summary["n_books"] == 3
    assert summary["book_providers"] == ["BookA", "BookB", "BookC"]
    # Median consensus opens across the three books.
    assert summary["open_home_moneyline"] == -130
    assert summary["open_away_moneyline"] == 110


def test_summarize_n_books_counts_only_parsed_items() -> None:
    junk = {"provider": {"name": "GhostBook"}}  # no odds fields at all
    items = [
        _book_item(home_ml=-150, away_ml=130, name="BookA"),
        junk,
        _book_item(home_ml=-140, away_ml=120, name="BookB"),
    ]
    summary = summarize_book_items(items)
    assert summary["n_books"] == 2
    assert summary["book_providers"] == ["BookA", "BookB"]
    # No open objects supplied → open fields absent, never fabricated.
    assert "open_home_moneyline" not in summary
    assert "open_away_moneyline" not in summary


def test_summarize_all_unparsed_returns_empty() -> None:
    assert summarize_book_items([{"provider": {"name": "GhostBook"}}]) == {}


def test_apply_enrichment_to_market_pure_merge() -> None:
    market = {"home_moneyline": -110, "away_moneyline": 100}
    assert apply_enrichment_to_market(dict(market), {}) == market
    out = apply_enrichment_to_market(
        dict(market),
        {"n_books": 3, "best_home_ml": -105, "open_home_moneyline": -120},
    )
    assert out["n_books"] == 3
    assert out["open_home_moneyline"] == -120
    assert out["line_shopping_edge_pp"] is not None


def test_budget_cutoff_disables_enrichment_for_rest_of_build() -> None:
    reset_enrichment_budget()
    calls = {"n": 0}

    def _slow_fetch(url: str, timeout: int = 5, retries: int = 1) -> dict:
        calls["n"] += 1
        time.sleep(0.05)
        return {"items": [_book_item(home_ml=-150, away_ml=130)]}

    market = {"home_moneyline": -145, "away_moneyline": 125}
    env = {"LIVE_MULTI_BOOK": "1", "LIVE_MULTI_BOOK_BUDGET_S": "0.02"}
    try:
        with patch.dict("os.environ", env, clear=False):
            with patch("web.live_odds_enrichment._get_json", side_effect=_slow_fetch):
                first = enrich_market_dict(dict(market), "nba", "401")
                assert enrichment_budget_exhausted()
                second = enrich_market_dict(dict(market), "nba", "402")
    finally:
        reset_enrichment_budget()

    assert first["n_books"] == 1  # first fetch completed and merged
    assert second == market  # budget spent → input returned unchanged
    assert calls["n"] == 1  # no second network call


def test_reset_enrichment_budget_restores_fetching() -> None:
    from web.live_odds_enrichment import _charge_budget

    reset_enrichment_budget()
    _charge_budget(1.0)
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK_BUDGET_S": "0.5"}, clear=False):
        assert enrichment_budget_exhausted() is True
        reset_enrichment_budget()
        assert enrichment_budget_exhausted() is False
