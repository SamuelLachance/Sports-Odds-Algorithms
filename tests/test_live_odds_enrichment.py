"""Unit tests for live multi-book odds enrichment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.live_odds_enrichment import (  # noqa: E402
    apply_enrichment_to_market,
    best_american_odds,
    best_available_for_pick,
    enrich_market_dict,
    enrichment_budget_exhausted,
    enrichment_budget_remaining_s,
    fetch_multi_book_odds,
    line_shopping_edge_from_market,
    line_shopping_fields_for_pick,
    line_shopping_status,
    multi_book_enabled,
    odds_path_for_league,
    reset_enrichment_budget,
    shopping_edge_pp,
    summarize_book_items,
)
from web.nba_odds_espn import _valid_american  # noqa: E402


def test_valid_american_maps_even_zero_to_plus_100() -> None:
    assert _valid_american(0) == 100.0
    assert _valid_american(100) == 100.0
    assert _valid_american(-110) == -110.0
    assert _valid_american(1.91) is None
    assert _valid_american(None) is None


def test_valid_handicap_line_rejects_ml_sized_dumps() -> None:
    from web.nba_odds_espn import _valid_handicap_line

    assert _valid_handicap_line(-1.5, max_abs=7.0) == -1.5
    assert _valid_handicap_line(0.0, max_abs=7.0) == 0.0
    assert _valid_handicap_line(152.0, max_abs=7.0) is None
    assert _valid_handicap_line(-110.0, max_abs=5.0) is None
    # Raised college caps must still reject juice/ML magnitudes.
    assert _valid_handicap_line(-110.0, max_abs=120.0) is None
    assert _valid_handicap_line(55.5, max_abs=120.0) == 55.5


def test_nba_provider_line_rejects_ml_sized_point_spread() -> None:
    from web.nba_odds_espn import _provider_line

    item = {
        "provider": {"name": "DraftKings"},
        "spread": -7.5,
        "homeTeamOdds": {
            "favorite": True,
            "moneyLine": -280,
            "close": {
                "pointSpread": {"american": -280},
                "moneyLine": {"american": -280},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "moneyLine": 230,
            "close": {
                "pointSpread": {"american": 230},
                "moneyLine": {"american": 230},
                "spread": {"american": -110},
            },
        },
    }
    line = _provider_line(item)
    # Nested pointSpread dumped ML; fall back to flat signed spread.
    assert line["home_close_spread"] == -7.5
    assert line["away_close_spread"] == 7.5


def test_summarize_mlb_rejects_fake_run_line_from_moneyline() -> None:
    """Live multi-book MLB path must use run-line validation (not NBA parser)."""
    item = {
        "provider": {"name": "FanDuel"},
        "spread": 1.5,
        "homeTeamOdds": {
            "favorite": False,
            "moneyLine": 140,
            "close": {
                "pointSpread": {"american": 140},
                "moneyLine": {"american": 140},
                "spread": {"american": -115},
            },
        },
        "awayTeamOdds": {
            "favorite": True,
            "moneyLine": -160,
            "close": {
                "pointSpread": {"american": -160},
                "moneyLine": {"american": -160},
                "spread": {"american": -105},
            },
        },
    }
    summary = summarize_book_items([item], league="mlb")
    assert summary["consensus_home_spread"] == 1.5
    assert abs(summary["consensus_home_spread"]) < 10


def test_mlb_provider_line_drops_moneyline_as_run_line() -> None:
    from web.mlb_odds_espn import _provider_line_mlb

    item = {
        "homeTeamOdds": {
            "favorite": False,
            "moneyLine": 150,
            "close": {
                "moneyLine": {"american": 150},
                "pointSpread": {"american": 152},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": True,
            "moneyLine": -170,
            "close": {
                "moneyLine": {"american": -170},
                "pointSpread": {"american": -152},
                "spread": {"american": -110},
            },
        },
        "spread": 1.5,
        "overUnder": 8.5,
    }
    line = _provider_line_mlb(item)
    assert line["home_close_ml"] == 150.0
    assert line["away_close_ml"] == -170.0
    # Nested pointSpread was ML-sized; fall back to signed flat spread magnitude.
    assert line["home_close_spread"] == 1.5
    assert line["away_close_spread"] == -1.5


def test_mlb_provider_line_keeps_real_run_line() -> None:
    from web.mlb_odds_espn import _provider_line_mlb

    item = {
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "moneyLine": {"american": -140},
                "pointSpread": {"american": -1.5},
                "spread": {"american": -115},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "moneyLine": {"american": 120},
                "pointSpread": {"american": 1.5},
                "spread": {"american": -105},
            },
        },
    }
    line = _provider_line_mlb(item)
    assert line["home_close_spread"] == -1.5
    assert line["away_close_spread"] == 1.5
    assert line["home_spread_odds"] == -115.0


def test_nhl_provider_line_drops_moneyline_as_puck_line() -> None:
    from web.nhl_odds_espn import _provider_line_nhl

    item = {
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "moneyLine": {"american": -150},
                "pointSpread": {"american": -150},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "moneyLine": {"american": 130},
                "pointSpread": {"american": 130},
                "spread": {"american": -110},
            },
        },
        "spread": 1.5,
    }
    line = _provider_line_nhl(item)
    assert line["home_close_spread"] == -1.5
    assert line["away_close_spread"] == 1.5


def test_nba_odds_collect_preserves_missing_spread_juice() -> None:
    """Historical collectors must leave missing juice as None, not invent -110."""
    from web import nba_odds_espn as nba_odds

    assert nba_odds._median([None, None]) is None
    # Simulate the collect_day_rows packing contract after the fix.
    consensus = {"home_spread_odds": None, "away_spread_odds": None}
    row = {
        "home_spread_odds": consensus["home_spread_odds"],
        "away_spread_odds": consensus["away_spread_odds"],
    }
    assert row["home_spread_odds"] is None
    assert row["away_spread_odds"] is None
    # Falsy EVEN would have been poisoned by `or -110` before _valid_american;
    # after validation, 0 maps to 100 and must survive packing.
    assert (100.0 or -110) == 100.0


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


def test_line_shopping_fields_use_side_edge_not_game_max() -> None:
    """Home pick must not inherit the away side's larger shopping edge."""
    from web.live_odds_enrichment import shopping_edge_pp

    market = {
        "n_books": 4,
        "home_moneyline": -110,
        "away_moneyline": 100,
        "best_home_ml": -105,  # small home edge
        "best_away_ml": 130,  # large away edge
        "line_shopping_edge_pp": 99.0,  # poisoned game-level max
        "consensus_home_ml": -108,
        "consensus_away_ml": 105,
    }
    home_fields = line_shopping_fields_for_pick(
        market, side="home", bet_type="moneyline"
    )
    away_fields = line_shopping_fields_for_pick(
        market, side="away", bet_type="moneyline"
    )
    home_edge = shopping_edge_pp(-110, -105)
    away_edge = shopping_edge_pp(100, 130)
    assert home_edge is not None and away_edge is not None
    assert away_edge > home_edge
    assert home_fields["line_shopping_edge_pp"] == pytest.approx(home_edge)
    assert home_fields["best_vs_espn_pp"] == pytest.approx(home_edge)
    assert away_fields["line_shopping_edge_pp"] == pytest.approx(away_edge)


def test_line_shopping_omits_edge_when_espn_side_missing() -> None:
    """Missing ESPN juice for the pick side must not inherit game-level max."""
    market = {
        "n_books": 4,
        "home_moneyline": None,
        "away_moneyline": 100,
        "best_home_ml": -105,
        "best_away_ml": 130,
        "line_shopping_edge_pp": 99.0,
    }
    home_fields = line_shopping_fields_for_pick(
        market, side="home", bet_type="moneyline"
    )
    assert home_fields["best_available_odds"] == -105
    assert "line_shopping_edge_pp" not in home_fields
    assert "best_vs_espn_pp" not in home_fields


def test_line_shopping_omits_edge_when_best_missing() -> None:
    """No shopped price → do not attach the game-level shopping edge to the pick."""
    market = {
        "n_books": 4,
        "home_moneyline": -110,
        "away_moneyline": 100,
        "best_home_ml": None,
        "best_away_ml": 130,
        "line_shopping_edge_pp": 99.0,
    }
    home_fields = line_shopping_fields_for_pick(
        market, side="home", bet_type="moneyline"
    )
    assert "best_available_odds" not in home_fields
    assert "line_shopping_edge_pp" not in home_fields


def test_line_shopping_accepts_float_string_espn_odds() -> None:
    """JSON/CSV float strings must not crash pick attach via bare int()."""
    market = {
        "n_books": 3,
        "home_moneyline": "-110.0",
        "away_moneyline": "100.0",
        "best_home_ml": -105,
        "best_away_ml": 105,
    }
    fields = line_shopping_fields_for_pick(market, side="home", bet_type="moneyline")
    assert fields["best_available_odds"] == -105
    assert fields["line_shopping_edge_pp"] == shopping_edge_pp(-110, -105)


def test_line_shopping_reports_ev_at_best_without_changing_espn_ev() -> None:
    from web.bet_advisor import expected_value_pct

    market = {
        "n_books": 4,
        "home_moneyline": -110,
        "away_moneyline": 100,
        "best_home_ml": -105,
        "best_away_ml": 110,
    }
    model_prob = 55.0
    fields = line_shopping_fields_for_pick(
        market,
        side="home",
        bet_type="moneyline",
        model_prob_pct=model_prob,
    )
    espn_ev = expected_value_pct(model_prob, -110)
    best_ev = expected_value_pct(model_prob, -105)
    assert fields["best_available_odds"] == -105
    assert fields["ev_pct_at_best"] == round(best_ev, 2)
    assert fields["ev_pct_at_best"] > round(espn_ev, 2)


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


def test_line_shopping_status_labels() -> None:
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "0"}, clear=False):
        assert line_shopping_status() == "off"
    with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "1"}, clear=False):
        assert line_shopping_status() == "on"
    with patch.dict("os.environ", {"FAST_DAILY_BUILD": "1"}, clear=False):
        os.environ.pop("LIVE_MULTI_BOOK", None)
        assert line_shopping_status() == "skipped_fast_build"
    with patch.dict(
        "os.environ",
        {"FAST_DAILY_BUILD": "1", "LIVE_MULTI_BOOK": "1"},
        clear=False,
    ):
        assert line_shopping_status() == "on"


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


def test_mlb_consensus_n_books_ignores_empty_providers() -> None:
    """Empty ESPN provider shells must not inflate MLB n_books."""
    from web.mlb_odds_espn import _consensus_mlb

    real = {
        "provider": {"name": "DraftKings"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "moneyLine": {"american": -130},
                "pointSpread": {"american": -1.5},
                "spread": {"american": -115},
            },
            "open": {"moneyLine": {"american": -125}},
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "moneyLine": {"american": 110},
                "pointSpread": {"american": 1.5},
                "spread": {"american": -105},
            },
            "open": {"moneyLine": {"american": 105}},
        },
        "close": {"total": 8.5},
        "open": {"total": 8.0},
    }
    empty = {"provider": {"name": "GhostBook"}}
    consensus = _consensus_mlb([real, empty])
    assert consensus["n_books"] == 1
    assert consensus["home_close_ml"] == -130
    assert consensus["away_close_ml"] == 110


def test_nhl_consensus_n_books_ignores_empty_providers() -> None:
    """Empty ESPN provider shells must not inflate NHL n_books."""
    from web.nhl_odds_espn import _consensus_nhl

    real = {
        "provider": {"name": "DraftKings"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "moneyLine": {"american": -140},
                "pointSpread": {"american": -1.5},
                "spread": {"american": -110},
            },
            "open": {"moneyLine": {"american": -135}},
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "moneyLine": {"american": 120},
                "pointSpread": {"american": 1.5},
                "spread": {"american": -110},
            },
            "open": {"moneyLine": {"american": 115}},
        },
        "close": {"total": 6.0},
        "open": {"total": 5.5},
    }
    empty = {"provider": {"name": "GhostBook"}}
    consensus = _consensus_nhl([real, empty])
    assert consensus["n_books"] == 1
    assert consensus["home_close_ml"] == -140
    assert consensus["away_close_ml"] == 120


def test_nba_consensus_n_books_ignores_empty_providers() -> None:
    """Empty ESPN provider shells must not inflate NBA/college n_books."""
    from web.nba_odds_espn import _consensus

    real = {
        "provider": {"name": "DraftKings"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "moneyLine": {"american": -150},
                "pointSpread": {"american": -4.5},
                "spread": {"american": -110},
            },
            "open": {
                "moneyLine": {"american": -140},
                "pointSpread": {"american": -4.0},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "moneyLine": {"american": 130},
                "pointSpread": {"american": 4.5},
                "spread": {"american": -110},
            },
            "open": {"moneyLine": {"american": 120}},
        },
        "close": {"total": 220.5},
        "open": {"total": 218.0},
    }
    empty = {"provider": {"name": "GhostBook"}}
    consensus = _consensus([real, empty])
    assert consensus["n_books"] == 1
    assert consensus["home_close_ml"] == -150
    assert consensus["away_close_ml"] == 130


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
    """Budget accounting must not rely on real wall-clock sleeps (flake-prone)."""
    reset_enrichment_budget()
    calls = {"n": 0}
    clock = {"t": 100.0}

    def _mono() -> float:
        return clock["t"]

    def _slow_fetch(url: str, timeout: int = 5, retries: int = 1) -> dict:
        calls["n"] += 1
        clock["t"] += 0.05
        return {"items": [_book_item(home_ml=-150, away_ml=130)]}

    market = {"home_moneyline": -145, "away_moneyline": 125}
    env = {"LIVE_MULTI_BOOK": "1", "LIVE_MULTI_BOOK_BUDGET_S": "0.02"}
    try:
        with patch.dict("os.environ", env, clear=False):
            with patch("web.live_odds_enrichment.time.monotonic", side_effect=_mono):
                with patch(
                    "web.live_odds_enrichment._get_json",
                    side_effect=_slow_fetch,
                ):
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


def test_odds_path_for_league_known_and_unknown() -> None:
    assert odds_path_for_league("nba") == "basketball/leagues/nba"
    assert odds_path_for_league("NHL") == "hockey/leagues/nhl"
    assert odds_path_for_league("epl") is None
    assert odds_path_for_league("") is None


def test_fetch_skips_network_when_path_or_budget_blocks() -> None:
    """Clear failure paths must not call ESPN or charge budget."""
    reset_enrichment_budget()
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("network should not be reached")

    with patch("web.live_odds_enrichment._get_json", side_effect=_boom):
        with patch.dict("os.environ", {"LIVE_MULTI_BOOK": "1"}, clear=False):
            assert fetch_multi_book_odds("nba", "") == {}
            assert fetch_multi_book_odds("epl", "401") == {}
        with patch.dict(
            "os.environ",
            {"LIVE_MULTI_BOOK": "1", "LIVE_MULTI_BOOK_BUDGET_S": "0"},
            clear=False,
        ):
            assert enrichment_budget_remaining_s() == 0.0
            assert fetch_multi_book_odds("nba", "401") == {}
    assert calls["n"] == 0
    reset_enrichment_budget()


def test_fetch_network_failure_charges_budget_then_soft_fails() -> None:
    reset_enrichment_budget()
    env = {"LIVE_MULTI_BOOK": "1", "LIVE_MULTI_BOOK_BUDGET_S": "0.05"}
    clock = {"t": 50.0}

    def _mono() -> float:
        return clock["t"]

    def _slow_fail(*_a, **_k):
        clock["t"] += 0.06
        raise OSError("espn down")

    try:
        with patch.dict("os.environ", env, clear=False):
            with patch("web.live_odds_enrichment.time.monotonic", side_effect=_mono):
                with patch(
                    "web.live_odds_enrichment._get_json",
                    side_effect=_slow_fail,
                ):
                    assert fetch_multi_book_odds("nba", "401") == {}
                    assert enrichment_budget_exhausted()
                # Second call is a clear budget no-op (no network).
                with patch(
                    "web.live_odds_enrichment._get_json",
                    side_effect=AssertionError("should not fetch"),
                ):
                    assert fetch_multi_book_odds("nba", "402") == {}
    finally:
        reset_enrichment_budget()
