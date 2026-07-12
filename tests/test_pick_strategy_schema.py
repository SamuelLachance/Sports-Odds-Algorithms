"""Validate pick_strategy.json schema helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.pick_strategy_schema import (  # noqa: E402
    validate_pick_strategy_payload,
    validate_strategy_entry,
)


def test_validate_strategy_entry_rejects_bad_bet_type() -> None:
    assert validate_strategy_entry({"bet_type": "parlay", "enabled": True}) is None
    assert validate_strategy_entry("not-a-dict") is None


def test_validate_strategy_entry_coerces_numbers_and_bools() -> None:
    cleaned = validate_strategy_entry(
        {
            "bet_type": "Spread",
            "enabled": 1,
            "min_ev_pct": "2.5",
            "min_market_gap_pp": 4,
            "backtest_bets": "12",
            "allowed_sides": ["Home", "away", 3],
            "note": "tuned",
            "garbage": {"nested": True},
        }
    )
    assert cleaned is not None
    assert cleaned["bet_type"] == "spread"
    assert cleaned["enabled"] is True
    assert cleaned["min_ev_pct"] == 2.5
    assert cleaned["min_market_gap_pp"] == 4.0
    assert cleaned["backtest_bets"] == 12
    assert cleaned["allowed_sides"] == ["home", "away"]
    assert cleaned["note"] == "tuned"
    assert "garbage" not in cleaned


def test_validate_pick_strategy_payload_drops_corrupt_entries() -> None:
    payload = validate_pick_strategy_payload(
        {
            "policy": "ok",
            "generated_at": "2026-07-12",
            "default": {"min_ev_pct": 2.0, "enabled": True},
            "nba": {"bet_type": "spread", "min_spread_cover_gap_pp": 10},
            "bad_league": {"bet_type": "futures"},
            "also_bad": [1, 2, 3],
            "": {"enabled": True},
        }
    )
    assert payload["policy"] == "ok"
    assert payload["generated_at"] == "2026-07-12"
    assert payload["default"]["min_ev_pct"] == 2.0
    assert payload["nba"]["bet_type"] == "spread"
    assert "bad_league" not in payload
    assert "also_bad" not in payload
    assert "" not in payload


def test_load_pick_strategy_uses_schema_and_keeps_live_gates() -> None:
    from web.hubacek_picks import clear_strategy_cache
    from web.pick_strategy import get_pick_thresholds, load_pick_strategy

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    config = load_pick_strategy()
    assert isinstance(config.get("default"), dict)
    assert config["mlb"]["enabled"] is True
    assert config["nfl"]["enabled"] is False
    assert get_pick_thresholds("mlb")["min_market_gap_pp"] == 6.7
