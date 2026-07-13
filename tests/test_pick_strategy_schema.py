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
    # Corrupt bet_type keeps an explicit disable (fail closed) instead of None.
    assert validate_strategy_entry({"bet_type": "parlay", "enabled": True}) == {
        "enabled": False
    }
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


def test_validate_strategy_entry_drops_non_finite_numbers() -> None:
    """NaN/inf thresholds must not load — Hubáček floors would fail open."""
    cleaned = validate_strategy_entry(
        {
            "bet_type": "moneyline",
            "min_ev_pct": float("nan"),
            "min_market_gap_pp": float("inf"),
            "min_win_confidence_pp": 20.0,
            "ml_lo": float("-inf"),
            "ml_hi": 200,
        }
    )
    assert cleaned is not None
    assert "min_ev_pct" not in cleaned
    assert "min_market_gap_pp" not in cleaned
    assert "ml_lo" not in cleaned
    assert cleaned["min_win_confidence_pp"] == 20.0
    assert cleaned["ml_hi"] == 200.0

    """String 'false' must not become True via bool('false')."""
    cleaned = validate_strategy_entry({"bet_type": "spread", "enabled": "false"})
    assert cleaned is not None
    assert cleaned["enabled"] is False
    assert validate_strategy_entry({"enabled": "true"})["enabled"] is True
    assert validate_strategy_entry({"enabled": "off"})["enabled"] is False


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
    # Corrupt bet_type keeps enabled=false so gated leagues do not fall open.
    assert payload["bad_league"] == {"enabled": False}
    assert "also_bad" not in payload
    assert "" not in payload


def test_corrupt_league_bet_type_does_not_reenable_via_default() -> None:
    """Invalid NFL bet_type used to drop the block → default enabled=true."""
    from unittest.mock import patch

    from web.pick_strategy import get_pick_thresholds

    payload = validate_pick_strategy_payload(
        {
            "nfl": {"bet_type": "spreads", "enabled": False},
            "default": {"enabled": True, "bet_type": "moneyline"},
        }
    )
    assert payload["nfl"] == {"enabled": False}

    with patch("web.pick_strategy.load_pick_strategy", return_value=payload):
        thresholds = get_pick_thresholds("nfl")
    assert thresholds["enabled"] is False


def test_null_or_garbage_enabled_fails_closed() -> None:
    """``enabled: null`` / unknown strings must not omit the key (fail-open)."""
    assert validate_strategy_entry({"bet_type": "spread", "enabled": None}) == {
        "bet_type": "spread",
        "enabled": False,
    }
    assert validate_strategy_entry({"bet_type": "spread", "enabled": "nope"})[
        "enabled"
    ] is False


def test_load_pick_strategy_uses_schema_and_keeps_live_gates() -> None:
    from web.hubacek_picks import clear_strategy_cache
    from web.pick_strategy import get_pick_thresholds, load_pick_strategy

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    config = load_pick_strategy()
    assert isinstance(config.get("default"), dict)
    assert config["mlb"]["enabled"] is True
    assert config["nfl"]["enabled"] is False
    assert config["cbb"]["enabled"] is True
    assert get_pick_thresholds("mlb")["min_market_gap_pp"] == 6.0
