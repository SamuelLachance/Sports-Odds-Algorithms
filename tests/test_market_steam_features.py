"""Unit tests for shared market/steam helpers (no invented open=close)."""

from __future__ import annotations

from web.market_steam_features import soccer_steam_from_row, steam_features_from_game


def test_steam_requires_real_open_close() -> None:
    close_only = steam_features_from_game(
        {"home_ml": -150, "away_ml": 130, "home_spread": -3.5}
    )
    assert close_only["has_market"] == 1.0
    assert close_only["has_steam"] == 0.0
    assert close_only["ml_steam_pp"] == 0.0

    with_open = steam_features_from_game(
        {
            "home_ml": -150,
            "away_ml": 130,
            "home_open_ml": -120,
            "away_open_ml": 100,
            "home_spread": -3.5,
            "home_open_spread": -2.5,
        }
    )
    assert with_open["has_steam"] == 1.0
    assert with_open["has_open_line"] == 1.0
    assert with_open["spread_move"] == -1.0


def test_steam_features_are_finite() -> None:
    feats = steam_features_from_game({})
    assert feats["has_market"] == 0.0
    assert feats["mkt_home_prob"] == 0.5
    assert feats["mkt_total"] == 0.0
    for value in feats.values():
        assert value == value  # not NaN
        assert abs(value) < 1e9


def test_soccer_steam_from_open_close() -> None:
    row = {
        "open_home": 2.10,
        "open_draw": 3.40,
        "open_away": 3.50,
        "close_home": 1.90,
        "close_draw": 3.50,
        "close_away": 4.00,
    }
    feats = soccer_steam_from_row(row)
    assert feats["has_steam"] == 1.0
    assert feats["has_market"] == 1.0
    assert feats["ml_steam_pp"] != 0.0
