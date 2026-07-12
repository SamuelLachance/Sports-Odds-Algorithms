"""Regression tests for opening-line steam margin nudges."""

from __future__ import annotations

import pytest

from web.cbb_opening import STEAM_MARGIN_THRESHOLD, opening_steam_adjustment


def test_opening_steam_nudges_toward_market_not_away() -> None:
    """Model home+7 vs market home−3 must shrink home margin toward the book."""
    model_margin = 7.0
    market_spread = -3.0  # book: home favored by 3
    # model_spread = -7 → diff = -4 (≥ threshold) → model too home-bullish.
    assert abs((-model_margin) - market_spread) >= STEAM_MARGIN_THRESHOLD

    adj, meta = opening_steam_adjustment(model_margin, market_spread)
    assert meta["steam_signal"] is True
    assert meta["steam_direction"] == "away"
    assert adj < model_margin
    assert adj == pytest.approx(model_margin - min(4.0 * 0.15, 1.5), abs=1e-9)


def test_opening_steam_rejects_bool_false_as_pickem() -> None:
    """False must fail closed like None — not float(False)=0 inventing steam."""
    adj_f, meta_f = opening_steam_adjustment(5.0, False)
    adj_n, meta_n = opening_steam_adjustment(5.0, None)
    assert adj_f == 5.0
    assert meta_f["steam_signal"] is False
    assert adj_n == 5.0
    assert meta_n["steam_signal"] is False
    # True would also coerce to 1.0 — reject bools entirely.
    adj_t, meta_t = opening_steam_adjustment(5.0, True)
    assert adj_t == 5.0
    assert meta_t["steam_signal"] is False


def test_opening_steam_nudges_toward_market_when_model_too_away() -> None:
    """Model home−4 vs market home−8 must increase home margin toward the book."""
    model_margin = -4.0  # away favored by 4 in margin space
    market_spread = -8.0  # book: home favored by 8
    # model_spread = +4 → diff = +12 → model too away-bullish.
    adj, meta = opening_steam_adjustment(model_margin, market_spread)
    assert meta["steam_signal"] is True
    assert meta["steam_direction"] == "home"
    assert adj > model_margin


def test_opening_steam_move_uses_true_opening_not_close() -> None:
    """steam_move must reflect open→live, not close masquerading as open."""
    _, meta = opening_steam_adjustment(
        5.0,
        market_spread=-4.5,
        opening_spread=-3.0,
    )
    assert meta["steam_move"] == pytest.approx(-1.5)
    # Without a true open, ref falls back to market → zero move (fail closed).
    _, meta_no_open = opening_steam_adjustment(5.0, market_spread=-4.5)
    assert meta_no_open["steam_move"] == 0.0


def test_soccer_implied_shift_soft_fails_invalid_odds() -> None:
    """|odds| < 100 must return None, not raise into slate steam meta."""
    from web.soccer_opening import _implied_shift_pp, soccer_opening_steam_meta

    assert _implied_shift_pp(50, -110) is None
    assert _implied_shift_pp(-110, 75) is None
    assert _implied_shift_pp(0, -110) is not None  # ESPN EVEN → +100

    meta = soccer_opening_steam_meta(
        home_ml=-120,
        draw_ml=250,
        away_ml=320,
        open_home_ml=50,  # invalid open — shift skipped, no crash
        open_draw_ml=250,
        open_away_ml=320,
        model_home=48.0,
        model_draw=28.0,
        model_away=24.0,
    )
    assert meta["steam_signal"] is False
    assert meta["implied_shifts_pp"]["home"] is None


def test_soccer_steam_ignores_adverse_move_on_model_favorite() -> None:
    """Largest |shift| into a drop on the model favorite must not confirm steam."""
    from web.soccer_opening import soccer_opening_steam_meta

    # Home is model favorite; open→close home implied drops hard (adverse).
    # Draw/away flat so |home shift| wins the old abs-based steam pick.
    meta = soccer_opening_steam_meta(
        home_ml=100,
        draw_ml=250,
        away_ml=400,
        open_home_ml=-200,
        open_draw_ml=250,
        open_away_ml=400,
        model_home=58.0,
        model_draw=24.0,
        model_away=18.0,
    )
    assert meta["model_best_outcome"] == "home"
    assert meta["implied_shifts_pp"]["home"] is not None
    assert meta["implied_shifts_pp"]["home"] < -2.5
    assert meta["steam_signal"] is False
    assert meta.get("steam_direction") is None


def test_soccer_steam_requires_positive_move_into_favorite() -> None:
    """Confirming steam: market implied prob rises into the model favorite."""
    from web.soccer_opening import soccer_opening_steam_meta

    # Open: home dog; close: home shortens, but market still below model 62%.
    meta = soccer_opening_steam_meta(
        home_ml=-130,
        draw_ml=250,
        away_ml=320,
        open_home_ml=110,
        open_draw_ml=250,
        open_away_ml=280,
        model_home=62.0,
        model_draw=22.0,
        model_away=16.0,
    )
    assert meta["model_best_outcome"] == "home"
    assert meta["steam_signal"] is True
    assert meta["steam_direction"] == "home"
    assert meta["steam_move_pp"] > 0


def test_soccer_steam_edge_uses_raw_model_not_inflated_pick() -> None:
    """Decorrelated pick probs must not be what crosses STEAM_MODEL_EDGE_PP."""
    from web.soccer_opening import STEAM_MODEL_EDGE_PP, soccer_opening_steam_meta

    # Close home ~42.8% de-vig; open→close home implied rises into favorite.
    close_home, close_draw, close_away = 120, 240, 220
    open_home, open_draw, open_away = 160, 240, 200
    display_meta = soccer_opening_steam_meta(
        home_ml=close_home,
        draw_ml=close_draw,
        away_ml=close_away,
        open_home_ml=open_home,
        open_draw_ml=open_draw,
        open_away_ml=open_away,
        model_home=46.5,  # edge ~3.7pp < threshold
        model_draw=28.0,
        model_away=25.5,
    )
    pick_meta = soccer_opening_steam_meta(
        home_ml=close_home,
        draw_ml=close_draw,
        away_ml=close_away,
        open_home_ml=open_home,
        open_draw_ml=open_draw,
        open_away_ml=open_away,
        model_home=47.5,  # Hubáček-inflated edge ~4.7pp
        model_draw=27.5,
        model_away=25.0,
    )
    assert display_meta["model_edge_pp"] < STEAM_MODEL_EDGE_PP
    assert pick_meta["model_edge_pp"] >= STEAM_MODEL_EDGE_PP
    # Callers must pass display probs so borderline edges stay below threshold.
    assert display_meta["steam_signal"] is False
    assert pick_meta["steam_signal"] is True

def test_soccer_opening_does_not_fabricate_open_from_close(monkeypatch) -> None:
    """Missing open must stay None — copying close invents 0pp steam."""
    from web import soccer_opening as so

    block = {
        "moneyline": {
            "home": {"close": {"odds": "-120"}},
            "away": {"close": {"odds": "250"}},
            "draw": {"open": {"odds": "240"}, "close": {"odds": "280"}},
        }
    }
    monkeypatch.setattr(so, "_fetch_event_odds_block", lambda *_a, **_k: block)
    open_h, open_d, open_a = so.fetch_soccer_opening_moneylines("epl", "123")
    assert open_h is None
    assert open_a is None
    assert open_d == 240
