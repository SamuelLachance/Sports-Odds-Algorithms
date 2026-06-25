"""Tests for nfelo-style football prediction layer."""

from web.football_pred_model import (
    calc_shift,
    elo_to_prob,
    is_football_league,
    predict_matchup_from_model,
    build_football_model,
)


def test_is_football_league() -> None:
    assert is_football_league("nfl")
    assert is_football_league("cfb")
    assert not is_football_league("nba")


def test_elo_to_prob_favorite() -> None:
    prob = elo_to_prob(50.0, z=400.0)
    assert prob > 0.5


def test_calc_shift_positive_when_beat_expectation() -> None:
    shift = calc_shift(10.0, -3.0, -3.0, 9.0, 10.0, 0.0, True)
    assert shift > 0


def test_predict_matchup_from_built_model() -> None:
    games = []
    for _ in range(25):
        games.append(("kc", "den", "Chiefs", "Broncos", 27, 20))
        games.append(("den", "kc", "Broncos", "Chiefs", 17, 24))

    model = build_football_model(games, "nfl")
    assert model is not None
    result = predict_matchup_from_model(model, "kc", "den")
    assert result is not None
    assert 0 < result["home_win_probability"] < 100


def test_projected_spread_uses_book_sign_when_home_favored() -> None:
    """Negative projected_spread = home favored (matches market spread convention)."""
    games = []
    for _ in range(25):
        games.append(("kc", "den", "Chiefs", "Broncos", 30, 10))
        games.append(("den", "kc", "Broncos", "Chiefs", 14, 28))

    model = build_football_model(games, "nfl")
    assert model is not None
    result = predict_matchup_from_model(model, "kc", "den")
    assert result is not None
    assert result["home_win_probability"] > 50
    assert result["projected_spread"] < 0


def test_projected_spread_layer_margin_for_agreement() -> None:
    """Third-layer spread agreement expects positive home margin from book-sign spread."""
    from web.blend_service import _layer_home_margin

    layer = {"projected_spread": -3.5}
    assert _layer_home_margin(layer, "nfl") == 3.5
