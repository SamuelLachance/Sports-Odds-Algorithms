"""Soccer Path A 1X2 pick signal flags (predictions only — no official tracking)."""

from __future__ import annotations

from typing import Any

DISAGREEMENT_THRESHOLD_PP = 4.0
HIGH_CONFIDENCE_DISAGREEMENT_PP = 8.0


def _best_outcome(home_p: float, draw_p: float, away_p: float) -> str:
    if draw_p >= home_p and draw_p >= away_p:
        return "draw"
    return "home" if home_p >= away_p else "away"


def build_soccer_pick_signals(
    *,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_ml: int | None = None,
    draw_ml: int | None = None,
    away_ml: int | None = None,
    home_games: int = 0,
    away_games: int = 0,
    steam_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from web.bet_advisor import _odds_edge, soccer_model_moneylines

    signals: dict[str, Any] = {
        "disagreement_signal": False,
        "high_confidence_disagreement": False,
        "model_best_outcome": _best_outcome(home_prob, draw_prob, away_prob),
        "steam_signal": bool((steam_meta or {}).get("steam_signal")),
        "home_games": home_games,
        "away_games": away_games,
    }

    away_proj, draw_proj, home_proj = soccer_model_moneylines(
        home_prob, draw_prob, away_prob
    )
    max_edge = 0.0
    best_edge_outcome: str | None = None
    for outcome, prob, proj, market in (
        ("away", away_prob, away_proj, away_ml),
        ("draw", draw_prob, draw_proj, draw_ml),
        ("home", home_prob, home_proj, home_ml),
    ):
        if market is None:
            continue
        edge = _odds_edge(proj, market, prob)
        if edge > max_edge:
            max_edge = edge
            best_edge_outcome = outcome
    signals["max_edge_pp"] = round(max_edge, 2)
    signals["best_edge_outcome"] = best_edge_outcome
    if max_edge >= DISAGREEMENT_THRESHOLD_PP:
        signals["disagreement_signal"] = True
    if max_edge >= HIGH_CONFIDENCE_DISAGREEMENT_PP:
        signals["high_confidence_disagreement"] = True

    if steam_meta:
        signals["opening_steam"] = steam_meta

    return signals
