"""Walk-forward feature rows for Soccer Path A XGBoost 3-way classifier."""

from __future__ import annotations

from typing import Any

import numpy as np

from web.soccer_efficiency import elo_edge, pi_edge, xg_margin

FEATURE_NAMES: tuple[str, ...] = (
    "elo_edge",
    "pi_expected_gd",
    "dc_lam",
    "dc_mu",
    "xg_margin",
    "form_diff",
    "stat_home_prob",
    "stat_draw_prob",
    "stat_away_prob",
    "home_games",
    "away_games",
)


def build_matchup_features(
    *,
    elo_home: float,
    elo_away: float,
    pi_expected_gd: float,
    dc_lam: float,
    dc_mu: float,
    form_diff: float,
    stat_home: float,
    stat_draw: float,
    stat_away: float,
    home_games: int,
    away_games: int,
) -> dict[str, float]:
    return {
        "elo_edge": elo_edge(elo_home, elo_away),
        "pi_expected_gd": pi_edge(pi_expected_gd),
        "dc_lam": float(dc_lam),
        "dc_mu": float(dc_mu),
        "xg_margin": xg_margin(dc_lam, dc_mu),
        "form_diff": float(form_diff),
        "stat_home_prob": stat_home,
        "stat_draw_prob": stat_draw,
        "stat_away_prob": stat_away,
        "home_games": float(home_games),
        "away_games": float(away_games),
    }


def features_to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([float(feats[name]) for name in FEATURE_NAMES], dtype=float)


def blend_stat_and_xgb(
    stat_probs: tuple[float, float, float],
    xgb_probs: tuple[float, float, float],
    *,
    stat_weight: float = 0.45,
) -> tuple[float, float, float]:
    """Blend stat-layer stack with XGB 3-way output."""
    w = min(max(stat_weight, 0.0), 1.0)
    home = w * stat_probs[0] + (1.0 - w) * xgb_probs[0]
    draw = w * stat_probs[1] + (1.0 - w) * xgb_probs[1]
    away = w * stat_probs[2] + (1.0 - w) * xgb_probs[2]
    total = home + draw + away
    if total <= 0:
        return stat_probs
    scale = 100.0 / total
    return home * scale, draw * scale, away * scale
