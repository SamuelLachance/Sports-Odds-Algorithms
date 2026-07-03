"""Dixon-Coles xG margin helpers for Soccer Path A."""

from __future__ import annotations

import math


def xg_margin(home_lam: float, away_mu: float) -> float:
    """Expected home goals minus away goals."""
    return float(home_lam) - float(away_mu)


def pi_edge(pi_expected_gd: float) -> float:
    """Pi-rating expected goal-difference edge (home perspective)."""
    return float(pi_expected_gd)


def elo_edge(elo_home: float, elo_away: float, home_adv: float = 65.0) -> float:
    """Elo rating differential with home advantage."""
    return float(elo_home) + home_adv - float(elo_away)


def margin_to_home_win_proxy(margin: float, scale: float = 1.35) -> float:
    """Map xG margin to home win probability (%), draw implicit in 3-way head."""
    if scale <= 0:
        scale = 1.35
    prob = 1.0 / (1.0 + math.exp(-margin / scale))
    return round(prob * 100.0, 2)
