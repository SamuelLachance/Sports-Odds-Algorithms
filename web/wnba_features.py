"""Walk-forward feature rows for WNBA Elo + efficiency + schedule context."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from web.wnba_efficiency import EfficiencyState, TeamEfficiency
from web.wnba_elo import EloState, elo_home_win_prob, elo_predicted_margin

FEATURE_NAMES: tuple[str, ...] = (
    "elo_diff",
    "elo_home_prob",
    "elo_margin",
    "home_net_ewma",
    "away_net_ewma",
    "net_ewma_diff",
    "home_ortg",
    "home_drtg",
    "away_ortg",
    "away_drtg",
    "pace_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_games",
    "away_games",
)


def _parse_date(iso_date: str) -> datetime | None:
    raw = str(iso_date or "").strip()[:10]
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None


def _rest_days(last_date: str | None, game_date: str) -> float:
    last = _parse_date(last_date or "")
    current = _parse_date(game_date)
    if not last or not current:
        return 3.0
    return max((current - last).days, 0)


def build_matchup_features(
    home: str,
    away: str,
    *,
    elo: EloState,
    efficiency: EfficiencyState,
    game_date: str,
    last_game_date: dict[str, str],
    team_game_counts: dict[str, int],
) -> dict[str, float]:
    home = home.lower()
    away = away.lower()
    home_elo = elo.rating(home)
    away_elo = elo.rating(away)
    home_eff = efficiency.team(home)
    away_eff = efficiency.team(away)

    home_rest = _rest_days(last_game_date.get(home), game_date)
    away_rest = _rest_days(last_game_date.get(away), game_date)

    return {
        "elo_diff": home_elo - away_elo,
        "elo_home_prob": elo_home_win_prob(home_elo, away_elo),
        "elo_margin": elo_predicted_margin(home_elo, away_elo),
        "home_net_ewma": home_eff.net_ewma,
        "away_net_ewma": away_eff.net_ewma,
        "net_ewma_diff": home_eff.net_ewma - away_eff.net_ewma,
        "home_ortg": home_eff.ortg,
        "home_drtg": home_eff.drtg,
        "away_ortg": away_eff.ortg,
        "away_drtg": away_eff.drtg,
        "pace_diff": home_eff.pace - away_eff.pace,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_b2b": 1.0 if home_rest <= 1.0 else 0.0,
        "away_b2b": 1.0 if away_rest <= 1.0 else 0.0,
        "home_games": float(team_game_counts.get(home, home_eff.games)),
        "away_games": float(team_game_counts.get(away, away_eff.games)),
    }


def features_to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)


def raw_prob_from_elo_and_efficiency(
    home: str,
    away: str,
    *,
    elo: EloState,
    efficiency: EfficiencyState,
    sigma: float,
    hca: float,
) -> tuple[float, float, float, float]:
    from web.wnba_efficiency import expected_margin_from_efficiency, margin_to_home_win_prob

    home_eff = efficiency.team(home)
    away_eff = efficiency.team(away)
    _hs, _as, eff_margin = expected_margin_from_efficiency(home_eff, away_eff, hca=hca)
    elo_margin = elo_predicted_margin(elo.rating(home), elo.rating(away))
    margin = 0.45 * eff_margin + 0.55 * elo_margin
    elo_prob = elo_home_win_prob(elo.rating(home), elo.rating(away)) * 100.0
    eff_prob = margin_to_home_win_prob(margin, sigma=sigma)
    blended = 0.35 * elo_prob + 0.65 * eff_prob
    return _hs, _as, margin, blended
