"""Walk-forward features for MLB RunCast + XGBoost layer."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from web.mlb_efficiency import RunsEfficiencyState, expected_runs_for_matchup
from web.mlb_runs import expected_margin, simulate_home_win_probability

FEATURE_NAMES: tuple[str, ...] = (
    "home_rs_ewma",
    "home_ra_ewma",
    "away_rs_ewma",
    "away_ra_ewma",
    "net_ewma_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_games",
    "away_games",
    "pitcher_margin",
    "expected_home_runs",
    "expected_away_runs",
    "expected_margin",
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
        return 2.0
    return max((current - last).days, 0)


def build_matchup_features(
    home: str,
    away: str,
    *,
    efficiency: RunsEfficiencyState,
    game_date: str,
    last_game_date: dict[str, str],
    team_game_counts: dict[str, int],
    pitcher_margin: float = 0.0,
    hca: float = 0.22,
) -> dict[str, float]:
    home = home.lower()
    away = away.lower()
    home_prof = efficiency.team(home)
    away_prof = efficiency.team(away)
    pitcher_home_adj = max(min(pitcher_margin, 0.85), -0.85)
    pitcher_away_adj = -pitcher_home_adj
    home_mu, away_mu = expected_runs_for_matchup(
        home,
        away,
        efficiency,
        hca=hca,
        pitcher_home_adj=pitcher_home_adj,
        pitcher_away_adj=pitcher_away_adj,
    )
    margin = expected_margin(home_mu, away_mu)
    home_rest = _rest_days(last_game_date.get(home), game_date)
    away_rest = _rest_days(last_game_date.get(away), game_date)
    return {
        "home_rs_ewma": home_prof.rs_ewma,
        "home_ra_ewma": home_prof.ra_ewma,
        "away_rs_ewma": away_prof.rs_ewma,
        "away_ra_ewma": away_prof.ra_ewma,
        "net_ewma_diff": home_prof.net_ewma - away_prof.net_ewma,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_b2b": 1.0 if home_rest <= 1.0 else 0.0,
        "away_b2b": 1.0 if away_rest <= 1.0 else 0.0,
        "home_games": float(team_game_counts.get(home, home_prof.games)),
        "away_games": float(team_game_counts.get(away, away_prof.games)),
        "pitcher_margin": pitcher_margin,
        "expected_home_runs": home_mu,
        "expected_away_runs": away_mu,
        "expected_margin": margin,
    }


def features_to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)


def raw_prob_from_runs_sim(
    home: str,
    away: str,
    *,
    efficiency: RunsEfficiencyState,
    pitcher_margin: float = 0.0,
    hca: float = 0.22,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    pitcher_home_adj = max(min(pitcher_margin, 0.85), -0.85)
    pitcher_away_adj = -pitcher_home_adj
    home_mu, away_mu = expected_runs_for_matchup(
        home,
        away,
        efficiency,
        hca=hca,
        pitcher_home_adj=pitcher_home_adj,
        pitcher_away_adj=pitcher_away_adj,
    )
    margin = expected_margin(home_mu, away_mu)
    prob = simulate_home_win_probability(home_mu, away_mu, seed=seed)
    return round(home_mu, 2), round(away_mu, 2), margin, prob
