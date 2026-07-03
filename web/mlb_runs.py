"""Monte Carlo run simulation for MLB win probability (mlb-bet-engine pattern)."""

from __future__ import annotations

import numpy as np

DEFAULT_SIMS = 2500
RUN_SD = 1.15
ENV_SD = 0.07


def simulate_home_win_probability(
    home_mu: float,
    away_mu: float,
    *,
    n_sims: int = DEFAULT_SIMS,
    run_sd: float = RUN_SD,
    env_sd: float = ENV_SD,
    seed: int = 42,
) -> float:
    """Correlated log-normal environment + Gaussian run draws."""
    rng = np.random.default_rng(seed)
    env = rng.lognormal(mean=0.0, sigma=env_sd, size=n_sims)
    home_draws = np.maximum(rng.normal(home_mu * env, run_sd), 0.0)
    away_draws = np.maximum(rng.normal(away_mu * env, run_sd), 0.0)
    home_wins = float(np.mean(home_draws > away_draws))
    ties = float(np.mean(home_draws == away_draws))
    prob = home_wins + 0.5 * ties
    return round(min(max(prob * 100.0, 1.0), 99.0), 2)


def expected_margin(home_mu: float, away_mu: float) -> float:
    return round(home_mu - away_mu, 2)
