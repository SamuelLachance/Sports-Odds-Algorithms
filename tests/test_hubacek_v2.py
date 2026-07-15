"""Unit tests for the Hubáček β system: objective math, portfolio, evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from web.hubacek_v2.eval import evaluate_rounds, hubacek_report, quadrants
from web.hubacek_v2.features import is_market_derived, strip_market_features
from web.hubacek_v2.objective import (
    DecorrelatedLogloss,
    decorrelated_ders,
    decorrelated_loss,
    sigmoid,
)
from web.hubacek_v2.portfolio import (
    expected_profit,
    max_sharpe_bets,
    profit_variance,
    uniform_bets,
)


def test_decorrelated_gradient_matches_numeric() -> None:
    rng = np.random.default_rng(7)
    s = rng.normal(0, 1.5, 64)
    y = (rng.uniform(size=64) < 0.55).astype(float)
    q = np.clip(rng.uniform(0.1, 0.9, 64), 1e-6, 1 - 1e-6)
    q[::7] = np.nan  # rows without a market: plain log-loss only
    c = 0.6
    der1, der2 = decorrelated_ders(s, y, q, c)
    eps = 1e-6
    for i in (0, 3, 7, 21, 63):
        num = (
            decorrelated_loss(np.array([s[i] + eps]), y[i : i + 1], q[i : i + 1], c)
            - decorrelated_loss(np.array([s[i] - eps]), y[i : i + 1], q[i : i + 1], c)
        ) / (2 * eps)
        # der1 is the derivative of the MAXIMIZED objective = -loss.
        assert der1[i] == pytest.approx(-num, abs=1e-5)
    # Hessian stays strictly negative (stability clamp).
    assert np.all(der2 < 0)


def test_decorrelation_term_pushes_away_from_book() -> None:
    # With y ambiguous (p near book), the c-term gradient points away from q.
    s = np.array([0.0])
    y = np.array([1.0])
    q = np.array([0.6])
    base1, _ = decorrelated_ders(s, y, q, 0.0)
    dec1, _ = decorrelated_ders(s, y, q, 0.8)
    # p=0.5 < q=0.6 -> (p-q)<0 -> decorrelation pushes score DOWN vs base.
    assert dec1[0] < base1[0]


def test_catboost_objective_wrapper_shapes() -> None:
    q = np.array([0.5, 0.7, np.nan])
    obj = DecorrelatedLogloss(q, c=0.4)
    ders = obj.calc_ders_range([0.1, -0.2, 0.3], [1, 0, 1], None)
    assert len(ders) == 3
    assert all(len(pair) == 2 for pair in ders)
    weighted = obj.calc_ders_range([0.1, -0.2, 0.3], [1, 0, 1], [2.0, 2.0, 2.0])
    assert weighted[0][0] == pytest.approx(2.0 * ders[0][0])


def test_max_sharpe_allocates_only_positive_ev() -> None:
    probs = np.array([0.60, 0.45, 0.52])
    odds = np.array([2.0, 1.8, 2.1])  # EVs: +0.20, -0.19, +0.092
    bets = max_sharpe_bets(probs, odds)
    assert bets[1] == 0.0
    assert bets.sum() == pytest.approx(1.0)
    assert bets[0] > bets[2] > 0  # bigger edge earns the bigger allocation


def test_max_sharpe_beats_uniform_sharpe() -> None:
    rng = np.random.default_rng(11)
    probs = np.clip(rng.uniform(0.45, 0.75, 12), 0, 1)
    odds = 1.0 / np.clip(probs - rng.uniform(0.0, 0.08, 12), 0.05, 0.95)
    b_opt = max_sharpe_bets(probs, odds)
    b_unif = uniform_bets(probs, odds)
    if b_opt.sum() > 0 and b_unif.sum() > 0:
        sharpe_opt = expected_profit(probs, odds, b_opt) / np.sqrt(profit_variance(probs, odds, b_opt))
        sharpe_unif = expected_profit(probs, odds, b_unif) / np.sqrt(profit_variance(probs, odds, b_unif))
        assert sharpe_opt >= sharpe_unif - 1e-9


def test_max_sharpe_never_bets_both_sides() -> None:
    probs = np.array([0.65, 0.40])  # home & away of the SAME game, both "+EV"
    odds = np.array([1.7, 2.9])
    bets = max_sharpe_bets(probs, odds, game_ids=np.array([1, 1]))
    assert (bets > 0).sum() <= 1


def test_quadrants_sum_to_100() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.8, 200)
    q = np.clip(p + rng.normal(0, 0.1, 200), 0.05, 0.95)
    y = (rng.uniform(size=200) < p).astype(float)
    quad = quadrants(p, q, y)
    assert quad["consensus"] + quad["upset"] + quad["missed"] + quad["spotted"] == pytest.approx(100.0, abs=0.1)


def _toy_oos(n_days: int = 40, games_per_day: int = 8, edge: float = 0.06, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for _ in range(games_per_day):
            p_true = rng.uniform(0.35, 0.75)
            q = np.clip(p_true + rng.normal(0, 0.05), 0.05, 0.95)
            p_model = np.clip(p_true + rng.normal(0, 0.05 - edge * 0.3), 0.05, 0.95)
            rows.append(
                {
                    "date": f"2024-01-{d + 1:02d}" if d < 28 else f"2024-02-{d - 27:02d}",
                    "season": 2024,
                    "home_win": float(rng.uniform() < p_true),
                    "model_prob": p_model,
                    "market_home_prob": q,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_rounds_runs_and_reports() -> None:
    oos = _toy_oos()
    res = evaluate_rounds(oos, price="close", phi=0.1, strategy="opt", round_key="date", margin_haircut=0.025)
    assert res is not None
    assert res["n_rounds"] > 30
    assert res["rho_model_book"] is not None
    assert -100 < res["mean_profit_per_round_pct"] < 100
    report = hubacek_report(oos, phis=(0.0, 0.1), prices=("close",))
    assert report["settings"]
    assert "best_opt" in report


def test_phi_filter_reduces_bets() -> None:
    oos = _toy_oos()
    loose = evaluate_rounds(oos, phi=0.0, strategy="unif", round_key="date", margin_haircut=0.025)
    tight = evaluate_rounds(oos, phi=0.2, strategy="unif", round_key="date", margin_haircut=0.025)
    assert loose["n_bets"] > tight["n_bets"]


def test_market_feature_registry() -> None:
    assert is_market_derived("mkt_home_prob")
    assert is_market_derived("ml_steam_pp")
    assert is_market_derived("reverse_line_move")
    assert not is_market_derived("elo_diff")
    cols = ["elo_diff", "mkt_home_prob", "has_steam", "adj_epa_off_diff", "n_books"]
    assert strip_market_features(cols) == ["elo_diff", "adj_epa_off_diff"]
