"""Nested walk-forward backtest for the NBA ML models.

Honest methodology:
- Expanding-window walk-forward by season: for each test season, train only on
  prior seasons (no lookahead; features are already point-in-time).
- Nested threshold selection: the betting threshold is chosen on a validation
  slice (the last training season), never on the test season.
- Isotonic calibration is fit on the validation slice, not the test season.
- Primary metrics are ROI graded at the REAL closing line and CLV (opening ->
  closing movement in the bet's direction), plus probability calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from web.nba_ml.calibrate import (
    apply_isotonic,
    brier,
    calibration_table,
    fit_isotonic,
    log_loss,
)
from web.nba_ml.model import cover_probability, train_models

MIN_PRIOR_GAMES = 8
MIN_TRAIN_SEASONS = 3
MIN_VALIDATION_BETS = 20
SPREAD_BREAK_EVEN = 0.5238  # -110


def _sane_american(odds: float, default: float = float("nan")) -> float:
    """American odds must have magnitude >= 100; return default otherwise."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return default
    # ESPN EVEN sometimes arrives as numeric 0 — treat as +100.
    if odds == 0:
        return 100.0
    if 100.0 <= abs(odds) <= 100000.0:
        return odds
    return default


def american_profit(odds: float, won: bool) -> float:
    if not won:
        return -1.0
    odds = _sane_american(odds)
    if math.isnan(odds):
        return float("nan")
    return odds / 100.0 if odds > 0 else 100.0 / -odds


def implied_prob(odds: float) -> float:
    odds = _sane_american(float(odds))
    if math.isnan(odds):
        return float("nan")
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def _ats_result(side: str, home_margin: float, home_spread: float) -> str:
    line = home_spread if side == "home" else -home_spread
    diff = (home_margin + line) if side == "home" else (-home_margin + line)
    if diff > 0:
        return "win"
    if diff < 0:
        return "loss"
    return "push"


@dataclass
class FoldConfig:
    ev_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09)
    ensemble_weight: float = 0.5


@dataclass
class BacktestResult:
    ats: dict = field(default_factory=dict)
    moneyline: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    clv: dict = field(default_factory=dict)
    bets: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


def _ats_bets(frame: pd.DataFrame, pred_margin, sigma, ev_threshold):
    cover_home = cover_probability(pred_margin, frame["home_close_spread"].to_numpy(), sigma)
    records = []
    home_spread_odds = frame["home_spread_odds"].to_numpy()
    away_spread_odds = frame["away_spread_odds"].to_numpy()
    margins = frame["home_margin"].to_numpy()
    spreads = frame["home_close_spread"].to_numpy()
    open_spreads = frame["home_open_spread"].to_numpy() if "home_open_spread" in frame else np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if np.isnan(spreads[i]):
            continue
        ch = float(cover_home[i])
        ho = _sane_american(home_spread_odds[i])
        ao = _sane_american(away_spread_odds[i])
        ev_home = (
            ch * american_profit(ho, True) - (1 - ch)
            if not math.isnan(ho)
            else float("-inf")
        )
        ev_away = (
            (1 - ch) * american_profit(ao, True) - ch
            if not math.isnan(ao)
            else float("-inf")
        )
        if not math.isnan(ho) and ev_home >= ev_threshold and ev_home >= ev_away:
            side, odds = "home", ho
        elif not math.isnan(ao) and ev_away >= ev_threshold:
            side, odds = "away", ao
        else:
            continue
        result = _ats_result(side, margins[i], spreads[i])
        won = result == "win"
        profit = 0.0 if result == "push" else american_profit(odds, won)
        if math.isnan(profit):
            continue
        clv = np.nan
        if not np.isnan(open_spreads[i]):
            clv = (open_spreads[i] - spreads[i]) if side == "home" else (spreads[i] - open_spreads[i])
        records.append(
            {
                "market": "ats",
                "side": side,
                "odds": odds,
                "result": result,
                "profit": profit,
                "cover_prob": ch,
                "clv_points": clv,
            }
        )
    return records


def _ml_bets(frame: pd.DataFrame, win_prob, ev_threshold):
    records = []
    home_ml = frame["home_close_ml"].to_numpy()
    away_ml = frame["away_close_ml"].to_numpy()
    home_win = frame["home_win"].to_numpy()
    for i in range(len(frame)):
        if np.isnan(home_ml[i]) or np.isnan(away_ml[i]):
            continue
        home_odds = _sane_american(float(home_ml[i]), default=float("nan"))
        away_odds = _sane_american(float(away_ml[i]), default=float("nan"))
        if np.isnan(home_odds) or np.isnan(away_odds):
            continue
        if abs(home_odds) < 100 or abs(away_odds) < 100:
            continue
        p = float(win_prob[i])
        ev_home = p * american_profit(home_odds, True) - (1 - p)
        ev_away = (1 - p) * american_profit(away_odds, True) - p
        if ev_home >= ev_threshold and ev_home >= ev_away:
            side, odds, won = "home", home_odds, home_win[i] == 1
        elif ev_away >= ev_threshold:
            side, odds, won = "away", away_odds, home_win[i] == 0
        else:
            continue
        records.append(
            {
                "market": "ml",
                "side": side,
                "result": "win" if won else "loss",
                "profit": american_profit(odds, won),
                "win_prob": p,
            }
        )
    return records


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"bets": 0, "roi_pct": 0.0, "units": 0.0, "wins": 0, "losses": 0, "pushes": 0}
    profits = [r["profit"] for r in records]
    wins = sum(1 for r in records if r["result"] == "win")
    losses = sum(1 for r in records if r["result"] == "loss")
    pushes = sum(1 for r in records if r["result"] == "push")
    units = float(np.sum(profits))
    graded = wins + losses
    return {
        "bets": len(records),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": round(100.0 * wins / graded, 2) if graded else 0.0,
        "units": round(units, 2),
        "roi_pct": round(100.0 * units / len(records), 2),
    }


def _pick_threshold(frame, pred_margin, win_prob, sigma, cfg, market):
    best_thr, best_units, best_bets = cfg.ev_grid[0], float("-inf"), 0
    for thr in cfg.ev_grid:
        recs = (
            _ats_bets(frame, pred_margin, sigma, thr)
            if market == "ats"
            else _ml_bets(frame, win_prob, thr)
        )
        if len(recs) < MIN_VALIDATION_BETS:
            continue
        units = sum(r["profit"] for r in recs)
        if units > best_units:
            best_units, best_thr, best_bets = units, thr, len(recs)
    return best_thr, best_bets


def walk_forward(df: pd.DataFrame, cfg: FoldConfig | None = None) -> BacktestResult:
    cfg = cfg or FoldConfig()
    df = df[df["n_prior_games"] >= MIN_PRIOR_GAMES].copy()
    seasons = sorted(df["season"].unique())
    if len(seasons) <= MIN_TRAIN_SEASONS:
        raise ValueError("Not enough seasons for walk-forward backtest.")

    ats_records: list[dict] = []
    ml_records: list[dict] = []
    cal_probs: list[float] = []
    cal_market: list[float] = []
    cal_outcomes: list[float] = []
    fold_meta: list[dict] = []

    for idx in range(MIN_TRAIN_SEASONS, len(seasons)):
        test_season = seasons[idx]
        train_seasons = seasons[:idx]
        val_season = train_seasons[-1]

        train_df = df[df["season"].isin(train_seasons)]
        fit_df = train_df[train_df["season"] != val_season]
        val_df = train_df[train_df["season"] == val_season]
        test_df = df[df["season"] == test_season]
        if len(fit_df) < 500 or val_df.empty or test_df.empty:
            continue

        # Fit on fit_df; calibrate + choose thresholds on validation.
        fit_models = train_models(fit_df)
        val_margin = fit_models.predict_margin(val_df)
        val_prob_raw = fit_models.predict_winprob(val_df)
        iso = fit_isotonic(val_prob_raw, val_df["home_win"].to_numpy())
        val_prob = apply_isotonic(iso, val_prob_raw)

        ats_thr, _ = _pick_threshold(val_df, val_margin, val_prob, fit_models.margin_sigma, cfg, "ats")
        ml_thr, _ = _pick_threshold(val_df, None, val_prob, fit_models.margin_sigma, cfg, "ml")

        # Refit on full training window for the strongest test-season model.
        full_models = train_models(train_df)
        test_margin = full_models.predict_margin(test_df)
        test_prob = apply_isotonic(iso, full_models.predict_winprob(test_df))

        ats_records.extend(_ats_bets(test_df, test_margin, full_models.margin_sigma, ats_thr))
        ml_records.extend(_ml_bets(test_df, test_prob, ml_thr))

        cal_probs.extend(test_prob.tolist())
        cal_market.extend(test_df["market_devig_home_prob"].tolist())
        cal_outcomes.extend(test_df["home_win"].tolist())

        fold_meta.append(
            {
                "test_season": int(test_season),
                "train_games": int(len(train_df)),
                "test_games": int(len(test_df)),
                "ats_ev_threshold": ats_thr,
                "ml_ev_threshold": ml_thr,
            }
        )

    ats_summary = _summarize(ats_records)
    ml_summary = _summarize(ml_records)

    clv_vals = [r["clv_points"] for r in ats_records if not np.isnan(r.get("clv_points", np.nan))]
    clv_summary = {
        "n_with_open": len(clv_vals),
        "mean_clv_points": round(float(np.mean(clv_vals)), 3) if clv_vals else None,
        "pct_positive_clv": round(100.0 * np.mean([v > 0 for v in clv_vals]), 1) if clv_vals else None,
    }

    cal_probs_arr = np.array(cal_probs)
    cal_out_arr = np.array(cal_outcomes)
    cal_mkt_arr = np.array(cal_market)
    mkt_mask = ~np.isnan(cal_mkt_arr)
    calibration = {
        "model_log_loss": round(log_loss(cal_probs_arr, cal_out_arr), 4),
        "model_brier": round(brier(cal_probs_arr, cal_out_arr), 4),
        "market_log_loss": round(log_loss(cal_mkt_arr[mkt_mask], cal_out_arr[mkt_mask]), 4)
        if mkt_mask.any()
        else None,
        "market_brier": round(brier(cal_mkt_arr[mkt_mask], cal_out_arr[mkt_mask]), 4)
        if mkt_mask.any()
        else None,
        "reliability": calibration_table(cal_probs_arr, cal_out_arr),
    }

    all_records = [dict(r, **{"market": "ats"}) for r in ats_records] + [
        dict(r, **{"market": "ml"}) for r in ml_records
    ]
    bets_df = pd.DataFrame(all_records) if all_records else None

    return BacktestResult(
        ats=ats_summary,
        moneyline=ml_summary,
        calibration=calibration,
        clv=clv_summary,
        bets=bets_df,
        meta={"folds": fold_meta, "oos_seasons": [f["test_season"] for f in fold_meta]},
    )
