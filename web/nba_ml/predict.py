"""Cheap NBA ML inference from committed artifacts (no training at runtime).

Rebuilds the point-in-time ``FeatureState`` by replaying the historical odds
table once (cached), then scores a live matchup. Safe to import even when the
model artifacts are absent -- callers should gate on ``nba_ml_enabled()``.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from web.nba_ml.config import (
    FEATURE_COLUMNS,
    MARGIN_MODEL_PATH,
    METADATA_PATH,
    ODDS_CSV,
    WINPROB_MODEL_PATH,
    nba_ml_artifacts_present,
)
from web.nba_ml.features import FeatureState, devig_home_prob, parse_game_date, spread_to_home_prob
from web.nba_ml.model import cover_probability, ensemble_home_prob


@lru_cache(maxsize=1)
def _load_models():
    margin = joblib.load(MARGIN_MODEL_PATH)
    winprob = joblib.load(WINPROB_MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return margin["model"], float(margin["sigma"]), winprob["model"], winprob.get("isotonic"), meta


@lru_cache(maxsize=32)
def _replayed_state(stop_before: str | None = None) -> FeatureState:
    """Replay completed games strictly before ``stop_before`` (ISO YYYY-MM-DD).

    Without a cutoff the full table is used (live tip uses today's date so
    same-day results never leak into Elo/rest). Sorted CSV → early ``break``.
    """
    state = FeatureState()
    frame = pd.read_csv(ODDS_CSV).dropna(subset=["home_final", "away_final"])
    frame = frame.sort_values("date")
    for _, row in frame.iterrows():
        game_day = parse_game_date(row["date"])
        if stop_before is not None and game_day.isoformat() >= stop_before:
            break
        state.update(
            str(row["home_key"]).lower(),
            str(row["away_key"]).lower(),
            game_day,
            int(row["home_final"]),
            int(row["away_final"]),
        )
    return state


def clear_caches() -> None:
    _load_models.cache_clear()
    _replayed_state.cache_clear()


def _parse_cutoff(cutoff_date: str | None) -> date:
    if not cutoff_date:
        return date.today()
    text = str(cutoff_date).strip()
    for fmt in ("%m-%d-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    # Unparseable cutoffs must not silently become "today" (wrong rest/Elo).
    raise ValueError(f"invalid NBA ML cutoff date: {cutoff_date!r}")


def predict_nba(
    home_abbr: str,
    away_abbr: str,
    *,
    cutoff_date: str | None = None,
    market_spread: float | None = None,
    home_ml: float | None = None,
    away_ml: float | None = None,
) -> dict[str, Any] | None:
    """Return calibrated NBA predictions, or None if the model can't score it.

    ``market_spread`` (home-oriented, negative = home favored) and moneylines,
    when provided, sharpen the prediction; the model still runs without them.
    """
    if not nba_ml_artifacts_present():
        return None
    home = home_abbr.lower()
    away = away_abbr.lower()

    margin_model, sigma, winprob_model, iso, meta = _load_models()
    try:
        game_day = _parse_cutoff(cutoff_date)
    except ValueError:
        return None
    state = _replayed_state(game_day.isoformat())
    if home not in state.teams or away not in state.teams:
        return None

    feats = state.features_for(
        home,
        away,
        game_day,
        market_spread=market_spread,
        market_home_ml=home_ml,
        market_away_ml=away_ml,
    )
    frame = pd.DataFrame([feats])[list(FEATURE_COLUMNS)].astype(float)

    pred_margin = float(margin_model.predict(frame)[0])
    raw_win = float(winprob_model.predict_proba(frame)[0, 1])
    if iso is not None:
        raw_win = float(iso.predict([raw_win])[0])
    win_prob = min(max(raw_win, 1e-4), 1.0 - 1e-4)

    market_prob = devig_home_prob(home_ml, away_ml)
    if market_prob is None:
        market_prob = spread_to_home_prob(market_spread)
    from web.market_decorrelation import DEFAULT_BINARY_DECORRELATION_WEIGHT

    decor_weight = float(meta.get("decorrelation_weight", DEFAULT_BINARY_DECORRELATION_WEIGHT))
    ensemble = float(
        ensemble_home_prob(
            win_prob,
            market_prob if market_prob is not None else float("nan"),
            weight=decor_weight,
        )
    )
    market_decorrelated = market_prob is not None

    cover_prob = None
    if market_spread is not None:
        cover_prob = float(cover_probability(pred_margin, market_spread, sigma))

    edge_pp = None
    if market_prob is not None:
        edge_pp = round((win_prob - market_prob) * 100.0, 2)

    return {
        "algorithm": "NBA-ML (XGBoost)",
        "source": "nba_ml",
        "home_win_probability": round(ensemble * 100.0, 2),
        "market_decorrelated": market_decorrelated,
        "model_home_win_probability": round(win_prob * 100.0, 2),
        "predicted_home_margin": round(pred_margin, 2),
        "home_cover_probability": round(cover_prob * 100.0, 2) if cover_prob is not None else None,
        "market_home_win_probability": round(market_prob * 100.0, 2) if market_prob is not None else None,
        "model_market_edge_pp": edge_pp,
        "margin_sigma": round(sigma, 2),
    }
