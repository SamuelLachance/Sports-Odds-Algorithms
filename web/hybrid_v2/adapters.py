"""Per-league adapters: paths, features, period keys, baselines."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class LeagueAdapter:
    league: str
    table_path: Path
    hybrid_dir: Path
    v2_dir: Path
    feature_module: str
    first_eval_season: int
    baseline_ll: float
    baseline_margin_mae: float | None = None
    end_season: int = 2025
    multiclass: bool = False
    has_margin: bool = True
    min_calibration_pool: int = 1500
    core_features: tuple[str, ...] = ()
    period_fn: Callable[[pd.DataFrame], pd.Series] | None = None
    market_attach_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    home_col: str = "home"
    away_col: str = "away"

    def feature_columns(self) -> tuple[str, ...]:
        mod = importlib.import_module(self.feature_module)
        return tuple(mod.FEATURE_COLUMNS)

    def resolve_core(self) -> tuple[str, ...]:
        if self.core_features:
            return self.core_features
        cols = self.feature_columns()
        # Prefer strength / rest / form style columns that usually exist.
        preferred = (
            "elo_diff",
            "net_rtg_slow_diff",
            "net_rtg_diff",
            "adj_margin_ewma_diff",
            "margin_pg_diff",
            "pythag_diff",
            "win_ewma_diff",
            "sos_elo_diff",
            "rest_diff",
            "home_b2b",
            "away_b2b",
            "pace_sum",
            "elo_x_season_frac",
            "season_frac",
            "h2h_margin_ewma",
            "xg_diff",
            "goal_diff_ewma",
            "sp_era_diff",
            "goalie_gsax_diff",
        )
        found = [c for c in preferred if c in cols]
        if len(found) >= 8:
            return tuple(found)
        return cols[: min(20, len(cols))]


def _month_thirds(
    frame: pd.DataFrame,
    *,
    early: set[int],
    mid: set[int],
) -> pd.Series:
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(2, index=frame.index)
    period[months.isin(early)] = 0
    period[months.isin(mid)] = 1
    return period


def _wnba_periods(frame: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(2, index=frame.index)
    period[months <= 6] = 0
    period[months == 7] = 1
    return period


def _nba_periods(frame: pd.DataFrame) -> pd.Series:
    return _month_thirds(frame, early={10, 11, 12}, mid={1, 2, 3})


def _nhl_periods(frame: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(2, index=frame.index)
    period[months >= 9] = 0
    period[months.isin({1, 2})] = 1
    return period


def _mlb_periods(frame: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(2, index=frame.index)
    period[months <= 5] = 0
    period[months.isin({6, 7})] = 1
    return period


def _nfl_periods(frame: pd.DataFrame) -> pd.Series:
    return _month_thirds(frame, early={9, 10}, mid={11})


def _cfb_periods(frame: pd.DataFrame) -> pd.Series:
    return _month_thirds(frame, early={8, 9}, mid={10})


def _cbb_periods(frame: pd.DataFrame) -> pd.Series:
    return _month_thirds(frame, early={11, 12}, mid={1})


def _soccer_periods(frame: pd.DataFrame) -> pd.Series:
    # Season-only walk-forward: single period per season.
    return pd.Series(0, index=frame.index)


def _devig_home_prob(home_ml: float, away_ml: float) -> float | None:
    def implied(american: float) -> float | None:
        try:
            american = float(american)
        except (TypeError, ValueError):
            return None
        if american == 0:
            american = 100.0
        if abs(american) < 100:
            return None
        if american > 0:
            return 100.0 / (american + 100.0)
        return abs(american) / (abs(american) + 100.0)

    ph = implied(home_ml)
    pa = implied(away_ml)
    if ph is None or pa is None or ph + pa <= 0:
        return None
    return ph / (ph + pa)


def _attach_binary_market(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "market_home_prob" not in frame.columns:
        if "home_ml" in frame.columns and "away_ml" in frame.columns:
            frame["market_home_prob"] = [
                _devig_home_prob(h, a) if pd.notna(h) and pd.notna(a) else np.nan
                for h, a in zip(frame["home_ml"], frame["away_ml"])
            ]
        else:
            frame["market_home_prob"] = np.nan
    mkt = frame["market_home_prob"]
    frame["has_market"] = mkt.notna().astype(float)
    frame["mkt_home_prob"] = mkt.fillna(0.5)
    # Prefer steam already baked into the training table (CBB full-feature build).
    if "ml_steam_pp" not in frame.columns:
        frame["ml_steam_pp"] = 0.0
    if "has_steam" not in frame.columns:
        frame["has_steam"] = 0.0
    if "spread_move" not in frame.columns:
        frame["spread_move"] = 0.0
    return frame


def _attach_soccer_market(frame: pd.DataFrame) -> pd.DataFrame:
    from web.soccer_v2.data import devig_decimal

    frame = frame.copy()
    n = len(frame)
    homes = np.full(n, 1 / 3, dtype=float)
    draws = np.full(n, 1 / 3, dtype=float)
    aways = np.full(n, 1 / 3, dtype=float)
    has = np.zeros(n, dtype=float)
    if {"open_home", "open_draw", "open_away"}.issubset(frame.columns):
        oh = frame["open_home"].to_numpy()
        od = frame["open_draw"].to_numpy()
        oa = frame["open_away"].to_numpy()
        for i in range(n):
            trip = devig_decimal(oh[i], od[i], oa[i])
            if trip is None:
                continue
            homes[i], draws[i], aways[i] = trip
            has[i] = 1.0
    frame["mkt_open_home"] = homes
    frame["mkt_open_draw"] = draws
    frame["mkt_open_away"] = aways
    frame["has_market"] = has
    frame["market_home_prob"] = homes
    frame["mkt_home_prob"] = homes
    frame["ml_steam_pp"] = 0.0
    frame["has_steam"] = 0.0
    return frame


def _read_baseline(league: str, key: str = "oos_model_logloss") -> float:
    path = PROJECT_ROOT / "data" / "models" / f"{league}_v2" / "metadata.json"
    if not path.is_file():
        return 1.0
    meta = json.loads(path.read_text(encoding="utf-8"))
    return float(meta.get(key, 1.0))


def _read_margin_mae(league: str) -> float | None:
    path = PROJECT_ROOT / "data" / "models" / f"{league}_v2" / "metadata.json"
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    val = meta.get("oos_margin_mae")
    return float(val) if val is not None else None


WNBA_CORE = (
    "elo_diff",
    "net_rtg_slow_diff",
    "adj_margin_ewma_diff",
    "margin_pg_diff",
    "elo_x_season_frac",
    "pythag_diff",
    "pace_sum",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "home_travel_km",
    "away_travel_km",
    "win_ewma_diff",
    "sos_elo_diff",
    "dnp_star_rate_diff",
    "star_avail_diff",
    "margin_vol_diff",
    "close_win_ewma_diff",
    "h2h_margin_ewma",
)


def get_adapter(league: str) -> LeagueAdapter:
    league = league.lower().strip()
    specs: dict[str, dict[str, Any]] = {
        "wnba": {
            "feature_module": "web.wnba_v2.feature_engine",
            "first_eval_season": 2010,
            "period_fn": _wnba_periods,
            "core_features": WNBA_CORE,
            "has_margin": True,
        },
        "nba": {
            "feature_module": "web.nba_v2.feature_engine",
            "first_eval_season": 2010,
            "period_fn": _nba_periods,
            "has_margin": True,
        },
        "nhl": {
            "feature_module": "web.nhl_v2.feature_engine",
            "first_eval_season": 2014,
            "period_fn": _nhl_periods,
            "has_margin": False,
        },
        "mlb": {
            "feature_module": "web.mlb_v2.feature_engine",
            "first_eval_season": 2016,
            "period_fn": _mlb_periods,
            "has_margin": False,
            "home_col": "home_key",
            "away_col": "away_key",
        },
        "nfl": {
            "feature_module": "web.nfl_v2.feature_engine",
            "first_eval_season": 2010,
            "period_fn": _nfl_periods,
            "has_margin": True,
        },
        "cfb": {
            "feature_module": "web.cfb_v2.feature_engine",
            "first_eval_season": 2023,
            "period_fn": _cfb_periods,
            "has_margin": True,
            "min_calibration_pool": 400,
        },
        "cbb": {
            "feature_module": "web.cbb_v2.feature_engine",
            "first_eval_season": 2022,
            "period_fn": _cbb_periods,
            "has_margin": True,
            "min_calibration_pool": 800,
        },
        "soccer": {
            "feature_module": "web.soccer_v2.feature_engine",
            "first_eval_season": 2006,
            "period_fn": _soccer_periods,
            "has_margin": False,
            "multiclass": True,
            "market_attach_fn": _attach_soccer_market,
            "min_calibration_pool": 2000,
        },
    }
    if league not in specs:
        raise KeyError(f"Unknown league for hybrid adapter: {league}")
    s = specs[league]
    return LeagueAdapter(
        league=league,
        table_path=PROJECT_ROOT / "data" / f"{league}_history" / "training_table.csv",
        hybrid_dir=PROJECT_ROOT / "data" / "models" / f"{league}_hybrid",
        v2_dir=PROJECT_ROOT / "data" / "models" / f"{league}_v2",
        feature_module=s["feature_module"],
        first_eval_season=int(s["first_eval_season"]),
        baseline_ll=_read_baseline(league),
        baseline_margin_mae=_read_margin_mae(league),
        multiclass=bool(s.get("multiclass", False)),
        has_margin=bool(s.get("has_margin", True)),
        min_calibration_pool=int(s.get("min_calibration_pool", 1500)),
        core_features=tuple(s.get("core_features") or ()),
        period_fn=s["period_fn"],
        market_attach_fn=s.get("market_attach_fn") or _attach_binary_market,
        home_col=str(s.get("home_col") or "home"),
        away_col=str(s.get("away_col") or "away"),
    )


ALL_LEAGUES = ("wnba", "nba", "nhl", "mlb", "nfl", "cfb", "cbb", "soccer")
