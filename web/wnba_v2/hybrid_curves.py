"""Stage-1 strength-curve projector for the WNBA hybrid trainer.

Architecture (foundation-model hybrid without DL overfit):

* Maintain chronological team Elo + net-rating curves (leak-free).
* Project strength to tip-off with uncertainty:
  - **identity**: last pre-game state + rolling volatility (always safe)
  - **curvefm**: LightGBM quantile on lagged strengths, fit only on *past*
    seasons inside each walk-forward fold
  - **chronos**: optional Amazon Chronos if installed

Projected strengths feed Stage-2 CatBoost / LightGBM match models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LEAGUE_ELO = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 70.0
ELO_SEASON_CARRY = 0.70
ALPHA_NET = 0.12


@dataclass
class TeamCurvePoint:
    date: str
    season: int
    elo: float
    net: float


def _lag_matrix(values: list[float], lags: tuple[int, ...] = (1, 2, 3, 5, 8)) -> np.ndarray:
    n = len(values)
    out = np.zeros((n, len(lags)), dtype=float)
    for j, lag in enumerate(lags):
        for i in range(n):
            out[i, j] = values[i - lag] if i >= lag else values[0]
    return out


class CurveFM:
    """LightGBM quantile projector fit on past team-strength trajectories."""

    def __init__(self, *, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)):
        self.quantiles = quantiles
        self.models: dict[str, dict[float, Any]] = {"elo": {}, "net": {}}

    @property
    def fitted(self) -> bool:
        return bool(self.models["elo"])

    def fit(self, histories: dict[str, list[TeamCurvePoint]]) -> None:
        import lightgbm as lgb

        for series_name in ("elo", "net"):
            xs: list[np.ndarray] = []
            ys: list[float] = []
            for pts in histories.values():
                if len(pts) < 12:
                    continue
                vals = [float(getattr(p, series_name)) for p in pts]
                lags = _lag_matrix(vals)
                for i in range(8, len(vals) - 1):
                    xs.append(lags[i])
                    ys.append(vals[i + 1])
            if len(xs) < 200:
                continue
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            x_df = pd.DataFrame(x, columns=[f"lag_{i}" for i in (1, 2, 3, 5, 8)])
            for q in self.quantiles:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=180,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=25,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    verbosity=-1,
                    random_state=42,
                )
                model.fit(x_df, y)
                self.models[series_name][q] = model

    def project_from_values(self, values: list[float], series_name: str) -> tuple[float, float, float]:
        if not values:
            default = LEAGUE_ELO if series_name == "elo" else 0.0
            return default, default, default
        last = values[-1]
        models = self.models.get(series_name) or {}
        if not models or len(values) < 9:
            # identity + empirical unc from recent deltas
            if len(values) >= 4:
                deltas = np.diff(values[-8:])
                spread = float(np.std(deltas)) if len(deltas) else 0.0
            else:
                spread = 0.0
            return last, last - 1.28 * spread, last + 1.28 * spread
        x = pd.DataFrame(_lag_matrix(values)[-1:].astype(float), columns=[f"lag_{i}" for i in (1, 2, 3, 5, 8)])
        preds = {q: float(m.predict(x)[0]) for q, m in models.items()}
        mid = preds.get(0.5, last)
        return mid, preds.get(0.1, mid), preds.get(0.9, mid)


def attach_curve_features(
    frame: pd.DataFrame,
    *,
    projector: CurveFM | None = None,
    refit_each_season: bool = True,
) -> pd.DataFrame:
    """Stream games chronologically; emit pre-game projected strength features.

    When ``refit_each_season`` is True, CurveFM is refit on all *prior* seasons
    at each season boundary (walk-forward safe).
    """
    frame = frame.sort_values(["date", "event_id"]).reset_index(drop=True)
    proj = projector or CurveFM()
    history: dict[str, list[TeamCurvePoint]] = {}
    teams: dict[str, dict[str, float]] = {}
    season_seen: dict[str, int] = {}
    fitted_through: int | None = None
    rows_out: list[dict[str, float]] = []

    def state(key: str) -> dict[str, float]:
        if key not in teams:
            teams[key] = {"elo": LEAGUE_ELO, "net": 0.0}
            history[key] = []
        return teams[key]

    def maybe_refit(season: int) -> None:
        nonlocal fitted_through
        if not refit_each_season:
            if not proj.fitted:
                proj.fit(history)
                fitted_through = season - 1
            return
        prior = season - 1
        if fitted_through is not None and fitted_through >= prior:
            return
        if prior < 2004:
            return
        fit_hist = {
            k: [p for p in pts if p.season <= prior]
            for k, pts in history.items()
            if any(p.season <= prior for p in pts)
        }
        if sum(len(v) for v in fit_hist.values()) < 500:
            return
        fresh = CurveFM()
        fresh.fit(fit_hist)
        if fresh.fitted:
            proj.models = fresh.models
            fitted_through = prior

    for row in frame.itertuples(index=False):
        home = str(row.home)
        away = str(row.away)
        season = int(row.season)
        maybe_refit(season)

        for key in (home, away):
            st = state(key)
            prev = season_seen.get(key)
            if prev is not None and prev != season:
                st["elo"] = LEAGUE_ELO + ELO_SEASON_CARRY * (st["elo"] - LEAGUE_ELO)
                st["net"] *= ELO_SEASON_CARRY
            season_seen[key] = season

        hs = state(home)
        as_ = state(away)
        # Pre-game projection from history (excludes current game)
        he = [p.elo for p in history[home]]
        hn = [p.net for p in history[home]]
        ae = [p.elo for p in history[away]]
        an = [p.net for p in history[away]]
        # Seed with current state so identity path works before history grows
        if not he:
            he, hn = [hs["elo"]], [hs["net"]]
        if not ae:
            ae, an = [as_["elo"]], [as_["net"]]
        # Use live pre-game state as the tip-off prior; CurveFM adjusts trajectory
        he = he + [hs["elo"]]
        hn = hn + [hs["net"]]
        ae = ae + [as_["elo"]]
        an = an + [as_["net"]]

        ph_elo, ph_lo, ph_hi = proj.project_from_values(he, "elo")
        pa_elo, pa_lo, pa_hi = proj.project_from_values(ae, "elo")
        ph_net, ph_nlo, ph_nhi = proj.project_from_values(hn, "net")
        pa_net, pa_nlo, pa_nhi = proj.project_from_values(an, "net")
        unc = 0.5 * (
            (abs(ph_hi - ph_lo) + abs(pa_hi - pa_lo)) / 40.0
            + (abs(ph_nhi - ph_nlo) + abs(pa_nhi - pa_nlo)) / 8.0
        )
        rows_out.append(
            {
                "curve_elo_diff": ph_elo - pa_elo + ELO_HOME_ADV,
                "curve_net_diff": ph_net - pa_net,
                "curve_elo_raw_diff": hs["elo"] - as_["elo"] + ELO_HOME_ADV,
                "curve_unc_sum": unc,
                "curve_unc_diff": (abs(ph_hi - ph_lo) - abs(pa_hi - pa_lo)) / 40.0,
                "curve_backend": 1.0 if proj.fitted else 0.0,
            }
        )

        home_win = float(row.home_win) > 0.5
        margin = abs(float(row.home_score) - float(row.away_score))
        signed = float(row.home_score) - float(row.away_score)
        home_edge = hs["elo"] - as_["elo"] + ELO_HOME_ADV
        exp_home = 1.0 / (1.0 + 10.0 ** (-home_edge / 400.0))
        winner_edge = home_edge if home_win else -home_edge
        mov = math.log(max(margin, 1.0) + 1.0) * (2.2 / (0.001 * max(winner_edge, 0.0) + 2.2))
        delta = ELO_K * mov * ((1.0 if home_win else 0.0) - exp_home)
        hs["elo"] += delta
        as_["elo"] -= delta
        hs["net"] += ALPHA_NET * (signed - hs["net"])
        as_["net"] += ALPHA_NET * (-signed - as_["net"])
        date = str(row.date)[:10]
        history[home].append(TeamCurvePoint(date, season, hs["elo"], hs["net"]))
        history[away].append(TeamCurvePoint(date, season, as_["elo"], as_["net"]))

    return pd.concat([frame, pd.DataFrame(rows_out)], axis=1)


CURVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "curve_elo_diff",
    "curve_net_diff",
    "curve_elo_raw_diff",
    "curve_unc_sum",
    "curve_unc_diff",
    "curve_backend",
)
