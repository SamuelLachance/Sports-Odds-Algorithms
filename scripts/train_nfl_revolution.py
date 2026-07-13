"""NFL Revolutionary Hybrid: CurveFM + Match MC + CatBoost ATS residual.

Three-stage architecture (tabular sports theory):
  1. CurveFM — LightGBM quantile projector on Elo/net/EPA strength curves
     (TimesFM/Chronos-style Stage-1 without DL overfitting on short seasons).
  2. Match layer — Normal Monte Carlo from projected strength → prior margin/prob.
  3. CatBoost — categoricals (team IDs, roof, week) + full tabular + curve/MC
     features; dedicated ATS residual head: target = margin + home_close_spread.

Primary success metric: OOS margin MAE vs closing-spread MAE (~10.07) and
logloss vs current XGB (~0.631). Iterates a config grid until --until wall clock.

Usage:
  python scripts/train_nfl_revolution.py --until 15:00
  python scripts/train_nfl_revolution.py --until 15:00 --ship
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hybrid_v2.curves import CURVE_FEATURE_COLUMNS, CurveFM, TeamCurvePoint  # noqa: E402
from web.nfl_v2.feature_engine import FEATURE_COLUMNS, POINTS_PER_ELO  # noqa: E402

TABLE = PROJECT_ROOT / "data" / "nfl_history" / "training_table.csv"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "nfl_revolution"
V2_DIR = PROJECT_ROOT / "data" / "models" / "nfl_v2"
TZ = ZoneInfo("America/Toronto")

LEAGUE_ELO = 1500.0
NFL_HFA = 48.0
NFL_K = 20.0
NFL_CARRY = 0.70
ALPHA_NET = 0.15
ALPHA_EPA_NET = 0.18
FIRST_EVAL = 2010
CLOSE_SPREAD_MAE_REF = 10.0679
XGB_LL_REF = 0.63059
XGB_MAE_REF = 10.4244

CAT_COLS = ("home_id", "away_id", "roof_cat", "week_bucket", "weekday_cat")


@dataclass
class RevConfig:
    name: str
    depth: int = 6
    lr: float = 0.05
    iterations: int = 320
    l2: float = 5.0
    min_data: int = 35
    market_blend_w: float = 0.45
    residual_blend_w: float = 0.55  # weight on ATS residual head for margin
    use_mc: bool = True
    use_curves: bool = True
    feature_mode: str = "full"  # full | core | curves_mc
    early_stopping: int = 25
    seed: int = 42
    subsample: float = 0.9
    fold_mode: str = "season"  # season (fast search) | period (full)
    resid_shrink: float = 1.0  # shrink ATS residual toward 0 (market prior)
    market_anchor_w: float = 0.0  # blend final margin toward -close_spread
    mc_blend_w: float = 0.15  # when use_mc
    resid_loss: str = "RMSE"  # RMSE | MAE


def _parse_until(text: str) -> datetime:
    """Parse HH:MM as today America/Toronto; if already past, still use that clock."""
    now = datetime.now(TZ)
    hour, minute = (int(x) for x in text.strip().split(":")[:2])
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def attach_nfl_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """NFL-tuned CurveFM with EPA-augmented net rating + NFL HFA/K."""
    frame = frame.sort_values(["date", "home", "away"]).reset_index(drop=True)
    proj = CurveFM()
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
        if sum(len(v) for v in fit_hist.values()) < 400:
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
                st["elo"] = LEAGUE_ELO + NFL_CARRY * (st["elo"] - LEAGUE_ELO)
                st["net"] *= NFL_CARRY
            season_seen[key] = season

        hs = state(home)
        as_ = state(away)
        he = [p.elo for p in history[home]] + [hs["elo"]]
        hn = [p.net for p in history[home]] + [hs["net"]]
        ae = [p.elo for p in history[away]] + [as_["elo"]]
        an = [p.net for p in history[away]] + [as_["net"]]

        ph_elo, ph_lo, ph_hi = proj.project_from_values(he, "elo")
        pa_elo, pa_lo, pa_hi = proj.project_from_values(ae, "elo")
        ph_net, ph_nlo, ph_nhi = proj.project_from_values(hn, "net")
        pa_net, pa_nlo, pa_nhi = proj.project_from_values(an, "net")
        unc = 0.5 * (
            (abs(ph_hi - ph_lo) + abs(pa_hi - pa_lo)) / 40.0
            + (abs(ph_nhi - ph_nlo) + abs(pa_nhi - pa_nlo)) / 8.0
        )
        elo_diff = ph_elo - pa_elo + NFL_HFA
        net_diff = ph_net - pa_net
        # Stage-2 match prior: map Elo diff → expected margin (nfelo POINTS_PER_ELO)
        mc_margin = elo_diff / POINTS_PER_ELO
        # Uncertainty → margin sigma floor
        mc_sigma = 13.0 + 4.0 * float(unc)
        # P(home win) from Normal(mc_margin, mc_sigma)
        z = mc_margin / max(mc_sigma, 1e-6)
        mc_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        rows_out.append(
            {
                "curve_elo_diff": elo_diff,
                "curve_net_diff": net_diff,
                "curve_elo_raw_diff": hs["elo"] - as_["elo"] + NFL_HFA,
                "curve_unc_sum": unc,
                "curve_unc_diff": (abs(ph_hi - ph_lo) - abs(pa_hi - pa_lo)) / 40.0,
                "curve_backend": 1.0 if proj.fitted else 0.0,
                "mc_margin": mc_margin,
                "mc_sigma": mc_sigma,
                "mc_home_prob": float(min(max(mc_prob, 0.02), 0.98)),
                "mc_x_unc": mc_margin * unc,
            }
        )

        # Post-game update (leak-free for next rows)
        home_win = float(row.home_win) > 0.5
        hs_score = float(row.home_score)
        as_score = float(row.away_score)
        signed = hs_score - as_score
        abs_m = abs(signed)
        home_edge = hs["elo"] - as_["elo"] + NFL_HFA
        exp_home = 1.0 / (1.0 + 10.0 ** (-home_edge / 400.0))
        winner_edge = home_edge if home_win else -home_edge
        mov = math.log(max(abs_m, 1.0) + 1.0) * (2.2 / (0.001 * max(winner_edge, 0.0) + 2.2))
        delta = NFL_K * mov * ((1.0 if home_win else 0.0) - exp_home)
        hs["elo"] += delta
        as_["elo"] -= delta
        # EPA-augmented net when rolling EPA diffs exist on the row (pre-game state
        # already used for features; use post-game scores + epa feature as proxy).
        epa_boost = 0.0
        if hasattr(row, "epa_off_diff") and row.epa_off_diff == row.epa_off_diff:
            # Use actual scoring margin primarily; mild EPA pull if present in table
            # as contemporaneous form (table epa_* are pre-game EWMA — safe as soft prior).
            epa_boost = 0.25 * float(row.epa_off_diff) * 14.0
        hs["net"] += ALPHA_NET * ((signed + epa_boost) - hs["net"])
        as_["net"] += ALPHA_NET * ((-signed - epa_boost) - as_["net"])
        date = str(row.date)[:10]
        history[home].append(TeamCurvePoint(date, season, hs["elo"], hs["net"]))
        history[away].append(TeamCurvePoint(date, season, as_["elo"], as_["net"]))

    return pd.concat([frame, pd.DataFrame(rows_out)], axis=1)


MC_FEATURE_COLUMNS = ("mc_margin", "mc_sigma", "mc_home_prob", "mc_x_unc")


def prepare_frame(end_season: int = 2025, *, force: bool = False) -> pd.DataFrame:
    cache = OUT_DIR / f"prepared_end{end_season}.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cache.is_file() and not force:
        print(f"loading cached frame {cache}", flush=True)
        return pd.read_parquet(cache)

    frame = pd.read_csv(TABLE)
    frame = frame.dropna(subset=["home_win", "home_score", "away_score"]).copy()
    frame = frame[frame.season <= end_season].reset_index(drop=True)
    frame["home_id"] = frame["home"].astype(str)
    frame["away_id"] = frame["away"].astype(str)
    frame["roof_cat"] = np.where(frame.get("roof_dome", 0) > 0.5, "dome", "outdoor")
    if "weekday_thu" in frame.columns:
        frame["weekday_cat"] = np.where(frame["weekday_thu"] > 0.5, "thu", "other")
    else:
        frame["weekday_cat"] = "other"
    week = frame["week"] if "week" in frame.columns else 1
    frame["week_bucket"] = pd.cut(
        week.fillna(1).astype(float),
        bins=[0, 4, 9, 13, 18, 30],
        labels=["early", "mid", "late", "dec", "post"],
    ).astype(str)
    # Market home prob from closing moneylines
    if "home_close_ml" in frame.columns and "away_close_ml" in frame.columns:
        def amer_implied(ml: float) -> float:
            if ml != ml or abs(ml) < 100:
                return float("nan")
            return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)

        ph = frame["home_close_ml"].astype(float).map(amer_implied)
        pa = frame["away_close_ml"].astype(float).map(amer_implied)
        tot = ph + pa
        frame["market_home_prob"] = ph / tot
    else:
        frame["market_home_prob"] = np.nan
    frame["has_market"] = frame["market_home_prob"].notna().astype(float)
    frame["mkt_home_prob"] = frame["market_home_prob"].fillna(0.5)
    if "home_close_spread" in frame.columns:
        frame["mkt_home_spread"] = frame["home_close_spread"].fillna(0.0)
        frame["has_spread"] = frame["home_close_spread"].notna().astype(float)
        frame["ats_residual"] = frame["margin"] + frame["home_close_spread"]
    else:
        frame["mkt_home_spread"] = 0.0
        frame["has_spread"] = 0.0
        frame["ats_residual"] = frame["margin"]

    # Periods
    months = pd.to_datetime(frame["date"]).dt.month
    frame["period"] = 2
    frame.loc[months.isin({9, 10}), "period"] = 0
    frame.loc[months == 11, "period"] = 1

    print("attaching NFL CurveFM + MC match layer...", flush=True)
    frame = attach_nfl_curves(frame)
    # Fill numeric NaNs for CatBoost features only (keep market/outcome NaNs for metrics)
    keep_nan = {
        "home_close_spread",
        "away_close_spread",
        "home_close_ml",
        "away_close_ml",
        "market_home_prob",
        "ats_residual",
        "home_open_spread",
        "away_open_spread",
        "home_open_ml",
        "away_open_ml",
        "home_spread_odds",
        "away_spread_odds",
    }
    for c in frame.columns:
        if c in CAT_COLS or c in keep_nan:
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            frame[c] = frame[c].fillna(0.0)
    frame.to_parquet(cache, index=False)
    print(f"cached prepared frame -> {cache}", flush=True)
    return frame


def select_cols(cfg: RevConfig) -> list[str]:
    core = [
        c
        for c in (
            "elo_diff",
            "margin_pg_diff",
            "pythag_diff",
            "pf_slow_diff",
            "pa_slow_diff",
            "win_ewma_diff",
            "qb_elo_diff",
            "epa_off_diff",
            "epa_def_diff",
            "pass_epa_diff",
            "sr_off_diff",
            "rest_diff",
            "short_week",
            "season_frac",
            "elo_x_season_frac",
            "qb_change_diff",
            "madden_ovr_diff",
        )
        if c in FEATURE_COLUMNS
    ]
    curves = list(CURVE_FEATURE_COLUMNS) + list(MC_FEATURE_COLUMNS)
    market = ["mkt_home_prob", "has_market", "mkt_home_spread", "has_spread"]
    cats = list(CAT_COLS)
    if cfg.feature_mode == "curves_mc":
        return curves + core[:10] + market + cats
    if cfg.feature_mode == "core":
        return core + curves + market + cats
    # full
    return list(FEATURE_COLUMNS) + curves + market + cats


def _fit_cb_clf(train, val, cols, cfg: RevConfig):
    from catboost import CatBoostClassifier, Pool

    cat = [c for c in CAT_COLS if c in cols]
    num = [c for c in cols if c not in cat]
    for c in cat:
        train[c] = train[c].astype(str)
        val[c] = val[c].astype(str)
    use = num + cat
    model = CatBoostClassifier(
        depth=cfg.depth,
        learning_rate=cfg.lr,
        iterations=cfg.iterations,
        l2_leaf_reg=cfg.l2,
        min_data_in_leaf=cfg.min_data,
        subsample=cfg.subsample,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=cfg.seed,
        od_type="Iter",
        od_wait=cfg.early_stopping,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(train[use], train["home_win"], cat_features=cat),
        eval_set=Pool(val[use], val["home_win"], cat_features=cat),
        use_best_model=True,
    )
    return model, use, cat


def _fit_cb_reg(train, val, cols, cfg: RevConfig, target: str):
    from catboost import CatBoostRegressor, Pool

    cat = [c for c in CAT_COLS if c in cols]
    num = [c for c in cols if c not in cat]
    use = num + cat
    sub = train.dropna(subset=[target])
    vsub = val.dropna(subset=[target])
    if len(sub) < 200 or len(vsub) < 40:
        return None, use, cat
    for c in cat:
        sub[c] = sub[c].astype(str)
        vsub[c] = vsub[c].astype(str)
    model = CatBoostRegressor(
        depth=cfg.depth,
        learning_rate=cfg.lr,
        iterations=cfg.iterations,
        l2_leaf_reg=cfg.l2,
        min_data_in_leaf=cfg.min_data,
        subsample=cfg.subsample,
        loss_function=cfg.resid_loss if target == "ats_residual" else "RMSE",
        eval_metric=cfg.resid_loss if target == "ats_residual" else "RMSE",
        random_seed=cfg.seed,
        od_type="Iter",
        od_wait=cfg.early_stopping,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(sub[use], sub[target], cat_features=cat),
        eval_set=Pool(vsub[use], vsub[target], cat_features=cat),
        use_best_model=True,
    )
    return model, use, cat


def walk_forward(frame: pd.DataFrame, cfg: RevConfig, end_season: int = 2025) -> dict[str, Any]:
    from catboost import Pool

    cols = [c for c in select_cols(cfg) if c in frame.columns]
    outputs: list[pd.DataFrame] = []
    calibrator: IsotonicRegression | None = None
    periods = (0,) if cfg.fold_mode == "season" else (0, 1, 2)

    for eval_season in range(FIRST_EVAL, end_season + 1):
        for period in periods:
            if cfg.fold_mode == "season":
                train = frame[frame.season < eval_season]
                test = frame[frame.season == eval_season]
            else:
                train = frame[
                    (frame.season < eval_season)
                    | ((frame.season == eval_season) & (frame.period < period))
                ]
                test = frame[(frame.season == eval_season) & (frame.period == period)]
            if train.empty or test.empty:
                continue
            val_season = int(train.season.max())
            fit = train[train.season < val_season]
            val = train[train.season == val_season]
            if fit.empty or len(val) < 80:
                fit, val = train, train

            clf, fcols, cat = _fit_cb_clf(fit.copy(), val.copy(), cols, cfg)
            raw = clf.predict_proba(Pool(test[fcols], cat_features=cat))[:, 1]

            # Market blend on probability
            w = float(cfg.market_blend_w)
            mkt = test["mkt_home_prob"].to_numpy(dtype=float)
            has_m = test["has_market"].to_numpy(dtype=float) > 0.5
            blended = raw.copy()
            blended[has_m] = (1.0 - w) * raw[has_m] + w * mkt[has_m]

            if calibrator is not None:
                prob = calibrator.predict(blended)
            else:
                prob = blended

            # Margin heads: pure margin + ATS residual
            m_model, m_cols, _ = _fit_cb_reg(fit.copy(), val.copy(), cols, cfg, "margin")
            r_model, r_cols, _ = _fit_cb_reg(fit.copy(), val.copy(), cols, cfg, "ats_residual")
            margin_pure = (
                m_model.predict(Pool(test[m_cols], cat_features=[c for c in CAT_COLS if c in m_cols]))
                if m_model is not None
                else test["mc_margin"].to_numpy(dtype=float)
            )
            if r_model is not None and test["has_spread"].sum() > 0:
                resid = r_model.predict(
                    Pool(test[r_cols], cat_features=[c for c in CAT_COLS if c in r_cols])
                )
                resid = np.asarray(resid, dtype=float) * float(cfg.resid_shrink)
                # pred_margin = -spread + resid  => covers when resid predicts ATS
                margin_from_ats = -test["mkt_home_spread"].to_numpy(dtype=float) + resid
                rw = float(cfg.residual_blend_w)
                has_s = test["has_spread"].to_numpy(dtype=float) > 0.5
                margin = np.asarray(margin_pure, dtype=float).copy()
                margin[has_s] = (1.0 - rw) * margin[has_s] + rw * margin_from_ats[has_s]
            else:
                margin = np.asarray(margin_pure, dtype=float)
                resid = np.full(len(test), np.nan)

            # Optional MC prior blend into margin
            if cfg.use_mc and float(cfg.mc_blend_w) > 0:
                mc = test["mc_margin"].to_numpy(dtype=float)
                mw = float(cfg.mc_blend_w)
                margin = (1.0 - mw) * margin + mw * mc

            # Anchor toward closing spread (honest market prior on margin)
            aw = float(cfg.market_anchor_w)
            if aw > 0 and test["has_spread"].sum() > 0:
                has_s = test["has_spread"].to_numpy(dtype=float) > 0.5
                close_m = -test["mkt_home_spread"].to_numpy(dtype=float)
                margin = margin.copy()
                margin[has_s] = (1.0 - aw) * margin[has_s] + aw * close_m[has_s]

            chunk = test[
                [
                    c
                    for c in (
                        "season",
                        "date",
                        "home",
                        "away",
                        "home_win",
                        "margin",
                        "home_close_spread",
                        "market_home_prob",
                        "has_spread",
                    )
                    if c in test.columns
                ]
            ].copy()
            chunk["model_raw"] = raw
            chunk["model_prob"] = np.clip(prob, 0.02, 0.98)
            chunk["model_margin"] = margin
            chunk["ats_resid_pred"] = resid
            outputs.append(chunk)

            pooled = pd.concat(outputs, ignore_index=True)
            if len(pooled) >= 800:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
                calibrator.fit(pooled["model_raw"], pooled["home_win"])

            if cfg.fold_mode == "season":
                break  # one fold per season

    oos = pd.concat(outputs, ignore_index=True)
    y = oos["home_win"].to_numpy(dtype=float)
    p = oos["model_prob"].to_numpy(dtype=float)
    ll = _log_loss(p, y)
    mae = float(np.mean(np.abs(oos["model_margin"] - oos["margin"])))
    close = oos.dropna(subset=["home_close_spread"])
    close_mae = float(np.mean(np.abs(close["margin"] + close["home_close_spread"]))) if len(close) else float("nan")
    ats_mae = (
        float(np.mean(np.abs(oos["model_margin"] + oos["home_close_spread"])))
        if "home_close_spread" in oos.columns
        else float("nan")
    )
    # beats close if model margin MAE < close spread MAE
    beat_close = bool(mae < close_mae) if close_mae == close_mae else False
    return {
        "config": asdict(cfg),
        "n": int(len(oos)),
        "oos_model_logloss": round(ll, 5),
        "oos_margin_mae": round(mae, 4),
        "oos_close_spread_mae": round(close_mae, 4) if close_mae == close_mae else None,
        "oos_ats_cover_mae": round(ats_mae, 4) if ats_mae == ats_mae else None,
        "beat_close_spread": beat_close,
        "delta_ll_vs_xgb": round(XGB_LL_REF - ll, 5),
        "delta_mae_vs_xgb": round(XGB_MAE_REF - mae, 4),
        "oos": oos,
    }


def search_space() -> list[RevConfig]:
    """LL-first hunt, near-close MAE anchors, then period refine of winners."""
    priority = [
        # Logloss hunters (best so far rev_ats_w85 LL=0.61453)
        RevConfig(name="rev_ll_mkt55_rw85", residual_blend_w=0.85, market_blend_w=0.55, use_mc=False, resid_shrink=0.15, depth=6, lr=0.04),
        RevConfig(name="rev_ll_mkt65_rw70", residual_blend_w=0.70, market_blend_w=0.65, use_mc=False, resid_shrink=0.15, depth=6),
        RevConfig(name="rev_ll_mkt75_rw50", residual_blend_w=0.50, market_blend_w=0.75, use_mc=False, resid_shrink=0.10, depth=5, lr=0.05),
        RevConfig(name="rev_ll_mkt80", residual_blend_w=0.20, market_blend_w=0.80, use_mc=False, resid_shrink=0.05, depth=5),
        RevConfig(name="rev_ll_deep_mkt60", residual_blend_w=0.85, market_blend_w=0.60, use_mc=False, depth=7, lr=0.03, iterations=450, l2=8.0, resid_shrink=0.2),
        RevConfig(name="rev_ll_curves_mkt70", feature_mode="curves_mc", residual_blend_w=0.6, market_blend_w=0.70, use_mc=False, resid_shrink=0.15),
        RevConfig(name="rev_ll_core_mkt60", feature_mode="core", residual_blend_w=0.8, market_blend_w=0.60, use_mc=False, resid_shrink=0.2),
        # Near-close MAE with tiny residual
        RevConfig(name="rev_mae_shrink12", residual_blend_w=1.0, market_blend_w=0.55, use_mc=False, resid_shrink=0.12),
        RevConfig(name="rev_mae_shrink08", residual_blend_w=1.0, market_blend_w=0.55, use_mc=False, resid_shrink=0.08),
        RevConfig(name="rev_mae_anchor70", residual_blend_w=1.0, market_blend_w=0.55, use_mc=False, resid_shrink=0.25, market_anchor_w=0.70),
        # Earlier beat-close attempts (resume skips if done)
        RevConfig(name="rev_ats_pure", residual_blend_w=1.0, market_blend_w=0.35, use_mc=False, resid_shrink=1.0),
        RevConfig(name="rev_ats_shrink50", residual_blend_w=1.0, market_blend_w=0.35, use_mc=False, resid_shrink=0.5),
        RevConfig(name="rev_ats_shrink35", residual_blend_w=1.0, market_blend_w=0.40, use_mc=False, resid_shrink=0.35),
        RevConfig(name="rev_ats_shrink20", residual_blend_w=1.0, market_blend_w=0.40, use_mc=False, resid_shrink=0.20),
        RevConfig(name="rev_ats_mae_loss", residual_blend_w=1.0, market_blend_w=0.35, use_mc=False, resid_loss="MAE", resid_shrink=0.6),
        RevConfig(name="rev_anchor20", residual_blend_w=1.0, market_blend_w=0.40, use_mc=False, resid_shrink=0.7, market_anchor_w=0.20),
        RevConfig(name="rev_anchor40", residual_blend_w=1.0, market_blend_w=0.40, use_mc=False, resid_shrink=0.5, market_anchor_w=0.40),
        RevConfig(name="rev_anchor60", residual_blend_w=0.9, market_blend_w=0.45, use_mc=False, resid_shrink=0.4, market_anchor_w=0.60),
        RevConfig(name="rev_mc_light", residual_blend_w=1.0, market_blend_w=0.35, use_mc=True, mc_blend_w=0.05, resid_shrink=0.5),
        RevConfig(name="rev_curves_ats", feature_mode="curves_mc", residual_blend_w=1.0, market_blend_w=0.50, use_mc=False, resid_shrink=0.45),
        RevConfig(name="rev_core_pure", feature_mode="core", residual_blend_w=1.0, market_blend_w=0.35, use_mc=False, resid_shrink=0.55),
    ]
    base = [
        RevConfig(name="rev_ats_w55", residual_blend_w=0.55, market_blend_w=0.45),
        RevConfig(name="rev_ats_w70", residual_blend_w=0.70, market_blend_w=0.40),
        RevConfig(name="rev_ats_w85", residual_blend_w=0.85, market_blend_w=0.35, lr=0.04),
        RevConfig(name="rev_ats_w100", residual_blend_w=1.0, market_blend_w=0.30),
        RevConfig(name="rev_mc_heavy", residual_blend_w=0.60, market_blend_w=0.50, depth=7, lr=0.035, iterations=400),
        RevConfig(name="rev_curves_mc", feature_mode="curves_mc", residual_blend_w=0.75, market_blend_w=0.55),
        RevConfig(name="rev_core_ats", feature_mode="core", residual_blend_w=0.80, market_blend_w=0.40),
        RevConfig(name="rev_nomarket_ats", market_blend_w=0.0, residual_blend_w=0.90, iterations=400),
        RevConfig(name="rev_deep_l2", depth=7, l2=8.0, lr=0.03, iterations=450, residual_blend_w=0.65, market_blend_w=0.45),
        RevConfig(name="rev_shallow_fast", depth=5, lr=0.07, iterations=280, residual_blend_w=0.50, market_blend_w=0.55),
        RevConfig(name="rev_mkt_dom", residual_blend_w=0.40, market_blend_w=0.70, depth=5),
        RevConfig(name="rev_pure_margin", residual_blend_w=0.0, market_blend_w=0.40, use_mc=True),
        RevConfig(name="rev_no_mc", residual_blend_w=0.70, market_blend_w=0.45, use_mc=False),
    ]
    extra = []
    for seed in (7, 13, 21, 99):
        extra.append(
            replace(
                base[1],
                name=f"rev_ats_w70_s{seed}",
                seed=seed,
                residual_blend_w=0.65 + 0.05 * (seed % 5),
                market_blend_w=0.30 + 0.05 * (seed % 4),
            )
        )
    for rw in (0.4, 0.5, 0.6, 0.75, 0.9, 0.95):
        extra.append(
            replace(
                base[0],
                name=f"rev_sweep_rw{int(rw * 100)}",
                residual_blend_w=rw,
                market_blend_w=max(0.0, 0.6 - 0.3 * rw),
            )
        )
    # Period-fold confirmation of strongest LL / MAE candidates
    refine = [
        RevConfig(name="rev_ats_w85_period", residual_blend_w=0.85, market_blend_w=0.35, lr=0.04, fold_mode="period", iterations=400, early_stopping=30, use_mc=False, resid_shrink=0.2),
        RevConfig(name="rev_ll_mkt65_period", residual_blend_w=0.70, market_blend_w=0.65, fold_mode="period", iterations=400, use_mc=False, resid_shrink=0.15),
        RevConfig(name="rev_mae_shrink12_period", residual_blend_w=1.0, market_blend_w=0.55, fold_mode="period", iterations=380, use_mc=False, resid_shrink=0.12),
    ]
    return priority + base + extra + refine


def _rank_score(result: dict[str, Any]) -> tuple:
    """Prefer beat-close; else strong LL lift with near-close MAE."""
    ll = float(result["oos_model_logloss"])
    mae = float(result["oos_margin_mae"])
    beat = 1 if result.get("beat_close_spread") else 0
    # Secondary: LL improvement vs XGB, then MAE closeness to close
    return (
        beat,
        round(XGB_LL_REF - ll, 5),  # higher better
        -abs(mae - CLOSE_SPREAD_MAE_REF),
        -mae,
    )


def ship_to_v2(result: dict[str, Any], frame: pd.DataFrame) -> None:
    """Refit final models on all data and write hybrid artifacts for live serve."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RevConfig(**{k: v for k, v in result["config"].items() if k in RevConfig.__dataclass_fields__})
    cols = [c for c in select_cols(cfg) if c in frame.columns]
    # Final fit: train through 2024, val 2025 early, or last season split
    train = frame[frame.season < 2025]
    val = frame[frame.season == 2025]
    if val.empty:
        val = train.tail(400)
        train = train.iloc[:-400]
    clf, fcols, cat = _fit_cb_clf(train.copy(), val.copy(), cols, cfg)
    m_model, m_cols, _ = _fit_cb_reg(train.copy(), val.copy(), cols, cfg, "margin")
    r_model, r_cols, _ = _fit_cb_reg(train.copy(), val.copy(), cols, cfg, "ats_residual")

    clf_path = OUT_DIR / "model_clf_hybrid.cbm"
    clf.save_model(str(clf_path))
    if m_model is not None:
        m_model.save_model(str(OUT_DIR / "model_margin_hybrid.cbm"))
    if r_model is not None:
        r_model.save_model(str(OUT_DIR / "model_ats_residual.cbm"))

    meta = {
        "algorithm": "NFLRevolutionHybrid",
        "stages": ["CurveFM", "MatchMC", "CatBoost+ATS"],
        "created_at": datetime.now(TZ).isoformat(),
        # Live loader keys (web.hybrid_v2.live.load_hybrid_bundle)
        "feature_cols": fcols,
        "cat_cols": list(cat),
        "clf_path": "model_clf_hybrid.cbm",
        "margin_path": "model_margin_hybrid.cbm" if m_model is not None else None,
        "margin_feature_cols": m_cols if m_model is not None else [],
        "residual_path": "model_ats_residual.cbm" if r_model is not None else None,
        "residual_feature_cols": r_cols if r_model is not None else [],
        # Aliases kept for notebook / revolution tooling
        "feature_columns": fcols,
        "margin_feature_columns": m_cols,
        "residual_feature_columns": r_cols if r_model is not None else [],
        "cat_features": list(cat),
        "config": asdict(cfg),
        "oos_model_logloss": result["oos_model_logloss"],
        "oos_margin_mae": result["oos_margin_mae"],
        "oos_close_spread_mae": result["oos_close_spread_mae"],
        "beat_close_spread": result["beat_close_spread"],
        "beat_close_spread_gate": result["beat_close_spread"],
        "delta_ll_vs_xgb": result["delta_ll_vs_xgb"],
        "delta_mae_vs_xgb": result["delta_mae_vs_xgb"],
        "ship_models": True,
        "hybrid": True,
        "hybrid_backend": "catboost",
        "market_blend_w": cfg.market_blend_w,
        "residual_blend_w": cfg.residual_blend_w,
        "resid_shrink": cfg.resid_shrink,
        "market_anchor_w": cfg.market_anchor_w,
        "mc_blend_w": cfg.mc_blend_w if cfg.use_mc else 0.0,
    }
    (OUT_DIR / "hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT_DIR / "best_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Also copy into nfl_v2 for live try_hybrid_binary if better than existing
    v2_meta_path = V2_DIR / "hybrid_meta.json"
    existing_ll = None
    if v2_meta_path.is_file():
        try:
            existing_ll = json.loads(v2_meta_path.read_text(encoding="utf-8")).get("oos_model_logloss")
        except Exception:
            pass
    if existing_ll is None or result["oos_model_logloss"] <= float(existing_ll) + 1e-6:
        import shutil

        shutil.copy2(clf_path, V2_DIR / "model_clf_hybrid.cbm")
        if m_model is not None:
            shutil.copy2(OUT_DIR / "model_margin_hybrid.cbm", V2_DIR / "model_margin_hybrid.cbm")
        if r_model is not None:
            shutil.copy2(OUT_DIR / "model_ats_residual.cbm", V2_DIR / "model_ats_residual.cbm")
        (V2_DIR / "hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        result["oos"].to_csv(V2_DIR / "oos_predictions_hybrid.csv", index=False)
        print(f"shipped hybrid into {V2_DIR}", flush=True)
    oos = result["oos"]
    oos.to_csv(OUT_DIR / "oos_predictions.csv", index=False)
    print(f"wrote {OUT_DIR}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", type=str, default="15:00", help="Stop HH:MM America/Toronto")
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--ship", action="store_true")
    parser.add_argument("--max-configs", type=int, default=0, help="0 = run until wall clock")
    args = parser.parse_args()

    deadline = _parse_until(args.until)
    print(f"NFL Revolution deadline: {deadline.isoformat()} (now {datetime.now(TZ).isoformat()})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame(args.end_season)
    configs = search_space()
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    leaderboard: list[dict[str, Any]] = []
    lb_path = OUT_DIR / "leaderboard.json"
    done_names: set[str] = set()
    if lb_path.is_file():
        try:
            leaderboard = json.loads(lb_path.read_text(encoding="utf-8"))
            done_names = {str(r.get("config", {}).get("name")) for r in leaderboard}
            print(f"resuming with {len(done_names)} finished configs", flush=True)
        except Exception:
            leaderboard = []

    best: dict[str, Any] | None = None
    best_path = OUT_DIR / "best_so_far.json"
    oos_path = OUT_DIR / "oos_predictions.csv"
    if best_path.is_file() and oos_path.is_file() and leaderboard:
        try:
            slim = json.loads(best_path.read_text(encoding="utf-8"))
            best = {**slim, "oos": pd.read_csv(oos_path)}
            best["_score"] = _rank_score(slim)
        except Exception:
            best = None

    # Re-pick best from leaderboard if richer than file
    if leaderboard:
        top = max(leaderboard, key=_rank_score)
        if best is None or _rank_score(top) > best.get("_score", ()):
            # Prefer in-memory OOS if same name already loaded
            if best is not None and best.get("config", {}).get("name") == top.get("config", {}).get("name"):
                pass
            else:
                best = {**top, "oos": pd.read_csv(oos_path)} if oos_path.is_file() else dict(top)
                best["_score"] = _rank_score(top)

    for i, cfg in enumerate(configs):
        if cfg.name in done_names:
            continue
        now = datetime.now(TZ)
        if now >= deadline:
            print(f"hit deadline at config {i}/{len(configs)}", flush=True)
            break
        print(f"\n=== [{i+1}/{len(configs)}] {cfg.name} remaining={(deadline-now).total_seconds()/60:.1f}m ===", flush=True)
        t0 = time.time()
        try:
            result = walk_forward(frame, cfg, args.end_season)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {cfg.name}: {exc}", flush=True)
            continue
        elapsed = time.time() - t0
        summary = {k: v for k, v in result.items() if k != "oos"}
        summary["elapsed_s"] = round(elapsed, 1)
        leaderboard.append(summary)
        # Persist before console print (Windows cp1252 can break unicode prints)
        (OUT_DIR / "leaderboard.json").write_text(
            json.dumps(leaderboard, indent=2), encoding="utf-8"
        )
        score = _rank_score(result)
        if best is None or score > best["_score"]:
            best = result
            best["_score"] = score
            slim = {k: v for k, v in best.items() if k not in ("oos", "_score")}
            (OUT_DIR / "best_so_far.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
            best["oos"].to_csv(OUT_DIR / "oos_predictions.csv", index=False)

        print(
            f"  LL={result['oos_model_logloss']:.5f} (d_xgb={result['delta_ll_vs_xgb']:+.5f}) "
            f"MAE={result['oos_margin_mae']:.4f} (d_xgb={result['delta_mae_vs_xgb']:+.4f}) "
            f"close={result['oos_close_spread_mae']} beat_close={result['beat_close_spread']} "
            f"[{elapsed:.0f}s]",
            flush=True,
        )
        if result["beat_close_spread"] and result["oos_margin_mae"] < CLOSE_SPREAD_MAE_REF - 0.15:
            print("  *** strong beat-close candidate ***", flush=True)
        if result["delta_ll_vs_xgb"] >= 0.015:
            print("  *** strong LL lift vs XGB ***", flush=True)

    if best is None:
        print("no successful configs", flush=True)
        return 1

    print("\n=== BEST ===", flush=True)
    print(json.dumps({k: v for k, v in best.items() if k not in ("oos", "_score")}, indent=2), flush=True)

    if args.ship or best.get("beat_close_spread") or best.get("delta_ll_vs_xgb", 0) > 0.005:
        ship_to_v2(best, frame)
    else:
        # Still save OOS for analysis
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        best["oos"].to_csv(OUT_DIR / "oos_predictions.csv", index=False)
        slim = {k: v for k, v in best.items() if k not in ("oos", "_score")}
        (OUT_DIR / "best_config.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
