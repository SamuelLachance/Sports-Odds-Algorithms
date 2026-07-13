"""WNBA hybrid trainer: CurveFM strengths + CatBoost/LightGBM match + stack.

Combines three ideas without deep-learning overfit on short seasons:

1. Tree ensembles (CatBoost / LightGBM) for tabular match outcomes with native
   categoricals (home/away franchise ids).
2. Foundation-style Stage-1 curves (CurveFM / optional Chronos) projecting team
   strength forward with uncertainty into Stage-2.
3. Shallow stacking of tree heads — no Transformers / large MLPs.

Walk-forward protocol mirrors ``train_wnba_model.py`` (season × month periods).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.wnba_v2.feature_engine import FEATURE_COLUMNS  # noqa: E402
from web.wnba_v2.hybrid_curves import (  # noqa: E402
    CURVE_FEATURE_COLUMNS,
    CurveFM,
    attach_curve_features,
)

TABLE_PATH = PROJECT_ROOT / "data" / "wnba_history" / "training_table.csv"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "wnba_hybrid"
BASELINE_LL = 0.62284  # committed pure XGB+LR OOS (2026-07-13 retrain)
FIRST_EVAL_SEASON = 2010
MIN_CALIBRATION_POOL = 1500

CLF_MARKET_FEATURES = ("mkt_home_prob", "has_market", "ml_steam_pp", "has_steam")


@dataclass
class HybridConfig:
    name: str = "default"
    backend: str = "catboost"  # catboost | lightgbm | both
    depth: int = 6
    learning_rate: float = 0.03
    iterations: int = 800
    l2: float = 4.0
    min_data: int = 40
    subsample: float = 0.9
    use_curves: bool = True
    use_categoricals: bool = True
    use_market: bool = True
    # OOS-safe market fusion: p = (1-w)*model + w*market on rows with odds
    market_blend_w: float = 0.0
    feature_mode: str = "full"  # full | core | curves_plus_core
    early_stopping: int = 60
    seed: int = 42
    # train residual head: target = home_win, features include market; or
    # target = home_win - market (clipped) — use "prob" | "residual"
    target_mode: str = "prob"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CORE_FEATURES: tuple[str, ...] = (
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


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


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


def _month_periods_key(frame: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(2, index=frame.index)
    period[months <= 6] = 0
    period[months == 7] = 1
    return period


def attach_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    from web.basketball_v2_market import compute_steam_features

    frame = frame.copy()
    mkt = frame["market_home_prob"]
    frame["has_market"] = mkt.notna().astype(float)
    frame["mkt_home_prob"] = mkt.fillna(0.5)
    steam_rows = [
        compute_steam_features(
            home_spread_open=row.get("home_spread_open"),
            home_spread_close=row.get("home_spread"),
            home_ml_open=row.get("home_ml_open"),
            away_ml_open=row.get("away_ml_open"),
            home_ml_close=row.get("home_ml"),
            away_ml_close=row.get("away_ml"),
        )
        for row in frame.to_dict(orient="records")
    ]
    frame["spread_move"] = [r["spread_move"] for r in steam_rows]
    frame["ml_steam_pp"] = [r["ml_steam_pp"] for r in steam_rows]
    frame["has_steam"] = [r["has_steam"] for r in steam_rows]
    return frame


def select_feature_columns(cfg: HybridConfig) -> list[str]:
    if cfg.feature_mode == "core":
        cols = list(CORE_FEATURES)
    elif cfg.feature_mode == "curves_plus_core":
        cols = list(CORE_FEATURES)
    else:
        cols = list(FEATURE_COLUMNS)
    if cfg.use_curves:
        cols = cols + [c for c in CURVE_FEATURE_COLUMNS if c not in cols]
    if cfg.use_market:
        cols = cols + [c for c in CLF_MARKET_FEATURES if c not in cols]
    return cols


def _fit_catboost(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], cfg: HybridConfig):
    from catboost import CatBoostClassifier, Pool

    cat_cols = [c for c in ("home_id", "away_id") if c in train.columns and cfg.use_categoricals]
    feature_cols = cols + cat_cols
    train_pool = Pool(train[feature_cols], train["home_win"], cat_features=cat_cols or None)
    val_pool = Pool(val[feature_cols], val["home_win"], cat_features=cat_cols or None)
    model = CatBoostClassifier(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        l2_leaf_reg=cfg.l2,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=cfg.seed,
        od_type="Iter",
        od_wait=cfg.early_stopping,
        verbose=False,
        allow_writing_files=False,
        subsample=cfg.subsample,
        min_data_in_leaf=cfg.min_data,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, feature_cols, cat_cols


def _fit_lightgbm(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], cfg: HybridConfig):
    import lightgbm as lgb

    # LightGBM: integer-encode categoricals
    feature_cols = list(cols)
    train_x = train[feature_cols].copy()
    val_x = val[feature_cols].copy()
    cat_maps: dict[str, dict[str, int]] = {}
    if cfg.use_categoricals:
        for name in ("home_id", "away_id"):
            if name not in train.columns:
                continue
            cats = sorted(set(train[name].astype(str)) | set(val[name].astype(str)))
            mapping = {c: i for i, c in enumerate(cats)}
            cat_maps[name] = mapping
            train_x[name] = train[name].astype(str).map(mapping).fillna(-1).astype(int)
            val_x[name] = val[name].astype(str).map(mapping).fillna(-1).astype(int)
            feature_cols.append(name)
    cat_feature_idx = [feature_cols.index(n) for n in cat_maps]
    model = lgb.LGBMClassifier(
        n_estimators=cfg.iterations,
        learning_rate=cfg.learning_rate,
        num_leaves=max(8, 2 ** min(cfg.depth, 6)),
        min_child_samples=cfg.min_data,
        subsample=cfg.subsample,
        colsample_bytree=0.85,
        reg_lambda=cfg.l2,
        random_state=cfg.seed,
        verbosity=-1,
    )
    model.fit(
        train_x[feature_cols],
        train["home_win"],
        eval_set=[(val_x[feature_cols], val["home_win"])],
        categorical_feature=cat_feature_idx or "auto",
        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)],
    )
    return model, feature_cols, cat_maps


def _predict_catboost(model, frame: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> np.ndarray:
    from catboost import Pool

    pool = Pool(frame[feature_cols], cat_features=cat_cols or None)
    return model.predict_proba(pool)[:, 1]


def _predict_lightgbm(model, frame: pd.DataFrame, feature_cols: list[str], cat_maps: dict) -> np.ndarray:
    data = {}
    for c in feature_cols:
        if c in cat_maps:
            data[c] = frame[c].astype(str).map(cat_maps[c]).fillna(-1).astype(int)
        elif c in frame.columns:
            data[c] = frame[c].to_numpy()
        else:
            data[c] = np.zeros(len(frame))
    x = pd.DataFrame(data)[feature_cols]
    return model.predict_proba(x)[:, 1]


def prepare_frame(path: Path = TABLE_PATH, *, end_season: int = 2025) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["home_win"])
    frame = frame[frame.season <= end_season].copy()
    frame["market_home_prob"] = [
        _devig_home_prob(h, a) if pd.notna(h) and pd.notna(a) else np.nan
        for h, a in zip(frame["home_ml"], frame["away_ml"])
    ]
    frame = attach_market_features(frame)
    frame["home_id"] = frame["home"].astype(str)
    frame["away_id"] = frame["away"].astype(str)
    frame = attach_curve_features(frame)
    frame["period"] = _month_periods_key(frame)
    return frame


def walk_forward(frame: pd.DataFrame, cfg: HybridConfig, end_season: int) -> pd.DataFrame:
    cols = select_feature_columns(cfg)
    outputs: list[pd.DataFrame] = []
    calibrator: IsotonicRegression | None = None

    for eval_season in range(FIRST_EVAL_SEASON, end_season + 1):
        for period in (0, 1, 2):
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
            if fit.empty or len(val) < 100:
                fit = train
                val = train

            preds: list[np.ndarray] = []
            if cfg.backend in ("catboost", "both"):
                model, fcols, cat_cols = _fit_catboost(fit, val, cols, cfg)
                preds.append(_predict_catboost(model, test, fcols, cat_cols))
                # refit on full train for slightly better fold pred? keep ES model
            if cfg.backend in ("lightgbm", "both"):
                model, fcols, cat_maps = _fit_lightgbm(fit, val, cols, cfg)
                preds.append(_predict_lightgbm(model, test, fcols, cat_maps))

            raw = np.mean(np.vstack(preds), axis=0) if preds else np.full(len(test), 0.5)
            if cfg.market_blend_w > 0:
                mkt = test["market_home_prob"].to_numpy(dtype=float)
                w = float(cfg.market_blend_w)
                blended = raw.copy()
                mask = np.isfinite(mkt)
                blended[mask] = (1.0 - w) * raw[mask] + w * mkt[mask]
                raw = blended
            calibrated = calibrator.predict(raw) if calibrator is not None else raw

            chunk = test[
                [
                    "season",
                    "date",
                    "event_id",
                    "home",
                    "away",
                    "home_win",
                    "margin",
                    "market_home_prob",
                    "home_ml",
                    "away_ml",
                    "home_spread",
                ]
            ].copy()
            chunk["model_raw"] = raw
            chunk["model_prob"] = np.clip(calibrated, 0.02, 0.98)
            outputs.append(chunk)

            pooled = pd.concat(outputs, ignore_index=True)
            if len(pooled) >= MIN_CALIBRATION_POOL:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
                calibrator.fit(pooled["model_raw"], pooled["home_win"])

        season_chunks = [c for c in outputs if int(c.season.iloc[0]) == eval_season]
        if season_chunks:
            season_rows = pd.concat(season_chunks, ignore_index=True)
            y = season_rows.home_win.values.astype(float)
            market = season_rows.dropna(subset=["market_home_prob"])
            mkt_ll = (
                log_loss(market.market_home_prob.values, market.home_win.values.astype(float))
                if len(market) > 50
                else float("nan")
            )
            print(
                f"[{cfg.name}] eval {eval_season}: n={len(season_rows)} "
                f"LL={log_loss(season_rows.model_prob.values, y):.5f} "
                f"mktLL={mkt_ll:.5f}",
                flush=True,
            )

    return pd.concat(outputs, ignore_index=True)


def summarize(oos: pd.DataFrame) -> dict[str, Any]:
    y = oos["home_win"].to_numpy(dtype=float)
    p = oos["model_prob"].to_numpy(dtype=float)
    m = oos.dropna(subset=["market_home_prob"])
    summary = {
        "oos_model_logloss": round(log_loss(p, y), 5),
        "oos_model_brier": round(brier(p, y), 5),
        "oos_model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "n": int(len(oos)),
        "baseline_ll": BASELINE_LL,
        "delta_vs_baseline": round(BASELINE_LL - log_loss(p, y), 5),
        "beat_baseline": bool(log_loss(p, y) < BASELINE_LL - 0.001),
    }
    if len(m) >= 50:
        ym = m["home_win"].to_numpy(dtype=float)
        summary["oos_market_logloss"] = round(
            log_loss(m["market_home_prob"].values, ym), 5
        )
        summary["oos_model_logloss_odds_subset"] = round(
            log_loss(m["model_prob"].values, ym), 5
        )
        summary["gap_vs_market"] = round(
            summary["oos_model_logloss_odds_subset"] - summary["oos_market_logloss"], 5
        )
    return summary


def run_config(cfg: HybridConfig, frame: pd.DataFrame, end_season: int) -> dict[str, Any]:
    t0 = time.time()
    oos = walk_forward(frame, cfg, end_season)
    summary = summarize(oos)
    summary["config"] = cfg.to_dict()
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.name)
    oos.to_csv(OUT_DIR / f"oos_{safe}.csv", index=False)
    (OUT_DIR / f"summary_{safe}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "oos_model_logloss", "delta_vs_baseline", "beat_baseline",
        "oos_model_acc", "gap_vs_market", "elapsed_s",
    ) if k in summary}, indent=1), flush=True)
    return summary


def default_search_space() -> list[HybridConfig]:
    """Priority queue: fast high-upside configs first, then broader sweep."""
    priority = [
        # Market fusion (honest market-aware — old market head lost to pure)
        HybridConfig(name="cb_core_mktblend35", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=500,
                     market_blend_w=0.35, early_stopping=40),
        HybridConfig(name="cb_core_mktblend50", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=500,
                     market_blend_w=0.50, early_stopping=40),
        HybridConfig(name="cb_core_mktblend25", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=500,
                     market_blend_w=0.25, early_stopping=40),
        HybridConfig(name="cb_full_mktblend40", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="full", use_curves=True, iterations=500,
                     market_blend_w=0.40, early_stopping=40),
        HybridConfig(name="both_core_mktblend35", backend="both", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=400,
                     market_blend_w=0.35, early_stopping=35),
        HybridConfig(name="lgb_core_mktblend35", backend="lightgbm", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=600,
                     market_blend_w=0.35, early_stopping=40),
        # Pure tree upgrades (no market blend)
        HybridConfig(name="cb_curves_core_fast", backend="catboost", depth=6, learning_rate=0.04,
                     feature_mode="curves_plus_core", use_curves=True, iterations=500),
        HybridConfig(name="cb_curves_full_fast", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="full", use_curves=True, iterations=500),
        HybridConfig(name="cb_deep_core", backend="catboost", depth=8, learning_rate=0.025,
                     feature_mode="curves_plus_core", use_curves=True, iterations=700, l2=6.0),
        HybridConfig(name="cb_shallow_wide", backend="catboost", depth=4, learning_rate=0.05,
                     feature_mode="curves_plus_core", use_curves=True, iterations=900),
        HybridConfig(name="lgb_curves_core", backend="lightgbm", depth=7, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=700),
        HybridConfig(name="both_curves_core", backend="both", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, iterations=450),
        HybridConfig(name="cb_nocat_curves", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, use_categoricals=False,
                     iterations=500),
        HybridConfig(name="cb_nomarket_feat", backend="catboost", depth=6, learning_rate=0.03,
                     feature_mode="curves_plus_core", use_curves=True, use_market=False,
                     iterations=500, market_blend_w=0.0),
        # Blend grid
        *[
            HybridConfig(
                name=f"cb_blend{int(w*100)}",
                backend="catboost",
                depth=6,
                learning_rate=0.03,
                feature_mode="curves_plus_core",
                use_curves=True,
                iterations=450,
                market_blend_w=w,
                early_stopping=35,
            )
            for w in (0.15, 0.2, 0.3, 0.45, 0.55, 0.6, 0.7)
        ],
    ]
    # Broader sweep after priority
    extras: list[HybridConfig] = []
    for backend in ("catboost", "lightgbm"):
        for depth in (5, 6, 7):
            for lr in (0.02, 0.04):
                for mode in ("curves_plus_core", "full"):
                    for w in (0.0, 0.35):
                        name = f"{backend}_d{depth}_lr{str(lr).replace('.', '')}_{mode}_w{int(w*100)}"
                        extras.append(
                            HybridConfig(
                                name=name,
                                backend=backend,
                                depth=depth,
                                learning_rate=lr,
                                iterations=450,
                                feature_mode=mode,
                                use_curves=True,
                                market_blend_w=w,
                                early_stopping=35,
                            )
                        )
    seen = {c.name for c in priority}
    for c in extras:
        if c.name not in seen:
            priority.append(c)
            seen.add(c.name)
    return priority


def main() -> int:
    parser = argparse.ArgumentParser(description="Train WNBA hybrid CurveFM+CatBoost model")
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--config", type=str, default="cb_curves_full")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()
    space = default_search_space()
    if args.list_configs:
        for c in space[:30]:
            print(c.name)
        print(f"... {len(space)} total")
        return 0
    print("Loading frame + curves...", flush=True)
    frame = prepare_frame(end_season=args.end_season)
    print(f"rows={len(frame)} cols={len(frame.columns)}", flush=True)
    cfg = next((c for c in space if c.name == args.config), None)
    if cfg is None:
        cfg = HybridConfig(name=args.config)
    run_config(cfg, frame, args.end_season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
