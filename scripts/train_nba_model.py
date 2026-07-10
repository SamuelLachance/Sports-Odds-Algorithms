"""Train NBA v2 win-probability + margin models with walk-forward evaluation.

Outputs to data/models/nba_v2/:
  model_clf.json          XGBoost win classifier
  model_lr.json           logistic regression coefficients (ensemble partner)
  model_margin.json       margin regressor (home - away, for spread picks)
  model_score_home.json   home points regressor (display)
  model_score_away.json   away points regressor (display)
  calibrator.json         isotonic calibration (x/y breakpoints)
  state_{season}.json.gz  feature-engine snapshots (last two seasons)
  metadata.json           walk-forward metrics vs closing market
  oos_predictions.csv     out-of-sample predictions for bet backtests
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_nba_training_table import load_season  # noqa: E402
from web.nba_v2.feature_engine import FEATURE_COLUMNS, NbaFeatureEngine  # noqa: E402
from web.nba_v2.replay import replay_season  # noqa: E402

TABLE_PATH = PROJECT_ROOT / "data" / "nba_history" / "training_table.csv"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "nba_v2"

FIRST_EVAL_SEASON = 2010

CLF_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=10.0,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=4.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=60,
)

MARGIN_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=14.0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

SCORE_PARAMS = dict(
    n_estimators=350,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=14.0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

LR_C = 0.05
XGB_WEIGHT = 0.5
MIN_CALIBRATION_POOL = 1500

# market-aware heads: devigged consensus moneyline (clf) / spread (margin)
CLF_MARKET_FEATURES = ("mkt_home_prob", "has_market")
MARGIN_MARKET_FEATURES = ("mkt_home_spread", "has_spread")


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _devig_home_prob(home_ml: float, away_ml: float) -> float | None:
    def implied(american: float) -> float | None:
        if american >= 100:
            return 100.0 / (american + 100.0)
        if american <= -100:
            return -american / (-american + 100.0)
        return None

    ph = implied(home_ml)
    pa = implied(away_ml)
    if ph is None or pa is None or ph + pa <= 0:
        return None
    return ph / (ph + pa)


def _month_periods_key(frame: pd.DataFrame) -> pd.Series:
    """NBA in-season periods: Oct-Dec=0, Jan-Mar=1, Apr-Jun=2."""
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(1, index=frame.index)
    period[months.isin([10, 11, 12])] = 0
    period[months.isin([4, 5, 6])] = 2
    return period


def attach_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    mkt = frame["market_home_prob"]
    frame["has_market"] = mkt.notna().astype(float)
    frame["mkt_home_prob"] = mkt.fillna(0.5)
    spread = frame["home_spread"]
    frame["has_spread"] = spread.notna().astype(float)
    frame["mkt_home_spread"] = spread.fillna(0.0)
    return frame


def fit_classifier(train: pd.DataFrame, val: pd.DataFrame, cols: list[str]) -> XGBClassifier:
    model = XGBClassifier(**CLF_PARAMS)
    model.fit(
        train[cols],
        train["home_win"],
        eval_set=[(val[cols], val["home_win"])],
        verbose=False,
    )
    return model


def fit_logistic(train: pd.DataFrame, cols: list[str]):
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=3000))
    lr.fit(train[cols], train["home_win"])
    return lr


def predict_ensemble(xgb: XGBClassifier, lr, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    px = xgb.predict_proba(frame[cols])[:, 1]
    pl = lr.predict_proba(frame[cols])[:, 1]
    return XGB_WEIGHT * px + (1.0 - XGB_WEIGHT) * pl


def walk_forward(frame: pd.DataFrame, end_season: int) -> pd.DataFrame:
    frame = attach_market_features(frame)
    frame["period"] = _month_periods_key(frame)
    pure_cols = list(FEATURE_COLUMNS)
    mkt_cols = pure_cols + list(CLF_MARKET_FEATURES)
    margin_cols = pure_cols + list(MARGIN_MARKET_FEATURES)
    outputs: list[pd.DataFrame] = []
    calibrator: IsotonicRegression | None = None
    calibrator_mkt: IsotonicRegression | None = None

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
            xgb = fit_classifier(fit, val, pure_cols)
            lr = fit_logistic(train, pure_cols)
            raw = predict_ensemble(xgb, lr, test, pure_cols)
            calibrated = calibrator.predict(raw) if calibrator is not None else raw

            xgb_mkt = fit_classifier(fit, val, mkt_cols)
            lr_mkt = fit_logistic(train, mkt_cols)
            raw_mkt = predict_ensemble(xgb_mkt, lr_mkt, test, mkt_cols)
            calibrated_mkt = (
                calibrator_mkt.predict(raw_mkt) if calibrator_mkt is not None else raw_mkt
            )

            margin_model = XGBRegressor(**MARGIN_PARAMS)
            margin_model.fit(train[pure_cols], train["margin"])
            margin_pred = margin_model.predict(test[pure_cols])

            margin_model_mkt = XGBRegressor(**MARGIN_PARAMS)
            margin_model_mkt.fit(train[margin_cols], train["margin"])
            margin_pred_mkt = margin_model_mkt.predict(test[margin_cols])

            keep_cols = [
                "season", "date", "event_id", "home", "away", "home_win",
                "home_score", "away_score", "margin", "season_type",
                "home_ml", "away_ml", "home_spread",
                "spread_home_odds", "spread_away_odds", "total_line", "books",
                "home_ml_open", "away_ml_open", "home_spread_open",
                "best_home_ml", "best_away_ml",
                "best_spread_home_odds", "best_spread_away_odds",
                "market_home_prob",
            ]
            chunk = test[keep_cols].copy()
            chunk["model_raw"] = raw
            chunk["model_prob"] = np.clip(calibrated, 0.02, 0.98)
            chunk["model_raw_mkt"] = raw_mkt
            chunk["model_prob_mkt"] = np.clip(calibrated_mkt, 0.02, 0.98)
            chunk["model_margin"] = margin_pred
            chunk["model_margin_mkt"] = margin_pred_mkt
            outputs.append(chunk)

            pooled = pd.concat(outputs, ignore_index=True)
            if len(pooled) >= MIN_CALIBRATION_POOL:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
                calibrator.fit(pooled["model_raw"], pooled["home_win"])
                calibrator_mkt = IsotonicRegression(
                    out_of_bounds="clip", y_min=0.02, y_max=0.98
                )
                calibrator_mkt.fit(pooled["model_raw_mkt"], pooled["home_win"])

        season_chunks = [c for c in outputs if int(c.season.iloc[0]) == eval_season]
        if not season_chunks:
            continue
        season_rows = pd.concat(season_chunks, ignore_index=True)
        market = season_rows.dropna(subset=["market_home_prob"])
        market_ll = (
            log_loss(market.market_home_prob.values, market.home_win.values.astype(float))
            if len(market) > 50
            else float("nan")
        )
        margin_mae = float(np.mean(np.abs(season_rows.model_margin - season_rows.margin)))
        y_season = season_rows.home_win.values.astype(float)
        print(
            f"eval {eval_season}: n={len(season_rows)} "
            f"LL={log_loss(season_rows.model_prob.values, y_season):.5f} "
            f"mktawareLL={log_loss(season_rows.model_prob_mkt.values, y_season):.5f} "
            f"mktLL={market_ll:.5f} marginMAE={margin_mae:.2f}",
            flush=True,
        )

    return pd.concat(outputs, ignore_index=True)


def season_metrics(oos: pd.DataFrame) -> list[dict]:
    rows = []
    for season, grp in oos.groupby("season"):
        y = grp["home_win"].to_numpy(dtype=float)
        p = grp["model_prob"].to_numpy(dtype=float)
        entry = {
            "season": int(season),
            "n": int(len(grp)),
            "model_logloss": round(log_loss(p, y), 5),
            "model_brier": round(brier(p, y), 5),
            "model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
            "margin_mae": round(float(np.mean(np.abs(grp.model_margin - grp.margin))), 2),
        }
        m = grp.dropna(subset=["market_home_prob"])
        if len(m) >= 50:
            ym = m["home_win"].to_numpy(dtype=float)
            pm = m["model_prob"].to_numpy(dtype=float)
            mk = m["market_home_prob"].to_numpy(dtype=float)
            entry.update(
                {
                    "n_with_odds": int(len(m)),
                    "model_logloss_odds_subset": round(log_loss(pm, ym), 5),
                    "market_logloss": round(log_loss(mk, ym), 5),
                    "market_acc": round(float(np.mean((mk > 0.5) == (ym > 0.5))), 4),
                }
            )
        rows.append(entry)
    return rows


def _logistic_to_json(lr_pipeline) -> dict:
    scaler: StandardScaler = lr_pipeline.named_steps["standardscaler"]
    logreg: LogisticRegression = lr_pipeline.named_steps["logisticregression"]
    return {
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in logreg.coef_[0]],
        "intercept": float(logreg.intercept_[0]),
        "xgb_weight": XGB_WEIGHT,
    }


def train_final_artifacts(frame: pd.DataFrame, oos: pd.DataFrame, end_season: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = attach_market_features(frame)
    pure_cols = list(FEATURE_COLUMNS)
    mkt_cols = pure_cols + list(CLF_MARKET_FEATURES)
    margin_cols = pure_cols + list(MARGIN_MARKET_FEATURES)

    val = frame[frame.season == end_season]
    fit = frame[frame.season < end_season]
    if val.empty:
        fit = frame
        val = frame
    clf = fit_classifier(fit, val, pure_cols)
    clf.get_booster().save_model(str(OUT_DIR / "model_clf.json"))
    clf_mkt = fit_classifier(fit, val, mkt_cols)
    clf_mkt.get_booster().save_model(str(OUT_DIR / "model_clf_market.json"))

    (OUT_DIR / "model_lr.json").write_text(
        json.dumps(_logistic_to_json(fit_logistic(frame, pure_cols))), encoding="utf-8"
    )
    (OUT_DIR / "model_lr_market.json").write_text(
        json.dumps(_logistic_to_json(fit_logistic(frame, mkt_cols))), encoding="utf-8"
    )

    margin_model = XGBRegressor(**MARGIN_PARAMS)
    margin_model.fit(frame[pure_cols], frame["margin"])
    margin_model.get_booster().save_model(str(OUT_DIR / "model_margin.json"))
    margin_model_mkt = XGBRegressor(**MARGIN_PARAMS)
    margin_model_mkt.fit(frame[margin_cols], frame["margin"])
    margin_model_mkt.get_booster().save_model(str(OUT_DIR / "model_margin_market.json"))

    score_home = XGBRegressor(**SCORE_PARAMS)
    score_home.fit(frame[pure_cols], frame["home_score"])
    score_home.get_booster().save_model(str(OUT_DIR / "model_score_home.json"))
    score_away = XGBRegressor(**SCORE_PARAMS)
    score_away.fit(frame[pure_cols], frame["away_score"])
    score_away.get_booster().save_model(str(OUT_DIR / "model_score_away.json"))

    grid = np.linspace(0.0, 1.0, 201)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calibrator.fit(oos["model_raw"], oos["home_win"])
    (OUT_DIR / "calibrator.json").write_text(
        json.dumps(
            {
                "kind": "isotonic",
                "x": [round(float(v), 6) for v in grid],
                "y": [round(float(v), 6) for v in calibrator.predict(grid)],
                "fitted_on": int(len(oos)),
            }
        ),
        encoding="utf-8",
    )
    calibrator_mkt = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calibrator_mkt.fit(oos["model_raw_mkt"], oos["home_win"])
    (OUT_DIR / "calibrator_market.json").write_text(
        json.dumps(
            {
                "kind": "isotonic",
                "x": [round(float(v), 6) for v in grid],
                "y": [round(float(v), 6) for v in calibrator_mkt.predict(grid)],
                "fitted_on": int(len(oos)),
            }
        ),
        encoding="utf-8",
    )

    margin_sigma = float(np.std(oos["margin"] - oos["model_margin"]))
    margin_sigma_mkt = float(np.std(oos["margin"] - oos["model_margin_mkt"]))

    # engine snapshots at the last two COMPLETED season boundaries (the current
    # calendar-year season is in progress; the live path replays it on top)
    now_season = datetime.now(timezone.utc).year
    completed = sorted(
        {int(s) for s in frame.season.unique() if int(s) <= end_season and int(s) < now_season}
    )
    snapshot_seasons = set(completed[-2:])
    engine = NbaFeatureEngine()
    for season in range(1997, max(snapshot_seasons) + 1):
        games = load_season(season)
        if not games:
            continue
        replay_season(engine, games)
        if season in snapshot_seasons:
            with gzip.open(OUT_DIR / f"state_{season}.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(engine.to_dict(), fh, separators=(",", ":"))

    per_season = season_metrics(oos)
    y = oos["home_win"].to_numpy(dtype=float)
    p = oos["model_prob"].to_numpy(dtype=float)
    m = oos.dropna(subset=["market_home_prob"])
    metadata = {
        "algorithm": "NBAGradientBoost v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "clf_market_features": list(CLF_MARKET_FEATURES),
        "margin_market_features": list(MARGIN_MARKET_FEATURES),
        "ensemble": {"xgb_weight": XGB_WEIGHT, "lr_C": LR_C},
        "train_rows": int(len(frame)),
        "train_seasons": [int(frame.season.min()), int(end_season)],
        "eval_seasons": [FIRST_EVAL_SEASON, int(end_season)],
        "snapshot_seasons": sorted(snapshot_seasons),
        "margin_sigma": round(margin_sigma, 3),
        "margin_sigma_market": round(margin_sigma_mkt, 3),
        "oos_model_logloss": round(log_loss(p, y), 5),
        "oos_model_brier": round(brier(p, y), 5),
        "oos_model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "oos_margin_mae": round(float(np.mean(np.abs(oos.model_margin - oos.margin))), 3),
        "oos_with_odds": int(len(m)),
        "oos_model_logloss_odds_subset": round(
            log_loss(m["model_prob"].values, m["home_win"].values.astype(float)), 5
        ) if len(m) else None,
        "oos_market_aware_logloss_odds_subset": round(
            log_loss(m["model_prob_mkt"].values, m["home_win"].values.astype(float)), 5
        ) if len(m) else None,
        "oos_market_logloss": round(
            log_loss(m["market_home_prob"].values, m["home_win"].values.astype(float)), 5
        ) if len(m) else None,
        "per_season": per_season,
        "clf_params": {k: v for k, v in CLF_PARAMS.items() if k != "early_stopping_rounds"},
        "feature_importance": {
            name: round(float(score), 5)
            for name, score in sorted(
                zip(FEATURE_COLUMNS, clf.feature_importances_),
                key=lambda kv: -kv[1],
            )
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    oos.to_csv(OUT_DIR / "oos_predictions.csv", index=False)

    print(json.dumps({k: metadata[k] for k in (
        "oos_model_logloss", "oos_market_logloss", "oos_model_logloss_odds_subset",
        "oos_market_aware_logloss_odds_subset",
        "oos_model_acc", "oos_margin_mae", "margin_sigma", "margin_sigma_market",
        "train_rows",
    )}, indent=1))
    print("\nper-season:")
    for row in per_season:
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NBA v2 model")
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(TABLE_PATH)
    frame = frame.dropna(subset=["home_win"])
    frame["market_home_prob"] = [
        _devig_home_prob(h, a) if pd.notna(h) and pd.notna(a) else np.nan
        for h, a in zip(frame["home_ml"], frame["away_ml"])
    ]
    end_season = min(args.end_season, int(frame.season.max()))

    oos = walk_forward(frame[frame.season <= end_season], end_season)
    if not args.skip_artifacts:
        train_final_artifacts(frame[frame.season <= end_season], oos, end_season)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        oos.to_csv(OUT_DIR / "oos_predictions.csv", index=False)
        for row in season_metrics(oos):
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
