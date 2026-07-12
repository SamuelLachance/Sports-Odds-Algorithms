"""Train MLB v2 win-probability model with walk-forward evaluation.

Outputs to data/models/mlb_v2/:
  model_clf.json        XGBoost win classifier
  model_runs_home.json  Poisson runs regressor (home)
  model_runs_away.json  Poisson runs regressor (away)
  calibrator.json       isotonic calibration (x/y breakpoints)
  state_2025.json.gz    feature-engine snapshot at end of prior season
  metadata.json         walk-forward metrics vs closing market
  oos_predictions.csv   out-of-sample predictions for bet backtests
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
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

from web.mlb_v2.feature_engine import FEATURE_COLUMNS, MlbFeatureEngine  # noqa: E402
from web.mlb_v2.replay import write_mlb_feature_snapshot  # noqa: E402

TABLE_PATH = PROJECT_ROOT / "data" / "mlb_history" / "training_table.csv"
CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "mlb-history"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "mlb_v2"

FIRST_EVAL_SEASON = 2016

CLF_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=12.0,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=60,
)

LR_C = 0.05
XGB_WEIGHT = 0.55  # ensemble weight for XGBoost vs logistic regression
MIN_CALIBRATION_POOL = 4000

# in-season retrain checkpoints (month boundaries) mirroring production refits
SEASON_PERIODS: tuple[tuple[int, int], ...] = ((1, 5), (6, 7), (8, 12))

RUNS_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=15.0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="count:poisson",
    random_state=42,
)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_classifier(train: pd.DataFrame, val: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(**CLF_PARAMS)
    model.fit(
        train[list(FEATURE_COLUMNS)],
        train["home_win"],
        eval_set=[(val[list(FEATURE_COLUMNS)], val["home_win"])],
        verbose=False,
    )
    return model


def fit_logistic(train: pd.DataFrame):
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(C=LR_C, max_iter=3000)
    )
    lr.fit(train[list(FEATURE_COLUMNS)], train["home_win"])
    return lr


def predict_ensemble(xgb: XGBClassifier, lr, frame: pd.DataFrame) -> np.ndarray:
    px = xgb.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    pl = lr.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    return XGB_WEIGHT * px + (1.0 - XGB_WEIGHT) * pl


def walk_forward(frame: pd.DataFrame, end_season: int) -> pd.DataFrame:
    """Walk-forward OOS predictions with in-season refit checkpoints."""
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.month
    outputs: list[pd.DataFrame] = []
    calibrator: IsotonicRegression | None = None

    for eval_season in range(FIRST_EVAL_SEASON, end_season + 1):
        for month_lo, month_hi in SEASON_PERIODS:
            train = frame[
                (frame.season < eval_season)
                | ((frame.season == eval_season) & (frame.month < month_lo))
            ]
            test = frame[
                (frame.season == eval_season)
                & (frame.month >= month_lo)
                & (frame.month <= month_hi)
            ]
            if train.empty or test.empty:
                continue
            val_season = int(train.season.max())
            fit = train[train.season < val_season]
            val = train[train.season == val_season]
            if fit.empty or len(val) < 200:
                fit = train
                val = train
            xgb = fit_classifier(fit, val)
            lr = fit_logistic(train)
            raw = predict_ensemble(xgb, lr, test)
            if calibrator is not None:
                calibrated = calibrator.predict(raw)
            else:
                calibrated = raw
            keep_cols = [
                "season",
                "date",
                "game_pk",
                "home_key",
                "away_key",
                "home_win",
                "home_close_ml",
                "away_close_ml",
                "market_home_prob",
                "home_score",
                "away_score",
            ]
            for optional in ("home_open_ml", "away_open_ml"):
                if optional in test.columns:
                    keep_cols.append(optional)
            chunk = test[keep_cols].copy()
            chunk["model_raw"] = raw
            chunk["model_prob"] = np.clip(calibrated, 0.02, 0.98)
            outputs.append(chunk)

            pooled = pd.concat(outputs, ignore_index=True)
            if len(pooled) >= MIN_CALIBRATION_POOL:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
                calibrator.fit(pooled["model_raw"], pooled["home_win"])
        season_rows = pd.concat(
            [c for c in outputs if int(c.season.iloc[0]) == eval_season], ignore_index=True
        )
        print(
            f"eval {eval_season}: n={len(season_rows)} "
            f"LL={log_loss(season_rows.model_prob.values, season_rows.home_win.values.astype(float)):.5f}",
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

    # final classifier: all data, early stop on most recent season
    val = frame[frame.season == end_season]
    fit = frame[frame.season < end_season]
    if val.empty:
        fit = frame
        val = frame
    clf = fit_classifier(fit, val)
    clf.get_booster().save_model(str(OUT_DIR / "model_clf.json"))

    lr_final = fit_logistic(frame)
    (OUT_DIR / "model_lr.json").write_text(
        json.dumps(_logistic_to_json(lr_final)), encoding="utf-8"
    )

    runs_home = XGBRegressor(**RUNS_PARAMS)
    runs_home.fit(frame[list(FEATURE_COLUMNS)], frame["home_score"])
    runs_home.get_booster().save_model(str(OUT_DIR / "model_runs_home.json"))
    runs_away = XGBRegressor(**RUNS_PARAMS)
    runs_away.fit(frame[list(FEATURE_COLUMNS)], frame["away_score"])
    runs_away.get_booster().save_model(str(OUT_DIR / "model_runs_away.json"))

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calibrator.fit(oos["model_raw"], oos["home_win"])
    grid = np.linspace(0.0, 1.0, 201)
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

    # engine snapshot at end of the season before the current live season
    snapshot_season = end_season - 1
    engine = MlbFeatureEngine()
    write_mlb_feature_snapshot(
        engine,
        OUT_DIR,
        snapshot_season=snapshot_season,
        start_season=int(frame.season.min()),
        cache_root=CACHE_ROOT,
    )

    per_season = season_metrics(oos)
    y = oos["home_win"].to_numpy(dtype=float)
    p = oos["model_prob"].to_numpy(dtype=float)
    m = oos.dropna(subset=["market_home_prob"])
    metadata = {
        "algorithm": "MLBGradientBoost v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "ensemble": {"xgb_weight": XGB_WEIGHT, "lr_C": LR_C},
        "train_rows": int(len(frame)),
        "train_seasons": [int(frame.season.min()), int(end_season)],
        "eval_seasons": [FIRST_EVAL_SEASON, int(end_season)],
        "snapshot_season": snapshot_season,
        "oos_model_logloss": round(log_loss(p, y), 5),
        "oos_model_brier": round(brier(p, y), 5),
        "oos_model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "oos_with_odds": int(len(m)),
        "oos_model_logloss_odds_subset": round(
            log_loss(m["model_prob"].values, m["home_win"].values.astype(float)), 5
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
        "oos_model_acc", "train_rows",
    )}, indent=1))
    print("\nper-season:")
    for row in per_season:
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MLB v2 model")
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()

    frame = pd.read_csv(TABLE_PATH)
    frame = frame.dropna(subset=["home_win"])
    end_season = min(args.end_season, int(frame.season.max()))

    oos = walk_forward(frame, end_season)
    train_final_artifacts(frame, oos, end_season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
