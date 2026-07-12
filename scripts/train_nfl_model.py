"""Train NFL v2 win-probability + margin models with walk-forward evaluation.

Outputs to data/models/nfl_v2/:
  model_clf.json          XGBoost win classifier
  model_lr.json           logistic regression coefficients (ensemble partner)
  model_margin.json       margin regressor (home - away)
  model_score_home.json   home points regressor
  model_score_away.json   away points regressor
  calibrator.json         isotonic calibration
  state_{season}.json.gz  feature-engine snapshots
  metadata.json           walk-forward metrics vs Elo / closing market
  oos_predictions.csv     out-of-sample predictions for bet backtests

Ship gate: OOS margin MAE < 10.69 (nfelo) OR model logloss clearly beats Elo.
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

from web.nfl_v2.feature_engine import (  # noqa: E402
    ELO_Z,
    FEATURE_COLUMNS,
    POINTS_PER_ELO,
    NflFeatureEngine,
)
from web.nfl_v2.replay import nflverse_rows_to_games, replay_games  # noqa: E402
from web.football_pred_model import elo_to_prob  # noqa: E402

TABLE_PATH = PROJECT_ROOT / "data" / "nfl_history" / "training_table.csv"
ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nflverse_games.csv"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "nfl_v2"

# Warm 1999–2009; first OOS eval season is 2010
FIRST_EVAL_SEASON = 2010
ELO_BASELINE_MAE = 10.69

CLF_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=12.0,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=4.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=50,
)

MARGIN_PARAMS = dict(
    n_estimators=350,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=14.0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

SCORE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=14.0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

LR_C = 0.05
XGB_WEIGHT = 0.55
MIN_CALIBRATION_POOL = 400

# NFL season periods: early (Sep-Oct), mid (Nov), late+playoffs (Dec-Feb)
SEASON_PERIODS: tuple[tuple[int, ...], ...] = (
    (9, 10),
    (11,),
    (12, 1, 2),
)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _period_key(frame: pd.DataFrame) -> pd.Series:
    month = pd.to_datetime(frame["date"]).dt.month
    out = pd.Series(0, index=frame.index, dtype=int)
    for idx, months in enumerate(SEASON_PERIODS):
        out = out.where(~month.isin(months), idx)
    return out


def _devig_home_prob(home_ml: float, away_ml: float) -> float | None:
    def implied(american: float) -> float | None:
        try:
            american = float(american)
        except (TypeError, ValueError):
            return None
        if abs(american) < 100:
            return None
        if american > 0:
            return 100.0 / (american + 100.0)
        return abs(american) / (abs(american) + 100.0)

    ih = implied(home_ml)
    ia = implied(away_ml)
    if ih is None or ia is None or ih + ia <= 0:
        return None
    return ih / (ih + ia)


def attach_market(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    probs = []
    for row in frame.itertuples(index=False):
        probs.append(_devig_home_prob(row.home_close_ml, row.away_close_ml))
    frame["market_home_prob"] = probs
    frame["elo_home_prob"] = frame["elo_diff"].map(
        lambda d: elo_to_prob(float(d), z=ELO_Z)
    )
    frame["elo_margin"] = frame["elo_diff"] / POINTS_PER_ELO
    return frame


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
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=3000))
    lr.fit(train[list(FEATURE_COLUMNS)], train["home_win"])
    return lr


def predict_ensemble(xgb: XGBClassifier, lr, frame: pd.DataFrame) -> np.ndarray:
    px = xgb.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    pl = lr.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    return XGB_WEIGHT * px + (1.0 - XGB_WEIGHT) * pl


def walk_forward(frame: pd.DataFrame, end_season: int) -> pd.DataFrame:
    frame = attach_market(frame)
    frame["period"] = _period_key(frame)
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
            if train.season.max() < FIRST_EVAL_SEASON - 1 and eval_season == FIRST_EVAL_SEASON and period == 0:
                if len(train) < 200:
                    continue
            val_season = int(train.season.max())
            fit = train[train.season < val_season]
            val = train[train.season == val_season]
            if fit.empty or len(val) < 80:
                fit = train
                val = train
            xgb = fit_classifier(fit, val)
            lr = fit_logistic(train)
            raw = predict_ensemble(xgb, lr, test)
            calibrated = calibrator.predict(raw) if calibrator is not None else raw

            margin_model = XGBRegressor(**MARGIN_PARAMS)
            margin_model.fit(train[list(FEATURE_COLUMNS)], train["margin"])
            margin_pred = margin_model.predict(test[list(FEATURE_COLUMNS)])

            keep_cols = [
                "season",
                "date",
                "home",
                "away",
                "home_win",
                "home_score",
                "away_score",
                "margin",
                "home_close_ml",
                "away_close_ml",
                "home_close_spread",
                "away_close_spread",
                "home_spread_odds",
                "away_spread_odds",
                "market_home_prob",
                "elo_home_prob",
                "elo_margin",
            ]
            chunk = test[keep_cols].copy()
            chunk["model_raw"] = raw
            chunk["model_prob"] = np.clip(calibrated, 0.02, 0.98)
            chunk["model_margin"] = margin_pred
            outputs.append(chunk)

            pooled = pd.concat(outputs, ignore_index=True)
            if len(pooled) >= MIN_CALIBRATION_POOL:
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
                calibrator.fit(pooled["model_raw"], pooled["home_win"])

        season_chunks = [c for c in outputs if int(c.season.iloc[0]) == eval_season]
        if not season_chunks:
            continue
        season_rows = pd.concat(season_chunks, ignore_index=True)
        y = season_rows.home_win.values.astype(float)
        margin_mae = float(np.mean(np.abs(season_rows.model_margin - season_rows.margin)))
        elo_mae = float(np.mean(np.abs(season_rows.elo_margin - season_rows.margin)))
        print(
            f"eval {eval_season}: n={len(season_rows)} "
            f"LL={log_loss(season_rows.model_prob.values, y):.5f} "
            f"eloLL={log_loss(season_rows.elo_home_prob.values, y):.5f} "
            f"marginMAE={margin_mae:.2f} eloMAE={elo_mae:.2f}",
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
            "margin_mae": round(float(np.mean(np.abs(grp.model_margin - grp.margin))), 3),
            "elo_margin_mae": round(float(np.mean(np.abs(grp.elo_margin - grp.margin))), 3),
            "elo_logloss": round(log_loss(grp.elo_home_prob.values, y), 5),
        }
        m = grp.dropna(subset=["market_home_prob"])
        if len(m) >= 50:
            ym = m["home_win"].to_numpy(dtype=float)
            entry["n_with_odds"] = int(len(m))
            entry["market_logloss"] = round(log_loss(m.market_home_prob.values, ym), 5)
            entry["close_spread_mae"] = round(
                float(np.mean(np.abs(m.margin + m.home_close_spread))), 3
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


def train_final_artifacts(frame: pd.DataFrame, oos: pd.DataFrame, end_season: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = attach_market(frame)

    val = frame[frame.season == end_season]
    fit = frame[frame.season < end_season]
    if val.empty or fit.empty:
        fit = frame
        val = frame
    clf = fit_classifier(fit, val)
    clf.get_booster().save_model(str(OUT_DIR / "model_clf.json"))

    lr_final = fit_logistic(frame)
    (OUT_DIR / "model_lr.json").write_text(
        json.dumps(_logistic_to_json(lr_final)), encoding="utf-8"
    )

    margin_model = XGBRegressor(**MARGIN_PARAMS)
    margin_model.fit(frame[list(FEATURE_COLUMNS)], frame["margin"])
    margin_model.get_booster().save_model(str(OUT_DIR / "model_margin.json"))

    score_home = XGBRegressor(**SCORE_PARAMS)
    score_home.fit(frame[list(FEATURE_COLUMNS)], frame["home_score"])
    score_home.get_booster().save_model(str(OUT_DIR / "model_score_home.json"))
    score_away = XGBRegressor(**SCORE_PARAMS)
    score_away.fit(frame[list(FEATURE_COLUMNS)], frame["away_score"])
    score_away.get_booster().save_model(str(OUT_DIR / "model_score_away.json"))

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

    odds = pd.read_csv(ODDS_CSV)
    odds = odds[odds.game_type.isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
    odds = odds.dropna(subset=["home_score", "away_score"])
    all_games = nflverse_rows_to_games(odds.to_dict(orient="records"))

    snapshot_seasons = sorted({int(g["season"]) for g in all_games if int(g["season"]) < end_season})
    for snap_season in snapshot_seasons[-2:]:
        engine = NflFeatureEngine()
        replay_games(
            engine,
            [g for g in all_games if int(g["season"]) <= snap_season],
        )
        path = OUT_DIR / f"state_{snap_season}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(engine.to_dict(), handle, separators=(",", ":"))
        print(f"wrote snapshot {path}", flush=True)

    y = oos["home_win"].to_numpy(dtype=float)
    p = oos["model_prob"].to_numpy(dtype=float)
    oos_margin_mae = float(np.mean(np.abs(oos.model_margin - oos.margin)))
    oos_elo_mae = float(np.mean(np.abs(oos.elo_margin - oos.margin)))
    oos_elo_ll = log_loss(oos.elo_home_prob.values, y)
    oos_model_ll = log_loss(p, y)
    margin_sigma = float(np.std(oos.margin - oos.model_margin))
    close_subset = oos.dropna(subset=["home_close_spread"])
    close_mae = float(
        np.mean(np.abs(close_subset.margin + close_subset.home_close_spread))
    ) if len(close_subset) else float("nan")

    ship_models = bool(
        oos_margin_mae < ELO_BASELINE_MAE or oos_model_ll < oos_elo_ll - 0.01
    )
    per_season = season_metrics(oos)
    metadata = {
        "algorithm": "NFLGradientBoost v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "ensemble": {"xgb_weight": XGB_WEIGHT, "lr_C": LR_C},
        "train_rows": int(len(frame)),
        "train_seasons": [int(frame.season.min()), int(end_season)],
        "eval_seasons": [FIRST_EVAL_SEASON, int(end_season)],
        "oos_model_logloss": round(oos_model_ll, 5),
        "oos_model_brier": round(brier(p, y), 5),
        "oos_model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "oos_margin_mae": round(oos_margin_mae, 4),
        "oos_elo_margin_mae": round(oos_elo_mae, 4),
        "oos_elo_logloss": round(oos_elo_ll, 5),
        "elo_baseline_mae_reference": ELO_BASELINE_MAE,
        "oos_close_spread_mae": round(close_mae, 4) if close_mae == close_mae else None,
        "margin_sigma": round(margin_sigma, 4),
        "ship_models": ship_models,
        "elo_params": {
            "k": 20.0,
            "hfa": 48.0,
            "points_per_elo": POINTS_PER_ELO,
            "z": ELO_Z,
        },
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
    m = oos.dropna(subset=["market_home_prob"])
    if len(m):
        metadata["oos_market_logloss"] = round(
            log_loss(m.market_home_prob.values, m.home_win.values.astype(float)), 5
        )
        metadata["oos_with_odds"] = int(len(m))

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export = oos.rename(
        columns={
            "home_close_spread": "home_spread",
            "home_spread_odds": "spread_home_odds",
            "away_spread_odds": "spread_away_odds",
        }
    )
    export.to_csv(OUT_DIR / "oos_predictions.csv", index=False)

    print(json.dumps({
        "oos_model_logloss": metadata["oos_model_logloss"],
        "oos_elo_logloss": metadata["oos_elo_logloss"],
        "oos_margin_mae": metadata["oos_margin_mae"],
        "oos_elo_margin_mae": metadata["oos_elo_margin_mae"],
        "elo_baseline_mae_reference": ELO_BASELINE_MAE,
        "oos_close_spread_mae": metadata["oos_close_spread_mae"],
        "ship_models": ship_models,
        "n_features": metadata["n_features"],
        "train_rows": metadata["train_rows"],
    }, indent=1))
    print("\nper-season:")
    for row in per_season:
        print(row)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train NFL v2 win-probability + margin models (walk-forward OOS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            f"Requires training table at {TABLE_PATH}. "
            "Build first: python scripts/build_nfl_training_table.py. "
            f"Writes artifacts under {OUT_DIR}/."
        ),
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last season included in training/eval (default: max in table)",
    )
    args = parser.parse_args()

    if not TABLE_PATH.is_file():
        print(
            f"ERROR: missing training table: {TABLE_PATH}\n"
            "  Build it first:\n"
            "    python scripts/build_nfl_training_table.py\n"
            "  That script needs data/supplemental/closing-odds/nflverse_games.csv.",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_csv(TABLE_PATH)
    frame = frame.dropna(subset=["home_win", "margin"])
    if frame.empty:
        print(
            f"ERROR: training table is empty after dropping NaN labels: {TABLE_PATH}",
            file=sys.stderr,
        )
        return 1
    end_season = args.end_season
    if end_season is None:
        end_season = int(frame.season.max())
    end_season = min(end_season, int(frame.season.max()))

    oos = walk_forward(frame, end_season)
    if oos.empty:
        print(
            "ERROR: no OOS rows produced "
            f"(end_season={end_season}; need seasons after the warm-up window).",
            file=sys.stderr,
        )
        return 1
    train_final_artifacts(frame, oos, end_season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
