"""Train CBB v2 win-probability + margin models with walk-forward evaluation.

Outputs to data/models/cbb_v2/. Features are PIT-safe (ESPN completed games);
Torvik season-end ratings are not used. Gate target: OOS logloss <= ~0.55
(ensemble holdout baseline) or meaningful margin MAE improvement.
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

from scripts.build_cbb_training_table import load_season  # noqa: E402
from web.cbb_v2.feature_engine import FEATURE_COLUMNS, CbbFeatureEngine  # noqa: E402
from web.cbb_v2.replay import replay_season  # noqa: E402

TABLE_PATH = PROJECT_ROOT / "data" / "cbb_history" / "training_table.csv"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "cbb_v2"

FIRST_EVAL_SEASON = 2022
ENSEMBLE_BASELINE_LOGLOSS = 0.55

CLF_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.025,
    max_depth=3,
    min_child_weight=12.0,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_lambda=5.0,
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
    min_child_weight=16.0,
    subsample=0.85,
    colsample_bytree=0.75,
    objective="reg:squarederror",
    random_state=42,
)

SCORE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=16.0,
    subsample=0.85,
    colsample_bytree=0.75,
    objective="reg:squarederror",
    random_state=42,
)

LR_C = 0.04
XGB_WEIGHT = 0.55
MIN_CALIBRATION_POOL = 2000


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


def _period_key(frame: pd.DataFrame) -> pd.Series:
    """CBB season thirds: Nov–Dec, Jan, Feb–Apr."""
    months = pd.to_datetime(frame["date"]).dt.month
    period = pd.Series(2, index=frame.index)
    period[months.isin([11, 12])] = 0
    period[months == 1] = 1
    return period


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
    frame = frame.copy()
    frame["period"] = _period_key(frame)
    cols = list(FEATURE_COLUMNS)
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
            if fit.empty or len(val) < 200:
                fit = train
                val = train
            xgb = fit_classifier(fit, val, cols)
            lr = fit_logistic(train, cols)
            raw = predict_ensemble(xgb, lr, test, cols)
            calibrated = calibrator.predict(raw) if calibrator is not None else raw

            margin_model = XGBRegressor(**MARGIN_PARAMS)
            margin_model.fit(train[cols], train["margin"])
            margin_pred = margin_model.predict(test[cols])

            keep_cols = [
                "season", "date", "event_id", "home", "away",
                "home_abbr", "away_abbr", "home_win",
                "home_score", "away_score", "margin", "season_type",
                "home_ml", "away_ml", "home_spread", "books",
                "market_home_prob",
            ]
            present = [c for c in keep_cols if c in test.columns]
            chunk = test[present].copy()
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
        y_season = season_rows.home_win.values.astype(float)
        margin_mae = float(np.mean(np.abs(season_rows.model_margin - season_rows.margin)))
        print(
            f"eval {eval_season}: n={len(season_rows)} "
            f"LL={log_loss(season_rows.model_prob.values, y_season):.5f} "
            f"acc={float(np.mean((season_rows.model_prob.values > 0.5) == (y_season > 0.5))):.3f} "
            f"marginMAE={margin_mae:.2f}",
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
        if "market_home_prob" in grp.columns:
            m = grp.dropna(subset=["market_home_prob"])
            if len(m) >= 50:
                ym = m["home_win"].to_numpy(dtype=float)
                mk = m["market_home_prob"].to_numpy(dtype=float)
                entry.update(
                    {
                        "n_with_odds": int(len(m)),
                        "market_logloss": round(log_loss(mk, ym), 5),
                        "model_logloss_odds_subset": round(
                            log_loss(m["model_prob"].values, ym), 5
                        ),
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


def train_final_artifacts(frame: pd.DataFrame, oos: pd.DataFrame, end_season: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(FEATURE_COLUMNS)

    val = frame[frame.season == end_season]
    fit = frame[frame.season < end_season]
    if val.empty or fit.empty:
        fit = frame
        val = frame
    clf = fit_classifier(fit, val, cols)
    clf.get_booster().save_model(str(OUT_DIR / "model_clf.json"))

    (OUT_DIR / "model_lr.json").write_text(
        json.dumps(_logistic_to_json(fit_logistic(frame, cols))), encoding="utf-8"
    )

    margin_model = XGBRegressor(**MARGIN_PARAMS)
    margin_model.fit(frame[cols], frame["margin"])
    margin_model.get_booster().save_model(str(OUT_DIR / "model_margin.json"))

    score_home = XGBRegressor(**SCORE_PARAMS)
    score_home.fit(frame[cols], frame["home_score"])
    score_home.get_booster().save_model(str(OUT_DIR / "model_score_home.json"))
    score_away = XGBRegressor(**SCORE_PARAMS)
    score_away.fit(frame[cols], frame["away_score"])
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

    margin_sigma = float(np.std(oos["margin"] - oos["model_margin"]))

    now_season = cbb_now_season()
    completed = sorted(
        {int(s) for s in frame.season.unique() if int(s) <= end_season and int(s) < now_season}
    )
    snapshot_seasons = set(completed[-2:]) if completed else set()
    if not snapshot_seasons and completed:
        snapshot_seasons = {completed[-1]}
    # Always snapshot the latest fully available season <= end_season for live warm-start
    if end_season in frame.season.values and end_season not in snapshot_seasons:
        # if current season in progress, prefer previous
        prior = [s for s in completed if s < now_season]
        snapshot_seasons = set((prior or completed)[-2:])

    engine = CbbFeatureEngine()
    start_season = int(frame.season.min())
    # replay from FIRST_SEASON warm-up even if emit-from is later
    from web.cbb_v2.data import FIRST_SEASON

    for season in range(FIRST_SEASON, max(snapshot_seasons or [end_season]) + 1):
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
    oos_ll = log_loss(p, y)
    oos_mae = float(np.mean(np.abs(oos.model_margin - oos.margin)))
    gate_pass = oos_ll <= ENSEMBLE_BASELINE_LOGLOSS

    m = oos.dropna(subset=["market_home_prob"]) if "market_home_prob" in oos.columns else oos.iloc[0:0]
    metadata = {
        "algorithm": "CBBGradientBoost v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "ensemble": {"xgb_weight": XGB_WEIGHT, "lr_C": LR_C},
        "train_rows": int(len(frame)),
        "train_seasons": [int(frame.season.min()), int(end_season)],
        "eval_seasons": [FIRST_EVAL_SEASON, int(end_season)],
        "snapshot_seasons": sorted(snapshot_seasons),
        "margin_sigma": round(margin_sigma, 3),
        "oos_model_logloss": round(oos_ll, 5),
        "oos_model_brier": round(brier(p, y), 5),
        "oos_model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "oos_margin_mae": round(oos_mae, 3),
        "ensemble_baseline_logloss": ENSEMBLE_BASELINE_LOGLOSS,
        "gate_pass_vs_ensemble_ll": gate_pass,
        "pit_safe": True,
        "training_source": "espn_completed_games",
        "torvik_used_in_training": False,
        "oos_with_odds": int(len(m)),
        "oos_market_logloss": round(
            log_loss(m["market_home_prob"].values, m["home_win"].values.astype(float)), 5
        )
        if len(m)
        else None,
        "official_picks_recommendation": (
            "keep_disabled_no_historical_odds_backtest"
            if len(m) < 200
            else ("enable_if_worst_season_roi_positive" if gate_pass else "keep_disabled_gate_fail")
        ),
        "per_season": per_season,
        "clf_params": {k: v for k, v in CLF_PARAMS.items() if k != "early_stopping_rounds"},
        "feature_importance": {
            name: round(float(score), 5)
            for name, score in sorted(
                zip(FEATURE_COLUMNS, clf.feature_importances_),
                key=lambda kv: -kv[1],
            )
        },
        "bet_policy_note": (
            "Official CBB picks remain disabled until a positive worst-season "
            "walk-forward backtest on historical closing odds is available."
        ),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    oos.to_csv(OUT_DIR / "oos_predictions.csv", index=False)

    # Minimal bet policy stub — picks stay disabled.
    (OUT_DIR / "bet_policy.json").write_text(
        json.dumps(
            {
                "enabled": False,
                "reason": metadata["bet_policy_note"],
                "gate_pass_vs_ensemble_ll": gate_pass,
                "oos_model_logloss": metadata["oos_model_logloss"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                k: metadata[k]
                for k in (
                    "oos_model_logloss",
                    "oos_model_acc",
                    "oos_margin_mae",
                    "gate_pass_vs_ensemble_ll",
                    "oos_market_logloss",
                    "train_rows",
                    "n_features",
                    "official_picks_recommendation",
                )
            },
            indent=1,
        )
    )
    print("\nper-season:")
    for row in per_season:
        print(row)
    return metadata


def cbb_now_season() -> int:
    now = datetime.now(timezone.utc)
    if now.month >= 8:
        return now.year + 1
    return now.year


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CBB v2 win-probability + margin models (walk-forward OOS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            f"Requires training table at {TABLE_PATH}. "
            "Build first: python scripts/build_cbb_training_table.py. "
            f"Writes artifacts under {OUT_DIR}/. "
            "Official picks stay gated until spread backtest clears."
        ),
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2026,
        help="Last season included in training/eval",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Only run walk-forward OOS; do not write model artifacts",
    )
    args = parser.parse_args()

    if not TABLE_PATH.is_file():
        print(
            f"ERROR: missing training table: {TABLE_PATH}\n"
            "  Build it first:\n"
            "    python scripts/build_cbb_training_table.py\n"
            "  Optional odds join: data/supplemental/closing-odds/cbb.csv.",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_csv(TABLE_PATH)
    frame = frame.dropna(subset=["home_win"])
    if frame.empty:
        print(
            f"ERROR: training table is empty after dropping NaN labels: {TABLE_PATH}",
            file=sys.stderr,
        )
        return 1
    if "home_ml" in frame.columns and "away_ml" in frame.columns:
        frame["market_home_prob"] = [
            _devig_home_prob(h, a) if pd.notna(h) and pd.notna(a) else np.nan
            for h, a in zip(frame["home_ml"], frame["away_ml"])
        ]
    else:
        frame["market_home_prob"] = np.nan

    end_season = min(args.end_season, int(frame.season.max()))
    oos = walk_forward(frame[frame.season <= end_season], end_season)
    if oos.empty:
        print(
            "ERROR: no OOS rows produced "
            f"(end_season={end_season}; need seasons after the warm-up window).",
            file=sys.stderr,
        )
        return 1
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
