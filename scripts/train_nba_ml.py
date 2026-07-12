"""Offline NBA ML training: build dataset, honest walk-forward, save artifacts.

Run locally / in CI (not in the daily site build). Produces:
- data/supplemental/nba-ml/feature_matrix.(parquet|csv)
- data/supplemental/nba-ml/backtest_report.(json|md)
- data/models/nba/margin_model.joblib, winprob_model.joblib, metadata.json

    python scripts/train_nba_ml.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_ml.backtest import FoldConfig, walk_forward  # noqa: E402
from web.nba_ml.calibrate import apply_isotonic, fit_isotonic  # noqa: E402
from web.nba_ml.config import (  # noqa: E402
    BACKTEST_REPORT_MD,
    BACKTEST_REPORT_PATH,
    FEATURE_COLUMNS,
    MARGIN_MODEL_PATH,
    METADATA_PATH,
    WINPROB_MODEL_PATH,
)
from web.nba_ml.dataset import build_dataset, save_dataset  # noqa: E402
from web.nba_ml.model import train_models  # noqa: E402


def _median(values):
    clean = [v for v in values if v is not None]
    return float(np.median(clean)) if clean else 0.0


def _write_report(result, dataset_rows, date_min, date_max) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_rows": dataset_rows,
        "date_range": [date_min, date_max],
        "oos_seasons": result.meta.get("oos_seasons"),
        "ats": result.ats,
        "moneyline": result.moneyline,
        "clv": result.clv,
        "calibration": result.calibration,
        "folds": result.meta.get("folds"),
    }
    BACKTEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cal = result.calibration
    lines = [
        "# NBA ML Honest Backtest Report",
        "",
        f"Generated: {payload['generated_at']}",
        f"Dataset: {dataset_rows} games, {date_min} -> {date_max}",
        f"Out-of-sample seasons: {payload['oos_seasons']}",
        "",
        "## Against the spread (graded at REAL closing line)",
        f"- Bets: {result.ats.get('bets')}",
        f"- Record: {result.ats.get('wins')}-{result.ats.get('losses')}-{result.ats.get('pushes')} "
        f"({result.ats.get('win_pct')}% win)",
        f"- Units: {result.ats.get('units')} | ROI: {result.ats.get('roi_pct')}%",
        "",
        "## Moneyline (graded at REAL closing line)",
        f"- Bets: {result.moneyline.get('bets')}",
        f"- Record: {result.moneyline.get('wins')}-{result.moneyline.get('losses')} "
        f"({result.moneyline.get('win_pct')}% win)",
        f"- Units: {result.moneyline.get('units')} | ROI: {result.moneyline.get('roi_pct')}%",
        "",
        "## Closing Line Value (ATS bets with opening data)",
        f"- Bets with opening line: {result.clv.get('n_with_open')}",
        f"- Mean CLV (points): {result.clv.get('mean_clv_points')}",
        f"- Positive CLV: {result.clv.get('pct_positive_clv')}%",
        "",
        "## Probability calibration (win prob)",
        f"- Model log-loss: {cal.get('model_log_loss')} | Brier: {cal.get('model_brier')}",
        f"- Market log-loss: {cal.get('market_log_loss')} | Brier: {cal.get('market_brier')}",
        "",
        "Interpretation: beating the closing line is extremely hard. Positive CLV",
        "and ROI at the close on a filtered subset are the real signals; a model",
        "log-loss at or slightly above the market's is expected and honest.",
    ]
    BACKTEST_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("Building NBA feature matrix...", flush=True)
    df = build_dataset()
    path = save_dataset(df)
    date_min, date_max = str(df["date"].min()), str(df["date"].max())
    print(f"  {len(df)} rows -> {path}")

    print("Running nested walk-forward backtest...", flush=True)
    result = walk_forward(df, FoldConfig())
    print(f"  ATS: {result.ats}")
    print(f"  ML:  {result.moneyline}")
    print(f"  CLV: {result.clv}")
    print(f"  Cal: {result.calibration.get('model_log_loss')} vs market "
          f"{result.calibration.get('market_log_loss')}")
    _write_report(result, len(df), date_min, date_max)

    print("Training deployable models on full history...", flush=True)
    seasons = sorted(df["season"].unique())
    hold = df[df["season"] == seasons[-1]]
    prior = df[df["season"] != seasons[-1]]
    hold_models = train_models(prior)
    iso = fit_isotonic(hold_models.predict_winprob(hold), hold["home_win"].to_numpy())

    final_models = train_models(df)

    MARGIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_models.margin, "sigma": final_models.margin_sigma},
        MARGIN_MODEL_PATH,
    )
    joblib.dump({"model": final_models.winprob, "isotonic": iso}, WINPROB_MODEL_PATH)

    ats_thr = _median([f["ats_ev_threshold"] for f in result.meta["folds"]])
    ml_thr = _median([f["ml_ev_threshold"] for f in result.meta["folds"]])
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "feature_columns": list(FEATURE_COLUMNS),
        "margin_sigma": round(final_models.margin_sigma, 3),
        "dataset_rows": len(df),
        "date_range": [date_min, date_max],
        "recommended_ats_ev_threshold": round(ats_thr, 4),
        "recommended_ml_ev_threshold": round(ml_thr, 4),
        "ensemble_weight": 0.5,
        "backtest": {
            "ats": result.ats,
            "moneyline": result.moneyline,
            "clv": result.clv,
            "calibration": {
                k: v for k, v in result.calibration.items() if k != "reliability"
            },
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved artifacts to {MARGIN_MODEL_PATH.parent}")
    print(f"Report: {BACKTEST_REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
