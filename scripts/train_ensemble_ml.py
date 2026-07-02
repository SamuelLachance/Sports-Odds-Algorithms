"""Train per-sport XGBoost ensemble mega-models from all layer outputs."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.ensemble_ml.config import (  # noqa: E402
    MIN_TRAIN_ROWS,
    NHL_TARGET_LOGLOSS,
    TRAIN_LEAGUES,
    get_dataset_profile,
)
from web.ensemble_ml.dataset import build_league_dataset  # noqa: E402
from web.ensemble_ml.model import (  # noqa: E402
    evaluate_holdout_logloss,
    save_ensemble,
    train_binary_ensemble,
    train_soccer_ensemble,
)
from web.ensemble_ml.predict import clear_ensemble_caches  # noqa: E402
from web.league_profiles import is_soccer_league  # noqa: E402
from web.nba_ml.calibrate import brier, log_loss  # noqa: E402

CORE_LEAGUES = (
    "nba",
    "wnba",
    "cbb",
    "nhl",
    "mlb",
    "nfl",
    "cfb",
    "ncaah",
    "ncaabb",
    "epl",
    "bundesliga",
    "laliga",
    "seriea",
    "ligue1",
)


def _cutoff_today() -> str:
    today = date.today()
    return f"{today.month}-{today.day}-{today.year}"


def train_league(
    league: str,
    cutoff: str,
    *,
    profile: dict | None = None,
    rebuild_dataset: bool = False,
) -> dict | None:
    league = league.lower()
    profile = profile or get_dataset_profile(league)
    print(f"Building dataset for {league}...", flush=True)
    frame = build_league_dataset(
        league, cutoff, profile=profile, rebuild=rebuild_dataset
    )
    if len(frame) < MIN_TRAIN_ROWS:
        print(f"  skipped ({len(frame)} rows < {MIN_TRAIN_ROWS})", flush=True)
        return None

    print(f"  {len(frame)} rows — training...", flush=True)
    if is_soccer_league(league):
        model = train_soccer_ensemble(league, frame)
        if not model:
            return None
        holdout = frame.tail(max(30, len(frame) // 5))
        from web.ensemble_ml.model import predict_soccer
        from web.ensemble_ml.features import extract_soccer_features

        # Quick holdout log-loss proxy on home outcome
        losses = []
        for _, row in holdout.iterrows():
            feats = {col: row.get(col) for col in model.feature_columns}
            pred = predict_soccer(model, feats)
            p = pred["home_win_probability"] / 100.0
            losses.append(-(row["home_win"] * __import__("math").log(max(p, 1e-6))))
        meta = {
            "league": league,
            "model_type": "soccer_threeway",
            "train_rows": len(frame),
            "holdout_home_logloss": round(sum(losses) / len(losses), 4) if losses else None,
            "trained_at": date.today().isoformat(),
            "features": list(model.feature_columns),
        }
        save_ensemble(league, model, meta)
        print(f"  saved soccer ensemble (holdout home logloss {meta['holdout_home_logloss']})", flush=True)
        return meta

    model = train_binary_ensemble(league, frame)
    if not model:
        return None

    holdout = frame.tail(max(30, len(frame) // 5))
    from web.ensemble_ml.model import predict_binary

    probs = []
    outcomes = []
    for _, row in holdout.iterrows():
        feats = {col: row.get(col) for col in model.feature_columns}
        pred = predict_binary(model, feats)
        probs.append(pred["home_win_probability"] / 100.0)
        outcomes.append(int(row["home_win"]))

    import numpy as np

    prob_arr = np.asarray(probs, dtype=float)
    out_arr = np.asarray(outcomes, dtype=float)
    metrics = evaluate_holdout_logloss(model, frame, league=league)
    nhl_score = metrics.get("all_market_logloss", metrics.get("nhl_market_holdout_logloss"))
    meta = {
        "league": league,
        "model_type": "binary_spread" if model.spread_league else "binary_moneyline",
        "train_rows": len(frame),
        "games_available": profile.get("games_available"),
        "market_blend_weight": model.market_blend_weight,
        "temperature": model.temperature,
        "dataset_profile": {
            key: profile[key]
            for key in (
                "max_calibration_games",
                "target_rows",
                "power_train_window",
                "dated_source",
            )
            if key in profile
        },
        "holdout_logloss": round(metrics.get("holdout_logloss", log_loss(prob_arr, out_arr)), 5),
        "market_holdout_logloss": metrics.get("market_holdout_logloss"),
        "holdout_logloss_market_games": metrics.get("holdout_logloss_market_games"),
        "nhl_market_holdout_logloss": metrics.get("nhl_market_holdout_logloss"),
        "nhl_market_baseline_logloss": metrics.get("nhl_market_baseline_logloss"),
        "all_market_logloss": metrics.get("all_market_logloss"),
        "all_market_baseline_logloss": metrics.get("all_market_baseline_logloss"),
        "all_market_games": metrics.get("all_market_games"),
        "target_logloss": NHL_TARGET_LOGLOSS if league == "nhl" else None,
        "beats_market_target": (
            nhl_score < NHL_TARGET_LOGLOSS
            if league == "nhl" and nhl_score is not None
            else None
        ),
        "holdout_brier": round(brier(prob_arr, out_arr), 4),
        "margin_sigma": model.margin_sigma,
        "trained_at": date.today().isoformat(),
        "features": list(model.feature_columns),
    }
    save_ensemble(league, model, meta)
    print(
        f"  saved ({meta['model_type']}) logloss={meta['holdout_logloss']} "
        f"market={meta.get('market_holdout_logloss')} "
        f"brier={meta['holdout_brier']}"
        + (
            f" all_market={meta.get('all_market_logloss')}"
            f" beats_target={meta.get('beats_market_target')}"
            if league == "nhl"
            else ""
        ),
        flush=True,
    )
    return meta


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train per-sport ensemble ML mega-models.")
    parser.add_argument("--leagues", default=",".join(CORE_LEAGUES))
    parser.add_argument("--cutoff", default=_cutoff_today())
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use legacy 250-game subsample caps (not recommended).",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Ignore cached walk-forward dataset and rebuild from scratch.",
    )
    args = parser.parse_args()
    leagues = [item.strip().lower() for item in args.leagues.split(",") if item.strip()]

    summary: dict[str, dict] = {}
    trained = 0
    for league in leagues:
        if league not in TRAIN_LEAGUES:
            print(f"Skipping unknown league {league}", flush=True)
            continue
        profile = dict(get_dataset_profile(league, fast=args.fast))
        from web.ensemble_ml.dataset import _load_dated_games

        profile["games_available"] = len(
            _load_dated_games(
                league,
                args.cutoff,
                dated_source=str(profile.get("dated_source", "espn")),
            )
        )
        entry = train_league(
            league,
            args.cutoff,
            profile=profile,
            rebuild_dataset=args.rebuild_dataset,
        )
        if entry:
            summary[league] = entry
            trained += 1

    clear_ensemble_caches()
    report_path = PROJECT_ROOT / "data" / "supplemental" / "ensemble-ml" / "train_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"trained_at": date.today().isoformat(), "leagues": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Trained {trained} leagues -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
