"""Train Soccer GradientBoost v2 with walk-forward validation.

Pooled top-5-league 1X2 classifier: XGBoost multi:softprob + multinomial
logistic regression ensemble, per-class isotonic calibration. Every season
2006-2025 is predicted by models trained only on earlier seasons; the market
benchmark is the devigged Pinnacle/average closing line (opening odds where
close is unavailable, pre-2015).

Artifacts -> data/models/soccer_v2/:
  model_clf.json                XGBoost softprob booster
  model_lr.json                 multinomial LR (mean/scale/coef/intercept)
  model_goals_{home,away}.json  goals regressors (display xG)
  calibrator.json               per-class isotonic (x/y for numpy.interp)
  state_{season}.json.gz        country-pool engine snapshots (last 2 seasons)
  metadata.json                 feature list, metrics, per-season table
  oos_predictions.csv           walk-forward OOS predictions (gitignored)
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_v2.data import (  # noqa: E402
    DIVISION_COUNTRY,
    HISTORY_DIR,
    all_division_codes,
    current_season,
    load_division_season,
    seasons_for_division,
)
from web.soccer_v2.feature_engine import FEATURE_COLUMNS, SoccerFeatureEngine  # noqa: E402
from web.soccer_v2.replay import merge_country_rows, replay_country  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "soccer_v2"
TABLE_PATH = HISTORY_DIR / "training_table.csv"

FIRST_EVAL_SEASON = 2006
MIN_GAMES_PLAYED = 5
CLASSES = ("H", "D", "A")

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 800,
    "learning_rate": 0.02,
    "max_depth": 5,
    "min_child_weight": 20.0,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}
LR_C = 0.05
XGB_WEIGHT = 0.55


def _devig(row: pd.Series, prefix: str) -> tuple[float, float, float] | None:
    h, d, a = (
        row.get(f"{prefix}_home"),
        row.get(f"{prefix}_draw"),
        row.get(f"{prefix}_away"),
    )
    if pd.isna(h) or pd.isna(d) or pd.isna(a):
        return None
    if h <= 1.0 or d <= 1.0 or a <= 1.0:
        return None
    inv = (1.0 / h, 1.0 / d, 1.0 / a)
    total = sum(inv)
    return inv[0] / total, inv[1] / total, inv[2] / total


def _logloss_row(probs: tuple[float, float, float], label: int) -> float:
    return -float(np.log(max(probs[label], 1e-12)))


def _fit_lr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=LR_C, max_iter=3000)
    lr.fit(scaler.transform(x), y)
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "classes": lr.classes_.tolist(),
        "xgb_weight": XGB_WEIGHT,
    }


def _lr_probs(lr: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(lr["mean"], dtype=float)
    scale = np.asarray(lr["scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    z = (x - mean) / scale @ np.asarray(lr["coef"], dtype=float).T + np.asarray(
        lr["intercept"], dtype=float
    )
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    order = [int(np.where(np.asarray(lr["classes"]) == cls)[0][0]) for cls in range(3)]
    return probs[:, order]


def _fit_isotonic(oos_probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    calibrator: dict[str, Any] = {"classes": list(CLASSES), "x": [], "y": []}
    for cls in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.97)
        iso.fit(oos_probs[:, cls], (labels == cls).astype(float))
        xs = np.linspace(0.0, 1.0, 201)
        ys = iso.predict(xs)
        calibrator["x"].append(xs.tolist())
        calibrator["y"].append(ys.tolist())
    return calibrator


def _apply_isotonic(calibrator: dict[str, Any], probs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs)
    for cls in range(3):
        out[:, cls] = np.interp(
            probs[:, cls], calibrator["x"][cls], calibrator["y"][cls]
        )
    out = np.clip(out, 1e-4, None)
    return out / out.sum(axis=1, keepdims=True)


MARKET_FEATURES = ("mkt_open_home", "mkt_open_draw", "mkt_open_away", "has_market")


def _attach_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devigged opening-odds probabilities as features (0 when missing)."""
    inv_h = np.where(df["open_home"] > 1.0, 1.0 / df["open_home"], np.nan)
    inv_d = np.where(df["open_draw"] > 1.0, 1.0 / df["open_draw"], np.nan)
    inv_a = np.where(df["open_away"] > 1.0, 1.0 / df["open_away"], np.nan)
    total = inv_h + inv_d + inv_a
    with np.errstate(invalid="ignore"):
        mh = inv_h / total
        md = inv_d / total
        ma = inv_a / total
    has = ~np.isnan(mh)
    df["mkt_open_home"] = np.where(has, mh, 0.0)
    df["mkt_open_draw"] = np.where(has, md, 0.0)
    df["mkt_open_away"] = np.where(has, ma, 0.0)
    df["has_market"] = has.astype(float)
    return df


def main() -> None:
    from xgboost import XGBClassifier, XGBRegressor

    df = pd.read_csv(TABLE_PATH)
    df = df[
        (df["home_games_played"] >= MIN_GAMES_PLAYED)
        & (df["away_games_played"] >= MIN_GAMES_PLAYED)
    ].reset_index(drop=True)
    df["label"] = df["result"].map({"H": 0, "D": 1, "A": 2}).astype(int)
    df = _attach_market_features(df)
    feature_cols = list(FEATURE_COLUMNS)
    market_cols = feature_cols + list(MARKET_FEATURES)
    print(f"rows: {len(df)}, features: {len(feature_cols)}", flush=True)

    seasons = sorted(df["season"].unique())
    eval_seasons = [s for s in seasons if s >= FIRST_EVAL_SEASON]

    oos_frames: list[pd.DataFrame] = []
    per_season: list[dict[str, Any]] = []

    for season in eval_seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if len(train) < 4000 or test.empty:
            continue
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)

        def _ensemble(cols: list[str]) -> np.ndarray:
            x_train = train[cols].to_numpy(dtype=float)
            x_test = test[cols].to_numpy(dtype=float)
            clf = XGBClassifier(**XGB_PARAMS)
            clf.fit(x_train, y_train, verbose=False)
            xgb_probs = clf.predict_proba(x_test)
            lr = _fit_lr(x_train, y_train)
            lr_probs = _lr_probs(lr, x_test)
            raw = XGB_WEIGHT * xgb_probs + (1.0 - XGB_WEIGHT) * lr_probs
            return raw / raw.sum(axis=1, keepdims=True)

        raw_pure = _ensemble(feature_cols)
        raw_mkt = _ensemble(market_cols)

        # walk-forward calibration: isotonic fit on OOS predictions from
        # earlier eval seasons only (identity when sample is thin)
        prior = pd.concat(oos_frames, ignore_index=True) if oos_frames else None
        if prior is not None and len(prior) >= 3000:
            calib_pure = _fit_isotonic(
                prior[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
                prior["label"].to_numpy(dtype=int),
            )
            cal_pure = _apply_isotonic(calib_pure, raw_pure)
            calib_mkt = _fit_isotonic(
                prior[["pm_home", "pm_draw", "pm_away"]].to_numpy(dtype=float),
                prior["label"].to_numpy(dtype=int),
            )
            cal_mkt = _apply_isotonic(calib_mkt, raw_mkt)
        else:
            cal_pure = raw_pure
            cal_mkt = raw_mkt

        frame = test[
            [
                "league",
                "season",
                "date",
                "home",
                "away",
                "label",
                "result",
                "open_home",
                "open_draw",
                "open_away",
                "close_home",
                "close_draw",
                "close_away",
                "b365_home",
                "b365_draw",
                "b365_away",
                "b365c_home",
                "b365c_draw",
                "b365c_away",
                "max_home",
                "max_draw",
                "max_away",
            ]
        ].copy()
        frame["p_home"] = raw_pure[:, 0]
        frame["p_draw"] = raw_pure[:, 1]
        frame["p_away"] = raw_pure[:, 2]
        frame["pm_home"] = raw_mkt[:, 0]
        frame["pm_draw"] = raw_mkt[:, 1]
        frame["pm_away"] = raw_mkt[:, 2]
        frame["cal_home"] = cal_pure[:, 0]
        frame["cal_draw"] = cal_pure[:, 1]
        frame["cal_away"] = cal_pure[:, 2]
        frame["calm_home"] = cal_mkt[:, 0]
        frame["calm_draw"] = cal_mkt[:, 1]
        frame["calm_away"] = cal_mkt[:, 2]
        oos_frames.append(frame)

        model_ll = float(
            np.mean(
                [_logloss_row(tuple(cal_pure[i]), y_test[i]) for i in range(len(test))]
            )
        )
        model_mkt_ll = float(
            np.mean(
                [_logloss_row(tuple(cal_mkt[i]), y_test[i]) for i in range(len(test))]
            )
        )

        market_lls: list[float] = []
        model_lls_market_subset: list[float] = []
        model_mkt_lls_subset: list[float] = []
        for i, (_, row) in enumerate(test.iterrows()):
            market = _devig(row, "close") or _devig(row, "open")
            if market is None:
                continue
            market_lls.append(_logloss_row(market, y_test[i]))
            model_lls_market_subset.append(_logloss_row(tuple(cal_pure[i]), y_test[i]))
            model_mkt_lls_subset.append(_logloss_row(tuple(cal_mkt[i]), y_test[i]))

        acc = float(np.mean(cal_mkt.argmax(axis=1) == y_test))
        per_season.append(
            {
                "season": int(season),
                "n": int(len(test)),
                "model_logloss": round(model_ll, 5),
                "model_market_aware_logloss": round(model_mkt_ll, 5),
                "model_acc": round(acc, 4),
                "n_with_odds": len(market_lls),
                "model_logloss_odds_subset": round(
                    float(np.mean(model_lls_market_subset)), 5
                )
                if model_lls_market_subset
                else None,
                "model_market_aware_logloss_odds_subset": round(
                    float(np.mean(model_mkt_lls_subset)), 5
                )
                if model_mkt_lls_subset
                else None,
                "market_logloss": round(float(np.mean(market_lls)), 5)
                if market_lls
                else None,
            }
        )
        print(
            f"season {season}: n={len(test)} pure_ll={model_ll:.5f} "
            f"mkt_aware_ll={model_mkt_ll:.5f} "
            f"market_ll={per_season[-1]['market_logloss']} acc={acc:.4f}",
            flush=True,
        )

    oos = pd.concat(oos_frames, ignore_index=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    oos.to_csv(MODEL_DIR / "oos_predictions.csv", index=False)

    overall_model = float(
        np.mean(
            [
                _logloss_row((row.cal_home, row.cal_draw, row.cal_away), int(row.label))
                for row in oos.itertuples()
            ]
        )
    )
    subset_model: list[float] = []
    subset_model_mkt: list[float] = []
    subset_market: list[float] = []
    for row in oos.itertuples():
        market = _devig(
            pd.Series(
                {
                    "close_home": row.close_home,
                    "close_draw": row.close_draw,
                    "close_away": row.close_away,
                }
            ),
            "close",
        ) or _devig(
            pd.Series(
                {
                    "open_home": row.open_home,
                    "open_draw": row.open_draw,
                    "open_away": row.open_away,
                }
            ),
            "open",
        )
        if market is None:
            continue
        subset_market.append(_logloss_row(market, int(row.label)))
        subset_model.append(
            _logloss_row((row.cal_home, row.cal_draw, row.cal_away), int(row.label))
        )
        subset_model_mkt.append(
            _logloss_row((row.calm_home, row.calm_draw, row.calm_away), int(row.label))
        )
    print(
        f"OVERALL: pure_ll={overall_model:.5f} "
        f"pure_ll_odds_subset={np.mean(subset_model):.5f} "
        f"mkt_aware_ll_odds_subset={np.mean(subset_model_mkt):.5f} "
        f"market_ll={np.mean(subset_market):.5f} n_odds={len(subset_market)}",
        flush=True,
    )

    # ---- final artifacts: train on everything -----------------------------
    y_all = df["label"].to_numpy(dtype=int)

    def _save_final(cols: list[str], suffix: str) -> Any:
        x_all = df[cols].to_numpy(dtype=float)
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(x_all, y_all, verbose=False)
        clf.get_booster().save_model(str(MODEL_DIR / f"model_clf{suffix}.json"))
        lr = _fit_lr(x_all, y_all)
        (MODEL_DIR / f"model_lr{suffix}.json").write_text(
            json.dumps(lr), encoding="utf-8"
        )
        return clf

    final_clf = _save_final(feature_cols, "")
    _save_final(market_cols, "_market")

    final_cal = _fit_isotonic(
        oos[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
        oos["label"].to_numpy(dtype=int),
    )
    (MODEL_DIR / "calibrator.json").write_text(json.dumps(final_cal), encoding="utf-8")
    final_cal_mkt = _fit_isotonic(
        oos[["pm_home", "pm_draw", "pm_away"]].to_numpy(dtype=float),
        oos["label"].to_numpy(dtype=int),
    )
    (MODEL_DIR / "calibrator_market.json").write_text(
        json.dumps(final_cal_mkt), encoding="utf-8"
    )

    x_all = df[feature_cols].to_numpy(dtype=float)

    goals_params = {**XGB_PARAMS}
    goals_params.pop("num_class")
    goals_params["objective"] = "reg:squarederror"
    goals_params["eval_metric"] = "rmse"
    goals_params["n_estimators"] = 500
    for target, name in (
        ("home_goals", "model_goals_home"),
        ("away_goals", "model_goals_away"),
    ):
        reg = XGBRegressor(**goals_params)
        reg.fit(x_all, df[target].to_numpy(dtype=float), verbose=False)
        reg.get_booster().save_model(str(MODEL_DIR / f"{name}.json"))

    # ---- engine snapshots (end of last two completed seasons) -------------
    through = current_season()
    completed = [int(s) for s in seasons if int(s) < through]
    snapshot_seasons = sorted(completed)[-2:]
    by_country: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for division in all_division_codes():
        rows: list[dict[str, Any]] = []
        for season in seasons_for_division(division, through=through):
            rows.extend(load_division_season(division, season))
        by_country[DIVISION_COUNTRY[division]][division] = rows

    for snap_season in snapshot_seasons:
        pools: dict[str, Any] = {}
        for country, divisions in sorted(by_country.items()):
            engine = SoccerFeatureEngine(country)
            merged = [
                row
                for row in merge_country_rows(divisions)
                if int(row["season"]) <= snap_season
            ]
            replay_country(engine, merged)
            pools[country] = engine.to_dict()
        payload = {"season": snap_season, "pools": pools}
        with gzip.open(
            MODEL_DIR / f"state_{snap_season}.json.gz", "wt", encoding="utf-8"
        ) as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"snapshot saved: season {snap_season}", flush=True)

    importance = final_clf.get_booster().get_score(importance_type="gain")
    total_gain = sum(importance.values()) or 1.0
    named = {
        feature_cols[int(key[1:])]: round(value / total_gain, 5)
        for key, value in importance.items()
    }

    metadata = {
        "algorithm": "SoccerGradientBoost v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_cols,
        "market_feature_columns": list(MARKET_FEATURES),
        "ensemble": {"xgb_weight": XGB_WEIGHT, "lr_C": LR_C},
        "train_rows": int(len(df)),
        "train_seasons": [int(seasons[0]), int(seasons[-1])],
        "eval_seasons": [int(eval_seasons[0]), int(eval_seasons[-1])],
        "snapshot_seasons": [int(s) for s in snapshot_seasons],
        "oos_model_logloss": round(overall_model, 5),
        "oos_with_odds": len(subset_market),
        "oos_model_logloss_odds_subset": round(float(np.mean(subset_model)), 5),
        "oos_market_aware_logloss_odds_subset": round(
            float(np.mean(subset_model_mkt)), 5
        ),
        "oos_market_logloss": round(float(np.mean(subset_market)), 5),
        "per_season": per_season,
        "clf_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
        "feature_importance": dict(sorted(named.items(), key=lambda item: -item[1])),
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"artifacts saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
