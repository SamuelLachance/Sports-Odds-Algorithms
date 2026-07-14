"""Ship hybrid winners into data/models/{league}_v2 for live serving."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from web.hybrid_v2.adapters import LeagueAdapter, get_adapter
from web.hybrid_v2.config import HybridConfig
from web.hybrid_v2.train import prepare_frame, select_feature_columns


def _load_best(adapter: LeagueAdapter) -> dict[str, Any] | None:
    path = adapter.hybrid_dir / "best_config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ship_league(league: str, *, force: bool = False) -> dict[str, Any]:
    """Fit final CatBoost on full history and write hybrid live artifacts.

    Keeps existing XGB margin/score boosters unless hybrid margin beats baseline MAE.
    Only ships classifier path when beat_baseline is True (unless force=True).
    """
    adapter = get_adapter(league)
    best = _load_best(adapter)
    if best is None:
        raise FileNotFoundError(f"No best_config.json for {league}")
    if not best.get("beat_baseline") and not force:
        return {
            "league": league,
            "shipped": False,
            "reason": "did_not_beat_baseline",
            "best": best,
        }

    cfg = HybridConfig(**best["config"])
    frame = prepare_frame(adapter)
    cols = select_feature_columns(adapter, cfg)
    # Final fit: last season held for early stopping
    end_season = int(frame.season.max())
    train = frame[frame.season < end_season]
    val = frame[frame.season == end_season]
    if train.empty or len(val) < 50:
        train = frame
        val = frame

    adapter.v2_dir.mkdir(parents=True, exist_ok=True)
    hybrid_meta: dict[str, Any] = {
        "backend": cfg.backend if cfg.backend != "both" else "catboost",
        "market_blend_w": cfg.market_blend_w,
        "target_mode": cfg.target_mode,
        "feature_mode": cfg.feature_mode,
        "use_curves": cfg.use_curves,
        "use_categoricals": cfg.use_categoricals,
        "use_market": cfg.use_market,
        "multiclass": adapter.multiclass,
        "config": cfg.to_dict(),
        "oos_summary": {k: best[k] for k in best if k != "config"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    shipped_margin = False
    if adapter.multiclass:
        from catboost import CatBoostClassifier, Pool

        cat_cols = [c for c in ("home_id", "away_id") if c in train.columns and cfg.use_categoricals]
        feature_cols = [c for c in cols if c in train.columns] + cat_cols
        model = CatBoostClassifier(
            iterations=cfg.iterations,
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            l2_leaf_reg=cfg.l2,
            loss_function="MultiClass",
            classes_count=3,
            random_seed=cfg.seed,
            od_type="Iter",
            od_wait=cfg.early_stopping,
            verbose=False,
            allow_writing_files=False,
            bootstrap_type="Bernoulli",
            subsample=cfg.subsample,
            min_data_in_leaf=cfg.min_data,
        )
        model.fit(
            Pool(train[feature_cols], train["label"], cat_features=cat_cols or None),
            eval_set=Pool(val[feature_cols], val["label"], cat_features=cat_cols or None),
            use_best_model=True,
        )
        clf_path = adapter.v2_dir / "model_clf_hybrid.cbm"
        model.save_model(str(clf_path))
        hybrid_meta["feature_cols"] = feature_cols
        hybrid_meta["cat_cols"] = cat_cols
        hybrid_meta["clf_path"] = clf_path.name
    else:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        # Prefer catboost even if best was "both" — ship single backend for serve simplicity
        cat_cols = [c for c in ("home_id", "away_id") if c in train.columns and cfg.use_categoricals]
        feature_cols = [c for c in cols if c in train.columns] + cat_cols
        model = CatBoostClassifier(
            iterations=cfg.iterations,
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            l2_leaf_reg=cfg.l2,
            loss_function="Logloss",
            random_seed=cfg.seed,
            od_type="Iter",
            od_wait=cfg.early_stopping,
            verbose=False,
            allow_writing_files=False,
            subsample=cfg.subsample,
            min_data_in_leaf=cfg.min_data,
        )
        from web.hybrid_v2.train import _market_logit_baseline

        offset = cfg.target_mode == "offset"
        model.fit(
            Pool(
                train[feature_cols],
                train["home_win"],
                cat_features=cat_cols or None,
                baseline=_market_logit_baseline(train) if offset else None,
            ),
            eval_set=Pool(
                val[feature_cols],
                val["home_win"],
                cat_features=cat_cols or None,
                baseline=_market_logit_baseline(val) if offset else None,
            ),
            use_best_model=True,
        )
        clf_path = adapter.v2_dir / "model_clf_hybrid.cbm"
        model.save_model(str(clf_path))
        hybrid_meta["feature_cols"] = feature_cols
        hybrid_meta["cat_cols"] = cat_cols
        hybrid_meta["clf_path"] = clf_path.name

        if adapter.has_margin and "margin" in train.columns and best.get("beat_margin_baseline"):
            m_cols = [c for c in cols if c in train.columns]
            m_model = CatBoostRegressor(
                iterations=min(cfg.iterations, 400),
                depth=min(cfg.depth, 6),
                learning_rate=cfg.learning_rate,
                l2_leaf_reg=cfg.l2,
                loss_function="MAE",
                random_seed=cfg.seed,
                od_type="Iter",
                od_wait=cfg.early_stopping,
                verbose=False,
                allow_writing_files=False,
            )
            m_model.fit(
                Pool(train[m_cols], train["margin"]),
                eval_set=Pool(val[m_cols], val["margin"]),
                use_best_model=True,
            )
            m_path = adapter.v2_dir / "model_margin_hybrid.cbm"
            m_model.save_model(str(m_path))
            hybrid_meta["margin_path"] = m_path.name
            hybrid_meta["margin_feature_cols"] = m_cols
            shipped_margin = True

    # Copy best OOS for bet backtests if present
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.name)
    oos_src = adapter.hybrid_dir / f"oos_{safe}.csv"
    if oos_src.is_file():
        shutil.copy2(oos_src, adapter.v2_dir / "oos_predictions_hybrid.csv")

    # Update metadata — keep prior XGB metrics under baseline_* keys
    meta_path = adapter.v2_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta["algorithm"] = "HybridGradientBoost v2"
    meta["hybrid"] = True
    meta["hybrid_backend"] = hybrid_meta["backend"]
    meta["hybrid_market_blend_w"] = cfg.market_blend_w
    meta["baseline_xgb_logloss"] = adapter.baseline_ll
    meta["oos_model_logloss"] = best.get("oos_model_logloss")
    meta["oos_model_acc"] = best.get("oos_model_acc")
    meta["oos_model_brier"] = best.get("oos_model_brier")
    if best.get("oos_margin_mae") is not None and shipped_margin:
        meta["oos_margin_mae"] = best["oos_margin_mae"]
    for k in ("oos_market_logloss", "oos_model_logloss_odds_subset", "gap_vs_market"):
        if k in best:
            meta[k] = best[k]
    meta["hybrid_shipped_at"] = datetime.now(timezone.utc).isoformat()

    # Real post-blend calibrator fit on the winner's walk-forward OOS preds so
    # live probs match evaluated behavior; identity fallback when OOS is absent.
    # Runs before the metadata writes: logit_stack freezes its final market
    # weight here and it must land in hybrid_meta.json.
    cal: dict[str, Any] = {"x": [0.0, 0.5, 1.0], "y": [0.005, 0.5, 0.995], "kind": "identity_clip"}
    if not adapter.multiclass and oos_src.is_file():
        try:
            from sklearn.isotonic import IsotonicRegression

            oos_frame = pd.read_csv(oos_src)
            if {"model_raw", "home_win"}.issubset(oos_frame.columns) and len(oos_frame) >= 500:
                cal_values = oos_frame["model_raw"].astype(float).to_numpy()
                y_oos = oos_frame["home_win"].astype(float).to_numpy()
                iso = IsotonicRegression(
                    out_of_bounds="clip", y_min=0.005, y_max=0.995
                )
                iso.fit(cal_values, y_oos)
                if cfg.target_mode == "logit_stack":
                    # Live order is calibrate -> logit-stack: freeze the final
                    # market weight fit on calibrated model probs vs the close.
                    from web.hybrid_v2.train import fit_logit_stack_w

                    mkt = (
                        oos_frame["market_home_prob"].astype(float).to_numpy()
                        if "market_home_prob" in oos_frame.columns
                        else np.full(len(oos_frame), np.nan)
                    )
                    w_final = fit_logit_stack_w(iso.predict(cal_values), mkt, y_oos)
                    hybrid_meta["market_blend_w"] = w_final
                    meta["hybrid_market_blend_w"] = w_final
                cal = {
                    "x": [float(v) for v in iso.X_thresholds_],
                    "y": [float(v) for v in iso.y_thresholds_],
                    "kind": "isotonic_oos",
                    "n_fit": int(len(oos_frame)),
                }
        except Exception:  # noqa: BLE001 — calibrator is optional, never block a ship
            pass
    (adapter.v2_dir / "calibrator_hybrid.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (adapter.v2_dir / "hybrid_meta.json").write_text(json.dumps(hybrid_meta, indent=2), encoding="utf-8")

    return {
        "league": league,
        "shipped": True,
        "shipped_margin": shipped_margin,
        "hybrid_meta": hybrid_meta,
        "oos_model_logloss": best.get("oos_model_logloss"),
        "baseline_ll": adapter.baseline_ll,
    }
