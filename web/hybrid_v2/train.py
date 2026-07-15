"""Walk-forward hybrid training for binary and soccer multiclass targets."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from web.hybrid_v2.adapters import LeagueAdapter, get_adapter
from web.hybrid_v2.config import HybridConfig, focused_search_space
from web.hybrid_v2.curves import CURVE_FEATURE_COLUMNS, attach_curve_features

CLF_MARKET_FEATURES = ("mkt_home_prob", "has_market", "ml_steam_pp", "has_steam")
SOCCER_MARKET_FEATURES = ("mkt_open_home", "mkt_open_draw", "mkt_open_away", "has_market")
SOCCER_CLASSES = ("H", "D", "A")


def log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_multi(proba: np.ndarray, y: np.ndarray) -> float:
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    rows = np.arange(len(y))
    return float(-np.mean(np.log(proba[rows, y.astype(int)])))


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def select_feature_columns(adapter: LeagueAdapter, cfg: HybridConfig) -> list[str]:
    full = list(adapter.feature_columns())
    core = list(adapter.resolve_core())
    if cfg.feature_mode == "core":
        cols = [c for c in core if c in full or c in CURVE_FEATURE_COLUMNS]
    elif cfg.feature_mode == "curves_plus_core":
        cols = [c for c in core if c in full]
    else:
        cols = list(full)
    # Keep only columns that will exist after prepare (filter unknowns later)
    if cfg.use_curves:
        cols = cols + [c for c in CURVE_FEATURE_COLUMNS if c not in cols]
    if cfg.use_market:
        mkt = SOCCER_MARKET_FEATURES if adapter.multiclass else CLF_MARKET_FEATURES
        cols = cols + [c for c in mkt if c not in cols]
    if cfg.target_mode == "decorrelated" and not cfg.use_market:
        # A truly odds-blind Hubáček arm drops every market-derived feature
        # (steam, line moves, totals...), not just the two direct columns.
        from web.hubacek_v2.features import strip_market_features

        cols = strip_market_features(cols)
    return cols


def _market_logit_baseline(frame) -> np.ndarray:
    """logit(de-vigged close) as CatBoost init predictions; 0 (=50%) when absent."""
    mkt = frame["market_home_prob"].to_numpy(dtype=float) if "market_home_prob" in frame.columns else np.full(len(frame), np.nan)
    p = np.clip(mkt, 1e-4, 1 - 1e-4)
    out = np.where(np.isfinite(mkt), np.log(p / (1 - p)), 0.0)
    return out.astype(float)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


LOGIT_STACK_DEFAULT_W = 0.85
LOGIT_STACK_MIN_POOL = 300


def fit_logit_stack_w(model_p: np.ndarray, market_p: np.ndarray, y: np.ndarray) -> float:
    """Grid-fit the logit-space market weight minimizing log-loss."""
    mask = np.isfinite(market_p) & np.isfinite(model_p) & np.isfinite(y)
    if mask.sum() < LOGIT_STACK_MIN_POOL:
        return LOGIT_STACK_DEFAULT_W
    lm, lk, yy = _logit(model_p[mask]), _logit(market_p[mask]), y[mask]
    best_w, best_ll = LOGIT_STACK_DEFAULT_W, np.inf
    for w in np.arange(0.0, 1.0001, 0.05):
        p = 1.0 / (1.0 + np.exp(-((1.0 - w) * lm + w * lk)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        cur = float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
        if cur < best_ll:
            best_ll, best_w = cur, float(w)
    return best_w


def apply_logit_stack(model_p: np.ndarray, market_p: np.ndarray, w: float) -> np.ndarray:
    """Logit-space mix on rows with a market; raw model prob elsewhere."""
    out = model_p.astype(float).copy()
    mask = np.isfinite(market_p)
    out[mask] = 1.0 / (
        1.0 + np.exp(-((1.0 - w) * _logit(model_p[mask]) + w * _logit(market_p[mask])))
    )
    return out


def _fit_catboost_binary(train, val, cols, cfg):
    from catboost import CatBoostClassifier, Pool

    cat_cols = [c for c in ("home_id", "away_id") if c in train.columns and cfg.use_categoricals]
    feature_cols = [c for c in cols if c in train.columns] + cat_cols
    offset = cfg.target_mode == "offset"
    train_pool = Pool(
        train[feature_cols],
        train["home_win"],
        cat_features=cat_cols or None,
        baseline=_market_logit_baseline(train) if offset else None,
    )
    val_pool = Pool(
        val[feature_cols],
        val["home_win"],
        cat_features=cat_cols or None,
        baseline=_market_logit_baseline(val) if offset else None,
    )
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


def _fit_catboost_decorrelated(train, cols, cfg):
    """Hubáček β-model: CatBoostRegressor on raw scores with the decorrelated
    custom objective (logloss − c·(p̂−q)²). Fixed iterations — the python
    objective has no paired eval metric for early stopping. Prediction is
    sigmoid(raw score)."""
    from catboost import CatBoostRegressor, Pool

    from web.hubacek_v2.objective import DecorrelatedLogloss

    cat_cols = [c for c in ("home_id", "away_id") if c in train.columns and cfg.use_categoricals]
    feature_cols = [c for c in cols if c in train.columns] + cat_cols
    q_train = (
        train["market_home_prob"].to_numpy(dtype=float)
        if "market_home_prob" in train.columns
        else np.full(len(train), np.nan)
    )
    model = CatBoostRegressor(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        l2_leaf_reg=cfg.l2,
        loss_function=DecorrelatedLogloss(q_train, cfg.decorrelation_c),
        eval_metric="RMSE",
        random_seed=cfg.seed,
        verbose=False,
        allow_writing_files=False,
        min_data_in_leaf=cfg.min_data,
        # Python custom objectives require plain boosting on CPU.
        boosting_type="Plain",
    )
    model.fit(Pool(train[feature_cols], train["home_win"], cat_features=cat_cols or None))
    return model, feature_cols, cat_cols


def _predict_catboost_decorrelated(model, frame, feature_cols, cat_cols):
    from catboost import Pool

    from web.hubacek_v2.objective import sigmoid

    raw = model.predict(Pool(frame[feature_cols], cat_features=cat_cols or None))
    return sigmoid(np.asarray(raw, dtype=float))


def _fit_catboost_multi(train, val, cols, cfg):
    from catboost import CatBoostClassifier, Pool

    cat_cols = [c for c in ("home_id", "away_id") if c in train.columns and cfg.use_categoricals]
    feature_cols = [c for c in cols if c in train.columns] + cat_cols
    train_pool = Pool(train[feature_cols], train["label"], cat_features=cat_cols or None)
    val_pool = Pool(val[feature_cols], val["label"], cat_features=cat_cols or None)
    model = CatBoostClassifier(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        l2_leaf_reg=cfg.l2,
        loss_function="MultiClass",
        eval_metric="MultiClass",
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
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, feature_cols, cat_cols


def _fit_lightgbm_binary(train, val, cols, cfg):
    import lightgbm as lgb

    feature_cols = [c for c in cols if c in train.columns]
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


def _fit_lightgbm_multi(train, val, cols, cfg):
    import lightgbm as lgb

    feature_cols = [c for c in cols if c in train.columns]
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
        objective="multiclass",
        num_class=3,
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
        train["label"],
        eval_set=[(val_x[feature_cols], val["label"])],
        categorical_feature=cat_feature_idx or "auto",
        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)],
    )
    return model, feature_cols, cat_maps


def _predict_catboost(model, frame, feature_cols, cat_cols, *, multiclass: bool, offset: bool = False):
    from catboost import Pool

    pool = Pool(
        frame[feature_cols],
        cat_features=cat_cols or None,
        baseline=None if multiclass or not offset else _market_logit_baseline(frame),
    )
    proba = model.predict_proba(pool)
    if multiclass:
        return np.asarray(proba, dtype=float)
    return np.asarray(proba, dtype=float)[:, 1]


def _predict_lightgbm(model, frame, feature_cols, cat_maps, *, multiclass: bool):
    data = {}
    for c in feature_cols:
        if c in cat_maps:
            data[c] = frame[c].astype(str).map(cat_maps[c]).fillna(-1).astype(int)
        elif c in frame.columns:
            data[c] = frame[c].to_numpy()
        else:
            data[c] = np.zeros(len(frame))
    x = pd.DataFrame(data)[feature_cols]
    proba = model.predict_proba(x)
    if multiclass:
        return np.asarray(proba, dtype=float)
    return np.asarray(proba, dtype=float)[:, 1]


def _fit_margin_catboost(train, val, cols, cfg):
    from catboost import CatBoostRegressor, Pool

    feature_cols = [c for c in cols if c in train.columns]
    if "margin" not in train.columns:
        return None, feature_cols
    train_pool = Pool(train[feature_cols], train["margin"])
    val_pool = Pool(val[feature_cols], val["margin"])
    model = CatBoostRegressor(
        iterations=min(cfg.iterations, 400),
        depth=min(cfg.depth, 6),
        learning_rate=cfg.learning_rate,
        l2_leaf_reg=cfg.l2,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=cfg.seed,
        od_type="Iter",
        od_wait=cfg.early_stopping,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, feature_cols


def prepare_frame(adapter: LeagueAdapter, *, end_season: int | None = None) -> pd.DataFrame:
    end_season = end_season or adapter.end_season
    if not adapter.table_path.is_file():
        raise FileNotFoundError(f"Missing training table: {adapter.table_path}")
    frame = pd.read_csv(adapter.table_path)
    if adapter.multiclass:
        if "label" not in frame.columns and "result" in frame.columns:
            mapping = {"H": 0, "D": 1, "A": 2}
            frame["label"] = frame["result"].map(mapping)
        frame = frame.dropna(subset=["label"])
        # Binary home_win for CurveFM Elo updates (home win only; draws → 0.5 via soft)
        if "home_win" not in frame.columns:
            frame["home_win"] = (frame["label"] == 0).astype(float)
            # Treat draws as half-win for Elo curve only
            frame.loc[frame["label"] == 1, "home_win"] = 0.5
    else:
        frame = frame.dropna(subset=["home_win"])
    frame = frame[frame.season <= end_season].copy()
    if adapter.market_attach_fn is not None:
        frame = adapter.market_attach_fn(frame)
    frame["home_id"] = frame[adapter.home_col].astype(str)
    frame["away_id"] = frame[adapter.away_col].astype(str)
    # Ensure score cols for CurveFM when available
    score_h = "home_score" if "home_score" in frame.columns else (
        "home_goals" if "home_goals" in frame.columns else None
    )
    score_a = "away_score" if "away_score" in frame.columns else (
        "away_goals" if "away_goals" in frame.columns else None
    )
    kwargs: dict = {}
    if score_h and score_a:
        kwargs = {"score_home_col": score_h, "score_away_col": score_a}
    if adapter.multiclass:
        # Soccer tables are huge; identity Elo CurveFM via itertuples is too slow.
        # Fill neutral curve columns; tree model still gets tabular + market blend.
        for c in CURVE_FEATURE_COLUMNS:
            frame[c] = 0.0
    else:
        frame = attach_curve_features(
            frame,
            home_col=adapter.home_col,
            away_col=adapter.away_col,
            refit_each_season=False,
            **kwargs,
        )
    if adapter.period_fn is not None:
        frame["period"] = adapter.period_fn(frame)
    else:
        frame["period"] = 0
    if adapter.has_margin and "margin" not in frame.columns and score_h and score_a:
        frame["margin"] = frame[score_h].astype(float) - frame[score_a].astype(float)
    return frame


def walk_forward(
    adapter: LeagueAdapter,
    frame: pd.DataFrame,
    cfg: HybridConfig,
    end_season: int,
) -> pd.DataFrame:
    cols = select_feature_columns(adapter, cfg)
    outputs: list[pd.DataFrame] = []
    calibrator: IsotonicRegression | None = None
    multi = adapter.multiclass

    for eval_season in range(adapter.first_eval_season, end_season + 1):
        periods = (0,) if multi else (0, 1, 2)
        for period in periods:
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
            min_val = 50 if multi else 100
            if fit.empty or len(val) < min_val:
                # No prior-season history (first fold): validate on a
                # chronological tail instead of early-stopping on the fit set.
                ordered = train.sort_values("date")
                n_val = max(min_val, int(round(0.15 * len(ordered))))
                if len(ordered) >= 2 * n_val:
                    fit = ordered.iloc[:-n_val]
                    val = ordered.iloc[-n_val:]
                else:
                    fit = train
                    val = train

            preds: list[np.ndarray] = []
            margin_preds: list[np.ndarray] = []
            if cfg.backend in ("catboost", "both"):
                if multi:
                    model, fcols, cat_cols = _fit_catboost_multi(fit, val, cols, cfg)
                    preds.append(_predict_catboost(model, test, fcols, cat_cols, multiclass=True))
                elif cfg.target_mode == "decorrelated":
                    model, fcols, cat_cols = _fit_catboost_decorrelated(train, cols, cfg)
                    preds.append(_predict_catboost_decorrelated(model, test, fcols, cat_cols))
                else:
                    model, fcols, cat_cols = _fit_catboost_binary(fit, val, cols, cfg)
                    preds.append(
                        _predict_catboost(
                            model, test, fcols, cat_cols,
                            multiclass=False, offset=cfg.target_mode == "offset",
                        )
                    )
                    if adapter.has_margin and "margin" in fit.columns:
                        m_model, m_cols = _fit_margin_catboost(fit, val, cols, cfg)
                        if m_model is not None:
                            from catboost import Pool

                            margin_preds.append(m_model.predict(Pool(test[m_cols])))
            if cfg.backend in ("lightgbm", "both"):
                if multi:
                    model, fcols, cat_maps = _fit_lightgbm_multi(fit, val, cols, cfg)
                    preds.append(_predict_lightgbm(model, test, fcols, cat_maps, multiclass=True))
                else:
                    model, fcols, cat_maps = _fit_lightgbm_binary(fit, val, cols, cfg)
                    preds.append(_predict_lightgbm(model, test, fcols, cat_maps, multiclass=False))

            if multi:
                raw = np.mean(np.stack(preds, axis=0), axis=0) if preds else np.full((len(test), 3), 1 / 3)
                if cfg.market_blend_w > 0:
                    w = float(cfg.market_blend_w)
                    mkt = test[["mkt_open_home", "mkt_open_draw", "mkt_open_away"]].to_numpy(dtype=float)
                    mask = test["has_market"].to_numpy(dtype=float) > 0.5
                    blended = raw.copy()
                    blended[mask] = (1.0 - w) * raw[mask] + w * mkt[mask]
                    # renormalize
                    blended = blended / blended.sum(axis=1, keepdims=True).clip(min=1e-9)
                    raw = blended
                # Calibrate home class only via isotonic on P(H)
                home_raw = raw[:, 0]
                home_cal = calibrator.predict(home_raw) if calibrator is not None else home_raw
                # redistribute: scale home, keep draw/away ratio
                scale = np.where(home_raw > 1e-6, home_cal / home_raw, 1.0)
                calibrated = raw.copy()
                calibrated[:, 0] = home_cal
                rest = raw[:, 1:] * scale[:, None]
                rest_sum = rest.sum(axis=1)
                target_rest = np.clip(1.0 - home_cal, 0.0, 1.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rest = np.where(
                        rest_sum[:, None] > 1e-9,
                        rest * (target_rest / rest_sum)[:, None],
                        np.column_stack([target_rest / 2, target_rest / 2]),
                    )
                calibrated[:, 1:] = rest
                chunk = test[
                    [c for c in ("season", "date", "event_id", "home", "away", "label", "result", "market_home_prob") if c in test.columns]
                ].copy()
                chunk["model_raw_h"] = raw[:, 0]
                chunk["model_prob_h"] = calibrated[:, 0]
                chunk["model_prob_d"] = calibrated[:, 1]
                chunk["model_prob_a"] = calibrated[:, 2]
                chunk["model_prob"] = calibrated[:, 0]  # for binary-style summarize helpers
                chunk["home_win"] = (test["label"] == 0).astype(float)
                outputs.append(chunk)
                pooled = pd.concat(outputs, ignore_index=True)
                if len(pooled) >= adapter.min_calibration_pool:
                    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.95)
                    calibrator.fit(pooled["model_raw_h"], pooled["home_win"])
            else:
                raw = np.mean(np.vstack(preds), axis=0) if preds else np.full(len(test), 0.5)
                raw_pre_blend = raw.copy()
                mkt = (
                    test["market_home_prob"].to_numpy(dtype=float)
                    if "market_home_prob" in test.columns
                    else np.full(len(test), np.nan)
                )
                if cfg.target_mode == "logit_stack":
                    # Calibrate the model first (expanding isotonic on prior
                    # chunks), THEN logit-stack with the market using a weight
                    # fit only on pooled prior chunks. No post-stack
                    # recalibration — restacking calibrated probs preserves the
                    # market's sharpness.
                    model_cal = calibrator.predict(raw) if calibrator is not None else raw
                    if outputs:
                        pooled_prior = pd.concat(outputs, ignore_index=True)
                        w = fit_logit_stack_w(
                            pooled_prior["model_cal"].to_numpy(dtype=float),
                            pooled_prior["market_home_prob"].to_numpy(dtype=float)
                            if "market_home_prob" in pooled_prior.columns
                            else np.full(len(pooled_prior), np.nan),
                            pooled_prior["home_win"].to_numpy(dtype=float),
                        )
                    else:
                        w = LOGIT_STACK_DEFAULT_W
                    calibrated = apply_logit_stack(model_cal, mkt, w)
                else:
                    if cfg.market_blend_w > 0:
                        w = float(cfg.market_blend_w)
                        blended = raw.copy()
                        mask = np.isfinite(mkt)
                        blended[mask] = (1.0 - w) * raw[mask] + w * mkt[mask]
                        raw = blended
                    calibrated = calibrator.predict(raw) if calibrator is not None else raw
                keep = [
                    c
                    for c in (
                        "season",
                        "date",
                        "event_id",
                        "game_pk",
                        adapter.home_col,
                        adapter.away_col,
                        "home",
                        "away",
                        "home_win",
                        "margin",
                        "market_home_prob",
                        "home_ml",
                        "away_ml",
                        "home_spread",
                        "home_close_ml",
                        "away_close_ml",
                        "home_open_ml",
                        "away_open_ml",
                        "home_open_spread",
                        "away_open_spread",
                        "home_close_spread",
                        "away_close_spread",
                        "open_total",
                        "close_total",
                        "has_steam",
                        "ml_steam_pp",
                        "label",
                        "result",
                    )
                    if c in test.columns
                ]
                chunk = test[keep].copy()
                if adapter.home_col in chunk.columns and "home" not in chunk.columns:
                    chunk["home"] = chunk[adapter.home_col]
                    chunk["away"] = chunk[adapter.away_col]
                # logit_stack keeps the pre-blend model prob in model_raw (the
                # expanding isotonic is fit on it) and its calibrated version
                # in model_cal (the market-stack weight is fit on it).
                chunk["model_raw"] = raw_pre_blend if cfg.target_mode == "logit_stack" else raw
                if cfg.target_mode == "logit_stack":
                    chunk["model_cal"] = model_cal
                # 0.98 caps tax true near-certain favorites (CFB buy games hit
                # 99%+); isotonic never extrapolates past observed rates, so
                # wider caps are safe for every league.
                chunk["model_prob"] = np.clip(calibrated, 0.005, 0.995)
                if margin_preds:
                    chunk["model_margin"] = np.mean(np.vstack(margin_preds), axis=0)
                outputs.append(chunk)
                pooled = pd.concat(outputs, ignore_index=True)
                if len(pooled) >= adapter.min_calibration_pool:
                    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
                    calibrator.fit(pooled["model_raw"], pooled["home_win"])

        season_chunks = [c for c in outputs if int(c.season.iloc[0]) == eval_season]
        if season_chunks:
            season_rows = pd.concat(season_chunks, ignore_index=True)
            if multi:
                proba = season_rows[["model_prob_h", "model_prob_d", "model_prob_a"]].to_numpy()
                y = season_rows["label"].to_numpy()
                ll = log_loss_multi(proba, y)
            else:
                y = season_rows.home_win.values.astype(float)
                ll = log_loss_binary(season_rows.model_prob.values, y)
            print(f"[{adapter.league}/{cfg.name}] eval {eval_season}: n={len(season_rows)} LL={ll:.5f}", flush=True)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def summarize(adapter: LeagueAdapter, oos: pd.DataFrame) -> dict[str, Any]:
    if oos.empty:
        return {"n": 0, "beat_baseline": False, "oos_model_logloss": None}
    if adapter.multiclass and {"model_prob_h", "model_prob_d", "model_prob_a", "label"}.issubset(oos.columns):
        proba = oos[["model_prob_h", "model_prob_d", "model_prob_a"]].to_numpy(dtype=float)
        y = oos["label"].to_numpy(dtype=int)
        ll = log_loss_multi(proba, y)
        pred = proba.argmax(axis=1)
        acc = float(np.mean(pred == y))
        summary: dict[str, Any] = {
            "oos_model_logloss": round(ll, 5),
            "oos_model_acc": round(acc, 4),
            "n": int(len(oos)),
            "baseline_ll": adapter.baseline_ll,
            "delta_vs_baseline": round(adapter.baseline_ll - ll, 5),
            "beat_baseline": bool(ll < adapter.baseline_ll - 0.001),
            "multiclass": True,
        }
        return summary

    y = oos["home_win"].to_numpy(dtype=float)
    p = oos["model_prob"].to_numpy(dtype=float)
    ll = log_loss_binary(p, y)
    summary = {
        "oos_model_logloss": round(ll, 5),
        "oos_model_brier": round(brier_binary(p, y), 5),
        "oos_model_acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "n": int(len(oos)),
        "baseline_ll": adapter.baseline_ll,
        "delta_vs_baseline": round(adapter.baseline_ll - ll, 5),
        "beat_baseline": bool(ll < adapter.baseline_ll - 0.001),
        "multiclass": False,
    }
    m = oos.dropna(subset=["market_home_prob"]) if "market_home_prob" in oos.columns else oos.iloc[0:0]
    if len(m) >= 50:
        ym = m["home_win"].to_numpy(dtype=float)
        summary["oos_market_logloss"] = round(log_loss_binary(m["market_home_prob"].values, ym), 5)
        summary["oos_model_logloss_odds_subset"] = round(log_loss_binary(m["model_prob"].values, ym), 5)
        summary["gap_vs_market"] = round(
            summary["oos_model_logloss_odds_subset"] - summary["oos_market_logloss"], 5
        )
    if "model_margin" in oos.columns and "margin" in oos.columns:
        mm = oos.dropna(subset=["model_margin", "margin"])
        if len(mm) >= 50:
            mae = float(np.mean(np.abs(mm["model_margin"] - mm["margin"])))
            summary["oos_margin_mae"] = round(mae, 4)
            if adapter.baseline_margin_mae is not None:
                summary["baseline_margin_mae"] = adapter.baseline_margin_mae
                summary["beat_margin_baseline"] = bool(mae < adapter.baseline_margin_mae - 0.01)
    return summary


def hubacek_round_key(league: str, oos: pd.DataFrame) -> pd.Series:
    """Round grouping for betting simulation: ISO week for weekly football
    leagues, calendar date elsewhere (paper: one league round)."""
    dates = pd.to_datetime(oos["date"], errors="coerce")
    if league in ("nfl", "cfb"):
        iso = dates.dt.isocalendar()
        return iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return dates.dt.date.astype(str)


def run_config(
    adapter: LeagueAdapter,
    cfg: HybridConfig,
    frame: pd.DataFrame,
    end_season: int,
) -> dict[str, Any]:
    t0 = time.time()
    oos = walk_forward(adapter, frame, cfg, end_season)
    summary = summarize(adapter, oos)
    if cfg.target_mode == "decorrelated" and not adapter.multiclass and not oos.empty:
        try:
            from web.hubacek_v2.eval import hubacek_report

            oos = oos.copy()
            oos["hub_round"] = hubacek_round_key(adapter.league, oos)
            prices = ("close", "open") if {"home_open_ml", "away_open_ml"}.issubset(oos.columns) and oos["home_open_ml"].notna().any() else ("close",)
            summary["hubacek"] = hubacek_report(oos, round_key="hub_round", prices=prices)
        except Exception as exc:  # noqa: BLE001 — telemetry must not kill a run
            summary["hubacek_error"] = str(exc)[:200]
    summary["config"] = cfg.to_dict()
    summary["league"] = adapter.league
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    adapter.hybrid_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.name)
    if not oos.empty:
        oos.to_csv(adapter.hybrid_dir / f"oos_{safe}.csv", index=False)
    (adapter.hybrid_dir / f"summary_{safe}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "league", "oos_model_logloss", "delta_vs_baseline", "beat_baseline",
        "oos_model_acc", "oos_margin_mae", "gap_vs_market", "elapsed_s",
    ) if k in summary}, indent=1), flush=True)
    return summary


def run_focused_grid(
    league: str,
    *,
    end_season: int | None = None,
    max_configs: int = 10,
) -> dict[str, Any]:
    adapter = get_adapter(league)
    end_season = end_season or adapter.end_season
    print(f"=== {league} hybrid focused grid (end_season={end_season}) ===", flush=True)
    print(f"baseline_ll={adapter.baseline_ll} table={adapter.table_path}", flush=True)
    frame = prepare_frame(adapter, end_season=end_season)
    print(f"rows={len(frame)} cols={len(frame.columns)}", flush=True)
    board: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in focused_search_space()[:max_configs]:
        summary = run_config(adapter, cfg, frame, end_season)
        board.append(summary)
        if best is None or (
            summary.get("oos_model_logloss") is not None
            and summary["oos_model_logloss"] < best.get("oos_model_logloss", 9e9)
        ):
            best = summary
            (adapter.hybrid_dir / "best_config.json").write_text(
                json.dumps(best, indent=2), encoding="utf-8"
            )
    board_sorted = sorted(
        [b for b in board if b.get("oos_model_logloss") is not None],
        key=lambda x: x["oos_model_logloss"],
    )
    final = {
        "league": league,
        "baseline_ll": adapter.baseline_ll,
        "baseline_margin_mae": adapter.baseline_margin_mae,
        "best": best,
        "board": board_sorted,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (adapter.hybrid_dir / "focused_final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final
