"""Live CFB v2 inference: artifact loading + current-season replay + prediction.

Artifacts in data/models/cfb_v2/ include a feature-engine snapshot at the end
of a prior season. At prediction time we fetch completed current-season games
from ESPN via season_games, replay them through the training engine, then
score today's matchups with the persisted XGBoost + logistic ensemble
(isotonic-calibrated) and the margin/score regressors.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import date as date_cls
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from web.basketball_v2_market import apply_market_features, resolve_market_heads
from web.cfb_v2.feature_engine import FEATURE_COLUMNS, CfbFeatureEngine, cfb_season_of
from web.cfb_v2.replay import events_to_results, replay_season

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "cfb_v2"
LIVE_CACHE_DIR = PROJECT_ROOT / ".build-cache" / "cfb-v2-live"

EVENTS_TTL_SECONDS = 3 * 3600  # fresher than 6h Pages cadence

CLF_MARKET_FEATURES = ("mkt_home_prob", "has_market")
MARGIN_MARKET_FEATURES = ("mkt_home_spread", "has_spread")
MARKET_FEATURES = CLF_MARKET_FEATURES + MARGIN_MARKET_FEATURES


def artifacts_available() -> bool:
    required = (
        "model_clf.json",
        "model_lr.json",
        "model_margin.json",
        "calibrator.json",
        "metadata.json",
    )
    if not all((MODEL_DIR / name).is_file() for name in required):
        return False
    return any(MODEL_DIR.glob("state_*.json.gz"))


@lru_cache(maxsize=1)
def _load_artifacts() -> dict[str, Any] | None:
    if not artifacts_available():
        return None
    try:
        from xgboost import Booster  # noqa: F401
    except ImportError:
        return None

    try:
        def _booster(name: str):
            path = MODEL_DIR / name
            if not path.is_file():
                return None
            try:
                booster = Booster()
                booster.load_model(str(path))
                return booster
            except Exception:  # noqa: BLE001 - corrupt booster → unavailable
                return None

        metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        snapshots: dict[int, Path] = {}
        for path in MODEL_DIR.glob("state_*.json.gz"):
            try:
                snapshots[int(path.stem.split("_")[1].split(".")[0])] = path
            except (IndexError, ValueError):
                continue

        def _optional_json(name: str) -> dict[str, Any] | None:
            path = MODEL_DIR / name
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        clf = _booster("model_clf.json")
        if clf is None or not snapshots:
            return None
        return {
            "clf": clf,
            "lr": json.loads((MODEL_DIR / "model_lr.json").read_text(encoding="utf-8")),
            "calibrator": json.loads((MODEL_DIR / "calibrator.json").read_text(encoding="utf-8")),
            "margin": _booster("model_margin.json"),
            "score_home": _booster("model_score_home.json"),
            "score_away": _booster("model_score_away.json"),
            "clf_market": _booster("model_clf_market.json"),
            "lr_market": _optional_json("model_lr_market.json"),
            "calibrator_market": _optional_json("calibrator_market.json"),
            "margin_market": _booster("model_margin_market.json"),
            "metadata": metadata,
            "snapshots": snapshots,
            "feature_columns": metadata.get("feature_columns") or list(FEATURE_COLUMNS),
            "clf_market_features": list(
                metadata.get("clf_market_features") or CLF_MARKET_FEATURES
            ),
            "margin_market_features": list(
                metadata.get("margin_market_features") or MARGIN_MARKET_FEATURES
            ),
        }
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def _load_snapshot_state(art: dict[str, Any], target_season: int) -> tuple[int, Any] | None:
    eligible = [season for season in art["snapshots"] if season < target_season]
    if not eligible:
        return None
    season = max(eligible)
    try:
        with gzip.open(art["snapshots"][season], "rt", encoding="utf-8") as handle:
            return season, json.load(handle)
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile, EOFError, TypeError, ValueError):
        return None


def _read_cache(path: Path, ttl: int) -> Any | None:
    if not path.is_file():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _fetch_completed_season_games(season: int, *, stop_before: str) -> list[dict[str, Any]]:
    """Completed CFB games for ``season`` before ``stop_before`` via ESPN scoreboard."""
    cache_path = LIVE_CACHE_DIR / f"events_{season}_{stop_before}.json"
    cached = _read_cache(cache_path, EVENTS_TTL_SECONDS)
    if cached is not None:
        try:
            return events_to_results(cached, season)
        except (TypeError, ValueError, KeyError):
            pass

    try:
        from web.season_games import _event_date_iso, _load_league_game_map
    except Exception:  # noqa: BLE001 - live fetch must not break slate builds
        return []

    try:
        merged = _load_league_game_map("cfb", stop_before, for_backtest=True)
    except Exception:  # noqa: BLE001
        return []

    filtered: list[dict[str, Any]] = []
    for event_id, (event_date, game_tuple) in merged.items():
        day_iso = _event_date_iso(event_date)[:10]
        if not day_iso or day_iso >= stop_before:
            continue
        try:
            day = date_cls.fromisoformat(day_iso)
        except ValueError:
            continue
        if cfb_season_of(day) != season:
            continue
        home, away, _hn, _an, hs, aw = game_tuple
        if hs is None or aw is None:
            continue
        filtered.append(
            {
                "event_id": event_id,
                "date": day_iso,
                "home": str(home).lower(),
                "away": str(away).lower(),
                "home_score": int(hs),
                "away_score": int(aw),
                "completed": True,
            }
        )

    if filtered:
        _write_cache(cache_path, filtered)
    return events_to_results(filtered, season)


@lru_cache(maxsize=8)
def get_live_context(day_iso: str) -> dict[str, Any] | None:
    art = _load_artifacts()
    if art is None:
        return None
    try:
        day = date_cls.fromisoformat(day_iso[:10])
    except ValueError:
        return None
    season = cfb_season_of(day)
    snap = _load_snapshot_state(art, season)
    if snap is None:
        # No prior-season snapshot → cold-start would invent Elo/form. Fail closed.
        return None
    snapshot_season, payload = snap
    engine = CfbFeatureEngine.from_dict(payload)

    # Replay any full seasons between the snapshot and the target season.
    for gap_season in range(snapshot_season + 1, season):
        gap_games = _fetch_completed_season_games(gap_season, stop_before=day_iso[:10])
        if not gap_games:
            # Missing intermediate season would skip Elo/form state — fail closed.
            return None
        replay_season(engine, gap_games)

    games = _fetch_completed_season_games(season, stop_before=day_iso[:10])
    replay_season(engine, games, stop_before_date=day_iso[:10])
    return {
        "engine": engine,
        "artifacts": art,
        "season": season,
        "day_iso": day_iso[:10],
    }


def _predict_probability(
    art: dict[str, Any],
    features: dict[str, float],
    *,
    cols: list[str] | None = None,
    clf=None,
    lr: dict[str, Any] | None = None,
    calibrator: dict[str, Any] | None = None,
) -> float:
    from xgboost import DMatrix

    cols = list(cols or art["feature_columns"])
    clf = clf if clf is not None else art["clf"]
    lr = lr if lr is not None else art["lr"]
    calibrator = calibrator if calibrator is not None else art["calibrator"]
    vector = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    xgb_p = float(clf.predict(DMatrix(vector, feature_names=list(cols)))[0])

    mean = np.asarray(lr["mean"], dtype=float)
    scale = np.asarray(lr["scale"], dtype=float)
    coef = np.asarray(lr["coef"], dtype=float)
    z = float(np.dot((vector[0] - mean) / np.where(scale == 0, 1.0, scale), coef)) + float(
        lr["intercept"]
    )
    lr_p = 1.0 / (1.0 + np.exp(-z))
    weight = float(lr.get("xgb_weight", 0.55))
    raw = weight * xgb_p + (1.0 - weight) * lr_p
    calibrated = float(np.interp(raw, calibrator["x"], calibrator["y"]))
    return float(min(max(calibrated, 0.02), 0.98))


def _predict_regressor(
    booster,
    art: dict[str, Any],
    features: dict[str, float],
    *,
    cols: list[str] | None = None,
) -> float | None:
    if booster is None:
        return None
    from xgboost import DMatrix

    cols = list(cols or art["feature_columns"])
    vector = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    return float(booster.predict(DMatrix(vector, feature_names=list(cols)))[0])


def predict_matchup_v2(
    day_iso: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_moneyline: int | float | None = None,
    away_moneyline: int | float | None = None,
    home_spread: float | None = None,
) -> dict[str, Any] | None:
    """Predict a CFB matchup by ESPN abbreviations for a given slate date.

    Soft-fails to the pure head when odds or market boosters are missing.
    """
    context = get_live_context(day_iso[:10])
    if context is None:
        return None
    home = str(home_abbr or "").lower().strip()
    away = str(away_abbr or "").lower().strip()
    engine: CfbFeatureEngine = context["engine"]
    if home not in engine.teams or away not in engine.teams:
        return None
    art = context["artifacts"]

    game: dict[str, Any] = {
        "date": day_iso[:10],
        "season": context["season"],
        "home": home,
        "away": away,
    }
    features = engine.features_for_game(game)
    has_market, has_spread = apply_market_features(
        features,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
        home_spread=home_spread,
    )
    (
        use_market_clf,
        use_market_margin,
        model_variant,
        clf_cols,
        margin_cols,
    ) = resolve_market_heads(art, has_market=has_market, has_spread=has_spread)

    if use_market_clf:
        prob_home = _predict_probability(
            art,
            features,
            cols=clf_cols,
            clf=art["clf_market"],
            lr=art["lr_market"],
            calibrator=art["calibrator_market"],
        )
    else:
        prob_home = _predict_probability(art, features, cols=clf_cols)

    if use_market_margin:
        margin = _predict_regressor(
            art["margin_market"], art, features, cols=margin_cols
        )
    else:
        margin = _predict_regressor(art["margin"], art, features, cols=margin_cols)

    # Prefer shipped hybrid CatBoost+market-blend win prob when available.
    try:
        from web.hybrid_v2.live import try_hybrid_binary

        mkt_p = float(features["mkt_home_prob"]) if has_market else None
        hybrid = try_hybrid_binary(
            "cfb",
            features,
            home_id=home,
            away_id=away,
            market_home_prob=mkt_p,
        )
        if hybrid is not None:
            prob_home = float(hybrid["home_win_prob"])
            model_variant = "hybrid"
            if hybrid.get("predicted_margin") is not None:
                margin = float(hybrid["predicted_margin"])
    except Exception:  # noqa: BLE001 — hybrid is best-effort overlay
        pass

    pure_cols = list(art["feature_columns"])
    score_home = _predict_regressor(art["score_home"], art, features, cols=pure_cols)
    score_away = _predict_regressor(art["score_away"], art, features, cols=pure_cols)

    home_team = engine.team(home)
    away_team = engine.team(away)
    payload: dict[str, Any] = {
        "model_version": "v2",
        "algorithm": "CFBGradientBoost v2",
        "model_variant": model_variant,
        "home_win_probability": round(prob_home * 100.0, 2),
        "features_used": len(clf_cols),
        "home_games": int(home_team.games_played),
        "away_games": int(away_team.games_played),
        "home_elo": round(home_team.elo, 1),
        "away_elo": round(away_team.elo, 1),
        "home_rest_days": features["home_rest_days"],
        "away_rest_days": features["away_rest_days"],
        "home_win_pct": round(home_team.win_pct(), 3),
        "away_win_pct": round(away_team.win_pct(), 3),
        "neutral_site": bool(features["neutral_site"]),
        "is_conference_game": bool(features["is_conference_game"]),
        "has_market": has_market,
        "has_spread": has_spread,
    }
    if margin is not None:
        payload["predicted_margin"] = round(margin, 2)
    if score_home is not None and score_away is not None:
        total = score_home + score_away
        if margin is not None:
            payload["predicted_home_score"] = round((total + margin) / 2.0, 1)
            payload["predicted_away_score"] = round((total - margin) / 2.0, 1)
        else:
            payload["predicted_home_score"] = round(score_home, 1)
            payload["predicted_away_score"] = round(score_away, 1)
        payload["predicted_total"] = round(total, 1)
    return payload


def clear_live_caches() -> None:
    get_live_context.cache_clear()
    _load_artifacts.cache_clear()
