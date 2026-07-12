"""Live NBA v2 inference: artifact loading + current-season replay + prediction.

The artifact (data/models/nba_v2/) contains a feature-engine snapshot at the
end of the previous season. At prediction time we fetch the current season
from ESPN (events + team box scores, 3h TTL), replay it through the training
engine, then score today's matchups with the persisted XGBoost + logistic
ensemble (isotonic-calibrated) and the margin/score regressors.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date as date_cls
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from web.basketball_v2_market import (
    apply_market_features,
    resolve_market_heads,
    devig_home_prob as _devig_home_prob,  # noqa: F401 — re-exported for tests
)
from web.nba_v2.data import canon_franchise, fetch_box_score, fetch_season_events
from web.nba_v2.feature_engine import FEATURE_COLUMNS, NbaFeatureEngine
from web.nba_v2.replay import merge_season_games, replay_season

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "nba_v2"
LIVE_CACHE_DIR = PROJECT_ROOT / ".build-cache" / "nba-v2-live"

EVENTS_TTL_SECONDS = 3 * 3600  # fresher than 6h Pages cadence
PAST_SEASON_TTL_SECONDS = 30 * 86400
BOX_WORKERS = 6


def nba_season_for_date(target: date_cls | str) -> int:
    """NBA season = ending calendar year (Oct Y-1 through Jun Y -> season Y)."""
    if isinstance(target, str):
        target = date_cls.fromisoformat(target[:10])
    if target.month >= 10:
        return target.year + 1
    return target.year


def artifacts_available() -> bool:
    required = (
        "model_clf.json", "model_lr.json", "model_margin.json",
        "calibrator.json", "metadata.json",
    )
    if not all((MODEL_DIR / name).is_file() for name in required):
        return False
    return any(MODEL_DIR.glob("state_*.json.gz"))


@lru_cache(maxsize=1)
def _load_artifacts() -> dict[str, Any] | None:
    if not artifacts_available():
        return None
    try:
        from xgboost import Booster, DMatrix  # noqa: F401
    except ImportError:
        return None

    def _booster(name: str):
        path = MODEL_DIR / name
        if not path.is_file():
            return None
        booster = Booster()
        booster.load_model(str(path))
        return booster

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

    return {
        "clf": _booster("model_clf.json"),
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
            metadata.get("clf_market_features") or ("mkt_home_prob", "has_market")
        ),
        "margin_market_features": list(
            metadata.get("margin_market_features") or ("mkt_home_spread", "has_spread")
        ),
    }


def _load_snapshot_state(art: dict[str, Any], target_season: int) -> tuple[int, Any] | None:
    """Newest snapshot strictly older than the target season."""
    eligible = [season for season in art["snapshots"] if season < target_season]
    if not eligible:
        return None
    season = max(eligible)
    with gzip.open(art["snapshots"][season], "rt", encoding="utf-8") as handle:
        return season, json.load(handle)


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


def _fetch_events_cached(season: int, *, current: bool) -> list[dict[str, Any]]:
    path = LIVE_CACHE_DIR / f"events_{season}.json"
    ttl = EVENTS_TTL_SECONDS if current else PAST_SEASON_TTL_SECONDS
    cached = _read_cache(path, ttl)
    if cached is not None:
        return cached.get("events", [])
    try:
        events = fetch_season_events(season)
    except OSError:
        stale = _read_cache(path, 90 * 86400)
        return (stale or {}).get("events", [])
    _write_cache(path, {"events": events})
    return events


def _fetch_boxes_cached(season: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Incremental per-event box cache: only missing final events are fetched."""
    path = LIVE_CACHE_DIR / f"boxes_{season}.json"
    boxes: dict[str, Any] = {}
    cached = _read_cache(path, 365 * 86400)
    if isinstance(cached, dict) and cached:
        boxes = cached
    else:
        history_boxes = (
            PROJECT_ROOT / ".build-cache" / "nba-history" / str(season) / "boxes.json"
        )
        if history_boxes.is_file():
            try:
                seeded = json.loads(history_boxes.read_text(encoding="utf-8"))
                if isinstance(seeded, dict):
                    boxes = seeded
            except (json.JSONDecodeError, OSError):
                pass
    todo = [
        event["event_id"]
        for event in events
        if event.get("completed") and event["event_id"] not in boxes
    ]
    if not todo:
        if boxes and not path.is_file():
            _write_cache(path, boxes)
        return boxes

    def one(event_id: str) -> tuple[str, Any]:
        try:
            return event_id, fetch_box_score(event_id)
        except OSError:
            return event_id, None

    with ThreadPoolExecutor(max_workers=BOX_WORKERS) as pool:
        for event_id, box in pool.map(one, todo):
            boxes[event_id] = box
    _write_cache(path, boxes)
    return boxes


def _history_season(season: int) -> list[dict[str, Any]] | None:
    """Completed-season games from the offline history cache when present."""
    season_dir = PROJECT_ROOT / ".build-cache" / "nba-history" / str(season)
    events_path = season_dir / "events.json"
    if not events_path.is_file():
        return None
    try:
        events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
        results_path = season_dir / "results.json"
        results = (
            json.loads(results_path.read_text(encoding="utf-8")).get("results", [])
            if results_path.is_file()
            else []
        )
        boxes_path = season_dir / "boxes.json"
        boxes = (
            json.loads(boxes_path.read_text(encoding="utf-8"))
            if boxes_path.is_file()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        return None
    return merge_season_games(results, events, boxes, season=season)


def _live_season_games(season: int, *, current: bool) -> list[dict[str, Any]]:
    """Prefer offline history for past seasons; refresh current season via ESPN."""
    if not current:
        cached = _history_season(season)
        if cached is not None:
            return cached
    events = _fetch_events_cached(season, current=current)
    if not events:
        return _history_season(season) or []
    boxes = _fetch_boxes_cached(season, events)
    return merge_season_games([], events, boxes, season=season)


@lru_cache(maxsize=8)
def get_live_context(day_iso: str) -> dict[str, Any] | None:
    """Engine state replayed to `day_iso` morning + today's scheduled games."""
    art = _load_artifacts()
    if art is None:
        return None
    try:
        target = date_cls.fromisoformat(day_iso)
    except ValueError:
        return None

    season = nba_season_for_date(target)
    loaded = _load_snapshot_state(art, season)
    if loaded is None:
        return None
    snapshot_season, state = loaded

    engine = NbaFeatureEngine.from_dict(state)

    # replay any full seasons between the snapshot and the target season
    for gap_season in range(snapshot_season + 1, season):
        gap_games = _live_season_games(gap_season, current=False)
        if gap_games:
            replay_season(engine, gap_games)

    events = _fetch_events_cached(season, current=True)
    games = _live_season_games(season, current=True) if events else []
    replay_season(engine, games, stop_before_date=day_iso)

    todays: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if str(event.get("date") or "")[:10] != day_iso:
            continue
        home = canon_franchise(event.get("home_abbr") or "")
        away = canon_franchise(event.get("away_abbr") or "")
        todays[(home, away)] = event

    return {
        "engine": engine,
        "artifacts": art,
        "todays_games": todays,
        "season": season,
        "day_iso": day_iso,
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

    weight = float(lr.get("xgb_weight", 0.5))
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
    """Predict a NBA matchup by ESPN abbreviations for a given slate date.

    When live moneylines and/or spreads are available and market-aware artifacts
    exist, scores the market heads used by ``bet_policy.json`` (model_margin_mkt).
    Falls back to the pure head when odds are missing.
    """
    context = get_live_context(day_iso)
    if context is None:
        return None
    home = canon_franchise(home_abbr)
    away = canon_franchise(away_abbr)
    engine: NbaFeatureEngine = context["engine"]
    if home not in engine.teams or away not in engine.teams:
        return None
    art = context["artifacts"]

    event = context["todays_games"].get((home, away))
    game: dict[str, Any] = {
        "date": day_iso,
        "season": context["season"],
        "season_type": int((event or {}).get("season_type") or 2),
        "home": home,
        "away": away,
        "neutral_site": bool((event or {}).get("neutral_site")),
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

    pure_cols = list(art["feature_columns"])
    score_home = _predict_regressor(art["score_home"], art, features, cols=pure_cols)
    score_away = _predict_regressor(art["score_away"], art, features, cols=pure_cols)

    home_team = engine.team(home)
    away_team = engine.team(away)

    payload: dict[str, Any] = {
        "model_version": "v2",
        "algorithm": "NBAGradientBoost v2",
        "model_variant": model_variant,
        "home_win_probability": round(prob_home * 100.0, 2),
        "features_used": len(clf_cols),
        "home_games": int(home_team.games_played),
        "away_games": int(away_team.games_played),
        "home_elo": round(home_team.elo, 1),
        "away_elo": round(away_team.elo, 1),
        "home_net_rtg": round(home_team.ortg_fast - home_team.drtg_fast, 2),
        "away_net_rtg": round(away_team.ortg_fast - away_team.drtg_fast, 2),
        "home_pace": round(home_team.pace_ewma, 1),
        "away_pace": round(away_team.pace_ewma, 1),
        "home_rest_days": features["home_rest_days"],
        "away_rest_days": features["away_rest_days"],
        "home_b2b": bool(features["home_b2b"]),
        "away_b2b": bool(features["away_b2b"]),
        "home_win_pct": round(home_team.win_pct(), 3),
        "away_win_pct": round(away_team.win_pct(), 3),
        "home_expansion": bool(features["home_expansion"]),
        "away_expansion": bool(features["away_expansion"]),
        "has_market": has_market,
        "has_spread": has_spread,
    }
    if margin is not None:
        payload["predicted_margin"] = round(margin, 2)
    if score_home is not None and score_away is not None:
        # reconcile display scores with the dedicated margin head
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
