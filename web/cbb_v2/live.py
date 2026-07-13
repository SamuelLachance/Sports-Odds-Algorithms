"""Live CBB v2 inference: artifact loading + current-season replay + prediction."""

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
from web.cbb_v2.data import (
    build_abbr_id_map,
    canon_abbr,
    cbb_season_for_date,
    fetch_box_score,
    fetch_season_events,
    force_postseason_neutral_site,
    load_season_boxes,
    team_key,
)
from web.cbb_v2.feature_engine import FEATURE_COLUMNS, CbbFeatureEngine
from web.cbb_v2.replay import merge_season_games, replay_season

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "cbb_v2"
LIVE_CACHE_DIR = PROJECT_ROOT / ".build-cache" / "cbb-v2-live"

EVENTS_TTL_SECONDS = 3 * 3600  # fresher than 6h Pages cadence
PAST_SEASON_TTL_SECONDS = 30 * 86400
BOX_WORKERS = 6

# ML + spread market heads (same shape as NBA/WNBA when closing odds exist).
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


def _fetch_events_cached(season: int, *, current: bool) -> tuple[list[dict[str, Any]], bool]:
    """Returns ``(events, live_inputs_stale)``; soft-serves long-TTL cache on fetch failure."""
    path = LIVE_CACHE_DIR / f"events_{season}.json"
    ttl = EVENTS_TTL_SECONDS if current else PAST_SEASON_TTL_SECONDS
    cached = _read_cache(path, ttl)
    if isinstance(cached, dict):
        events = cached.get("events")
        if isinstance(events, list):
            return events, False
    history = PROJECT_ROOT / ".build-cache" / "cbb-history" / str(season) / "events.json"
    if history.is_file() and not current:
        try:
            payload = json.loads(history.read_text(encoding="utf-8"))
            events = payload.get("events", []) if isinstance(payload, dict) else []
            if isinstance(events, list):
                _write_cache(path, {"events": events})
                return events, False
        except (json.JSONDecodeError, OSError):
            pass
    try:
        events = fetch_season_events(season, use_cache=True)
    except OSError:
        soft = _read_cache(path, 90 * 86400)
        if isinstance(soft, dict):
            return list(soft.get("events") or []), True
        # Hard fail with no soft cache: mark stale so callers stay honest.
        return [], True
    if not isinstance(events, list):
        return [], True
    _write_cache(path, {"events": events})
    return events, False


def _fetch_boxes_cached(
    season: int, events: list[dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """Incremental per-event box cache: only missing final events are fetched.

    Failed fetches are not persisted as ``null`` (that would poison the year-long
    TTL and block retries). Returns ``(boxes, live_inputs_stale)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    path = LIVE_CACHE_DIR / f"boxes_{season}.json"
    boxes: dict[str, Any] = {}
    cached = _read_cache(path, 365 * 86400)
    if isinstance(cached, dict) and cached:
        boxes = {k: v for k, v in cached.items() if isinstance(v, dict)}
    else:
        seeded = load_season_boxes(season)
        if seeded:
            boxes = dict(seeded)
    todo = [
        str(event["event_id"])
        for event in events
        if event.get("completed") and event.get("event_id") and str(event["event_id"]) not in boxes
    ]
    if not todo:
        if boxes and not path.is_file():
            _write_cache(path, boxes)
        return boxes, False

    stale = False

    def one(event_id: str) -> tuple[str, Any]:
        try:
            return event_id, fetch_box_score(event_id)
        except OSError:
            return event_id, None

    with ThreadPoolExecutor(max_workers=BOX_WORKERS) as pool:
        for event_id, box in pool.map(one, todo):
            if isinstance(box, dict):
                boxes[event_id] = box
            else:
                stale = True
    _write_cache(path, boxes)
    return boxes, stale


def _live_season_games(
    season: int, *, current: bool
) -> tuple[list[dict[str, Any]], bool]:
    events, stale = _fetch_events_cached(season, current=current)
    if not events:
        return [], stale
    boxes, boxes_stale = _fetch_boxes_cached(season, events)
    return merge_season_games([], events, boxes, season=season), stale or boxes_stale


@lru_cache(maxsize=8)
def get_live_context(day_iso: str) -> dict[str, Any] | None:
    art = _load_artifacts()
    if art is None:
        return None
    try:
        target = date_cls.fromisoformat(day_iso)
    except ValueError:
        return None

    season = cbb_season_for_date(target)
    loaded = _load_snapshot_state(art, season)
    if loaded is None:
        return None
    snapshot_season, state = loaded
    engine = CbbFeatureEngine.from_dict(state)
    live_inputs_stale = False

    for gap_season in range(snapshot_season + 1, season):
        gap_games, gap_stale = _live_season_games(gap_season, current=False)
        live_inputs_stale = live_inputs_stale or gap_stale
        if not gap_games:
            # Missing intermediate season would skip Elo/form state — fail closed.
            return None
        replay_season(engine, gap_games)

    events, events_stale = _fetch_events_cached(season, current=True)
    live_inputs_stale = live_inputs_stale or events_stale
    boxes, boxes_stale = _fetch_boxes_cached(season, events) if events else ({}, False)
    live_inputs_stale = live_inputs_stale or boxes_stale
    games = merge_season_games([], events, boxes, season=season) if events else []
    replay_season(engine, games, stop_before_date=day_iso)

    abbr_map = build_abbr_id_map(events)
    for abbr, eid in abbr_map.items():
        engine.register_abbr(abbr, eid)

    todays: dict[tuple[str, str], dict[str, Any]] = {}
    from web.season_games import _event_date_iso

    for event in events:
        raw_date = str(event.get("date") or "")
        # Prefer already-normalized YYYY-MM-DD; map leftover ISO via Toronto.
        event_day = (
            _event_date_iso(raw_date) if "T" in raw_date else raw_date[:10]
        )
        if event_day != day_iso:
            continue
        home = team_key(event.get("home_id") or "", event.get("home_abbr") or "")
        away = team_key(event.get("away_id") or "", event.get("away_abbr") or "")
        todays[(home, away)] = event
        todays[(canon_abbr(str(event.get("home_abbr") or "")),
                canon_abbr(str(event.get("away_abbr") or "")))] = event

    return {
        "engine": engine,
        "artifacts": art,
        "todays_games": todays,
        "season": season,
        "day_iso": day_iso,
        "abbr_map": abbr_map,
        "live_inputs_stale": live_inputs_stale,
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
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    home_moneyline: int | float | None = None,
    away_moneyline: int | float | None = None,
    home_spread: float | None = None,
) -> dict[str, Any] | None:
    """Predict a CBB matchup by ESPN abbreviations (and optional ids).

    When live moneylines/spreads are available and market-aware artifacts exist,
    scores the market heads. Soft-fails to pure when odds or boosters are missing.
    """
    context = get_live_context(day_iso)
    if context is None:
        return None
    engine: CbbFeatureEngine = context["engine"]
    home = engine.resolve_team_id(home_abbr, home_espn_id)
    away = engine.resolve_team_id(away_abbr, away_espn_id)
    if not home or not away or home not in engine.teams or away not in engine.teams:
        return None
    art = context["artifacts"]

    event = (
        context["todays_games"].get((home, away))
        or context["todays_games"].get(
            (canon_abbr(home_abbr), canon_abbr(away_abbr))
        )
    )
    game: dict[str, Any] = {
        "date": day_iso,
        "season": context["season"],
        "season_type": int((event or {}).get("season_type") or 2),
        "home": home,
        "away": away,
        "home_abbr": canon_abbr(home_abbr),
        "away_abbr": canon_abbr(away_abbr),
        "neutral_site": bool((event or {}).get("neutral_site")),
        "conference_game": bool((event or {}).get("conference_game")),
        "home_conference_id": str((event or {}).get("home_conference_id") or ""),
        "away_conference_id": str((event or {}).get("away_conference_id") or ""),
    }
    # Match training/replay: postseason March/April often lack ESPN neutralSite.
    force_postseason_neutral_site(game)

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
            "cbb",
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
        "algorithm": "CBBGradientBoost v2",
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
        "is_conference": bool(features["is_conference"]),
        "home_win_pct": round(home_team.win_pct(), 3),
        "away_win_pct": round(away_team.win_pct(), 3),
        "pit_safe": True,
        "training_source": "espn_completed_games",
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
    if context.get("live_inputs_stale"):
        payload["live_inputs_stale"] = True
    return payload


def clear_live_caches() -> None:
    get_live_context.cache_clear()
    _load_artifacts.cache_clear()
