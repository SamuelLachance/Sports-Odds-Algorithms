"""Live MLB v2 inference: artifact loading + current-season replay + prediction.

The artifact (data/models/mlb_v2/) contains a feature-engine snapshot at the
end of a prior season. At prediction time we load the newest snapshot strictly
older than the target season, gap-replay any intermediate seasons, fetch the
current season from the MLB Stats API (daily cached), replay through today,
then score today's matchups with the persisted XGBoost + logistic ensemble and
isotonic calibrator.
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

from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID
from web.mlb_v2.feature_engine import FEATURE_COLUMNS, MlbFeatureEngine
from web.mlb_v2.replay import is_final_game, replay_season
from web.mlb_v2.statsapi_data import (
    fetch_pitcher_logs,
    fetch_season_games,
    fetch_team_ids,
    fetch_team_logs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "mlb_v2"
LIVE_CACHE_DIR = PROJECT_ROOT / ".build-cache" / "mlb-v2-live"

LIVE_CACHE_TTL_SECONDS = 3 * 3600  # fresher than 6h Pages cadence
LIVE_CACHE_STALE_SECONDS = 10 * 86400  # soft-serve window on network failure

# Stats API codedGameState values treated as finished for doubleheader pick.
_FINAL_STATUSES = frozenset({"F", "O", "C", "D"})


def artifacts_available() -> bool:
    required = ("model_clf.json", "model_lr.json", "calibrator.json", "metadata.json")
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

        clf = _booster("model_clf.json")
        if clf is None or not snapshots:
            return None

        feature_columns = metadata.get("feature_columns") or list(FEATURE_COLUMNS)
        return {
            "clf": clf,
            "lr": json.loads((MODEL_DIR / "model_lr.json").read_text(encoding="utf-8")),
            "calibrator": json.loads(
                (MODEL_DIR / "calibrator.json").read_text(encoding="utf-8")
            ),
            "metadata": metadata,
            "runs_home": _booster("model_runs_home.json"),
            "runs_away": _booster("model_runs_away.json"),
            "snapshots": snapshots,
            "feature_columns": feature_columns,
        }
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def _load_snapshot_state(art: dict[str, Any], target_season: int) -> tuple[int, Any] | None:
    """Newest snapshot strictly older than the target season."""
    eligible = [season for season in art["snapshots"] if season < target_season]
    if not eligible:
        return None
    season = max(eligible)
    try:
        with gzip.open(art["snapshots"][season], "rt", encoding="utf-8") as handle:
            return season, json.load(handle)
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile, EOFError, TypeError, ValueError):
        return None


def _cache_path(name: str, season: int, day_iso: str) -> Path:
    return LIVE_CACHE_DIR / f"{season}" / f"{day_iso}_{name}.json"


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


def _fetch_current_season_bundle(
    season: int, day_iso: str
) -> tuple[dict[str, Any] | None, bool]:
    """Season-to-date games, pitcher logs and team logs (cached per day).

    Returns ``(bundle, live_inputs_stale)``. On network failure, may soft-return
    a longer-TTL cache and set ``live_inputs_stale=True``.
    """
    bundle_path = _cache_path("bundle", season, day_iso)
    cached = _read_cache(bundle_path, LIVE_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached, False
    try:
        games = fetch_season_games(season, include_names=True)
        pitcher_ids = sorted(
            {
                int(pid)
                for game in games
                for pid in (game.get("home_pp_id"), game.get("away_pp_id"))
                if pid
            }
        )
        pitchers = fetch_pitcher_logs(season, pitcher_ids)
        team_ids = fetch_team_ids(season)
        team_hitting = fetch_team_logs(season, team_ids, "hitting")
        team_pitching = fetch_team_logs(season, team_ids, "pitching")
    except OSError:
        stale = _read_cache(bundle_path, LIVE_CACHE_STALE_SECONDS)
        return stale, stale is not None
    bundle = {
        "games": games,
        "pitchers": pitchers,
        "team_hitting": team_hitting,
        "team_pitching": team_pitching,
    }
    _write_cache(bundle_path, bundle)
    return bundle, False


def _prefer_todays_game(
    existing: dict[str, Any] | None, candidate: dict[str, Any]
) -> dict[str, Any]:
    """For doubleheaders, prefer the not-yet-final game (live starters)."""
    if existing is None:
        return candidate
    existing_final = str(existing.get("status") or "") in _FINAL_STATUSES
    candidate_final = str(candidate.get("status") or "") in _FINAL_STATUSES
    if existing_final and not candidate_final:
        return candidate
    if (not existing_final) and candidate_final:
        return existing
    # Same finality: prefer higher game_number (game 2 of DH) when both pending,
    # else keep the earlier entry for stable training-style first-game semantics.
    if not existing_final and not candidate_final:
        if int(candidate.get("game_number") or 1) > int(existing.get("game_number") or 1):
            return candidate
    return existing


def _predict_probability(art: dict[str, Any], features: dict[str, float]) -> float:
    from xgboost import DMatrix

    cols = art["feature_columns"]
    vector = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    xgb_p = float(art["clf"].predict(DMatrix(vector, feature_names=list(cols)))[0])

    lr = art["lr"]
    mean = np.asarray(lr["mean"], dtype=float)
    scale = np.asarray(lr["scale"], dtype=float)
    coef = np.asarray(lr["coef"], dtype=float)
    z = float(np.dot((vector[0] - mean) / np.where(scale == 0, 1.0, scale), coef)) + float(
        lr["intercept"]
    )
    lr_p = 1.0 / (1.0 + np.exp(-z))

    weight = float(lr.get("xgb_weight", 0.55))
    raw = weight * xgb_p + (1.0 - weight) * lr_p

    cal = art["calibrator"]
    calibrated = float(np.interp(raw, cal["x"], cal["y"]))
    return float(min(max(calibrated, 0.02), 0.98))


def _predict_runs(art: dict[str, Any], features: dict[str, float]) -> tuple[float, float] | None:
    if art["runs_home"] is None or art["runs_away"] is None:
        return None
    from xgboost import DMatrix

    cols = art["feature_columns"]
    vector = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    matrix = DMatrix(vector, feature_names=list(cols))
    home = float(art["runs_home"].predict(matrix)[0])
    away = float(art["runs_away"].predict(matrix)[0])
    return round(home, 2), round(away, 2)


@lru_cache(maxsize=8)
def get_live_context(day_iso: str) -> dict[str, Any] | None:
    """Engine state replayed to `day_iso` morning + today's schedule."""
    art = _load_artifacts()
    if art is None:
        return None
    try:
        target = date_cls.fromisoformat(day_iso)
    except ValueError:
        return None

    season = target.year
    loaded = _load_snapshot_state(art, season)
    if loaded is None:
        return None
    snapshot_season, state = loaded

    engine = MlbFeatureEngine.from_dict(state)
    live_inputs_stale = False

    # Replay any full seasons between the snapshot and the target season.
    for gap_season in range(snapshot_season + 1, season):
        gap_bundle, gap_stale = _fetch_current_season_bundle(gap_season, day_iso)
        live_inputs_stale = live_inputs_stale or gap_stale
        if gap_bundle is None:
            # Missing intermediate season would skip Elo/form/SP state — fail closed.
            return None
        replay_season(
            engine,
            gap_season,
            gap_bundle["games"],
            gap_bundle["pitchers"],
            gap_bundle["team_hitting"],
            gap_bundle["team_pitching"],
        )

    bundle, bundle_stale = _fetch_current_season_bundle(season, day_iso)
    live_inputs_stale = live_inputs_stale or bundle_stale
    if bundle is None:
        return None

    replay_season(
        engine,
        season,
        bundle["games"],
        bundle["pitchers"],
        bundle["team_hitting"],
        bundle["team_pitching"],
        stop_before_date=day_iso,
    )

    todays_games: dict[tuple[int, int], dict[str, Any]] = {}
    todays_finals: list[dict[str, Any]] = []
    for game in bundle["games"]:
        if str(game.get("date")) != day_iso:
            continue
        key = (int(game["home_id"]), int(game["away_id"]))
        todays_games[key] = _prefer_todays_game(todays_games.get(key), game)
        if is_final_game(game):
            todays_finals.append(game)

    pitcher_names: dict[int, str] = {}
    for game in bundle["games"]:
        for side in ("home", "away"):
            pid = game.get(f"{side}_pp_id")
            name = game.get(f"{side}_pp_name")
            if pid and name:
                pitcher_names[int(pid)] = str(name)

    return {
        "engine": engine,
        "artifacts": art,
        "todays_games": todays_games,
        "todays_finals": todays_finals,
        "pitcher_names": pitcher_names,
        "season": season,
        "day_iso": day_iso,
        "live_inputs_stale": live_inputs_stale,
    }


def predict_matchup_v2(
    day_iso: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Predict today's matchup by ESPN abbreviations. Returns None when unavailable."""
    context = get_live_context(day_iso)
    if context is None:
        return None
    home_id = ESPN_TO_MLB_TEAM_ID.get(home_abbr.lower())
    away_id = ESPN_TO_MLB_TEAM_ID.get(away_abbr.lower())
    if not home_id or not away_id:
        return None

    game = context["todays_games"].get((home_id, away_id))
    if game is None:
        # scheduled game missing (e.g. postponed) -> synthetic neutral row
        game = {
            "gamePk": None,
            "date": day_iso,
            "status": "S",
            "day_night": "night",
            "double_header": "N",
            "game_number": 1,
            "venue_id": None,
            "home_id": home_id,
            "away_id": away_id,
            "home_pp_id": None,
            "away_pp_id": None,
        }

    engine: MlbFeatureEngine = context["engine"]
    art = context["artifacts"]

    # Afternoon DH: morning-of replay stops before day_iso, so apply earlier
    # same-day finals for this matchup on a cloned engine before scoring game 2+.
    selected_gn = int(game.get("game_number") or 1)
    earlier_finals = [
        prior
        for prior in context.get("todays_finals") or []
        if int(prior.get("home_id") or 0) == home_id
        and int(prior.get("away_id") or 0) == away_id
        and int(prior.get("game_number") or 1) < selected_gn
    ]
    if earlier_finals:
        engine = MlbFeatureEngine.from_dict(engine.to_dict())
        for prior in sorted(earlier_finals, key=lambda g: int(g.get("game_number") or 1)):
            engine.update_after_game(prior)

    features = engine.features_for_game(game)
    prob_home = _predict_probability(art, features)
    runs = _predict_runs(art, features)

    names = context["pitcher_names"]
    home_pp_id = game.get("home_pp_id")
    away_pp_id = game.get("away_pp_id")

    home_team = engine.team(home_id)
    away_team = engine.team(away_id)

    payload: dict[str, Any] = {
        "model_version": "v2",
        "home_win_probability": round(prob_home * 100.0, 2),
        "features_used": len(art["feature_columns"]),
        "home_games": int(home_team.season_wins + home_team.season_losses),
        "away_games": int(away_team.season_wins + away_team.season_losses),
        "home_elo": round(home_team.elo, 1),
        "away_elo": round(away_team.elo, 1),
        "home_probable_pitcher": names.get(int(home_pp_id)) if home_pp_id else None,
        "away_probable_pitcher": names.get(int(away_pp_id)) if away_pp_id else None,
        "home_sp_fip": round(float(features["home_sp_fip_blend"]), 2),
        "away_sp_fip": round(float(features["away_sp_fip_blend"]), 2),
        "park_factor": round(float(features["park_factor"]), 3),
    }
    if runs is not None:
        payload["predicted_home_runs"] = runs[0]
        payload["predicted_away_runs"] = runs[1]
        payload["predicted_margin"] = round(runs[0] - runs[1], 2)
    if context.get("live_inputs_stale"):
        payload["live_inputs_stale"] = True
    return payload


def clear_live_caches() -> None:
    get_live_context.cache_clear()
    _load_artifacts.cache_clear()
