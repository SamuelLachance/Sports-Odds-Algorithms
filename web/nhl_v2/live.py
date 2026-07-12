"""Live NHL v2 inference: artifact loading + current-season replay + prediction.

The artifact (data/models/nhl_v2/) contains a feature-engine snapshot at the
end of the previous season. At prediction time we fetch the current season
(NHL Stats API results + goalie logs, MoneyPuck team xG, MoneyPuck shot-level
goalie xG), replay it through the same engine used in training, then score
today's matchups with the persisted XGBoost + logistic ensemble and isotonic
calibrator. Starting goalies come from the NHL scoreboard when available,
falling back to the team's most-used recent starter.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import time
import zipfile
from collections import defaultdict
from datetime import date as date_cls
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from web.basketball_v2_market import apply_market_features, resolve_market_heads
from web.nhl_v2.data import (
    MONEYPUCK_TEAMS_URL,
    MP_METRICS,
    MP_SITUATIONS,
    build_game_index,
    canon_team,
    fetch_goalie_games,
    fetch_schedule_day,
    fetch_team_games,
    get_bytes,
)
from web.nhl_v2.feature_engine import FEATURE_COLUMNS, NhlFeatureEngine
from web.nhl_v2.replay import build_goalie_index, replay_season

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "nhl_v2"
LIVE_CACHE_DIR = PROJECT_ROOT / ".build-cache" / "nhl-v2-live"

STATS_TTL_SECONDS = 3 * 3600  # fresher than 6h Pages cadence
MP_TTL_SECONDS = 24 * 3600

SHOTS_URL = "https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip"

# Moneyline-only market head (no spread margin head for NHL goals models).
MARKET_FEATURES = ("mkt_home_prob", "has_market")

# ESPN lowercase keys -> NHL API abbreviations.
ESPN_TO_NHL: dict[str, str] = {
    "ana": "ANA", "ari": "ARI", "phx": "ARI", "bos": "BOS", "buf": "BUF",
    "car": "CAR", "cbj": "CBJ", "cgy": "CGY", "chi": "CHI", "col": "COL",
    "dal": "DAL", "det": "DET", "edm": "EDM", "fla": "FLA", "la": "LAK",
    "min": "MIN", "mtl": "MTL", "nj": "NJD", "nsh": "NSH", "nyi": "NYI",
    "nyr": "NYR", "ott": "OTT", "phi": "PHI", "pit": "PIT", "sea": "SEA",
    "sj": "SJS", "stl": "STL", "tb": "TBL", "tor": "TOR", "uta": "UTA",
    "utah": "UTA", "van": "VAN", "vgk": "VGK", "wpg": "WPG", "wsh": "WSH",
}


def nhl_season_for_date(target: date_cls) -> int:
    """NHL season start-year for a calendar date (Jul-Dec -> that year)."""
    return target.year if target.month >= 7 else target.year - 1


def infer_nhl_game_type(day_iso: str) -> int:
    """Best-effort regular (2) vs playoff (3) when the scoreboard row is missing.

    NHL playoffs typically run mid-April through June. Early April still has
    regular-season games, so we only flip after the 15th.
    """
    try:
        day = date_cls.fromisoformat(day_iso[:10])
    except ValueError:
        return 2
    if day.month in (5, 6):
        return 3
    if day.month == 4 and day.day >= 15:
        return 3
    return 2


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

        def _optional_json(name: str) -> dict[str, Any] | None:
            path = MODEL_DIR / name
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

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
            "goals_home": _booster("model_goals_home.json"),
            "goals_away": _booster("model_goals_away.json"),
            "clf_market": _booster("model_clf_market.json"),
            "lr_market": _optional_json("model_lr_market.json"),
            "calibrator_market": _optional_json("calibrator_market.json"),
            "snapshots": snapshots,
            "feature_columns": feature_columns,
            "clf_market_features": list(
                metadata.get("clf_market_features") or MARKET_FEATURES
            ),
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


def _fetch_stats_bundle(season: int) -> tuple[dict[str, Any] | None, bool]:
    """Current-season team results + goalie logs (6h TTL).

    Returns ``(bundle, live_inputs_stale)``. On network failure, may soft-return
    a longer-TTL cache and set ``live_inputs_stale=True``.
    """
    path = LIVE_CACHE_DIR / f"stats_{season}.json"
    cached = _read_cache(path, STATS_TTL_SECONDS)
    if cached is not None:
        return cached, False
    try:
        team_rows = fetch_team_games(season)
        goalie_rows = fetch_goalie_games(season)
    except OSError:
        stale = _read_cache(path, 10 * 86400)
        return stale, stale is not None
    bundle = {"team_games": team_rows, "goalies": goalie_rows}
    _write_cache(path, bundle)
    return bundle, False


def _fetch_moneypuck_slices(
    seasons: list[int],
) -> tuple[dict[int, dict[str, dict[str, dict[str, float]]]], bool]:
    """MoneyPuck team metrics per requested season, keyed by gameId (24h TTL).

    Downloads the (large) all-seasons file at most once per call, only when at
    least one requested season slice is stale.
    Returns ``(slices_by_season, live_inputs_stale)``.
    """
    out: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    stale: list[int] = []
    used_stale = False
    for season in seasons:
        cached = _read_cache(LIVE_CACHE_DIR / f"mp_{season}.json", MP_TTL_SECONDS)
        if cached is not None:
            out[season] = cached
        else:
            stale.append(season)
    if not stale:
        return out, False
    try:
        payload = get_bytes(MONEYPUCK_TEAMS_URL)
    except OSError:
        for season in stale:
            fallback = _read_cache(LIVE_CACHE_DIR / f"mp_{season}.json", 30 * 86400)
            out[season] = fallback or {}
            if fallback is not None:
                used_stale = True
            else:
                # Empty after network failure — callers treat empty as fail-closed;
                # still mark stale so soft-cache hits remain distinguishable.
                used_stale = True
        return out, used_stale
    wanted = set(stale)
    slices: dict[int, dict[str, dict[str, dict[str, float]]]] = {s: {} for s in stale}
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8", "ignore")))
    for row in reader:
        try:
            season = int(row.get("season") or 0)
        except ValueError:
            continue
        if season not in wanted:
            continue
        situation = row.get("situation") or ""
        if situation not in MP_SITUATIONS:
            continue
        try:
            game_id = str(int(row["gameId"]))
        except (TypeError, ValueError):
            continue
        team = canon_team(row.get("team") or "")
        if not team:
            continue
        bucket = slices[season].setdefault(game_id, {}).setdefault(team, {})
        for metric in MP_METRICS:
            raw = row.get(metric)
            if raw in (None, ""):
                continue
            try:
                bucket[f"{situation}_{metric}"] = float(raw)
            except ValueError:
                continue
    for season, games in slices.items():
        _write_cache(LIVE_CACHE_DIR / f"mp_{season}.json", games)
        out[season] = games
    return out, False


def _fetch_goalie_xg_slice(season: int) -> tuple[dict[str, list[float]], bool]:
    """Current-season per-goalie per-game xG from MoneyPuck shots (24h TTL).

    Keyed by "gameId:goalieId" -> [shots, xg, goals].
    Returns ``(slice, live_inputs_stale)``.
    """
    path = LIVE_CACHE_DIR / f"goalie_xg_{season}.json"
    cached = _read_cache(path, MP_TTL_SECONDS)
    if cached is not None:
        return cached, False
    try:
        payload = get_bytes(SHOTS_URL.format(season=season))
    except OSError:
        stale = _read_cache(path, 30 * 86400)
        if stale is not None:
            return stale, True
        return {}, False
    # [shots, xg, goals, hd_shots, hd_xg, hd_goals]
    out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hd_threshold = 0.20
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = next(
                (n for n in archive.namelist() if n.lower().endswith(".csv")), None
            )
            if name:
                with archive.open(name) as handle:
                    reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
                    for row in reader:
                        try:
                            goalie_id = int(float(row.get("goalieIdForShot") or 0))
                        except ValueError:
                            continue
                        if goalie_id <= 0:
                            continue
                        if row.get("shotOnEmptyNet") in ("1", "1.0"):
                            continue
                        raw_gid = str(row.get("game_id") or "").strip()
                        try:
                            value = int(float(raw_gid))
                        except ValueError:
                            continue
                        game_id = value if value > 2000000000 else int(
                            f"{season}0{value:05d}"
                        )
                        try:
                            xg = float(row.get("xGoal") or 0.0)
                        except ValueError:
                            xg = 0.0
                        goal = 1.0 if row.get("goal") in ("1", "1.0") else 0.0
                        bucket = out[f"{game_id}:{goalie_id}"]
                        bucket[0] += 1.0
                        bucket[1] += xg
                        bucket[2] += goal
                        if xg >= hd_threshold:
                            bucket[3] += 1.0
                            bucket[4] += xg
                            bucket[5] += goal
    except (zipfile.BadZipFile, OSError):
        return {}, False
    result = dict(out)
    _write_cache(path, result)
    return result, False


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


def _predict_goals(art: dict[str, Any], features: dict[str, float]) -> tuple[float, float] | None:
    if art["goals_home"] is None or art["goals_away"] is None:
        return None
    from xgboost import DMatrix

    cols = art["feature_columns"]
    vector = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=float)
    matrix = DMatrix(vector, feature_names=list(cols))
    home = float(art["goals_home"].predict(matrix)[0])
    away = float(art["goals_away"].predict(matrix)[0])
    return round(home, 2), round(away, 2)


def _goalie_xg_map(
    raw: dict[str, list[float]] | None,
) -> dict[tuple[int, int], tuple[float, float, float, float, float, float]]:
    """Parse MoneyPuck goalie-xG cache keys ``gameId:goalieId`` -> float tuple."""
    out: dict[tuple[int, int], tuple[float, float, float, float, float, float]] = {}
    for key, vals in (raw or {}).items():
        try:
            gid_str, goalie_str = key.split(":")
            shots = float(vals[0]) if len(vals) > 0 else 0.0
            xg = float(vals[1]) if len(vals) > 1 else 0.0
            goals = float(vals[2]) if len(vals) > 2 else 0.0
            hd_shots = float(vals[3]) if len(vals) > 3 else 0.0
            hd_xg = float(vals[4]) if len(vals) > 4 else 0.0
            hd_goals = float(vals[5]) if len(vals) > 5 else 0.0
            out[(int(gid_str), int(goalie_str))] = (
                shots, xg, goals, hd_shots, hd_xg, hd_goals
            )
        except (ValueError, TypeError, IndexError):
            continue
    return out


def _depth_chart(goalie_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """team -> goalies sorted by recent starts (current season)."""
    by_team: dict[str, dict[str, dict[str, Any]]] = {}
    for row in goalie_rows:
        team = str(row.get("teamAbbrev") or "")
        player_id = str(row.get("playerId") or "")
        if not team or not player_id:
            continue
        entry = by_team.setdefault(team, {}).setdefault(
            player_id,
            {"id": player_id, "name": row.get("goalieFullName"), "starts": 0,
             "last_date": ""},
        )
        if int(row.get("gamesStarted") or 0) == 1:
            entry["starts"] += 1
            entry["last_date"] = max(str(entry["last_date"]), str(row.get("gameDate") or ""))
    charts: dict[str, list[dict[str, Any]]] = {}
    for team, goalies in by_team.items():
        charts[team] = sorted(
            goalies.values(), key=lambda g: (-int(g["starts"]), str(g["last_date"])),
        )
    return charts


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

    season = nhl_season_for_date(target)
    loaded = _load_snapshot_state(art, season)
    if loaded is None:
        return None
    snapshot_season, state = loaded

    live_inputs_stale = False
    stats, stats_stale = _fetch_stats_bundle(season)
    live_inputs_stale = live_inputs_stale or stats_stale
    if stats is None:
        return None
    season_has_games = bool(stats.get("team_games"))
    if season_has_games:
        mp_all, mp_stale = _fetch_moneypuck_slices([season])
        live_inputs_stale = live_inputs_stale or mp_stale
        mp_slice = mp_all.get(season) or {}
        if not mp_slice:
            # Empty MoneyPuck would replay Elo without xG/shot features — fail closed
            # (same policy as gap seasons).
            return None
        goalie_xg_raw, xg_stale = _fetch_goalie_xg_slice(season)
        live_inputs_stale = live_inputs_stale or xg_stale
    else:
        mp_slice = {}
        goalie_xg_raw = {}

    mp_games = {int(gid): teams for gid, teams in (mp_slice or {}).items()}
    goalie_xg = _goalie_xg_map(goalie_xg_raw)

    engine = NhlFeatureEngine.from_dict(state)

    # replay any full seasons between the snapshot and the target season
    for gap_season in range(snapshot_season + 1, season):
        gap_stats, gap_stale = _fetch_stats_bundle(gap_season)
        live_inputs_stale = live_inputs_stale or gap_stale
        if gap_stats is None:
            # Missing intermediate season would skip Elo/form state — fail closed.
            return None
        gap_mp_all, gap_mp_stale = _fetch_moneypuck_slices([gap_season])
        live_inputs_stale = live_inputs_stale or gap_mp_stale
        gap_mp = gap_mp_all.get(gap_season) or {}
        gap_games = list(build_game_index(gap_stats["team_games"]).values())
        if not gap_games:
            # Empty bundle (soft-cache / blank API) would skip Elo/form — fail closed.
            return None
        if not gap_mp:
            # Empty MoneyPuck would replay Elo without xG/shot features — fail closed.
            return None
        gap_xg_raw, gap_xg_stale = _fetch_goalie_xg_slice(gap_season)
        live_inputs_stale = live_inputs_stale or gap_xg_stale
        gap_goalies = build_goalie_index(gap_stats["goalies"])
        # Match training / current-season replay: attach MoneyPuck goalie xG.
        replay_season(
            engine, gap_season, gap_games,
            {int(gid): teams for gid, teams in gap_mp.items()},
            gap_goalies,
            goalie_xg=_goalie_xg_map(gap_xg_raw),
        )

    games = list(build_game_index(stats["team_games"]).values())
    goalie_index = build_goalie_index(stats["goalies"])
    replay_season(
        engine, season, games, mp_games, goalie_index,
        goalie_xg=goalie_xg, stop_before_date=day_iso,
    )

    try:
        schedule = fetch_schedule_day(day_iso)
    except OSError:
        schedule = []
        live_inputs_stale = True
    todays: dict[tuple[str, str], dict[str, Any]] = {}
    for game in schedule:
        if game.get("home") and game.get("away"):
            todays[(game["home"], game["away"])] = game

    # Depth chart from the current season; before any games have been played
    # (offseason/opening night) fall back to last season's starter usage.
    goalie_rows: list[dict[str, Any]] = list(stats["goalies"] or [])
    if not goalie_rows:
        prev, prev_stale = _fetch_stats_bundle(season - 1)
        live_inputs_stale = live_inputs_stale or prev_stale
        if prev:
            goalie_rows = list(prev.get("goalies") or [])

    goalie_names: dict[str, str] = {}
    for row in goalie_rows:
        player_id = str(row.get("playerId") or "")
        name = row.get("goalieFullName")
        if player_id and name:
            goalie_names[player_id] = str(name)

    return {
        "engine": engine,
        "artifacts": art,
        "todays_games": todays,
        "depth_chart": _depth_chart(goalie_rows),
        "goalie_names": goalie_names,
        "season": season,
        "day_iso": day_iso,
        "live_inputs_stale": live_inputs_stale,
    }


def _resolve_starter(
    context: dict[str, Any],
    team: str,
    scoreboard_starter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Starter playerId/name/confirmed from scoreboard, else depth chart."""
    if scoreboard_starter and scoreboard_starter.get("id"):
        player_id = str(scoreboard_starter["id"])
        name = scoreboard_starter.get("name")
        if isinstance(name, dict):
            name = name.get("default")
        return {
            "id": player_id,
            "name": name or context["goalie_names"].get(player_id),
            "confirmed": bool(scoreboard_starter.get("confirmed")),
            "source": "scoreboard",
        }
    chart = context["depth_chart"].get(team) or []
    if chart:
        top = chart[0]
        return {
            "id": str(top["id"]),
            "name": top.get("name"),
            "confirmed": False,
            "source": "depth-chart",
        }
    return None


def predict_matchup_v2(
    day_iso: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_moneyline: int | float | None = None,
    away_moneyline: int | float | None = None,
) -> dict[str, Any] | None:
    """Predict today's NHL matchup by ESPN abbreviations.

    When live moneylines are available and market-aware artifacts exist, scores
    the market classifier head. Soft-fails to the pure head when odds or
    market boosters are missing.
    """
    context = get_live_context(day_iso)
    if context is None:
        return None
    home = ESPN_TO_NHL.get(home_abbr.lower(), home_abbr.upper())
    away = ESPN_TO_NHL.get(away_abbr.lower(), away_abbr.upper())

    game = context["todays_games"].get((home, away))
    if game is None:
        game = {
            "gameId": 0,
            "date": day_iso,
            "game_type": infer_nhl_game_type(day_iso),
            "home": home,
            "away": away,
        }

    # starting goalies: scoreboard first, then most-used current-season starter
    try:
        from web.hockey_goalie import _find_matchup_starters

        starters_meta = _find_matchup_starters(
            home_abbr, away_abbr, date_cls.fromisoformat(day_iso)
        )
    except Exception:  # noqa: BLE001 - scoreboard probe is best-effort
        starters_meta = {}

    home_starter = _resolve_starter(context, home, starters_meta.get("home_starter"))
    away_starter = _resolve_starter(context, away, starters_meta.get("away_starter"))
    if home_starter:
        game["home_goalie_id"] = home_starter["id"]
    if away_starter:
        game["away_goalie_id"] = away_starter["id"]

    engine: NhlFeatureEngine = context["engine"]
    art = context["artifacts"]
    features = engine.features_for_game(game)
    has_market, _has_spread = apply_market_features(
        features,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
    )
    (
        use_market_clf,
        _use_market_margin,
        model_variant,
        clf_cols,
        _margin_cols,
    ) = resolve_market_heads(art, has_market=has_market, has_spread=False)

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
    goals = _predict_goals(art, features)

    home_team = engine.team(home)
    away_team = engine.team(away)

    payload: dict[str, Any] = {
        "model_version": "v2",
        "model_variant": model_variant,
        "home_win_probability": round(prob_home * 100.0, 2),
        "features_used": len(clf_cols),
        "home_games": int(home_team.games_played),
        "away_games": int(away_team.games_played),
        "home_elo": round(home_team.elo, 1),
        "away_elo": round(away_team.elo, 1),
        "home_xgf_pg": round(home_team.xgf_fast, 2),
        "home_xga_pg": round(home_team.xga_fast, 2),
        "away_xgf_pg": round(away_team.xgf_fast, 2),
        "away_xga_pg": round(away_team.xga_fast, 2),
        "home_points_pct": round(home_team.points_pct(), 3),
        "away_points_pct": round(away_team.points_pct(), 3),
        "home_rest_days": features["home_rest_days"],
        "away_rest_days": features["away_rest_days"],
        "home_b2b": bool(features["home_b2b"]),
        "away_b2b": bool(features["away_b2b"]),
        "home_goalie": (home_starter or {}).get("name"),
        "away_goalie": (away_starter or {}).get("name"),
        "home_goalie_confirmed": bool((home_starter or {}).get("confirmed")),
        "away_goalie_confirmed": bool((away_starter or {}).get("confirmed")),
        "home_goalie_gsax100": round(float(features["home_goalie_gsax"]), 2),
        "away_goalie_gsax100": round(float(features["away_goalie_gsax"]), 2),
        "has_market": has_market,
    }
    if goals is not None:
        payload["predicted_home_goals"] = goals[0]
        payload["predicted_away_goals"] = goals[1]
        payload["predicted_margin"] = round(goals[0] - goals[1], 2)
    if context.get("live_inputs_stale"):
        payload["live_inputs_stale"] = True
    return payload


def clear_live_caches() -> None:
    get_live_context.cache_clear()
    _load_artifacts.cache_clear()
