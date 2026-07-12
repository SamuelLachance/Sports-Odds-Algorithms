"""Soccer v2 live inference: artifacts + season replay + matchup prediction.

Loads the pooled XGB/LR ensemble trained by scripts/train_soccer_model.py,
restores the newest engine snapshot older than the target season, replays any
newer completed seasons plus the current season's football-data CSVs (top
flight AND second division so promoted-team state stays fresh), then builds
features for a live fixture.

Two heads:
  pure          — market-free probabilities (display fallback, diagnostics)
  market-aware  — devigged live 1X2 odds as extra features (display + picks)

Pick probabilities are Hubáček-decorrelated from the live market.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from web.soccer_v2.data import (
    DIVISION_COUNTRY,
    LEAGUE_DIVISIONS,
    SECOND_DIVISIONS,
    devig_decimal,
    fetch_csv_text,
    parse_season_csv,
)
from web.soccer_v2.feature_engine import FEATURE_COLUMNS, SoccerFeatureEngine
from web.soccer_v2.replay import merge_country_rows, replay_country

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "soccer_v2"
LIVE_CACHE_DIR = PROJECT_ROOT / ".build-cache" / "soccer-v2-live"
CURRENT_CSV_TTL_SECONDS = 3 * 3600  # fresher than 6h Pages cadence

MARKET_FEATURES = ("mkt_open_home", "mkt_open_draw", "mkt_open_away", "has_market")

# Hubáček push-away weight for pick probabilities (gap gets ×(1+w)).
PICK_DECORRELATION_WEIGHT = 0.15

MODEL_VERSION = "soccer_v2.2026.07"


_NAME_STOPWORDS = {
    "fc",
    "cf",
    "afc",
    "ac",
    "as",
    "ss",
    "ssc",
    "sc",
    "cd",
    "rcd",
    "rc",
    "sd",
    "ud",
    "us",
    "vfb",
    "vfl",
    "fsv",
    "tsg",
    "sv",
    "og",
    "olympique",
}


def _norm_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    tokens = [
        token
        for token in text.split()
        if token not in _NAME_STOPWORDS and not token.isdigit()
    ]
    return " ".join(tokens)


# extra ESPN-name → football-data-name hints where fuzzy matching fails
_NAME_HINTS: dict[str, str] = {
    "wolverhampton wanderers": "Wolves",
    "brighton hove albion": "Brighton",
    "west ham united": "West Ham",
    "nottingham forest": "Nott'm Forest",
    "borussia dortmund": "Dortmund",
    "bayern munich": "Bayern Munich",
    "borussia monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "bayer leverkusen": "Leverkusen",
    "koln": "FC Koln",
    "cologne": "FC Koln",
    "st pauli": "St Pauli",
    "hamburg": "Hamburg",
    "hamburger": "Hamburg",
    "atletico madrid": "Ath Madrid",
    "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "real betis": "Betis",
    "celta vigo": "Celta",
    "rayo vallecano": "Vallecano",
    "real oviedo": "Oviedo",
    "deportivo alaves": "Alaves",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "milan": "Milan",
    "hellas verona": "Verona",
    "paris saint germain": "Paris SG",
    "marseille": "Marseille",
    "saint etienne": "St Etienne",
    "paris": "Paris FC",
}


def artifacts_available() -> bool:
    required = (
        MODEL_DIR / "model_clf.json",
        MODEL_DIR / "model_lr.json",
        MODEL_DIR / "model_clf_market.json",
        MODEL_DIR / "model_lr_market.json",
        MODEL_DIR / "calibrator.json",
        MODEL_DIR / "calibrator_market.json",
        MODEL_DIR / "metadata.json",
    )
    return all(path.is_file() for path in required) and bool(
        list(MODEL_DIR.glob("state_*.json.gz"))
    )


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

        snapshots: dict[int, Path] = {}
        for path in MODEL_DIR.glob("state_*.json.gz"):
            try:
                snapshots[int(path.stem.split("_")[1].split(".")[0])] = path
            except (IndexError, ValueError):
                continue

        clf = _booster("model_clf.json")
        clf_market = _booster("model_clf_market.json")
        if clf is None or clf_market is None or not snapshots:
            return None

        return {
            "clf": clf,
            "clf_market": clf_market,
            "lr": json.loads((MODEL_DIR / "model_lr.json").read_text(encoding="utf-8")),
            "lr_market": json.loads(
                (MODEL_DIR / "model_lr_market.json").read_text(encoding="utf-8")
            ),
            "calibrator": json.loads(
                (MODEL_DIR / "calibrator.json").read_text(encoding="utf-8")
            ),
            "calibrator_market": json.loads(
                (MODEL_DIR / "calibrator_market.json").read_text(encoding="utf-8")
            ),
            "metadata": json.loads(
                (MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
            ),
            "snapshots": snapshots,
            "goals_home": _booster("model_goals_home.json"),
            "goals_away": _booster("model_goals_away.json"),
        }
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def soccer_season_for_date(date_iso: str) -> int:
    year = int(date_iso[:4])
    month = int(date_iso[5:7])
    return year if month >= 7 else year - 1


def _fetch_current_csv(division: str, season: int) -> tuple[list[dict[str, Any]], bool]:
    """Current-season rows with a TTL cache (football-data updates ~daily).

    Returns ``(rows, live_inputs_stale)``. On network failure, prefer an existing
    on-disk CSV over wiping the cache to empty.
    """
    LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = LIVE_CACHE_DIR / f"{division}_{season}.csv"
    if (
        cache.is_file()
        and time.time() - cache.stat().st_mtime < CURRENT_CSV_TTL_SECONDS
    ):
        text = cache.read_text(encoding="utf-8", errors="replace")
        rows = parse_season_csv(text, division, season) if text.strip() else []
        return rows, False
    try:
        text = fetch_csv_text(division, season, cache_dir=LIVE_CACHE_DIR, force=True)
    except OSError:
        text = None
    if text is None:
        # Soft-fail: keep prior CSV rather than negative-caching an empty wipe.
        if cache.is_file():
            prior = cache.read_text(encoding="utf-8", errors="replace")
            if prior.strip():
                return parse_season_csv(prior, division, season), True
        cache.write_text("", encoding="utf-8")  # negative-cache missing files
        return [], False
    return parse_season_csv(text, division, season), False


@lru_cache(maxsize=8)
def get_live_context(cutoff_iso: str) -> dict[str, Any] | None:
    """Engine pools replayed through matches strictly before ``cutoff_iso``."""
    artifacts = _load_artifacts()
    if not artifacts:
        return None
    target_season = soccer_season_for_date(cutoff_iso)

    snapshots: dict[int, Path] = artifacts["snapshots"]
    usable = [season for season in snapshots if season < target_season]
    if not usable:
        return None
    snap_season = max(usable)
    try:
        with gzip.open(snapshots[snap_season], "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        pools = {
            country: SoccerFeatureEngine.from_dict(data)
            for country, data in (payload.get("pools") or {}).items()
        }
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile, EOFError, KeyError, TypeError, ValueError):
        return None

    live_inputs_stale = False
    # replay every season after the snapshot up to the cutoff (handles both
    # in-season updates and multi-season snapshot gaps)
    for season in range(snap_season + 1, target_season + 1):
        rows_by_country: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for top_division in LEAGUE_DIVISIONS.values():
            for division in (top_division, SECOND_DIVISIONS.get(top_division)):
                if not division:
                    continue
                rows, csv_stale = _fetch_current_csv(division, season)
                live_inputs_stale = live_inputs_stale or csv_stale
                if not rows:
                    continue
                country = DIVISION_COUNTRY[division]
                rows_by_country.setdefault(country, {})[division] = rows
        for country, divisions in rows_by_country.items():
            engine = pools.get(country)
            if engine is None:
                continue
            merged = merge_country_rows(divisions)
            replay_country(engine, merged, stop_before_date=cutoff_iso)

    return {
        "pools": pools,
        "snapshot_season": snap_season,
        "cutoff": cutoff_iso,
        "live_inputs_stale": live_inputs_stale,
    }


def _team_lookup(engine: SoccerFeatureEngine) -> dict[str, str]:
    return {_norm_name(name): name for name in engine.teams}


def resolve_team(
    engine: SoccerFeatureEngine,
    league: str,
    abbr: str | None,
    name: str | None,
) -> str | None:
    """Map an ESPN abbr/display-name to a football-data team key."""
    from web.football_data_uk import TEAM_TO_ABBR

    mapping = TEAM_TO_ABBR.get(league, {})
    if abbr:
        abbr_lower = abbr.lower()
        for fd_name, fd_abbr in mapping.items():
            if fd_abbr == abbr_lower and fd_name in engine.teams:
                return fd_name

    if not name:
        return None
    normalized = _norm_name(name)
    hint = _NAME_HINTS.get(normalized)
    if hint and hint in engine.teams:
        return hint

    lookup = _team_lookup(engine)
    if normalized in lookup:
        return lookup[normalized]
    # containment either way ("Hamburg SV" vs "Hamburg", "Betis" vs "Real Betis")
    candidates = [
        fd_name
        for norm, fd_name in lookup.items()
        if norm and (norm in normalized or normalized in norm)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _predict_probs(
    artifacts: dict[str, Any],
    features: dict[str, float],
    market: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(pure, market_aware) calibrated probability triples."""
    import xgboost as xgb

    base = [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]
    x_pure = np.asarray([base], dtype=float)

    market_extra = (
        [market[0], market[1], market[2], 1.0] if market else [0.0, 0.0, 0.0, 0.0]
    )
    x_market = np.asarray([base + market_extra], dtype=float)

    def _ensemble(clf, lr, calibrator, x) -> tuple[float, float, float]:
        xgb_probs = clf.predict(xgb.DMatrix(x))[0]
        lr_probs = _lr_probs(lr, x)[0]
        weight = float(lr.get("xgb_weight") or 0.55)
        raw = weight * xgb_probs + (1.0 - weight) * lr_probs
        raw = raw / raw.sum()
        cal = np.array(
            [
                np.interp(raw[cls], calibrator["x"][cls], calibrator["y"][cls])
                for cls in range(3)
            ]
        )
        cal = np.clip(cal, 1e-4, None)
        cal = cal / cal.sum()
        return float(cal[0]), float(cal[1]), float(cal[2])

    pure = _ensemble(artifacts["clf"], artifacts["lr"], artifacts["calibrator"], x_pure)
    market_aware = _ensemble(
        artifacts["clf_market"],
        artifacts["lr_market"],
        artifacts["calibrator_market"],
        x_market,
    )
    return pure, market_aware


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


def _predict_goals(
    artifacts: dict[str, Any], features: dict[str, float]
) -> tuple[float | None, float | None]:
    import xgboost as xgb

    if artifacts["goals_home"] is None or artifacts["goals_away"] is None:
        return None, None
    x = np.asarray(
        [[float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]], dtype=float
    )
    dm = xgb.DMatrix(x)
    home = float(artifacts["goals_home"].predict(dm)[0])
    away = float(artifacts["goals_away"].predict(dm)[0])
    return max(home, 0.1), max(away, 0.1)


def _decorrelate(
    probs: tuple[float, float, float],
    market: tuple[float, float, float] | None,
    weight: float = PICK_DECORRELATION_WEIGHT,
) -> tuple[float, float, float]:
    if market is None:
        return probs
    pushed = [max(p + weight * (p - m), 0.005) for p, m in zip(probs, market)]
    total = sum(pushed)
    return pushed[0] / total, pushed[1] / total, pushed[2] / total


def american_to_decimal(ml: int | None) -> float | None:
    if ml is None:
        return None
    ml = int(ml)
    if ml > 0:
        return 1.0 + ml / 100.0
    if ml < 0:
        return 1.0 + 100.0 / abs(ml)
    return None


def predict_matchup_v2(
    league: str,
    cutoff_iso: str,
    *,
    home_abbr: str | None,
    away_abbr: str | None,
    home_name: str | None = None,
    away_name: str | None = None,
    home_ml: int | None = None,
    draw_ml: int | None = None,
    away_ml: int | None = None,
    match_date: str | None = None,
) -> dict[str, Any] | None:
    league = (league or "").lower()
    if league not in LEAGUE_DIVISIONS:
        return None
    artifacts = _load_artifacts()
    if not artifacts:
        return None
    context = get_live_context(cutoff_iso)
    if not context:
        return None

    division = LEAGUE_DIVISIONS[league]
    engine: SoccerFeatureEngine | None = context["pools"].get(
        DIVISION_COUNTRY[division]
    )
    if engine is None:
        return None

    home_key = resolve_team(engine, league, home_abbr, home_name)
    away_key = resolve_team(engine, league, away_abbr, away_name)
    if not home_key or not away_key or home_key == away_key:
        return None

    home_state = engine.teams.get(home_key)
    away_state = engine.teams.get(away_key)
    if home_state is None or away_state is None:
        return None
    if home_state.games_played < 5 or away_state.games_played < 5:
        return None

    row = {
        "date": match_date or cutoff_iso,
        "home": home_key,
        "away": away_key,
    }
    features = engine.features_for_match(row, league)

    market = None
    if home_ml is not None and draw_ml is not None and away_ml is not None:
        market = devig_decimal(
            american_to_decimal(home_ml),
            american_to_decimal(draw_ml),
            american_to_decimal(away_ml),
        )

    pure, market_aware = _predict_probs(artifacts, features, market)
    display = market_aware if market is not None else pure
    pick = _decorrelate(display, market)

    exp_home, exp_away = _predict_goals(artifacts, features)

    payload: dict[str, Any] = {
        "algorithm": "SoccerGradientBoost v2",
        "model_version": MODEL_VERSION,
        "source": "soccer-v2-xgb-ensemble",
        "league": league,
        "home_team": home_key,
        "away_team": away_key,
        "home_win_probability": round(display[0] * 100.0, 2),
        "draw_probability": round(display[1] * 100.0, 2),
        "away_win_probability": round(display[2] * 100.0, 2),
        "pick_home_win_probability": round(pick[0] * 100.0, 2),
        "pick_draw_probability": round(pick[1] * 100.0, 2),
        "pick_away_win_probability": round(pick[2] * 100.0, 2),
        "raw_home_win_probability": round(pure[0] * 100.0, 2),
        "raw_draw_probability": round(pure[1] * 100.0, 2),
        "raw_away_win_probability": round(pure[2] * 100.0, 2),
        "market_aware_home_win_probability": round(market_aware[0] * 100.0, 2),
        "market_aware_draw_probability": round(market_aware[1] * 100.0, 2),
        "market_aware_away_win_probability": round(market_aware[2] * 100.0, 2),
        "expected_home_goals": round(exp_home, 2) if exp_home is not None else None,
        "expected_away_goals": round(exp_away, 2) if exp_away is not None else None,
        "elo_home": round(home_state.elo, 1),
        "elo_away": round(away_state.elo, 1),
        "home_games": home_state.games_played,
        "away_games": away_state.games_played,
        "home_rest_days": features.get("home_rest_days"),
        "away_rest_days": features.get("away_rest_days"),
        "home_promoted": bool(features.get("home_promoted")),
        "away_promoted": bool(features.get("away_promoted")),
        "market_decorrelated": market is not None,
        "market_calibrated": market is not None,
        "decorrelation_weight": PICK_DECORRELATION_WEIGHT,
        "snapshot_season": context["snapshot_season"],
    }
    if context.get("live_inputs_stale"):
        payload["live_inputs_stale"] = True
    return payload


def clear_live_caches() -> None:
    get_live_context.cache_clear()
    _load_artifacts.cache_clear()
