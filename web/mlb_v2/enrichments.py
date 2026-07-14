"""Load MLB supplemental enrichments for training + live inference."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUP = PROJECT_ROOT / "data" / "supplemental"


@lru_cache(maxsize=1)
def _weather_map() -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    openmeteo = SUP / "mlb-weather" / "game_weather.csv"
    mlbapi = SUP / "mlb-weather" / "mlb_api_weather.csv"
    if openmeteo.is_file():
        for row in pd.read_csv(openmeteo).to_dict(orient="records"):
            if float(row.get("has_weather") or 0) < 0.5:
                continue
            try:
                key = (str(row["date"])[:10], int(row["venue_id"]))
            except (TypeError, ValueError, KeyError):
                continue
            out[key] = {
                "temp_c": float(row.get("temp_c") or 22.0),
                "humidity": float(row.get("humidity") or 50.0),
                "precip_mm": float(row.get("precip_mm") or 0.0),
                "wind_kph": float(row.get("wind_kph") or 0.0),
                "has_weather": 1.0,
                "is_roof": float(row.get("is_roof") or 0.0),
            }
    if mlbapi.is_file():
        for row in pd.read_csv(mlbapi).to_dict(orient="records"):
            if float(row.get("has_weather") or 0) < 0.5:
                continue
            try:
                key = (str(row["date"])[:10], int(row["venue_id"]))
            except (TypeError, ValueError, KeyError):
                continue
            if key in out:
                if float(row.get("is_roof") or 0) >= 0.5:
                    out[key]["is_roof"] = 1.0
                continue
            out[key] = {
                "temp_c": float(row.get("temp_c") or 22.0),
                "humidity": 50.0,
                "precip_mm": 0.0,
                "wind_kph": float(row.get("wind_kph") or 0.0),
                "has_weather": 1.0,
                "is_roof": float(row.get("is_roof") or 0.0),
            }
    return out


@lru_cache(maxsize=1)
def _ump_by_pk() -> dict[int, int]:
    path = SUP / "mlb-umpires" / "game_umpires.csv"
    if not path.is_file():
        return {}
    out: dict[int, int] = {}
    for row in pd.read_csv(path).to_dict(orient="records"):
        if float(row.get("has_ump") or 0) < 0.5:
            continue
        try:
            out[int(row["game_pk"])] = int(row["hp_ump_id"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


@lru_cache(maxsize=1)
def _il_map() -> dict[tuple[str, int], dict[str, float]]:
    path = SUP / "mlb-injuries" / "team_il_daily.csv"
    if not path.is_file():
        return {}
    out: dict[tuple[str, int], dict[str, float]] = {}
    for row in pd.read_csv(path).to_dict(orient="records"):
        try:
            key = (str(row["date"])[:10], int(row["team_id"]))
        except (TypeError, ValueError, KeyError):
            continue
        out[key] = {
            "il_count": float(row.get("il_count") or 0.0),
            "il_pitchers": float(row.get("il_pitchers") or 0.0),
            "has_il": float(row.get("has_il") or 1.0),
        }
    return out


@lru_cache(maxsize=1)
def _sp_xera() -> dict[tuple[int, int], float]:
    path = SUP / "mlb-statcast" / "pitcher_prev_xera.csv"
    if not path.is_file():
        return {}
    out: dict[tuple[int, int], float] = {}
    for row in pd.read_csv(path).to_dict(orient="records"):
        try:
            season = int(row.get("for_season") or (int(row["season"]) + 1))
            out[(season, int(row["player_id"]))] = float(row["xera"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


@lru_cache(maxsize=1)
def _team_xwoba() -> dict[tuple[int, int], float]:
    path = SUP / "mlb-statcast" / "team_prev_xwoba.csv"
    if not path.is_file():
        return {}
    out: dict[tuple[int, int], float] = {}
    for row in pd.read_csv(path).to_dict(orient="records"):
        try:
            season = int(row.get("for_season") or (int(row["season"]) + 1))
            out[(season, int(row["team_id"]))] = float(row["xwoba"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


@lru_cache(maxsize=1)
def _team_xwoba_rolling() -> list[tuple[str, int, int, float]]:
    """Sorted (asof_date, for_season, team_id, xwoba) for PIT lookup."""
    path = SUP / "mlb-statcast" / "team_xwoba_rolling.csv"
    if not path.is_file():
        return []
    rows: list[tuple[str, int, int, float]] = []
    for row in pd.read_csv(path).to_dict(orient="records"):
        try:
            asof = str(row.get("asof_date") or "")[:10]
            tid = int(row["team_id"])
            xw = float(row["xwoba"])
            for_season = int(row.get("for_season") or (int(row["season"]) + 1))
        except (TypeError, ValueError, KeyError):
            continue
        if asof:
            rows.append((asof, for_season, tid, xw))
    rows.sort()
    return rows


def _rolling_xwoba(team_id: int, day: str, season: int) -> float | None:
    """Prior-season (or earlier) snapshot only — never same-season final leak."""
    best: float | None = None
    for asof, for_season, tid, xw in _team_xwoba_rolling():
        if tid != int(team_id):
            continue
        # Prefer for_season match (season-end prior for this season)
        if for_season == int(season) and asof < day:
            best = xw
        elif for_season < int(season) and asof < day:
            best = xw
    return best


def attach_game_enrichments(game: dict[str, Any], *, season: int | None = None) -> None:
    """Mutate ``game`` with weather / ump / IL / Statcast priors when available."""
    day = str(game.get("date") or "")[:10]
    if season is None:
        try:
            season = int(day[:4])
        except (TypeError, ValueError):
            season = date.today().year

    venue = game.get("venue_id")
    if venue is not None:
        wx = _weather_map().get((day, int(venue)))
        if wx:
            game.update(wx)

    pk = game.get("gamePk")
    if pk is not None:
        ump_id = _ump_by_pk().get(int(pk))
        if ump_id:
            game["hp_ump_id"] = ump_id

    home_id = game.get("home_id")
    away_id = game.get("away_id")
    if home_id is not None and away_id is not None:
        hil = _il_map().get((day, int(home_id)))
        ail = _il_map().get((day, int(away_id)))
        if hil and ail:
            game["home_il_count"] = hil["il_count"]
            game["away_il_count"] = ail["il_count"]
            game["home_il_pitchers"] = hil["il_pitchers"]
            game["away_il_pitchers"] = ail["il_pitchers"]
            game["has_il"] = 1.0
        # If today's IL row missing, try yesterday (build lag).
        elif hil is None or ail is None:
            try:
                yday = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
            except ValueError:
                yday = None
            if yday:
                hil = hil or _il_map().get((yday, int(home_id)))
                ail = ail or _il_map().get((yday, int(away_id)))
                if hil and ail:
                    game["home_il_count"] = hil["il_count"]
                    game["away_il_count"] = ail["il_count"]
                    game["home_il_pitchers"] = hil["il_pitchers"]
                    game["away_il_pitchers"] = ail["il_pitchers"]
                    game["has_il"] = 1.0

    home_pp = game.get("home_pp_id")
    away_pp = game.get("away_pp_id")
    if home_pp:
        x = _sp_xera().get((int(season), int(home_pp)))
        if x is not None:
            game["home_sp_prev_xera"] = x
    if away_pp:
        x = _sp_xera().get((int(season), int(away_pp)))
        if x is not None:
            game["away_sp_prev_xera"] = x

    if home_id is not None:
        hx = _team_xwoba().get((int(season), int(home_id)))
        if hx is not None:
            game["home_prev_xwoba"] = hx
        hx_roll = _rolling_xwoba(int(home_id), day, int(season))
        if hx_roll is not None:
            game["home_roll_xwoba"] = hx_roll
    if away_id is not None:
        ax = _team_xwoba().get((int(season), int(away_id)))
        if ax is not None:
            game["away_prev_xwoba"] = ax
        ax_roll = _rolling_xwoba(int(away_id), day, int(season))
        if ax_roll is not None:
            game["away_roll_xwoba"] = ax_roll
    if game.get("home_roll_xwoba") is not None or game.get("away_roll_xwoba") is not None:
        game["has_statcast_rolling"] = 1.0
        game["roll_xwoba_diff"] = float(
            (game.get("home_roll_xwoba") or 0.320) - (game.get("away_roll_xwoba") or 0.320)
        )
