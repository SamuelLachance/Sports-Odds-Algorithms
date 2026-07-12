"""Chronological replay of NFL seasons through the feature engine.

Walks completed games (nflverse CSV and/or ESPN scoreboard events),
emitting leak-free feature rows before folding each result into state.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any, Callable

from web.nfl_v2.feature_engine import (
    NflFeatureEngine,
    infer_neutral_site,
    infer_playoff,
    nfl_season_of,
)


def _espn_local_date(event: dict[str, Any]) -> str:
    """America/Toronto calendar day — same keying as season_games / live paths."""
    from web.season_games import _event_date_iso

    return _event_date_iso(str(event.get("date") or ""))


def events_to_results(events: list[dict[str, Any]], season: int | None = None) -> list[dict[str, Any]]:
    """Authoritative game list from completed ESPN / season_games events."""
    rows: list[dict[str, Any]] = []
    for event in events:
        completed = event.get("completed")
        if completed is False:
            continue
        home_score = event.get("home_score", event.get("home_final"))
        away_score = event.get("away_score", event.get("away_final"))
        if home_score is None or away_score is None:
            continue
        home = str(event.get("home") or event.get("home_abbr") or event.get("home_key") or "").lower()
        away = str(event.get("away") or event.get("away_abbr") or event.get("away_key") or "").lower()
        if not home or not away:
            continue
        raw_date = str(event.get("date") or "")
        day_iso = _espn_local_date(event) if "T" in raw_date else raw_date[:10]
        day = date_cls.fromisoformat(day_iso) if day_iso else None
        game_season = int(season if season is not None else (nfl_season_of(day) if day else 0))
        row = {
            "date": day_iso,
            "season": game_season,
            "home": home,
            "away": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "neutral_site": bool(event.get("neutral_site")) if event.get("neutral_site") is not None else None,
            "event_id": event.get("event_id"),
            "week": event.get("week"),
            "game_type": event.get("game_type"),
        }
        if row["neutral_site"] is None:
            row["neutral_site"] = infer_neutral_site(row)
        rows.append(row)
    rows.sort(key=lambda r: (r["date"], r["home"], r["away"]))
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
        if out != out:  # NaN
            return None
        return out
    except (TypeError, ValueError):
        return None


def nflverse_rows_to_games(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert nflverse_games.csv dict rows into engine game dicts."""
    games: list[dict[str, Any]] = []
    for row in rows:
        day_iso = str(row.get("gameday") or row.get("date") or "")[:10]
        if not day_iso:
            continue
        day = date_cls.fromisoformat(day_iso)
        season = int(row.get("season") or nfl_season_of(day))
        home = str(row.get("home_team") or row.get("home_key") or row.get("home") or "").lower()
        away = str(row.get("away_team") or row.get("away_key") or row.get("away") or "").lower()
        if not home or not away:
            continue
        home_score = row.get("home_score", row.get("home_final"))
        away_score = row.get("away_score", row.get("away_final"))
        if home_score is None or away_score is None:
            continue
        try:
            home_score_i = int(float(home_score))
            away_score_i = int(float(away_score))
        except (TypeError, ValueError):
            continue

        game_type = str(row.get("game_type") or "REG").upper()
        location = str(row.get("location") or "Home")
        spread_line = _optional_float(row.get("spread_line"))
        # nflverse spread_line = expected home margin; betting home_spread = -spread_line
        home_close_spread = -spread_line if spread_line is not None else None
        away_close_spread = spread_line if spread_line is not None else None

        game = {
            "date": day_iso,
            "season": season,
            "home": home,
            "away": away,
            "home_score": home_score_i,
            "away_score": away_score_i,
            "week": row.get("week"),
            "game_type": game_type,
            "location": location,
            "neutral_site": location.strip().lower() != "home",
            "is_playoff": infer_playoff({"game_type": game_type}),
            "div_game": bool(int(float(row["div_game"]))) if row.get("div_game") not in (None, "") else False,
            "roof": row.get("roof"),
            "surface": row.get("surface"),
            "temp": row.get("temp"),
            "wind": row.get("wind"),
            "weekday": row.get("weekday"),
            "home_rest": row.get("home_rest"),
            "away_rest": row.get("away_rest"),
            "home_qb_id": row.get("home_qb_id"),
            "away_qb_id": row.get("away_qb_id"),
            "home_qb_name": row.get("home_qb_name"),
            "away_qb_name": row.get("away_qb_name"),
            "home_coach": row.get("home_coach"),
            "away_coach": row.get("away_coach"),
            "home_close_ml": row.get("home_moneyline", row.get("home_close_ml")),
            "away_close_ml": row.get("away_moneyline", row.get("away_close_ml")),
            "home_close_spread": home_close_spread,
            "away_close_spread": away_close_spread,
            "home_spread_odds": row.get("home_spread_odds"),
            "away_spread_odds": row.get("away_spread_odds"),
            "close_total": row.get("total_line", row.get("close_total")),
            "game_id": row.get("game_id"),
        }
        games.append(game)
    games.sort(key=lambda g: (g["date"], g["home"], g["away"]))
    return games


def csv_rows_to_games(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept either nflverse or simplified nfl.csv closing-odds rows."""
    if rows and ("home_team" in rows[0] or "gameday" in rows[0]):
        return nflverse_rows_to_games(rows)
    games: list[dict[str, Any]] = []
    for row in rows:
        day_iso = str(row.get("date") or row.get("gameday") or "")[:10]
        if not day_iso:
            continue
        day = date_cls.fromisoformat(day_iso)
        season = int(row.get("season") or nfl_season_of(day))
        home = str(row.get("home_key") or row.get("home") or "").lower()
        away = str(row.get("away_key") or row.get("away") or "").lower()
        if not home or not away:
            continue
        home_score = row.get("home_final", row.get("home_score"))
        away_score = row.get("away_final", row.get("away_score"))
        if home_score is None or away_score is None:
            continue
        game = {
            "date": day_iso,
            "season": season,
            "home": home,
            "away": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "neutral_site": infer_neutral_site({"date": day_iso, "neutral_site": row.get("neutral_site")}),
            "home_close_ml": row.get("home_close_ml"),
            "away_close_ml": row.get("away_close_ml"),
            "home_close_spread": row.get("home_close_spread"),
            "away_close_spread": row.get("away_close_spread"),
            "home_spread_odds": row.get("home_spread_odds"),
            "away_spread_odds": row.get("away_spread_odds"),
            "close_total": row.get("close_total"),
        }
        games.append(game)
    games.sort(key=lambda g: (g["date"], g["home"], g["away"]))
    return games


def replay_season(
    engine: NflFeatureEngine,
    games: list[dict[str, Any]],
    *,
    stop_before_date: str | None = None,
    emit: Callable[[dict[str, Any], dict[str, float]], None] | None = None,
) -> None:
    """Walk games chronologically; optionally emit (game, features) rows."""
    for game in games:
        if stop_before_date and str(game.get("date") or "") >= stop_before_date:
            break
        if emit is not None:
            features = engine.features_for_game(game)
            emit(game, features)
        engine.update_after_game(game)


def replay_games(
    engine: NflFeatureEngine,
    games: list[dict[str, Any]],
    *,
    stop_before_date: str | None = None,
    emit: Callable[[dict[str, Any], dict[str, float]], None] | None = None,
) -> None:
    """Alias for multi-season chronological replay."""
    replay_season(engine, games, stop_before_date=stop_before_date, emit=emit)
