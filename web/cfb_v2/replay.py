"""Chronological replay of CFB seasons through the feature engine.

Walks completed games (closing-odds CSV and/or ESPN scoreboard events),
emitting leak-free feature rows before folding each result into state.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any, Callable

from web.cfb_v2.feature_engine import (
    CfbFeatureEngine,
    cfb_season_of,
    infer_bowl,
    infer_neutral_site,
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
        game_season = int(season if season is not None else (cfb_season_of(day) if day else 0))
        row = {
            "date": day_iso,
            "season": game_season,
            "home": home,
            "away": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "neutral_site": bool(event.get("neutral_site")) if event.get("neutral_site") is not None else None,
            "event_id": event.get("event_id"),
            "conference_game": event.get("conference_game"),
        }
        if row["neutral_site"] is None:
            row["neutral_site"] = infer_neutral_site(row)
        rows.append(row)
    rows.sort(key=lambda r: (r["date"], r["home"], r["away"]))
    return rows


def csv_rows_to_games(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert closing-odds CSV dict rows into engine game dicts."""
    games: list[dict[str, Any]] = []
    for row in rows:
        day_iso = str(row.get("date") or "")[:10]
        if not day_iso:
            continue
        day = date_cls.fromisoformat(day_iso)
        season = int(row.get("season") or cfb_season_of(day))
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
            "is_bowl": infer_bowl({"date": day_iso}),
            "home_close_ml": row.get("home_close_ml"),
            "away_close_ml": row.get("away_close_ml"),
            "home_close_spread": row.get("home_close_spread"),
            "away_close_spread": row.get("away_close_spread"),
            "home_spread_odds": row.get("home_spread_odds"),
            "away_spread_odds": row.get("away_spread_odds"),
            "home_open_spread": row.get("home_open_spread"),
            "away_open_spread": row.get("away_open_spread"),
            "close_total": row.get("close_total"),
            "n_books": row.get("n_books"),
        }
        games.append(game)
    games.sort(key=lambda g: (g["date"], g["home"], g["away"]))
    return games


def replay_season(
    engine: CfbFeatureEngine,
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
    engine: CfbFeatureEngine,
    games: list[dict[str, Any]],
    *,
    stop_before_date: str | None = None,
    emit: Callable[[dict[str, Any], dict[str, float]], None] | None = None,
) -> None:
    """Alias for multi-season chronological replay."""
    replay_season(engine, games, stop_before_date=stop_before_date, emit=emit)
