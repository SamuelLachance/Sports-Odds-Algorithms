"""Chronological replay of NBA seasons through the feature engine.

Walks ESPN events (authoritative scores for 1996+) with box scores attached,
emitting leak-free feature rows before folding each result into state.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from typing import Any, Callable

from web.nba_v2.data import franchise_for_espn_id
from web.nba_v2.feature_engine import NbaFeatureEngine


def _iso_shift(iso: str, days: int) -> str:
    try:
        return (date_cls.fromisoformat(iso[:10]) + timedelta(days=days)).isoformat()
    except ValueError:
        return iso


def _espn_local_date(event: dict[str, Any]) -> str:
    """America/Toronto calendar day — same keying as fetch_season_events."""
    from web.season_games import _event_date_iso

    return _event_date_iso(str(event.get("date") or ""))


def events_to_results(events: list[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    """Authoritative game list from completed ESPN events."""
    rows: list[dict[str, Any]] = []
    for event in events:
        if not event.get("completed"):
            continue
        home_score = event.get("home_score")
        away_score = event.get("away_score")
        if home_score is None or away_score is None:
            continue
        rows.append(
            {
                "date": _espn_local_date(event),
                "season": season,
                "season_type": int(event.get("season_type") or 2),
                "home": franchise_for_espn_id(
                    event.get("home_id") or "", event.get("home_abbr") or ""
                ),
                "away": franchise_for_espn_id(
                    event.get("away_id") or "", event.get("away_abbr") or ""
                ),
                "home_score": int(home_score),
                "away_score": int(away_score),
                "neutral_site": bool(event.get("neutral_site")),
                "event_id": event.get("event_id"),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def build_espn_index(events: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        home = franchise_for_espn_id(event.get("home_id") or "", event.get("home_abbr") or "")
        away = franchise_for_espn_id(event.get("away_id") or "", event.get("away_abbr") or "")
        date = _espn_local_date(event)
        if home and away and date:
            index[(date, home, away)] = event
    return index


def match_espn_event(
    index: dict[tuple[str, str, str], dict[str, Any]],
    date: str,
    home: str,
    away: str,
) -> dict[str, Any] | None:
    for offset in (0, 1, -1):
        key = (_iso_shift(date, offset), home, away)
        event = index.get(key)
        if event is not None:
            return event
    return None


def merge_season_games(
    results: list[dict[str, Any]],
    events: list[dict[str, Any]],
    boxes: dict[str, Any] | None = None,
    *,
    season: int | None = None,
) -> list[dict[str, Any]]:
    """Authoritative game list with ESPN event ids and box stats attached."""
    if not results and events and season is not None:
        results = events_to_results(events, season)
    index = build_espn_index(events)
    boxes = boxes or {}
    games: list[dict[str, Any]] = []
    for row in results:
        game = dict(row)
        # 2019-20 restart / bubble games in Orlando were neutral-site.
        # ESPN sometimes flags them; force neutral for Jul–Oct 2020 dates.
        date = str(row.get("date") or "")
        if date >= "2020-07-01" and date <= "2020-10-15":
            game["neutral_site"] = True
        event = None
        if game.get("event_id"):
            for ev in events:
                if str(ev.get("event_id")) == str(game["event_id"]):
                    event = ev
                    break
        if event is None:
            event = match_espn_event(index, row["date"], row["home"], row["away"])
        if event is not None:
            game["event_id"] = event.get("event_id")
            game["neutral_site"] = game.get("neutral_site") or event.get("neutral_site", False)
            box = boxes.get(str(event.get("event_id")))
            if isinstance(box, dict):
                home_box = box.get(str(event.get("home_id")))
                away_box = box.get(str(event.get("away_id")))
                if home_box and away_box:
                    game["home_box"] = home_box
                    game["away_box"] = away_box
        games.append(game)
    games.sort(key=lambda g: (str(g.get("date")), str(g.get("event_id") or "")))
    return games


def replay_season(
    engine: NbaFeatureEngine,
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
