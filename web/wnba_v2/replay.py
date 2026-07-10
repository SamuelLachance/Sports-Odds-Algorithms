"""Chronological replay of WNBA seasons through the feature engine.

Merges data.wnba.com results (authoritative scores) with ESPN events
(box scores, odds join) by franchise pair and date (+/- 1 day for the
UTC-vs-local skew), then walks games in order emitting leak-free
feature rows before folding each result into state.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from typing import Any, Callable

from web.wnba_v2.data import franchise_for_espn_id
from web.wnba_v2.feature_engine import WnbaFeatureEngine


def _iso_shift(iso: str, days: int) -> str:
    try:
        return (date_cls.fromisoformat(iso[:10]) + timedelta(days=days)).isoformat()
    except ValueError:
        return iso


def _espn_local_date(event: dict[str, Any]) -> str:
    """US-local calendar date from an ESPN UTC timestamp (ET approximation)."""
    raw = str(event.get("date") or "")
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (stamp - timedelta(hours=5)).date().isoformat()
    except ValueError:
        return raw[:10]


def events_to_results(events: list[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    """Synthesize the authoritative game list from ESPN events.

    Used for seasons data.wnba.com does not host (2007, current). ESPN events
    carry final scores from 2002 onward.
    """
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
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def build_espn_index(events: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(date, home_franchise, away_franchise) -> espn event (all dates +/-0)."""
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        home = franchise_for_espn_id(event.get("home_id") or "", event.get("home_abbr") or "")
        away = franchise_for_espn_id(event.get("away_id") or "", event.get("away_abbr") or "")
        date = str(event.get("date") or "")[:10]
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
        # 2020 was played entirely in the IMG Academy bubble; ESPN does not
        # flag those games neutral, so no team gets home-court that season.
        if int(row.get("season") or 0) == 2020:
            game["neutral_site"] = True
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
    engine: WnbaFeatureEngine,
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
