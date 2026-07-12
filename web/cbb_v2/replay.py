"""Chronological replay of CBB seasons through the feature engine."""

from __future__ import annotations

from typing import Any, Callable

from web.cbb_v2.data import canon_abbr, force_postseason_neutral_site, team_key
from web.cbb_v2.feature_engine import CbbFeatureEngine


def _espn_local_date(event: dict[str, Any]) -> str:
    """America/Toronto calendar day — same keying as NBA/WNBA and season_games."""
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
        home = team_key(event.get("home_id") or "", event.get("home_abbr") or "")
        away = team_key(event.get("away_id") or "", event.get("away_abbr") or "")
        if not home or not away:
            continue
        rows.append(
            {
                "date": _espn_local_date(event),
                "season": int(event.get("season") or season),
                "season_type": int(event.get("season_type") or 2),
                "home": home,
                "away": away,
                "home_abbr": canon_abbr(str(event.get("home_abbr") or "")),
                "away_abbr": canon_abbr(str(event.get("away_abbr") or "")),
                "home_score": int(home_score),
                "away_score": int(away_score),
                "neutral_site": bool(event.get("neutral_site")),
                "conference_game": bool(event.get("conference_game")),
                "home_conference_id": str(event.get("home_conference_id") or ""),
                "away_conference_id": str(event.get("away_conference_id") or ""),
                "event_id": event.get("event_id"),
            }
        )
    rows.sort(key=lambda r: (str(r.get("date")), str(r.get("event_id") or "")))
    return rows


def merge_season_games(
    results: list[dict[str, Any]],
    events: list[dict[str, Any]],
    boxes: dict[str, Any] | None = None,
    *,
    season: int | None = None,
) -> list[dict[str, Any]]:
    """Completed games with ESPN metadata and optional box stats attached."""
    if not results and events and season is not None:
        results = events_to_results(events, season)
    by_id = {str(e.get("event_id")): e for e in events if e.get("event_id")}
    boxes = boxes or {}
    games: list[dict[str, Any]] = []
    for row in results:
        game = dict(row)
        event = by_id.get(str(game.get("event_id") or ""))
        if event is not None:
            game["neutral_site"] = game.get("neutral_site") or event.get("neutral_site", False)
            game["conference_game"] = game.get("conference_game") or event.get(
                "conference_game", False
            )
            game.setdefault("home_abbr", canon_abbr(str(event.get("home_abbr") or "")))
            game.setdefault("away_abbr", canon_abbr(str(event.get("away_abbr") or "")))
            game.setdefault(
                "home_conference_id", str(event.get("home_conference_id") or "")
            )
            game.setdefault(
                "away_conference_id", str(event.get("away_conference_id") or "")
            )
            box = boxes.get(str(event.get("event_id")))
            if isinstance(box, dict):
                home_box = box.get(str(event.get("home_id")))
                away_box = box.get(str(event.get("away_id")))
                if home_box and away_box:
                    game["home_box"] = home_box
                    game["away_box"] = away_box
        force_postseason_neutral_site(game)
        games.append(game)
    games.sort(key=lambda g: (str(g.get("date")), str(g.get("event_id") or "")))
    return games


def replay_season(
    engine: CbbFeatureEngine,
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
