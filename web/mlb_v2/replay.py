"""Chronological season replay driving MlbFeatureEngine (offline + live parity).

Semantics: for each calendar date, feature rows for that date's games are
built BEFORE any of that date's results are folded into state ("morning
slate" view, identical to production inference timing).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from web.mlb_v2.feature_engine import MlbFeatureEngine

RowSink = Callable[[dict[str, Any], dict[str, float]], None]


def _index_team_logs(
    raw: dict[str, list[dict[str, Any]]],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    index: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for team_id, rows in (raw or {}).items():
        for row in rows:
            index[(int(team_id), str(row.get("date") or ""))].append(row)
    return index


def _index_pitcher_appearances(
    raw: dict[str, Any],
) -> dict[str, list[tuple[int, str, dict[str, Any]]]]:
    index: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for pid, payload in (raw or {}).items():
        hand = str(payload.get("hand") or "R")
        for log in payload.get("logs") or []:
            index[str(log.get("date") or "")].append((int(pid), hand, log))
    return index


def _starter_log(
    pitchers_raw: dict[str, Any], pitcher_id: Any, game_date: str
) -> dict[str, Any] | None:
    if not pitcher_id:
        return None
    payload = (pitchers_raw or {}).get(str(pitcher_id))
    if not payload:
        return None
    for log in payload.get("logs") or []:
        if str(log.get("date")) == game_date and int(log.get("gs") or 0) >= 1:
            return log
    return None


def _bullpen_line(
    team_log: dict[str, Any] | None, starter_log: dict[str, Any] | None
) -> dict[str, float] | None:
    if not team_log:
        return None
    if not starter_log:
        return {
            "outs": float(team_log.get("outs") or 0),
            "r": float(team_log.get("r") or 0),
            "so": float(team_log.get("so") or 0),
        }
    return {
        "outs": max(float(team_log.get("outs") or 0) - float(starter_log.get("outs") or 0), 0.0),
        "r": max(float(team_log.get("r") or 0) - float(starter_log.get("r") or 0), 0.0),
        "so": max(float(team_log.get("so") or 0) - float(starter_log.get("so") or 0), 0.0),
    }


def is_final_game(game: dict[str, Any]) -> bool:
    # MLB Stats API uses F (Final) and O (Game Over) for completed games.
    status = str(game.get("status") or "")
    return (
        status in {"F", "O"}
        and game.get("home_score") is not None
        and game.get("away_score") is not None
    )


def replay_season(
    engine: MlbFeatureEngine,
    season: int,
    games: list[dict[str, Any]],
    pitchers: dict[str, Any],
    team_hitting: dict[str, list[dict[str, Any]]],
    team_pitching: dict[str, list[dict[str, Any]]],
    *,
    on_row: RowSink | None = None,
    stop_before_date: str | None = None,
) -> None:
    """Replay one season in date order.

    on_row(game, features) is called for every final game (before updates).
    stop_before_date: replay state only for games strictly before this ISO
    date (used by live inference to position state for today's slate).
    """
    engine.begin_season(season)
    hit_index = _index_team_logs(team_hitting)
    pitch_index = _index_team_logs(team_pitching)
    appearances = _index_pitcher_appearances(pitchers)

    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_date[str(game.get("date") or "")].append(game)

    for game_date in sorted(by_date):
        if not game_date:
            continue
        if stop_before_date is not None and game_date >= stop_before_date:
            break
        day_games = sorted(by_date[game_date], key=lambda g: (g.get("gamePk") or 0))
        finals = [g for g in day_games if is_final_game(g)]

        if on_row is not None:
            for game in finals:
                on_row(game, engine.features_for_game(game))

        occurrence: dict[int, int] = defaultdict(int)
        for game in finals:
            home_id = int(game["home_id"])
            away_id = int(game["away_id"])
            home_occ = occurrence[home_id]
            away_occ = occurrence[away_id]
            occurrence[home_id] += 1
            occurrence[away_id] += 1

            home_hits = hit_index.get((home_id, game_date)) or []
            away_hits = hit_index.get((away_id, game_date)) or []
            home_pitch = pitch_index.get((home_id, game_date)) or []
            away_pitch = pitch_index.get((away_id, game_date)) or []

            home_hit_log = home_hits[home_occ] if home_occ < len(home_hits) else None
            away_hit_log = away_hits[away_occ] if away_occ < len(away_hits) else None
            home_team_pitch = home_pitch[home_occ] if home_occ < len(home_pitch) else None
            away_team_pitch = away_pitch[away_occ] if away_occ < len(away_pitch) else None

            home_starter = _starter_log(pitchers, game.get("home_pp_id"), game_date)
            away_starter = _starter_log(pitchers, game.get("away_pp_id"), game_date)

            engine.update_after_game(
                game,
                home_hit_log=home_hit_log,
                away_hit_log=away_hit_log,
                home_bullpen=_bullpen_line(home_team_pitch, home_starter),
                away_bullpen=_bullpen_line(away_team_pitch, away_starter),
            )

        for pitcher_id, hand, log in appearances.get(game_date, []):
            engine.apply_pitcher_appearance(pitcher_id, hand, log)
