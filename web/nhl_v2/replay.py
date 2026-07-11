"""Chronological NHL season replay through the feature engine (no leakage).

Merges NHL Stats API game index, MoneyPuck situational xG and goalie starter
logs into engine-ready game dicts, emits feature rows morning-of-game, then
folds results into state.
"""

from __future__ import annotations

from typing import Any, Callable

from web.nhl_v2.feature_engine import NhlFeatureEngine

OnRow = Callable[[dict[str, Any], dict[str, float]], None]


def attach_moneypuck(
    game: dict[str, Any],
    mp_games: dict[int, dict[str, dict[str, float]]],
) -> None:
    mp = mp_games.get(int(game["gameId"]))
    if mp:
        game["mp"] = mp


def build_goalie_index(
    goalie_rows: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """(gameId, team) -> starter row (gamesStarted == 1)."""
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in goalie_rows:
        game_id = int(row.get("gameId") or 0)
        team = str(row.get("teamAbbrev") or "")
        if not game_id or not team:
            continue
        key = (game_id, team)
        if int(row.get("gamesStarted") or 0) == 1 or key not in index:
            if int(row.get("gamesStarted") or 0) == 1 or int(
                index.get(key, {}).get("gamesStarted") or 0
            ) != 1:
                index[key] = row
    return index


GoalieXgRow = tuple[float, float, float, float, float, float]


def attach_goalies(
    game: dict[str, Any],
    goalie_index: dict[tuple[int, str], dict[str, Any]],
    goalie_xg: dict[tuple[int, int], GoalieXgRow] | None = None,
) -> None:
    game_id = int(game["gameId"])
    for side in ("home", "away"):
        row = goalie_index.get((game_id, game[side]))
        if not row:
            continue
        goalie_id = row.get("playerId")
        game[f"{side}_goalie_id"] = goalie_id
        game[f"{side}_goalie_name"] = row.get("goalieFullName")
        game[f"{game[side]}_goalie_shots"] = row.get("shotsAgainst")
        game[f"{game[side]}_goalie_saves"] = row.get("saves")
        if goalie_xg and goalie_id:
            xg_row = goalie_xg.get((game_id, int(goalie_id)))
            if xg_row:
                shots, xg, goals, hd_shots, hd_xg, hd_goals = xg_row
                game[f"{game[side]}_goalie_xg_shots"] = shots
                game[f"{game[side]}_goalie_xg"] = xg
                game[f"{game[side]}_goalie_xg_goals"] = goals
                game[f"{game[side]}_goalie_hd_shots"] = hd_shots
                game[f"{game[side]}_goalie_hd_xg"] = hd_xg
                game[f"{game[side]}_goalie_hd_goals"] = hd_goals


def replay_season(
    engine: NhlFeatureEngine,
    season: int,
    games: list[dict[str, Any]],
    mp_games: dict[int, dict[str, dict[str, float]]],
    goalie_index: dict[tuple[int, str], dict[str, Any]],
    *,
    goalie_xg: dict[tuple[int, int], GoalieXgRow] | None = None,
    stop_before_date: str | None = None,
    on_row: OnRow | None = None,
) -> None:
    """Replay one season chronologically. Games must be date-sorted."""
    engine.start_season(season)
    ordered = sorted(games, key=lambda g: (str(g.get("date")), int(g.get("gameId") or 0)))
    for game in ordered:
        game_date = str(game.get("date") or "")
        if stop_before_date and game_date >= stop_before_date:
            break
        if not game.get("home") or not game.get("away"):
            continue
        attach_moneypuck(game, mp_games)
        attach_goalies(game, goalie_index, goalie_xg)
        if on_row is not None:
            features = engine.features_for_game(game)
            on_row(game, features)
        engine.update_after_game(game)


def load_goalie_xg_csv(path: Any) -> dict[tuple[int, int], GoalieXgRow]:
    """goalie_xg_games.csv -> (gameId, goalieId) -> (shots, xg, goals, hd_shots, hd_xg, hd_goals)."""
    import csv
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[tuple[int, int], GoalieXgRow] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (int(row["gameId"]), int(row["goalieId"]))
                out[key] = (
                    float(row["shots_faced"]),
                    float(row["xg_faced"]),
                    float(row["goals_allowed"]),
                    float(row.get("hd_shots_faced") or 0.0),
                    float(row.get("hd_xg_faced") or 0.0),
                    float(row.get("hd_goals_allowed") or 0.0),
                )
            except (KeyError, TypeError, ValueError):
                continue
    return out
