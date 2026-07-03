"""WNBA availability, injuries, and early-season roster uncertainty."""

from __future__ import annotations

from web.availability_signals import _count_team_injuries

EARLY_SEASON_GAME_THRESHOLD = 8
MAX_AVAILABILITY_SHIFT_PP = 2.5


def availability_home_shift_pp(
    league: str,
    home_espn_id: str | None,
    away_espn_id: str | None,
    cutoff_date: str,
    home_games: int,
    away_games: int,
) -> float:
    _ = cutoff_date
    home_injuries, home_out = (0, 0)
    away_injuries, away_out = (0, 0)

    if home_espn_id:
        home_injuries, home_out = _count_team_injuries(league, home_espn_id)
    if away_espn_id:
        away_injuries, away_out = _count_team_injuries(league, away_espn_id)

    home_burden = home_out + 0.45 * max(home_injuries - home_out, 0)
    away_burden = away_out + 0.45 * max(away_injuries - away_out, 0)
    shift = (away_burden - home_burden) * 0.4

    min_games = min(home_games, away_games)
    if min_games < EARLY_SEASON_GAME_THRESHOLD:
        deficit = EARLY_SEASON_GAME_THRESHOLD - min_games
        shift *= max(0.3, 1.0 - deficit * 0.1)

    return round(
        max(min(shift, MAX_AVAILABILITY_SHIFT_PP), -MAX_AVAILABILITY_SHIFT_PP),
        2,
    )
