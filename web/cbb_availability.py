"""CBB availability and early-season uncertainty signals."""

from __future__ import annotations

from web.availability_signals import _count_team_injuries

EARLY_SEASON_GAME_THRESHOLD = 5
MAX_AVAILABILITY_SHIFT_PP = 2.5
EARLY_SEASON_UNCERTAINTY_PP = 0.75


def availability_home_shift_pp(
    league: str,
    home_espn_id: str | None,
    away_espn_id: str | None,
    cutoff_date: str,
    home_games: int,
    away_games: int,
) -> float:
    """
    Signed percentage-point shift to home win probability from injuries and
    early-season roster churn (low games played increases uncertainty, nudging
    toward neutral when sample is thin).
    """
    _ = cutoff_date
    home_injuries, home_out = (0, 0)
    away_injuries, away_out = (0, 0)

    if home_espn_id:
        home_injuries, home_out = _count_team_injuries(league, home_espn_id)
    if away_espn_id:
        away_injuries, away_out = _count_team_injuries(league, away_espn_id)

    home_burden = home_out + 0.4 * max(home_injuries - home_out, 0)
    away_burden = away_out + 0.4 * max(away_injuries - away_out, 0)
    injury_shift = away_burden - home_burden
    shift = injury_shift * 0.35

    # Early-season roster churn proxy: thin samples reduce confidence.
    min_games = min(home_games, away_games)
    if min_games < EARLY_SEASON_GAME_THRESHOLD:
        deficit = EARLY_SEASON_GAME_THRESHOLD - min_games
        shift *= max(0.35, 1.0 - deficit * 0.12)

    return round(
        max(min(shift, MAX_AVAILABILITY_SHIFT_PP), -MAX_AVAILABILITY_SHIFT_PP),
        2,
    )
