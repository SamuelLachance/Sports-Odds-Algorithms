"""NHL season-boundary logic, extracted so it can be TESTED.

The NHL analog of `nfl_season_guards.py`. Every function here first matters in
October, when the 2026-27 schedule appears while the spine still ends in
2025-26 — and each of the three defects below was verified against the real
spine before this module existed.

1. `current_season` — the serve derived the current season as
   `max(g["season"] for g in games)` over the spine, which holds only COMPLETED
   games. Between the day the new schedule posts and the day the first new
   final lands, that expression still returns the OLD season: the payload
   serves next season's upcoming games beside last season's completed games,
   standings and current-season log-loss, labelled with last season's id.
   No crash — just a site that is quietly a season out of date.

2. `display_teams` — the team set was `{t for t in rec}`, i.e. teams with a
   COMPLETED game this season. On opening night that is 2 teams, so the
   standings and ratings pages would list 2 of 32 and fill in over a week.
   The set must be "active teams" (current or previous season), which still
   drops defunct codes (ATL/PHX/ARI) because they appear in neither.

3. `fmt_metric` — with no completed games the realized LL/accuracy are None,
   and the serve's closing `print(f"...{cur_ll:.5f}...")` raises TypeError on
   None. This is why the three fixes ship together: correcting (1) so the
   season rolls on schedule is exactly what makes `done` empty on opening day,
   which is what makes (3) reachable. Fixing the season roll alone would turn a
   silent staleness bug into a hard crash on the season's first morning.

Dependency-free (no numpy) so the boundary is testable in isolation.
"""
from __future__ import annotations

# NHL game ids are 10 digits: SSSS TT NNNN — a four-digit season START year,
# a two-digit game type (02 regular, 03 playoff), then the game number.
_ID_LEN = 10
_PLAYOFF_TYPE = "03"


def season_of_game_id(game_id) -> int | None:
    """Season id (e.g. 20262027) encoded in an NHL game id, or None if unparseable.

    Upcoming games arrive from the schedule API with an id and a date but no
    season field, so the season has to come from the id itself.
    """
    try:
        s = str(int(game_id))
    except (TypeError, ValueError):
        return None
    if len(s) != _ID_LEN:
        return None
    y = int(s[:4])
    return y * 10000 + (y + 1)


def is_playoff(game_id) -> bool:
    try:
        s = str(int(game_id))
    except (TypeError, ValueError):
        return False
    return len(s) == _ID_LEN and s[4:6] == _PLAYOFF_TYPE


def current_season(completed_seasons, upcoming_ids=()) -> int | None:
    """Newest season anywhere — completed OR merely scheduled.

    Taking the max over both is what makes the payload roll to the new season
    the moment its schedule posts, instead of lagging until its first final.
    """
    seasons = [s for s in completed_seasons if s is not None]
    seasons += [s for s in map(season_of_game_id, upcoming_ids) if s is not None]
    return max(seasons) if seasons else None


def display_teams(rated: dict, cur_teams, prev_teams) -> dict:
    """Rated teams active in the current or previous season, preserving order.

    Including the previous season is what keeps all 32 present on opening night,
    when the current season's schedule names only the teams playing this week
    (`nhl_upcoming.json` covers a 10-day window, not the full season).
    """
    active = set(cur_teams) | set(prev_teams)
    return {t: v for t, v in rated.items() if t in active}


def fmt_metric(value, nd: int) -> str:
    """Format a realized metric that is None until the season's first game."""
    return "n/a" if value is None else f"{value:.{nd}f}"
