"""NHL season-boundary logic. All of it first matters on opening night, October.

Two of the three defects covered here were SILENT (a season-stale payload, a
2-of-32 standings table) and one was a hard crash reachable only after the first
is fixed. They were reproduced against the real spine before the module existed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))

import nhl_season_boundary as B  # noqa: E402


# ---- game-id decoding ------------------------------------------------------

def test_season_is_decoded_from_the_game_id():
    """Upcoming games carry an id and a date but NO season field, so the season
    has to be recovered from the id or the payload cannot roll."""
    assert B.season_of_game_id(2026020001) == 20262027
    assert B.season_of_game_id(2025020500) == 20252026
    assert B.season_of_game_id("2026030111") == 20262027   # str form too


def test_unparseable_ids_are_None_not_a_guess():
    for bad in (None, "", "abc", 123, 12345678901, [], {}):
        assert B.season_of_game_id(bad) is None


def test_playoff_flag_reads_the_game_type_digits():
    assert B.is_playoff(2025030111) is True     # 03 = playoff
    assert B.is_playoff(2025020111) is False    # 02 = regular
    assert B.is_playoff("nonsense") is False


# ---- defect 1: the season roll --------------------------------------------

def test_season_rolls_as_soon_as_the_new_schedule_posts():
    """THE OCTOBER BUG. The spine holds only COMPLETED games, so deriving the
    current season from it alone leaves the payload serving 2026-27 upcoming
    games beside 2025-26 standings, labelled 20252026 — for as long as it takes
    the first new final to land."""
    completed = [20232024, 20242025, 20252026]
    assert B.current_season(completed) == 20252026          # spine alone: stale
    assert B.current_season(completed, [2026020001]) == 20262027


def test_offseason_keeps_the_last_completed_season():
    """August: the schedule has not posted, so nothing rolls."""
    assert B.current_season([20242025, 20252026], []) == 20252026


def test_a_stray_old_upcoming_id_cannot_roll_the_season_backwards():
    assert B.current_season([20252026], [2019020001]) == 20252026


def test_empty_everything_is_None_not_an_exception():
    assert B.current_season([], []) is None


# ---- defect 2: the team set -----------------------------------------------

_RATED = {"TOR": 1550, "MTL": 1480, "BOS": 1520, "FLA": 1530, "ARI": 1490}


def test_all_active_teams_show_on_opening_night():
    """The standings previously came from teams with a COMPLETED game, so on
    opening night the page listed 2 of 32 and filled in over a week. The 10-day
    upcoming window does not name every team either, which is why the previous
    season is unioned in."""
    cur = {"TOR", "MTL"}                     # only tonight's game
    prev = {"TOR", "MTL", "BOS", "FLA"}      # last season's full slate
    out = B.display_teams(_RATED, cur, prev)
    assert set(out) == {"TOR", "MTL", "BOS", "FLA"}


def test_defunct_franchises_are_still_dropped():
    """The union must not resurrect relocated clubs: ARI is rated but appears in
    neither recent season, so it stays out of the standings."""
    out = B.display_teams(_RATED, {"TOR"}, {"TOR", "MTL", "BOS", "FLA"})
    assert "ARI" not in out


def test_midseason_is_unchanged_by_the_union():
    """Once both seasons name the same 4 clubs the fix is a no-op — it must not
    alter what the site shows for the other eleven months."""
    full = {"TOR", "MTL", "BOS", "FLA"}
    assert set(B.display_teams(_RATED, full, full)) == full


def test_team_order_is_preserved():
    """`teams` is emitted as JSON and the SPA ranks from it; a set-ordered dict
    would reshuffle the payload on every serve and churn the git diff."""
    out = B.display_teams(_RATED, {"TOR", "MTL", "BOS", "FLA"}, set())
    assert list(out) == ["TOR", "MTL", "BOS", "FLA"]


# ---- defect 3: the crash that fixing defect 1 unlocks ----------------------

def test_none_metrics_format_instead_of_raising():
    """With the season rolled and no game played yet, realized LL/acc are None
    and the serve's closing f-string raised TypeError. Verified: an f-string
    format spec on None is a hard error, not a fallback."""
    import pytest
    with pytest.raises(TypeError):
        f"{None:.5f}"                       # the original expression
    assert B.fmt_metric(None, 5) == "n/a"
    assert B.fmt_metric(None, 4) == "n/a"


def test_real_metrics_keep_their_precision():
    assert B.fmt_metric(0.685314, 5) == "0.68531"
    assert B.fmt_metric(0.5404, 4) == "0.5404"
    assert B.fmt_metric(0.0, 4) == "0.0000", "0.0 is a real value, not a missing one"
