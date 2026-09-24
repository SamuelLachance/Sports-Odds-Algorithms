"""NHL serving at a season boundary: the two bugs that only appear when the
spine ends in one season and every served game is in the next."""
import sys

sys.path.insert(0, "phase0")
from nhl_serve_guards import needs_boundary_regression, schedule_rest  # noqa: E402


def test_regression_applies_before_the_first_game_of_a_new_season():
    assert needs_boundary_regression(20252026, 20262027)


def test_no_second_regression_once_the_walk_is_inside_the_season():
    """After opening night the walk itself crosses the boundary and regresses;
    applying it again would double-shrink every rating."""
    assert not needs_boundary_regression(20262027, 20262027)


def _g(i, d, h, a):
    return {"id": i, "d": d, "home": h, "away": a}


def test_a_back_to_back_inside_the_window_is_seen():
    up = [_g(1, "2026-10-07", "BOS", "MTL"), _g(2, "2026-10-08", "TOR", "BOS")]
    r = schedule_rest({}, up)
    assert r[1][3] == 1.0          # BOS is the away side, second night
    assert r[1][1] == 1            # one day of rest


def test_rest_counts_scheduled_games_not_only_played_ones():
    last = {"BOS": "2026-04-16"}
    up = [_g(1, "2026-10-07", "BOS", "MTL"), _g(2, "2026-10-10", "BOS", "NYR")]
    r = schedule_rest(last, up)
    assert r[0][0] == 5            # capped: months since April
    assert r[1][0] == 3            # three days after the Oct 7 game, not months


def test_results_come_back_in_the_callers_order():
    up = [_g(2, "2026-10-08", "TOR", "BOS"), _g(1, "2026-10-07", "BOS", "MTL")]
    r = schedule_rest({}, up)
    assert r[0][3] == 1.0 and r[1][2] == 0.0


def test_a_team_with_no_history_gets_the_default_rest():
    r = schedule_rest({}, [_g(1, "2026-10-07", "SEA", "UTA")])
    assert r[0][:2] == (3, 3)
