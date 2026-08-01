"""mlbwp/db.py — the standings/roster/player database behind the site's DB pages.

It runs on every refresh and had no tests. Two defects were found by reading it
with the same lens that caught the NHL boundary bugs:

  - a bare `except Exception: continue` dropped a whole team's roster on any
    fetch hiccup, and main() wrote the result straight over the last good
    db.json. An empty roster on the site was indistinguishable from a team we
    could not ask about.
  - main(season=None) resolved the default at only one of four use sites, so a
    bare `python -m mlbwp.db` sent None into build_teams, build_players and the
    payload. (Self-inflicted, iteration 64; refresh.py always passes a season,
    so production never saw it.)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlbwp import db  # noqa: E402


# ---- the write guard -------------------------------------------------------

def test_first_build_always_writes():
    assert db.write_is_safe(0, 0) is True
    assert db.write_is_safe(900, 0) is True


def test_routine_roster_churn_still_writes():
    """Call-ups and IL moves shift a handful of players; that is a refresh."""
    assert db.write_is_safe(986, 1000) is True
    assert db.write_is_safe(1014, 1000) is True


def test_a_lost_roster_blocks_the_write():
    """Losing whole teams (~26 players each) is a fetch failure, not churn, and
    must not overwrite the last good database."""
    assert db.write_is_safe(700, 1000) is False
    assert db.write_is_safe(0, 1000) is False


def test_the_floor_is_where_it_claims_to_be():
    assert db.write_is_safe(800, 1000) is True     # exactly the floor
    assert db.write_is_safe(799, 1000) is False


def test_a_negative_or_absent_previous_count_never_blocks():
    """A corrupt or missing db.json must not wedge the build permanently."""
    assert db.write_is_safe(5, -1) is True


def test_a_blocked_write_skips_rather_than_raising():
    """main() runs BEFORE the NHL serve and build_site.build(). Raising here
    would turn a transient MLB API blip into 'the whole site failed to rebuild',
    which is worse than one cycle of stale player data. The guard exists to keep
    the last good database, and skipping does that."""
    src = inspect.getsource(db.main)
    guard = src.split("if not write_is_safe", 1)[1].split("_write_atomic", 1)[0]
    code = "\n".join(ln.split("#")[0] for ln in guard.splitlines())
    assert "return" in code, "the guard must skip the write"
    assert "raise" not in code, (
        "raising here halts the NHL serve and the site rebuild downstream")


# ---- the season-resolution regression -------------------------------------

def test_main_resolves_the_season_before_using_it():
    """It flows to the finals path, build_teams, build_players AND the payload.
    Defaulting it inline at one call site leaves None in the other three."""
    src = inspect.getsource(db.main)
    assert "season = season or pred.serve_season" in src
    body = src.split("season = season or pred.serve_season", 1)[1]
    assert "season or pred.serve_season" not in body, (
        "season is still being re-defaulted downstream — resolve it once")


def test_payload_records_a_real_season():
    src = inspect.getsource(db.main)
    assert '"season": season' in src
    # and the resolution happens before the payload is built
    assert src.index("season = season or") < src.index('"season": season')


# ---- pure helpers ----------------------------------------------------------

def test_pythag_is_a_probability_and_symmetric():
    assert db.pythag(0, 0) == 0.5, "no games played is a coin flip, not a crash"
    assert db.pythag(700, 700) == 0.5
    assert 0.0 <= db.pythag(900, 600) <= 1.0
    assert db.pythag(900, 600) > 0.5 > db.pythag(600, 900)
    assert abs(db.pythag(900, 600) + db.pythag(600, 900) - 1.0) < 1e-12


def test_pythag_handles_a_shutout_season_without_dividing_by_zero():
    assert db.pythag(0, 700) == 0.0
    assert db.pythag(700, 0) == 1.0


def test_phi_is_the_normal_cdf():
    assert abs(db._phi(0) - 0.5) < 1e-12
    assert abs(db._phi(1.96) - 0.975) < 1e-3
    assert db._phi(-3) < db._phi(0) < db._phi(3)


def test_f_returns_None_rather_than_raising_on_junk():
    """Stat fields arrive from the API as strings like '-.--' for no data."""
    assert db._f("3.14") == 3.14
    for junk in (None, "", "-.--", "N/A", [], {}):
        assert db._f(junk) is None


def test_split_and_l10_degrade_to_placeholders():
    assert db._split({}, "home") == "-"
    assert db._l10({}) == (0, 0)
    recs = {"splitRecords": [{"type": "lastTen", "wins": 6, "losses": 4},
                             {"type": "home", "wins": 30, "losses": 21}]}
    assert db._l10(recs) == (6, 4)
    assert db._split(recs, "home") == "30-21"
    assert db._split(recs, "away") == "-", "a missing split must not match another"
