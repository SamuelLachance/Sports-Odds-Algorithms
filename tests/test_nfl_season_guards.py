"""The two NFL in-season guards. Both fire for the first time on 2026-09-09.

Neither had any coverage until now: they were inline expressions in
nfl_season_serve.py, exercised once by a synthetic dry run. One fails LOUDLY
(the stale-npy stop) and one fails SILENTLY (double-regressed ratings), and the
silent one is the dangerous half.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))

import nfl_season_guards as G  # noqa: E402


# ---- guard 1: the manual season-boundary transform -------------------------

def test_boundary_is_on_while_the_walk_is_still_in_2025():
    """Today's state, and every day until week 1: the walk ended in 2025, so the
    manual transform is still the thing that moves states into the serve season."""
    assert G.apply_2026_boundary(2025) is True
    assert G.apply_2026_boundary(2024) is True


def test_boundary_turns_off_the_moment_a_2026_FINAL_lands():
    """The silent failure this exists to prevent. Once the walk itself crosses
    into 2026 its in-loop season-change trigger has already regressed the
    ratings; applying the manual transform again would double-regress them —
    no crash, just quietly wrong probabilities for the whole season."""
    assert G.apply_2026_boundary(2026) is False
    assert G.apply_2026_boundary(2027) is False


def test_boundary_is_parameterised_by_the_serve_season():
    """It must not be hardcoded to 2026 forever — 2027 needs the same logic with
    no code edit beyond the season."""
    assert G.apply_2026_boundary(2026, serve_season=2027) is True
    assert G.apply_2026_boundary(2027, serve_season=2027) is False


# ---- guard 2: the frozen v7 feature column --------------------------------

def test_matching_lengths_pass_silently():
    assert G.v7_npy_error(7276, 7276) is None


def test_shorter_npy_names_the_real_cause_and_the_fix():
    """The opening-day case: a 2026 final is appended and the column is short.
    A bare shape error at 6am on a game day is useless — the message must say
    what happened AND the command that repairs it."""
    msg = G.v7_npy_error(7276, 7277)
    assert msg is not None
    assert "7276 rows vs 7277 games" in msg
    assert "new finals landed" in msg
    assert "python phase0/nfl_v7_feature_gen.py" in msg


def test_longer_npy_is_diagnosed_as_a_DIFFERENT_problem():
    """More rows than games means the spine went BACKWARDS — a data problem, not
    a regeneration one. Telling the operator to regenerate would be wrong advice,
    so the two directions must not share a message."""
    msg = G.v7_npy_error(7277, 7276)
    assert msg is not None
    assert "rolled back" in msg
    assert "new finals landed" not in msg


def test_the_two_directions_never_produce_the_same_message():
    assert G.v7_npy_error(10, 11) != G.v7_npy_error(11, 10)


def test_shipped_artifacts_currently_agree():
    """Contract against the real files: if this fails, the serve is already
    broken and the next run would stop."""
    import csv

    import pytest
    np = pytest.importorskip("numpy")   # stdlib-only CI: skip, don't fail
    root = Path(__file__).resolve().parents[1]
    npy, spine = root / "data" / "nfl_v7_feature.npy", root / "data" / "nfl_games.csv"
    if not (npy.is_file() and spine.is_file()):
        import pytest
        pytest.skip("NFL artifacts not present")
    n_games = sum(1 for r in csv.DictReader(spine.open(encoding="utf-8"))
                  if r.get("home_score") not in (None, ""))
    assert G.v7_npy_error(len(np.load(npy)), n_games) is None
