"""The NHL announced-lineup archive — the dedup key that decides what survives.

This archive is the deployable path for ledger row 10 (scratch-absence, the most
promising null): nobody publishes announced NHL lineups historically, so the
project captures them itself from October on. Everything the row-10 reopen
eventually tests comes through here, and until the season starts the file does
not exist, so none of it has ever run against real game-day data.

The key used to be `(gid, frozenset(_pid_set(sub)))` — the flat union of every
player id in the snapshot. That union is BLIND to the one event the archive
exists for: a player moving from `dressed` to `railScratches` is still in the
union, so the key matches the previous snapshot and the late scratch is dropped
as a duplicate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nhl_update as U  # noqa: E402


def _snap(dressed, scratches=()):
    return {
        "dressed": [{"teamId": 1, "playerId": p, "positionCode": "C",
                     "sweaterNumber": p % 100} for p in dressed],
        "railScratches": {"homeTeam": list(scratches)},
        "headCoach": {"homeTeam": "A Coach"},
    }


# ---- the bug this replaced -------------------------------------------------

def test_a_late_scratch_is_a_different_snapshot():
    """THE point of the change. Player 8477492 moves from dressed to scratched:
    the flat id union is identical, so the old key could not tell these apart
    and the scratch — the whole reason the archive exists — was discarded."""
    before = _snap([8477492, 8478402, 8479318])
    after = _snap([8478402, 8479318], scratches=[8477492])

    assert U._pid_set(before) == U._pid_set(after), (
        "precondition: the flat id union really is blind to this")
    assert U.lineup_fingerprint(before) != U.lineup_fingerprint(after), (
        "a late scratch must not key as a duplicate")


def test_a_goalie_swap_is_a_different_snapshot():
    a = {"goalieComparison": {"homeTeam": [8476945]}}
    b = {"goalieComparison": {"homeTeam": [8477970]}}
    assert U.lineup_fingerprint(a) != U.lineup_fingerprint(b)


def test_a_coach_change_is_visible_even_with_no_ids_at_all():
    """A snapshot carrying only a coach has an EMPTY id set, so every such
    snapshot keyed identically — the degenerate-key failure."""
    a = {"headCoach": {"homeTeam": "First Coach"}}
    b = {"headCoach": {"homeTeam": "Second Coach"}}
    assert U._pid_set(a) == U._pid_set(b) == set()
    assert U.lineup_fingerprint(a) != U.lineup_fingerprint(b)


# ---- and it must still suppress genuine duplicates -------------------------

def test_an_identical_refetch_is_still_a_duplicate():
    """The 4-hourly cadence re-fetches the same game repeatedly; unchanged
    snapshots must not accumulate."""
    s = _snap([8477492, 8478402], scratches=[8479318])
    assert U.lineup_fingerprint(s) == U.lineup_fingerprint(_snap(
        [8477492, 8478402], scratches=[8479318]))


def test_reordering_is_not_a_change():
    """The API may return rosterSpots in any order. Without canonicalisation a
    re-ordering would archive a duplicate every single cycle."""
    a = _snap([8477492, 8478402, 8479318])
    b = _snap([8479318, 8477492, 8478402])
    assert a["dressed"] != b["dressed"], "precondition: raw lists differ"
    assert U.lineup_fingerprint(a) == U.lineup_fingerprint(b)


def test_nested_reordering_is_not_a_change():
    a = {"railScratches": {"homeTeam": [1, 2, 3], "awayTeam": [9]}}
    b = {"railScratches": {"awayTeam": [9], "homeTeam": [3, 1, 2]}}
    assert U.lineup_fingerprint(a) == U.lineup_fingerprint(b)


# ---- shape robustness ------------------------------------------------------

def test_the_fingerprint_survives_shapes_we_have_not_seen_yet():
    """The pre-game payload shape is genuinely unknown until October — the
    module says so — so the key must not assume any particular structure."""
    for odd in ({}, {"k": None}, {"a": [[1, 2], [3]]}, {"x": {"y": {"z": [1]}}},
                {"mixed": [1, "two", None, {"three": 3}]}):
        assert isinstance(U.lineup_fingerprint(odd), str)


def test_the_fingerprint_is_deterministic_across_calls():
    s = _snap([8477492, 8478402], scratches=[8479318])
    assert len({U.lineup_fingerprint(s) for _ in range(5)}) == 1


def test_distinct_content_never_collides():
    seen = {}
    for n in range(1, 25):
        s = _snap(list(range(8470000, 8470000 + n)))
        fp = U.lineup_fingerprint(s)
        assert fp not in seen, f"collision between {n} and {seen[fp]} dressed players"
        seen[fp] = n
