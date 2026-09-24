"""The NHL pre-game freeze keeps the track record honest once nhl_serve runs
every 4 hours in-season: a played game shows the number published before puck
drop, never a later replay."""
import sys

sys.path.insert(0, "phase0")
from nhl_hp_freeze import freeze  # noqa: E402

S = 20262027


def _row(i, hp, hs=None, as_=None):
    return {"id": i, "hp": hp, "hs": hs, "as": as_, "ct": {"elo": 1.0}}


def test_unplayed_rows_refresh_their_entry_with_time_and_tier():
    sched = [_row(2026020001, 0.61)]
    led, nf, nr = freeze(sched, {}, "2026-09-28T12:00:00Z", S)
    assert led["2026020001"] == {"hp": 0.61, "ct": {"elo": 1.0},
                                 "t": "2026-09-28T12:00:00Z", "tier": "EARLY"}
    assert (nf, nr) == (0, 0)


def test_played_rows_are_restored_to_the_pre_game_number():
    led = {"2026020001": {"hp": 0.61, "ct": {"elo": 1.0}, "t": "T0", "tier": "EARLY"}}
    sched = [_row(2026020001, 0.70, 3, 2)]          # the replay says 0.70
    led, nf, nr = freeze(sched, led, "T1", S)
    assert sched[0]["hp"] == 0.61 and sched[0]["frozen_at"] == "T0"
    assert led["2026020001"]["t"] == "T0"           # the entry is not overwritten
    assert (nf, nr) == (1, 0)


def test_a_played_row_with_no_entry_is_marked_replay_not_passed_off():
    warned = []
    sched = [_row(2026020002, 0.55, 1, 4)]
    freeze(sched, {}, "T", S, warn=warned.append)
    assert sched[0]["replay"] is True and warned


def test_earlier_seasons_are_left_alone():
    sched = [_row(2025020001, 0.52, 2, 1)]
    led, nf, nr = freeze(sched, {}, "T", S)
    assert led == {} and "replay" not in sched[0] and (nf, nr) == (0, 0)
