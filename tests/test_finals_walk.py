"""The season finals walk — the most consequential silent failure in the chain.

`season_finals` makes one request per day across the season and swallowed any
failing day. That list is not a display artifact: it is the model's replay input
AND the ledger's grading source. A dropped day means the ratings walked without
those games (so every subsequent probability is computed from a wrong state) and
the missing games become picks that can never grade.

`refresh.py` then wrote the result straight over the cached file, so one bad
walk destroyed the good data it should have fallen back to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlbwp import live  # noqa: E402


# ---- the monotonicity guard ------------------------------------------------

def test_a_growing_walk_writes():
    assert live.finals_write_is_safe(1670, 1653) is True
    assert live.finals_write_is_safe(1653, 1653) is True, "no new finals is normal"


def test_a_shrunken_walk_is_blocked():
    """Completed games never un-complete. Fewer games means failed requests."""
    assert live.finals_write_is_safe(1652, 1653) is False
    assert live.finals_write_is_safe(0, 1653) is False, "a total outage must not wipe it"


def test_the_first_build_of_a_season_writes():
    """At the rollover the new season's cache does not exist yet, so the walk
    starts from zero and must not be blocked by its own emptiness."""
    assert live.finals_write_is_safe(0, 0) is True
    assert live.finals_write_is_safe(30, 0) is True


def test_the_block_is_self_healing():
    """A block keeps the good file for one cycle; once more games complete the
    next walk clears the bar without intervention. Pin that the guard is a
    comparison and not a latch."""
    assert live.finals_write_is_safe(1652, 1653) is False   # blocked
    assert live.finals_write_is_safe(1668, 1653) is True    # next cycle, unblocked


# ---- the failing-day walk, actually executed ------------------------------

def _game(date_, home_win=True):
    return {"date": date_, "home_win": home_win}


def test_a_failing_day_warns_instead_of_vanishing(monkeypatch, capsys):
    """The whole point: a dropped day used to leave no trace anywhere."""
    monkeypatch.setattr(live.time, "sleep", lambda s: None)

    def flaky(day, probables=False):
        if day == "2026-03-16":
            raise ConnectionError("simulated blip")
        return [_game(day)]

    monkeypatch.setattr(live, "schedule", flaky)
    out = live.season_finals(2026, start_md="03-15", end_date="2026-03-17")

    assert len(out) == 2, "the reachable days must still be collected"
    err = capsys.readouterr().out
    assert "1 day(s) failed" in err
    assert "2026-03-16" in err
    assert "cannot grade" in err, "the warning must say what the consequence is"


def test_a_clean_walk_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    monkeypatch.setattr(live, "schedule", lambda day, probables=False: [_game(day)])
    out = live.season_finals(2026, start_md="03-15", end_date="2026-03-17")
    assert len(out) == 3
    assert "failed" not in capsys.readouterr().out, "must not cry wolf"


def test_results_stay_sorted_by_date(monkeypatch):
    """The replay walks these in order; an unsorted list would apply games out
    of sequence and corrupt every rating."""
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    monkeypatch.setattr(live, "schedule", lambda day, probables=False: [_game(day)])
    out = live.season_finals(2026, start_md="03-15", end_date="2026-03-20")
    assert [g["date"] for g in out] == sorted(g["date"] for g in out)


def test_only_completed_games_are_collected(monkeypatch):
    """A scheduled-but-unplayed game has no home_win and must not enter the
    replay as if it were a result."""
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    monkeypatch.setattr(live, "schedule", lambda day, probables=False: [
        _game(day), {"date": day},                      # no home_win -> not final
    ])
    out = live.season_finals(2026, start_md="03-15", end_date="2026-03-16")
    assert len(out) == 2 and all("home_win" in g for g in out)


# ---- duplicates: one game, counted once -----------------------------------

def _susp(pk, date_, score=(7, 2)):
    return {"game_pk": pk, "date": date_, "home_win": 0.0,
            "away_score": score[0], "home_score": score[1]}


def test_a_suspended_game_listed_on_two_dates_is_kept_once():
    """Found live: game_pk 824912 (SF @ ATL) appeared as both 2026-06-16 and
    2026-06-17 — same pk, same 7-2 final. The day-by-day walk sees a suspended
    or postponed game under its original date AND its completion date, so the
    season's finals list held 1653 entries for 1652 distinct games.

    The replay walks this list in order, so that result was applied to both
    teams' Elo twice: SF ended 1.82 too high and ATL 1.77 too low, with drift
    reaching 18 of 33 teams through subsequent games.
    """
    out = live.dedupe_finals([_susp(824912, "2026-06-16"), _susp(824912, "2026-06-17")])
    assert len(out) == 1
    assert out[0]["date"] == "2026-06-17", (
        "the later date wins — that is when the game concluded, which is when "
        "its result can legitimately enter a leak-safe replay")


def test_distinct_games_are_never_merged():
    games = [_susp(1, "2026-06-16"), _susp(2, "2026-06-16"), _susp(3, "2026-06-17")]
    assert len(live.dedupe_finals(games)) == 3


def test_a_doubleheader_keeps_both_games():
    """Two games, same teams, same day — different game_pks, so both are real."""
    out = live.dedupe_finals([_susp(10, "2026-06-16"), _susp(11, "2026-06-16")])
    assert len(out) == 2


def test_games_without_an_id_are_never_collapsed_together():
    """Keying on a missing game_pk would make every unidentified game the same
    game. Caught by the existing walk tests, whose fixtures carry no pk."""
    out = live.dedupe_finals([{"date": "2026-06-16"}, {"date": "2026-06-17"},
                              {"date": "2026-06-18"}])
    assert len(out) == 3


def test_scores_survive_deduplication():
    out = live.dedupe_finals([_susp(5, "2026-06-16"), _susp(5, "2026-06-17")])
    assert out[0]["away_score"] == 7 and out[0]["home_score"] == 2
    assert out[0]["home_win"] == 0.0


def test_the_walk_itself_emits_no_duplicates(monkeypatch):
    """Integration: a schedule endpoint that reports the same game on two
    consecutive days must still yield one entry."""
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    monkeypatch.setattr(live, "schedule",
                        lambda day, probables=False: [_susp(824912, day)])
    out = live.season_finals(2026, start_md="06-16", end_date="2026-06-17")
    assert len(out) == 1 and out[0]["date"] == "2026-06-17"


def test_the_shipped_finals_file_has_no_duplicates():
    """Contract against the real artifact: if this fails the model is currently
    replaying some game more than once."""
    import collections

    from mlbwp import season_paths
    p = season_paths.finals_cache()
    if not p.is_file():
        import pytest
        pytest.skip("finals cache not present")
    fin = live.dedupe_finals(json.loads(p.read_text(encoding="utf-8")))
    c = collections.Counter(str(g["game_pk"]) for g in fin)
    assert not [k for k, v in c.items() if v > 1]


# ---- the caller ------------------------------------------------------------

def test_refresh_keeps_the_cached_finals_and_continues():
    """refresh.py must not abort on a blocked write. Everything downstream reads
    this file, so continuing rebuilds from the last good finals — a board a few
    hours stale but correct, rather than no board at all."""
    src = (ROOT / "refresh.py").read_text(encoding="utf-8")
    assert "if finals_write_is_safe(" in src, "the guard is not wired in"
    block = src.split("if finals_write_is_safe(", 1)[1].split("predict_slate.main", 1)[0]
    # statement-level, not substring: the warning text itself contains "returned"
    stmts = [ln.split("#")[0].strip() for ln in block.splitlines()]
    offenders = [s for s in stmts if s.startswith(("return", "raise", "sys.exit"))]
    assert not offenders, (
        f"aborting here skips the board, the NHL serve and the site rebuild: {offenders}")
