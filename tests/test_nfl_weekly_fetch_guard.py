"""The NFL weekly fetch guard — protecting the spine the model replays from.

`data/nfl_games.csv` is a REQUIRED full rewrite from nflverse on every weekly
run, and the whole NFL model replays from it. The only validation was byte size
(> 1000), which a truncated response passes easily: a partial CDN read can be
megabytes and still be short thousands of rows.

A stale-spine tripwire does exist downstream — nfl_season_guards.v7_npy_error
asks "was data/nfl_games.csv rolled back?" — but it fires at SERVE time, after
the short file has been written and committed. By then the good copy is gone.
Refusing the write is what keeps it.

These tables are append-mostly: completed games and logged injuries do not
un-happen. A small shrink is an upstream correction; a large one is truncation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import nfl_weekly as W  # noqa: E402


# ---- row counting ----------------------------------------------------------

def test_rows_excludes_the_header():
    assert W.csv_rows(b"h1,h2\na,b\nc,d\n") == 2
    assert W.csv_rows(b"h1,h2\na,b\nc,d") == 2, "a missing trailing newline still counts"


def test_a_header_only_table_has_no_rows():
    assert W.csv_rows(b"game_id,season,week\n") == 0


def test_unparseable_input_is_minus_one_not_zero():
    """-1 means 'not a countable table', which the guard must treat as unknown
    rather than as an empty file — otherwise a non-CSV would look like a total
    truncation and block every future fetch."""
    assert W.csv_rows(b"") == -1
    assert W.csv_rows(b"no newlines at all") == -1
    assert W.csv_rows(None) == -1


# ---- the guard -------------------------------------------------------------

def test_growth_and_steady_state_are_allowed():
    assert W.fetch_is_safe(7549, 7548) is True
    assert W.fetch_is_safe(7548, 7548) is True, "an unchanged table is normal"
    assert W.fetch_is_safe(9000, 7548) is True, "a new season's schedule lands at once"


def test_a_truncated_response_is_refused():
    """The case byte size cannot catch: megabytes of valid CSV, thousands of
    rows short."""
    assert W.fetch_is_safe(4000, 7548) is False
    assert W.fetch_is_safe(0, 7548) is False, "a header-only response must not win"


def test_a_small_upstream_correction_still_writes():
    """nflverse does occasionally remove a row. Blocking that would wedge the
    fetch until the next season's schedule arrived."""
    assert W.fetch_is_safe(7545, 7548) is True
    assert W.fetch_is_safe(7543, 7548) is True     # exactly the tolerance
    assert W.fetch_is_safe(7542, 7548) is False    # one past it


def test_the_first_fetch_is_never_blocked():
    """2026 files do not exist until nflverse publishes them mid-September."""
    assert W.fetch_is_safe(500, 0) is True
    assert W.fetch_is_safe(0, 0) is True


def test_an_uncountable_new_file_is_not_blocked():
    """csv_rows returning -1 means we could not count it, which is not evidence
    of truncation — the byte-size check remains the gate for that case."""
    assert W.fetch_is_safe(-1, 7548) is True


# ---- the wiring ------------------------------------------------------------

def test_the_guard_skips_rather_than_raising():
    """One bad upstream table must not stop the rest of the weekly chain; the
    other four fetches and the payload steps still need to run. Same rule as the
    MLB finals and db guards."""
    import inspect
    src = inspect.getsource(W.fetch)
    block = src.split("if not fetch_is_safe", 1)[1].split("with open(tmp", 1)[0]
    code = "\n".join(ln.split("#")[0] for ln in block.splitlines())
    assert "return" in code, "the guard must skip the write"
    assert "raise" not in code, "raising here would abort the remaining fetches"


def test_the_existing_file_is_untouched_when_refused(tmp_path, monkeypatch, capsys):
    """The whole point: the good spine must survive a truncated fetch."""
    target = tmp_path / "nfl_games.csv"
    # rows long enough that the SHORT version still clears the >1000-byte
    # check — otherwise the old size guard catches it and this never exercises
    # the row guard at all (my first fixture made exactly that mistake).
    row = b"2020_01_AAA_BBB,2020,REG,1,2020-09-10,Thu,20:20,AAA,17,BBB,24,home\n"
    good = b"game_id,season,game_type,week,gameday,weekday,gametime,away,as,home,hs,loc\n" \
           + row * 2000
    target.write_bytes(good)

    short = b"game_id,season,game_type,week,gameday,weekday,gametime,away,as,home,hs,loc\n" \
            + row * 50
    assert len(short) > 1000, "fixture must get past the byte check to reach the row guard"

    class _R:
        def read(self): return short
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(W.urllib.request, "urlopen", lambda *a, **k: _R())

    W.fetch(str(target), "https://example.invalid/games.csv", True)

    assert target.read_bytes() == good, "a truncated fetch overwrote the good spine"
    assert not (tmp_path / "nfl_games.csv.tmp").exists(), "temp file left behind"
    assert "REFUSED" in capsys.readouterr().out


def test_a_healthy_fetch_still_writes(tmp_path, monkeypatch):
    target = tmp_path / "inj_2026.csv"
    target.write_bytes(b"h,x\n" + b"".join(b"a,1\n" for _ in range(500)))
    bigger = b"h,x\n" + b"".join(b"a,1\n" for _ in range(900))

    class _R:
        def read(self): return bigger
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(W.urllib.request, "urlopen", lambda *a, **k: _R())

    W.fetch(str(target), "https://example.invalid/inj.csv", False)
    assert target.read_bytes() == bigger
