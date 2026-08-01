"""The lines cache — the season's raw per-game data the model is replayed on.

`ensure_lines_cache` fetches one feed per completed game and swallowed any
failure. Unlike the finals walk this one self-heals (an uncached game_pk is
retried on the next run), so the severity is lower — but a game that fails
EVERY run is never cached, and its plate appearances, baserunning and bullpen
work stay permanently absent from the replay with nothing to say so.

The reported line was also unanswerable: "cache N games" cannot tell you whether
every completed game made it in, because the cache legitimately holds games from
outside the current finals window (25 such locally when this was written). The
question that matters is coverage of the CURRENT finals list, so that is what it
now prints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlbwp import predict_slate as PS  # noqa: E402


def _final(pk, date="2026-08-01"):
    return {"game_pk": pk, "date": date}


def _gd(pk):
    return {"home": "BOS", "away": "NYY", "pa": [{"pk": pk}], "baserun": {}}


def test_coverage_counts_the_finals_list_not_the_cache_size(monkeypatch, capsys, tmp_path):
    """The cache holds games outside the current finals window, so its size
    cannot answer 'is every completed game in the replay?'."""
    monkeypatch.setattr(PS.time, "sleep", lambda s: None)
    monkeypatch.setattr(PS, "game_data", _gd)
    cache_path = tmp_path / "lines.json"
    # two stale games from an earlier window, already cached
    cache_path.write_text(json.dumps({"900": {}, "901": {}}), encoding="utf-8")

    PS.ensure_lines_cache([_final(1), _final(2)], cache_path=cache_path)
    out = capsys.readouterr().out
    assert "replay coverage 2/2 finals" in out
    assert "cache 4 games" in out, "cache size still reported, but it is not coverage"


def test_a_failing_feed_warns_and_is_reported_as_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(PS.time, "sleep", lambda s: None)

    def flaky(pk):
        if pk == 2:
            raise ConnectionError("simulated bad feed")
        return _gd(pk)

    monkeypatch.setattr(PS, "game_data", flaky)
    cache_path = tmp_path / "lines.json"

    cache = PS.ensure_lines_cache([_final(1), _final(2), _final(3)], cache_path=cache_path)
    assert set(cache) == {"1", "3"}, "the reachable games must still cache"
    out = capsys.readouterr().out
    assert "replay coverage 2/3 finals" in out, "the gap must be visible in the count"
    assert "1 game feed(s) failed" in out
    assert "MISSING from the replay" in out
    assert "ConnectionError" in out


def test_a_failure_is_retried_on_the_next_run(monkeypatch, tmp_path):
    """Self-healing is what makes skipping acceptable here — pin it, because the
    warning's claim that it is 'retried next run' has to be true."""
    monkeypatch.setattr(PS.time, "sleep", lambda s: None)
    cache_path = tmp_path / "lines.json"

    calls = []

    def once_bad(pk):
        calls.append(pk)
        if pk == 2 and calls.count(2) == 1:
            raise TimeoutError("first attempt fails")
        return _gd(pk)

    monkeypatch.setattr(PS, "game_data", once_bad)
    finals = [_final(1), _final(2)]
    c1 = PS.ensure_lines_cache(finals, cache_path=cache_path)
    assert "2" not in c1
    c2 = PS.ensure_lines_cache(finals, cache_path=cache_path)
    assert "2" in c2, "an uncached game must be re-attempted"


def test_already_cached_games_are_never_refetched(monkeypatch, tmp_path):
    """'never re-fetched once cached' is the docstring's promise and the reason
    the daily run is cheap."""
    monkeypatch.setattr(PS.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(PS, "game_data", lambda pk: (calls.append(pk), _gd(pk))[1])
    cache_path = tmp_path / "lines.json"
    cache_path.write_text(json.dumps({"1": {"pre": "existing"}}), encoding="utf-8")

    cache = PS.ensure_lines_cache([_final(1), _final(2)], cache_path=cache_path)
    assert calls == [2], "a cached game must not be re-fetched"
    assert cache["1"] == {"pre": "existing"}, "and must not be overwritten"


def test_a_clean_run_prints_no_warning(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(PS.time, "sleep", lambda s: None)
    monkeypatch.setattr(PS, "game_data", _gd)
    PS.ensure_lines_cache([_final(1)], cache_path=tmp_path / "lines.json")
    assert "WARNING" not in capsys.readouterr().out


def test_nothing_is_written_when_nothing_was_fetched(monkeypatch, tmp_path):
    """A no-op run must not rewrite the file — it is committed by CI, and a
    rewrite would churn the diff on every cycle."""
    monkeypatch.setattr(PS.time, "sleep", lambda s: None)
    monkeypatch.setattr(PS, "game_data", _gd)
    cache_path = tmp_path / "lines.json"
    cache_path.write_text(json.dumps({"1": {"pre": "existing"}}), encoding="utf-8")
    before = cache_path.stat().st_mtime_ns

    PS.ensure_lines_cache([_final(1)], cache_path=cache_path)
    assert cache_path.stat().st_mtime_ns == before, "untouched run rewrote the cache"
