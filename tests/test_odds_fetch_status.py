"""Why the odds fetch returned nothing — the difference between a quiet market
and a dead pipeline.

`fetch_consensus` degrades to [] on every failure so the board build never
breaks. That is correct, but it collapsed five very different situations into
one indistinguishable empty list, and every caller printed the same line:
"0 games with >=20% EV vs opening consensus". A dead API key was reported in
exactly the words of a market with no value in it.

Found live on 2026-08-01: `data/opening_odds_2026.json` held 40 entries all
captured 2026-07-29T19:23:35Z, for games dated 2026-07-30, while the board
carried 36 pending picks — and refresh_odds.py completed in 0.56s, too fast to
have made three HTTP calls. The edge layer had been dead for ~3 days and
data/edge_ledger.json, the evidence stream the live-CLV promotion gate depends
on, held zero rows. Nothing anywhere said so.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market import odds  # noqa: E402


def _api_game(gid="g1", home="Boston Red Sox", away="New York Yankees",
              commence="2099-01-01T00:00:00Z", n_books=4):
    return {
        "id": gid, "home_team": home, "away_team": away, "commence_time": commence,
        "bookmakers": [
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": 1.80}, {"name": away, "price": 2.10}]}]}
            for _ in range(n_books)
        ],
    }


def _patch_http(monkeypatch, payload):
    class _R:
        def read(self): return __import__("json").dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(odds.urllib.request, "urlopen", lambda *a, **k: _R())


# ---- the five outcomes are distinguishable --------------------------------

def test_missing_key_says_so(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert odds.fetch_consensus() == []
    assert odds.LAST_FETCH["reason"] == "no_api_key"
    assert "ODDS_API_KEY is unset" in odds.fetch_status()


def test_a_failed_request_names_the_error(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("simulated")
    monkeypatch.setattr(odds.urllib.request, "urlopen", boom)
    assert odds.fetch_consensus(api_key="k") == []
    assert odds.LAST_FETCH["reason"] == "fetch_failed"
    s = odds.fetch_status()
    assert "the request failed" in s and "TimeoutError" in s
    assert "quota" in s, "the operator needs to know where to look"


def test_an_empty_response_is_not_the_same_as_a_failure(monkeypatch):
    """Out of quota often returns 200 with an empty list — that must not be
    reported as a healthy fetch."""
    _patch_http(monkeypatch, [])
    assert odds.fetch_consensus(api_key="k") == []
    assert odds.LAST_FETCH["reason"] == "empty_response"
    assert "zero games" in odds.fetch_status()


def test_games_returned_but_none_priced_is_its_own_reason(monkeypatch):
    """The dangerous quiet failure: the API is healthy and the key is fine, but
    a team-name map drift silently drops every game."""
    _patch_http(monkeypatch, [_api_game(home="Boston Red Sox FC")])   # unmapped
    assert odds.fetch_consensus(api_key="k") == []
    assert odds.LAST_FETCH["reason"] == "all_filtered"
    assert odds.LAST_FETCH["skipped"]["unmapped_team"] == 1
    s = odds.fetch_status()
    assert "none" in s and "team-name map drift" in s


def test_a_healthy_fetch_reports_how_many_it_priced(monkeypatch):
    _patch_http(monkeypatch, [_api_game()])
    out = odds.fetch_consensus(api_key="k")
    assert len(out) == 1
    assert odds.LAST_FETCH["reason"] == "ok"
    assert odds.fetch_status() == "ok: 1 priced of 1 returned"


# ---- the filters are attributed to the right bucket -----------------------

def test_a_started_game_is_counted_as_started_not_as_a_map_failure(monkeypatch):
    _patch_http(monkeypatch, [_api_game(commence="2000-01-01T00:00:00Z")])
    assert odds.fetch_consensus(api_key="k") == []
    assert odds.LAST_FETCH["skipped"]["already_started"] == 1
    assert odds.LAST_FETCH["skipped"]["unmapped_team"] == 0


def test_thin_books_are_counted_separately(monkeypatch):
    """Fewer than three books is a real consensus problem, not a key problem."""
    _patch_http(monkeypatch, [_api_game(n_books=2)])
    assert odds.fetch_consensus(api_key="k") == []
    assert odds.LAST_FETCH["skipped"]["too_few_books"] == 1


def test_an_unparseable_commence_time_does_not_crash(monkeypatch):
    _patch_http(monkeypatch, [_api_game(commence="not-a-time")])
    assert odds.fetch_consensus(api_key="k") == []
    assert odds.LAST_FETCH["reason"] == "all_filtered"


# ---- the contract callers rely on -----------------------------------------

def test_every_path_still_returns_a_list(monkeypatch):
    """The whole point of the swallow is that the board build never breaks. The
    diagnosis must not change that."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert isinstance(odds.fetch_consensus(), list)
    monkeypatch.setattr(odds.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert isinstance(odds.fetch_consensus(api_key="k"), list)


def test_the_refresh_prints_the_status_for_every_league():
    src = (ROOT / "refresh_odds.py").read_text(encoding="utf-8")
    assert src.count("odds.fetch_status()") == 3, (
        "each league's fetch is separate; one shared status would misreport two")
