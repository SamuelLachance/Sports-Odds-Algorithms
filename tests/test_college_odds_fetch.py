"""CFB/CBB odds fetch must keep blowout spreads and not invent juice."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_odds_espn import MAX_NBA_SPREAD, _consensus  # noqa: E402


def _blowout_item(spread: float = -45.5) -> dict:
    return {
        "provider": {"name": "consensus"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": spread},
                "moneyLine": {"american": -5000},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": -spread},
                "moneyLine": {"american": 1800},
                "spread": {"american": -110},
            },
        },
        "spread": spread,
    }


def test_nba_consensus_default_drops_cfb_sized_blowout() -> None:
    consensus = _consensus([_blowout_item(-45.5)])
    assert consensus["home_close_spread"] is None
    assert consensus["home_close_ml"] == -5000.0


def test_cfb_consensus_keeps_blowout_spread() -> None:
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    consensus = _consensus([_blowout_item(-45.5)], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] == -45.5
    assert consensus["away_close_spread"] == 45.5
    assert MAX_CFB_SPREAD > MAX_NBA_SPREAD


def test_cbb_consensus_keeps_blowout_spread() -> None:
    from scripts.fetch_cbb_odds import MAX_CBB_SPREAD

    consensus = _consensus([_blowout_item(-45.5)], max_handicap_abs=MAX_CBB_SPREAD)
    assert consensus["home_close_spread"] == -45.5
    assert MAX_CBB_SPREAD > MAX_NBA_SPREAD


def test_cfb_odds_row_does_not_invent_spread_juice() -> None:
    """Missing juice must stay None — never fake -110."""
    from scripts import fetch_cfb_odds as mod

    event = {
        "date": "2024-09-01",
        "home_key": "ala",
        "away_key": "uga",
        "event": "1",
        "comp": "1",
        "home_final": 28,
        "away_final": 21,
    }
    payload = {
        "items": [
            {
                "provider": {"name": "book"},
                "homeTeamOdds": {
                    "favorite": True,
                    "close": {
                        "pointSpread": {"american": -7.5},
                        "moneyLine": {"american": -300},
                        # spread juice omitted
                    },
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "close": {
                        "pointSpread": {"american": 7.5},
                        "moneyLine": {"american": 250},
                    },
                },
                "spread": -7.5,
            }
        ]
    }
    with patch.object(mod, "_throttled_get", return_value=payload):
        row = mod._odds_row(event)
    assert row["home_close_spread"] == -7.5
    assert row["home_spread_odds"] is None
    assert row["away_spread_odds"] is None
    assert row["home_close_ml"] == -300.0


def test_cfb_provider_rejects_juice_sized_point_spread() -> None:
    """CFB max_abs=120 must not keep −110 juice dumped into pointSpread."""
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    item = {
        "provider": {"name": "book"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": -110},  # juice dump
                "moneyLine": {"american": -300},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": 110},
                "moneyLine": {"american": 250},
                "spread": {"american": -110},
            },
        },
        "spread": -7.5,
    }
    consensus = _consensus([item], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] == -7.5
    assert consensus["away_close_spread"] == 7.5
    assert consensus["home_close_ml"] == -300.0


def test_cfb_provider_rejects_juice_point_spread_without_flat_spread() -> None:
    """Missing flat `spread` must still drop |pointSpread| >= 100 juice dumps."""
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    item = {
        "provider": {"name": "book"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": -110},
                "moneyLine": {"american": -200},
                "spread": {"american": -110},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": 110},
                "moneyLine": {"american": 170},
                "spread": {"american": -110},
            },
        },
        # no top-level "spread"
    }
    consensus = _consensus([item], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] is None
    assert consensus["away_close_spread"] is None
    assert consensus["home_close_ml"] == -200.0


def test_cfb_provider_rejects_juice_in_flat_spread_fallback() -> None:
    """CFB must not reintroduce −110 via flat `spread` after juice-clearing nested fields."""
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    item = {
        "provider": {"name": "book"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "pointSpread": {"american": -110},
                "moneyLine": {"american": -200},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": 110},
                "moneyLine": {"american": 170},
            },
        },
        "spread": -110,  # juice also dumped into flat field
    }
    consensus = _consensus([item], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] is None
    assert consensus["away_close_spread"] is None


def test_cfb_provider_rejects_away_only_juice_point_spread() -> None:
    """Away-only juice nested field must not leave asymmetric away_close_spread=110."""
    from scripts.fetch_cfb_odds import MAX_CFB_SPREAD

    item = {
        "provider": {"name": "book"},
        "homeTeamOdds": {
            "favorite": True,
            "close": {
                "moneyLine": {"american": -200},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "close": {
                "pointSpread": {"american": 110},  # juice only on away
                "moneyLine": {"american": 170},
            },
        },
        "spread": -7.5,
    }
    consensus = _consensus([item], max_handicap_abs=MAX_CFB_SPREAD)
    assert consensus["home_close_spread"] == -7.5
    assert consensus["away_close_spread"] == 7.5


def test_cbb_odds_row_excludes_live_books_from_consensus() -> None:
    """Live/in-game books must not pollute CBB closing consensus."""
    from scripts import fetch_cbb_odds as mod

    event = {
        "date": "2024-01-15",
        "home_key": "duke",
        "away_key": "unc",
        "event": "1",
        "comp": "1",
        "home_final": 80,
        "away_final": 72,
    }
    payload = {
        "items": [
            {
                "provider": {"name": "DraftKings"},
                "homeTeamOdds": {
                    "favorite": True,
                    "close": {
                        "pointSpread": {"american": -5.5},
                        "moneyLine": {"american": -220},
                        "spread": {"american": -110},
                    },
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "close": {
                        "pointSpread": {"american": 5.5},
                        "moneyLine": {"american": 180},
                        "spread": {"american": -110},
                    },
                },
                "spread": -5.5,
            },
            {
                "provider": {"name": "DraftKings Live"},
                "homeTeamOdds": {
                    "favorite": True,
                    "close": {
                        "pointSpread": {"american": -12.5},
                        "moneyLine": {"american": -500},
                        "spread": {"american": -115},
                    },
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "close": {
                        "pointSpread": {"american": 12.5},
                        "moneyLine": {"american": 375},
                        "spread": {"american": -105},
                    },
                },
                "spread": -12.5,
            },
        ]
    }
    with patch.object(mod, "_throttled_get", return_value=payload):
        row = mod._odds_row(event)
    assert row["home_close_spread"] == -5.5
    assert row["n_books"] == 1


def test_cbb_merge_rows_keeps_prior_closes_when_fresh_empty() -> None:
    """Fresh n_books=0 shells must not wipe existing closing lines."""
    from scripts.fetch_cbb_odds import merge_rows

    existing = [
        {
            "date": "2024-01-15",
            "home_key": "duke",
            "away_key": "unc",
            "home_close_ml": "-150",
            "away_close_ml": "130",
            "home_close_spread": "-5.5",
            "away_close_spread": "5.5",
            "n_books": "2",
            "source": "espn-core",
        }
    ]
    fresh = [
        {
            "date": "2024-01-15",
            "home_key": "duke",
            "away_key": "unc",
            "home_close_ml": None,
            "away_close_ml": None,
            "home_close_spread": None,
            "away_close_spread": None,
            "n_books": 0,
            "source": "espn-core",
        }
    ]
    merged = merge_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["home_close_ml"] == "-150"
    assert merged[0]["n_books"] == "2"


def test_cfb_collect_day_rows_refetches_empty_odds_cache(tmp_path, monkeypatch) -> None:
    """Cached n_books=0 rows must not permanently block a later successful fetch."""
    import json
    from datetime import date
    from unittest.mock import patch

    from scripts import fetch_cfb_odds as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    day = date(2024, 9, 1)
    cache_file = tmp_path / f"{day.strftime('%Y%m%d')}.json"
    cache_file.write_text(
        json.dumps(
            [
                {
                    "date": "2024-09-01",
                    "home_key": "ala",
                    "away_key": "uga",
                    "n_books": 0,
                    "home_close_ml": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    filled = [
        {
            "date": "2024-09-01",
            "home_key": "ala",
            "away_key": "uga",
            "n_books": 2,
            "home_close_ml": -200,
        }
    ]
    with patch.object(mod, "_iter_completed_events", return_value=[{"id": "1"}]):
        with patch.object(mod, "_odds_row", return_value=filled[0]):
            rows = mod.collect_day_rows(day, use_cache=True)
    assert rows[0]["n_books"] == 2
    assert rows[0]["home_close_ml"] == -200


def test_cfb_collect_day_rows_refetches_partial_day_cache(tmp_path, monkeypatch) -> None:
    """One filled book row must not freeze siblings still at n_books=0."""
    import json
    from datetime import date
    from unittest.mock import patch

    from scripts import fetch_cfb_odds as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    day = date(2024, 9, 3)
    cache_file = tmp_path / f"{day.strftime('%Y%m%d')}.json"
    cache_file.write_text(
        json.dumps(
            [
                {
                    "date": "2024-09-03",
                    "home_key": "ala",
                    "away_key": "uga",
                    "n_books": 2,
                    "home_close_ml": -200,
                },
                {
                    "date": "2024-09-03",
                    "home_key": "osu",
                    "away_key": "mich",
                    "n_books": 0,
                    "home_close_ml": None,
                },
            ]
        ),
        encoding="utf-8",
    )
    filled = [
        {
            "date": "2024-09-03",
            "home_key": "ala",
            "away_key": "uga",
            "n_books": 2,
            "home_close_ml": -200,
        },
        {
            "date": "2024-09-03",
            "home_key": "osu",
            "away_key": "mich",
            "n_books": 3,
            "home_close_ml": -140,
        },
    ]
    with patch.object(mod, "_iter_completed_events", return_value=[{"id": "1"}, {"id": "2"}]):
        with patch.object(mod, "_odds_row", side_effect=filled):
            rows = mod.collect_day_rows(day, use_cache=True)
    assert len(rows) == 2
    assert rows[1]["n_books"] == 3
    assert rows[1]["home_close_ml"] == -140


def test_cfb_collect_day_rows_refetches_sticky_empty_list(tmp_path, monkeypatch) -> None:
    """Cached [] from a scoreboard outage / mid-day empty slate must not stick."""
    from datetime import date
    from unittest.mock import patch

    from scripts import fetch_cfb_odds as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    day = date(2024, 9, 2)
    cache_file = tmp_path / f"{day.strftime('%Y%m%d')}.json"
    cache_file.write_text("[]", encoding="utf-8")
    filled = {
        "date": "2024-09-02",
        "home_key": "osu",
        "away_key": "mich",
        "n_books": 3,
        "home_close_ml": -140,
    }
    with patch.object(mod, "_iter_completed_events", return_value=[{"id": "2"}]):
        with patch.object(mod, "_odds_row", return_value=filled):
            rows = mod.collect_day_rows(day, use_cache=True)
    assert len(rows) == 1
    assert rows[0]["n_books"] == 3
    assert rows[0]["home_close_ml"] == -140


def test_nba_collect_day_rows_refetches_empty_odds_cache(tmp_path, monkeypatch) -> None:
    """Sticky [] after an odds outage must not permanently skip a later refill."""
    from datetime import date
    from unittest.mock import patch

    from web import nba_odds_espn as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    day = date(2024, 1, 15)
    cache_file = tmp_path / f"{day.strftime('%Y%m%d')}.json"
    cache_file.write_text("[]", encoding="utf-8")
    event = {
        "date": "2024-01-15",
        "event": "1",
        "comp": "1",
        "home_key": "bos",
        "away_key": "nyk",
        "home_final": 110,
        "away_final": 105,
    }
    filled = {
        **mod._empty_odds_row(event),
        "home_close_ml": -150,
        "away_close_ml": 130,
        "home_close_spread": -3.5,
        "away_close_spread": 3.5,
        "n_books": 3,
    }
    with patch.object(mod, "_iter_completed_events", return_value=[event]):
        with patch("web.nba_odds_espn.ThreadPoolExecutor") as pool_cls:
            pool = pool_cls.return_value.__enter__.return_value
            pool.map.return_value = [filled]
            rows = mod.collect_day_rows(day, use_cache=True)
    assert len(rows) == 1
    assert rows[0]["n_books"] == 3
    assert rows[0]["home_close_ml"] == -150
    # Successful refill must be trusted on the next call.
    rows2 = mod.collect_day_rows(day, use_cache=True)
    assert rows2[0]["n_books"] == 3


def test_nba_collect_day_rows_refetches_all_zero_books_cache(tmp_path, monkeypatch) -> None:
    """Cached n_books=0 stubs must not permanently block a later successful fetch."""
    import json
    from datetime import date
    from unittest.mock import patch

    from web import nba_odds_espn as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    day = date(2024, 1, 16)
    event = {
        "date": "2024-01-16",
        "event": "2",
        "comp": "2",
        "home_key": "mia",
        "away_key": "chi",
        "home_final": 100,
        "away_final": 98,
    }
    cache_file = tmp_path / f"{day.strftime('%Y%m%d')}.json"
    cache_file.write_text(json.dumps([mod._empty_odds_row(event)]), encoding="utf-8")
    filled = {
        **mod._empty_odds_row(event),
        "home_close_ml": -120,
        "away_close_ml": 100,
        "n_books": 2,
    }
    with patch.object(mod, "_iter_completed_events", return_value=[event]):
        with patch("web.nba_odds_espn.ThreadPoolExecutor") as pool_cls:
            pool = pool_cls.return_value.__enter__.return_value
            pool.map.return_value = [filled]
            rows = mod.collect_day_rows(day, use_cache=True)
    assert rows[0]["n_books"] == 2


def test_cbb_scoreboard_always_live_fetches(tmp_path, monkeypatch) -> None:
    """Truncated mid-day scoreboard must not stick; later runs see new games."""
    from datetime import date
    from unittest.mock import patch

    from scripts import fetch_cbb_odds as mod

    day = date(2024, 1, 20)

    def _event(eid: str, home: str, away: str) -> dict:
        return {
            "id": eid,
            "competitions": [
                {
                    "id": eid,
                    "status": {"type": {"completed": True}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "70",
                            "team": {"abbreviation": home},
                        },
                        {
                            "homeAway": "away",
                            "score": "65",
                            "team": {"abbreviation": away},
                        },
                    ],
                }
            ],
        }

    payloads = [
        {"events": [_event("1", "DUKE", "UNC")]},
        {"events": [_event("1", "DUKE", "UNC"), _event("2", "KU", "UK")]},
    ]
    with patch.object(mod, "_throttled_get", side_effect=payloads) as fetch:
        first = list(mod._iter_completed_events(day))
        second = list(mod._iter_completed_events(day))
    assert len(first) == 1
    assert len(second) == 2
    assert fetch.call_count == 2
