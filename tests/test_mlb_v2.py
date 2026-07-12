"""MLB v2 live / schedule defensive tests."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    """Corrupt on-disk JSON must not raise — live layer returns None instead."""
    import web.mlb_v2.live as live

    model_dir = tmp_path / "mlb_v2"
    model_dir.mkdir()
    for name in ("model_clf.json", "model_lr.json", "calibrator.json", "metadata.json"):
        (model_dir / name).write_text("{not-json", encoding="utf-8")
    with gzip.open(model_dir / "state_2024.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    monkeypatch.setattr(live, "MODEL_DIR", model_dir)
    live._load_artifacts.cache_clear()
    assert live.artifacts_available() is True
    assert live._load_artifacts() is None


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path) -> None:
    import web.mlb_v2.live as live

    bad = tmp_path / "state_2024.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2024: bad}}
    assert live._load_snapshot_state(art, 2025) is None


def test_prefer_todays_game_picks_non_final_doubleheader() -> None:
    import web.mlb_v2.live as live

    game1 = {
        "gamePk": 1,
        "home_id": 147,
        "away_id": 111,
        "status": "F",
        "game_number": 1,
        "home_pp_id": 100,
    }
    game2 = {
        "gamePk": 2,
        "home_id": 147,
        "away_id": 111,
        "status": "S",
        "game_number": 2,
        "home_pp_id": 200,
    }
    assert live._prefer_todays_game(game1, game2)["gamePk"] == 2
    assert live._prefer_todays_game(game2, game1)["gamePk"] == 2
    # Both pending: prefer later game_number (game 2 of DH).
    pending1 = {**game1, "status": "S"}
    assert live._prefer_todays_game(pending1, game2)["gamePk"] == 2


def test_fetch_season_bundle_flags_stale_on_network_failure(tmp_path, monkeypatch) -> None:
    import time

    import web.mlb_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)
    path = live._cache_path("bundle", 2024, "2024-10-01")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"games": [{"gamePk": 1}], "pitchers": {}, "team_hitting": {}, "team_pitching": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    aged = time.time() - (live.LIVE_CACHE_TTL_SECONDS + 60)
    import os

    os.utime(path, (aged, aged))

    def _boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(live, "fetch_season_games", _boom)
    bundle, stale = live._fetch_current_season_bundle(2024, "2024-10-01")
    assert stale is True
    assert bundle == payload


def test_games_last7_excludes_same_day_and_future() -> None:
    from web.mlb_v2.feature_engine import MlbFeatureEngine

    engine = MlbFeatureEngine()
    team = SimpleNamespace(
        recent_dates=["2026-07-05", "2026-07-08", "2026-07-11", "2026-07-12", "2026-07-13"]
    )
    # Window is (2026-07-05, 2026-07-12) exclusive of game_date → 07-05, 07-08, 07-11
    assert engine._games_last7(team, "2026-07-12") == 3.0
    assert engine._games_last7(team, "bad-date") == 0.0


def test_parse_schedule_games_includes_playoff_type() -> None:
    from web.mlb_v2.statsapi_data import _parse_schedule_games

    payload = {
        "dates": [
            {
                "date": "2024-10-05",
                "games": [
                    {
                        "gamePk": 99,
                        "officialDate": "2024-10-05",
                        "gameType": "F",
                        "status": {"codedGameState": "S"},
                        "dayNight": "night",
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "venue": {"id": 1},
                        "teams": {
                            "home": {
                                "team": {"id": 147},
                                "probablePitcher": {"id": 1, "fullName": "A"},
                            },
                            "away": {
                                "team": {"id": 111},
                                "probablePitcher": {"id": 2, "fullName": "B"},
                            },
                        },
                    }
                ],
            }
        ]
    }
    rows = _parse_schedule_games(payload, include_names=True)
    assert len(rows) == 1
    assert rows[0]["game_type"] == "F"
    assert rows[0]["home_pp_name"] == "A"


def test_fetch_season_games_merges_regular_and_playoff(monkeypatch) -> None:
    from web.mlb_v2 import statsapi_data as api

    def fake_get_json(url: str, **_k):
        if "gameType=R" in url:
            return {
                "dates": [
                    {
                        "date": "2024-04-01",
                        "games": [
                            {
                                "gamePk": 1,
                                "officialDate": "2024-04-01",
                                "gameType": "R",
                                "status": {"codedGameState": "F"},
                                "teams": {
                                    "home": {"team": {"id": 147}},
                                    "away": {"team": {"id": 111}},
                                },
                            }
                        ],
                    }
                ]
            }
        if "gameType=F" in url:
            return {
                "dates": [
                    {
                        "date": "2024-10-01",
                        "games": [
                            {
                                "gamePk": 2,
                                "officialDate": "2024-10-01",
                                "gameType": "F",
                                "status": {"codedGameState": "S"},
                                "teams": {
                                    "home": {"team": {"id": 147}},
                                    "away": {"team": {"id": 111}},
                                },
                            }
                        ],
                    }
                ]
            }
        raise OSError("no games")

    monkeypatch.setattr(api, "get_json", fake_get_json)
    games = api.fetch_season_games(2024)
    pks = {g["gamePk"] for g in games}
    assert pks == {1, 2}
    assert any(g["game_type"] == "F" for g in games)
