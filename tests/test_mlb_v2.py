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
    # Both pending: prefer earlier game_number (game 1 of DH).
    pending1 = {**game1, "status": "S"}
    assert live._prefer_todays_game(pending1, game2)["gamePk"] == 1
    assert live._prefer_todays_game(game2, pending1)["gamePk"] == 1
    # Game Over (O) is terminal for preference, same as Final (F).
    over1 = {**game1, "status": "O"}
    assert live._prefer_todays_game(over1, game2)["gamePk"] == 2


def test_is_final_game_accepts_game_over_status() -> None:
    from web.mlb_v2.replay import is_final_game

    base = {"home_score": 5, "away_score": 3}
    assert is_final_game({**base, "status": "F"})
    assert is_final_game({**base, "status": "O"})
    assert not is_final_game({**base, "status": "S"})
    assert not is_final_game({"status": "O", "home_score": 5})  # missing away
    assert not is_final_game({**base, "status": "C"})  # cancelled ≠ completed
    assert not is_final_game({**base, "status": "D"})  # postponed ≠ completed


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


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    """Hard-missing intermediate seasons must not silently skip Elo/form state."""
    from unittest.mock import MagicMock

    import web.mlb_v2.live as live

    live.get_live_context.cache_clear()
    art = {
        "snapshots": {2023: MagicMock()},
        "feature_columns": [],
        "clf": None,
        "lr": {},
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "runs_home": None,
        "runs_away": None,
    }
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}, "pitchers": {}, "venues": {}}),
    )
    monkeypatch.setattr(
        live.MlbFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )

    calls: list[int] = []

    def fake_bundle(season: int, _day: str):
        calls.append(season)
        if season == 2024:
            return None, False
        return (
            {
                "games": [],
                "pitchers": {},
                "team_hitting": {},
                "team_pitching": {},
            },
            False,
        )

    monkeypatch.setattr(live, "_fetch_current_season_bundle", fake_bundle)
    assert live.get_live_context("2025-07-12") is None
    assert 2024 in calls


def test_predict_matchup_applies_earlier_dh_final(monkeypatch) -> None:
    """Game-2 prediction must fold same-day game-1 final into a cloned engine."""
    from unittest.mock import MagicMock

    import web.mlb_v2.live as live

    home_id = 147  # NYY
    away_id = 111  # BOS
    game1 = {
        "gamePk": 1,
        "date": "2026-07-12",
        "status": "O",  # Game Over — must count as final for same-day DH updates
        "game_number": 1,
        "home_id": home_id,
        "away_id": away_id,
        "home_score": 5,
        "away_score": 2,
        "home_pp_id": 100,
        "away_pp_id": 200,
    }
    game2 = {
        "gamePk": 2,
        "date": "2026-07-12",
        "status": "S",
        "game_number": 2,
        "home_id": home_id,
        "away_id": away_id,
        "home_pp_id": 101,
        "away_pp_id": 201,
        "day_night": "night",
        "double_header": "Y",
        "venue_id": 1,
    }

    engine = MagicMock()
    engine.to_dict.return_value = {"cloned": True}
    engine.features_for_game.return_value = {
        "home_sp_fip_blend": 3.5,
        "away_sp_fip_blend": 4.0,
        "park_factor": 1.0,
    }
    engine.team.return_value = MagicMock(
        season_wins=40, season_losses=40, elo=1500.0
    )
    cloned = MagicMock()
    cloned.features_for_game.return_value = engine.features_for_game.return_value
    cloned.team.return_value = engine.team.return_value

    art = {
        "feature_columns": ["home_sp_fip_blend", "away_sp_fip_blend", "park_factor"],
        "clf": MagicMock(),
        "lr": {
            "mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "coef": [0.0, 0.0, 0.0],
            "intercept": 0.0,
            "xgb_weight": 0.5,
        },
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "runs_home": None,
        "runs_away": None,
    }

    monkeypatch.setattr(
        live,
        "get_live_context",
        lambda _day: {
            "engine": engine,
            "artifacts": art,
            "todays_games": {(home_id, away_id): game2},
            "todays_finals": [game1],
            "pitcher_names": {},
            "season": 2026,
            "day_iso": "2026-07-12",
            "live_inputs_stale": False,
        },
    )
    monkeypatch.setattr(
        live.MlbFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: cloned),
    )
    monkeypatch.setattr(live, "_predict_probability", lambda _art, _feat: 0.55)
    monkeypatch.setattr(live, "_predict_runs", lambda _art, _feat: None)

    # ESPN abbrs map via ESPN_TO_MLB_TEAM_ID — use known keys
    from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID

    home_abbr = next(k for k, v in ESPN_TO_MLB_TEAM_ID.items() if v == home_id)
    away_abbr = next(k for k, v in ESPN_TO_MLB_TEAM_ID.items() if v == away_id)

    out = live.predict_matchup_v2("2026-07-12", home_abbr, away_abbr)
    assert out is not None
    cloned.update_after_game.assert_called_once_with(game1)
    engine.update_after_game.assert_not_called()
    cloned.features_for_game.assert_called_once_with(game2)
