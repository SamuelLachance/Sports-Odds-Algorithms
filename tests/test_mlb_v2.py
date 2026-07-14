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


def test_select_matchup_game_distinguishes_pending_doubleheader() -> None:
    import web.mlb_v2.live as live

    game1 = {
        "gamePk": 1,
        "status": "S",
        "game_number": 1,
        "home_pp_id": 100,
        "game_datetime": "2026-07-12T17:05:00Z",
    }
    game2 = {
        "gamePk": 2,
        "status": "S",
        "game_number": 2,
        "home_pp_id": 200,
        "game_datetime": "2026-07-12T23:10:00Z",
    }
    games = [game1, game2]
    assert live._select_matchup_game(games, game_number=2)["gamePk"] == 2
    assert live._select_matchup_game(games, game_pk=2)["home_pp_id"] == 200
    # ESPN evening kickoff must pick game 2 starters, not morning game 1.
    selected = live._select_matchup_game(
        games, kickoff_iso="2026-07-12T23:05:00Z"
    )
    assert selected["gamePk"] == 2
    assert selected["home_pp_id"] == 200
    # Without selectors, prefer fold still keeps game 1 when both pending.
    assert live._select_matchup_game(games)["gamePk"] == 1


def test_prefer_todays_game_keeps_delayed_game1() -> None:
    """Rain delay (D) is not terminal — do not promote pending game 2."""
    import web.mlb_v2.live as live

    delayed1 = {
        "gamePk": 1,
        "status": "D",
        "game_number": 1,
        "home_pp_id": 100,
    }
    pending2 = {
        "gamePk": 2,
        "status": "S",
        "game_number": 2,
        "home_pp_id": 200,
    }
    assert live._prefer_todays_game(delayed1, pending2)["gamePk"] == 1
    assert live._prefer_todays_game(pending2, delayed1)["gamePk"] == 1
    # Cancelled game 1 may still promote game 2.
    cancelled1 = {**delayed1, "status": "C"}
    assert live._prefer_todays_game(cancelled1, pending2)["gamePk"] == 2


def test_select_matchup_game_estimates_missing_nightcap_datetime() -> None:
    """Nightcap ESPN kickoff must not fall back to G1 when G2 lacks gameDate."""
    import web.mlb_v2.live as live

    game1 = {
        "gamePk": 1,
        "status": "S",
        "game_number": 1,
        "home_pp_id": 100,
        "game_datetime": "2026-07-12T17:05:00Z",
    }
    game2 = {
        "gamePk": 2,
        "status": "S",
        "game_number": 2,
        "home_pp_id": 200,
        # TBA / missing Stats API datetime is common on nightcaps.
    }
    selected = live._select_matchup_game(
        [game1, game2], kickoff_iso="2026-07-12T23:05:00Z"
    )
    assert selected["gamePk"] == 2
    assert selected["home_pp_id"] == 200


def test_select_matchup_game_estimates_missing_opener_datetime() -> None:
    """When only G2 has a timestamp, evening kickoff must still pick G2 starters."""
    import web.mlb_v2.live as live

    game1 = {
        "gamePk": 1,
        "status": "S",
        "game_number": 1,
        "home_pp_id": 100,
        # Missing G1 datetime previously cloned G2's time → equal delta → G1 wins.
    }
    game2 = {
        "gamePk": 2,
        "status": "S",
        "game_number": 2,
        "home_pp_id": 200,
        "game_datetime": "2026-07-12T23:10:00Z",
    }
    selected = live._select_matchup_game(
        [game1, game2], kickoff_iso="2026-07-12T23:05:00Z"
    )
    assert selected["gamePk"] == 2
    assert selected["home_pp_id"] == 200


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


def test_get_live_context_fails_closed_on_empty_gap_games(monkeypatch) -> None:
    """Empty intermediate-season game bundles must fail closed (not silent Elo skip)."""
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
    monkeypatch.setattr(
        live,
        "_fetch_current_season_bundle",
        lambda season, _day: (
            {
                "games": [],
                "pitchers": {},
                "team_hitting": {},
                "team_pitching": {},
            },
            False,
        ),
    )
    assert live.get_live_context("2025-07-12") is None


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
            "pitchers": {},
            "team_hitting": {},
            "team_pitching": {},
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
    apply_calls: list[tuple] = []

    def fake_apply(eng, game, pitchers, hitting, pitching, *, home_occ=0, away_occ=0):
        apply_calls.append((eng, game, home_occ, away_occ))
        eng.update_after_game(
            game,
            home_hit_log=None,
            away_hit_log=None,
            home_bullpen=None,
            away_bullpen=None,
        )

    monkeypatch.setattr(live, "apply_final_game_with_logs", fake_apply)
    monkeypatch.setattr(live, "_predict_probability", lambda _art, _feat, **_kw: 0.55)
    monkeypatch.setattr(live, "_predict_runs", lambda _art, _feat, **_kw: None)

    # ESPN abbrs map via ESPN_TO_MLB_TEAM_ID — use known keys
    from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID

    home_abbr = next(k for k, v in ESPN_TO_MLB_TEAM_ID.items() if v == home_id)
    away_abbr = next(k for k, v in ESPN_TO_MLB_TEAM_ID.items() if v == away_id)

    out = live.predict_matchup_v2("2026-07-12", home_abbr, away_abbr)
    assert out is not None
    assert len(apply_calls) == 1
    assert apply_calls[0][0] is cloned
    assert apply_calls[0][1] is game1
    assert apply_calls[0][2] == 0
    assert apply_calls[0][3] == 0
    engine.update_after_game.assert_not_called()
    cloned.features_for_game.assert_called_once_with(game2)


def test_game_matches_slate_day_includes_west_coast_spillover() -> None:
    """10pm PT on July 11 is officialDate 07-11 but Toronto slate day 07-12."""
    import web.mlb_v2.live as live

    # 2026-07-12 05:10 UTC = 2026-07-11 22:10 America/Los_Angeles
    # and 2026-07-12 01:10 America/Toronto.
    late = {
        "date": "2026-07-11",
        "game_datetime": "2026-07-12T05:10:00Z",
    }
    assert live._game_matches_slate_day(late, "2026-07-12") is True
    assert live._game_matches_slate_day(late, "2026-07-11") is True  # officialDate match
    afternoon = {
        "date": "2026-07-11",
        "game_datetime": "2026-07-11T20:10:00Z",  # still July 11 in Toronto
    }
    assert live._game_matches_slate_day(afternoon, "2026-07-12") is False
    same_day = {"date": "2026-07-12", "game_datetime": "2026-07-12T23:10:00Z"}
    assert live._game_matches_slate_day(same_day, "2026-07-12") is True


def test_batting_ewma_updates_on_zero_obp_slg() -> None:
    """Hitless 0.000 OBP/SLG must not fall through truthy `or` to league averages."""
    import pytest

    from web.mlb_v2.feature_engine import (
        ALPHA_BATTING,
        LEAGUE_OBP,
        LEAGUE_SLG,
        MlbFeatureEngine,
    )

    engine = MlbFeatureEngine()
    prior_obp = engine.team(111).obp
    prior_slg = engine.team(111).slg
    assert prior_obp == LEAGUE_OBP
    assert prior_slg == LEAGUE_SLG
    engine.update_after_game(
        {
            "date": "2024-06-15",
            "home_id": 111,
            "away_id": 147,
            "home_score": 1,
            "away_score": 0,
        },
        home_hit_log={
            "pa": 28,
            "ab": 27,
            "h": 0,
            "obp": 0.0,
            "slg": 0.0,
            "hr": 0,
            "bb": 0,
            "so": 12,
            "d2": 0,
            "d3": 0,
        },
    )
    expected_obp = (1.0 - ALPHA_BATTING) * LEAGUE_OBP + ALPHA_BATTING * 0.0
    expected_slg = (1.0 - ALPHA_BATTING) * LEAGUE_SLG + ALPHA_BATTING * 0.0
    assert engine.team(111).obp == pytest.approx(expected_obp)
    assert engine.team(111).slg == pytest.approx(expected_slg)
    assert engine.team(111).obp != LEAGUE_OBP


def test_platoon_matchup_features_from_sp_hand_and_rates() -> None:
    """platoon_k/obp_matchup = away_LHP×home_rates − home_LHP×away_rates."""
    import pytest

    from web.mlb_v2.feature_engine import FEATURE_COLUMNS, MlbFeatureEngine

    assert "platoon_k_matchup" in FEATURE_COLUMNS
    assert "platoon_obp_matchup" in FEATURE_COLUMNS
    assert 150 <= len(FEATURE_COLUMNS) <= 220
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    for col in (
        "has_open_line",
        "has_steam",
        "ml_steam_pp",
        "ba_diff",
        "ops_diff",
        "win_streak_diff",
        "travel_diff",
        "home_sp_short_rest",
        "home_bp_fried",
        "has_weather",
        "coors_extreme",
        "elo_x_season_frac",
        "april_noise",
        "has_ump",
        "has_il",
        "has_statcast_sp",
        "steam_to_fav",
    ):
        assert col in FEATURE_COLUMNS

    engine = MlbFeatureEngine()
    home = engine.team(111)
    away = engine.team(147)
    home.so_rate = 0.28
    home.obp = 0.340
    away.so_rate = 0.18
    away.obp = 0.300

    # Away SP is LHP → home offense × LHP; home SP missing → RHP prior (0)
    engine.pitcher(9001, hand="L")

    feats = engine.features_for_game(
        {
            "date": "2024-06-15",
            "home_id": 111,
            "away_id": 147,
            "home_pp_id": None,
            "away_pp_id": 9001,
            "venue_id": None,
        }
    )
    assert set(feats.keys()) == set(FEATURE_COLUMNS)
    assert all(v == v and isinstance(v, float) for v in feats.values())
    assert feats["away_sp_is_lhp"] == 1.0
    assert feats["home_sp_is_lhp"] == 0.0
    assert feats["platoon_k_matchup"] == pytest.approx(0.28)
    assert feats["platoon_obp_matchup"] == pytest.approx(0.340)
    assert feats["has_steam"] == 0.0
    assert feats["has_weather"] == 0.0
    assert feats["temp_c"] == 22.0
    assert feats["mkt_home_prob"] == 0.5



def test_mlb_elo_mov_stays_positive_on_huge_underdog_upset() -> None:
    """Underdog wins with a massive Elo gap must not flip MOV (denom ≤ 0)."""
    import math

    import pytest

    from web.mlb_v2.feature_engine import ELO_HOME_ADV, ELO_K, MlbFeatureEngine

    engine = MlbFeatureEngine()
    home = engine.team(111)
    away = engine.team(147)
    home.elo = 2800.0
    away.elo = 500.0
    pre_home, pre_away = home.elo, away.elo

    # Without max(winner_diff, 0): winner_diff=-2300 → mov_mult < 0 → favorite
    # gains Elo after losing.
    winner_diff = away.elo - home.elo
    assert winner_diff < -2200
    buggy_mov = math.log(5 + 1.0) * (2.2 / (0.001 * winner_diff + 2.2))
    assert buggy_mov < 0

    engine.update_after_game(
        {
            "date": "2024-06-15",
            "home_id": 111,
            "away_id": 147,
            "home_score": 1,
            "away_score": 6,
        }
    )
    assert home.elo < pre_home
    assert away.elo > pre_away
    expected_home = 1.0 / (
        1.0 + 10 ** ((pre_away - pre_home - ELO_HOME_ADV) / 400.0)
    )
    clamped_mov = math.log(5 + 1.0) * (2.2 / (0.001 * max(winner_diff, 0.0) + 2.2))
    expected_delta = ELO_K * clamped_mov * (0.0 - expected_home)
    assert home.elo == pytest.approx(pre_home + expected_delta, abs=1e-9)


def test_write_mlb_feature_snapshot_refuses_mislabeled_gap_season(tmp_path) -> None:
    """Missing snapshot-season cache must not write state_N with end-of-(N-1) state."""
    import json
    import pytest

    from web.mlb_v2.feature_engine import MlbFeatureEngine
    from web.mlb_v2.replay import write_mlb_feature_snapshot

    cache = tmp_path / "cache"
    season_2022 = cache / "2022"
    season_2022.mkdir(parents=True)
    for name, payload in (
        ("games", []),
        ("pitchers", {}),
        ("team_hitting", {}),
        ("team_pitching", {}),
    ):
        (season_2022 / f"{name}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    engine = MlbFeatureEngine()
    with pytest.raises(ValueError, match="mislabeled"):
        write_mlb_feature_snapshot(
            engine,
            tmp_path / "out",
            snapshot_season=2023,
            start_season=2022,
            cache_root=cache,
        )
    assert not (tmp_path / "out" / "state_2023.json.gz").exists()

def test_predict_matchup_v2_market_aware_wiring_with_stub_context() -> None:
    """Odds + market artifacts flip model_variant without needing a live season fetch."""
    from unittest.mock import MagicMock, patch

    from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID
    from web.mlb_v2 import live as mlb_live

    home_abbr = next(iter(ESPN_TO_MLB_TEAM_ID))
    away_abbr = [k for k in ESPN_TO_MLB_TEAM_ID if k != home_abbr][0]
    home_id = ESPN_TO_MLB_TEAM_ID[home_abbr]
    away_id = ESPN_TO_MLB_TEAM_ID[away_abbr]

    team = MagicMock()
    team.season_wins = 40
    team.season_losses = 40
    team.elo = 1500.0

    engine = MagicMock()
    engine.team = MagicMock(return_value=team)
    engine.features_for_game = MagicMock(
        return_value={
            "home_sp_fip_blend": 3.8,
            "away_sp_fip_blend": 4.1,
            "park_factor": 1.0,
        }
    )

    art = {
        "feature_columns": ["elo_diff"],
        "clf_market_features": ["mkt_home_prob", "has_market"],
        "clf": object(),
        "lr": {"xgb_weight": 0.55},
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "clf_market": object(),
        "lr_market": {"xgb_weight": 0.55},
        "calibrator_market": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "runs_home": None,
        "runs_away": None,
    }
    context = {
        "engine": engine,
        "artifacts": art,
        "todays_games": {},
        "todays_matchup_games": {},
        "todays_finals": [],
        "pitcher_names": {},
        "pitchers": {},
        "team_hitting": {},
        "team_pitching": {},
        "season": 2026,
    }

    with (
        patch.object(mlb_live, "get_live_context", return_value=context),
        patch.object(mlb_live, "_predict_probability", return_value=0.58) as mock_prob,
        patch.object(mlb_live, "_predict_runs", return_value=None),
        patch("web.hybrid_v2.live.try_hybrid_binary", return_value=None),
    ):
        result = mlb_live.predict_matchup_v2(
            "2026-07-12",
            home_abbr,
            away_abbr,
            home_moneyline=-140,
            away_moneyline=120,
        )

    assert result is not None
    assert result["model_variant"] == "market_aware"
    assert result["has_market"] is True
    assert mock_prob.call_args.kwargs.get("clf") is art["clf_market"]
