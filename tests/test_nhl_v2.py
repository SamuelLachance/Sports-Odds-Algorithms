"""NHL v2 live artifact loading defensive tests."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    """Corrupt on-disk JSON must not raise — live layer returns None instead."""
    import web.nhl_v2.live as live

    model_dir = tmp_path / "nhl_v2"
    model_dir.mkdir()
    for name in ("model_clf.json", "model_lr.json", "calibrator.json", "metadata.json"):
        (model_dir / name).write_text("{not-json", encoding="utf-8")
    with gzip.open(model_dir / "state_2024.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    monkeypatch.setattr(live, "MODEL_DIR", model_dir)
    live._load_artifacts.cache_clear()
    assert live.artifacts_available() is True
    assert live._load_artifacts() is None


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path, monkeypatch) -> None:
    import web.nhl_v2.live as live

    model_dir = tmp_path / "nhl_v2"
    model_dir.mkdir()
    bad = model_dir / "state_2024.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2024: bad}}
    assert live._load_snapshot_state(art, 2025) is None


def test_fetch_stats_bundle_flags_stale_on_network_failure(tmp_path, monkeypatch) -> None:
    import json
    import time

    import web.nhl_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)
    cache = tmp_path / "stats_2024.json"
    payload = {"team_games": [{"id": 1}], "goalies": []}
    cache.write_text(json.dumps(payload), encoding="utf-8")
    # Age the file past the fresh TTL but within the soft-fail window.
    aged = time.time() - (live.STATS_TTL_SECONDS + 60)
    import os

    os.utime(cache, (aged, aged))

    def _boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(live, "fetch_team_games", _boom)
    monkeypatch.setattr(live, "fetch_goalie_games", _boom)
    bundle, stale = live._fetch_stats_bundle(2024)
    assert stale is True
    assert bundle == payload


def test_blend_nhl_survives_hockey_pred_crash() -> None:
    """NHL blend must fall back to Algo V1 when hockey pred raises."""
    from web.blend_service import blend_predictions
    import web.blend_service as blend_module

    original = blend_module.run_hockey_pred_model

    def _boom(*_a, **_k):
        raise RuntimeError("simulated nhl v2 failure")

    try:
        blend_module.run_hockey_pred_model = _boom
        result = blend_predictions(
            legacy_total_score=-4.5,
            legacy_win_probability=58.0,
            league="nhl",
            cutoff_date="1-15-2026",
            home_abbr="tor",
            away_abbr="bos",
            home_moneyline=-140,
            away_moneyline=120,
        )
    finally:
        blend_module.run_hockey_pred_model = original

    assert result["blend_mode"] == "algo_v1"
    assert result["algorithm"] == "Algo_V1"
    assert result.get("hockey_pred") is None


def test_infer_nhl_game_type_playoff_window() -> None:
    from web.nhl_v2.live import infer_nhl_game_type

    assert infer_nhl_game_type("2026-01-15") == 2
    assert infer_nhl_game_type("2026-04-10") == 2
    assert infer_nhl_game_type("2026-04-15") == 3
    assert infer_nhl_game_type("2026-05-20") == 3
    assert infer_nhl_game_type("2026-06-15") == 3
    assert infer_nhl_game_type("not-a-date") == 2


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    """Hard-missing intermediate seasons must not silently skip Elo/form state."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)

    calls: list[int] = []

    def fake_bundle(season: int):
        calls.append(season)
        if season == 2024:
            return None, False
        return {"team_games": [], "goalies": []}, False

    monkeypatch.setattr(live, "_fetch_stats_bundle", fake_bundle)
    assert live.get_live_context("2025-01-15") is None
    assert 2024 in calls


def test_get_live_context_fails_closed_on_empty_gap_games(monkeypatch) -> None:
    """Empty intermediate-season game bundles must fail closed (not silent Elo skip)."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)
    monkeypatch.setattr(
        live,
        "_fetch_stats_bundle",
        lambda season: ({"team_games": [], "goalies": []}, False),
    )
    monkeypatch.setattr(live, "_fetch_moneypuck_slices", lambda seasons: ({}, False))
    assert live.get_live_context("2025-01-15") is None


def test_get_live_context_fails_closed_on_empty_gap_moneypuck(monkeypatch) -> None:
    """Gap seasons with Stats games but empty MoneyPuck must fail closed."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)

    monkeypatch.setattr(
        live,
        "_fetch_stats_bundle",
        lambda season: (
            {"team_games": [{"gameId": 1, "teamAbbrev": "TOR"}], "goalies": []},
            False,
        ),
    )
    monkeypatch.setattr(
        live,
        "_fetch_moneypuck_slices",
        lambda seasons: ({}, False),
    )
    monkeypatch.setattr(
        live,
        "build_game_index",
        lambda _rows: {1: {"home": "TOR", "away": "MTL"}},
    )
    assert live.get_live_context("2025-01-15") is None


def test_get_live_context_fails_closed_on_empty_current_moneypuck(monkeypatch) -> None:
    """Current season with Stats games but empty MoneyPuck must fail closed."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2024: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2024, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2024)
    monkeypatch.setattr(
        live,
        "_fetch_stats_bundle",
        lambda season: (
            {"team_games": [{"gameId": 1, "teamAbbrev": "TOR"}], "goalies": []},
            False,
        ),
    )
    monkeypatch.setattr(
        live,
        "_fetch_moneypuck_slices",
        lambda seasons: ({}, False),
    )
    assert live.get_live_context("2024-12-15") is None


def test_espn_to_nhl_maps_phx_alias() -> None:
    """Historical ESPN/closing keys still use phx for the Coyotes franchise."""
    from web.nhl_v2.live import ESPN_TO_NHL

    assert ESPN_TO_NHL["phx"] == "ARI"
    assert ESPN_TO_NHL["ari"] == "ARI"


def test_nhl_features_precede_update_rest_and_elo() -> None:
    """Emit-before-update: rest/B2B must not see the current game; Elo moves after."""
    from web.nhl_v2.feature_engine import ELO_HOME_ADV, NhlFeatureEngine

    engine = NhlFeatureEngine()
    game1 = {
        "gameId": 1,
        "date": "2024-10-10",
        "game_type": 2,
        "home": "TOR",
        "away": "BOS",
        "home_goals": 4,
        "away_goals": 2,
        "home_win": 1,
        "home_shots": 30,
        "away_shots": 28,
    }
    before = engine.features_for_game(game1)
    assert before["elo_diff"] == ELO_HOME_ADV / 100.0
    assert before["home_b2b"] == 0.0
    engine.update_after_game(game1)
    game2 = {
        "gameId": 2,
        "date": "2024-10-11",
        "game_type": 2,
        "home": "MTL",
        "away": "TOR",
        "home_goals": 0,
        "away_goals": 0,
        "home_win": 0,
        "home_shots": 20,
        "away_shots": 20,
    }
    after = engine.features_for_game(game2)
    # TOR played yesterday → B2B; features still pre-update for game2.
    assert after["away_b2b"] == 1.0
    assert after["away_rest_days"] == 1.0
    assert engine.team("TOR").elo > engine.team("BOS").elo
    # Updating game2 must not be required for the Elo move from game1.
    assert engine.team("TOR").games_played == 1


def test_giveaway_diff_from_moneypuck_dzone() -> None:
    """MoneyPuck dZoneGiveawaysFor folds into EWMA and emits giveaway_diff leak-free."""
    from web.nhl_v2.feature_engine import FEATURE_COLUMNS, LEAGUE_DZ_GIVEAWAYS, NhlFeatureEngine

    engine = NhlFeatureEngine()
    game1 = {
        "gameId": 10,
        "date": "2024-10-12",
        "game_type": 2,
        "home": "COL",
        "away": "DAL",
        "home_goals": 3,
        "away_goals": 2,
        "home_win": 1,
        "home_shots": 32,
        "away_shots": 28,
        "mp": {
            "COL": {"all_dZoneGiveawaysFor": 12.0},
            "DAL": {"all_dZoneGiveawaysFor": 2.0},
        },
    }
    before = engine.features_for_game(game1)
    assert before["giveaway_diff"] == 0.0
    assert "giveaway_diff" in FEATURE_COLUMNS
    engine.update_after_game(game1)
    assert engine.team("COL").giveaways > LEAGUE_DZ_GIVEAWAYS
    assert engine.team("DAL").giveaways < LEAGUE_DZ_GIVEAWAYS

    after = engine.features_for_game(
        {
            "gameId": 11,
            "date": "2024-10-14",
            "game_type": 2,
            "home": "COL",
            "away": "DAL",
            "home_goals": 0,
            "away_goals": 0,
            "home_win": 0,
            "home_shots": 20,
            "away_shots": 20,
        }
    )
    assert after["giveaway_diff"] > 0.0
    snapshot = engine.team("COL").giveaways
    _ = engine.features_for_game(
        {
            "gameId": 12,
            "date": "2024-10-16",
            "game_type": 2,
            "home": "COL",
            "away": "MIN",
            "home_goals": 0,
            "away_goals": 0,
            "home_win": 0,
            "home_shots": 20,
            "away_shots": 20,
        }
    )
    assert engine.team("COL").giveaways == snapshot


def test_skater_depth_proxies_from_unused_moneypuck() -> None:
    """Unused team MP columns fold into skater/roster-quality FEATURE_COLUMNS."""
    from web.nhl_v2.feature_engine import (
        FEATURE_COLUMNS,
        LEAGUE_EV_XG_SHARE,
        LEAGUE_HITS,
        NhlFeatureEngine,
    )

    skater_cols = {
        "hits_against_diff",
        "corsi_all_pct_diff",
        "ev_xg_share_diff",
        "pp_toi_share_diff",
    }
    assert skater_cols.issubset(set(FEATURE_COLUMNS))
    expanded = {
        "pp_pct_diff",
        "home_xgf_vs_away_xga",
        "home_stand_len",
        "has_steam",
        "ml_steam_pp",
        "regulation_win_share_diff",
        "backup_gsax_diff",
    }
    assert expanded.issubset(set(FEATURE_COLUMNS))
    assert 90 <= len(FEATURE_COLUMNS) <= 200
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    engine = NhlFeatureEngine()
    game1 = {
        "gameId": 20,
        "date": "2024-10-12",
        "game_type": 2,
        "home": "COL",
        "away": "DAL",
        "home_goals": 4,
        "away_goals": 1,
        "home_win": 1,
        "home_shots": 35,
        "away_shots": 22,
        "mp": {
            "COL": {
                "all_hitsAgainst": 35.0,
                "all_shotAttemptsFor": 70.0,
                "all_shotAttemptsAgainst": 40.0,
                "5on5_xGoalsFor": 2.8,
                "all_xGoalsFor": 3.2,
                "5on4_iceTime": 400.0,
                "all_iceTime": 18000.0,
            },
            "DAL": {
                "all_hitsAgainst": 10.0,
                "all_shotAttemptsFor": 35.0,
                "all_shotAttemptsAgainst": 65.0,
                "5on5_xGoalsFor": 1.0,
                "all_xGoalsFor": 2.5,
                "5on4_iceTime": 200.0,
                "all_iceTime": 18000.0,
            },
        },
    }
    before = engine.features_for_game(game1)
    assert before["hits_against_diff"] == 0.0
    assert before["corsi_all_pct_diff"] == 0.0
    engine.update_after_game(game1)
    assert engine.team("COL").hits_against > LEAGUE_HITS
    assert engine.team("DAL").hits_against < LEAGUE_HITS
    assert engine.team("COL").corsi_all_pct > 0.5
    assert engine.team("COL").ev_xg_share > LEAGUE_EV_XG_SHARE - 0.01

    after = engine.features_for_game(
        {
            "gameId": 21,
            "date": "2024-10-14",
            "game_type": 2,
            "home": "COL",
            "away": "DAL",
            "home_goals": 0,
            "away_goals": 0,
            "home_win": 0,
            "home_shots": 20,
            "away_shots": 20,
        }
    )
    assert after["hits_against_diff"] > 0.0
    assert after["corsi_all_pct_diff"] > 0.0
    assert after["ev_xg_share_diff"] > 0.0
    assert after["pp_toi_share_diff"] > 0.0
    snapshot = engine.team("COL").corsi_all_pct
    _ = engine.features_for_game(
        {
            "gameId": 22,
            "date": "2024-10-16",
            "game_type": 2,
            "home": "COL",
            "away": "MIN",
            "home_goals": 0,
            "away_goals": 0,
            "home_win": 0,
            "home_shots": 20,
            "away_shots": 20,
        }
    )
    assert engine.team("COL").corsi_all_pct == snapshot


def test_explicit_home_win_zero_not_overridden_by_scores() -> None:
    """`home_win=0` is falsy; must not fall through to score-based `or` fallback."""
    from web.nhl_v2.feature_engine import NhlFeatureEngine

    engine = NhlFeatureEngine()
    engine.update_after_game(
        {
            "gameId": 99,
            "date": "2024-10-12",
            "game_type": 2,
            "home": "EDM",
            "away": "CGY",
            # Scores alone would imply a home win; explicit flag says away won.
            "home_goals": 5,
            "away_goals": 1,
            "home_win": 0,
            "home_shots": 30,
            "away_shots": 20,
        }
    )
    assert engine.team("EDM").season_wins == 0
    assert engine.team("EDM").season_losses == 1
    assert engine.team("CGY").season_wins == 1
    assert engine.team("CGY").elo > engine.team("EDM").elo


def test_gap_season_replay_passes_goalie_xg(monkeypatch) -> None:
    """Gap seasons must attach MoneyPuck goalie xG like the current-season path."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)

    def fake_bundle(season: int):
        return (
            {"team_games": [{"gameId": season}], "goalies": []},
            False,
        )

    monkeypatch.setattr(live, "_fetch_stats_bundle", fake_bundle)
    monkeypatch.setattr(
        live,
        "_fetch_moneypuck_slices",
        lambda seasons: ({s: {s: {"xG": 1.0}} for s in seasons}, False),
    )
    xg_seasons: list[int] = []

    def fake_xg(season: int):
        xg_seasons.append(season)
        return {f"{season}:99": [10.0, 2.5, 3.0, 1.0, 0.4, 0.0]}, False

    monkeypatch.setattr(live, "_fetch_goalie_xg_slice", fake_xg)
    monkeypatch.setattr(
        live,
        "build_game_index",
        lambda rows: {
            int(r["gameId"]): {
                "gameId": int(r["gameId"]),
                "date": f"{r['gameId']}-10-10",
                "home": "TOR",
                "away": "MTL",
            }
            for r in rows
        },
    )
    monkeypatch.setattr(live, "build_goalie_index", lambda _rows: {})
    monkeypatch.setattr(live, "fetch_schedule_day", lambda _day: [])

    replay_kwargs: list[dict] = []

    def fake_replay(*_args, **kwargs):
        replay_kwargs.append(kwargs)

    monkeypatch.setattr(live, "replay_season", fake_replay)

    ctx = live.get_live_context("2025-01-15")
    assert ctx is not None
    assert 2024 in xg_seasons  # gap season fetched
    assert 2025 in xg_seasons  # current season fetched
    gap_calls = [kw for kw in replay_kwargs if kw.get("goalie_xg")]
    assert gap_calls, "gap replay must pass goalie_xg="
    assert (2024, 99) in gap_calls[0]["goalie_xg"]


def test_faceoff_ewma_updates_on_zero_pct() -> None:
    """0.0 faceoff % must not be skipped by a truthiness check."""
    from web.nhl_v2.feature_engine import ALPHA_SPECIAL, LEAGUE_FO_PCT, NhlFeatureEngine

    engine = NhlFeatureEngine()
    prior = engine.team("bos").faceoff
    assert prior == LEAGUE_FO_PCT
    engine.update_after_game(
        {
            "date": "2024-10-15",
            "home": "bos",
            "away": "mtl",
            "home_goals": 3,
            "away_goals": 2,
            "home_win": 1,
            "home_faceoff_pct": 0.0,
            "away_faceoff_pct": 0.55,
        }
    )
    expected = (1.0 - ALPHA_SPECIAL) * LEAGUE_FO_PCT + ALPHA_SPECIAL * 0.0
    assert engine.team("bos").faceoff == pytest.approx(expected)
    assert engine.team("mtl").faceoff != LEAGUE_FO_PCT


def test_sog_ewma_updates_on_zero_shots() -> None:
    """Explicit 0 SOG must not fall through truthy `or` to LEAGUE_SOG."""
    from web.nhl_v2.feature_engine import ALPHA_FAST, LEAGUE_SOG, NhlFeatureEngine

    engine = NhlFeatureEngine()
    prior = engine.team("bos").sog_for
    assert prior == LEAGUE_SOG
    engine.update_after_game(
        {
            "date": "2024-10-15",
            "home": "bos",
            "away": "mtl",
            "home_goals": 3,
            "away_goals": 2,
            "home_win": 1,
            "home_shots": 0,
            "away_shots": 0,
            "mp": {},
        }
    )
    expected = (1.0 - ALPHA_FAST) * LEAGUE_SOG + ALPHA_FAST * 0.0
    assert engine.team("bos").sog_for == pytest.approx(expected)
    assert engine.team("bos").sog_for != LEAGUE_SOG


def test_equal_score_without_home_win_is_regulation_tie() -> None:
    """Missing home_win + equal goals must not count as a full away regulation win."""
    from web.nhl_v2.feature_engine import NhlFeatureEngine

    engine = NhlFeatureEngine()
    elo_before_home = engine.team("TOR").elo
    elo_before_away = engine.team("MTL").elo
    engine.update_after_game(
        {
            "home": "TOR",
            "away": "MTL",
            "date": "2003-11-01",
            "home_goals": 2,
            "away_goals": 2,
            "game_type": 2,
        }
    )
    assert engine.team("TOR").season_wins == 0
    assert engine.team("TOR").season_losses == 0
    assert engine.team("MTL").season_wins == 0
    assert engine.team("MTL").season_losses == 0
    # Regulation-tie goals still count even though W-L stays flat.
    assert engine.team("TOR").season_gf == 2
    assert engine.team("TOR").season_ga == 2
    assert engine.team("MTL").season_gf == 2
    assert engine.team("MTL").season_ga == 2
    assert engine.team("TOR").games_played == 1
    # Half-credit Elo: home (with HFA) should lose a little Elo vs even opponent.
    assert engine.team("TOR").elo < elo_before_home
    assert engine.team("MTL").elo > elo_before_away
    assert engine.team("TOR").h2h.get("MTL") in (None, [])

def test_predict_matchup_v2_market_aware_wiring_with_stub_context() -> None:
    """Odds + market artifacts flip model_variant without needing a live season fetch."""
    from unittest.mock import MagicMock, patch

    from web.nhl_v2 import live as nhl_live

    team = MagicMock()
    team.games_played = 40
    team.elo = 1500.0
    team.xgf_fast = 2.8
    team.xga_fast = 2.5
    team.points_pct = MagicMock(return_value=0.55)

    engine = MagicMock()
    engine.team = MagicMock(return_value=team)
    engine.features_for_game = MagicMock(
        return_value={
            "home_rest_days": 1.0,
            "away_rest_days": 1.0,
            "home_b2b": 0.0,
            "away_b2b": 0.0,
            "home_goalie_gsax": 0.5,
            "away_goalie_gsax": -0.2,
        }
    )

    art = {
        "feature_columns": ["elo_diff"],
        "clf_market_features": ["mkt_home_prob", "has_market"],
        "clf": object(),
        "lr": {"xgb_weight": 0.5, "intercept": 0.0, "mean": [0.0], "scale": [1.0], "coef": [0.0]},
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "clf_market": object(),
        "lr_market": {
            "xgb_weight": 0.5,
            "intercept": 0.0,
            "mean": [0.0, 0.5, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "coef": [0.0, 0.0, 0.0],
        },
        "calibrator_market": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "goals_home": None,
        "goals_away": None,
    }
    context = {
        "engine": engine,
        "artifacts": art,
        "todays_games": {},
        "depth_chart": {},
        "goalie_names": {},
        "season": 2025,
    }

    with (
        patch.object(nhl_live, "get_live_context", return_value=context),
        patch.object(nhl_live, "_predict_probability", return_value=0.61) as mock_prob,
        patch.object(nhl_live, "_predict_goals", return_value=None),
        patch("web.hybrid_v2.live.try_hybrid_binary", return_value=None),
    ):
        result = nhl_live.predict_matchup_v2(
            "2026-01-10",
            "tor",
            "bos",
            home_moneyline=-150,
            away_moneyline=130,
        )

    assert result is not None
    assert result["model_variant"] == "market_aware"
    assert result["has_market"] is True
    assert mock_prob.call_args.kwargs.get("clf") is art["clf_market"]


def test_predict_matchup_v2_soft_fails_to_pure_without_market_artifacts() -> None:
    from unittest.mock import MagicMock, patch

    from web.nhl_v2 import live as nhl_live

    team = MagicMock()
    team.games_played = 40
    team.elo = 1500.0
    team.xgf_fast = 2.8
    team.xga_fast = 2.5
    team.points_pct = MagicMock(return_value=0.55)

    engine = MagicMock()
    engine.team = MagicMock(return_value=team)
    engine.features_for_game = MagicMock(
        return_value={
            "home_rest_days": 1.0,
            "away_rest_days": 1.0,
            "home_b2b": 0.0,
            "away_b2b": 0.0,
            "home_goalie_gsax": 0.0,
            "away_goalie_gsax": 0.0,
        }
    )
    art = {
        "feature_columns": ["elo_diff"],
        "clf_market_features": ["mkt_home_prob", "has_market"],
        "clf": object(),
        "lr": {"xgb_weight": 0.5},
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "clf_market": None,
        "lr_market": None,
        "calibrator_market": None,
        "goals_home": None,
        "goals_away": None,
    }
    context = {
        "engine": engine,
        "artifacts": art,
        "todays_games": {},
        "depth_chart": {},
        "goalie_names": {},
        "season": 2025,
    }
    with (
        patch.object(nhl_live, "get_live_context", return_value=context),
        patch.object(nhl_live, "_predict_probability", return_value=0.55) as mock_prob,
        patch.object(nhl_live, "_predict_goals", return_value=None),
        patch("web.hybrid_v2.live.try_hybrid_binary", return_value=None),
    ):
        result = nhl_live.predict_matchup_v2(
            "2026-01-10", "tor", "bos", home_moneyline=-150, away_moneyline=130
        )
    assert result["model_variant"] == "pure"
    assert mock_prob.call_args.kwargs.get("clf") is None
