"""NBA v2 feature engine, replay, data-parsing, and live-path unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_v2.data import (  # noqa: E402
    _signed_spread_from_details,
    canon_franchise,
    devig_two_way,
    franchise_for_espn_id,
)
from web.nba_v2.feature_engine import (  # noqa: E402
    ELO_HOME_ADV,
    FEATURE_COLUMNS,
    NbaFeatureEngine,
)
from web.nba_v2.replay import (  # noqa: E402
    events_to_results,
    merge_season_games,
    replay_season,
)


def _game(
    date: str,
    home: str,
    away: str,
    hs: int,
    as_: int,
    *,
    season: int = 2025,
    season_type: int = 2,
    neutral: bool = False,
) -> dict:
    return {
        "date": date,
        "season": season,
        "season_type": season_type,
        "home": home,
        "away": away,
        "home_score": hs,
        "away_score": as_,
        "neutral_site": neutral,
    }


def test_canon_franchise_follows_relocation_chains() -> None:
    # Seattle SuperSonics -> Oklahoma City Thunder
    assert canon_franchise("sea") == "okc"
    assert canon_franchise("okc") == "okc"
    # Vancouver Grizzlies -> Memphis
    assert canon_franchise("van") == "mem"
    assert canon_franchise("mem") == "mem"
    # New Jersey Nets -> Brooklyn
    assert canon_franchise("njn") == "bkn"
    assert canon_franchise("bkn") == "bkn"
    # ESPN ids stay stable across rebrands (25 = SEA/OKC)
    assert franchise_for_espn_id("25") == "okc"
    assert franchise_for_espn_id("", "bos") == "bos"


def test_engine_features_precede_update_and_elo_moves_to_winner() -> None:
    engine = NbaFeatureEngine()
    first = engine.features_for_game(_game("2025-01-16", "bos", "ny", 0, 0))
    assert first["elo_diff"] == ELO_HOME_ADV
    assert set(first.keys()) == set(FEATURE_COLUMNS)
    for day in range(16, 22):
        engine.update_after_game(_game(f"2025-01-{day:02d}", "bos", "ny", 120, 100))
    assert engine.teams["bos"].elo > engine.teams["ny"].elo
    later = engine.features_for_game(_game("2025-02-10", "bos", "ny", 0, 0))
    assert later["elo_diff"] > first["elo_diff"]
    assert later["win_pct_diff"] == 1.0
    assert later["h2h_home_win_rate"] == 1.0


def test_neutral_site_removes_home_court() -> None:
    engine = NbaFeatureEngine()
    neutral = engine.features_for_game(
        _game("2025-01-16", "bos", "ny", 0, 0, neutral=True)
    )
    assert neutral["elo_diff"] == 0.0
    assert neutral["neutral_site"] == 1.0


def test_rest_and_b2b_flags() -> None:
    engine = NbaFeatureEngine()
    engine.update_after_game(_game("2025-01-16", "bos", "ny", 110, 100))
    features = engine.features_for_game(_game("2025-01-17", "bos", "chi", 0, 0))
    assert features["home_rest_days"] == 1.0
    assert features["home_b2b"] == 1.0
    assert features["away_b2b"] == 0.0


def test_season_rollover_carries_elo_and_resets_records() -> None:
    engine = NbaFeatureEngine()
    for day in range(16, 26):
        engine.update_after_game(
            _game(f"2025-01-{day:02d}", "bos", "ny", 115, 95, season=2025)
        )
    elo_2025 = engine.teams["bos"].elo
    features = engine.features_for_game(
        _game("2025-10-22", "bos", "ny", 0, 0, season=2026)
    )
    bos = engine.teams["bos"]
    assert bos.wins == 0 and bos.losses == 0
    assert 1500.0 < bos.elo < elo_2025
    assert features["prev_win_pct_diff"] == 1.0
    assert features["home_games_played"] == 0.0


def test_engine_snapshot_round_trip() -> None:
    engine = NbaFeatureEngine()
    for day in range(16, 20):
        engine.update_after_game(_game(f"2025-01-{day:02d}", "bos", "ny", 112, 105))
    restored = NbaFeatureEngine.from_dict(engine.to_dict())
    assert restored.teams["bos"].elo == engine.teams["bos"].elo
    assert restored.league_ppg == engine.league_ppg
    original = engine.features_for_game(_game("2025-02-01", "bos", "ny", 0, 0))
    round_trip = restored.features_for_game(_game("2025-02-01", "bos", "ny", 0, 0))
    assert original == round_trip


def test_events_to_results_synthesizes_completed_games() -> None:
    events = [
        {
            "event_id": "401", "date": "2026-01-11T00:00Z", "season_type": 2,
            "completed": True, "home_id": "2", "away_id": "18",
            "home_abbr": "bos", "away_abbr": "ny",
            "home_score": 110, "away_score": 104,
        },
        {
            "event_id": "402", "date": "2026-01-11T02:00Z", "season_type": 2,
            "completed": False, "home_id": "4", "away_id": "5",
            "home_abbr": "chi", "away_abbr": "cle",
            "home_score": None, "away_score": None,
        },
    ]
    rows = events_to_results(events, 2026)
    assert len(rows) == 1
    assert rows[0]["home"] == "bos" and rows[0]["away"] == "ny"
    assert rows[0]["date"] == "2026-01-10"


def test_merge_season_games_flags_2020_bubble_and_attaches_boxes() -> None:
    results = [_game("2020-08-01", "bos", "ny", 112, 104, season=2020)]
    results[0].pop("neutral_site")
    events = [
        {
            "event_id": "999", "date": "2020-08-01", "season_type": 2,
            "completed": True, "neutral_site": False,
            "home_id": "2", "away_id": "18",
            "home_abbr": "bos", "away_abbr": "ny",
            "home_score": 112, "away_score": 104,
        }
    ]
    boxes = {"999": {"2": {"fga": 90.0, "fgm": 40.0}, "18": {"fga": 88.0, "fgm": 38.0}}}
    games = merge_season_games(results, events, boxes, season=2020)
    assert games[0]["neutral_site"] is True
    assert games[0]["event_id"] == "999"
    assert games[0]["home_box"]["fga"] == 90.0


def test_replay_season_respects_cutoff() -> None:
    engine = NbaFeatureEngine()
    games = [
        _game("2025-01-16", "bos", "ny", 110, 100),
        _game("2025-01-20", "ny", "bos", 105, 99),
    ]
    emitted: list[str] = []
    replay_season(
        engine,
        games,
        stop_before_date="2025-01-20",
        emit=lambda game, _f: emitted.append(str(game["date"])),
    )
    assert emitted == ["2025-01-16"]
    assert engine.teams["bos"].games_played == 1


def test_signed_spread_and_devig() -> None:
    assert _signed_spread_from_details("BOS -6.5", "bos", "ny") == -6.5
    assert _signed_spread_from_details("NY -3", "bos", "ny") == 3.0
    assert _signed_spread_from_details("PK", "bos", "ny") is None
    probs = devig_two_way(-150, 130)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1]
    assert devig_two_way(-50, 130) is None


def _box(players: list[list] | None = None, *, dnp_ids: list[str] | None = None) -> dict:
    box = {
        "fgm": 40.0, "fga": 88.0, "tpm": 12.0, "tpa": 32.0,
        "ftm": 15.0, "fta": 20.0, "orb": 10.0, "drb": 32.0,
        "tov": 13.0, "ast": 24.0,
    }
    if players is not None:
        box["players"] = players
    if dnp_ids is not None:
        box["dnp_ids"] = dnp_ids
    return box


def _rich_players(prefix: str, *, star_out: bool = False) -> list[list]:
    """Synthetic rich rows: [id, min, fga, ast, tov, pf, +/-]."""
    base = [
        [f"{prefix}a", 38, 22, 6, 3, 2, 8],
        [f"{prefix}b", 36, 18, 5, 2, 3, 4],
        [f"{prefix}c", 32, 14, 4, 2, 4, 2],
        [f"{prefix}d", 28, 10, 3, 1, 2, -1],
        [f"{prefix}e", 24, 8, 2, 1, 3, -2],
        [f"{prefix}f", 20, 6, 2, 2, 2, -3],
        [f"{prefix}g", 16, 4, 1, 1, 1, -4],
        [f"{prefix}h", 12, 3, 1, 1, 2, -5],
    ]
    if star_out:
        # drop top-2 stars; replacements soak minutes
        return [
            [f"{prefix}c", 36, 16, 4, 2, 5, -6],
            [f"{prefix}d", 34, 14, 3, 2, 4, -4],
            [f"{prefix}e", 30, 12, 3, 1, 3, -2],
            [f"{prefix}f", 26, 10, 2, 2, 3, -3],
            [f"{prefix}g", 22, 8, 2, 1, 2, -2],
            [f"{prefix}h", 18, 6, 1, 1, 2, -1],
            [f"{prefix}q", 16, 5, 1, 1, 1, 0],
            [f"{prefix}r", 14, 4, 1, 1, 1, 0],
        ]
    return base


def test_schedule_run_features_track_stands_and_trips() -> None:
    engine = NbaFeatureEngine()
    for day in (10, 12, 14):
        engine.update_after_game(_game(f"2025-01-{day:02d}", "bos", "ny", 110, 100))
    features = engine.features_for_game(_game("2025-01-16", "bos", "ny", 0, 0))
    assert features["home_stand_len"] == 4.0  # 3 prior home games + this one
    assert features["away_trip_len"] == 4.0
    flipped = engine.features_for_game(_game("2025-01-16", "ny", "bos", 0, 0))
    assert flipped["home_stand_len"] == 1.0  # ny was away, bos was home
    assert flipped["away_trip_len"] == 1.0


def test_3in4_flag_and_tz_altitude() -> None:
    engine = NbaFeatureEngine()
    engine.update_after_game(_game("2025-01-13", "bos", "ny", 110, 100))
    engine.update_after_game(_game("2025-01-15", "bos", "ny", 110, 100))
    features = engine.features_for_game(_game("2025-01-16", "bos", "ny", 0, 0))
    assert features["home_3in4"] == 1.0
    # bos hosting den: venue altitude ~0, den comes from altitude
    features = engine.features_for_game(_game("2025-01-20", "bos", "den", 0, 0))
    assert features["venue_altitude_km"] < 0.1
    assert features["away_altitude_gap"] < -1.0  # downhill for Denver
    den_home = engine.features_for_game(_game("2025-01-22", "den", "bos", 0, 0))
    assert den_home["venue_altitude_km"] > 1.5
    assert den_home["away_altitude_gap"] > 1.5
    # bos (last in Boston) visiting lal shifts ~3 timezones west
    lal_host = engine.features_for_game(_game("2025-01-24", "lal", "bos", 0, 0))
    assert lal_host["away_tz_shift"] < -2.5


def test_close_game_and_blowout_and_margin_volatility() -> None:
    engine = NbaFeatureEngine()
    for day in range(10, 16):
        engine.update_after_game(_game(f"2025-01-{day:02d}", "bos", "ny", 103, 100))
    features = engine.features_for_game(_game("2025-01-20", "bos", "ny", 0, 0))
    assert features["close_win_ewma_diff"] > 0.3  # bos wins all close games
    assert features["blowout_net_ewma_diff"] == 0.0  # no blowouts yet
    assert features["margin_vol_diff"] == 0.0  # symmetric 3-point games
    engine.update_after_game(_game("2025-01-20", "bos", "ny", 130, 100))
    features = engine.features_for_game(_game("2025-01-24", "bos", "ny", 0, 0))
    assert features["blowout_net_ewma_diff"] > 0.15
    assert features["h2h_margin_ewma"] > 3.0
    assert features["elo_mom5_diff"] > 0.0


def test_availability_proxies_from_player_rows() -> None:
    engine = NbaFeatureEngine()
    full = [["a", 36], ["b", 34], ["c", 30], ["d", 28], ["e", 25],
            ["f", 22], ["g", 18], ["h", 15]]
    ny_full = [["x", 36], ["y", 34], ["z", 30], ["w", 28], ["v", 25],
               ["u", 22], ["t", 18], ["s", 15]]
    for day in range(2, 8):  # six games: stars build a cumulative-minutes lead
        game = _game(f"2025-01-{day:02d}", "bos", "ny", 110, 100)
        game["home_box"] = _box(full)
        game["away_box"] = _box(ny_full)
        engine.update_after_game(game)
    even = engine.features_for_game(_game("2025-01-09", "bos", "ny", 0, 0))
    assert even["roster_continuity_diff"] == 0.0
    assert even["star_avail_diff"] == 0.0
    assert even["top1_min_share_diff"] == 0.0
    assert even["rotation_depth_diff"] == 0.0
    # ny loses its top-2 stars; replacements soak the minutes
    ny_short = [["z", 34], ["w", 32], ["v", 30], ["u", 26], ["t", 24],
                ["s", 20], ["q", 18], ["r", 16]]
    game_short = _game("2025-01-09", "bos", "ny", 110, 100)
    game_short["home_box"] = _box(full)
    game_short["away_box"] = _box(ny_short)
    engine.update_after_game(game_short)
    hurt = engine.features_for_game(_game("2025-01-11", "bos", "ny", 0, 0))
    assert hurt["star_avail_diff"] > 0.5  # bos 3/3 vs ny 1/3
    assert hurt["roster_continuity_diff"] > 0.1
    assert hurt["dnp_star_rate_diff"] < 0.0  # ny missing more stars
    assert hurt["star_min_gap_diff"] > 0.0  # bos stars still logging minutes


def test_rich_player_rotation_features() -> None:
    engine = NbaFeatureEngine()
    for day in range(2, 10):
        game = _game(f"2025-01-{day:02d}", "bos", "ny", 110, 100)
        game["home_box"] = _box(_rich_players("h"))
        game["away_box"] = _box(_rich_players("a"))
        engine.update_after_game(game)
    even = engine.features_for_game(_game("2025-01-12", "bos", "ny", 0, 0))
    assert abs(even["top1_usage_diff"]) < 1e-6
    assert abs(even["bench_pm_diff"]) < 1e-6
    assert abs(even["high_min_ast_tov_diff"]) < 1e-6
    concentrated = [
        ["ha", 42, 28, 8, 4, 8, 12],
        ["hb", 30, 12, 3, 2, 6, 2],
        ["hc", 24, 8, 2, 1, 5, 0],
        ["hd", 20, 6, 2, 1, 4, -1],
        ["he", 16, 4, 1, 1, 1, -2],
        ["hf", 12, 3, 1, 1, 1, -3],
        ["hg", 10, 2, 0, 1, 1, -4],
        ["hh", 8, 2, 0, 0, 1, -5],
    ]
    game = _game("2025-01-12", "bos", "ny", 110, 100)
    game["home_box"] = _box(concentrated)
    game["away_box"] = _box(_rich_players("a", star_out=True), dnp_ids=["aa", "ab"])
    engine.update_after_game(game)
    feat = engine.features_for_game(_game("2025-01-14", "bos", "ny", 0, 0))
    assert feat["top1_min_share_diff"] > 0.0
    assert feat["top1_usage_diff"] > 0.0
    assert feat["min_hhi_diff"] > 0.0
    assert feat["bench_pm_diff"] != 0.0
    assert feat["dnp_star_rate_diff"] < 0.0


def test_shooting_profile_updates_from_boxes() -> None:
    engine = NbaFeatureEngine()
    game = _game("2025-01-10", "bos", "ny", 110, 100)
    game["home_box"] = dict(_box(), tpa=45.0, tpm=20.0)
    game["away_box"] = dict(_box(), tpa=18.0, tpm=4.0)
    engine.update_after_game(game)
    features = engine.features_for_game(_game("2025-01-12", "bos", "ny", 0, 0))
    assert features["tpa_rate_diff"] > 0.04
    assert features["tp_pct_diff"] > 0.02
    assert features["tp_pct_against_diff"] < -0.02


def test_from_dict_defaults_new_fields_for_old_snapshots() -> None:
    engine = NbaFeatureEngine()
    game = _game("2025-01-10", "bos", "ny", 110, 100)
    game["home_box"] = _box([["a", 36], ["b", 30]])
    game["away_box"] = _box([["x", 36], ["y", 30]])
    engine.update_after_game(game)
    payload = engine.to_dict()
    legacy_keys = (
        "efg_for_slow", "tpa_rate", "tp_pct", "ft_pct", "tp_pct_against",
        "close_win_ewma", "blowout_net_ewma", "recent_margins", "elo_pre_hist",
        "loc_streak", "last_players", "prev_player_ids", "season_minutes",
        "h2h_margin",
        "top1_min_share", "top3_min_share", "top1_usage", "top3_usage",
        "high_min_ast_tov", "high_min_foul_rate", "star_min_ewma",
        "star_min_season_avg", "bench_pm", "dnp_star_rate", "rotation_depth",
        "min_hhi", "bench_min_share",
    )
    for team_payload in payload["teams"].values():
        for key in legacy_keys:
            team_payload.pop(key, None)
    restored = NbaFeatureEngine.from_dict(payload)
    features = restored.features_for_game(_game("2025-01-12", "bos", "ny", 0, 0))
    assert set(features.keys()) == set(FEATURE_COLUMNS)
    assert features["roster_continuity_diff"] == 0.0
    assert features["star_avail_diff"] == 0.0
    assert features["top1_min_share_diff"] == 0.0
    assert features["star_min_gap_diff"] == 0.0
    assert features["margin_vol_diff"] == 0.0
    assert features["h2h_margin_ewma"] == 0.0
    assert features["elo_mom5_diff"] == 0.0
    assert len(FEATURE_COLUMNS) == 89


def test_live_prediction_when_artifacts_present() -> None:
    from web.nba_v2.live import artifacts_available, predict_matchup_v2, nba_season_for_date

    if not artifacts_available():
        return
    assert nba_season_for_date("2026-01-10") == 2026
    assert nba_season_for_date("2025-11-01") == 2026
    result = predict_matchup_v2("2026-01-10", "bos", "ny")
    if result is None:
        return
    assert result["algorithm"] == "NBAGradientBoost v2"
    assert 0.0 <= result["home_win_probability"] <= 100.0
    assert result["home_elo"] > 1000
    assert "predicted_margin" in result
    assert result.get("model_variant") == "pure"


def test_devig_home_prob_and_market_variant_helpers() -> None:
    from web.nba_v2.live import _devig_home_prob

    assert _devig_home_prob(None, -110) is None
    assert _devig_home_prob(-110, 100) is not None
    # even juice ≈ 0.5
    p = _devig_home_prob(-110, -110)
    assert p is not None and abs(p - 0.5) < 0.02
    # heavy home favorite
    fav = _devig_home_prob(-200, 170)
    assert fav is not None and fav > 0.6


def test_live_market_aware_when_odds_provided() -> None:
    from web.nba_v2.live import artifacts_available, predict_matchup_v2

    if not artifacts_available():
        return
    pure = predict_matchup_v2("2026-01-10", "bos", "ny")
    mkt = predict_matchup_v2(
        "2026-01-10",
        "bos",
        "ny",
        home_moneyline=-150,
        away_moneyline=130,
        home_spread=-4.5,
    )
    if pure is None or mkt is None:
        return
    assert mkt["model_variant"] == "market_aware"
    assert mkt["has_market"] is True
    assert mkt["has_spread"] is True
    assert "predicted_margin" in mkt
    # Market head should be able to differ from pure when odds are present
    assert mkt["home_win_probability"] != pure["home_win_probability"] or mkt[
        "predicted_margin"
    ] != pure["predicted_margin"]


def test_nba_market_aware_artifacts_present() -> None:
    """Market heads shipped with NBA v2 must load for live market-aware scoring."""
    from web.nba_v2.live import _load_artifacts, artifacts_available

    assert artifacts_available()
    art = _load_artifacts()
    assert art is not None
    assert art.get("clf_market") is not None
    assert art.get("lr_market") is not None
    assert art.get("calibrator_market") is not None
    assert art.get("margin_market") is not None
    assert "mkt_home_prob" in art["clf_market_features"]
    assert "has_market" in art["clf_market_features"]
    assert "mkt_home_spread" in art["margin_market_features"]
    assert "has_spread" in art["margin_market_features"]


def test_predict_matchup_v2_market_aware_wiring_with_stub_context() -> None:
    """Odds + market artifacts flip model_variant without needing a live season fetch."""
    from unittest.mock import MagicMock, patch

    from web.nba_v2 import live as nba_live

    team = MagicMock()
    team.games_played = 40
    team.elo = 1500.0
    team.ortg_fast = 112.0
    team.drtg_fast = 108.0
    team.pace_ewma = 100.0
    team.win_pct = MagicMock(return_value=0.55)

    engine = MagicMock()
    engine.teams = {"bos": team, "ny": team}
    engine.team = MagicMock(return_value=team)
    engine.features_for_game = MagicMock(
        return_value={
            "home_rest_days": 1.0,
            "away_rest_days": 1.0,
            "home_b2b": 0.0,
            "away_b2b": 0.0,
            "home_expansion": 0.0,
            "away_expansion": 0.0,
        }
    )

    art = {
        "feature_columns": ["elo_diff"],
        "clf_market_features": ["mkt_home_prob", "has_market"],
        "margin_market_features": ["mkt_home_spread", "has_spread"],
        "clf": object(),
        "lr": {"xgb_weight": 0.5, "intercept": 0.0, "coefs": [0.0]},
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "clf_market": object(),
        "lr_market": {"xgb_weight": 0.5, "intercept": 0.0, "coefs": [0.0, 0.0, 0.0]},
        "calibrator_market": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "margin": object(),
        "margin_market": object(),
        "score_home": object(),
        "score_away": object(),
    }
    context = {
        "engine": engine,
        "artifacts": art,
        "todays_games": {},
        "season": 2026,
    }

    with (
        patch.object(nba_live, "get_live_context", return_value=context),
        patch.object(nba_live, "canon_franchise", side_effect=lambda x: x.lower()),
        patch.object(nba_live, "_predict_probability", return_value=0.62) as mock_prob,
        patch.object(nba_live, "_predict_regressor", side_effect=[5.0, 110.0, 105.0]),
    ):
        result = nba_live.predict_matchup_v2(
            "2026-01-10",
            "bos",
            "ny",
            home_moneyline=-150,
            away_moneyline=130,
            home_spread=-4.5,
        )

    assert result is not None
    assert result["model_variant"] == "market_aware"
    assert result["has_market"] is True
    assert result["has_spread"] is True
    assert result["predicted_margin"] == 5.0
    # Market classifier path is selected when odds + market artifacts exist.
    assert mock_prob.call_args.kwargs.get("clf") is art["clf_market"]
