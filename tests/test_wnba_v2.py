"""WNBA v2 feature engine, replay, data-parsing, and live-path unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.wnba_v2.data import (  # noqa: E402
    _american,
    _event_calendar_date,
    _player_box_rows,
    _signed_spread_from_details,
    canon_franchise,
    devig_two_way,
    franchise_for_espn_id,
)
from web.wnba_v2.feature_engine import (  # noqa: E402
    ELO_HOME_ADV,
    FEATURE_COLUMNS,
    WnbaFeatureEngine,
)
from web.wnba_v2.replay import (  # noqa: E402
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
    # Utah Starzz -> San Antonio -> Las Vegas Aces share one franchise key
    assert canon_franchise("utah") == "lva"
    assert canon_franchise("sas") == "lva"
    assert canon_franchise("lva") == "lva"
    # Detroit Shock -> Tulsa -> Dallas Wings
    assert canon_franchise("det") == "dal"
    assert canon_franchise("tul") == "dal"
    # Orlando Miracle -> Connecticut Sun
    assert canon_franchise("orl") == "con"
    # ESPN ids stay stable across rebrands (17 = Utah/SA/LV franchise)
    assert franchise_for_espn_id("17") == "lva"
    assert franchise_for_espn_id("", "sea") == "sea"
    # Historical ESPN ids must resolve to relocation targets, not stale tricodes.
    assert franchise_for_espn_id("4") == "dal"  # Shock → Wings
    assert franchise_for_espn_id("21") == "lva"  # Starzz → Aces
    assert franchise_for_espn_id("10") == "con"  # Miracle → Sun


def test_engine_features_precede_update_and_elo_moves_to_winner() -> None:
    engine = WnbaFeatureEngine()
    first = engine.features_for_game(_game("2025-05-16", "lva", "sea", 0, 0))
    assert first["elo_diff"] == ELO_HOME_ADV  # equal priors + home court
    assert set(first.keys()) == set(FEATURE_COLUMNS)
    for day in range(16, 22):
        engine.update_after_game(_game(f"2025-05-{day:02d}", "lva", "sea", 90, 70))
    assert engine.teams["lva"].elo > engine.teams["sea"].elo
    later = engine.features_for_game(_game("2025-06-10", "lva", "sea", 0, 0))
    assert later["elo_diff"] > first["elo_diff"]
    assert later["win_pct_diff"] == 1.0
    assert later["h2h_margin_ewma"] > 0.0


def test_neutral_site_removes_home_court() -> None:
    engine = WnbaFeatureEngine()
    neutral = engine.features_for_game(
        _game("2025-05-16", "lva", "sea", 0, 0, neutral=True)
    )
    assert neutral["elo_diff"] == 0.0
    assert "neutral_site" not in neutral
    home = engine.features_for_game(_game("2025-05-16", "lva", "sea", 0, 0))
    assert home["elo_diff"] == ELO_HOME_ADV


def test_rest_and_b2b_flags() -> None:
    engine = WnbaFeatureEngine()
    engine.update_after_game(_game("2025-05-16", "lva", "sea", 90, 80))
    features = engine.features_for_game(_game("2025-05-17", "lva", "min", 0, 0))
    assert features["home_rest_days"] == 1.0
    assert features["home_b2b"] == 1.0
    assert features["away_b2b"] == 0.0  # min has not played -> 5-day prior


def test_season_rollover_carries_elo_and_resets_records() -> None:
    engine = WnbaFeatureEngine()
    for day in range(16, 26):
        engine.update_after_game(
            _game(f"2025-05-{day:02d}", "lva", "sea", 95, 75, season=2025)
        )
    # Finish prior season on the road so last_market is the away venue.
    engine.update_after_game(
        _game("2025-09-10", "sea", "lva", 70, 80, season=2025)
    )
    assert engine.teams["lva"].last_market is not None
    elo_2025 = engine.teams["lva"].elo
    features = engine.features_for_game(_game("2026-05-16", "lva", "sea", 0, 0, season=2026))
    lva = engine.teams["lva"]
    assert lva.wins == 0 and lva.losses == 0  # season stats reset
    assert lva.last_market is None
    assert 1500.0 < lva.elo < elo_2025  # carryover regresses toward mean
    assert features["prev_win_pct_diff"] == 1.0  # last season's record retained
    assert features["home_games_played"] == 0.0
    assert features["home_travel_km"] == 0.0
    assert features["home_tz_shift"] == 0.0


def test_engine_snapshot_round_trip() -> None:
    engine = WnbaFeatureEngine()
    for day in range(16, 20):
        engine.update_after_game(_game(f"2025-05-{day:02d}", "lva", "sea", 88, 82))
    restored = WnbaFeatureEngine.from_dict(engine.to_dict())
    assert restored.teams["lva"].elo == engine.teams["lva"].elo
    assert restored.league_ppg == engine.league_ppg
    original = engine.features_for_game(_game("2025-06-01", "lva", "sea", 0, 0))
    round_trip = restored.features_for_game(_game("2025-06-01", "lva", "sea", 0, 0))
    assert original == round_trip


def test_events_to_results_synthesizes_completed_games() -> None:
    events = [
        {
            "event_id": "401", "date": "2026-07-11T00:00Z", "season_type": 2,
            "completed": True, "home_id": "17", "away_id": "14",
            "home_abbr": "lva", "away_abbr": "sea",
            "home_score": 90, "away_score": 84,
        },
        {
            "event_id": "402", "date": "2026-07-11T02:00Z", "season_type": 2,
            "completed": False, "home_id": "8", "away_id": "6",
            "home_abbr": "min", "away_abbr": "las",
            "home_score": None, "away_score": None,
        },
    ]
    rows = events_to_results(events, 2026)
    assert len(rows) == 1
    assert rows[0]["home"] == "lva" and rows[0]["away"] == "sea"
    assert rows[0]["date"] == "2026-07-10"  # UTC midnight → prior Toronto evening


def test_wnba_build_espn_index_uses_toronto_not_utc() -> None:
    from web.wnba_v2.data import _event_calendar_date
    from web.wnba_v2.replay import _espn_local_date, build_espn_index, events_to_results

    iso = "2026-07-11T04:30:00Z"
    assert _espn_local_date({"date": iso}) == _event_calendar_date(iso) == "2026-07-11"

    event = {
        "event_id": "401",
        "date": "2026-01-16T03:30:00Z",
        "season_type": 2,
        "completed": True,
        "home_id": "17",
        "away_id": "14",
        "home_abbr": "lva",
        "away_abbr": "sea",
        "home_score": 90,
        "away_score": 84,
    }
    rows = events_to_results([event], 2026)
    index = build_espn_index([event])
    key = (rows[0]["date"], rows[0]["home"], rows[0]["away"])
    assert key in index
    assert rows[0]["date"] == "2026-01-15"


def test_wnba_espn_local_date_does_not_shift_date_only() -> None:
    """Date-only stamps are calendar keys — midnight−5h must not roll them back."""
    rows = events_to_results(
        [
            {
                "event_id": "1",
                "date": "2026-07-11",
                "completed": True,
                "season_type": 2,
                "home_id": "17",
                "away_id": "14",
                "home_abbr": "lva",
                "away_abbr": "sea",
                "home_score": 90,
                "away_score": 84,
            }
        ],
        2026,
    )
    assert rows[0]["date"] == "2026-07-11"


def test_event_calendar_date_uses_toronto_not_utc_truncation() -> None:
    """Late-ET tips must key on America/Toronto day, not UTC [:10]."""
    assert _event_calendar_date("2026-01-16T03:00:00Z") == "2026-01-15"
    assert _event_calendar_date("2026-01-15") == "2026-01-15"


def test_merge_season_games_flags_2020_bubble_and_attaches_boxes() -> None:
    results = [_game("2020-07-25", "lva", "sea", 82, 74, season=2020)]
    results[0].pop("neutral_site")
    events = [
        {
            "event_id": "999", "date": "2020-07-25", "season_type": 2,
            "completed": True, "neutral_site": False,
            "home_id": "17", "away_id": "14",
            "home_abbr": "lva", "away_abbr": "sea",
            "home_score": 82, "away_score": 74,
        }
    ]
    boxes = {"999": {"17": {"fga": 70.0, "fgm": 30.0}, "14": {"fga": 68.0, "fgm": 27.0}}}
    games = merge_season_games(results, events, boxes, season=2020)
    assert games[0]["neutral_site"] is True  # bubble override
    assert games[0]["event_id"] == "999"
    assert games[0]["home_box"]["fga"] == 70.0


def test_replay_season_respects_cutoff() -> None:
    engine = WnbaFeatureEngine()
    games = [
        _game("2025-05-16", "lva", "sea", 90, 80),
        _game("2025-05-20", "sea", "lva", 85, 79),
    ]
    emitted: list[str] = []
    replay_season(
        engine,
        games,
        stop_before_date="2025-05-20",
        emit=lambda game, _f: emitted.append(str(game["date"])),
    )
    assert emitted == ["2025-05-16"]
    assert engine.teams["lva"].games_played == 1  # cutoff game not folded in


def test_signed_spread_and_devig() -> None:
    assert _signed_spread_from_details("LV -6.5", "lv", "sea") == -6.5
    assert _signed_spread_from_details("SEA -3", "lv", "sea") == 3.0
    assert _signed_spread_from_details("PK", "lv", "sea") is None
    probs = devig_two_way(-150, 130)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1]
    assert devig_two_way(-50, 130) is None  # invalid american price


@pytest.mark.slow
def test_live_prediction_when_artifacts_present() -> None:
    """Live smoke test (skips when model artifacts are not built)."""
    from web.wnba_v2.live import artifacts_available, predict_matchup_v2, wnba_season_for_date

    if not artifacts_available():
        return
    season = wnba_season_for_date("2026-07-10")
    assert season == 2026
    result = predict_matchup_v2("2026-07-10", "lva", "sea")
    if result is None:  # live season context unavailable offline
        return
    assert result["algorithm"] == "WNBAGradientBoost v2"
    assert 0.0 <= result["home_win_probability"] <= 100.0
    assert result["home_elo"] > 1000
    assert "predicted_margin" in result
    assert result.get("model_variant") == "pure"


def test_devig_home_prob_and_market_variant_helpers() -> None:
    from web.wnba_v2.live import _devig_home_prob

    assert _devig_home_prob(None, -110) is None
    assert _devig_home_prob(-110, 100) is not None
    p = _devig_home_prob(-110, -110)
    assert p is not None and abs(p - 0.5) < 0.02
    fav = _devig_home_prob(-200, 170)
    assert fav is not None and fav > 0.6


def test_wnba_v2_american_maps_espn_even_zero() -> None:
    assert _american(0) == 100.0
    assert _american("0") == 100.0
    assert _american("EVEN") == 100.0
    assert _american("PK") == 100.0
    assert _american(-110) == -110.0
    assert _american(50) is None


def test_wnba_v2_side_odds_maps_even_spread_juice() -> None:
    from web.wnba_v2.data import _side_odds, _to_float

    assert _to_float("PK") == 0.0
    assert _to_float("EVEN") == 0.0
    even = _side_odds({"moneyLine": "EVEN", "spreadOdds": "EVEN"})
    assert even["ml"] == 100.0
    assert even["spread_odds"] == 100.0
    junk = _side_odds({"moneyLine": -110, "spreadOdds": -50})
    assert junk["spread_odds"] is None


def test_wnba_v2_side_odds_reads_espn_close_nested_shape() -> None:
    from web.wnba_v2.data import _side_odds

    out = _side_odds(
        {
            "close": {
                "moneyLine": {"american": -160},
                "pointSpread": {"american": -3.5},
                "spread": {"american": -105},
            },
            "open": {
                "moneyLine": {"american": -155},
                "pointSpread": {"american": -3.0},
            },
        }
    )
    assert out["ml"] == -160.0
    assert out["spread_odds"] == -105.0
    assert out["point_spread"] == -3.5
    assert out["ml_open"] == -155.0
    assert out["spread_open"] == -3.0


def test_wnba_v2_preserves_signed_spread_when_favorite_missing() -> None:
    from unittest.mock import patch

    from web.wnba_v2 import data as wnba_data

    payload = {
        "items": [
            {
                "provider": {"name": "ESPN BET"},
                "spread": -5.5,
                "homeTeamOdds": {"favorite": False, "moneyLine": -220, "spreadOdds": -110},
                "awayTeamOdds": {"favorite": False, "moneyLine": 180, "spreadOdds": -110},
            }
        ]
    }
    with patch.object(wnba_data, "get_json", return_value=payload):
        rows = wnba_data.fetch_event_odds("99", "ny", "chi")
    assert len(rows) == 1
    assert rows[0]["home_spread"] == -5.5


def test_wnba_v2_rejects_ml_sized_nested_point_spread() -> None:
    """ML dumped into pointSpread must fall back to flat signed spread."""
    from unittest.mock import patch

    from web.wnba_v2 import data as wnba_data

    payload = {
        "items": [
            {
                "provider": {"name": "DraftKings"},
                "spread": -5.5,
                "homeTeamOdds": {
                    "favorite": True,
                    "moneyLine": -220,
                    "current": {"pointSpread": {"american": -220}},
                    "spreadOdds": -110,
                },
                "awayTeamOdds": {
                    "favorite": False,
                    "moneyLine": 180,
                    "current": {"pointSpread": {"american": 180}},
                    "spreadOdds": -110,
                },
            }
        ]
    }
    with patch.object(wnba_data, "get_json", return_value=payload):
        rows = wnba_data.fetch_event_odds("99", "ny", "chi")
    assert len(rows) == 1
    assert rows[0]["home_spread"] == -5.5
    assert abs(rows[0]["home_spread"]) <= 40


@pytest.mark.slow
def test_live_market_aware_when_odds_provided() -> None:
    from web.wnba_v2.live import artifacts_available, predict_matchup_v2

    if not artifacts_available():
        return
    pure = predict_matchup_v2("2026-07-10", "lva", "sea")
    mkt = predict_matchup_v2(
        "2026-07-10",
        "lva",
        "sea",
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
    assert mkt["home_win_probability"] != pure["home_win_probability"] or mkt[
        "predicted_margin"
    ] != pure["predicted_margin"]


def test_wnba_market_aware_artifacts_present() -> None:
    """Market heads shipped with WNBA v2 must load for live market-aware scoring."""
    from web.wnba_v2.live import _load_artifacts, artifacts_available

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


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    import gzip

    import web.wnba_v2.live as live

    model_dir = tmp_path / "wnba_v2"
    model_dir.mkdir()
    for name in (
        "model_clf.json",
        "model_lr.json",
        "model_margin.json",
        "calibrator.json",
        "metadata.json",
    ):
        (model_dir / name).write_text("{not-json", encoding="utf-8")
    with gzip.open(model_dir / "state_2024.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    monkeypatch.setattr(live, "MODEL_DIR", model_dir)
    live._load_artifacts.cache_clear()
    assert live.artifacts_available() is True
    assert live._load_artifacts() is None


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path) -> None:
    import web.wnba_v2.live as live

    bad = tmp_path / "state_2024.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2024: bad}}
    assert live._load_snapshot_state(art, 2025) is None


def test_predict_matchup_v2_market_aware_wiring_with_stub_context() -> None:
    """Odds + market artifacts flip model_variant without needing a live season fetch."""
    from unittest.mock import MagicMock, patch

    from web.wnba_v2 import live as wnba_live

    team = MagicMock()
    team.games_played = 20
    team.elo = 1500.0
    team.ortg_fast = 105.0
    team.drtg_fast = 100.0
    team.pace_ewma = 80.0
    team.win_pct = MagicMock(return_value=0.55)

    engine = MagicMock()
    engine.teams = {"lva": team, "sea": team}
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
        patch.object(wnba_live, "get_live_context", return_value=context),
        patch.object(wnba_live, "canon_franchise", side_effect=lambda x: x.lower()),
        patch.object(wnba_live, "_predict_probability", return_value=0.62) as mock_prob,
        patch.object(wnba_live, "_predict_regressor", side_effect=[5.0, 85.0, 80.0]),
        patch("web.hybrid_v2.live.try_hybrid_binary", return_value=None),
    ):
        result = wnba_live.predict_matchup_v2(
            "2026-07-10",
            "lva",
            "sea",
            home_moneyline=-150,
            away_moneyline=130,
            home_spread=-4.5,
        )

    assert result is not None
    assert result["model_variant"] == "market_aware"
    assert result["has_market"] is True
    assert result["has_spread"] is True
    assert result["predicted_margin"] == 5.0
    assert mock_prob.call_args.kwargs.get("clf") is art["clf_market"]


def test_fetch_boxes_cached_does_not_poison_null_and_retries(tmp_path, monkeypatch) -> None:
    import json

    import web.wnba_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)
    events = [{"event_id": "w1", "completed": True}]
    calls = {"n": 0}

    def flaky(_event_id: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return {"home": {"pts": 80}, "away": {"pts": 75}}

    monkeypatch.setattr(live, "fetch_box_score", flaky)
    boxes, stale = live._fetch_boxes_cached(2025, events)
    assert stale is True
    assert "w1" not in boxes

    boxes2, stale2 = live._fetch_boxes_cached(2025, events)
    assert stale2 is False
    assert isinstance(boxes2["w1"], dict)
    assert calls["n"] == 2
    cached = json.loads((tmp_path / "boxes_2025.json").read_text(encoding="utf-8"))
    assert isinstance(cached["w1"], dict)


def test_live_season_games_falls_back_to_history_when_events_empty(monkeypatch) -> None:
    import web.wnba_v2.live as live

    history = [{"date": "2025-06-01", "home": "lva", "away": "sea"}]
    monkeypatch.setattr(live, "_fetch_events_cached", lambda *_a, **_k: ([], False))
    monkeypatch.setattr(live, "_history_season", lambda _season: history)
    games, stale = live._live_season_games(2025, current=True)
    assert games == history
    assert stale is False


def test_fetch_events_cached_marks_stale_on_hard_fail(tmp_path, monkeypatch) -> None:
    import web.wnba_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)

    def boom(_season: int):
        raise OSError("network down")

    monkeypatch.setattr(live, "fetch_season_events", boom)
    events, stale = live._fetch_events_cached(2025, current=True)
    assert events == []
    assert stale is True


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import web.wnba_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.WnbaFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )

    calls: list[int] = []

    def fake_games(season: int, *, current: bool = False):
        calls.append(season)
        if season == 2024:
            return [], False
        return [{"date": f"{season}-06-01"}], False

    monkeypatch.setattr(live, "_live_season_games", fake_games)
    monkeypatch.setattr(live, "wnba_season_for_date", lambda _d: 2025)
    assert live.get_live_context("2025-06-15") is None
    assert 2024 in calls


def test_missing_player_box_preserves_rotation_continuity() -> None:
    """Score-only updates must not clear last_players into fake continuity=1."""
    engine = WnbaFeatureEngine()
    roster = [[f"p{i}", 30.0 - i] for i in range(8)]
    with_players = _game("2025-05-16", "lva", "sea", 85, 70)
    with_players["home_box"] = {"players": roster}
    with_players["away_box"] = {"players": [[f"a{i}", 28.0 - i] for i in range(8)]}
    engine.update_after_game(with_players)

    home = engine.teams["lva"]
    assert home.last_players
    # Drop two stars from the "last" lineup so continuity is imperfect.
    home.last_players = home.last_players[2:]
    before = home.top8_continuity()
    assert before < 1.0
    prior_last = [list(row) for row in home.last_players]
    prior_top1 = home.top1_min_share
    prior_depth = home.rotation_depth

    score_only = _game("2025-05-18", "lva", "sea", 80, 75)
    engine.update_after_game(score_only)
    assert home.last_players == prior_last
    assert home.top8_continuity() == before
    assert home.top1_min_share == prior_top1
    assert home.rotation_depth == prior_depth


def _box(players: list[list] | None = None, *, dnp_ids: list[str] | None = None) -> dict:
    box = {
        "fgm": 30.0, "fga": 68.0, "tpm": 8.0, "tpa": 24.0,
        "ftm": 12.0, "fta": 15.0, "orb": 8.0, "drb": 26.0,
        "tov": 12.0, "ast": 18.0,
    }
    if players is not None:
        box["players"] = players
    if dnp_ids is not None:
        box["dnp_ids"] = dnp_ids
    return box


def _rich_players(prefix: str, *, star_out: bool = False) -> list[list]:
    """Synthetic rich rows: [id, min, fga, ast, tov, pf, +/-]."""
    base = [
        [f"{prefix}a", 34, 18, 5, 3, 2, 6],
        [f"{prefix}b", 32, 15, 4, 2, 3, 3],
        [f"{prefix}c", 28, 12, 3, 2, 4, 1],
        [f"{prefix}d", 24, 9, 2, 1, 2, -1],
        [f"{prefix}e", 20, 7, 2, 1, 3, -2],
        [f"{prefix}f", 16, 5, 1, 2, 2, -3],
        [f"{prefix}g", 12, 3, 1, 1, 1, -2],
        [f"{prefix}h", 10, 2, 1, 1, 2, -2],
    ]
    if star_out:
        return [
            [f"{prefix}c", 34, 14, 3, 2, 5, -5],
            [f"{prefix}d", 30, 12, 3, 2, 4, -3],
            [f"{prefix}e", 26, 10, 2, 1, 3, -2],
            [f"{prefix}f", 22, 8, 2, 2, 3, -2],
            [f"{prefix}g", 18, 6, 1, 1, 2, -1],
            [f"{prefix}h", 14, 4, 1, 1, 2, 0],
            [f"{prefix}q", 12, 3, 1, 1, 1, 0],
            [f"{prefix}r", 10, 2, 1, 1, 1, 0],
        ]
    return base


def test_player_box_rows_rich_shape_and_dnp() -> None:
    summary = {
        "boxscore": {
            "players": [
                {
                    "team": {"id": "17"},
                    "statistics": [
                        {
                            "names": ["MIN", "FG", "AST", "TO", "PF", "+/-"],
                            "athletes": [
                                {
                                    "athlete": {"id": "100"},
                                    "stats": ["32", "8-16", "4", "2", "3", "+5"],
                                },
                                {
                                    "athlete": {"id": "101"},
                                    "didNotPlay": True,
                                    "stats": [],
                                },
                                {
                                    "athlete": {"id": "102"},
                                    "stats": ["18", "3-7", "1", "1", "2", "-2"],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }
    players, dnp = _player_box_rows(summary)
    assert dnp["17"] == ["101"]
    assert players["17"] == [
        ["100", 32.0, 16.0, 4.0, 2.0, 3.0, 5.0],
        ["102", 18.0, 7.0, 1.0, 1.0, 2.0, -2.0],
    ]


def test_player_box_rows_soft_fail_missing_usage_cols() -> None:
    summary = {
        "boxscore": {
            "players": [
                {
                    "team": {"id": "14"},
                    "statistics": [
                        {
                            "names": ["MIN"],
                            "athletes": [
                                {"athlete": {"id": "200"}, "stats": ["28"]},
                                {"athlete": {"id": "201"}, "stats": ["0"]},
                            ],
                        }
                    ],
                }
            ]
        }
    }
    players, dnp = _player_box_rows(summary)
    assert dnp == {}
    assert players["14"] == [["200", 28.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def test_availability_proxies_from_player_rows() -> None:
    engine = WnbaFeatureEngine()
    full = [["a", 34], ["b", 32], ["c", 28], ["d", 26], ["e", 22],
            ["f", 18], ["g", 14], ["h", 12]]
    sea_full = [["x", 34], ["y", 32], ["z", 28], ["w", 26], ["v", 22],
                ["u", 18], ["t", 14], ["s", 12]]
    # Longer warmup so slow-EWMA stars keep a clear lead over role players.
    for day in range(2, 16):
        game = _game(f"2025-05-{day:02d}", "lva", "sea", 85, 70)
        game["home_box"] = _box(full)
        game["away_box"] = _box(sea_full)
        engine.update_after_game(game)
    even = engine.features_for_game(_game("2025-05-17", "lva", "sea", 0, 0))
    assert even["star_avail_diff"] == 0.0
    assert even["top1_min_share_diff"] == 0.0
    assert even["rotation_depth_diff"] == 0.0
    assert even["close_win_ewma_diff"] == 0.0
    assert even["bubble_season"] == 0.0
    assert "away_tz_shift" in even
    assert "home_3in4" in even
    assert "high_min_ast_tov_diff" not in even
    assert "bench_pm_diff" not in even
    sea_short = [["z", 32], ["w", 30], ["v", 28], ["u", 24], ["t", 20],
                 ["s", 16], ["q", 14], ["r", 12]]
    game_short = _game("2025-05-17", "lva", "sea", 85, 70)
    game_short["home_box"] = _box(full)
    game_short["away_box"] = _box(sea_short)
    engine.update_after_game(game_short)
    hurt = engine.features_for_game(_game("2025-05-19", "lva", "sea", 0, 0))
    assert hurt["star_avail_diff"] > 0.5
    assert hurt["top8_continuity_diff"] > 0.1
    assert hurt["dnp_star_rate_diff"] < 0.0
    assert hurt["star_min_gap_diff"] > 0.0


def test_rich_player_rotation_features() -> None:
    engine = WnbaFeatureEngine()
    for day in range(16, 24):
        game = _game(f"2025-05-{day:02d}", "lva", "sea", 85, 70)
        game["home_box"] = _box(_rich_players("h"))
        game["away_box"] = _box(_rich_players("a"))
        engine.update_after_game(game)
    even = engine.features_for_game(_game("2025-05-25", "lva", "sea", 0, 0))
    assert abs(even["top1_usage_diff"]) < 1e-6
    concentrated = [
        ["ha", 38, 24, 7, 3, 7, 10],
        ["hb", 28, 10, 3, 2, 5, 2],
        ["hc", 22, 7, 2, 1, 4, 0],
        ["hd", 18, 5, 2, 1, 3, -1],
        ["he", 14, 3, 1, 1, 1, -2],
        ["hf", 10, 2, 1, 1, 1, -3],
        ["hg", 8, 2, 0, 1, 1, -2],
        ["hh", 6, 1, 0, 0, 1, -4],
    ]
    game = _game("2025-05-25", "lva", "sea", 85, 70)
    game["home_box"] = _box(concentrated)
    game["away_box"] = _box(_rich_players("a", star_out=True), dnp_ids=["aa", "ab"])
    engine.update_after_game(game)
    feat = engine.features_for_game(_game("2025-05-27", "lva", "sea", 0, 0))
    assert feat["top1_min_share_diff"] > 0.0
    assert feat["top1_usage_diff"] > 0.0
    assert feat["min_hhi_diff"] > 0.0
    assert feat["dnp_star_rate_diff"] < 0.0


def test_from_dict_defaults_new_rotation_fields() -> None:
    engine = WnbaFeatureEngine()
    game = _game("2025-05-16", "lva", "sea", 85, 70)
    game["home_box"] = _box([["a", 34], ["b", 28]])
    game["away_box"] = _box([["x", 34], ["y", 28]])
    engine.update_after_game(game)
    payload = engine.to_dict()
    rotation_keys = (
        "prev_player_ids", "season_minutes",
        "top1_min_share", "top3_min_share", "top1_usage", "top3_usage",
        "high_min_ast_tov", "high_min_foul_rate", "star_min_ewma",
        "star_min_season_avg", "bench_pm", "dnp_star_rate", "rotation_depth",
        "min_hhi", "bench_min_share",
    )
    for team_payload in payload["teams"].values():
        for key in rotation_keys:
            team_payload.pop(key, None)
    restored = WnbaFeatureEngine.from_dict(payload)
    features = restored.features_for_game(_game("2025-05-18", "lva", "sea", 0, 0))
    assert set(features.keys()) == set(FEATURE_COLUMNS)
    assert features["star_avail_diff"] == 0.0
    assert features["top1_min_share_diff"] == 0.0
    assert features["star_min_gap_diff"] == 0.0
    assert features["margin_vol_diff"] == 0.0
    assert "star_avail_diff" in FEATURE_COLUMNS
    assert "bench_min_share_diff" in FEATURE_COLUMNS
    assert "bubble_season" in FEATURE_COLUMNS
    assert "away_tz_shift" in FEATURE_COLUMNS


def test_minutes_weighted_dnp_star_rate_prefers_star_minutes() -> None:
    from web.wnba_v2.feature_engine import _player_rotation_metrics

    # Top season minutes on aa/ab; only aa plays → missing weight ~ ab share.
    metrics = _player_rotation_metrics(
        [["aa", 34], ["c", 20], ["d", 18]],
        dnp_ids=["ab"],
        season_minutes={"aa": 400.0, "ab": 380.0, "c": 100.0, "d": 90.0,
                        "e": 80.0, "f": 70.0, "g": 60.0, "h": 50.0},
        games_played=10,
    )
    assert metrics["dnp_star_rate"] > 0.25
    assert metrics["dnp_star_rate"] < 0.55


def test_bubble_season_flag() -> None:
    engine = WnbaFeatureEngine()
    feat = engine.features_for_game(_game("2020-07-25", "lva", "sea", 0, 0, season=2020))
    assert feat["bubble_season"] == 1.0
    feat2 = engine.features_for_game(_game("2025-07-25", "lva", "sea", 0, 0, season=2025))
    assert feat2["bubble_season"] == 0.0


if __name__ == "__main__":
    test_canon_franchise_follows_relocation_chains()
    test_engine_features_precede_update_and_elo_moves_to_winner()
    test_neutral_site_removes_home_court()
    test_rest_and_b2b_flags()
    test_season_rollover_carries_elo_and_resets_records()
    test_engine_snapshot_round_trip()
    test_events_to_results_synthesizes_completed_games()
    test_merge_season_games_flags_2020_bubble_and_attaches_boxes()
    test_replay_season_respects_cutoff()
    test_signed_spread_and_devig()
    test_live_prediction_when_artifacts_present()
    test_devig_home_prob_and_market_variant_helpers()
    test_live_market_aware_when_odds_provided()
    test_wnba_market_aware_artifacts_present()
    test_predict_matchup_v2_market_aware_wiring_with_stub_context()
    test_missing_player_box_preserves_rotation_continuity()
    test_player_box_rows_rich_shape_and_dnp()
    test_player_box_rows_soft_fail_missing_usage_cols()
    test_availability_proxies_from_player_rows()
    test_rich_player_rotation_features()
    test_from_dict_defaults_new_rotation_fields()
    test_minutes_weighted_dnp_star_rate_prefers_star_minutes()
    test_bubble_season_flag()
    print("test_wnba_v2.py: all tests passed")
