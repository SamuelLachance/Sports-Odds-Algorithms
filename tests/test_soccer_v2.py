"""Soccer v2 feature engine, replay, and live-resolution unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_v2.data import devig_decimal, parse_season_csv  # noqa: E402
from web.soccer_v2.feature_engine import (  # noqa: E402
    FEATURE_COLUMNS,
    SoccerFeatureEngine,
    _dixon_coles_tau,
    poisson_1x2,
)
from web.soccer_v2.live import (  # noqa: E402
    _decorrelate,
    _norm_name,
    american_to_decimal,
    resolve_team,
)
from web.soccer_v2.replay import merge_country_rows, replay_country  # noqa: E402


def _row(
    date: str,
    home: str,
    away: str,
    hg: int,
    ag: int,
    *,
    division: str = "E0",
    season: int = 2024,
) -> dict:
    return {
        "division": division,
        "season": season,
        "date": date,
        "home": home,
        "away": away,
        "home_goals": hg,
        "away_goals": ag,
        "home_shots": 14,
        "away_shots": 9,
        "home_sot": 6,
        "away_sot": 3,
        "home_corners": 6,
        "away_corners": 4,
        "home_reds": 0,
        "away_reds": 0,
    }


def test_poisson_1x2_sums_to_one_and_favors_stronger_attack() -> None:
    home, draw, away = poisson_1x2(2.2, 0.9)
    assert abs(home + draw + away - 1.0) < 1e-9
    assert home > away
    even_home, even_draw, even_away = poisson_1x2(1.3, 1.3)
    assert abs(even_home + even_draw + even_away - 1.0) < 1e-9
    assert even_draw > draw  # closer matchup -> higher draw probability


def test_poisson_1x2_high_xg_stays_non_negative() -> None:
    """Dixon–Coles τ must clamp to ≥ 0 when λ > 1/|ρ| (~10)."""
    from web.soccer_v2.feature_engine import DC_RHO

    exp_home, exp_away = 12.0, 11.0
    # Unclamped factors would be negative: 1 + 12*(-0.10) = -0.20
    assert 1.0 + exp_home * DC_RHO < 0.0
    assert 1.0 + exp_away * DC_RHO < 0.0
    assert _dixon_coles_tau(0, 1, exp_home, exp_away, DC_RHO) == 0.0
    assert _dixon_coles_tau(1, 0, exp_home, exp_away, DC_RHO) == 0.0
    assert _dixon_coles_tau(0, 0, exp_home, exp_away, DC_RHO) >= 0.0
    home, draw, away = poisson_1x2(exp_home, exp_away)
    assert home >= 0.0 and draw >= 0.0 and away >= 0.0
    assert abs(home + draw + away - 1.0) < 1e-9


def test_engine_elo_moves_toward_winner_and_features_precede_update() -> None:
    engine = SoccerFeatureEngine("E")
    engine.start_season(2024)
    first = engine.features_for_match(
        {"date": "2024-08-10", "home": "Arsenal", "away": "Everton"}, "epl"
    )
    # neutral priors before any result: home edge comes only from home advantage
    assert first["elo_diff"] == 60.0
    for i in range(6):
        engine.update_after_match(
            _row(f"2024-08-{10 + i:02d}", "Arsenal", "Everton", 3, 0), tier=1
        )
    assert engine.teams["Arsenal"].elo > engine.teams["Everton"].elo
    later = engine.features_for_match(
        {"date": "2024-09-20", "home": "Arsenal", "away": "Everton"}, "epl"
    )
    assert later["elo_diff"] > first["elo_diff"]
    assert later["gf_fast_diff"] > 0
    assert set(FEATURE_COLUMNS).issuperset(later.keys()) or set(later).issuperset(
        FEATURE_COLUMNS
    )


def test_engine_snapshot_round_trip() -> None:
    engine = SoccerFeatureEngine("E")
    engine.start_season(2024)
    for i in range(4):
        engine.update_after_match(
            _row(f"2024-08-{10 + i:02d}", "Arsenal", "Everton", 2, 1), tier=1
        )
    restored = SoccerFeatureEngine.from_dict(engine.to_dict())
    assert restored.teams["Arsenal"].elo == engine.teams["Arsenal"].elo
    assert restored.league_gpg == engine.league_gpg
    original = engine.features_for_match(
        {"date": "2024-09-01", "home": "Arsenal", "away": "Everton"}, "epl"
    )
    round_trip = restored.features_for_match(
        {"date": "2024-09-01", "home": "Arsenal", "away": "Everton"}, "epl"
    )
    assert original == round_trip


def test_promoted_flag_via_tier_change() -> None:
    engine = SoccerFeatureEngine("E")
    engine.start_season(2023)
    engine.update_after_match(
        _row("2023-09-02", "Leeds", "Leicester", 2, 1, division="E1", season=2023),
        tier=2,
    )
    engine.start_season(2024)
    engine.update_after_match(
        _row("2024-08-17", "Leeds", "Everton", 1, 1, division="E0", season=2024),
        tier=1,
    )
    features = engine.features_for_match(
        {"date": "2024-08-24", "home": "Leeds", "away": "Everton"}, "epl"
    )
    assert features["home_promoted"] == 1.0
    assert features["away_promoted"] == 0.0


def test_replay_country_emits_tier1_only_and_respects_cutoff() -> None:
    engine = SoccerFeatureEngine("E")
    rows = [
        _row("2024-08-10", "Arsenal", "Everton", 2, 0, division="E0"),
        _row("2024-08-11", "Leeds", "Hull", 3, 1, division="E1"),
        _row("2024-08-18", "Everton", "Arsenal", 0, 1, division="E0"),
    ]
    emitted: list[str] = []
    replay_country(
        engine,
        merge_country_rows({"E0": [rows[0], rows[2]], "E1": [rows[1]]}),
        on_features=lambda row, _f: emitted.append(f"{row['home']}-{row['date']}"),
        stop_before_date="2024-08-18",
    )
    assert emitted == ["Arsenal-2024-08-10"]
    # tier-2 result still updated the pool
    assert engine.teams["Leeds"].games_played == 1
    # cutoff row not folded in
    assert engine.teams["Arsenal"].games_played == 1


def test_parse_season_csv_extracts_stats_and_odds() -> None:
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,HC,AC,"
        "PSH,PSD,PSA,B365H,B365D,B365A,PSCH,PSCD,PSCA,MaxH,MaxD,MaxA\n"
        "E0,16/08/25,Arsenal,Everton,2,0,15,7,7,2,8,3,"
        "1.45,4.6,7.5,1.44,4.5,7.0,1.42,4.75,8.0,1.5,4.8,8.2\n"
    )
    rows = parse_season_csv(text, "E0", 2025)
    assert len(rows) == 1
    row = rows[0]
    assert row["date"] == "2025-08-16"
    assert row["home_sot"] == 7
    assert row["open_home"] == 1.45
    assert row["close_home"] == 1.42
    assert row["max_home"] == 1.5
    assert row["b365_home"] == 1.44


def test_devig_decimal_sums_to_one() -> None:
    probs = devig_decimal(1.5, 4.5, 7.0)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1] > probs[2]
    assert devig_decimal(None, 4.5, 7.0) is None


def test_decorrelate_pushes_away_from_market() -> None:
    model = (0.50, 0.27, 0.23)
    market = (0.45, 0.28, 0.27)
    pushed = _decorrelate(model, market, weight=0.15)
    assert pushed[0] > model[0]  # model above market -> pushed higher
    assert abs(sum(pushed) - 1.0) < 1e-9
    assert _decorrelate(model, None) == model


def test_resolve_team_by_abbr_hint_and_fuzzy() -> None:
    engine = SoccerFeatureEngine("E")
    for name in ("Arsenal", "Nott'm Forest", "Wolves", "Sunderland"):
        engine.team(name)
    assert resolve_team(engine, "epl", "ars", "Arsenal") == "Arsenal"
    assert resolve_team(engine, "epl", "nfo", "Nottingham Forest") == "Nott'm Forest"
    assert resolve_team(engine, "epl", None, "Wolverhampton Wanderers") == "Wolves"
    assert resolve_team(engine, "epl", None, "Sunderland AFC") == "Sunderland"
    assert resolve_team(engine, "epl", "zzz", "Atlantis United") is None
    assert _norm_name("1. FC Köln") == "koln"


@pytest.mark.slow
def test_live_context_and_prediction_when_artifacts_present() -> None:
    """End-to-end live smoke test (skips when artifacts are not built)."""
    from web.soccer_v2.live import artifacts_available, predict_matchup_v2

    if not artifacts_available():
        return
    result = predict_matchup_v2(
        "epl",
        "2026-07-10",
        home_abbr="ars",
        away_abbr="liv",
        home_name="Arsenal",
        away_name="Liverpool",
        home_ml=-120,
        draw_ml=260,
        away_ml=300,
    )
    assert result is not None
    assert result["algorithm"] == "SoccerGradientBoost v2"
    total = (
        result["home_win_probability"]
        + result["draw_probability"]
        + result["away_win_probability"]
    )
    assert abs(total - 100.0) < 0.5
    assert result["market_decorrelated"] is True
    assert result["expected_home_goals"] is not None
    assert result["elo_home"] > 1000


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    """Corrupt on-disk JSON must not raise — live layer returns None instead."""
    import gzip

    import web.soccer_v2.live as live

    model_dir = tmp_path / "soccer_v2"
    model_dir.mkdir()
    required = (
        "model_clf.json",
        "model_lr.json",
        "model_clf_market.json",
        "model_lr_market.json",
        "calibrator.json",
        "calibrator_market.json",
        "metadata.json",
    )
    for name in required:
        (model_dir / name).write_text("{not-json", encoding="utf-8")
    with gzip.open(model_dir / "state_2024.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    monkeypatch.setattr(live, "MODEL_DIR", model_dir)
    live._load_artifacts.cache_clear()
    assert live.artifacts_available() is True
    assert live._load_artifacts() is None


def test_blend_soccer_survives_pred_model_crash() -> None:
    """Blend must fall back cleanly when soccer pred raises."""
    from web.blend_service import blend_predictions
    import web.blend_service as blend_module

    original = blend_module.run_soccer_pred_model

    def _boom(*_a, **_k):
        raise RuntimeError("simulated soccer v2 failure")

    try:
        blend_module.run_soccer_pred_model = _boom
        result = blend_predictions(
            legacy_total_score=-48.0,
            legacy_win_probability=48.0,
            league="epl",
            cutoff_date="11-15-2025",
            home_abbr="ars",
            away_abbr="liv",
            home_moneyline=-120,
            draw_moneyline=260,
            away_moneyline=320,
        )
    finally:
        blend_module.run_soccer_pred_model = original

    assert result["blend_mode"] == "soccer_path_a_unavailable"
    assert result["soccer_pred"] is None


def test_fetch_current_csv_keeps_prior_on_network_failure(tmp_path, monkeypatch) -> None:
    """Network failure must not wipe a usable CSV cache; flag as stale instead."""
    import time

    import web.soccer_v2.live as live
    from web.soccer_v2.data import parse_season_csv

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)
    csv_body = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR\n"
        "E0,10/08/24,Arsenal,Liverpool,2,1,H,10,8,5,3,8,10,4,3,1,2,0,0\n"
    )
    cache = tmp_path / "E0_2024.csv"
    cache.write_text(csv_body, encoding="utf-8")
    aged = time.time() - (live.CURRENT_CSV_TTL_SECONDS + 60)
    import os

    os.utime(cache, (aged, aged))

    def _boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(live, "fetch_csv_text", _boom)
    rows, stale = live._fetch_current_csv("E0", 2024)
    assert stale is True
    assert len(rows) == 1
    assert rows[0]["home"] == "Arsenal"
    # Cache must still contain the prior body (not emptied).
    assert "Arsenal" in cache.read_text(encoding="utf-8")
    expected = parse_season_csv(csv_body, "E0", 2024)
    assert rows[0]["away"] == expected[0]["away"]


def test_american_to_decimal_maps_espn_even_zero() -> None:
    assert american_to_decimal(0) == pytest.approx(2.0)
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(None) is None
    probs = devig_decimal(american_to_decimal(0), 3.5, 4.0)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9


def test_american_to_decimal_rejects_invalid_magnitude() -> None:
    """Garbage |ml| < 100 must not enable market-aware soccer heads."""
    assert american_to_decimal(-50) is None
    assert american_to_decimal(50) is None
    assert american_to_decimal(75) is None


if __name__ == "__main__":
    test_poisson_1x2_sums_to_one_and_favors_stronger_attack()
    test_engine_elo_moves_toward_winner_and_features_precede_update()
    test_engine_snapshot_round_trip()
    test_promoted_flag_via_tier_change()
    test_replay_country_emits_tier1_only_and_respects_cutoff()
    test_parse_season_csv_extracts_stats_and_odds()
    test_devig_decimal_sums_to_one()
    test_decorrelate_pushes_away_from_market()
    test_resolve_team_by_abbr_hint_and_fuzzy()
    test_live_context_and_prediction_when_artifacts_present()
    test_blend_soccer_survives_pred_crash()
    test_fetch_current_csv_keeps_prior_on_network_failure()
    test_american_to_decimal_maps_espn_even_zero()
    test_american_to_decimal_rejects_invalid_magnitude()
    print("test_soccer_v2.py: all tests passed")
