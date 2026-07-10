"""WNBA v2 feature engine, replay, data-parsing, and live-path unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.wnba_v2.data import (  # noqa: E402
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
    assert later["h2h_home_win_rate"] == 1.0


def test_neutral_site_removes_home_court() -> None:
    engine = WnbaFeatureEngine()
    neutral = engine.features_for_game(
        _game("2025-05-16", "lva", "sea", 0, 0, neutral=True)
    )
    assert neutral["elo_diff"] == 0.0
    assert neutral["neutral_site"] == 1.0


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
    elo_2025 = engine.teams["lva"].elo
    features = engine.features_for_game(_game("2026-05-16", "lva", "sea", 0, 0, season=2026))
    lva = engine.teams["lva"]
    assert lva.wins == 0 and lva.losses == 0  # season stats reset
    assert 1500.0 < lva.elo < elo_2025  # carryover regresses toward mean
    assert features["prev_win_pct_diff"] == 1.0  # last season's record retained
    assert features["home_games_played"] == 0.0


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
    assert rows[0]["date"] == "2026-07-10"  # UTC evening -> US local date


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
    print("test_wnba_v2.py: all tests passed")
