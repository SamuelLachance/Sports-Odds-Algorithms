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
