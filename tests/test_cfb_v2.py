"""CFB v2 feature engine, replay, and live-path unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cfb_v2.feature_engine import (  # noqa: E402
    ELO_HOME_ADV,
    FEATURE_COLUMNS,
    CfbFeatureEngine,
    conference_of,
    infer_neutral_site,
)
from web.cfb_v2.replay import csv_rows_to_games, events_to_results, replay_season  # noqa: E402


def _game(
    date: str,
    home: str,
    away: str,
    hs: int,
    as_: int,
    *,
    season: int = 2024,
    neutral: bool = False,
) -> dict:
    return {
        "date": date,
        "season": season,
        "home": home,
        "away": away,
        "home_score": hs,
        "away_score": as_,
        "neutral_site": neutral,
    }


def test_feature_columns_count_in_range() -> None:
    assert 40 <= len(FEATURE_COLUMNS) <= 70


def test_conference_proxy_maps_power_five() -> None:
    assert conference_of("ala") == "sec"
    assert conference_of("mich") == "b1g"
    assert conference_of("clem") == "acc"
    assert conference_of("unknown_xyz") == ""


def test_engine_features_precede_update_and_elo_moves() -> None:
    engine = CfbFeatureEngine()
    first = engine.features_for_game(_game("2024-09-07", "ala", "usf", 0, 0))
    assert set(first.keys()) == set(FEATURE_COLUMNS)
    assert first["elo_diff"] == ELO_HOME_ADV
    assert first["is_conference_game"] == 0.0
    for week, day in enumerate(("07", "14", "21", "28"), start=1):
        engine.update_after_game(_game(f"2024-09-{day}", "ala", "usf", 42, 10))
    assert engine.teams["ala"].elo > engine.teams["usf"].elo
    later = engine.features_for_game(_game("2024-10-05", "ala", "usf", 0, 0))
    assert later["elo_diff"] > first["elo_diff"]
    assert later["win_pct_diff"] == 1.0
    assert later["home_games_played"] == 4.0


def test_neutral_site_removes_hfa() -> None:
    engine = CfbFeatureEngine()
    home = engine.features_for_game(_game("2024-09-07", "ala", "uga", 0, 0))
    neutral = engine.features_for_game(
        _game("2025-01-01", "ala", "uga", 0, 0, season=2024, neutral=True)
    )
    assert home["neutral_site"] == 0.0
    assert neutral["neutral_site"] == 1.0
    assert abs(home["elo_diff"] - neutral["elo_diff"] - ELO_HOME_ADV) < 1e-9


def test_infer_bowl_neutral_from_date() -> None:
    assert infer_neutral_site({"date": "2025-01-01"}) is True
    assert infer_neutral_site({"date": "2024-12-20"}) is True
    assert infer_neutral_site({"date": "2024-09-14"}) is False


def test_conference_game_feature() -> None:
    engine = CfbFeatureEngine()
    feats = engine.features_for_game(_game("2024-10-12", "ala", "uga", 0, 0))
    assert feats["is_conference_game"] == 1.0


def test_rest_and_games_last14() -> None:
    engine = CfbFeatureEngine()
    engine.update_after_game(_game("2024-09-07", "ala", "usf", 35, 14))
    feats = engine.features_for_game(_game("2024-09-14", "ala", "usf", 0, 0))
    assert feats["home_rest_days"] == 7.0
    assert feats["home_games_last14"] == 1.0


def test_season_rollover_carries_elo() -> None:
    engine = CfbFeatureEngine()
    for day in ("07", "14", "21"):
        engine.update_after_game(_game(f"2024-09-{day}", "ala", "usf", 40, 7, season=2024))
    elo_end = engine.teams["ala"].elo
    feats = engine.features_for_game(_game("2025-08-30", "ala", "usf", 0, 0, season=2025))
    assert engine.teams["ala"].games_played == 0
    assert engine.teams["ala"].prev_win_pct == 1.0
    assert engine.teams["ala"].elo != 1500.0
    assert abs(engine.teams["ala"].elo - (1500.0 + 0.70 * (elo_end - 1500.0))) < 1e-6
    assert "prev_win_pct_diff" in feats


def test_engine_snapshot_round_trip() -> None:
    engine = CfbFeatureEngine()
    engine.update_after_game(_game("2024-09-07", "ala", "usf", 35, 17))
    engine.update_after_game(_game("2024-09-14", "uga", "ala", 24, 30))
    payload = engine.to_dict()
    restored = CfbFeatureEngine.from_dict(payload)
    assert restored.teams["ala"].elo == engine.teams["ala"].elo
    assert restored.teams["ala"].wins == engine.teams["ala"].wins
    assert restored.league_ppg == engine.league_ppg
    f1 = engine.features_for_game(_game("2024-09-21", "ala", "uga", 0, 0))
    f2 = restored.features_for_game(_game("2024-09-21", "ala", "uga", 0, 0))
    for col in FEATURE_COLUMNS:
        assert abs(f1[col] - f2[col]) < 1e-9


def test_replay_emits_before_update() -> None:
    engine = CfbFeatureEngine()
    games = [
        _game("2024-09-07", "ala", "usf", 42, 10),
        _game("2024-09-14", "ala", "uga", 28, 21),
    ]
    emitted: list[tuple] = []

    def emit(game, features):
        emitted.append((game["date"], features["home_games_played"], engine.teams.get("ala")))

    replay_season(engine, games, emit=emit)
    assert len(emitted) == 2
    assert emitted[0][1] == 0.0
    assert emitted[1][1] == 1.0
    assert engine.teams["ala"].games_played == 2


def test_csv_rows_to_games_and_events() -> None:
    rows = [
        {
            "date": "2024-09-07",
            "home_key": "ala",
            "away_key": "usf",
            "home_final": 42,
            "away_final": 10,
        }
    ]
    games = csv_rows_to_games(rows)
    assert len(games) == 1
    assert games[0]["home"] == "ala"
    assert games[0]["season"] == 2024

    events = events_to_results(
        [
            {
                "date": "2024-09-07T19:00Z",
                "home": "ala",
                "away": "usf",
                "home_score": 42,
                "away_score": 10,
                "completed": True,
            }
        ],
        2024,
    )
    assert len(events) == 1
    assert events[0]["home_score"] == 42


def test_live_artifacts_helper() -> None:
    from web.cfb_v2.live import artifacts_available

    # Soft check: function is callable; may be False before training.
    assert isinstance(artifacts_available(), bool)


def test_player_proxy_features_present() -> None:
    proxies = {
        "close_win_ewma_diff",
        "blowout_net_ewma_diff",
        "blowout_rate_diff",
        "margin_vol_diff",
        "scoring_vol_diff",
        "one_score_rate_diff",
    }
    assert proxies.issubset(set(FEATURE_COLUMNS))


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path) -> None:
    import web.cfb_v2.live as live

    bad = tmp_path / "state_2023.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2023: bad}}
    assert live._load_snapshot_state(art, 2024) is None


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import web.cfb_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.CfbFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )

    calls: list[int] = []

    def fake_games(season: int, *, stop_before: str):
        calls.append(season)
        if season == 2024:
            return []
        return [{"date": f"{season}-09-07", "home": "ala", "away": "uga"}]

    monkeypatch.setattr(live, "_fetch_completed_season_games", fake_games)
    monkeypatch.setattr(live, "cfb_season_of", lambda _d: 2025)
    assert live.get_live_context("2025-09-15") is None
    assert 2024 in calls


def test_espn_local_date_uses_toronto_not_fixed_utc_minus_5() -> None:
    from web.cfb_v2.replay import _espn_local_date
    from web.season_games import _event_date_iso

    iso = "2024-06-15T04:30:00Z"
    assert _espn_local_date({"date": iso}) == _event_date_iso(iso) == "2024-06-15"
    assert _espn_local_date({"date": "2024-09-01"}) == "2024-09-01"


def test_sos_elo_uses_pre_update_opponent_elo() -> None:
    from web.cfb_v2.feature_engine import LEAGUE_ELO

    engine = CfbFeatureEngine()
    engine.update_after_game(_game("2024-09-07", "ala", "uga", 34, 24))
    assert engine.teams["ala"].sos_elo() == LEAGUE_ELO
    assert engine.teams["uga"].sos_elo() == LEAGUE_ELO
    assert engine.teams["ala"].elo != engine.teams["uga"].elo
    assert engine.teams["ala"].sos_elo_sum == LEAGUE_ELO
    assert engine.teams["uga"].sos_elo_sum == LEAGUE_ELO


def test_tie_is_not_treated_as_away_win() -> None:
    engine = CfbFeatureEngine()
    engine.update_after_game(_game("1995-11-18", "ala", "aub", 17, 17, season=1995))
    home = engine.teams["ala"]
    away = engine.teams["aub"]
    assert home.wins == 0 and home.losses == 0 and home.ties == 1
    assert away.wins == 0 and away.losses == 0 and away.ties == 1
    assert home.streak == 0 and away.streak == 0
    assert home.win_pct() == 0.5
    # Home was favored by HFA; a tie pulls home Elo down and away Elo up.
    assert home.elo < 1500.0
    assert away.elo > 1500.0
    # Neither side should record an H2H win on a tie.
    assert home.h2h.get("aub", [0, 0]) == [0, 0]


def test_espn_fetch_accepts_iso_stop_before(monkeypatch, tmp_path) -> None:
    """CFB live has no nflverse fallback — ISO cutoff must parse for season replay."""
    import web.cfb_v2.live as live
    import web.season_games as season_games

    live.LIVE_CACHE_DIR = tmp_path
    cutoffs: list[str] = []

    def fake_map(league: str, cutoff_date: str, *, for_backtest: bool = False):
        cutoffs.append(cutoff_date)
        from web.live_data import _parse_cutoff

        parsed = _parse_cutoff(cutoff_date)
        assert (parsed.year, parsed.month, parsed.day) == (2025, 9, 15)
        return {
            "e1": (
                "2024-09-07T19:00Z",
                ("ala", "uga", "Alabama", "Georgia", 34, 24),
            )
        }

    monkeypatch.setattr(season_games, "_load_league_game_map", fake_map)
    rows = live._fetch_completed_season_games(2024, stop_before="2025-09-15")
    assert cutoffs == ["2025-09-15"]
    assert len(rows) == 1
    assert rows[0]["home"] == "ala"


def test_h2h_win_rate_survives_home_away_role_reversal() -> None:
    """W/L H2H must update both sides — reverse fixtures must not default to 0.5."""
    engine = CfbFeatureEngine()
    engine.update_after_game(_game("2024-09-01", "ala", "uga", 24, 20))
    same = engine.features_for_game(_game("2024-11-01", "ala", "uga", 0, 0))
    reverse = engine.features_for_game(_game("2024-11-01", "uga", "ala", 0, 0))
    assert same["h2h_home_win_rate"] == 1.0
    assert reverse["h2h_home_win_rate"] == 0.0
    assert reverse["h2h_margin_ewma"] < 0.0


def test_week_zero_is_not_replaced_by_date_derived_week() -> None:
    """Explicit week=0 (preseason) must not fall through truthy `or`."""
    engine = CfbFeatureEngine()
    game = _game("2024-08-24", "ala", "uga", 0, 0)
    game["week"] = 0
    feats = engine.features_for_game(game)
    assert feats["week"] == 0.0
