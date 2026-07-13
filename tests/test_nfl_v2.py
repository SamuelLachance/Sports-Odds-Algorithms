"""NFL v2 feature engine, replay, and live-path unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nfl_v2.feature_engine import (  # noqa: E402
    ELO_HOME_ADV,
    FEATURE_COLUMNS,
    NflFeatureEngine,
    infer_neutral_site,
    infer_playoff,
)
from web.nfl_v2.replay import (  # noqa: E402
    events_to_results,
    nflverse_rows_to_games,
    replay_season,
)


def _game(
    date: str,
    home: str,
    away: str,
    hs: int,
    as_: int,
    *,
    season: int = 2024,
    neutral: bool = False,
    week: int = 1,
    game_type: str = "REG",
    home_rest: float = 7.0,
    away_rest: float = 7.0,
    home_qb_id: str = "qb-home",
    away_qb_id: str = "qb-away",
    home_coach: str = "Coach Home",
    away_coach: str = "Coach Away",
) -> dict:
    return {
        "date": date,
        "season": season,
        "home": home,
        "away": away,
        "home_score": hs,
        "away_score": as_,
        "neutral_site": neutral,
        "week": week,
        "game_type": game_type,
        "location": "Neutral" if neutral else "Home",
        "div_game": False,
        "roof": "outdoors",
        "surface": "grass",
        "temp": 65,
        "wind": 8,
        "weekday": "Sunday",
        "home_rest": home_rest,
        "away_rest": away_rest,
        "home_qb_id": home_qb_id,
        "away_qb_id": away_qb_id,
        "home_coach": home_coach,
        "away_coach": away_coach,
    }


def test_feature_columns_count_in_range() -> None:
    assert 55 <= len(FEATURE_COLUMNS) <= 130
    assert "cold_game" in FEATURE_COLUMNS
    assert "wr1_snap_share_diff" in FEATURE_COLUMNS
    assert "sack_rate_def_diff" in FEATURE_COLUMNS


def test_engine_features_precede_update_and_elo_moves() -> None:
    engine = NflFeatureEngine()
    first = engine.features_for_game(_game("2024-09-08", "kc", "bal", 0, 0))
    assert set(first.keys()) == set(FEATURE_COLUMNS)
    assert first["elo_diff"] == ELO_HOME_ADV
    assert first["home_qb_known"] == 1.0
    assert first["away_qb_known"] == 1.0
    for week, day in enumerate(("08", "15", "22", "29"), start=1):
        engine.update_after_game(
            _game(f"2024-09-{day}", "kc", "bal", 27, 20, week=week)
        )
    assert engine.teams["kc"].elo > engine.teams["bal"].elo
    later = engine.features_for_game(_game("2024-10-06", "kc", "bal", 0, 0, week=5))
    assert later["elo_diff"] > first["elo_diff"]
    assert later["win_pct_diff"] == 1.0
    assert later["home_games_played"] == 4.0
    assert later["home_qb_streak"] == 4.0
    assert engine.qbs["qb-home"].elo > engine.qbs["qb-away"].elo


def test_no_leakage_qb_elo_updates_after_features() -> None:
    engine = NflFeatureEngine()
    feats = engine.features_for_game(_game("2024-09-08", "kc", "bal", 27, 20))
    assert feats["qb_elo_diff"] == 0.0
    engine.update_after_game(_game("2024-09-08", "kc", "bal", 27, 20))
    after = engine.features_for_game(_game("2024-09-15", "kc", "bal", 0, 0, week=2))
    assert after["qb_elo_diff"] != 0.0


def test_neutral_site_removes_hfa() -> None:
    engine = NflFeatureEngine()
    home = engine.features_for_game(_game("2024-09-08", "kc", "sf", 0, 0))
    neutral = engine.features_for_game(
        _game("2025-02-09", "kc", "sf", 0, 0, season=2024, neutral=True, game_type="SB")
    )
    assert home["neutral_site"] == 0.0
    assert neutral["neutral_site"] == 1.0
    assert abs(home["elo_diff"] - neutral["elo_diff"] - ELO_HOME_ADV) < 1e-9
    assert neutral["is_playoff"] == 1.0


def test_infer_playoff_and_neutral() -> None:
    assert infer_playoff({"game_type": "REG"}) is False
    assert infer_playoff({"game_type": "POST"}) is True
    assert infer_playoff({"game_type": "SB"}) is True
    assert infer_neutral_site({"location": "Home"}) is False
    assert infer_neutral_site({"location": "Neutral"}) is True
    # Stringy false/0 must not strip HFA via bool("false") is True.
    assert infer_neutral_site({"neutral_site": "false", "location": "Home"}) is False
    assert infer_neutral_site({"neutral_site": "0", "location": "Home"}) is False
    assert infer_neutral_site({"neutral_site": False, "location": "Home"}) is False
    assert infer_neutral_site({"neutral_site": "true"}) is True
    assert infer_neutral_site({"neutral_site": True}) is True


def test_false_rest_falls_back_to_computed_days() -> None:
    """bool False must not coerce to 0.0 rest (false short-week signal)."""
    engine = NflFeatureEngine()
    engine.update_after_game(_game("2024-09-08", "kc", "den", 24, 17))
    bad = _game("2024-09-15", "kc", "buf", 0, 0)
    bad["home_rest"] = False
    bad["away_rest"] = False
    ok = _game("2024-09-15", "kc", "buf", 0, 0)
    bad_feats = engine.features_for_game(bad)
    ok_feats = engine.features_for_game(ok)
    assert bad_feats["home_rest_days"] == ok_feats["home_rest_days"] == 7.0
    assert bad_feats["short_week"] == 0.0


def test_short_week_and_rest_from_nflverse_fields() -> None:
    engine = NflFeatureEngine()
    g = _game("2024-09-12", "kc", "bal", 0, 0, home_rest=4.0, away_rest=7.0)
    g["weekday"] = "Thursday"
    feats = engine.features_for_game(g)
    assert feats["home_rest_days"] == 4.0
    assert feats["away_rest_days"] == 7.0
    assert feats["short_week"] == 1.0
    assert feats["weekday_thu"] == 1.0
    assert feats["rest_diff"] == -3.0


def test_season_rollover_carries_elo() -> None:
    engine = NflFeatureEngine()
    for day in ("08", "15", "22"):
        engine.update_after_game(_game(f"2024-09-{day}", "kc", "bal", 28, 14, season=2024))
    elo_end = engine.teams["kc"].elo
    feats = engine.features_for_game(_game("2025-09-07", "kc", "bal", 0, 0, season=2025))
    assert engine.teams["kc"].games_played == 0
    assert engine.teams["kc"].prev_win_pct == 1.0
    assert engine.teams["kc"].elo != 1500.0
    assert abs(engine.teams["kc"].elo - (1500.0 + 0.70 * (elo_end - 1500.0))) < 1e-6
    assert "prev_win_pct_diff" in feats


def test_engine_snapshot_round_trip() -> None:
    engine = NflFeatureEngine()
    engine.update_after_game(_game("2024-09-08", "kc", "bal", 27, 20))
    engine.update_after_game(_game("2024-09-15", "sf", "kc", 24, 30, week=2))
    payload = engine.to_dict()
    restored = NflFeatureEngine.from_dict(payload)
    assert restored.teams["kc"].elo == engine.teams["kc"].elo
    assert restored.teams["kc"].wins == engine.teams["kc"].wins
    assert restored.qbs["qb-home"].elo == engine.qbs["qb-home"].elo
    assert restored.league_ppg == engine.league_ppg
    f1 = engine.features_for_game(_game("2024-09-22", "kc", "sf", 0, 0, week=3))
    f2 = restored.features_for_game(_game("2024-09-22", "kc", "sf", 0, 0, week=3))
    for col in FEATURE_COLUMNS:
        assert abs(f1[col] - f2[col]) < 1e-9


def test_replay_emits_before_update() -> None:
    engine = NflFeatureEngine()
    games = [
        _game("2024-09-08", "kc", "bal", 27, 20),
        _game("2024-09-15", "kc", "cin", 26, 25, week=2),
    ]
    emitted: list[tuple] = []

    def emit(game, features):
        emitted.append((game["date"], features["home_games_played"], engine.teams.get("kc")))

    replay_season(engine, games, emit=emit)
    assert len(emitted) == 2
    assert emitted[0][1] == 0.0
    assert emitted[1][1] == 1.0
    assert engine.teams["kc"].games_played == 2


def test_nflverse_rows_to_games_and_events() -> None:
    rows = [
        {
            "gameday": "2024-09-08",
            "season": 2024,
            "game_type": "REG",
            "week": 1,
            "home_team": "KC",
            "away_team": "BAL",
            "home_score": 27,
            "away_score": 20,
            "location": "Home",
            "spread_line": 3.0,
            "div_game": 0,
            "roof": "outdoors",
            "surface": "grass",
            "home_rest": 7,
            "away_rest": 7,
            "home_qb_id": "00-0033873",
            "away_qb_id": "00-0034796",
            "home_coach": "Andy Reid",
            "away_coach": "John Harbaugh",
            "weekday": "Sunday",
        }
    ]
    games = nflverse_rows_to_games(rows)
    assert len(games) == 1
    assert games[0]["home"] == "kc"
    assert games[0]["home_close_spread"] == -3.0
    assert games[0]["home_qb_id"] == "00-0033873"

    events = events_to_results(
        [
            {
                "date": "2024-09-08T17:00Z",
                "home": "kc",
                "away": "bal",
                "home_score": 27,
                "away_score": 20,
                "completed": True,
            }
        ],
        2024,
    )
    assert len(events) == 1
    assert events[0]["home_score"] == 27


def test_player_proxy_features_present() -> None:
    proxies = {
        "close_win_ewma_diff",
        "blowout_net_ewma_diff",
        "blowout_rate_diff",
        "margin_vol_diff",
        "scoring_vol_diff",
        "one_score_rate_diff",
        "qb_elo_diff",
        "coach_elo_diff",
    }
    assert proxies.issubset(set(FEATURE_COLUMNS))


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path) -> None:
    import web.nfl_v2.live as live

    bad = tmp_path / "state_2023.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2023: bad}}
    assert live._load_snapshot_state(art, 2024) is None


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    import gzip

    import web.nfl_v2.live as live

    model_dir = tmp_path / "nfl_v2"
    model_dir.mkdir()
    for name in ("model_clf.json", "model_lr.json", "model_margin.json", "calibrator.json"):
        (model_dir / name).write_text("{not-json", encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        '{"ship_models": true}', encoding="utf-8"
    )
    with gzip.open(model_dir / "state_2023.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    monkeypatch.setattr(live, "MODEL_DIR", model_dir)
    live._load_artifacts.cache_clear()
    assert live.artifacts_available() is True
    assert live._load_artifacts() is None


def test_live_artifacts_helper() -> None:
    from web.nfl_v2.live import artifacts_available

    assert isinstance(artifacts_available(), bool)


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import web.nfl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NflFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )

    calls: list[int] = []

    def fake_games(season: int, *, stop_before: str):
        calls.append(season)
        if season == 2024:
            return []
        return [{"date": f"{season}-09-08", "home": "kc", "away": "bal"}]

    monkeypatch.setattr(live, "_fetch_completed_season_games", fake_games)
    monkeypatch.setattr(live, "nfl_season_of", lambda _d: 2025)
    assert live.get_live_context("2025-09-15") is None
    assert 2024 in calls


def test_espn_local_date_uses_toronto_not_fixed_utc_minus_5() -> None:
    from web.nfl_v2.replay import _espn_local_date
    from web.season_games import _event_date_iso

    iso = "2024-06-15T04:30:00Z"
    assert _espn_local_date({"date": iso}) == _event_date_iso(iso) == "2024-06-15"
    assert _espn_local_date({"date": "2024-09-08"}) == "2024-09-08"


def test_get_live_context_fails_closed_without_snapshot(monkeypatch) -> None:
    import web.nfl_v2.live as live

    live.get_live_context.cache_clear()
    monkeypatch.setattr(live, "_load_artifacts", lambda: {"snapshots": {}})
    monkeypatch.setattr(live, "_load_snapshot_state", lambda _art, _season: None)
    monkeypatch.setattr(live, "nfl_season_of", lambda _d: 2025)
    assert live.get_live_context("2025-09-15") is None


def test_espn_fetch_accepts_iso_stop_before(monkeypatch, tmp_path) -> None:
    """Live day_iso is YYYY-MM-DD; ESPN path must not die in _parse_cutoff."""
    import web.nfl_v2.live as live
    import web.season_games as season_games

    live.LIVE_CACHE_DIR = tmp_path
    cutoffs: list[str] = []

    def fake_map(league: str, cutoff_date: str, *, for_backtest: bool = False):
        cutoffs.append(cutoff_date)
        # Regression: ISO used to raise ValueError (month=15) inside _parse_cutoff.
        from web.live_data import _parse_cutoff

        parsed = _parse_cutoff(cutoff_date)
        assert (parsed.year, parsed.month, parsed.day) == (2025, 9, 15)
        return {
            "e1": (
                "2024-09-08T17:00Z",
                ("kc", "bal", "Chiefs", "Ravens", 27, 20),
            )
        }

    monkeypatch.setattr(season_games, "_load_league_game_map", fake_map)
    rows = live._fetch_completed_season_games_espn(2024, stop_before="2025-09-15")
    assert cutoffs == ["2025-09-15"]
    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["home"] == "kc"


def test_sos_elo_uses_pre_update_opponent_elo() -> None:
    """SOS must accumulate opponent Elo before the game's Elo shift."""
    from web.nfl_v2.feature_engine import LEAGUE_ELO

    engine = NflFeatureEngine()
    engine.update_after_game(_game("2024-09-08", "kc", "bal", 27, 20))
    assert engine.teams["kc"].sos_elo() == LEAGUE_ELO
    assert engine.teams["bal"].sos_elo() == LEAGUE_ELO
    # After Elo moved, a wrong post-update reconstruction would drift SOS.
    assert engine.teams["kc"].elo != engine.teams["bal"].elo
    assert engine.teams["kc"].sos_elo_sum == LEAGUE_ELO
    assert engine.teams["bal"].sos_elo_sum == LEAGUE_ELO


def test_h2h_win_rate_survives_home_away_role_reversal() -> None:
    """W/L H2H must update both sides — reverse fixtures must not default to 0.5."""
    engine = NflFeatureEngine()
    engine.update_after_game(_game("2024-09-08", "kc", "sf", 27, 20))
    same = engine.features_for_game(_game("2024-10-06", "kc", "sf", 0, 0, week=5))
    reverse = engine.features_for_game(_game("2024-10-06", "sf", "kc", 0, 0, week=5))
    assert same["h2h_home_win_rate"] == 1.0
    assert reverse["h2h_home_win_rate"] == 0.0
    assert reverse["h2h_margin_ewma"] < 0.0


def test_coach_streak_emitted_and_accumulates() -> None:
    engine = NflFeatureEngine()
    engine.update_after_game(_game("2024-09-08", "kc", "bal", 27, 20))
    feats = engine.features_for_game(
        _game("2024-09-15", "kc", "cin", 0, 0, week=2, home_coach="Coach Home", away_coach="Coach Cin")
    )
    assert "home_coach_streak" in FEATURE_COLUMNS
    assert feats["home_coach_streak"] == 1.0
    engine.update_after_game(
        _game("2024-09-15", "kc", "cin", 24, 17, week=2, home_coach="Coach Home", away_coach="Coach Cin")
    )
    feats2 = engine.features_for_game(
        _game("2024-09-22", "kc", "lac", 0, 0, week=3, home_coach="Coach Home", away_coach="Coach Lac")
    )
    assert feats2["home_coach_streak"] == 2.0


def test_live_prefers_nflverse_context_fields(monkeypatch, tmp_path: Path) -> None:
    from web.nfl_v2 import live as nfl_live

    csv_path = tmp_path / "nflverse.csv"
    csv_path.write_text(
        "season,gameday,home_team,away_team,home_qb_id,away_qb_id,home_coach,away_coach,"
        "roof,surface,temp,wind,weekday,home_rest,away_rest,div_game,week,game_type,location\n"
        "2024,2024-09-08,KC,BAL,qb-mahomes,qb-jackson,Andy Reid,John Harbaugh,"
        "outdoors,grass,72,5,Sunday,7,7,0,1,REG,Home\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(nfl_live, "NFLVERSE_CSV", csv_path)
    ctx = nfl_live._lookup_nflverse_matchup_context("2024-09-08", "kc", "bal")
    assert ctx["home_qb_id"] == "qb-mahomes"
    assert ctx["away_coach"] == "John Harbaugh"
    assert ctx["temp"] == 72 or float(ctx["temp"]) == 72.0
    assert ctx["neutral_site"] is False


def test_qb_change_and_epa_features() -> None:
    engine = NflFeatureEngine()
    g1 = _game("2024-09-08", "kc", "bal", 27, 20, home_qb_id="qb-a", away_qb_id="qb-b")
    g1.update(
        {
            "home_epa_off": 0.2,
            "away_epa_off": -0.1,
            "home_epa_def": 0.0,
            "away_epa_def": 0.05,
            "home_sr_off": 0.5,
            "away_sr_off": 0.4,
            "home_sr_def": 0.45,
            "away_sr_def": 0.48,
            "home_explosive_off": 0.12,
            "away_explosive_off": 0.08,
            "home_pass_epa_off": 0.25,
            "away_pass_epa_off": -0.05,
        }
    )
    engine.features_for_game(g1)
    engine.update_after_game(g1)
    g2 = _game("2024-09-15", "kc", "cin", 24, 17, home_qb_id="qb-backup", away_qb_id="qb-c", week=2)
    feats = engine.features_for_game(g2)
    assert "epa_off_diff" in FEATURE_COLUMNS
    assert "qb_change_diff" in FEATURE_COLUMNS
    assert "rush_epa_diff" in FEATURE_COLUMNS
    assert "spread_move" in FEATURE_COLUMNS
    assert "ref_home_bias" in FEATURE_COLUMNS
    assert "tz_diff" in FEATURE_COLUMNS
    assert feats["qb_change_diff"] == 1.0
    assert feats["epa_off_diff"] != 0.0
    assert feats["madden_known"] in (0.0, 1.0)
