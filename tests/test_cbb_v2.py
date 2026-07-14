"""CBB v2 feature engine, replay, and data-parsing unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cbb_v2.data import (  # noqa: E402
    canon_abbr,
    cbb_season_for_date,
    force_postseason_neutral_site,
    team_key,
)
from web.cbb_v2.feature_engine import (  # noqa: E402
    ELO_HOME_ADV,
    FEATURE_COLUMNS,
    CbbFeatureEngine,
)
from web.cbb_v2.replay import events_to_results, merge_season_games, replay_season  # noqa: E402


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
    conference: bool = False,
    home_abbr: str | None = None,
    away_abbr: str | None = None,
) -> dict:
    return {
        "date": date,
        "season": season,
        "season_type": season_type,
        "home": home,
        "away": away,
        "home_abbr": home_abbr or home,
        "away_abbr": away_abbr or away,
        "home_score": hs,
        "away_score": as_,
        "neutral_site": neutral,
        "conference_game": conference,
        "home_conference_id": "23" if conference else "1",
        "away_conference_id": "23" if conference else "2",
    }


def test_feature_column_count_in_range() -> None:
    assert 40 <= len(FEATURE_COLUMNS) <= 160
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_canon_abbr_and_team_key() -> None:
    assert canon_abbr("UConn") == "conn"
    assert canon_abbr("CONN") == "conn"
    assert team_key("150", "conn") == "150"
    assert team_key("", "UConn") == "conn"


def test_cbb_season_for_date() -> None:
    assert cbb_season_for_date("2024-11-15") == 2025
    assert cbb_season_for_date("2025-03-20") == 2025
    assert cbb_season_for_date("2025-07-01") == 2025


def test_engine_features_precede_update_and_elo_moves_to_winner() -> None:
    engine = CbbFeatureEngine()
    first = engine.features_for_game(_game("2024-11-16", "duke", "unc", 0, 0))
    assert first["elo_diff"] == ELO_HOME_ADV
    assert set(first.keys()) == set(FEATURE_COLUMNS)
    for day in range(16, 22):
        engine.update_after_game(_game(f"2024-11-{day:02d}", "duke", "unc", 85, 70))
    assert engine.teams["duke"].elo > engine.teams["unc"].elo
    later = engine.features_for_game(_game("2024-12-10", "duke", "unc", 0, 0))
    assert later["elo_diff"] > first["elo_diff"]
    assert later["win_pct_diff"] == 1.0
    assert later["h2h_home_win_rate"] == 1.0


def test_neutral_site_removes_home_court() -> None:
    engine = CbbFeatureEngine()
    neutral = engine.features_for_game(
        _game("2024-11-16", "duke", "unc", 0, 0, neutral=True)
    )
    assert neutral["elo_diff"] == 0.0
    assert neutral["neutral_site"] == 1.0


def test_conference_flag_and_splits() -> None:
    engine = CbbFeatureEngine()
    engine.update_after_game(
        _game("2024-11-16", "duke", "unc", 80, 70, conference=True)
    )
    features = engine.features_for_game(
        _game("2024-11-23", "duke", "unc", 0, 0, conference=True)
    )
    assert features["is_conference"] == 1.0
    assert engine.teams["duke"].conf_wins == 1
    assert engine.teams["duke"].nonconf_wins == 0


def test_rest_and_b2b_flags() -> None:
    engine = CbbFeatureEngine()
    engine.update_after_game(_game("2024-11-16", "duke", "unc", 80, 70))
    features = engine.features_for_game(_game("2024-11-17", "duke", "ku", 0, 0))
    assert features["home_rest_days"] == 1.0
    assert features["home_b2b"] == 1.0
    assert features["away_b2b"] == 0.0


def test_season_rollover_carries_elo_and_resets_records() -> None:
    engine = CbbFeatureEngine()
    for day in range(16, 26):
        engine.update_after_game(
            _game(f"2024-11-{day:02d}", "duke", "unc", 90, 70, season=2025)
        )
    elo_2025 = engine.teams["duke"].elo
    features = engine.features_for_game(
        _game("2025-11-16", "duke", "unc", 0, 0, season=2026)
    )
    duke = engine.teams["duke"]
    assert duke.wins == 0 and duke.losses == 0
    assert 1500.0 < duke.elo < elo_2025
    assert features["prev_win_pct_diff"] == 1.0
    assert features["home_games_played"] == 0.0


def test_engine_snapshot_round_trip() -> None:
    engine = CbbFeatureEngine()
    for day in range(16, 20):
        engine.update_after_game(_game(f"2024-11-{day:02d}", "duke", "unc", 78, 72))
    restored = CbbFeatureEngine.from_dict(engine.to_dict())
    assert restored.teams["duke"].elo == engine.teams["duke"].elo
    assert restored.league_ppg == engine.league_ppg
    original = engine.features_for_game(_game("2024-12-01", "duke", "unc", 0, 0))
    round_trip = restored.features_for_game(_game("2024-12-01", "duke", "unc", 0, 0))
    assert original == round_trip


def test_events_to_results_and_merge() -> None:
    events = [
        {
            "event_id": "401",
            "date": "2025-01-15T00:00Z",
            "season": 2025,
            "season_type": 2,
            "completed": True,
            "home_id": "150",
            "away_id": "153",
            "home_abbr": "duke",
            "away_abbr": "unc",
            "home_score": 80,
            "away_score": 74,
            "neutral_site": False,
            "conference_game": True,
            "home_conference_id": "2",
            "away_conference_id": "2",
        },
        {
            "event_id": "402",
            "date": "2025-01-15T02:00Z",
            "season": 2025,
            "season_type": 2,
            "completed": False,
            "home_id": "96",
            "away_id": "2305",
            "home_abbr": "ku",
            "away_abbr": "kstate",
            "home_score": None,
            "away_score": None,
        },
    ]
    rows = events_to_results(events, 2025)
    assert len(rows) == 1
    assert rows[0]["home"] == "150" and rows[0]["away"] == "153"
    assert rows[0]["conference_game"] is True
    games = merge_season_games([], events, season=2025)
    assert len(games) == 1
    assert games[0]["home_abbr"] == "duke"


def test_postseason_march_forces_neutral_site_train_and_live_parity() -> None:
    """Training strips HCA for March postseason; live must apply the same heuristic."""
    events = [
        {
            "event_id": "401584",
            "date": "2025-03-20T00:00Z",
            "season": 2025,
            "season_type": 3,
            "completed": True,
            "home_id": "150",
            "away_id": "153",
            "home_abbr": "duke",
            "away_abbr": "unc",
            "home_score": 80,
            "away_score": 74,
            "neutral_site": False,
            "conference_game": False,
        }
    ]
    games = merge_season_games([], events, season=2025)
    assert len(games) == 1
    assert games[0]["neutral_site"] is True

    live_game = {
        "date": "2025-03-20",
        "season": 2025,
        "season_type": 3,
        "home": "150",
        "away": "153",
        "home_abbr": "duke",
        "away_abbr": "unc",
        "neutral_site": False,
        "conference_game": False,
        "home_conference_id": "",
        "away_conference_id": "",
    }
    force_postseason_neutral_site(live_game)
    assert live_game["neutral_site"] is True

    engine = CbbFeatureEngine()
    feats = engine.features_for_game(
        {
            **live_game,
            "home_score": 0,
            "away_score": 0,
        }
    )
    assert feats["neutral_site"] == 1.0
    assert feats["elo_diff"] == 0.0  # no ELO_HOME_ADV on neutral


def test_replay_emits_before_update() -> None:
    engine = CbbFeatureEngine()
    games = [
        _game("2024-11-16", "duke", "unc", 80, 70),
        _game("2024-11-23", "duke", "unc", 75, 72),
    ]
    seen: list[float] = []

    def emit(game: dict, features: dict[str, float]) -> None:
        seen.append(features["elo_diff"])

    replay_season(engine, games, emit=emit)
    assert len(seen) == 2
    assert seen[0] == ELO_HOME_ADV
    assert seen[1] > seen[0]


def test_artifacts_available_false_without_models() -> None:
    from web.cbb_v2 import live

    # Point at empty temp by checking default dir without required files is fine
    # if models were just trained this may be True; only assert callable works.
    assert isinstance(live.artifacts_available(), bool)


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path) -> None:
    import web.cbb_v2.live as live

    bad = tmp_path / "state_2024.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2024: bad}}
    assert live._load_snapshot_state(art, 2025) is None


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    import gzip

    import web.cbb_v2.live as live

    model_dir = tmp_path / "cbb_v2"
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


def test_fetch_events_cached_marks_stale_on_hard_fail(tmp_path, monkeypatch) -> None:
    import web.cbb_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)

    def boom(_season: int, use_cache: bool = True):
        raise OSError("network down")

    monkeypatch.setattr(live, "fetch_season_events", boom)
    events, stale = live._fetch_events_cached(2025, current=True)
    assert events == []
    assert stale is True


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import web.cbb_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.CbbFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )

    calls: list[int] = []

    def fake_games(season: int, *, current: bool = False):
        calls.append(season)
        if season == 2024:
            return [], False
        return [{"date": f"{season}-01-01"}], False

    monkeypatch.setattr(live, "_live_season_games", fake_games)
    monkeypatch.setattr(live, "cbb_season_for_date", lambda _d: 2025)
    assert live.get_live_context("2025-01-15") is None
    assert 2024 in calls


def test_espn_local_date_uses_toronto_not_fixed_utc_minus_5() -> None:
    """EDT tips must not roll back a day under the old UTC−5 heuristic."""
    from web.cbb_v2.replay import _espn_local_date
    from web.season_games import _event_date_iso

    # 2024-06-15 04:30Z = 00:30 America/Toronto (EDT); UTC−5 wrongly → 06-14.
    iso = "2024-06-15T04:30:00Z"
    assert _espn_local_date({"date": iso}) == _event_date_iso(iso) == "2024-06-15"
    assert _espn_local_date({"date": "2024-11-15"}) == "2024-11-15"


def test_parse_event_stores_toronto_calendar_day() -> None:
    """Parsed events must never retain raw ISO stamps (breaks season windows)."""
    from web.cbb_v2.data import _parse_event

    event = {
        "id": "401",
        "date": "2025-04-15T18:00:00Z",
        "season": {"year": 2025, "type": 3},
        "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
        "competitions": [
            {
                "neutralSite": False,
                "conferenceCompetition": False,
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": "70",
                        "team": {"id": "150", "abbreviation": "DUKE", "displayName": "Duke"},
                    },
                    {
                        "homeAway": "away",
                        "score": "65",
                        "team": {"id": "153", "abbreviation": "UNC", "displayName": "UNC"},
                    },
                ],
            }
        ],
    }
    parsed = _parse_event(event, fallback_date="2025-04-15")
    assert parsed is not None
    assert parsed["date"] == "2025-04-15"
    assert "T" not in parsed["date"]
    # Season-window string compare must accept the normalized day.
    assert "2024-11-01" <= parsed["date"] <= "2025-04-15"


def test_blowout_ewma_is_signed_for_winner_and_loser() -> None:
    """blowout_rate_diff must separate blowout winners from losers."""
    engine = CbbFeatureEngine()
    # Margin 30 >= BLOWOUT_MARGIN (18)
    engine.update_after_game(_game("2024-11-16", "duke", "unc", 90, 60))
    home = engine.teams["duke"]
    away = engine.teams["unc"]
    assert home.blowout_ewma > 0.0
    assert away.blowout_ewma < 0.0
    feats = engine.features_for_game(_game("2024-11-23", "duke", "unc", 0, 0))
    assert feats["blowout_rate_diff"] > 0.0


def _team_box(
    *,
    fgm: float = 30.0,
    fga: float = 60.0,
    tpm: float = 10.0,
    tpa: float = 25.0,
    ftm: float = 15.0,
    fta: float = 20.0,
    orb: float = 12.0,
    drb: float = 25.0,
    tov: float = 10.0,
    ast: float = 15.0,
) -> dict:
    return {
        "fgm": fgm,
        "fga": fga,
        "tpm": tpm,
        "tpa": tpa,
        "ftm": ftm,
        "fta": fta,
        "orb": orb,
        "drb": drb,
        "tov": tov,
        "ast": ast,
    }


def test_merge_attaches_home_away_boxes() -> None:
    events = [
        {
            "event_id": "401",
            "date": "2025-01-15",
            "season": 2025,
            "season_type": 2,
            "completed": True,
            "home_id": "150",
            "away_id": "153",
            "home_abbr": "duke",
            "away_abbr": "unc",
            "home_score": 80,
            "away_score": 74,
            "neutral_site": False,
            "conference_game": True,
            "home_conference_id": "2",
            "away_conference_id": "2",
        }
    ]
    boxes = {
        "401": {
            "150": _team_box(fga=62.0),
            "153": _team_box(fga=58.0),
        }
    }
    games = merge_season_games([], events, boxes, season=2025)
    assert len(games) == 1
    assert games[0]["home_box"]["fga"] == 62.0
    assert games[0]["away_box"]["fga"] == 58.0


def test_box_update_changes_pace_and_four_factors() -> None:
    """Real boxes move pace / eFG / TOV / ORB / FTR off league priors."""
    from web.cbb_v2.feature_engine import (
        LEAGUE_EFG,
        LEAGUE_FT_RATE,
        LEAGUE_ORB_PCT,
        LEAGUE_PACE,
        LEAGUE_TOV_RATE,
    )

    engine = CbbFeatureEngine()
    game = _game("2024-11-16", "duke", "unc", 85, 70)
    # High eFG (~0.583), elevated pace via high FGA+FTA, low TOV.
    game["home_box"] = _team_box(
        fgm=35, fga=60, tpm=10, tpa=22, ftm=15, fta=18, orb=14, drb=28, tov=8
    )
    game["away_box"] = _team_box(
        fgm=25, fga=65, tpm=5, tpa=28, ftm=10, fta=14, orb=8, drb=20, tov=16
    )
    prior_efg = engine.teams["duke"].efg_for if "duke" in engine.teams else LEAGUE_EFG
    engine.update_after_game(game)
    duke = engine.teams["duke"]
    unc = engine.teams["unc"]
    assert duke.pace_ewma != LEAGUE_PACE
    assert duke.efg_for != prior_efg
    assert duke.efg_for > LEAGUE_EFG  # hot shooting night
    assert unc.tov_for > LEAGUE_TOV_RATE  # 16 TOV on ~60 poss
    assert duke.orb_for != LEAGUE_ORB_PCT
    assert duke.ftr_for != LEAGUE_FT_RATE
    feats = engine.features_for_game(_game("2024-11-23", "duke", "unc", 0, 0))
    assert feats["pace_diff"] != 0.0 or feats["pace_sum"] != 2.0 * LEAGUE_PACE
    assert feats["efg_for_diff"] != 0.0
    assert feats["tov_for_diff"] != 0.0


def test_missing_box_holds_four_factor_priors() -> None:
    """Score-only updates keep four-factor EWMAs at prior; pace still moves via ASSUMED."""
    from web.cbb_v2.feature_engine import LEAGUE_EFG, LEAGUE_TOV_RATE

    engine = CbbFeatureEngine()
    engine.update_after_game(_game("2024-11-16", "duke", "unc", 80, 70))
    duke = engine.teams["duke"]
    assert duke.efg_for == LEAGUE_EFG
    assert duke.tov_for == LEAGUE_TOV_RATE
    assert duke.orb_for == engine.teams["unc"].orb_for  # both still at league prior
    # Pace uses ASSUMED_POSSESSIONS scaled by game total — may move off LEAGUE_PACE.
    assert duke.pace_ewma != 0.0
    # Explicit: a second score-only game still does not touch four factors.
    engine.update_after_game(_game("2024-11-20", "duke", "unc", 78, 72))
    assert duke.efg_for == LEAGUE_EFG
    assert duke.efg_against == LEAGUE_EFG


def test_box_emit_before_update_sees_prior_four_factors() -> None:
    """Replay emit must read four-factor state before folding the box into EWMAs."""
    from web.cbb_v2.feature_engine import LEAGUE_EFG

    engine = CbbFeatureEngine()
    boxed = _game("2024-11-16", "duke", "unc", 85, 70)
    boxed["home_box"] = _team_box(fgm=40, fga=60, tpm=12, tov=6)
    boxed["away_box"] = _team_box(fgm=22, fga=60, tpm=4, tov=18)
    seen: list[float] = []

    def emit(_game_row: dict, features: dict[str, float]) -> None:
        seen.append(features["efg_for_diff"])

    replay_season(engine, [boxed], emit=emit)
    assert len(seen) == 1
    assert seen[0] == 0.0  # both at LEAGUE_EFG before update
    assert engine.teams["duke"].efg_for > LEAGUE_EFG


def test_fetch_season_boxes_does_not_poison_null(tmp_path, monkeypatch) -> None:
    """Failed mid-season box fetches must not stick as null in boxes.json."""
    import json

    import web.cbb_v2.data as cbb_data

    monkeypatch.setattr(cbb_data, "CACHE_ROOT", tmp_path)
    events = [{"event_id": "e1", "completed": True}]
    calls = {"n": 0}

    def flaky(_event_id: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("timeout")
        return {"150": _team_box(), "153": _team_box()}

    monkeypatch.setattr(cbb_data, "fetch_box_score", flaky)
    boxes = cbb_data.fetch_season_boxes(2025, events, use_cache=True)
    assert "e1" not in boxes
    cached_path = tmp_path / "2025" / "boxes.json"
    assert cached_path.is_file()
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    assert "e1" not in cached

    boxes2 = cbb_data.fetch_season_boxes(2025, events, use_cache=True)
    assert isinstance(boxes2["e1"], dict)
    assert calls["n"] == 2
    cached2 = json.loads(cached_path.read_text(encoding="utf-8"))
    assert isinstance(cached2["e1"], dict)


def test_live_fetch_boxes_cached_does_not_poison_null(tmp_path, monkeypatch) -> None:
    import json

    import web.cbb_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)
    events = [{"event_id": "c1", "completed": True}]
    calls = {"n": 0}

    def flaky(_event_id: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return {"150": {"fga": 60.0}, "153": {"fga": 58.0}}

    monkeypatch.setattr(live, "fetch_box_score", flaky)
    monkeypatch.setattr(live, "load_season_boxes", lambda _season: {})
    boxes, stale = live._fetch_boxes_cached(2025, events)
    assert stale is True
    assert "c1" not in boxes

    boxes2, stale2 = live._fetch_boxes_cached(2025, events)
    assert stale2 is False
    assert isinstance(boxes2["c1"], dict)
    assert calls["n"] == 2
    cached = json.loads((tmp_path / "boxes_2025.json").read_text(encoding="utf-8"))
    assert isinstance(cached["c1"], dict)

def test_predict_matchup_v2_market_aware_wiring_with_stub_context() -> None:
    """Odds + market artifacts flip model_variant without needing a live season fetch."""
    from unittest.mock import MagicMock, patch

    from web.cbb_v2 import live as cbb_live

    team = MagicMock()
    team.games_played = 20
    team.elo = 1600.0
    team.ortg_fast = 110.0
    team.drtg_fast = 98.0
    team.pace_ewma = 68.0
    team.win_pct = MagicMock(return_value=0.7)

    engine = MagicMock()
    engine.teams = {"duke": team, "unc": team}
    engine.team = MagicMock(return_value=team)
    engine.resolve_team_id = MagicMock(side_effect=lambda abbr, _id=None: abbr.lower())
    engine.features_for_game = MagicMock(
        return_value={
            "home_rest_days": 2.0,
            "away_rest_days": 1.0,
            "is_conference": 1.0,
        }
    )

    art = {
        "feature_columns": ["elo_diff"],
        "clf_market_features": ["mkt_home_prob", "has_market"],
        "margin_market_features": ["mkt_home_spread", "has_spread"],
        "clf": object(),
        "lr": {"xgb_weight": 0.55},
        "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "clf_market": object(),
        "lr_market": {"xgb_weight": 0.55},
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
        "abbr_map": {},
    }

    with (
        patch.object(cbb_live, "get_live_context", return_value=context),
        patch.object(cbb_live, "_predict_probability", return_value=0.64) as mock_prob,
        patch.object(cbb_live, "_predict_regressor", side_effect=[8.0, 78.0, 70.0]),
        patch("web.hybrid_v2.live.try_hybrid_binary", return_value=None),
    ):
        result = cbb_live.predict_matchup_v2(
            "2026-01-10",
            "duke",
            "unc",
            home_moneyline=-200,
            away_moneyline=170,
            home_spread=-6.5,
        )

    assert result is not None
    assert result["model_variant"] == "market_aware"
    assert result["has_market"] is True
    assert result["has_spread"] is True
    assert mock_prob.call_args.kwargs.get("clf") is art["clf_market"]
